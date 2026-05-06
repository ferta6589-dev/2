from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import time

import httpx
import structlog

from . import arb, clob_ws, markets
from .config import Settings, load
from .executor import PaperExecutor
from .orderbook import OrderBook


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ]
    )


log = structlog.get_logger("polyarb")


class BotState:
    def __init__(self) -> None:
        self.market: markets.Market | None = None
        self.yes_book: OrderBook | None = None
        self.no_book: OrderBook | None = None
        self.subscribe_sig: tuple[str, str] | None = None  # (yes_id, no_id)
        self.swap_event = asyncio.Event()


def _apply_event(state: BotState, ev: dict) -> bool:
    """Update local books from a CLOB WS event. Returns True if either book changed."""
    if state.market is None or state.yes_book is None or state.no_book is None:
        return False

    asset_id = ev.get("asset_id") or ev.get("market") or ev.get("assetId")
    if asset_id == state.market.yes_token:
        book = state.yes_book
    elif asset_id == state.market.no_token:
        book = state.no_book
    else:
        return False

    ts_ms = int(ev.get("timestamp") or ev.get("ts") or clob_ws.now_ms())
    et = ev.get("event_type") or ev.get("type")

    if et == "book":
        bids = [(lv["price"], lv["size"]) for lv in ev.get("bids", [])]
        asks = [(lv["price"], lv["size"]) for lv in ev.get("asks", [])]
        book.replace(bids, asks, ts_ms)
        return True

    if et in ("price_change", "tick_size_change"):
        for ch in ev.get("changes", []) or [ev]:
            side = ch.get("side")
            price = ch.get("price")
            size = ch.get("size")
            if side is None or price is None or size is None:
                continue
            book.apply_change(str(side).upper(), float(price), float(size), ts_ms)
        return True

    return False


async def discovery_loop(state: BotState, settings: Settings, http: httpx.AsyncClient, stop: asyncio.Event):
    """Poll Gamma for the active 5m window; swap subscription when it rolls."""
    while not stop.is_set():
        try:
            found = await markets.discover(http, settings.gamma_host, settings.assets)
        except Exception as e:  # noqa: BLE001
            log.warning("discover_failed", err=str(e))
            found = []

        new_market = found[0] if found else None
        if new_market is not None:
            sig = (new_market.yes_token, new_market.no_token)
            if state.subscribe_sig != sig:
                log.info("market_swap", slug=new_market.slug, yes=sig[0][:10], no=sig[1][:10])
                state.market = new_market
                state.yes_book = OrderBook(asset_id=new_market.yes_token)
                state.no_book = OrderBook(asset_id=new_market.no_token)
                state.subscribe_sig = sig
                state.swap_event.set()

        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.discovery_interval_s)
        except asyncio.TimeoutError:
            pass


async def ws_loop(state: BotState, settings: Settings, executor: PaperExecutor, stop: asyncio.Event):
    """One WS subscription at a time, restarted on each market swap."""
    while not stop.is_set():
        if state.subscribe_sig is None:
            try:
                await asyncio.wait_for(state.swap_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            state.swap_event.clear()
            continue

        active_sig = state.subscribe_sig
        local_stop = asyncio.Event()

        async def watch_swap():
            while not stop.is_set() and not local_stop.is_set():
                if state.subscribe_sig != active_sig:
                    local_stop.set()
                    return
                try:
                    await asyncio.wait_for(state.swap_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                state.swap_event.clear()

        watcher = asyncio.create_task(watch_swap())
        try:
            async for ev in clob_ws.stream_market(settings.clob_ws_host, list(active_sig), local_stop):
                if not _apply_event(state, ev):
                    continue
                _check_arb(state, settings, executor)
                if local_stop.is_set() or stop.is_set():
                    break
        finally:
            local_stop.set()
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass


def _check_arb(state: BotState, settings: Settings, executor: PaperExecutor) -> None:
    if state.market is None or state.yes_book is None or state.no_book is None:
        return
    settle_in = state.market.time_to_settle()
    opp = arb.detect(
        slug=state.market.slug,
        yes_book=state.yes_book,
        no_book=state.no_book,
        fee_rate_bps=settings.fee_rate_bps,
        min_profit_bps=settings.profit_bps_min,
        min_order_size=state.market.minimum_order_size,
        settle_in_s=settle_in,
        min_time_to_settle_s=settings.min_time_to_settle_s,
    )
    if opp is not None:
        executor.fill(opp)


async def run_once(settings: Settings) -> int:
    """Single-shot diagnostic: discover markets, fetch top-of-book via REST, print state."""
    async with httpx.AsyncClient() as http:
        found = await markets.discover(http, settings.gamma_host, settings.assets)
        if not found:
            print("No active 5m markets discovered for assets:", settings.assets)
            return 1
        for m in found:
            print(f"\n=== {m.slug}  (settles in {m.time_to_settle():.1f}s) ===")
            for label, tok in (("YES", m.yes_token), ("NO", m.no_token)):
                try:
                    r = await http.get(
                        f"{settings.clob_http_host}/book",
                        params={"token_id": tok},
                        timeout=10.0,
                    )
                    r.raise_for_status()
                    book = r.json()
                except httpx.HTTPError as e:
                    print(f"  {label} {tok[:12]}: HTTP error {e}")
                    continue
                bids = book.get("bids") or []
                asks = book.get("asks") or []
                best_bid = max((float(lv["price"]) for lv in bids), default=None)
                best_ask = min((float(lv["price"]) for lv in asks), default=None)
                print(f"  {label} best_bid={best_bid} best_ask={best_ask}")
        return 0


async def run(settings: Settings) -> int:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            pass

    state = BotState()
    executor = PaperExecutor(settings.log_dir)
    async with httpx.AsyncClient() as http:
        await asyncio.gather(
            discovery_loop(state, settings, http, stop),
            ws_loop(state, settings, executor, stop),
        )
    executor.close()
    return 0


def cli():
    _setup_logging()
    parser = argparse.ArgumentParser("polyarb")
    parser.add_argument("--once", action="store_true", help="Diagnostic: print top of book and exit")
    args = parser.parse_args()
    settings = load()
    coro = run_once(settings) if args.once else run(settings)
    raise SystemExit(asyncio.run(coro))


if __name__ == "__main__":
    cli()
