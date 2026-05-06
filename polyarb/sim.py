"""Synthetic data feed for the arbitrage detector.

Produces a random-walking fair value, periodically pushes the asks below $1 to
simulate the kind of momentary mispricings the bot is built to capture, and
rolls the market every 5 minutes — the same window cadence as the real
Polymarket BTC Up/Down 5m markets.
"""

from __future__ import annotations

import asyncio
import random
import time

import structlog

from . import arb, clob_ws
from .config import Settings
from .executor import PaperExecutor
from .markets import WINDOW_S, Market, current_window_ts
from .orderbook import OrderBook
from .state import AppState

log = structlog.get_logger("polyarb.sim")


def _build_market(asset: str = "BTC") -> Market:
    ts = current_window_ts()
    return Market(
        slug=f"{asset.lower()}-updown-5m-{ts}-DEMO",
        asset=asset,
        yes_token=f"DEMO_{asset}_YES_{ts}",
        no_token=f"DEMO_{asset}_NO_{ts}",
        end_ts=float(ts + WINDOW_S),
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )


class Simulator:
    def __init__(self, state: AppState, settings: Settings, executor: PaperExecutor):
        self.state = state
        self.settings = settings
        self.executor = executor
        self.fair_value = 0.5
        self.tick_period_s = 0.5
        # Inject a clear arb opportunity roughly every 8-15 seconds.
        self._next_arb_in = random.randint(16, 30)

    async def run(self, stop: asyncio.Event) -> None:
        self.state.mode = "demo"
        self._roll_market()
        while not stop.is_set():
            self.state.tick += 1
            self._step()
            self._maybe_check_arb()
            if self.state.market is not None and self.state.market.time_to_settle() <= 0:
                self._roll_market()
            try:
                await asyncio.wait_for(stop.wait(), timeout=self.tick_period_s)
            except asyncio.TimeoutError:
                pass

    def _roll_market(self) -> None:
        market = _build_market("BTC")
        self.state.market = market
        self.state.yes_book = OrderBook(asset_id=market.yes_token)
        self.state.no_book = OrderBook(asset_id=market.no_token)
        self.fair_value = random.uniform(0.30, 0.70)
        log.info("sim_market_rolled", slug=market.slug)

    def _step(self) -> None:
        if self.state.yes_book is None or self.state.no_book is None:
            return

        # Random walk fair value, clipped away from the boundaries.
        self.fair_value += random.uniform(-0.012, 0.012)
        self.fair_value = max(0.10, min(0.90, self.fair_value))

        spread = random.uniform(0.01, 0.025)
        yes_ask_px = round(self.fair_value + spread / 2, 3)
        yes_bid_px = round(self.fair_value - spread / 2, 3)
        no_ask_px = round((1 - self.fair_value) + spread / 2, 3)
        no_bid_px = round((1 - self.fair_value) - spread / 2, 3)

        self._next_arb_in -= 1
        if self._next_arb_in <= 0:
            # Push both asks down so YES_ask + NO_ask drops well under $1
            # (after fees there's still real edge).
            squeeze = random.uniform(0.02, 0.05)
            yes_ask_px = max(0.02, round(yes_ask_px - squeeze, 3))
            no_ask_px = max(0.02, round(no_ask_px - squeeze, 3))
            self._next_arb_in = random.randint(16, 30)
            log.info(
                "sim_arb_injected",
                yes_ask=yes_ask_px,
                no_ask=no_ask_px,
                ask_sum=round(yes_ask_px + no_ask_px, 4),
            )

        yes_size = round(random.uniform(40, 250), 2)
        no_size = round(random.uniform(40, 250), 2)

        ts_ms = clob_ws.now_ms()
        self.state.yes_book.replace(
            bids=[(yes_bid_px, yes_size)],
            asks=[(yes_ask_px, yes_size)],
            ts_ms=ts_ms,
        )
        self.state.no_book.replace(
            bids=[(no_bid_px, no_size)],
            asks=[(no_ask_px, no_size)],
            ts_ms=ts_ms,
        )

    def _maybe_check_arb(self) -> None:
        market = self.state.market
        yes = self.state.yes_book
        no = self.state.no_book
        if market is None or yes is None or no is None:
            return
        opp = arb.detect(
            slug=market.slug,
            yes_book=yes,
            no_book=no,
            fee_rate_bps=self.settings.fee_rate_bps,
            min_profit_bps=self.settings.profit_bps_min,
            min_order_size=market.minimum_order_size,
            settle_in_s=market.time_to_settle(),
            min_time_to_settle_s=self.settings.min_time_to_settle_s,
        )
        if opp is not None:
            self.executor.fill(opp)


async def run_demo(state: AppState, settings: Settings, executor: PaperExecutor, stop: asyncio.Event) -> None:
    sim = Simulator(state, settings, executor)
    await sim.run(stop)


def _ts() -> float:
    return time.time()
