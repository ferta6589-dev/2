"""Orchestration for the METAR-driven weather strategy.

Four async loops:

* ``discovery_loop`` — finds open Polymarket daily-temperature events, maps
  each to its resolver ICAO station, REST-snapshots the bucket books, and
  registers tokens for the WS subscription.
* ``ws_loop`` — one multi-asset CLOB subscription, restarted on token-set
  change (same pattern as the crypto path).
* ``metar_loop`` — per resolver-station, adaptively polls NOAA tgftp.
  In the **publish window** (HH:25-40 / HH:55-10 UTC) we hit it every
  ``WEATHER_METAR_FAST_PERIOD_S`` (default 20s); otherwise every
  ``WEATHER_METAR_SLOW_PERIOD_S`` (default 180s).
* ``reaction_loop`` — on each fresh METAR, updates the DailyMaxTracker,
  reclassifies buckets, and submits buys (new leader) / sells (newly
  exceeded). After ``event.end_ts`` writes a ``resolve`` record.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Iterable

import httpx
import structlog

from . import clob_ws
from .config import Settings
from .metar import (
    DailyMaxTracker,
    MetarReport,
    _PollState,
    fetch_metar,
    in_publish_window,
    next_poll_delay,
)
from .orderbook import OrderBook
from .weather_executor import WeatherPaperExecutor
from .weather_markets import WeatherEvent, fetch_weather_events
from .weather_state import WeatherAppState
from .weather_strategy import (
    Verdict,
    classify_buckets,
    decide_entry,
    decide_exits,
    diff_verdicts,
)

log = structlog.get_logger("polyarb.weather")


def _apply_book_event(state: WeatherAppState, ev: dict) -> bool:
    asset_id = ev.get("asset_id") or ev.get("market") or ev.get("assetId")
    if asset_id is None:
        return False
    book = state.books.get(asset_id)
    if book is None:
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


async def _snapshot_books_rest(
    http: httpx.AsyncClient,
    clob_http_host: str,
    tokens: Iterable[str],
    state: WeatherAppState,
) -> None:
    for tok in tokens:
        try:
            r = await http.get(
                f"{clob_http_host}/book", params={"token_id": tok}, timeout=10.0
            )
            r.raise_for_status()
            data = r.json()
        except httpx.HTTPError as e:
            log.warning("clob_book_failed", token=tok[:12], err=str(e))
            continue
        bids = [(float(lv["price"]), float(lv["size"])) for lv in (data.get("bids") or [])]
        asks = [(float(lv["price"]), float(lv["size"])) for lv in (data.get("asks") or [])]
        book = state.books.setdefault(tok, OrderBook(asset_id=tok))
        book.replace(bids, asks, clob_ws.now_ms())


async def discovery_loop(
    state: WeatherAppState,
    settings: Settings,
    http: httpx.AsyncClient,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            events = await fetch_weather_events(
                http, settings.gamma_host, settings.weather_cities or None
            )
        except Exception as e:  # noqa: BLE001
            log.warning("weather_discovery_failed", err=str(e))
            events = []

        for event in events:
            if event.event_slug in state.events:
                continue
            if len(state.events) >= settings.weather_max_open_events:
                break
            tracker = DailyMaxTracker(target_date=event.target_date)
            state.add_event(event, tracker)
            await _snapshot_books_rest(
                http, settings.clob_http_host, [b.token_yes for b in event.buckets], state
            )
            log.info(
                "weather_event_added",
                event_slug=event.event_slug,
                city=event.city,
                station=event.station_id,
                n_buckets=len(event.buckets),
            )

        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.weather_discovery_interval_s)
        except asyncio.TimeoutError:
            pass


async def ws_loop(state: WeatherAppState, settings: Settings, stop: asyncio.Event) -> None:
    while not stop.is_set():
        if not state.subscribed_tokens:
            try:
                await asyncio.wait_for(state.swap_event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                continue
            state.swap_event.clear()
            continue
        active = set(state.subscribed_tokens)
        local_stop = asyncio.Event()

        async def watch_swap():
            while not stop.is_set() and not local_stop.is_set():
                if state.subscribed_tokens != active:
                    local_stop.set()
                    return
                try:
                    await asyncio.wait_for(state.swap_event.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                state.swap_event.clear()

        watcher = asyncio.create_task(watch_swap())
        try:
            async for ev in clob_ws.stream_market(settings.clob_ws_host, list(active), local_stop):
                _apply_book_event(state, ev)
                if local_stop.is_set() or stop.is_set():
                    break
        finally:
            local_stop.set()
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass


async def _process_metar(
    state: WeatherAppState,
    settings: Settings,
    executor: WeatherPaperExecutor,
    report: MetarReport,
) -> None:
    """Re-classify all events resolving on this station; fire buys/sells."""
    for event_slug, event in list(state.events.items()):
        if event.station_id != report.station_id:
            continue
        tracker = state.trackers.setdefault(
            event_slug, DailyMaxTracker(target_date=event.target_date)
        )
        tracker.update(report)
        prev = state.last_verdicts.get(event_slug, {})
        curr = classify_buckets(event.buckets, observed_max_c=tracker.max_c)
        state.last_verdicts[event_slug] = curr
        changed = diff_verdicts(prev, curr)
        if changed:
            log.info(
                "weather_verdicts_changed",
                event_slug=event_slug,
                observed_c=tracker.max_c,
                rounded_c=tracker.rounded_max_c,
                changes={k: f"{v[0]}->{v[1]}" for k, v in changed.items()},
            )

        positions = [
            (b, state.positions[(event_slug, b.slug)].qty)
            for b in event.buckets
            if (event_slug, b.slug) in state.positions
            and state.positions[(event_slug, b.slug)].qty > 0
        ]
        exits = decide_exits(positions, curr, state.books)
        for ex in exits:
            bucket = next(b for b in event.buckets if b.slug == ex.bucket_slug)
            await executor.fill_sell(event_slug, bucket, ex)

        entries = decide_entry(
            event.buckets, curr, state.books,
            max_price=settings.weather_max_price,
            budget_usd=settings.weather_per_event_budget_usd,
        )
        for order in entries:
            key = (event_slug, order.bucket.slug)
            existing = state.positions.get(key)
            if existing is not None and existing.qty >= order.qty:
                continue  # already filled enough
            pos = await executor.fill_buy(event_slug, order)
            state.positions[key] = pos


async def _resolve_finished_events(
    state: WeatherAppState,
    executor: WeatherPaperExecutor,
) -> None:
    now = datetime.now(timezone.utc).timestamp()
    for event_slug, event in list(state.events.items()):
        if now < event.end_ts:
            continue
        tracker = state.trackers.get(event_slug)
        max_c = tracker.rounded_max_c if tracker else None
        if max_c is None:
            continue  # don't resolve without observation
        for b in event.buckets:
            pos = state.positions.get((event_slug, b.slug))
            if pos is None or pos.qty <= 0:
                continue
            payout = 1.0 if b.contains_rounded(max_c) else 0.0
            await executor.record_resolve(pos, payout_per_share=payout)
        _retire_event(state, event_slug)


def _retire_event(state: WeatherAppState, event_slug: str) -> None:
    event = state.events.pop(event_slug, None)
    state.trackers.pop(event_slug, None)
    state.last_verdicts.pop(event_slug, None)
    if event is None:
        return
    for b in event.buckets:
        still_in_use = any(
            b.token_yes in (bb.token_yes for bb in ev.buckets)
            for ev in state.events.values()
        )
        if not still_in_use:
            state.subscribed_tokens.discard(b.token_yes)
            state.books.pop(b.token_yes, None)
    state.swap_event.set()


async def metar_loop(
    state: WeatherAppState,
    settings: Settings,
    executor: WeatherPaperExecutor,
    http: httpx.AsyncClient,
    stop: asyncio.Event,
) -> None:
    poll_states: dict[str, _PollState] = {}
    while not stop.is_set():
        stations = state.stations_in_use()
        if not stations:
            try:
                await asyncio.wait_for(state.swap_event.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                continue
            state.swap_event.clear()
            continue
        in_window = in_publish_window()
        for station in stations:
            ps = poll_states.setdefault(station, _PollState())
            state.metar_poll_count[station] = state.metar_poll_count.get(station, 0) + 1
            report = await fetch_metar(
                http,
                station,
                state=ps,
                use_cycle=True,
                use_ogimet_fallback=settings.weather_ogimet_fallback,
            )
            if report is None:
                continue
            prev = state.last_metar.get(station)
            if prev is not None and report.observation_ts <= prev.observation_ts:
                continue
            state.last_metar[station] = report
            log.info(
                "metar_fresh",
                station=station,
                temp_c=report.temperature_c,
                ts=report.observation_ts.isoformat(),
                source=report.source,
            )
            await _process_metar(state, settings, executor, report)
        await _resolve_finished_events(state, executor)
        delay = next_poll_delay(
            in_window=in_window,
            fast_period_s=settings.weather_metar_fast_period_s,
            slow_period_s=settings.weather_metar_slow_period_s,
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=delay)
        except asyncio.TimeoutError:
            pass


async def run(
    state: WeatherAppState,
    settings: Settings,
    executor: WeatherPaperExecutor,
    http: httpx.AsyncClient,
    stop: asyncio.Event,
) -> None:
    state.mode = "live"
    await asyncio.gather(
        discovery_loop(state, settings, http, stop),
        ws_loop(state, settings, stop),
        metar_loop(state, settings, executor, http, stop),
    )
