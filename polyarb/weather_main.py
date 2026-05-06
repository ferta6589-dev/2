"""Three-loop orchestration for the weather strategy.

* ``discovery_loop`` — list new daily-temperature events on Polymarket;
  for each, blend NWS+Open-Meteo into a forecast distribution, pick the best
  3-bucket window, snapshot books via REST, and submit paper buys.
* ``ws_loop`` — single multi-asset CLOB WS subscription, restarted whenever
  ``WeatherAppState.subscribed_tokens`` changes.
* ``monitoring_loop`` — every ``nws_poll_interval_s`` on the day-of, pull
  the resolver station's observation + remaining-max forecast, classify the
  three buckets, sell losers, hold the winner to resolution.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Iterable

import httpx
import structlog

from . import clob_ws
from .config import Settings
from .forecast import (
    ForecastDistribution,
    ObservedNow,
    fetch_distribution,
    nws_observation,
    nws_today_remaining_max,
)
from .orderbook import OrderBook
from .weather_executor import WeatherPaperExecutor
from .weather_markets import WeatherEvent, fetch_weather_events
from .weather_state import WeatherAppState
from .weather_strategy import (
    classify_buckets,
    decide_entry,
    decide_exits,
    select_window,
)

log = structlog.get_logger("polyarb.weather")


def _apply_event(state: WeatherAppState, ev: dict) -> bool:
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
                f"{clob_http_host}/book",
                params={"token_id": tok},
                timeout=10.0,
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
    executor: WeatherPaperExecutor,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            events = await fetch_weather_events(http, settings.gamma_host, settings.weather_cities or None)
        except Exception as e:  # noqa: BLE001
            log.warning("weather_discovery_failed", err=str(e))
            events = []

        for event in events:
            if event.event_slug in state.events:
                continue
            if len(state.events) >= settings.weather_max_open_events:
                log.info("weather_max_events_reached", n=len(state.events))
                break

            dist = await fetch_distribution(
                http,
                settings.nws_host,
                settings.open_meteo_host,
                event.lat,
                event.lon,
                event.target_date,
                user_agent=settings.nws_user_agent,
                use_open_meteo=settings.open_meteo_enabled,
            )
            if dist is None:
                log.info("weather_no_forecast", event_slug=event.event_slug)
                continue
            window = select_window(
                event.buckets,
                dist,
                min_window_prob=settings.weather_min_window_prob,
            )
            if not window:
                log.info(
                    "weather_low_confidence",
                    event_slug=event.event_slug,
                    mu=round(dist.mu_f, 2),
                    sigma=round(dist.sigma_f, 2),
                )
                continue

            state.add_event(event, window)
            state.last_forecast[event.event_slug] = dist
            await _snapshot_books_rest(
                http, settings.clob_http_host, [b.token_yes for b in window], state
            )

            entries = decide_entry(
                window,
                state.books,
                central_max_price=settings.weather_central_max_price,
                wing_max_price=settings.weather_wing_max_price,
                per_event_budget_usd=settings.weather_per_event_budget_usd,
            )
            for order in entries:
                pos = await executor.fill_buy(event.event_slug, order)
                state.positions[(event.event_slug, order.bucket.slug)] = pos

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
                _apply_event(state, ev)
                if local_stop.is_set() or stop.is_set():
                    break
        finally:
            local_stop.set()
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass


async def monitoring_loop(
    state: WeatherAppState,
    settings: Settings,
    http: httpx.AsyncClient,
    executor: WeatherPaperExecutor,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        now = datetime.now(timezone.utc)
        for event_slug, event in list(state.events.items()):
            window = state.windows.get(event_slug, [])
            if not window:
                continue

            is_day_of = now.date() == event.target_date
            past_resolution = now.timestamp() >= event.end_ts

            obs: ObservedNow | None = None
            remaining_max: float | None = None
            if is_day_of or past_resolution:
                obs = await nws_observation(
                    http, settings.nws_host, event.station_id, settings.nws_user_agent
                )
                if obs is not None:
                    state.last_observation[event_slug] = obs
                if not past_resolution:
                    remaining_max = await nws_today_remaining_max(
                        http,
                        settings.nws_host,
                        event.lat,
                        event.lon,
                        settings.nws_user_agent,
                        now=now,
                    )
                    if remaining_max is not None:
                        state.last_remaining_max[event_slug] = remaining_max

            verdicts = classify_buckets(
                window,
                observed_high_f=(obs.temp_f if obs else None),
                forecast_remaining_max_f=remaining_max,
            )
            state.last_verdicts[event_slug] = verdicts

            positions_iter = (
                (b, state.positions.get((event_slug, b.slug)).qty)
                for b in window
                if state.positions.get((event_slug, b.slug)) is not None
                and state.positions[(event_slug, b.slug)].qty > 0
            )
            exits = decide_exits(
                ((b, q) for b, q in positions_iter),
                verdicts,
                state.books,
                hold_winner=True,
            )
            for ex in exits:
                bucket = next((b for b in window if b.slug == ex.bucket_slug), None)
                if bucket is None:
                    continue
                await executor.fill_sell(event_slug, bucket, ex)

            if past_resolution:
                for b in window:
                    pos = state.positions.get((event_slug, b.slug))
                    if pos is None or pos.qty <= 0:
                        continue
                    payout = 1.0 if verdicts.get(b.slug) == "winner" else 0.0
                    if verdicts.get(b.slug) is None and obs is not None:
                        payout = 1.0 if b.contains(obs.temp_f) else 0.0
                    await executor.record_resolve(pos, payout_per_share=payout)
                _retire_event(state, event_slug)

        try:
            await asyncio.wait_for(stop.wait(), timeout=settings.nws_poll_interval_s)
        except asyncio.TimeoutError:
            pass


def _retire_event(state: WeatherAppState, event_slug: str) -> None:
    window = state.windows.pop(event_slug, [])
    state.events.pop(event_slug, None)
    state.last_forecast.pop(event_slug, None)
    state.last_observation.pop(event_slug, None)
    state.last_remaining_max.pop(event_slug, None)
    state.last_verdicts.pop(event_slug, None)
    for b in window:
        still_in_use = any(
            b.token_yes in (bb.token_yes for bb in w)
            for w in state.windows.values()
        )
        if not still_in_use:
            state.subscribed_tokens.discard(b.token_yes)
            state.books.pop(b.token_yes, None)
    state.swap_event.set()


async def run(
    state: WeatherAppState,
    settings: Settings,
    executor: WeatherPaperExecutor,
    http: httpx.AsyncClient,
    stop: asyncio.Event,
) -> None:
    state.mode = "live"
    await asyncio.gather(
        discovery_loop(state, settings, http, executor, stop),
        ws_loop(state, settings, stop),
        monitoring_loop(state, settings, http, executor, stop),
    )
