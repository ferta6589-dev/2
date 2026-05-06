"""Synthetic feed for the weather strategy.

Builds an in-process fake event with seven 5°F buckets, a static forecast
distribution, and a scripted observation timeline that walks through "below the
window → in the window → past the window" so we exercise the full classify /
exit / resolve flow without hitting NWS or Polymarket.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import date, datetime, timedelta, timezone

import structlog

from . import clob_ws
from .config import Settings
from .forecast import ForecastDistribution, ObservedNow
from .orderbook import OrderBook
from .weather_executor import WeatherPaperExecutor
from .weather_markets import WeatherBucket, WeatherEvent
from .weather_state import WeatherAppState
from .weather_strategy import (
    classify_buckets,
    decide_entry,
    decide_exits,
    select_window,
)

log = structlog.get_logger("polyarb.weather_sim")


def _build_event(target: date) -> WeatherEvent:
    starts = [55, 60, 65, 70, 75, 80, 85]
    buckets: list[WeatherBucket] = []
    for i, lo in enumerate(starts):
        hi = lo + 5
        slug = f"sim-nyc-{target.isoformat()}-{lo}-{hi}"
        buckets.append(WeatherBucket(
            slug=slug,
            title=f"Highest temperature in NYC on {target.isoformat()} between {lo} and {hi} degrees",
            token_yes=f"DEMO_YES_{lo}",
            token_no=f"DEMO_NO_{lo}",
            lo_f=float(lo),
            hi_f=float(hi),
            minimum_order_size=5.0,
            minimum_tick_size=0.01,
        ))
    end_ts = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc).timestamp() + 86400
    return WeatherEvent(
        event_slug=f"sim-nyc-weather-{target.isoformat()}",
        city="NYC",
        target_date=target,
        station_id="KNYC",
        lat=40.78,
        lon=-73.97,
        end_ts=end_ts,
        buckets=buckets,
    )


def _seed_books(state: WeatherAppState, event: WeatherEvent) -> None:
    """Cheap asks on every bucket so decide_entry has something to fill."""
    ts_ms = clob_ws.now_ms()
    for b in event.buckets:
        book = state.books.setdefault(b.token_yes, OrderBook(asset_id=b.token_yes))
        center = (b.lo_f or 0.0) + 2.5
        ask_px = max(0.05, min(0.45, 0.45 - abs(center - 70) * 0.05))
        bid_px = max(0.01, ask_px - 0.04)
        book.replace(
            bids=[(round(bid_px, 3), 200.0)],
            asks=[(round(ask_px, 3), 200.0)],
            ts_ms=ts_ms,
        )


def _drift_books(state: WeatherAppState, event: WeatherEvent, observed_high: float) -> None:
    """As the day progresses, push asks of unreachable / exceeded buckets up
    (so their bids drop, simulating market consensus moving away)."""
    ts_ms = clob_ws.now_ms()
    for b in event.buckets:
        book = state.books.get(b.token_yes)
        if book is None:
            continue
        if b.contains(observed_high):
            ask_px = 0.55
            bid_px = 0.50
        elif b.hi_f is not None and b.hi_f < observed_high:
            ask_px = 0.04
            bid_px = 0.02
        else:
            ask_px = 0.10
            bid_px = 0.06
        book.replace(
            bids=[(round(bid_px, 3), 150.0)],
            asks=[(round(ask_px, 3), 150.0)],
            ts_ms=ts_ms,
        )


async def run_demo(
    state: WeatherAppState,
    settings: Settings,
    executor: WeatherPaperExecutor,
    stop: asyncio.Event,
    *,
    tick_period_s: float = 0.5,
    days_ahead: int = 1,
) -> None:
    """Runs through entry → classification → exit → resolve in a few seconds."""
    state.mode = "demo"
    target = (datetime.now(timezone.utc).date() + timedelta(days=days_ahead))
    event = _build_event(target)
    _seed_books(state, event)

    dist = ForecastDistribution(mu_f=72.0, sigma_f=2.0, sources=["demo"])
    window = select_window(
        event.buckets,
        dist,
        min_window_prob=settings.weather_min_window_prob,
    )
    if not window:
        log.warning("sim_low_confidence")
        return
    state.add_event(event, window)
    state.last_forecast[event.event_slug] = dist

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

    timeline: list[tuple[float, float | None]] = [
        (62.0, 75.0),
        (66.0, 75.0),
        (71.0, 73.0),
        (72.5, 72.5),
        (72.5, None),
    ]

    for observed_high, remaining_max in timeline:
        if stop.is_set():
            return
        state.tick += 1
        _drift_books(state, event, observed_high)
        state.last_observation[event.event_slug] = ObservedNow(
            temp_f=observed_high, ts_utc=time.time()
        )
        if remaining_max is not None:
            state.last_remaining_max[event.event_slug] = remaining_max
        else:
            state.last_remaining_max.pop(event.event_slug, None)

        verdicts = classify_buckets(
            window,
            observed_high_f=observed_high,
            forecast_remaining_max_f=remaining_max,
        )
        state.last_verdicts[event.event_slug] = verdicts

        positions = [
            (b, state.positions[(event.event_slug, b.slug)].qty)
            for b in window
            if (event.event_slug, b.slug) in state.positions
        ]
        exits = decide_exits(positions, verdicts, state.books, hold_winner=True)
        for ex in exits:
            bucket = next(b for b in window if b.slug == ex.bucket_slug)
            await executor.fill_sell(event.event_slug, bucket, ex)

        try:
            await asyncio.wait_for(stop.wait(), timeout=tick_period_s)
        except asyncio.TimeoutError:
            pass

    final_obs = timeline[-1][0]
    for b in window:
        pos = state.positions.get((event.event_slug, b.slug))
        if pos is None or pos.qty <= 0:
            continue
        payout = 1.0 if b.contains(final_obs) else 0.0
        await executor.record_resolve(pos, payout_per_share=payout)
