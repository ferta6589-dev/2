"""Synthetic METAR + Polymarket feed for tests/demo.

Walks through a realistic Moscow-on-May-12 trading day at UUWW (Vnukovo): a
morning low, an afternoon climb that crosses bucket boundaries, and a final
fix. Each climb step emits a fake METAR, the strategy reacts, and we close
out with a resolve.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import structlog

from . import clob_ws
from .config import Settings
from .metar import DailyMaxTracker, MetarReport
from .orderbook import OrderBook
from .weather_executor import WeatherPaperExecutor
from .weather_markets import WeatherBucket, WeatherEvent
from .weather_state import WeatherAppState
from .weather_strategy import (
    classify_buckets,
    decide_entry,
    decide_exits,
)

log = structlog.get_logger("polyarb.weather_sim")


def _build_moscow_event() -> WeatherEvent:
    today = datetime.now(timezone.utc).date()
    starts_c = [11, 13, 15, 17, 19, 21, 23]
    buckets: list[WeatherBucket] = []
    for lo in starts_c:
        hi = lo + 2
        buckets.append(WeatherBucket(
            slug=f"sim-moscow-{today.isoformat()}-{lo}-{hi}",
            title=f"{lo}-{hi}°C",
            token_yes=f"DEMO_YES_{lo}",
            token_no=f"DEMO_NO_{lo}",
            lo_c=float(lo),
            hi_c=float(hi),
            minimum_order_size=5.0,
        ))
    end_ts = datetime.combine(today, datetime.min.time(), tzinfo=timezone.utc).timestamp() + 86400
    return WeatherEvent(
        event_slug=f"sim-moscow-weather-{today.isoformat()}",
        city="MOSCOW",
        target_date=today,
        station_id="UUWW",
        lat=55.59,
        lon=37.27,
        end_ts=end_ts,
        buckets=buckets,
    )


def _seed_books(state: WeatherAppState, event: WeatherEvent) -> None:
    ts_ms = clob_ws.now_ms()
    for b in event.buckets:
        book = state.books.setdefault(b.token_yes, OrderBook(asset_id=b.token_yes))
        center = (b.lo_c or 0.0) + 1.0
        ask_px = max(0.04, min(0.40, 0.40 - abs(center - 17) * 0.05))
        bid_px = max(0.01, ask_px - 0.04)
        book.replace([(round(bid_px, 3), 250.0)], [(round(ask_px, 3), 250.0)], ts_ms)


def _drift_books(state: WeatherAppState, event: WeatherEvent, observed_c: float) -> None:
    """Only drop EXCEEDED buckets toward 0; leave LEADER and UNREACHED at their
    seed prices. That asymmetry is the very lag the strategy exploits — the
    market catches up to losers faster than to the new leader."""
    ts_ms = clob_ws.now_ms()
    truncated = int(observed_c) if observed_c >= 0 else -int(-observed_c // 1)
    for b in event.buckets:
        book = state.books.get(b.token_yes)
        if book is None:
            continue
        if b.hi_c is not None and truncated > int(b.hi_c):
            book.replace([(0.02, 200.0)], [(0.05, 200.0)], ts_ms)


async def run_demo(
    state: WeatherAppState,
    settings: Settings,
    executor: WeatherPaperExecutor,
    stop: asyncio.Event,
    *,
    tick_period_s: float = 0.3,
) -> None:
    """One full sim day on Moscow / UUWW."""
    state.mode = "demo"
    event = _build_moscow_event()
    _seed_books(state, event)
    tracker = DailyMaxTracker(target_date=event.target_date)
    state.add_event(event, tracker)

    base = datetime.combine(event.target_date, datetime.min.time(), tzinfo=timezone.utc)
    timeline = [
        (5, 8.5), (7, 10.0), (8, 12.3), (10, 14.8),
        (12, 16.2), (13, 17.4), (14, 18.1), (15, 18.6),
        (16, 18.8), (17, 18.3), (18, 17.0), (20, 14.5),
    ]
    from .weather_strategy import classify_buckets, decide_entry, decide_exits

    for hour, temp_c in timeline:
        if stop.is_set():
            return
        state.tick += 1
        observation_ts = base + timedelta(hours=hour)
        report = MetarReport(
            station_id="UUWW",
            observation_ts=observation_ts,
            temperature_c=temp_c,
            raw=f"UUWW {observation_ts.strftime('%d%H%M')}Z DEMO",
            source="sim",
        )
        tracker.update(report)
        state.last_metar["UUWW"] = report
        _drift_books(state, event, tracker.max_c or 0.0)

        verdicts = classify_buckets(event.buckets, observed_max_c=tracker.max_c)
        state.last_verdicts[event.event_slug] = verdicts

        positions = [
            (b, state.positions[(event.event_slug, b.slug)].qty)
            for b in event.buckets
            if (event.event_slug, b.slug) in state.positions
        ]
        for ex in decide_exits(positions, verdicts, state.books):
            bucket = next(b for b in event.buckets if b.slug == ex.bucket_slug)
            await executor.fill_sell(event.event_slug, bucket, ex)

        for order in decide_entry(
            event.buckets, verdicts, state.books,
            max_price=settings.weather_max_price,
            budget_usd=settings.weather_per_event_budget_usd,
        ):
            key = (event.event_slug, order.bucket.slug)
            existing = state.positions.get(key)
            if existing is not None and existing.qty >= order.qty:
                continue
            pos = await executor.fill_buy(event.event_slug, order)
            state.positions[key] = pos

        log.info(
            "sim_tick",
            hour=hour,
            temp_c=temp_c,
            max_c=tracker.max_c,
            rounded_max_c=tracker.rounded_max_c,
        )
        try:
            await asyncio.wait_for(stop.wait(), timeout=tick_period_s)
        except asyncio.TimeoutError:
            pass

    rounded = tracker.rounded_max_c or 0
    for b in event.buckets:
        pos = state.positions.get((event.event_slug, b.slug))
        if pos is None or pos.qty <= 0:
            continue
        payout = 1.0 if b.contains_rounded(rounded) else 0.0
        await executor.record_resolve(pos, payout_per_share=payout)
