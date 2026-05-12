from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque

from .metar import DailyMaxTracker, MetarReport
from .orderbook import OrderBook
from .weather_executor import Position
from .weather_markets import WeatherBucket, WeatherEvent
from .weather_strategy import Verdict


class WeatherAppState:
    """State for the METAR-driven weather strategy."""

    def __init__(self, max_actions: int = 200):
        self.events: dict[str, WeatherEvent] = {}
        self.books: dict[str, OrderBook] = {}
        self.positions: dict[tuple[str, str], Position] = {}
        self.trackers: dict[str, DailyMaxTracker] = {}      # by event_slug
        self.last_metar: dict[str, MetarReport] = {}        # by station_id
        self.last_verdicts: dict[str, dict[str, Verdict]] = {}
        self.metar_poll_count: dict[str, int] = {}          # by station_id
        self.recent_actions: Deque[dict] = deque(maxlen=max_actions)
        self.subscribed_tokens: set[str] = set()
        self.swap_event: asyncio.Event = asyncio.Event()
        self.mode: str = "idle"
        self.tick: int = 0

    def add_event(self, event: WeatherEvent, tracker: DailyMaxTracker) -> None:
        self.events[event.event_slug] = event
        self.trackers[event.event_slug] = tracker
        for b in event.buckets:
            self.books.setdefault(b.token_yes, OrderBook(asset_id=b.token_yes))
            self.subscribed_tokens.add(b.token_yes)
        self.swap_event.set()

    def record_action(self, rec: dict) -> None:
        self.recent_actions.appendleft(rec)

    def stations_in_use(self) -> set[str]:
        return {e.station_id for e in self.events.values()}

    def snapshot(self) -> dict[str, Any]:
        events_out: list[dict] = []
        realized = 0.0
        open_cost = 0.0
        open_mtm = 0.0
        for slug, event in self.events.items():
            verdicts = self.last_verdicts.get(slug, {})
            tracker = self.trackers.get(slug)
            buckets_out: list[dict] = []
            for b in event.buckets:
                book = self.books.get(b.token_yes)
                best_bid = book.best_bid() if book else None
                best_ask = book.best_ask() if book else None
                pos = self.positions.get((slug, b.slug))
                if pos:
                    realized += pos.realized_pnl
                    if pos.qty > 0:
                        open_cost += pos.qty * pos.avg_cost
                        mark_px = best_bid.price if best_bid else pos.avg_cost
                        open_mtm += pos.qty * mark_px
                buckets_out.append({
                    "slug": b.slug,
                    "title": b.title,
                    "lo_c": b.lo_c,
                    "hi_c": b.hi_c,
                    "best_bid": _level(best_bid),
                    "best_ask": _level(best_ask),
                    "verdict": verdicts.get(b.slug),
                    "position": (
                        {
                            "qty": round(pos.qty, 4),
                            "avg_cost": round(pos.avg_cost, 4),
                            "realized_pnl": round(pos.realized_pnl, 4),
                        }
                        if pos else None
                    ),
                })
            metar = self.last_metar.get(event.station_id)
            events_out.append({
                "event_slug": slug,
                "city": event.city,
                "target_date": event.target_date.isoformat(),
                "station_id": event.station_id,
                "settle_in_s": round(event.time_to_resolution(), 1),
                "observed_max_c": tracker.max_c if tracker else None,
                "rounded_max_c": tracker.rounded_max_c if tracker else None,
                "last_metar": (
                    {
                        "ts": metar.observation_ts.isoformat(),
                        "temperature_c": metar.temperature_c,
                        "source": metar.source,
                        "raw": metar.raw,
                    } if metar else None
                ),
                "buckets": buckets_out,
            })
        return {
            "mode": self.mode,
            "tick": self.tick,
            "events": events_out,
            "metar_polls": dict(self.metar_poll_count),
            "totals": {
                "open_events": len(self.events),
                "realized_pnl": round(realized, 4),
                "open_cost_usd": round(open_cost, 4),
                "open_mtm_usd": round(open_mtm, 4),
                "open_unrealized": round(open_mtm - open_cost, 4),
            },
            "recent_actions": list(self.recent_actions)[:25],
        }


def _level(level) -> dict | None:
    if level is None:
        return None
    return {"price": round(level.price, 4), "size": round(level.size, 2)}
