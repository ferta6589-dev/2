from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Deque

from .forecast import ForecastDistribution, ObservedNow
from .orderbook import OrderBook
from .weather_executor import Position
from .weather_markets import WeatherBucket, WeatherEvent
from .weather_strategy import Verdict


class WeatherAppState:
    """State shared between the weather strategy loops and the dashboard."""

    def __init__(self, max_actions: int = 200):
        self.events: dict[str, WeatherEvent] = {}
        self.windows: dict[str, list[WeatherBucket]] = {}
        self.books: dict[str, OrderBook] = {}
        self.positions: dict[tuple[str, str], Position] = {}
        self.last_forecast: dict[str, ForecastDistribution] = {}
        self.last_observation: dict[str, ObservedNow] = {}
        self.last_remaining_max: dict[str, float] = {}
        self.last_verdicts: dict[str, dict[str, Verdict]] = {}
        self.recent_actions: Deque[dict] = deque(maxlen=max_actions)
        self.subscribed_tokens: set[str] = set()
        self.swap_event: asyncio.Event = asyncio.Event()
        self.mode: str = "idle"
        self.tick: int = 0

    def add_event(self, event: WeatherEvent, window: list[WeatherBucket]) -> None:
        self.events[event.event_slug] = event
        self.windows[event.event_slug] = window
        for b in window:
            self.books.setdefault(b.token_yes, OrderBook(asset_id=b.token_yes))
            self.subscribed_tokens.add(b.token_yes)
        self.swap_event.set()

    def record_action(self, rec: dict) -> None:
        self.recent_actions.appendleft(rec)

    def snapshot(self) -> dict[str, Any]:
        events_out: list[dict] = []
        total_realized = 0.0
        total_open_cost = 0.0
        total_open_mtm = 0.0
        for slug, event in self.events.items():
            window = self.windows.get(slug, [])
            verdicts = self.last_verdicts.get(slug, {})
            forecast = self.last_forecast.get(slug)
            obs = self.last_observation.get(slug)
            buckets_out: list[dict] = []
            for b in window:
                book = self.books.get(b.token_yes)
                best_bid = book.best_bid() if book else None
                best_ask = book.best_ask() if book else None
                pos = self.positions.get((slug, b.slug))
                if pos:
                    total_realized += pos.realized_pnl
                    if pos.qty > 0:
                        total_open_cost += pos.qty * pos.avg_cost
                        mark_px = best_bid.price if best_bid else pos.avg_cost
                        total_open_mtm += pos.qty * mark_px
                buckets_out.append({
                    "slug": b.slug,
                    "title": b.title,
                    "lo_f": b.lo_f,
                    "hi_f": b.hi_f,
                    "best_bid": _level(best_bid),
                    "best_ask": _level(best_ask),
                    "verdict": verdicts.get(b.slug),
                    "position": (
                        {
                            "qty": round(pos.qty, 4),
                            "avg_cost": round(pos.avg_cost, 4),
                            "realized_pnl": round(pos.realized_pnl, 4),
                        }
                        if pos
                        else None
                    ),
                })
            events_out.append({
                "event_slug": slug,
                "city": event.city,
                "target_date": event.target_date.isoformat(),
                "station_id": event.station_id,
                "settle_in_s": round(event.time_to_resolution(), 1),
                "forecast": (
                    {
                        "mu_f": round(forecast.mu_f, 2),
                        "sigma_f": round(forecast.sigma_f, 2),
                        "sources": forecast.sources,
                    }
                    if forecast
                    else None
                ),
                "observation_f": round(obs.temp_f, 2) if obs else None,
                "remaining_max_f": (
                    round(self.last_remaining_max[slug], 2)
                    if slug in self.last_remaining_max
                    else None
                ),
                "buckets": buckets_out,
            })
        return {
            "mode": self.mode,
            "tick": self.tick,
            "events": events_out,
            "totals": {
                "open_events": len(self.events),
                "realized_pnl": round(total_realized, 4),
                "open_cost_usd": round(total_open_cost, 4),
                "open_mtm_usd": round(total_open_mtm, 4),
                "open_unrealized": round(total_open_mtm - total_open_cost, 4),
            },
            "recent_actions": list(self.recent_actions)[:25],
        }


def _level(level) -> dict | None:
    if level is None:
        return None
    return {"price": round(level.price, 4), "size": round(level.size, 2)}
