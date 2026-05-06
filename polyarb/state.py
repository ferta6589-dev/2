from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import asdict
from typing import Any, Deque

from .arb import Opportunity
from .markets import Market
from .orderbook import OrderBook


class AppState:
    """In-process state shared between the trading loop and the web dashboard.

    Single-threaded asyncio access — no lock needed.
    """

    def __init__(self, max_opps: int = 200):
        self.market: Market | None = None
        self.yes_book: OrderBook | None = None
        self.no_book: OrderBook | None = None
        self.recent_opps: Deque[dict] = deque(maxlen=max_opps)
        self.totals = {"opps": 0, "profit_usd": 0.0, "edge_bps_sum": 0.0}
        self.mode: str = "idle"  # "live" | "demo" | "idle"
        self.tick: int = 0
        # Live-mode coordination (unused in demo mode):
        self.subscribe_sig: tuple[str, str] | None = None
        self.swap_event: asyncio.Event = asyncio.Event()

    def record_opportunity(self, opp: Opportunity, ts: float) -> None:
        rec = {"ts": ts, **asdict(opp)}
        self.recent_opps.appendleft(rec)
        self.totals["opps"] += 1
        self.totals["profit_usd"] += opp.profit_usd
        self.totals["edge_bps_sum"] += opp.edge_bps

    def snapshot(self) -> dict[str, Any]:
        m = self.market
        yes = self.yes_book
        no = self.no_book

        def best(book: OrderBook | None, side: str):
            if book is None:
                return None
            lv = book.best_bid() if side == "bid" else book.best_ask()
            return None if lv is None else {"price": lv.price, "size": lv.size}

        yes_ask = best(yes, "ask")
        no_ask = best(no, "ask")
        yes_bid = best(yes, "bid")
        no_bid = best(no, "bid")

        ask_sum = (yes_ask["price"] + no_ask["price"]) if yes_ask and no_ask else None
        bid_sum = (yes_bid["price"] + no_bid["price"]) if yes_bid and no_bid else None

        avg_edge = (
            self.totals["edge_bps_sum"] / self.totals["opps"]
            if self.totals["opps"]
            else 0.0
        )
        return {
            "mode": self.mode,
            "tick": self.tick,
            "market": (
                {
                    "slug": m.slug,
                    "asset": m.asset,
                    "yes_token": m.yes_token,
                    "no_token": m.no_token,
                    "settle_in_s": round(m.time_to_settle(), 1),
                    "minimum_order_size": m.minimum_order_size,
                }
                if m
                else None
            ),
            "yes": {"best_bid": yes_bid, "best_ask": yes_ask},
            "no": {"best_bid": no_bid, "best_ask": no_ask},
            "ask_sum": ask_sum,
            "bid_sum": bid_sum,
            "totals": {
                "opps": self.totals["opps"],
                "profit_usd": round(self.totals["profit_usd"], 4),
                "avg_edge_bps": round(avg_edge, 2),
            },
            "recent": list(self.recent_opps)[:25],
        }
