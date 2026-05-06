from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Level:
    price: float
    size: float


@dataclass
class OrderBook:
    """Single-side-stored binary token orderbook.

    Bids are kept in descending price order, asks in ascending. Sizes are in
    shares (each share pays out $1 if the token wins).
    """

    asset_id: str
    bids: list[Level] = field(default_factory=list)
    asks: list[Level] = field(default_factory=list)
    timestamp_ms: int = 0

    def replace(self, bids: Iterable[tuple[float, float]], asks: Iterable[tuple[float, float]], ts_ms: int) -> None:
        self.bids = sorted(
            (Level(float(p), float(s)) for p, s in bids if float(s) > 0),
            key=lambda lv: lv.price,
            reverse=True,
        )
        self.asks = sorted(
            (Level(float(p), float(s)) for p, s in asks if float(s) > 0),
            key=lambda lv: lv.price,
        )
        self.timestamp_ms = ts_ms

    def apply_change(self, side: str, price: float, size: float, ts_ms: int) -> None:
        levels = self.bids if side == "BUY" else self.asks
        for i, lv in enumerate(levels):
            if lv.price == price:
                if size <= 0:
                    levels.pop(i)
                else:
                    lv.size = size
                self.timestamp_ms = ts_ms
                return
        if size > 0:
            levels.append(Level(price, size))
            levels.sort(key=lambda lv: lv.price, reverse=(side == "BUY"))
        self.timestamp_ms = ts_ms

    def best_ask(self) -> Level | None:
        return self.asks[0] if self.asks else None

    def best_bid(self) -> Level | None:
        return self.bids[0] if self.bids else None
