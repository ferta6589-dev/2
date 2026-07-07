"""Coherence / Dutch-book arbitrage — the risk-free L0 layer of GHOST.

A weather event's buckets are mutually exclusive and exhaustive: exactly one
pays $1 at resolution. Therefore the fair sum of all YES prices is $1.

When the market is thin, the top-of-book can drift out of coherence:

* ``Σ best_ask(YES_i) < 1 − fee_wedge``  → buy one share of *every* bucket.
  Whatever the weather, one bucket pays $1; cost was < $1. Locked profit,
  **no weather view required**.

* ``Σ best_bid(YES_i) > 1 + fee_wedge``  → the mirror (sell every bucket).
  Only actionable if we hold inventory or the venue allows YES shorting via
  the complementary NO token, so we surface it but don't force it.

The buy-side is the money-maker on Polymarket weather books because the wings
(cheap tail buckets) are routinely left stale at 1-3¢, dragging the ask-sum
below $1 for minutes at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .orderbook import OrderBook
from .weather_markets import WeatherBucket


@dataclass(frozen=True)
class CoherenceLeg:
    bucket_slug: str
    token_yes: str
    price: float
    size: float


@dataclass(frozen=True)
class CoherenceArb:
    side: str                       # "buy_all" | "sell_all"
    legs: list[CoherenceLeg]
    matched_size: float             # shares per bucket (min across legs)
    price_sum: float                # Σ prices at the matched top-of-book
    edge_per_share: float           # $ locked per share-set after fees
    profit_usd: float               # edge_per_share * matched_size
    detail: dict = field(default_factory=dict)


def detect_dutch_book(
    buckets: list[WeatherBucket],
    books: dict[str, OrderBook],
    *,
    fee_wedge: float = 0.02,
    min_profit_usd: float = 0.50,
    max_set_cost_usd: float | None = None,
) -> CoherenceArb | None:
    """Return a risk-free buy-all opportunity if the ask-sum is below $1.

    ``fee_wedge`` accounts for Polymarket's taker fee + a safety buffer.
    ``matched_size`` is the minimum top-of-book ask size across all buckets,
    because we must buy the *same* number of share-sets to guarantee the $1.
    """
    if len(buckets) < 2:
        return None

    legs: list[CoherenceLeg] = []
    price_sum = 0.0
    min_size = float("inf")
    for b in buckets:
        book = books.get(b.token_yes)
        if book is None:
            return None
        ask = book.best_ask()
        if ask is None or ask.size <= 0:
            return None  # incomplete book — cannot guarantee coverage
        legs.append(CoherenceLeg(b.slug, b.token_yes, ask.price, ask.size))
        price_sum += ask.price
        min_size = min(min_size, ask.size)

    edge = 1.0 - price_sum - fee_wedge
    if edge <= 0:
        return None

    if max_set_cost_usd is not None:
        affordable = max_set_cost_usd / max(price_sum, 1e-6)
        min_size = min(min_size, affordable)

    profit = edge * min_size
    if profit < min_profit_usd:
        return None

    return CoherenceArb(
        side="buy_all",
        legs=legs,
        matched_size=round(min_size, 4),
        price_sum=round(price_sum, 4),
        edge_per_share=round(edge, 4),
        profit_usd=round(profit, 4),
        detail={"n_buckets": len(buckets), "fee_wedge": fee_wedge},
    )


def detect_overround(
    buckets: list[WeatherBucket],
    books: dict[str, OrderBook],
    *,
    fee_wedge: float = 0.02,
) -> CoherenceArb | None:
    """Detect Σ best_bid > $1 (sell-all opportunity). Informational unless we
    hold inventory / can short via NO tokens."""
    if len(buckets) < 2:
        return None
    legs: list[CoherenceLeg] = []
    price_sum = 0.0
    min_size = float("inf")
    for b in buckets:
        book = books.get(b.token_yes)
        if book is None:
            return None
        bid = book.best_bid()
        if bid is None or bid.size <= 0:
            return None
        legs.append(CoherenceLeg(b.slug, b.token_yes, bid.price, bid.size))
        price_sum += bid.price
        min_size = min(min_size, bid.size)

    edge = price_sum - 1.0 - fee_wedge
    if edge <= 0:
        return None
    return CoherenceArb(
        side="sell_all",
        legs=legs,
        matched_size=round(min_size, 4),
        price_sum=round(price_sum, 4),
        edge_per_share=round(edge, 4),
        profit_usd=round(edge * min_size, 4),
        detail={"n_buckets": len(buckets), "fee_wedge": fee_wedge},
    )
