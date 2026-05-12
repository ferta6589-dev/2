"""Pure logic for the METAR-driven weather strategy.

We classify each Polymarket bucket purely from the **observed running daily
max** (whole °C, matching the resolver's published precision):

* **LEADER** — the bucket that contains the current rounded max.
* **EXCEEDED** — buckets whose upper bound is already below the current max.
* **UNREACHED** — buckets whose lower bound is above the current max.

Trade decisions: when a new METAR shifts the leader,

1. **Buy** YES on the new leader at best ask, capped by ``WEATHER_MAX_PRICE``,
   sized by ``WEATHER_PER_EVENT_BUDGET_USD``. This is the "buy before quotes
   reprice" play.
2. **Sell** YES on freshly-exceeded buckets at best bid (recover what we can
   before the market reprices them to ~0).
3. **Hold** UNREACHED buckets — they could still become the leader if
   afternoon heat pushes the max into them.

After resolution time, the LEADER bucket (if any) pays $1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal

from .orderbook import OrderBook
from .weather_markets import WeatherBucket

Verdict = Literal["leader", "exceeded", "unreached"]


@dataclass(frozen=True)
class EntryOrder:
    bucket: WeatherBucket
    qty: float
    limit_price: float
    reason: str


@dataclass(frozen=True)
class ExitOrder:
    bucket_slug: str
    qty: float
    limit_price: float
    reason: str


def classify_buckets(
    buckets: list[WeatherBucket],
    *,
    observed_max_c: float | None,
) -> dict[str, Verdict]:
    """Pure classification from the running observed max.

    Resolver uses whole-degree °C, so we truncate ``observed_max_c`` toward
    zero before comparing to bucket bounds. A bucket with ``[lo, hi]`` matches
    if ``lo <= truncated_max <= hi`` (inclusive both ends — Polymarket buckets
    are typically half-open in practice, but treating the upper edge
    inclusively only matters when the boundary equals a resolver value, and
    we'd rather not auto-exit there).
    """
    if observed_max_c is None:
        return {b.slug: "unreached" for b in buckets}
    truncated = int(math.floor(observed_max_c)) if observed_max_c >= 0 else int(math.ceil(observed_max_c))
    out: dict[str, Verdict] = {}
    for b in buckets:
        if b.hi_c is not None and truncated > int(b.hi_c):
            out[b.slug] = "exceeded"
        elif b.lo_c is not None and truncated < int(b.lo_c):
            out[b.slug] = "unreached"
        else:
            out[b.slug] = "leader"
    return out


def decide_entry(
    buckets: list[WeatherBucket],
    verdicts: dict[str, Verdict],
    books: dict[str, OrderBook],
    *,
    max_price: float,
    budget_usd: float,
) -> list[EntryOrder]:
    """Buy YES on the current LEADER bucket(s) at best ask."""
    leaders = [b for b in buckets if verdicts.get(b.slug) == "leader"]
    if not leaders:
        return []
    per_leader = budget_usd / len(leaders)
    out: list[EntryOrder] = []
    for b in leaders:
        book = books.get(b.token_yes)
        if book is None:
            continue
        ask = book.best_ask()
        if ask is None or ask.size <= 0:
            continue
        if ask.price > max_price:
            continue
        qty = min(ask.size, per_leader / max(ask.price, 1e-6))
        if qty < b.minimum_order_size:
            continue
        out.append(EntryOrder(
            bucket=b,
            qty=qty,
            limit_price=ask.price,
            reason="leader_on_metar",
        ))
    return out


def decide_exits(
    positions: Iterable[tuple[WeatherBucket, float]],
    verdicts: dict[str, Verdict],
    books: dict[str, OrderBook],
    *,
    aggressive_tick: float = 0.0,
) -> list[ExitOrder]:
    """Sell YES on EXCEEDED buckets immediately at best bid."""
    out: list[ExitOrder] = []
    for bucket, qty in positions:
        if qty <= 0:
            continue
        if verdicts.get(bucket.slug) != "exceeded":
            continue
        book = books.get(bucket.token_yes)
        bid = book.best_bid() if book else None
        limit = max(0.01, (bid.price if bid else 0.01) + aggressive_tick)
        out.append(ExitOrder(
            bucket_slug=bucket.slug,
            qty=qty,
            limit_price=round(limit, 4),
            reason="exceeded_by_observation",
        ))
    return out


def diff_verdicts(
    prev: dict[str, Verdict],
    curr: dict[str, Verdict],
) -> dict[str, tuple[Verdict | None, Verdict]]:
    """Buckets whose verdict changed: ``{slug: (prev, curr)}``."""
    out: dict[str, tuple[Verdict | None, Verdict]] = {}
    for slug, v in curr.items():
        p = prev.get(slug)
        if p != v:
            out[slug] = (p, v)
    return out
