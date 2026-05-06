"""Pure logic for the weather strategy: bucket selection, classification,
entry / exit decisions. No IO, easy to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from .forecast import ForecastDistribution
from .orderbook import OrderBook
from .weather_markets import WeatherBucket

Verdict = Literal["winner", "competitor", "loser"]
BucketRole = Literal["central", "wing"]


@dataclass(frozen=True)
class EntryOrder:
    bucket: WeatherBucket
    role: BucketRole
    qty: float
    limit_price: float


@dataclass(frozen=True)
class ExitOrder:
    bucket_slug: str
    qty: float
    limit_price: float
    reason: str


def select_window(
    buckets: list[WeatherBucket],
    dist: ForecastDistribution,
    *,
    min_window_prob: float,
    width: int = 3,
) -> list[WeatherBucket]:
    """Pick the contiguous ``width``-bucket window with the highest combined
    probability mass under ``dist``. Returns ``[]`` if confidence is too low.
    """
    if len(buckets) < width:
        return []
    if dist is None:
        return []
    best: tuple[float, float, list[WeatherBucket]] | None = None
    for i in range(len(buckets) - width + 1):
        window = buckets[i : i + width]
        prob = sum(dist.prob_in(b.lo_f, b.hi_f) for b in window)
        center = _window_center(window)
        if center is None:
            continue
        centeredness = -abs(center - dist.mu_f)
        key = (prob, centeredness, window)
        if best is None or key > best:
            best = key
    if best is None or best[0] < min_window_prob:
        return []
    return best[2]


def _window_center(window: list[WeatherBucket]) -> float | None:
    finite_lo = [b.lo_f for b in window if b.lo_f is not None]
    finite_hi = [b.hi_f for b in window if b.hi_f is not None]
    if not finite_lo and not finite_hi:
        return None
    lo = min(finite_lo) if finite_lo else max(finite_hi) - 5.0
    hi = max(finite_hi) if finite_hi else min(finite_lo) + 5.0
    return (lo + hi) / 2.0


def role_for(window: list[WeatherBucket], bucket: WeatherBucket) -> BucketRole:
    if not window:
        return "wing"
    return "central" if bucket is window[len(window) // 2] else "wing"


def decide_entry(
    window: list[WeatherBucket],
    books: dict[str, OrderBook],
    *,
    central_max_price: float,
    wing_max_price: float,
    per_event_budget_usd: float,
) -> list[EntryOrder]:
    """Place buys at best ask up to per-bucket budget and price caps.

    Budget is split evenly across the window's buckets.
    """
    if not window:
        return []
    out: list[EntryOrder] = []
    per_bucket_budget = per_event_budget_usd / len(window)
    for b in window:
        book = books.get(b.token_yes)
        if book is None:
            continue
        ask = book.best_ask()
        if ask is None or ask.size <= 0:
            continue
        role = role_for(window, b)
        cap = central_max_price if role == "central" else wing_max_price
        if ask.price > cap:
            continue
        max_qty_by_budget = per_bucket_budget / max(ask.price, 1e-6)
        qty = min(ask.size, max_qty_by_budget)
        if qty < b.minimum_order_size:
            continue
        out.append(EntryOrder(bucket=b, role=role, qty=qty, limit_price=ask.price))
    return out


def classify_buckets(
    window: list[WeatherBucket],
    *,
    observed_high_f: float | None,
    forecast_remaining_max_f: float | None,
) -> dict[str, Verdict]:
    """Classify each bucket as winner / competitor / loser.

    A bucket is a *loser* if it's already exceeded (``hi < observed_high``) or
    unreachable (``lo > forecast_remaining_max``). The single live bucket
    after all losers are removed is the *winner*; otherwise live buckets are
    *competitors*.
    """
    if not window:
        return {}
    verdicts: dict[str, Verdict] = {}
    live: list[WeatherBucket] = []
    for b in window:
        is_exceeded = (
            observed_high_f is not None
            and b.hi_f is not None
            and b.hi_f < observed_high_f
        )
        is_unreachable = (
            forecast_remaining_max_f is not None
            and b.lo_f is not None
            and b.lo_f > forecast_remaining_max_f
            and (observed_high_f is None or observed_high_f < b.lo_f)
        )
        if is_exceeded or is_unreachable:
            verdicts[b.slug] = "loser"
        else:
            live.append(b)
    if len(live) == 1:
        verdicts[live[0].slug] = "winner"
    else:
        for b in live:
            verdicts[b.slug] = "competitor"
    return verdicts


def decide_exits(
    positions: Iterable[tuple[WeatherBucket, float]],
    classifications: dict[str, Verdict],
    books: dict[str, OrderBook],
    *,
    hold_winner: bool = True,
    aggressive_tick: float = 0.0,
) -> list[ExitOrder]:
    """Emit a sell at best bid (optionally ``+aggressive_tick``) for any
    position whose bucket has flipped to ``loser``. Winners are held by
    default (resolution will pay $1)."""
    out: list[ExitOrder] = []
    for bucket, qty in positions:
        if qty <= 0:
            continue
        verdict = classifications.get(bucket.slug)
        if verdict == "loser":
            book = books.get(bucket.token_yes)
            bid = book.best_bid() if book else None
            limit = max(0.01, (bid.price if bid else 0.01) + aggressive_tick)
            out.append(ExitOrder(
                bucket_slug=bucket.slug,
                qty=qty,
                limit_price=round(limit, 4),
                reason="loser_classified",
            ))
        elif verdict == "winner" and not hold_winner:
            book = books.get(bucket.token_yes)
            bid = book.best_bid() if book else None
            if bid is not None:
                out.append(ExitOrder(
                    bucket_slug=bucket.slug,
                    qty=qty,
                    limit_price=round(bid.price, 4),
                    reason="winner_take_profit",
                ))
    return out
