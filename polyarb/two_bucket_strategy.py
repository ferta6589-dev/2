"""Two-bucket pre-entry strategy with >90% confidence gate.

Logic:

1. At event discovery (2 days before resolution), compute a Gaussian forecast
   distribution ``N(μ, σ)`` from the multi-model ensemble (see ``forecast.py``).

2. **Reject events with high uncertainty**: if ``σ > WEATHER_MAX_SIGMA_C``
   (default 1.2 °C), skip — we are not confident enough.

3. Enumerate every pair of adjacent buckets. Score each by combined probability
   ``P(temp in bucket_A) + P(temp in bucket_B)`` under the forecast distribution.

4. Take the highest-scoring pair. Require ``combined_p > WEATHER_MIN_PAIR_PROB``
   (default 0.90). Otherwise skip.

5. For each chosen bucket: buy YES at best ask if price ≤ cap.

The intentional outcome: we trade only the high-confidence ~30-50% of events,
hit ≥ 90% of those (one of two buckets wins), and skip the rest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .forecast import ForecastDistribution
from .orderbook import OrderBook
from .weather_markets import WeatherBucket

PairRole = Literal["primary", "secondary"]


@dataclass(frozen=True)
class PairEntryOrder:
    bucket: WeatherBucket
    qty: float
    limit_price: float
    role: PairRole
    reason: str


@dataclass(frozen=True)
class PairSelection:
    primary: WeatherBucket  # bucket containing forecast μ
    secondary: WeatherBucket  # adjacent bucket with most remaining mass
    combined_prob: float
    p_primary: float
    p_secondary: float


def _bucket_center(b: WeatherBucket, default: float = 0.0) -> float:
    lo = b.lo_c
    hi = b.hi_c
    if lo is not None and hi is not None:
        return (lo + hi) / 2.0
    if lo is None and hi is not None:
        return hi - 1.0
    if hi is None and lo is not None:
        return lo + 1.0
    return default


def select_best_pair(
    buckets: list[WeatherBucket],
    dist: ForecastDistribution,
    *,
    min_combined_prob: float,
    max_sigma_c: float,
) -> PairSelection | None:
    """Return the best adjacent pair, or None if confidence too low."""
    if len(buckets) < 2:
        return None
    if dist.confidence_too_low(max_sigma_c):
        return None

    sorted_b = sorted(
        buckets,
        key=lambda b: _bucket_center(b, default=dist.mu_c),
    )

    best: PairSelection | None = None
    for a, b in zip(sorted_b, sorted_b[1:]):
        pa = dist.prob_in(a.lo_c, a.hi_c)
        pb = dist.prob_in(b.lo_c, b.hi_c)
        combined = pa + pb
        center_a = _bucket_center(a, default=dist.mu_c)
        center_b = _bucket_center(b, default=dist.mu_c)

        if pa >= pb:
            primary, secondary = a, b
            p_pri, p_sec = pa, pb
        else:
            primary, secondary = b, a
            p_pri, p_sec = pb, pa

        candidate = PairSelection(
            primary=primary,
            secondary=secondary,
            combined_prob=combined,
            p_primary=p_pri,
            p_secondary=p_sec,
        )
        if best is None or candidate.combined_prob > best.combined_prob:
            best = candidate

    if best is None or best.combined_prob < min_combined_prob:
        return None
    return best


def decide_pair_entry(
    pair: PairSelection,
    books: dict[str, OrderBook],
    *,
    primary_max_price: float,
    secondary_max_price: float,
    budget_usd: float,
    primary_budget_share: float = 0.6,
) -> list[PairEntryOrder]:
    """Place buys on primary + secondary at best ask, capped by price + budget."""
    primary_budget = budget_usd * primary_budget_share
    secondary_budget = budget_usd * (1 - primary_budget_share)

    orders: list[PairEntryOrder] = []
    for bucket, role, cap, alloc in (
        (pair.primary, "primary", primary_max_price, primary_budget),
        (pair.secondary, "secondary", secondary_max_price, secondary_budget),
    ):
        book = books.get(bucket.token_yes)
        if book is None:
            continue
        ask = book.best_ask()
        if ask is None or ask.size <= 0:
            continue
        if ask.price > cap:
            continue
        qty = min(ask.size, alloc / max(ask.price, 1e-6))
        if qty < bucket.minimum_order_size:
            continue
        orders.append(PairEntryOrder(
            bucket=bucket,
            qty=qty,
            limit_price=ask.price,
            role=role,
            reason=f"pair_entry_p{pair.combined_prob:.3f}",
        ))
    return orders


def explain_pair(pair: PairSelection, dist: ForecastDistribution) -> dict:
    """Compact diagnostic for logging / dashboard."""
    return {
        "mu_c": round(dist.mu_c, 2),
        "sigma_c": round(dist.sigma_c, 2),
        "primary": {
            "slug": pair.primary.slug,
            "lo_c": pair.primary.lo_c,
            "hi_c": pair.primary.hi_c,
            "p": round(pair.p_primary, 3),
        },
        "secondary": {
            "slug": pair.secondary.slug,
            "lo_c": pair.secondary.lo_c,
            "hi_c": pair.secondary.hi_c,
            "p": round(pair.p_secondary, 3),
        },
        "combined_prob": round(pair.combined_prob, 3),
        "sources": dist.sources[:8],
    }
