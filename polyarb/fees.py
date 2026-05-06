"""Polymarket dynamic taker-fee model.

Polymarket charges a taker fee proportional to ``min(price, 1-price)``, which
peaks at the 50/50 price and goes to zero at $0 or $1. ``fee_rate_bps`` is the
nominal rate; the realized peak rate is ``fee_rate_bps * 0.5 / 1.0``. With the
default ``fee_rate_bps = 200`` (2.00%), peak realized fee is ~1.00% per side, in
line with the documented 1.56% combined-side fee at 50% probability for 5-minute
crypto markets when both legs cross at the same price.
"""

from __future__ import annotations


def fee_for(price: float, size: float, fee_rate_bps: int) -> float:
    """USD fee for a fill of ``size`` shares at ``price`` (each share 0..1)."""
    if price <= 0 or price >= 1 or size <= 0:
        return 0.0
    rate = fee_rate_bps / 10_000.0
    return rate * min(price, 1.0 - price) * size
