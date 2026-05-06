from __future__ import annotations

from dataclasses import dataclass

from .fees import fee_for
from .orderbook import OrderBook


@dataclass
class Opportunity:
    slug: str
    yes_token: str
    no_token: str
    yes_price: float
    no_price: float
    size: float
    fees_usd: float
    cost_usd: float
    payout_usd: float
    profit_usd: float
    edge_bps: float
    settle_in_s: float


def detect(
    *,
    slug: str,
    yes_book: OrderBook,
    no_book: OrderBook,
    fee_rate_bps: int,
    min_profit_bps: int,
    min_order_size: float,
    settle_in_s: float,
    min_time_to_settle_s: int,
) -> Opportunity | None:
    """Return the best YES+NO arbitrage at the top of book, or None.

    Walks only the level-1 ask of each side: that's the matchable arb at any
    instant. If you wanted to size up across multiple levels you'd VWAP — kept
    simple for v1.
    """
    if settle_in_s < min_time_to_settle_s:
        return None

    yes_ask = yes_book.best_ask()
    no_ask = no_book.best_ask()
    if yes_ask is None or no_ask is None:
        return None

    size = min(yes_ask.size, no_ask.size)
    if size < min_order_size:
        return None

    yes_fee = fee_for(yes_ask.price, size, fee_rate_bps)
    no_fee = fee_for(no_ask.price, size, fee_rate_bps)
    cost = yes_ask.price * size + no_ask.price * size + yes_fee + no_fee
    payout = size * 1.0
    profit = payout - cost
    if cost <= 0:
        return None
    edge_bps = (profit / cost) * 10_000.0
    if edge_bps < min_profit_bps:
        return None

    return Opportunity(
        slug=slug,
        yes_token=yes_book.asset_id,
        no_token=no_book.asset_id,
        yes_price=yes_ask.price,
        no_price=no_ask.price,
        size=size,
        fees_usd=yes_fee + no_fee,
        cost_usd=cost,
        payout_usd=payout,
        profit_usd=profit,
        edge_bps=edge_bps,
        settle_in_s=settle_in_s,
    )
