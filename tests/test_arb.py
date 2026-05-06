from polyarb.arb import detect
from polyarb.orderbook import OrderBook


def _book(asset_id: str, asks: list[tuple[float, float]], bids: list[tuple[float, float]] | None = None):
    b = OrderBook(asset_id=asset_id)
    b.replace(bids=bids or [], asks=asks, ts_ms=1)
    return b


def test_no_opportunity_when_sum_above_one():
    yes = _book("yes", asks=[(0.55, 100)])
    no = _book("no", asks=[(0.50, 100)])
    opp = detect(
        slug="btc-updown-5m-1",
        yes_book=yes,
        no_book=no,
        fee_rate_bps=200,
        min_profit_bps=50,
        min_order_size=5,
        settle_in_s=120,
        min_time_to_settle_s=10,
    )
    assert opp is None


def test_opportunity_detected_with_clear_edge():
    # 0.40 + 0.45 = 0.85 raw; even with peak fees this is well above 50 bps.
    yes = _book("yes", asks=[(0.40, 50)])
    no = _book("no", asks=[(0.45, 80)])
    opp = detect(
        slug="btc-updown-5m-1",
        yes_book=yes,
        no_book=no,
        fee_rate_bps=200,
        min_profit_bps=50,
        min_order_size=5,
        settle_in_s=120,
        min_time_to_settle_s=10,
    )
    assert opp is not None
    assert opp.size == 50  # min(50, 80)
    assert opp.profit_usd > 0
    # raw cost 50*0.85 = 42.5, fees 200bps*0.40*50 + 200bps*0.45*50 = 4 + 4.5 = 8.5 — actually
    # fee = rate * min(p,1-p) * size = 0.02 * 0.40 * 50 + 0.02 * 0.45 * 50 = 0.40 + 0.45 = 0.85
    expected_fees = 0.02 * 0.40 * 50 + 0.02 * 0.45 * 50
    assert abs(opp.fees_usd - expected_fees) < 1e-9


def test_size_limited_by_min_order():
    yes = _book("yes", asks=[(0.40, 2)])
    no = _book("no", asks=[(0.45, 80)])
    opp = detect(
        slug="s",
        yes_book=yes,
        no_book=no,
        fee_rate_bps=200,
        min_profit_bps=50,
        min_order_size=5,
        settle_in_s=120,
        min_time_to_settle_s=10,
    )
    assert opp is None


def test_filtered_when_too_close_to_settle():
    yes = _book("yes", asks=[(0.40, 50)])
    no = _book("no", asks=[(0.45, 80)])
    opp = detect(
        slug="s",
        yes_book=yes,
        no_book=no,
        fee_rate_bps=200,
        min_profit_bps=50,
        min_order_size=5,
        settle_in_s=3,
        min_time_to_settle_s=10,
    )
    assert opp is None


def test_below_min_profit_threshold_filtered():
    # 0.49 + 0.50 = 0.99; profit ~1% gross but fees eat most of it.
    yes = _book("yes", asks=[(0.49, 50)])
    no = _book("no", asks=[(0.50, 50)])
    opp = detect(
        slug="s",
        yes_book=yes,
        no_book=no,
        fee_rate_bps=200,
        min_profit_bps=50,
        min_order_size=5,
        settle_in_s=120,
        min_time_to_settle_s=10,
    )
    assert opp is None
