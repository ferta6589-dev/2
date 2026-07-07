from polyarb.coherence import detect_dutch_book, detect_overround
from polyarb.orderbook import OrderBook
from polyarb.weather_markets import WeatherBucket


def _b(lo, hi, slug=None):
    slug = slug or f"b-{lo}-{hi}"
    return WeatherBucket(
        slug=slug, title=slug,
        token_yes=f"Y_{slug}", token_no=f"N_{slug}",
        lo_c=lo, hi_c=hi, minimum_order_size=1.0,
    )


def _book(asset, ask_px, ask_sz=100, bid_px=None, bid_sz=100):
    b = OrderBook(asset_id=asset)
    bid_px = bid_px if bid_px is not None else max(0.01, ask_px - 0.03)
    b.replace([(bid_px, bid_sz)], [(ask_px, ask_sz)], 0)
    return b


def _three_buckets():
    return [_b(15, 17, "a"), _b(17, 19, "b"), _b(19, 21, "c")]


def test_dutch_book_detected_when_asks_sum_below_one():
    buckets = _three_buckets()
    # asks sum to 0.90 → below 1 - 0.02 wedge → arb
    books = {
        "Y_a": _book("Y_a", 0.20),
        "Y_b": _book("Y_b", 0.45),
        "Y_c": _book("Y_c", 0.25),
    }
    arb = detect_dutch_book(buckets, books, fee_wedge=0.02, min_profit_usd=0.1)
    assert arb is not None
    assert arb.side == "buy_all"
    assert abs(arb.price_sum - 0.90) < 1e-9
    assert abs(arb.edge_per_share - (1.0 - 0.90 - 0.02)) < 1e-9
    assert len(arb.legs) == 3


def test_no_dutch_book_when_asks_sum_above_one():
    buckets = _three_buckets()
    books = {
        "Y_a": _book("Y_a", 0.35),
        "Y_b": _book("Y_b", 0.45),
        "Y_c": _book("Y_c", 0.30),
    }
    arb = detect_dutch_book(buckets, books, fee_wedge=0.02)
    assert arb is None


def test_dutch_book_matched_size_is_min_across_legs():
    buckets = _three_buckets()
    books = {
        "Y_a": _book("Y_a", 0.20, ask_sz=500),
        "Y_b": _book("Y_b", 0.45, ask_sz=80),   # bottleneck
        "Y_c": _book("Y_c", 0.25, ask_sz=300),
    }
    arb = detect_dutch_book(buckets, books, fee_wedge=0.02, min_profit_usd=0.1)
    assert arb is not None
    assert arb.matched_size == 80.0
    # profit = edge(0.08) * 80 = 6.4
    assert abs(arb.profit_usd - 0.08 * 80) < 1e-6


def test_dutch_book_respects_min_profit():
    buckets = _three_buckets()
    # tiny edge, tiny size → below min_profit
    books = {
        "Y_a": _book("Y_a", 0.31, ask_sz=2),
        "Y_b": _book("Y_b", 0.33, ask_sz=2),
        "Y_c": _book("Y_c", 0.33, ask_sz=2),
    }
    arb = detect_dutch_book(buckets, books, fee_wedge=0.01, min_profit_usd=0.50)
    assert arb is None  # edge 0.02 * size 2 = 0.04 < 0.50


def test_dutch_book_none_on_incomplete_book():
    buckets = _three_buckets()
    books = {
        "Y_a": _book("Y_a", 0.20),
        "Y_b": _book("Y_b", 0.45),
        # Y_c missing → cannot guarantee coverage
    }
    arb = detect_dutch_book(buckets, books)
    assert arb is None


def test_max_set_cost_caps_size():
    buckets = _three_buckets()
    books = {
        "Y_a": _book("Y_a", 0.20, ask_sz=10_000),
        "Y_b": _book("Y_b", 0.45, ask_sz=10_000),
        "Y_c": _book("Y_c", 0.25, ask_sz=10_000),
    }
    # cost per set = 0.90; cap $90 → 100 sets
    arb = detect_dutch_book(
        buckets, books, fee_wedge=0.02, min_profit_usd=0.1, max_set_cost_usd=90.0
    )
    assert arb is not None
    assert abs(arb.matched_size - 100.0) < 1e-6


def test_overround_sell_all_detected():
    buckets = _three_buckets()
    # bids sum to 1.10 → above 1 + 0.02 → sell-all signal
    books = {
        "Y_a": _book("Y_a", 0.40, bid_px=0.38),
        "Y_b": _book("Y_b", 0.42, bid_px=0.40),
        "Y_c": _book("Y_c", 0.34, bid_px=0.32),
    }
    arb = detect_overround(buckets, books, fee_wedge=0.02)
    assert arb is not None
    assert arb.side == "sell_all"
    assert abs(arb.price_sum - 1.10) < 1e-9
