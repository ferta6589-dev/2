from polyarb.forecast import ForecastDistribution
from polyarb.orderbook import OrderBook
from polyarb.weather_markets import WeatherBucket
from polyarb.weather_strategy import (
    classify_buckets,
    decide_entry,
    decide_exits,
    select_window,
)


def _bucket(lo, hi, slug=None):
    slug = slug or f"b-{lo}-{hi}"
    return WeatherBucket(
        slug=slug,
        title=f"{lo}-{hi}",
        token_yes=f"YES_{slug}",
        token_no=f"NO_{slug}",
        lo_f=lo,
        hi_f=hi,
        minimum_order_size=5.0,
        minimum_tick_size=0.01,
    )


def _lattice():
    starts = [55, 60, 65, 70, 75, 80, 85]
    return [_bucket(lo, lo + 5) for lo in starts]


def test_select_window_centers_on_forecast_mean():
    buckets = _lattice()
    dist = ForecastDistribution(mu_f=72.0, sigma_f=2.0, sources=["t"])
    window = select_window(buckets, dist, min_window_prob=0.5, width=3)
    assert [b.lo_f for b in window] == [65.0, 70.0, 75.0]


def test_select_window_low_confidence_returns_empty():
    buckets = _lattice()
    dist = ForecastDistribution(mu_f=120.0, sigma_f=1.0, sources=["t"])
    assert select_window(buckets, dist, min_window_prob=0.5, width=3) == []


def test_classify_buckets_marks_already_exceeded_and_unreachable():
    window = [_bucket(65, 70), _bucket(70, 75), _bucket(75, 80)]
    verdicts = classify_buckets(
        window,
        observed_high_f=72.0,
        forecast_remaining_max_f=74.0,
    )
    assert verdicts[window[0].slug] == "loser"
    assert verdicts[window[2].slug] == "loser"
    assert verdicts[window[1].slug] == "winner"


def test_classify_buckets_two_competitors():
    window = [_bucket(65, 70), _bucket(70, 75), _bucket(75, 80)]
    verdicts = classify_buckets(
        window,
        observed_high_f=66.0,
        forecast_remaining_max_f=78.0,
    )
    assert verdicts[window[1].slug] == "competitor"
    assert verdicts[window[2].slug] == "competitor"


def _book(asset_id, ask_px, ask_sz, bid_px, bid_sz):
    b = OrderBook(asset_id=asset_id)
    b.replace([(bid_px, bid_sz)], [(ask_px, ask_sz)], 0)
    return b


def test_decide_entry_respects_price_caps():
    window = [_bucket(65, 70, "wing-l"), _bucket(70, 75, "central"), _bucket(75, 80, "wing-r")]
    books = {
        "YES_wing-l": _book("YES_wing-l", ask_px=0.18, ask_sz=200, bid_px=0.15, bid_sz=200),
        "YES_central": _book("YES_central", ask_px=0.50, ask_sz=200, bid_px=0.45, bid_sz=200),
        "YES_wing-r": _book("YES_wing-r", ask_px=0.10, ask_sz=200, bid_px=0.08, bid_sz=200),
    }
    orders = decide_entry(
        window,
        books,
        central_max_price=0.40,
        wing_max_price=0.20,
        per_event_budget_usd=30.0,
    )
    slugs = {o.bucket.slug for o in orders}
    assert "central" not in slugs
    assert "wing-l" in slugs and "wing-r" in slugs


def test_decide_entry_skips_below_min_size():
    bucket = WeatherBucket(slug="b", title="t", token_yes="Y", token_no="N",
                           lo_f=70.0, hi_f=75.0, minimum_order_size=200.0)
    books = {"Y": _book("Y", ask_px=0.30, ask_sz=10, bid_px=0.25, bid_sz=10)}
    orders = decide_entry([bucket], books, central_max_price=0.40,
                          wing_max_price=0.40, per_event_budget_usd=30.0)
    assert orders == []


def test_decide_exits_sells_only_losers():
    window = [_bucket(65, 70, "wing-l"), _bucket(70, 75, "central"), _bucket(75, 80, "wing-r")]
    books = {
        "YES_wing-l": _book("YES_wing-l", 0.05, 100, 0.03, 100),
        "YES_central": _book("YES_central", 0.55, 100, 0.50, 100),
        "YES_wing-r": _book("YES_wing-r", 0.40, 100, 0.35, 100),
    }
    verdicts = {"wing-l": "loser", "central": "winner", "wing-r": "competitor"}
    positions = [(window[0], 30.0), (window[1], 30.0), (window[2], 30.0)]
    exits = decide_exits(positions, verdicts, books, hold_winner=True)
    assert len(exits) == 1
    assert exits[0].bucket_slug == "wing-l"
    assert exits[0].limit_price == 0.03
