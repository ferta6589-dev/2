from polyarb.orderbook import OrderBook
from polyarb.weather_markets import WeatherBucket
from polyarb.weather_strategy import (
    classify_buckets,
    decide_entry,
    decide_exits,
    diff_verdicts,
)


def _b(lo, hi, slug=None):
    slug = slug or f"b-{lo}-{hi}"
    return WeatherBucket(
        slug=slug,
        title=f"{lo}-{hi}",
        token_yes=f"YES_{slug}",
        token_no=f"NO_{slug}",
        lo_c=lo,
        hi_c=hi,
        minimum_order_size=5.0,
    )


def _moscow_lattice():
    starts = [11, 13, 15, 17, 19, 21, 23]
    return [_b(lo, lo + 2) for lo in starts]


def test_classify_unreached_when_no_observation():
    buckets = _moscow_lattice()
    verdicts = classify_buckets(buckets, observed_max_c=None)
    assert all(v == "unreached" for v in verdicts.values())


def test_classify_leader_on_observed_climb():
    buckets = _moscow_lattice()
    v = classify_buckets(buckets, observed_max_c=18.3)
    assert v[buckets[0].slug] == "exceeded"  # 11-13
    assert v[buckets[1].slug] == "exceeded"  # 13-15
    assert v[buckets[2].slug] == "exceeded"  # 15-17
    assert v[buckets[3].slug] == "leader"    # 17-19
    assert v[buckets[4].slug] == "unreached"  # 19-21


def test_classify_truncates_toward_zero():
    buckets = _moscow_lattice()
    v = classify_buckets(buckets, observed_max_c=17.0)
    assert v[buckets[3].slug] == "leader"  # 17-19 contains 17


def test_classify_handles_open_ended():
    buckets = [_b(None, 11), _b(11, 13), _b(13, None)]
    v = classify_buckets(buckets, observed_max_c=15.5)
    assert v[buckets[0].slug] == "exceeded"
    assert v[buckets[1].slug] == "exceeded"
    assert v[buckets[2].slug] == "leader"


def _book(asset, ask_px, ask_sz, bid_px, bid_sz):
    b = OrderBook(asset_id=asset)
    b.replace([(bid_px, bid_sz)], [(ask_px, ask_sz)], 0)
    return b


def test_decide_entry_buys_only_leader():
    buckets = _moscow_lattice()
    books = {b.token_yes: _book(b.token_yes, 0.20, 200, 0.18, 200) for b in buckets}
    v = classify_buckets(buckets, observed_max_c=18.3)
    orders = decide_entry(buckets, v, books, max_price=0.40, budget_usd=30.0)
    assert len(orders) == 1
    assert orders[0].bucket.lo_c == 17.0
    assert orders[0].limit_price == 0.20


def test_decide_entry_respects_max_price_cap():
    buckets = _moscow_lattice()
    books = {b.token_yes: _book(b.token_yes, 0.55, 200, 0.50, 200) for b in buckets}
    v = classify_buckets(buckets, observed_max_c=18.3)
    orders = decide_entry(buckets, v, books, max_price=0.40, budget_usd=30.0)
    assert orders == []


def test_decide_exits_only_sells_exceeded():
    buckets = _moscow_lattice()
    books = {b.token_yes: _book(b.token_yes, 0.20, 200, 0.18, 200) for b in buckets}
    v = classify_buckets(buckets, observed_max_c=18.3)
    positions = [(b, 50.0) for b in buckets]
    exits = decide_exits(positions, v, books)
    exit_slugs = {ex.bucket_slug for ex in exits}
    # 11-13, 13-15, 15-17 are exceeded
    assert exit_slugs == {buckets[0].slug, buckets[1].slug, buckets[2].slug}
    # All sell at best_bid (0.18)
    assert all(ex.limit_price == 0.18 for ex in exits)


def test_diff_verdicts_detects_changes():
    prev = {"a": "unreached", "b": "leader", "c": "unreached"}
    curr = {"a": "exceeded", "b": "exceeded", "c": "leader"}
    changes = diff_verdicts(prev, curr)
    assert changes["a"] == ("unreached", "exceeded")
    assert changes["b"] == ("leader", "exceeded")
    assert changes["c"] == ("unreached", "leader")
