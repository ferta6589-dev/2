from polyarb.forecast import ForecastDistribution
from polyarb.orderbook import OrderBook
from polyarb.two_bucket_strategy import (
    decide_pair_entry,
    select_best_pair,
)
from polyarb.weather_markets import WeatherBucket


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
    """Standard Polymarket Moscow lattice: 2°C-wide buckets from 11 to 23°C."""
    starts = [11, 13, 15, 17, 19, 21, 23]
    return [_b(lo, lo + 2) for lo in starts]


def _book(asset, ask_px, ask_sz=200, bid_px=None, bid_sz=200):
    b = OrderBook(asset_id=asset)
    bid_px = bid_px if bid_px is not None else max(0.01, ask_px - 0.05)
    b.replace([(bid_px, bid_sz)], [(ask_px, ask_sz)], 0)
    return b


def test_select_best_pair_picks_neighbors_of_mu():
    buckets = _moscow_lattice()
    # μ=18 in middle of 17-19; σ=0.7 → P(17-19)≈84%, P(neighbor)≈8% → combined≈92%
    dist = ForecastDistribution(mu_c=18.0, sigma_c=0.7, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.85, max_sigma_c=1.5)
    assert pair is not None
    assert pair.primary.lo_c == 17.0
    assert pair.secondary.lo_c in (15.0, 19.0)
    assert pair.combined_prob > 0.85


def test_select_best_pair_rejects_high_sigma():
    buckets = _moscow_lattice()
    dist = ForecastDistribution(mu_c=18.0, sigma_c=2.0, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.90, max_sigma_c=1.2)
    assert pair is None  # sigma above threshold


def test_select_best_pair_rejects_low_combined_prob():
    """Mu falls exactly between two buckets but sigma is too wide for >90%."""
    buckets = _moscow_lattice()
    dist = ForecastDistribution(mu_c=18.0, sigma_c=1.5, sources=["t"])
    # σ=1.5 → 4°C window covers ~81% which is below 0.90
    pair = select_best_pair(buckets, dist, min_combined_prob=0.90, max_sigma_c=1.5)
    assert pair is None


def test_select_best_pair_accepts_at_threshold():
    """μ exactly on bucket boundary (μ=17.0) with σ=1.0 → combined ≈ 95%."""
    buckets = _moscow_lattice()
    dist = ForecastDistribution(mu_c=17.0, sigma_c=1.0, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.90, max_sigma_c=1.5)
    assert pair is not None
    assert pair.combined_prob >= 0.90


def test_decide_pair_entry_buys_both_when_prices_ok():
    buckets = _moscow_lattice()
    dist = ForecastDistribution(mu_c=18.0, sigma_c=0.7, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.85, max_sigma_c=1.5)
    assert pair is not None

    books = {
        pair.primary.token_yes: _book(pair.primary.token_yes, 0.40),
        pair.secondary.token_yes: _book(pair.secondary.token_yes, 0.18),
    }
    orders = decide_pair_entry(
        pair, books,
        primary_max_price=0.45,
        secondary_max_price=0.25,
        budget_usd=30.0,
    )
    assert len(orders) == 2
    roles = {o.role for o in orders}
    assert roles == {"primary", "secondary"}


def test_decide_pair_entry_skips_primary_when_price_too_high():
    buckets = _moscow_lattice()
    dist = ForecastDistribution(mu_c=18.0, sigma_c=0.7, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.85, max_sigma_c=1.5)
    assert pair is not None

    books = {
        # primary at 0.50 (above cap 0.45)
        pair.primary.token_yes: _book(pair.primary.token_yes, 0.50),
        pair.secondary.token_yes: _book(pair.secondary.token_yes, 0.18),
    }
    orders = decide_pair_entry(
        pair, books,
        primary_max_price=0.45,
        secondary_max_price=0.25,
        budget_usd=30.0,
    )
    assert len(orders) == 1
    assert orders[0].role == "secondary"


def test_pair_handles_open_ended_buckets():
    """Open-ended buckets (<11°C, >23°C) at ends should still pair-ify."""
    buckets = [
        _b(None, 11, "lt11"),
        _b(11, 13),
        _b(13, 15),
        _b(15, 17),
        _b(17, 19),
        _b(19, 21),
        _b(21, 23),
        _b(23, None, "gt23"),
    ]
    dist = ForecastDistribution(mu_c=18.0, sigma_c=0.7, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.85, max_sigma_c=1.5)
    assert pair is not None
    assert pair.primary.lo_c == 17.0


def test_pair_handles_cold_forecast():
    """Cold mu (μ=5°C) selects the open-ended <11 bucket if available."""
    buckets = [
        _b(None, 11, "lt11"),
        _b(11, 13),
        _b(13, 15),
    ]
    dist = ForecastDistribution(mu_c=5.0, sigma_c=1.0, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.85, max_sigma_c=1.5)
    assert pair is not None
    assert pair.primary.slug == "lt11"


def test_budget_split_60_40():
    buckets = _moscow_lattice()
    dist = ForecastDistribution(mu_c=18.0, sigma_c=0.7, sources=["t"])
    pair = select_best_pair(buckets, dist, min_combined_prob=0.85, max_sigma_c=1.5)
    assert pair is not None
    # plenty of book depth; check qty allocation matches budget split
    books = {
        pair.primary.token_yes: _book(pair.primary.token_yes, 0.20, ask_sz=10_000),
        pair.secondary.token_yes: _book(pair.secondary.token_yes, 0.10, ask_sz=10_000),
    }
    orders = decide_pair_entry(
        pair, books,
        primary_max_price=0.50, secondary_max_price=0.25,
        budget_usd=100.0, primary_budget_share=0.6,
    )
    pri = next(o for o in orders if o.role == "primary")
    sec = next(o for o in orders if o.role == "secondary")
    # primary: $60 budget / $0.20 ask = 300 shares
    assert abs(pri.qty - 300.0) < 1.0
    # secondary: $40 budget / $0.10 ask = 400 shares
    assert abs(sec.qty - 400.0) < 1.0
