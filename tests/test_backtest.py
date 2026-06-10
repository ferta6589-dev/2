import math

from polyarb.backtest import (
    _winning_bucket,
    make_lattice,
    monte_carlo,
)


def test_make_lattice_contiguous_and_open_ended():
    lat = make_lattice(18.0)
    # first bucket open-ended low, last open-ended high
    assert lat[0].lo_c is None
    assert lat[-1].hi_c is None
    # interior buckets are 2°C wide and contiguous
    interior = [b for b in lat if b.lo_c is not None and b.hi_c is not None]
    for b in interior:
        assert abs((b.hi_c - b.lo_c) - 2.0) < 1e-9
    for a, b in zip(interior, interior[1:]):
        assert abs(a.hi_c - b.lo_c) < 1e-9


def test_winning_bucket_truncation():
    lat = make_lattice(18.0)
    # 18.7°C truncates to 18 → must fall in the [18,20) bucket
    w = _winning_bucket(lat, 18.7)
    assert w is not None
    assert w.lo_c == 18.0 and w.hi_c == 20.0
    # 17.9 → 17 → [16,18)
    w2 = _winning_bucket(lat, 17.9)
    assert w2.lo_c == 16.0 and w2.hi_c == 18.0


def test_winning_bucket_extremes():
    lat = make_lattice(18.0)
    # very cold falls in open-ended low
    w_cold = _winning_bucket(lat, -40.0)
    assert w_cold.lo_c is None
    # very hot falls in open-ended high
    w_hot = _winning_bucket(lat, 99.0)
    assert w_hot.hi_c is None


def test_monte_carlo_deterministic():
    r1 = monte_carlo(n_events=2000, seed=1)
    r2 = monte_carlo(n_events=2000, seed=1)
    assert r1.n_traded == r2.n_traded
    assert r1.n_hit == r2.n_hit


def test_monte_carlo_robust_gate_beats_90pct():
    """Core claim: robust gate stays above 90% at realistic calibration."""
    r = monte_carlo(
        n_events=20000,
        max_sigma_c=1.0,
        min_pair_prob=0.92,
        calibration_factor=1.15,
        seed=7,
    )
    assert r.hit_rate >= 0.90
    # and it actually trades a meaningful fraction
    assert 0.10 < r.coverage < 0.40


def test_monte_carlo_tighter_gate_higher_hitrate():
    """Tighter prob gate → higher hit-rate, lower coverage."""
    loose = monte_carlo(n_events=20000, max_sigma_c=1.4, min_pair_prob=0.88, seed=7)
    tight = monte_carlo(n_events=20000, max_sigma_c=1.0, min_pair_prob=0.92, seed=7)
    assert tight.hit_rate > loose.hit_rate
    assert tight.coverage < loose.coverage


def test_monte_carlo_miscalibration_degrades_hitrate():
    """Worse ensemble calibration → lower realised hit-rate (monotone)."""
    good = monte_carlo(n_events=20000, calibration_factor=1.0, seed=7)
    bad = monte_carlo(n_events=20000, calibration_factor=1.4, seed=7)
    assert good.hit_rate > bad.hit_rate
