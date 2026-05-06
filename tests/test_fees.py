import pytest

from polyarb.fees import fee_for


def test_zero_at_extremes():
    assert fee_for(0.0, 100, 200) == 0.0
    assert fee_for(1.0, 100, 200) == 0.0


def test_peak_at_50():
    # 200 bps * 0.5 * 100 shares = 1.0 USD
    assert fee_for(0.5, 100, 200) == 1.0


def test_symmetry():
    assert fee_for(0.3, 100, 200) == pytest.approx(fee_for(0.7, 100, 200))


def test_scales_linearly_in_size():
    base = fee_for(0.4, 10, 200)
    assert fee_for(0.4, 100, 200) == base * 10


def test_zero_size_or_negative():
    assert fee_for(0.5, 0, 200) == 0.0
    assert fee_for(0.5, -1, 200) == 0.0
