from datetime import date, datetime, timedelta, timezone

from polyarb.peak_detection import PeakTracker, solar_noon_utc
from polyarb.backtest import make_lattice


def _t(h, m=0):
    return datetime(2026, 5, 12, h, m, tzinfo=timezone.utc)


def test_solar_noon_moscow_is_late_morning_utc():
    # Moscow lon ≈ 37.6°E → solar noon well before 12:00 UTC (~09:20-09:40)
    noon = solar_noon_utc(37.27, date(2026, 5, 12))
    assert 8 <= noon.hour <= 10


def test_solar_noon_west_of_greenwich_is_afternoon_utc():
    # NYC lon ≈ -74° → solar noon ~17:00 UTC
    noon = solar_noon_utc(-73.97, date(2026, 5, 12))
    assert 16 <= noon.hour <= 18


def test_not_locked_while_still_rising():
    lattice = make_lattice(18.0)
    tr = PeakTracker(lat=55.59, lon=37.27, target_date=date(2026, 5, 12))
    # morning climb
    for h, temp in [(5, 8.0), (7, 11.0), (9, 14.0), (11, 17.0)]:
        tr.add(_t(h), temp)
    sig = tr.evaluate(lattice, now=_t(11, 5))
    assert not sig.locked


def test_locked_after_peak_passes():
    lattice = make_lattice(18.0)
    tr = PeakTracker(lat=55.59, lon=37.27, target_date=date(2026, 5, 12))
    # climb to a peak of 18.7 around 12:00 UTC (well past Moscow solar noon),
    # then a clear decline
    timeline = [
        (6, 9.0), (8, 12.0), (10, 15.5), (11, 17.5),
        (12, 18.7),                 # peak
        (13, 17.6), (14, 16.4), (15, 15.0),  # falling
    ]
    for h, temp in timeline:
        tr.add(_t(h), temp)
    sig = tr.evaluate(
        lattice,
        now=_t(15, 10),
        forecast_remaining_max_c=15.5,  # nothing left to beat 18.7
    )
    assert sig.observed_max_c == 18.7
    assert sig.winner_slug is not None
    assert sig.trend_ok
    assert sig.solar_ok
    assert sig.forecast_ok
    assert sig.locked
    assert sig.confidence >= 0.75


def test_not_locked_when_max_sits_on_boundary():
    # observed max 18.0 sits right on the 18-20 bucket's lower edge → unsafe
    lattice = make_lattice(18.0)
    tr = PeakTracker(lat=55.59, lon=37.27, target_date=date(2026, 5, 12))
    timeline = [(8, 12.0), (10, 15.0), (12, 18.0), (13, 16.5), (14, 15.0), (15, 14.0)]
    for h, temp in timeline:
        tr.add(_t(h), temp)
    sig = tr.evaluate(
        lattice, now=_t(15, 10),
        forecast_remaining_max_c=14.0,
        boundary_margin_c=0.6,
    )
    # 18.0 truncates to 18 → winner bucket [18,20); distance to lower edge = 0 < margin
    assert not sig.boundary_ok
    assert not sig.locked


def test_lock_needs_more_than_solar_alone():
    """Being past solar-peak time is not enough without trend/forecast."""
    lattice = make_lattice(18.0)
    tr = PeakTracker(lat=55.59, lon=37.27, target_date=date(2026, 5, 12))
    # single reading, late in day, no decline evidence
    tr.add(_t(15), 18.7)
    sig = tr.evaluate(lattice, now=_t(15, 30))
    assert sig.solar_ok
    assert not sig.trend_ok
    assert not sig.locked
