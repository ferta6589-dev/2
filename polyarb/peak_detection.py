"""Peak-lock detection — the highest-conviction alpha in the GHOST strategy.

Daily-max temperature follows a diurnal cycle: it rises from dawn, peaks a few
hours after solar noon, then falls. The Polymarket resolver settles on the
day's maximum. Therefore, **once the peak has passed, the daily max is final**
and the bucket containing it is the guaranteed winner — hours before the market
resolves.

This module fuses three orthogonal signals into a single lock confidence:

1. **Trend** — the observed max was reached ≥ ``hold_minutes`` ago and the most
   recent observations sit ``fall_margin_c`` below it (temperature is falling).
2. **Solar geometry** — current time is past ``solar_noon + peak_lag_hours``,
   so physically the peak should already be behind us.
3. **Forecast confirmation** — the forecast maximum for the rest of the day is
   below the observed max (nothing left to exceed it).

Plus a safety check: the observed max must sit comfortably inside a bucket
(``boundary_margin_c`` away from either edge), so a late 0.5 °C blip can't flip
the winner across a boundary.

When the combined confidence clears ``lock_threshold``, the winner is declared
locked: buy it if still underpriced, dump everything else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .weather_markets import WeatherBucket


def solar_noon_utc(lon_deg: float, on: date) -> datetime:
    """Approximate solar noon in UTC for a longitude (equation-of-time aware).

    Good to a few minutes — more than enough to gate "are we past the peak".
    """
    # Day of year for the equation of time.
    n = on.timetuple().tm_yday
    b = 2 * math.pi * (n - 81) / 364.0
    eot_min = 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)
    # Local solar noon occurs when mean sun crosses the meridian:
    # 12:00 local mean time minus longitude offset, minus EoT.
    noon_utc_hours = 12.0 - lon_deg / 15.0 - eot_min / 60.0
    base = datetime(on.year, on.month, on.day, tzinfo=timezone.utc)
    return base.replace(hour=0, minute=0, second=0) + _hours(noon_utc_hours)


def _hours(h: float):
    from datetime import timedelta
    return timedelta(hours=h)


@dataclass
class PeakSignals:
    trend_ok: bool
    solar_ok: bool
    forecast_ok: bool
    boundary_ok: bool
    observed_max_c: float | None
    minutes_since_peak: float | None
    confidence: float
    locked: bool
    winner_slug: str | None
    detail: dict = field(default_factory=dict)


@dataclass
class PeakTracker:
    """Accumulates same-day observations and reports peak-lock status."""

    lat: float
    lon: float
    target_date: date
    samples: list[tuple[datetime, float]] = field(default_factory=list)

    def add(self, ts: datetime, temp_c: float) -> None:
        if ts.date() != self.target_date:
            return
        self.samples.append((ts, temp_c))

    @property
    def observed_max_c(self) -> float | None:
        return max((t for _, t in self.samples), default=None)

    def _peak_time(self) -> datetime | None:
        if not self.samples:
            return None
        peak = max(self.samples, key=lambda s: s[1])
        return peak[0]

    def evaluate(
        self,
        buckets: list[WeatherBucket],
        *,
        now: datetime,
        forecast_remaining_max_c: float | None = None,
        peak_lag_hours: float = 2.5,
        hold_minutes: float = 60.0,
        fall_margin_c: float = 0.8,
        boundary_margin_c: float = 0.6,
        forecast_margin_c: float = 0.5,
        lock_threshold: float = 0.75,
    ) -> PeakSignals:
        obs_max = self.observed_max_c
        if obs_max is None:
            return PeakSignals(False, False, False, False, None, None, 0.0, False, None)

        # --- Signal 1: trend (peak reached a while ago + now falling) ---
        peak_ts = self._peak_time()
        minutes_since_peak = (
            (now - peak_ts).total_seconds() / 60.0 if peak_ts else None
        )
        recent = [t for ts, t in self.samples if (now - ts).total_seconds() <= 5400][-4:]
        falling = bool(recent) and all(t <= obs_max - fall_margin_c for t in recent[-2:]) if len(recent) >= 2 else False
        trend_ok = (
            minutes_since_peak is not None
            and minutes_since_peak >= hold_minutes
            and falling
        )

        # --- Signal 2: solar geometry (past solar noon + lag) ---
        noon = solar_noon_utc(self.lon, self.target_date)
        expected_peak = noon + _hours(peak_lag_hours)
        solar_ok = now >= expected_peak

        # --- Signal 3: forecast confirms nothing left to exceed obs_max ---
        forecast_ok = (
            forecast_remaining_max_c is not None
            and forecast_remaining_max_c <= obs_max - forecast_margin_c
        )

        # --- Winner + boundary safety ---
        winner = _winning_bucket(buckets, obs_max)
        boundary_ok = _inside_with_margin(winner, obs_max, boundary_margin_c) if winner else False

        # --- Fuse into confidence ---
        # Solar is necessary background; trend + forecast are the strong evidence.
        weights = {"trend": 0.45, "solar": 0.20, "forecast": 0.25, "boundary": 0.10}
        confidence = (
            weights["trend"] * trend_ok
            + weights["solar"] * solar_ok
            + weights["forecast"] * forecast_ok
            + weights["boundary"] * boundary_ok
        )
        locked = confidence >= lock_threshold and boundary_ok and (trend_ok or forecast_ok)

        return PeakSignals(
            trend_ok=trend_ok,
            solar_ok=solar_ok,
            forecast_ok=forecast_ok,
            boundary_ok=boundary_ok,
            observed_max_c=obs_max,
            minutes_since_peak=minutes_since_peak,
            confidence=round(confidence, 3),
            locked=locked,
            winner_slug=winner.slug if winner else None,
            detail={
                "solar_noon_utc": noon.isoformat(),
                "expected_peak_utc": expected_peak.isoformat(),
                "forecast_remaining_max_c": forecast_remaining_max_c,
                "n_samples": len(self.samples),
            },
        )


def _winning_bucket(buckets: list[WeatherBucket], observed_c: float) -> WeatherBucket | None:
    truncated = int(math.floor(observed_c)) if observed_c >= 0 else int(math.ceil(observed_c))
    for b in buckets:
        if b.lo_c is not None and b.hi_c is not None:
            if int(b.lo_c) <= truncated < int(b.hi_c):
                return b
        elif b.lo_c is None and b.hi_c is not None:
            if truncated < int(b.hi_c):
                return b
        elif b.hi_c is None and b.lo_c is not None:
            if truncated >= int(b.lo_c):
                return b
    return None


def _inside_with_margin(b: WeatherBucket, observed_c: float, margin_c: float) -> bool:
    """True if observed_c sits ≥ margin away from both bucket edges."""
    if b is None:
        return False
    if b.lo_c is not None and observed_c - b.lo_c < margin_c:
        return False
    if b.hi_c is not None and b.hi_c - observed_c < margin_c:
        return False
    return True
