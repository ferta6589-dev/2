"""Free meteorological data feeds for the weather strategy.

Two sources, blended by inverse-variance weighting:

* **NWS** (``api.weather.gov``) — same agency that resolves Polymarket weather
  markets via station observations, so it's the closest signal we can get.
* **Open-Meteo** (``api.open-meteo.com``) — free ensemble feed with no key,
  used to denoise the NWS point forecast.

The σ table reflects rough day-ahead error for daily highs (in °F) — finer
calibration is a v2 follow-up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import httpx
import structlog

log = structlog.get_logger("polyarb.forecast")


@dataclass(frozen=True)
class ForecastPoint:
    mu_f: float
    sigma_f: float
    source: str


@dataclass(frozen=True)
class ObservedNow:
    temp_f: float
    ts_utc: float


@dataclass(frozen=True)
class ForecastDistribution:
    mu_f: float
    sigma_f: float
    sources: list[str] = field(default_factory=list)

    def prob_in(self, lo_f: float | None, hi_f: float | None) -> float:
        return _gaussian_prob_between(self.mu_f, self.sigma_f, lo_f, hi_f)


def _sigma_for_lead_days(lead_days: int) -> float:
    if lead_days <= 0:
        return 2.0
    if lead_days <= 1:
        return 2.5
    if lead_days <= 3:
        return 3.0
    if lead_days <= 5:
        return 4.0
    return 5.0


def _gaussian_prob_between(mu: float, sigma: float, lo: float | None, hi: float | None) -> float:
    if sigma <= 0:
        sigma = 0.5
    z_lo = -math.inf if lo is None else (lo - mu) / (sigma * math.sqrt(2))
    z_hi = math.inf if hi is None else (hi - mu) / (sigma * math.sqrt(2))
    cdf_lo = 0.0 if z_lo == -math.inf else 0.5 * (1.0 + math.erf(z_lo))
    cdf_hi = 1.0 if z_hi == math.inf else 0.5 * (1.0 + math.erf(z_hi))
    return max(0.0, cdf_hi - cdf_lo)


def ensemble(forecasts: list[ForecastPoint]) -> ForecastDistribution | None:
    """Blend by inverse-variance weighting. Inflate σ if only one survives."""
    forecasts = [f for f in forecasts if f and f.sigma_f > 0]
    if not forecasts:
        return None
    if len(forecasts) == 1:
        f = forecasts[0]
        return ForecastDistribution(mu_f=f.mu_f, sigma_f=f.sigma_f * 1.25, sources=[f.source])
    weights = [1.0 / (f.sigma_f ** 2) for f in forecasts]
    total = sum(weights)
    mu = sum(w * f.mu_f for w, f in zip(weights, forecasts)) / total
    sigma = math.sqrt(1.0 / total)
    return ForecastDistribution(mu_f=mu, sigma_f=sigma, sources=[f.source for f in forecasts])


def _nws_headers(user_agent: str) -> dict[str, str]:
    return {"User-Agent": user_agent, "Accept": "application/geo+json"}


async def nws_long_range(
    client: httpx.AsyncClient,
    nws_host: str,
    lat: float,
    lon: float,
    target_date: date,
    user_agent: str,
) -> ForecastPoint | None:
    """Daily high °F from the NWS gridpoint forecast for ``target_date``."""
    try:
        r = await client.get(
            f"{nws_host}/points/{lat:.4f},{lon:.4f}",
            headers=_nws_headers(user_agent),
            timeout=10.0,
        )
        r.raise_for_status()
        forecast_url = r.json().get("properties", {}).get("forecast")
        if not forecast_url:
            return None
        r2 = await client.get(forecast_url, headers=_nws_headers(user_agent), timeout=10.0)
        r2.raise_for_status()
        periods = r2.json().get("properties", {}).get("periods", [])
    except httpx.HTTPError as e:
        log.warning("nws_long_range_failed", err=str(e))
        return None

    candidates: list[float] = []
    for p in periods:
        if not p.get("isDaytime"):
            continue
        start = p.get("startTime")
        if not start:
            continue
        try:
            day = datetime.fromisoformat(start.replace("Z", "+00:00")).date()
        except ValueError:
            continue
        if day == target_date:
            unit = (p.get("temperatureUnit") or "F").upper()
            t = float(p.get("temperature"))
            if unit == "C":
                t = t * 9.0 / 5.0 + 32.0
            candidates.append(t)
    if not candidates:
        return None
    mu = max(candidates)
    lead_days = (target_date - datetime.now(timezone.utc).date()).days
    return ForecastPoint(mu_f=mu, sigma_f=_sigma_for_lead_days(lead_days), source="nws")


async def open_meteo_long_range(
    client: httpx.AsyncClient,
    open_meteo_host: str,
    lat: float,
    lon: float,
    target_date: date,
) -> ForecastPoint | None:
    """Daily high °F from Open-Meteo's free forecast endpoint."""
    try:
        r = await client.get(
            f"{open_meteo_host}/v1/forecast",
            params={
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "daily": "temperature_2m_max",
                "temperature_unit": "fahrenheit",
                "timezone": "UTC",
                "start_date": target_date.isoformat(),
                "end_date": target_date.isoformat(),
            },
            timeout=10.0,
        )
        r.raise_for_status()
        daily = r.json().get("daily", {})
        times = daily.get("time", [])
        highs = daily.get("temperature_2m_max", [])
    except httpx.HTTPError as e:
        log.warning("open_meteo_failed", err=str(e))
        return None

    for t, h in zip(times, highs):
        try:
            d = date.fromisoformat(t)
        except ValueError:
            continue
        if d == target_date and h is not None:
            lead_days = (target_date - datetime.now(timezone.utc).date()).days
            return ForecastPoint(mu_f=float(h), sigma_f=_sigma_for_lead_days(lead_days) * 1.1, source="open_meteo")
    return None


async def nws_observation(
    client: httpx.AsyncClient,
    nws_host: str,
    station_id: str,
    user_agent: str,
) -> ObservedNow | None:
    try:
        r = await client.get(
            f"{nws_host}/stations/{station_id}/observations/latest",
            headers=_nws_headers(user_agent),
            timeout=10.0,
        )
        r.raise_for_status()
        props = r.json().get("properties", {})
    except httpx.HTTPError as e:
        log.warning("nws_observation_failed", err=str(e), station=station_id)
        return None

    temp = props.get("temperature", {}) or {}
    val = temp.get("value")
    unit = (temp.get("unitCode") or "").lower()
    if val is None:
        return None
    if "degc" in unit or "celsius" in unit or unit.endswith(":degc"):
        val_f = float(val) * 9.0 / 5.0 + 32.0
    elif "degf" in unit or "fahrenheit" in unit:
        val_f = float(val)
    else:
        val_f = float(val) * 9.0 / 5.0 + 32.0
    ts_iso = props.get("timestamp")
    ts = (
        datetime.fromisoformat(ts_iso.replace("Z", "+00:00")).timestamp()
        if ts_iso
        else datetime.now(timezone.utc).timestamp()
    )
    return ObservedNow(temp_f=val_f, ts_utc=ts)


async def nws_today_remaining_max(
    client: httpx.AsyncClient,
    nws_host: str,
    lat: float,
    lon: float,
    user_agent: str,
    now: datetime | None = None,
) -> float | None:
    """Max forecast °F from now until end-of-day UTC at the gridpoint."""
    now = now or datetime.now(timezone.utc)
    try:
        r = await client.get(
            f"{nws_host}/points/{lat:.4f},{lon:.4f}",
            headers=_nws_headers(user_agent),
            timeout=10.0,
        )
        r.raise_for_status()
        hourly_url = r.json().get("properties", {}).get("forecastHourly")
        if not hourly_url:
            return None
        r2 = await client.get(hourly_url, headers=_nws_headers(user_agent), timeout=10.0)
        r2.raise_for_status()
        periods = r2.json().get("properties", {}).get("periods", [])
    except httpx.HTTPError as e:
        log.warning("nws_hourly_failed", err=str(e))
        return None

    end_of_day = datetime(now.year, now.month, now.day, 23, 59, 59, tzinfo=timezone.utc)
    out: list[float] = []
    for p in periods:
        start_iso = p.get("startTime")
        if not start_iso:
            continue
        try:
            ts = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
        except ValueError:
            continue
        if ts < now or ts > end_of_day:
            continue
        unit = (p.get("temperatureUnit") or "F").upper()
        t = float(p.get("temperature"))
        if unit == "C":
            t = t * 9.0 / 5.0 + 32.0
        out.append(t)
    return max(out) if out else None


async def fetch_distribution(
    client: httpx.AsyncClient,
    nws_host: str,
    open_meteo_host: str,
    lat: float,
    lon: float,
    target_date: date,
    *,
    user_agent: str,
    use_open_meteo: bool,
) -> ForecastDistribution | None:
    points: list[ForecastPoint] = []
    nws = await nws_long_range(client, nws_host, lat, lon, target_date, user_agent)
    if nws is not None:
        points.append(nws)
    if use_open_meteo:
        om = await open_meteo_long_range(client, open_meteo_host, lat, lon, target_date)
        if om is not None:
            points.append(om)
    return ensemble(points)
