"""Multi-source forecast ensemble for the 2-bucket pre-entry strategy.

Combines free numerical weather prediction (NWP) models into a Gaussian
distribution ``N(μ, σ)`` for the target date's daily-max temperature, in °C.

Primary source: Open-Meteo multi-model endpoint (ECMWF IFS, GFS, ICON, GEM,
JMA, ARPEGE, UKMO) — free, no key, single HTTP call returns daily-max from
each model.

Secondary source: Open-Meteo ECMWF ENS — 51-member ensemble forecast that
gives a true probabilistic distribution per timestep.

We blend them by inverse-variance weighting. The ensemble σ is the dominant
source of uncertainty; multi-model σ provides bias robustness.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from datetime import date

import httpx
import structlog

log = structlog.get_logger("polyarb.forecast")


@dataclass(frozen=True)
class ForecastDistribution:
    mu_c: float
    sigma_c: float
    sources: list[str] = field(default_factory=list)
    members: list[float] = field(default_factory=list)

    def prob_in(self, lo_c: float | None, hi_c: float | None) -> float:
        """P(lo_c <= temp <= hi_c) under N(mu, sigma)."""
        return _gaussian_prob_between(self.mu_c, self.sigma_c, lo_c, hi_c)

    def confidence_too_low(self, max_sigma_c: float) -> bool:
        return self.sigma_c > max_sigma_c


def _gaussian_prob_between(
    mu: float, sigma: float, lo: float | None, hi: float | None
) -> float:
    if sigma <= 0:
        sigma = 0.3
    z_lo = -math.inf if lo is None else (lo - mu) / (sigma * math.sqrt(2))
    z_hi = math.inf if hi is None else (hi - mu) / (sigma * math.sqrt(2))
    cdf_lo = 0.0 if z_lo == -math.inf else 0.5 * (1.0 + math.erf(z_lo))
    cdf_hi = 1.0 if z_hi == math.inf else 0.5 * (1.0 + math.erf(z_hi))
    return max(0.0, cdf_hi - cdf_lo)


_OM_MODELS = "ecmwf_ifs04,gfs_seamless,icon_seamless,gem_seamless,jma_seamless,meteofrance_arpege_world,ukmo_seamless"


async def open_meteo_multi_model(
    client: httpx.AsyncClient,
    host: str,
    lat: float,
    lon: float,
    target: date,
) -> list[tuple[str, float]]:
    """Pull daily_max from each NWP model. Returns [(model_name, temp_c), ...]."""
    try:
        r = await client.get(
            f"{host}/v1/forecast",
            params={
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "daily": "temperature_2m_max",
                "temperature_unit": "celsius",
                "timezone": "UTC",
                "start_date": target.isoformat(),
                "end_date": target.isoformat(),
                "models": _OM_MODELS,
            },
            timeout=15.0,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        log.warning("open_meteo_multi_failed", err=str(e))
        return []

    out: list[tuple[str, float]] = []
    daily = data.get("daily", {})
    for key, value in daily.items():
        if not key.startswith("temperature_2m_max"):
            continue
        if isinstance(value, list) and value and value[0] is not None:
            model = key[len("temperature_2m_max_"):] if "_" in key else "default"
            out.append((model, float(value[0])))
    return out


async def open_meteo_ensemble(
    client: httpx.AsyncClient,
    host: str,
    lat: float,
    lon: float,
    target: date,
) -> list[float]:
    """Pull 51-member ECMWF ENS daily_max values for target_date."""
    try:
        r = await client.get(
            f"{host}/v1/ensemble",
            params={
                "latitude": f"{lat:.4f}",
                "longitude": f"{lon:.4f}",
                "hourly": "temperature_2m",
                "temperature_unit": "celsius",
                "timezone": "UTC",
                "start_date": target.isoformat(),
                "end_date": target.isoformat(),
                "models": "ecmwf_ifs025",
            },
            timeout=20.0,
        )
        r.raise_for_status()
        data = r.json()
    except httpx.HTTPError as e:
        log.warning("open_meteo_ensemble_failed", err=str(e))
        return []

    hourly = data.get("hourly", {})
    members: list[float] = []
    for key, value in hourly.items():
        if not key.startswith("temperature_2m_member"):
            continue
        if isinstance(value, list) and value:
            valid = [v for v in value if v is not None]
            if valid:
                members.append(max(valid))

    main_key_value = hourly.get("temperature_2m")
    if isinstance(main_key_value, list) and main_key_value:
        valid = [v for v in main_key_value if v is not None]
        if valid:
            members.insert(0, max(valid))

    return members


def _blend_to_distribution(
    multi_model: list[tuple[str, float]],
    ensemble_members: list[float],
    sigma_floor: float = 0.4,
) -> ForecastDistribution | None:
    """Inverse-variance blend of multi-model point forecasts + ENS spread."""
    if not multi_model and not ensemble_members:
        return None

    parts: list[tuple[float, float, str]] = []  # (mu, sigma, source_tag)
    sources: list[str] = []

    if ensemble_members and len(ensemble_members) >= 5:
        mu_ens = statistics.fmean(ensemble_members)
        sigma_ens = max(statistics.stdev(ensemble_members), sigma_floor)
        parts.append((mu_ens, sigma_ens, "ecmwf_ens"))
        sources.append(f"ecmwf_ens({len(ensemble_members)})")

    if multi_model:
        values = [v for _, v in multi_model]
        mu_mm = statistics.fmean(values)
        if len(values) >= 2:
            sigma_mm = max(statistics.stdev(values), sigma_floor)
        else:
            sigma_mm = 1.5
        parts.append((mu_mm, sigma_mm, "multi_model"))
        sources.extend([f"om:{m}" for m, _ in multi_model])

    if not parts:
        return None

    weights = [1.0 / (s ** 2) for _, s, _ in parts]
    total_w = sum(weights)
    mu = sum(w * m for w, (m, _, _) in zip(weights, parts)) / total_w
    sigma = math.sqrt(1.0 / total_w)
    sigma = max(sigma, sigma_floor)

    all_members = list(ensemble_members) + [v for _, v in multi_model]
    return ForecastDistribution(
        mu_c=mu,
        sigma_c=sigma,
        sources=sources,
        members=all_members,
    )


async def fetch_distribution(
    client: httpx.AsyncClient,
    open_meteo_host: str,
    ensemble_host: str,
    lat: float,
    lon: float,
    target: date,
    *,
    sigma_floor: float = 0.4,
) -> ForecastDistribution | None:
    """Public entrypoint: pull all sources, blend, return distribution."""
    multi = await open_meteo_multi_model(client, open_meteo_host, lat, lon, target)
    members = await open_meteo_ensemble(client, ensemble_host, lat, lon, target)
    dist = _blend_to_distribution(multi, members, sigma_floor=sigma_floor)
    if dist is not None:
        log.info(
            "forecast_distribution",
            mu_c=round(dist.mu_c, 2),
            sigma_c=round(dist.sigma_c, 2),
            n_sources=len(dist.sources),
            n_members=len(dist.members),
        )
    return dist
