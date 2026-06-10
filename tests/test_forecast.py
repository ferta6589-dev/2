import math
from datetime import date

import httpx
import pytest
import respx

from polyarb import forecast


def test_prob_in_two_buckets_centered():
    """Sanity: 4°C window around μ with σ=1.0 gives ~95%."""
    dist = forecast.ForecastDistribution(mu_c=18.0, sigma_c=1.0, sources=["x"])
    p = dist.prob_in(16.0, 20.0)
    assert math.isclose(p, 0.9545, abs_tol=2e-3)


def test_prob_in_two_buckets_sigma_1_2():
    """4°C window around μ with σ=1.2 should be just above 90%."""
    dist = forecast.ForecastDistribution(mu_c=18.0, sigma_c=1.2, sources=["x"])
    p = dist.prob_in(16.0, 20.0)
    assert 0.90 <= p <= 0.91


def test_confidence_too_low():
    dist = forecast.ForecastDistribution(mu_c=18.0, sigma_c=2.0, sources=["x"])
    assert dist.confidence_too_low(max_sigma_c=1.2)
    assert not dist.confidence_too_low(max_sigma_c=3.0)


def test_blend_uses_inverse_variance():
    mm = [("ecmwf", 18.0), ("gfs", 18.5), ("icon", 17.5)]
    members = [18.1, 18.3, 18.0, 17.9, 18.2, 18.4]
    dist = forecast._blend_to_distribution(mm, members, sigma_floor=0.3)
    assert dist is not None
    # μ should be near 18°C
    assert 17.5 < dist.mu_c < 18.5
    # σ should be tight (multi-source agreement)
    assert dist.sigma_c < 1.5
    assert any("ecmwf_ens" in s for s in dist.sources)


def test_blend_returns_none_if_no_data():
    assert forecast._blend_to_distribution([], [], sigma_floor=0.3) is None


def test_blend_sigma_floor():
    """Even with extremely agreeing inputs, σ stays above floor."""
    mm = [("ecmwf", 18.0), ("gfs", 18.0)]
    members = [18.0, 18.0, 18.0, 18.0, 18.0]
    dist = forecast._blend_to_distribution(mm, members, sigma_floor=0.5)
    assert dist is not None
    assert dist.sigma_c >= 0.5


@respx.mock
@pytest.mark.asyncio
async def test_open_meteo_multi_model_parses():
    target = date(2026, 5, 14)
    payload = {
        "daily": {
            "time": ["2026-05-14"],
            "temperature_2m_max_ecmwf_ifs04": [18.4],
            "temperature_2m_max_gfs_seamless": [18.7],
            "temperature_2m_max_icon_seamless": [18.2],
        }
    }
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json=payload)
    )
    async with httpx.AsyncClient() as c:
        result = await forecast.open_meteo_multi_model(
            c, "https://api.open-meteo.com", 55.59, 37.27, target
        )
    assert len(result) == 3
    names = {n for n, _ in result}
    assert "ecmwf_ifs04" in names


@respx.mock
@pytest.mark.asyncio
async def test_open_meteo_ensemble_extracts_members():
    target = date(2026, 5, 14)
    payload = {
        "hourly": {
            "time": ["2026-05-14T12:00"],
            "temperature_2m": [18.4],
            "temperature_2m_member01": [18.5],
            "temperature_2m_member02": [18.2],
            "temperature_2m_member03": [18.7],
        }
    }
    respx.get("https://ensemble-api.open-meteo.com/v1/ensemble").mock(
        return_value=httpx.Response(200, json=payload)
    )
    async with httpx.AsyncClient() as c:
        members = await forecast.open_meteo_ensemble(
            c, "https://ensemble-api.open-meteo.com", 55.59, 37.27, target
        )
    # Main + 3 numbered members = 4 values
    assert len(members) == 4
    assert all(18.0 < m < 19.0 for m in members)


@respx.mock
@pytest.mark.asyncio
async def test_fetch_distribution_returns_dist_on_clean_data():
    target = date(2026, 5, 14)
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2026-05-14"],
                "temperature_2m_max_ecmwf_ifs04": [18.0],
                "temperature_2m_max_gfs_seamless": [18.2],
            }
        })
    )
    respx.get("https://ensemble-api.open-meteo.com/v1/ensemble").mock(
        return_value=httpx.Response(200, json={
            "hourly": {
                "time": ["2026-05-14T12:00"],
                "temperature_2m": [18.1],
                **{f"temperature_2m_member{i:02d}": [18.0 + 0.1 * i] for i in range(1, 11)},
            }
        })
    )
    async with httpx.AsyncClient() as c:
        dist = await forecast.fetch_distribution(
            c,
            "https://api.open-meteo.com",
            "https://ensemble-api.open-meteo.com",
            55.59, 37.27, target,
        )
    assert dist is not None
    assert 17.5 < dist.mu_c < 19.0
    assert dist.sigma_c > 0
