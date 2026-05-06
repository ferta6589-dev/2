import math
from datetime import date, datetime, timezone

import httpx
import pytest
import respx

from polyarb import forecast


def test_prob_in_known_gaussian():
    dist = forecast.ForecastDistribution(mu_f=70.0, sigma_f=2.0, sources=["x"])
    p = dist.prob_in(68.0, 72.0)
    assert math.isclose(p, 0.6826894, abs_tol=1e-3)


def test_prob_in_open_ended():
    dist = forecast.ForecastDistribution(mu_f=70.0, sigma_f=2.0, sources=["x"])
    p_below = dist.prob_in(None, 70.0)
    p_above = dist.prob_in(70.0, None)
    assert math.isclose(p_below, 0.5, abs_tol=1e-6)
    assert math.isclose(p_above, 0.5, abs_tol=1e-6)


def test_ensemble_inverse_variance_blend():
    a = forecast.ForecastPoint(mu_f=70.0, sigma_f=2.0, source="nws")
    b = forecast.ForecastPoint(mu_f=72.0, sigma_f=2.0, source="open_meteo")
    blended = forecast.ensemble([a, b])
    assert blended is not None
    assert math.isclose(blended.mu_f, 71.0, abs_tol=1e-6)
    assert math.isclose(blended.sigma_f, math.sqrt(2.0), abs_tol=1e-6)
    assert set(blended.sources) == {"nws", "open_meteo"}


def test_ensemble_single_inflates_sigma():
    a = forecast.ForecastPoint(mu_f=70.0, sigma_f=2.0, source="nws")
    blended = forecast.ensemble([a])
    assert blended is not None
    assert math.isclose(blended.sigma_f, 2.5, abs_tol=1e-6)


def test_ensemble_empty_returns_none():
    assert forecast.ensemble([]) is None


@respx.mock
@pytest.mark.asyncio
async def test_open_meteo_long_range_parses():
    target = date(2026, 5, 7)
    respx.get("https://api.open-meteo.com/v1/forecast").mock(
        return_value=httpx.Response(200, json={
            "daily": {
                "time": ["2026-05-07"],
                "temperature_2m_max": [73.4],
            },
        })
    )
    async with httpx.AsyncClient() as client:
        fp = await forecast.open_meteo_long_range(
            client, "https://api.open-meteo.com", 40.78, -73.97, target
        )
    assert fp is not None
    assert fp.source == "open_meteo"
    assert math.isclose(fp.mu_f, 73.4, abs_tol=0.01)


@respx.mock
@pytest.mark.asyncio
async def test_nws_observation_celsius_to_f():
    respx.get("https://api.weather.gov/stations/KNYC/observations/latest").mock(
        return_value=httpx.Response(200, json={
            "properties": {
                "temperature": {"value": 20.0, "unitCode": "wmoUnit:degC"},
                "timestamp": "2026-05-07T18:00:00Z",
            }
        })
    )
    async with httpx.AsyncClient() as client:
        obs = await forecast.nws_observation(
            client, "https://api.weather.gov", "KNYC", "polyarb-test"
        )
    assert obs is not None
    assert math.isclose(obs.temp_f, 68.0, abs_tol=0.01)
