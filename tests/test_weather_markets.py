import json
from datetime import date

import httpx
import pytest
import respx

from polyarb import weather_markets


def test_parse_bucket_bounds_celsius_default():
    assert weather_markets.parse_bucket_bounds("17-19°C") == (17.0, 19.0)
    assert weather_markets.parse_bucket_bounds("between 19 and 21 degrees") == (19.0, 21.0)


def test_parse_bucket_bounds_below_above():
    assert weather_markets.parse_bucket_bounds("less than 11°C") == (None, 11.0)
    assert weather_markets.parse_bucket_bounds(">25°C") == (25.0, None)


def test_parse_bucket_bounds_fahrenheit_converts():
    lo, hi = weather_markets.parse_bucket_bounds("60-65°F")
    assert round(lo, 2) == 15.56
    assert round(hi, 2) == 18.33


def test_moscow_resolver_known():
    sid, lat, lon = weather_markets.CITY_RESOLVERS["MOSCOW"]
    assert sid == "UUWW"
    assert round(lat, 2) == 55.59


@respx.mock
@pytest.mark.asyncio
async def test_fetch_weather_events_moscow():
    payload = [{
        "slug": "highest-temperature-in-moscow-on-may-12-2026",
        "title": "Highest temperature in Moscow on May 12, 2026?",
        "endDate": "2026-05-13T03:00:00Z",
        "markets": [
            _mkt("less-than-11", "Less than 11°C", "Y_LT11"),
            _mkt("11-13", "Between 11 and 13°C", "Y_1113"),
            _mkt("13-15", "Between 13 and 15°C", "Y_1315"),
            _mkt("15-17", "Between 15 and 17°C", "Y_1517"),
            _mkt("17-19", "Between 17 and 19°C", "Y_1719"),
            _mkt("19-21", "Between 19 and 21°C", "Y_1921"),
            _mkt("21-23", "Between 21 and 23°C", "Y_2123"),
            _mkt("above-23", "Greater than 23°C", "Y_GT23"),
        ],
    }]
    respx.get("https://gamma-api.polymarket.com/events").mock(
        return_value=httpx.Response(200, json=payload)
    )
    async with httpx.AsyncClient() as client:
        events = await weather_markets.fetch_weather_events(
            client, "https://gamma-api.polymarket.com", cities=None
        )
    assert len(events) == 1
    e = events[0]
    assert e.city == "MOSCOW"
    assert e.station_id == "UUWW"
    assert e.target_date == date(2026, 5, 12)
    assert len(e.buckets) == 8


def _mkt(slug_suffix: str, question: str, token_prefix: str) -> dict:
    return {
        "slug": f"moscow-may-12-{slug_suffix}",
        "question": question,
        "clobTokenIds": json.dumps([f"{token_prefix}_YES", f"{token_prefix}_NO"]),
        "outcomes": json.dumps(["Yes", "No"]),
    }
