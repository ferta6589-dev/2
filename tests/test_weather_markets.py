import json
from datetime import date

import httpx
import pytest
import respx

from polyarb import weather_markets


def test_parse_bucket_bounds_range_fahrenheit():
    assert weather_markets.parse_bucket_bounds(
        "Highest temperature in NYC on May 7 between 65 and 70 degrees"
    ) == (65.0, 70.0)


def test_parse_bucket_bounds_range_with_unit_marker():
    assert weather_markets.parse_bucket_bounds("60-65°F") == (60.0, 65.0)


def test_parse_bucket_bounds_below():
    assert weather_markets.parse_bucket_bounds("Less than 60°F") == (None, 60.0)
    assert weather_markets.parse_bucket_bounds("<60") == (None, 60.0)


def test_parse_bucket_bounds_above():
    assert weather_markets.parse_bucket_bounds(">75°F") == (75.0, None)
    assert weather_markets.parse_bucket_bounds("at least 80 degrees") == (80.0, None)


def test_parse_bucket_bounds_celsius_converts():
    lo, hi = weather_markets.parse_bucket_bounds("20-25°C")
    assert round(lo, 1) == 68.0
    assert round(hi, 1) == 77.0


def test_parse_bucket_bounds_unparseable():
    assert weather_markets.parse_bucket_bounds("partly sunny") is None


def test_is_contiguous_accepts_open_ended_at_ends():
    buckets = [
        weather_markets.WeatherBucket(slug="a", title="<60", token_yes="A", token_no="An",
                                      lo_f=None, hi_f=60.0),
        weather_markets.WeatherBucket(slug="b", title="60-65", token_yes="B", token_no="Bn",
                                      lo_f=60.0, hi_f=65.0),
        weather_markets.WeatherBucket(slug="c", title=">65", token_yes="C", token_no="Cn",
                                      lo_f=65.0, hi_f=None),
    ]
    assert weather_markets._is_contiguous(buckets)


def test_is_contiguous_rejects_gap():
    buckets = [
        weather_markets.WeatherBucket(slug="a", title="60-65", token_yes="A", token_no="An",
                                      lo_f=60.0, hi_f=65.0),
        weather_markets.WeatherBucket(slug="b", title="70-75", token_yes="B", token_no="Bn",
                                      lo_f=70.0, hi_f=75.0),
    ]
    assert not weather_markets._is_contiguous(buckets)


@respx.mock
@pytest.mark.asyncio
async def test_fetch_weather_events_parses_buckets():
    payload = [{
        "slug": "nyc-temp-may-7",
        "title": "Highest temperature in NYC on May 7, 2026?",
        "endDate": "2026-05-08T04:00:00Z",
        "markets": [
            {
                "slug": "nyc-temp-may-7-below-60",
                "question": "Less than 60°F",
                "clobTokenIds": json.dumps(["YES_LT60", "NO_LT60"]),
                "outcomes": json.dumps(["Yes", "No"]),
            },
            {
                "slug": "nyc-temp-may-7-60-65",
                "question": "Between 60 and 65 degrees",
                "clobTokenIds": json.dumps(["YES_6065", "NO_6065"]),
                "outcomes": json.dumps(["Yes", "No"]),
            },
            {
                "slug": "nyc-temp-may-7-65-70",
                "question": "Between 65 and 70 degrees",
                "clobTokenIds": json.dumps(["YES_6570", "NO_6570"]),
                "outcomes": json.dumps(["Yes", "No"]),
            },
            {
                "slug": "nyc-temp-may-7-70-75",
                "question": "Between 70 and 75 degrees",
                "clobTokenIds": json.dumps(["YES_7075", "NO_7075"]),
                "outcomes": json.dumps(["Yes", "No"]),
            },
            {
                "slug": "nyc-temp-may-7-above-75",
                "question": "Greater than 75°F",
                "clobTokenIds": json.dumps(["YES_GT75", "NO_GT75"]),
                "outcomes": json.dumps(["Yes", "No"]),
            },
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
    assert e.city == "NYC"
    assert e.station_id == "KNYC"
    assert e.target_date == date(2026, 5, 7)
    assert len(e.buckets) == 5
    assert e.buckets[0].lo_f is None and e.buckets[0].hi_f == 60.0
    assert e.buckets[-1].lo_f == 75.0 and e.buckets[-1].hi_f is None


@respx.mock
@pytest.mark.asyncio
async def test_fetch_weather_events_filters_by_city():
    payload = [{
        "slug": "ldn-temp",
        "title": "Highest temperature in London on May 7, 2026?",
        "markets": [],
    }]
    respx.get("https://gamma-api.polymarket.com/events").mock(
        return_value=httpx.Response(200, json=payload)
    )
    async with httpx.AsyncClient() as client:
        events = await weather_markets.fetch_weather_events(
            client, "https://gamma-api.polymarket.com", cities=["NYC"]
        )
    assert events == []
