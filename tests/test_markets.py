import json

import httpx
import pytest
import respx

from polyarb import markets


def test_window_ts_aligned_to_300():
    assert markets.current_window_ts(now=1_700_000_123) == 1_700_000_100
    assert markets.current_window_ts(now=1_700_000_400) == 1_700_000_400


def test_slug_format():
    assert markets.slug_for("BTC", 1_700_000_100) == "btc-updown-5m-1700000100"


@respx.mock
@pytest.mark.asyncio
async def test_fetch_market_parses_yes_no_outcomes():
    payload = [{
        "slug": "btc-updown-5m-1700000100",
        "clobTokenIds": json.dumps(["TOK_YES", "TOK_NO"]),
        "outcomes": json.dumps(["Up", "Down"]),
        "endDate": "2024-11-14T20:30:00Z",
        "orderMinSize": 5,
        "orderPriceMinTickSize": 0.01,
        "closed": False,
    }]
    respx.get("https://gamma-api.polymarket.com/markets").mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        m = await markets.fetch_market(client, "https://gamma-api.polymarket.com", "btc-updown-5m-1700000100", "BTC")
    assert m is not None
    assert m.yes_token == "TOK_YES"
    assert m.no_token == "TOK_NO"
    assert m.minimum_order_size == 5.0


@respx.mock
@pytest.mark.asyncio
async def test_fetch_market_returns_none_when_closed():
    payload = [{
        "slug": "btc-updown-5m-1700000100",
        "clobTokenIds": ["TOK_YES", "TOK_NO"],
        "outcomes": ["Up", "Down"],
        "endDate": "2024-11-14T20:30:00Z",
        "closed": True,
    }]
    respx.get("https://gamma-api.polymarket.com/markets").mock(return_value=httpx.Response(200, json=payload))
    async with httpx.AsyncClient() as client:
        m = await markets.fetch_market(client, "https://gamma-api.polymarket.com", "btc-updown-5m-1700000100", "BTC")
    assert m is None


@respx.mock
@pytest.mark.asyncio
async def test_discover_iterates_assets():
    respx.get("https://gamma-api.polymarket.com/markets").mock(return_value=httpx.Response(200, json=[]))
    async with httpx.AsyncClient() as client:
        out = await markets.discover(client, "https://gamma-api.polymarket.com", ["BTC", "ETH"])
    assert out == []
