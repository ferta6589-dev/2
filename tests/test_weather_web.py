from fastapi.testclient import TestClient

from polyarb.state import AppState
from polyarb.weather_state import WeatherAppState
from polyarb.web import make_app


def test_weather_state_endpoint_disabled():
    app = make_app(AppState())
    client = TestClient(app)
    r = client.get("/weather/state.json")
    assert r.status_code == 200
    body = r.json()
    assert body == {"enabled": False, "events": []}


def test_weather_state_endpoint_enabled():
    ws = WeatherAppState()
    ws.mode = "demo"
    app = make_app(AppState(), weather_state=ws)
    client = TestClient(app)
    r = client.get("/weather/state.json")
    assert r.status_code == 200
    body = r.json()
    assert body["enabled"] is True
    assert body["mode"] == "demo"
    assert "events" in body
    assert "metar_polls" in body
    assert body["totals"]["open_events"] == 0


def test_crypto_state_endpoint_unchanged():
    app = make_app(AppState())
    client = TestClient(app)
    r = client.get("/state.json")
    assert r.status_code == 200
    assert r.json()["mode"] == "idle"
