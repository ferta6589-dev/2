import asyncio

import pytest

from polyarb.config import Settings
from polyarb.pnl import summarize
from polyarb.weather_executor import WeatherPaperExecutor
from polyarb.weather_sim import run_demo
from polyarb.weather_state import WeatherAppState


@pytest.mark.asyncio
async def test_metar_sim_end_to_end(tmp_path):
    settings = Settings(
        log_dir=tmp_path,
        weather_per_event_budget_usd=30.0,
        weather_max_price=0.40,
    )
    state = WeatherAppState()
    executor = WeatherPaperExecutor(
        tmp_path, prefix=settings.weather_log_filename_prefix, state=state
    )
    stop = asyncio.Event()

    await run_demo(state, settings, executor, stop, tick_period_s=0.0)
    executor.close()

    files = list(tmp_path.glob(f"{settings.weather_log_filename_prefix}_*.jsonl"))
    assert files, "expected at least one trade-log file"

    summary = summarize(tmp_path, prefix=settings.weather_log_filename_prefix)
    assert summary["buys"] >= 1
    assert summary["resolves"] >= 1
    assert summary["events_traded"] == 1


@pytest.mark.asyncio
async def test_metar_sim_updates_daily_max(tmp_path):
    settings = Settings(log_dir=tmp_path)
    state = WeatherAppState()
    executor = WeatherPaperExecutor(tmp_path, prefix="weather_trades", state=state)
    stop = asyncio.Event()
    await run_demo(state, settings, executor, stop, tick_period_s=0.0)
    executor.close()

    snap = state.snapshot()
    assert len(snap["events"]) == 1
    e = snap["events"][0]
    assert e["city"] == "MOSCOW"
    assert e["station_id"] == "UUWW"
    assert e["observed_max_c"] is not None
    assert e["observed_max_c"] >= 18.0  # peak of timeline
    # Resolver-precision rounding (whole °C)
    assert e["rounded_max_c"] == 18
