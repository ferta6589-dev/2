import math

import pytest

from polyarb.weather_executor import Position, WeatherLiveExecutor, WeatherPaperExecutor
from polyarb.weather_markets import WeatherBucket
from polyarb.weather_strategy import EntryOrder, ExitOrder


def _bucket(slug="b1"):
    return WeatherBucket(
        slug=slug,
        title="t",
        token_yes=f"YES_{slug}",
        token_no=f"NO_{slug}",
        lo_c=17.0,
        hi_c=19.0,
    )


def test_position_lifecycle():
    p = Position(event_slug="e", bucket_slug="b", token_yes="Y")
    p.apply_buy(qty=20, price=0.20, ts=1.0)
    p.apply_buy(qty=10, price=0.30, ts=2.0)
    assert math.isclose(p.avg_cost, (20 * 0.20 + 10 * 0.30) / 30, abs_tol=1e-6)
    pnl = p.apply_sell(qty=10, price=0.40)
    assert pnl > 0
    res = p.apply_resolve(payout_per_share=1.0)
    assert math.isclose(p.realized_pnl, pnl + res, abs_tol=1e-6)
    assert p.closed


@pytest.mark.asyncio
async def test_paper_executor_round_trip(tmp_path):
    ex = WeatherPaperExecutor(tmp_path, prefix="weather_trades")
    bucket = _bucket()
    pos = await ex.fill_buy(
        "event-1",
        EntryOrder(bucket=bucket, qty=20.0, limit_price=0.20, reason="leader_on_metar"),
    )
    assert pos.qty == 20.0
    sold = await ex.fill_sell(
        "event-1",
        bucket,
        ExitOrder(bucket_slug=bucket.slug, qty=10.0, limit_price=0.30, reason="exceeded_by_observation"),
    )
    assert math.isclose(sold, 1.0, abs_tol=1e-6)
    resolved = await ex.record_resolve(pos, payout_per_share=1.0)
    assert math.isclose(resolved, 8.0, abs_tol=1e-6)
    ex.close()

    ex2 = WeatherPaperExecutor(tmp_path, prefix="weather_trades")
    replayed = ex2.positions[("event-1", bucket.slug)]
    assert replayed.qty == 0
    assert replayed.closed
    assert math.isclose(replayed.realized_pnl, sold + resolved, abs_tol=1e-6)
    ex2.close()


@pytest.mark.asyncio
async def test_live_executor_refuses():
    with pytest.raises(NotImplementedError):
        WeatherLiveExecutor()
