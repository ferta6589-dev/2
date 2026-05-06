import math

import pytest

from polyarb.weather_executor import Position, WeatherPaperExecutor
from polyarb.weather_markets import WeatherBucket
from polyarb.weather_strategy import EntryOrder, ExitOrder


def _bucket(slug="b1"):
    return WeatherBucket(
        slug=slug,
        title="t",
        token_yes=f"YES_{slug}",
        token_no=f"NO_{slug}",
        lo_f=70.0,
        hi_f=75.0,
        minimum_order_size=5.0,
    )


def test_position_apply_buy_blends_avg_cost():
    p = Position(event_slug="e", bucket_slug="b", token_yes="Y")
    p.apply_buy(qty=10, price=0.30, ts=1.0)
    p.apply_buy(qty=10, price=0.20, ts=2.0)
    assert math.isclose(p.avg_cost, 0.25, abs_tol=1e-6)
    assert p.qty == 20


def test_position_apply_sell_realizes_pnl():
    p = Position(event_slug="e", bucket_slug="b", token_yes="Y")
    p.apply_buy(qty=10, price=0.20, ts=1.0)
    pnl = p.apply_sell(qty=10, price=0.30)
    assert math.isclose(pnl, 1.0, abs_tol=1e-6)
    assert p.qty == 0
    assert p.closed


def test_position_apply_resolve_pays_out():
    p = Position(event_slug="e", bucket_slug="b", token_yes="Y")
    p.apply_buy(qty=10, price=0.30, ts=1.0)
    pnl = p.apply_resolve(payout_per_share=1.0)
    assert math.isclose(pnl, 7.0, abs_tol=1e-6)
    assert p.qty == 0


@pytest.mark.asyncio
async def test_paper_executor_round_trip(tmp_path):
    ex = WeatherPaperExecutor(tmp_path, prefix="weather_trades")
    bucket = _bucket()
    order = EntryOrder(bucket=bucket, role="central", qty=20.0, limit_price=0.30)
    pos = await ex.fill_buy("event-1", order)
    assert pos.qty == 20.0
    sold = await ex.fill_sell(
        "event-1", bucket, ExitOrder(bucket_slug=bucket.slug, qty=10.0, limit_price=0.45, reason="loser_classified")
    )
    assert math.isclose(sold, 1.5, abs_tol=1e-6)
    resolved = await ex.record_resolve(pos, payout_per_share=1.0)
    assert math.isclose(resolved, 7.0, abs_tol=1e-6)
    ex.close()

    files = sorted(tmp_path.glob("weather_trades_*.jsonl"))
    assert files, "expected daily trade log"

    ex2 = WeatherPaperExecutor(tmp_path, prefix="weather_trades")
    replayed = ex2.positions[("event-1", bucket.slug)]
    assert replayed.qty == 0
    assert replayed.closed
    assert math.isclose(replayed.realized_pnl, sold + resolved, abs_tol=1e-6)
    ex2.close()


@pytest.mark.asyncio
async def test_live_executor_refuses():
    from polyarb.weather_executor import WeatherLiveExecutor
    with pytest.raises(NotImplementedError):
        WeatherLiveExecutor()
