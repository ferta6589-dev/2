from datetime import date, datetime, timezone

import httpx
import pytest
import respx

from polyarb import metar


UUWW_RAW = """2026/05/12 12:30
UUWW 121230Z 19006MPS 9999 SCT040 18/09 Q1015 R06/CLRD70 NOSIG"""

UUWW_T_GROUP = """2026/05/12 14:00
UUWW 121400Z 22008MPS 9999 BKN050 22/08 Q1014 R06/CLRD60 NOSIG RMK QFE748 T02180081"""


def test_parse_metar_body_temp():
    rep = metar.parse_metar(UUWW_RAW, fallback_date=date(2026, 5, 12))
    assert rep is not None
    assert rep.station_id == "UUWW"
    assert rep.temperature_c == 18.0
    assert rep.observation_ts == datetime(2026, 5, 12, 12, 30, tzinfo=timezone.utc)


def test_parse_metar_t_group_precision_wins():
    rep = metar.parse_metar(UUWW_T_GROUP, fallback_date=date(2026, 5, 12))
    assert rep is not None
    assert rep.temperature_c == 21.8  # T-group gives tenths
    assert rep.observation_ts.hour == 14


def test_parse_metar_negative_temperature():
    raw = "UUWW 121230Z 00000MPS 9999 OVC020 M05/M07 Q1020"
    assert metar.parse_metar_temp(raw) == -5.0


def test_parse_metar_t_group_negative():
    raw = "UUWW 121230Z 00000MPS 9999 OVC020 M05/M07 Q1020 RMK T10120023"
    assert metar.parse_metar_temp(raw) == -1.2


@respx.mock
@pytest.mark.asyncio
async def test_fetch_tgftp_station_returns_report():
    respx.get("https://tgftp.nws.noaa.gov/data/observations/metar/stations/UUWW.TXT").mock(
        return_value=httpx.Response(
            200, text=UUWW_RAW, headers={"Last-Modified": "Tue, 12 May 2026 12:31:00 GMT"}
        )
    )
    state = metar._PollState()
    async with httpx.AsyncClient() as c:
        rep = await metar.fetch_tgftp_station(c, "UUWW", state=state)
    assert rep is not None
    assert rep.temperature_c == 18.0
    assert state.last_modified == "Tue, 12 May 2026 12:31:00 GMT"


@respx.mock
@pytest.mark.asyncio
async def test_fetch_tgftp_station_uses_conditional_get():
    respx.get("https://tgftp.nws.noaa.gov/data/observations/metar/stations/UUWW.TXT").mock(
        return_value=httpx.Response(304)
    )
    state = metar._PollState(last_modified="Tue, 12 May 2026 12:31:00 GMT")
    async with httpx.AsyncClient() as c:
        rep = await metar.fetch_tgftp_station(c, "UUWW", state=state)
    assert rep is None


@respx.mock
@pytest.mark.asyncio
async def test_fetch_tgftp_cycle_picks_freshest_for_station():
    cycle_text = (
        "2026/05/12 13:00\nUUEE 121300Z 18005MPS 9999 SCT040 20/08 Q1015\n\n"
        "2026/05/12 13:00\nUUWW 121300Z 19008MPS 9999 SCT040 19/09 Q1015\n\n"
        "2026/05/12 13:30\nUUWW 121330Z 19008MPS 9999 SCT040 20/09 Q1015\n"
    )
    respx.get("https://tgftp.nws.noaa.gov/data/observations/metar/cycles/13Z.TXT").mock(
        return_value=httpx.Response(200, text=cycle_text)
    )
    async with httpx.AsyncClient() as c:
        rep = await metar.fetch_tgftp_cycle(
            c, "UUWW", now=datetime(2026, 5, 12, 13, 45, tzinfo=timezone.utc)
        )
    assert rep is not None
    assert rep.station_id == "UUWW"
    assert rep.observation_ts.minute == 30
    assert rep.temperature_c == 20.0


def test_publish_window():
    assert metar.in_publish_window(datetime(2026, 5, 12, 12, 30, tzinfo=timezone.utc))
    assert metar.in_publish_window(datetime(2026, 5, 12, 12, 58, tzinfo=timezone.utc))
    assert metar.in_publish_window(datetime(2026, 5, 12, 13, 5, tzinfo=timezone.utc))
    assert not metar.in_publish_window(datetime(2026, 5, 12, 12, 15, tzinfo=timezone.utc))


def test_next_poll_delay():
    assert metar.next_poll_delay(in_window=True, fast_period_s=20, slow_period_s=180) == 20
    assert metar.next_poll_delay(in_window=False, fast_period_s=20, slow_period_s=180) == 180


def test_daily_max_tracker_updates_and_rounds():
    tr = metar.DailyMaxTracker(target_date=date(2026, 5, 12))
    rep1 = metar.MetarReport("UUWW", datetime(2026, 5, 12, 6, 0, tzinfo=timezone.utc), 12.7, "r1", "x")
    rep2 = metar.MetarReport("UUWW", datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc), 18.3, "r2", "x")
    rep3 = metar.MetarReport("UUWW", datetime(2026, 5, 12, 15, 0, tzinfo=timezone.utc), 17.9, "r3", "x")
    assert tr.update(rep1) is True
    assert tr.update(rep2) is True
    assert tr.update(rep3) is False
    assert tr.max_c == 18.3
    assert tr.rounded_max_c == 18  # truncate toward zero, matches resolver


def test_daily_max_tracker_rejects_other_dates():
    tr = metar.DailyMaxTracker(target_date=date(2026, 5, 12))
    rep = metar.MetarReport("UUWW", datetime(2026, 5, 11, 23, 0, tzinfo=timezone.utc), 30.0, "r", "x")
    assert tr.update(rep) is False
    assert tr.max_c is None


def test_daily_max_tracker_roundtrip_persistence(tmp_path):
    tr = metar.DailyMaxTracker(target_date=date(2026, 5, 12))
    tr.update(metar.MetarReport("UUWW", datetime(2026, 5, 12, 12, 0, tzinfo=timezone.utc), 18.3, "r", "x"))
    metar.save_trackers(tmp_path / "trk.json", {"e1": tr})
    loaded = metar.load_trackers(tmp_path / "trk.json")
    assert loaded["e1"].max_c == 18.3
    assert loaded["e1"].target_date == date(2026, 5, 12)
