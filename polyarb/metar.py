"""Fast METAR ingestion for the weather strategy.

Primary source: NOAA tgftp per-station file
    https://tgftp.nws.noaa.gov/data/observations/metar/stations/{ICAO}.TXT

It's a tiny plain-text file overwritten in place each time a new METAR for that
station hits NOAA's GTS ingest. End-to-end latency from issue is typically
30-90 seconds — the fastest free path we found.

Fallback: NOAA tgftp cycles file (`cycles/{HH}Z.TXT`, all stations for the
current hour) and Ogimet HTML.

We expose three core pieces:

* :func:`parse_metar` — pulls (temperature_c, observation_ts) out of a raw
  METAR string. Prefers the high-precision ``Txxxx`` remark group when present
  (tenths of °C, signed); falls back to the ``TT/DD`` body group (whole °C).
* :func:`fetch_metar` — pulls the latest report from NOAA tgftp (primary) with
  Ogimet fallback, using ``If-Modified-Since`` to keep requests cheap.
* :class:`DailyMaxTracker` — accumulates observations for one UTC date and
  exposes the running max rounded to whole °C (matches Polymarket resolver
  precision).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from email.utils import formatdate, parsedate_to_datetime
from pathlib import Path

import httpx
import structlog

log = structlog.get_logger("polyarb.metar")

NOAA_TGFTP_HOST = "https://tgftp.nws.noaa.gov"


@dataclass(frozen=True)
class MetarReport:
    station_id: str
    observation_ts: datetime
    temperature_c: float
    raw: str
    source: str


@dataclass
class _PollState:
    last_modified: str | None = None
    last_raw: str | None = None


_TIME_RE = re.compile(r"\b(\d{2})(\d{2})(\d{2})Z\b")
_T_GROUP_RE = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_BODY_TEMP_RE = re.compile(r"\b(M?\d{2})/(M?\d{2}|//)\b")


def parse_metar_time(raw: str, *, fallback_date: date | None = None) -> datetime | None:
    """Return the UTC timestamp from the ``DDhhmmZ`` group, anchored to today.

    METARs only carry day-of-month + time, so we pin the year/month using the
    provided ``fallback_date`` (defaults to today UTC). If the parsed day is in
    the future relative to that, we roll the month back.
    """
    m = _TIME_RE.search(raw)
    if not m:
        return None
    day, hour, minute = (int(x) for x in m.groups())
    today = fallback_date or datetime.now(timezone.utc).date()
    year, month = today.year, today.month
    if day > today.day:
        month -= 1
        if month == 0:
            month, year = 12, year - 1
    try:
        return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_metar_temp(raw: str) -> float | None:
    """Return temperature in °C. Prefer ``Txxxx`` remark (tenths) over body."""
    m = _T_GROUP_RE.search(raw)
    if m:
        t_sign, t_val, _td_sign, _td_val = m.groups()
        sign = -1.0 if t_sign == "1" else 1.0
        return sign * int(t_val) / 10.0
    for m in _BODY_TEMP_RE.finditer(raw):
        t_raw = m.group(1)
        if t_raw.startswith("M"):
            return -float(t_raw[1:])
        if t_raw.isdigit():
            return float(t_raw)
    return None


def parse_metar(raw: str, *, fallback_date: date | None = None) -> MetarReport | None:
    raw = raw.strip()
    if not raw:
        return None
    # The tgftp per-station file is two lines: timestamp header + raw METAR.
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    if not lines:
        return None
    station_match = re.match(r"^([A-Z]{4})\b", lines[-1])
    if not station_match:
        for line in lines:
            station_match = re.match(r"^([A-Z]{4})\b", line)
            if station_match:
                break
    if not station_match:
        return None
    station = station_match.group(1)
    metar_line = next(l for l in lines if l.startswith(station))
    temp = parse_metar_temp(metar_line)
    ts = parse_metar_time(metar_line, fallback_date=fallback_date)
    if temp is None or ts is None:
        return None
    return MetarReport(
        station_id=station,
        observation_ts=ts,
        temperature_c=temp,
        raw=metar_line,
        source="raw",
    )


async def fetch_tgftp_station(
    client: httpx.AsyncClient,
    station_id: str,
    state: _PollState | None = None,
) -> MetarReport | None:
    """GET ``stations/{station}.TXT`` with conditional ``If-Modified-Since``.

    ``state`` is reused across calls to avoid re-downloading when unchanged.
    """
    state = state if state is not None else _PollState()
    url = f"{NOAA_TGFTP_HOST}/data/observations/metar/stations/{station_id}.TXT"
    headers: dict[str, str] = {"User-Agent": "polyarb-metar/0.2"}
    if state.last_modified:
        headers["If-Modified-Since"] = state.last_modified
    try:
        r = await client.get(url, headers=headers, timeout=5.0)
    except httpx.HTTPError as e:
        log.warning("tgftp_failed", station=station_id, err=str(e))
        return None
    if r.status_code == 304:
        return None
    if r.status_code != 200:
        log.warning("tgftp_status", station=station_id, status=r.status_code)
        return None
    body = r.text
    if body == state.last_raw:
        return None
    state.last_raw = body
    if "Last-Modified" in r.headers:
        state.last_modified = r.headers["Last-Modified"]
    rep = parse_metar(body)
    if rep is None:
        log.debug("tgftp_unparsed", station=station_id, body_head=body[:120])
        return None
    return MetarReport(
        station_id=rep.station_id,
        observation_ts=rep.observation_ts,
        temperature_c=rep.temperature_c,
        raw=rep.raw,
        source="tgftp_station",
    )


async def fetch_tgftp_cycle(
    client: httpx.AsyncClient,
    station_id: str,
    now: datetime | None = None,
) -> MetarReport | None:
    """Grep the per-hour cycle dump for ``station_id`` — sometimes lands
    seconds before the per-station file rolls.
    """
    now = now or datetime.now(timezone.utc)
    url = f"{NOAA_TGFTP_HOST}/data/observations/metar/cycles/{now.hour:02d}Z.TXT"
    try:
        r = await client.get(url, headers={"User-Agent": "polyarb-metar/0.2"}, timeout=8.0)
    except httpx.HTTPError as e:
        log.warning("tgftp_cycle_failed", err=str(e))
        return None
    if r.status_code != 200:
        return None
    best: MetarReport | None = None
    for chunk in r.text.split("\n\n"):
        chunk = chunk.strip()
        if not chunk or station_id not in chunk:
            continue
        rep = parse_metar(chunk)
        if rep is None or rep.station_id != station_id:
            continue
        if best is None or rep.observation_ts > best.observation_ts:
            best = MetarReport(
                station_id=rep.station_id,
                observation_ts=rep.observation_ts,
                temperature_c=rep.temperature_c,
                raw=rep.raw,
                source="tgftp_cycle",
            )
    return best


async def fetch_ogimet(client: httpx.AsyncClient, station_id: str) -> MetarReport | None:
    """Ogimet HTML fallback. Use sparingly — they ask for ≤ 1 req/min."""
    url = (
        f"https://www.ogimet.com/display_metars2.php?"
        f"lang=en&tipo=ALL&ord=REV&nil=SI&fmt=txt&lugar={station_id}&hora=1"
    )
    try:
        r = await client.get(
            url,
            headers={"User-Agent": "polyarb-metar/0.2 (paper-trading)"},
            timeout=10.0,
        )
    except httpx.HTTPError as e:
        log.warning("ogimet_failed", err=str(e))
        return None
    if r.status_code != 200:
        return None
    for line in r.text.splitlines():
        line = line.strip()
        if not line.startswith(station_id):
            continue
        rep = parse_metar(line)
        if rep is not None and rep.station_id == station_id:
            return MetarReport(
                station_id=rep.station_id,
                observation_ts=rep.observation_ts,
                temperature_c=rep.temperature_c,
                raw=rep.raw,
                source="ogimet",
            )
    return None


async def fetch_metar(
    client: httpx.AsyncClient,
    station_id: str,
    *,
    state: _PollState | None = None,
    use_cycle: bool = True,
    use_ogimet_fallback: bool = False,
) -> MetarReport | None:
    """Return the freshest METAR we can find. None means no new data."""
    rep = await fetch_tgftp_station(client, station_id, state=state)
    if rep is not None:
        return rep
    if use_cycle:
        rep = await fetch_tgftp_cycle(client, station_id)
        if rep is not None:
            return rep
    if use_ogimet_fallback:
        rep = await fetch_ogimet(client, station_id)
        if rep is not None:
            return rep
    return None


def in_publish_window(now: datetime | None = None) -> bool:
    """METAR publish windows for HH:00 and HH:30 cycles, with slack on both
    sides for the upstream-to-NOAA propagation lag.
    """
    now = now or datetime.now(timezone.utc)
    m = now.minute
    return (25 <= m <= 40) or m >= 55 or m <= 10


def next_poll_delay(
    *,
    in_window: bool,
    fast_period_s: int,
    slow_period_s: int,
) -> int:
    return fast_period_s if in_window else slow_period_s


@dataclass
class DailyMaxTracker:
    """Per-event running daily-max in whole °C (matches resolver precision)."""

    target_date: date
    samples: list[tuple[datetime, float]] = field(default_factory=list)
    _max_c: float = float("-inf")

    def update(self, report: MetarReport) -> bool:
        if report.observation_ts.date() != self.target_date:
            return False
        self.samples.append((report.observation_ts, report.temperature_c))
        if report.temperature_c > self._max_c:
            self._max_c = report.temperature_c
            return True
        return False

    @property
    def max_c(self) -> float | None:
        return None if self._max_c == float("-inf") else self._max_c

    @property
    def rounded_max_c(self) -> int | None:
        """Truncate toward zero (the published "Temp" column is whole-degree)."""
        m = self.max_c
        return None if m is None else int(m)

    def to_dict(self) -> dict:
        return {
            "target_date": self.target_date.isoformat(),
            "max_c": self.max_c,
            "rounded_max_c": self.rounded_max_c,
            "samples": [(t.isoformat(), v) for t, v in self.samples[-20:]],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DailyMaxTracker":
        tr = cls(target_date=date.fromisoformat(d["target_date"]))
        for t, v in d.get("samples", []):
            tr.samples.append((datetime.fromisoformat(t), float(v)))
            if float(v) > tr._max_c:
                tr._max_c = float(v)
        return tr


def save_trackers(path: Path, trackers: dict[str, DailyMaxTracker]) -> None:
    import json

    payload = {k: v.to_dict() for k, v in trackers.items()}
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def load_trackers(path: Path) -> dict[str, DailyMaxTracker]:
    import json

    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {k: DailyMaxTracker.from_dict(v) for k, v in raw.items()}


def http_date_now() -> str:
    """RFC 7231 ``Last-Modified``-style date string for testing."""
    return formatdate(timeval=None, localtime=False, usegmt=True)
