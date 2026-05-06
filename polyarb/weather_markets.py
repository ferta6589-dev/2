"""Polymarket Gamma discovery for daily city-temperature events.

Polymarket groups daily weather forecasts as a single Gamma event with several
binary markets, one per temperature bucket (e.g. "Highest temperature in NYC on
May 7 between 65 and 70 degrees"). We parse the title/slug of each child market
into ``(lo_f, hi_f)`` bucket bounds, drop events whose buckets don't form a
contiguous lattice, and return :class:`WeatherEvent`s ready for the strategy
loop.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import httpx
import structlog

log = structlog.get_logger("polyarb.weather_markets")


@dataclass(frozen=True)
class WeatherBucket:
    slug: str
    title: str
    token_yes: str
    token_no: str
    lo_f: float | None
    hi_f: float | None
    minimum_order_size: float = 5.0
    minimum_tick_size: float = 0.01

    def contains(self, temp_f: float) -> bool:
        if self.lo_f is not None and temp_f < self.lo_f:
            return False
        if self.hi_f is not None and temp_f > self.hi_f:
            return False
        return True


@dataclass
class WeatherEvent:
    event_slug: str
    city: str
    target_date: date
    station_id: str
    lat: float
    lon: float
    end_ts: float
    buckets: list[WeatherBucket] = field(default_factory=list)

    def time_to_resolution(self, now: float | None = None) -> float:
        return self.end_ts - (now if now is not None else time.time())


CITY_RESOLVERS: dict[str, tuple[str, float, float]] = {
    "NYC": ("KNYC", 40.78, -73.97),
    "NEW YORK": ("KNYC", 40.78, -73.97),
    "LA": ("KCQT", 34.05, -118.24),
    "LOS ANGELES": ("KCQT", 34.05, -118.24),
    "CHICAGO": ("KORD", 41.98, -87.90),
    "MIAMI": ("KMIA", 25.79, -80.32),
    "AUSTIN": ("KAUS", 30.18, -97.68),
    "PHILADELPHIA": ("KPHL", 39.87, -75.24),
    "PHILLY": ("KPHL", 39.87, -75.24),
    "DENVER": ("KDEN", 39.85, -104.66),
    "BOSTON": ("KBOS", 42.36, -71.01),
    "ATLANTA": ("KATL", 33.64, -84.43),
    "SEATTLE": ("KSEA", 47.45, -122.30),
    "HOUSTON": ("KIAH", 29.99, -95.36),
    "DALLAS": ("KDFW", 32.90, -97.04),
    "PHOENIX": ("KPHX", 33.43, -112.01),
}


def _c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


_RANGE_RE = re.compile(
    r"(?P<lo>-?\d+(?:\.\d+)?)\s*(?:°[fcFC]?\s*)?(?:-|–|to|and)\s*(?P<hi>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[fcFC]?)",
    re.IGNORECASE,
)
_BELOW_RE = re.compile(
    r"(?:<|less\s*than|below|under|≤|<=)\s*(?P<hi>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[fcFC]?)",
    re.IGNORECASE,
)
_ABOVE_RE = re.compile(
    r"(?:>|greater\s*than|more\s*than|above|over|≥|>=|at\s+least)\s*(?P<lo>-?\d+(?:\.\d+)?)\s*°?\s*(?P<unit>[fcFC]?)",
    re.IGNORECASE,
)


def parse_bucket_bounds(text: str) -> tuple[float | None, float | None] | None:
    """Parse a temperature bucket title into ``(lo_f, hi_f)``.

    Returns ``None`` when no bound can be extracted. An open-ended bucket is
    represented as ``(None, hi)`` (upper-capped) or ``(lo, None)``.
    Celsius is converted to Fahrenheit when the unit is explicit.
    """
    if not text:
        return None
    s = text.strip()
    is_celsius = bool(re.search(r"°\s*C\b|\bcelsius\b", s, re.IGNORECASE))

    m = _BELOW_RE.search(s)
    if m:
        hi = float(m.group("hi"))
        unit = m.group("unit") or ("C" if is_celsius else "F")
        if unit.lower() == "c":
            hi = _c_to_f(hi)
        return (None, hi)

    m = _ABOVE_RE.search(s)
    if m:
        lo = float(m.group("lo"))
        unit = m.group("unit") or ("C" if is_celsius else "F")
        if unit.lower() == "c":
            lo = _c_to_f(lo)
        return (lo, None)

    m = _RANGE_RE.search(s)
    if m:
        lo = float(m.group("lo"))
        hi = float(m.group("hi"))
        unit = m.group("unit") or ("C" if is_celsius else "F")
        if unit.lower() == "c":
            lo, hi = _c_to_f(lo), _c_to_f(hi)
        if lo > hi:
            lo, hi = hi, lo
        return (lo, hi)

    return None


def _parse_token_ids(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str):
        try:
            return [str(t) for t in json.loads(raw)]
        except json.JSONDecodeError:
            return []
    return []


def _parse_outcomes(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(o) for o in raw]
    if isinstance(raw, str):
        try:
            return [str(o) for o in json.loads(raw)]
        except json.JSONDecodeError:
            return []
    return []


def _yes_no_tokens(market: dict) -> tuple[str, str] | None:
    tokens = _parse_token_ids(market.get("clobTokenIds") or market.get("clob_token_ids"))
    outcomes = _parse_outcomes(market.get("outcomes"))
    if len(tokens) != 2:
        return None
    yes_idx = 0
    if outcomes:
        for i, o in enumerate(outcomes):
            if o.strip().lower() in ("yes", "y", "true"):
                yes_idx = i
                break
    return tokens[yes_idx], tokens[1 - yes_idx]


def _iso_to_ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


_CITY_RE = re.compile(r"\b(?:in|at|for)\s+([A-Z][A-Za-z .'-]+?)(?:\s+on|\s+at|,|\?|$)")
_DATE_RE = re.compile(
    r"\b(?:on\s+)?(?P<month>jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\s+(?P<day>\d{1,2})(?:[, ]+(?P<year>\d{4}))?",
    re.IGNORECASE,
)
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _infer_city_and_date(title: str) -> tuple[str | None, date | None]:
    city = None
    target_date = None
    m = _CITY_RE.search(title)
    if m:
        city = m.group(1).strip().rstrip("?.").upper()
    m = _DATE_RE.search(title)
    if m:
        month = _MONTHS[m.group("month").lower()[:4].rstrip()]
        day = int(m.group("day"))
        year = int(m.group("year")) if m.group("year") else datetime.now(timezone.utc).year
        try:
            target_date = date(year, month, day)
        except ValueError:
            target_date = None
    return city, target_date


def _resolver_for_city(city_name: str | None) -> tuple[str, str, float, float] | None:
    if not city_name:
        return None
    key = city_name.upper().strip()
    if key in CITY_RESOLVERS:
        sid, lat, lon = CITY_RESOLVERS[key]
        return key, sid, lat, lon
    for variant in (key.replace("CITY", "").strip(), key.split()[0] if key.split() else ""):
        if variant and variant in CITY_RESOLVERS:
            sid, lat, lon = CITY_RESOLVERS[variant]
            return key, sid, lat, lon
    return None


def _build_event(raw_event: dict, allowed_cities: list[str] | None) -> WeatherEvent | None:
    title = (raw_event.get("title") or raw_event.get("question") or raw_event.get("slug") or "")
    event_slug = raw_event.get("slug") or raw_event.get("id") or title

    city, target_date = _infer_city_and_date(title)
    if city is None or target_date is None:
        log.debug("weather_event_unparseable", slug=event_slug, title=title)
        return None
    if allowed_cities and city.upper() not in {c.upper() for c in allowed_cities}:
        return None

    resolver = _resolver_for_city(city)
    if resolver is None:
        log.info("weather_event_no_resolver", slug=event_slug, city=city)
        return None
    city_key, station_id, lat, lon = resolver

    end_iso = (
        raw_event.get("endDate")
        or raw_event.get("end_date_iso")
        or raw_event.get("endDateIso")
    )
    end_ts = _iso_to_ts(end_iso) if end_iso else (
        datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc).timestamp() + 86400
    )

    buckets: list[WeatherBucket] = []
    for m in raw_event.get("markets", []) or []:
        if m.get("closed") or m.get("archived"):
            continue
        title_or_slug = m.get("question") or m.get("title") or m.get("slug") or ""
        bounds = parse_bucket_bounds(title_or_slug)
        if bounds is None:
            log.debug("weather_bucket_unparseable", slug=m.get("slug"), title=title_or_slug)
            continue
        tokens = _yes_no_tokens(m)
        if tokens is None:
            continue
        lo_f, hi_f = bounds
        buckets.append(WeatherBucket(
            slug=str(m.get("slug") or m.get("id") or title_or_slug),
            title=title_or_slug,
            token_yes=tokens[0],
            token_no=tokens[1],
            lo_f=lo_f,
            hi_f=hi_f,
            minimum_order_size=float(m.get("orderMinSize") or m.get("minimum_order_size") or 5.0),
            minimum_tick_size=float(m.get("orderPriceMinTickSize") or m.get("minimum_tick_size") or 0.01),
        ))

    if len(buckets) < 3:
        log.info("weather_event_too_few_buckets", slug=event_slug, n=len(buckets))
        return None
    if not _is_contiguous(buckets):
        log.info("weather_event_non_contiguous", slug=event_slug)
        return None

    return WeatherEvent(
        event_slug=str(event_slug),
        city=city_key,
        target_date=target_date,
        station_id=station_id,
        lat=lat,
        lon=lon,
        end_ts=end_ts,
        buckets=sorted(buckets, key=_bucket_sort_key),
    )


def _bucket_sort_key(b: WeatherBucket) -> float:
    if b.lo_f is not None:
        return b.lo_f
    if b.hi_f is not None:
        return b.hi_f - 1000.0
    return 0.0


def _is_contiguous(buckets: list[WeatherBucket]) -> bool:
    """Buckets must tile a temperature range with no gaps. Open-ended buckets
    are allowed at either end and only at either end."""
    sorted_b = sorted(buckets, key=_bucket_sort_key)
    for i, b in enumerate(sorted_b):
        if b.lo_f is None and i != 0:
            return False
        if b.hi_f is None and i != len(sorted_b) - 1:
            return False
    for prev, cur in zip(sorted_b, sorted_b[1:]):
        if prev.hi_f is None or cur.lo_f is None:
            return False
        if abs(prev.hi_f - cur.lo_f) > 0.51:
            return False
    return True


async def fetch_weather_events(
    client: httpx.AsyncClient,
    gamma_host: str,
    cities: list[str] | None,
    target_date: date | None = None,
) -> list[WeatherEvent]:
    """Fetch all open daily-temperature events from Gamma.

    Empty ``cities`` means auto-discover all known resolver cities.
    """
    params = {
        "tag_slug": "weather",
        "closed": "false",
        "active": "true",
        "limit": "200",
    }
    try:
        r = await client.get(f"{gamma_host}/events", params=params, timeout=15.0)
        r.raise_for_status()
    except httpx.HTTPError as e:
        log.warning("gamma_events_failed", err=str(e))
        return []
    raw = r.json()
    events_raw = raw if isinstance(raw, list) else raw.get("events") or []

    allowed = list(cities) if cities else None
    out: list[WeatherEvent] = []
    for ev in events_raw:
        we = _build_event(ev, allowed)
        if we is None:
            continue
        if target_date is not None and we.target_date != target_date:
            continue
        out.append(we)
    return out
