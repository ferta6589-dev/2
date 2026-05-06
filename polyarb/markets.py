from __future__ import annotations

import json
import time
from dataclasses import dataclass

import httpx

WINDOW_S = 300


@dataclass(frozen=True)
class Market:
    slug: str
    asset: str
    yes_token: str
    no_token: str
    end_ts: float
    minimum_order_size: float
    minimum_tick_size: float

    def time_to_settle(self, now: float | None = None) -> float:
        return self.end_ts - (now if now is not None else time.time())


def current_window_ts(now: float | None = None) -> int:
    t = int(now if now is not None else time.time())
    return t - (t % WINDOW_S)


def slug_for(asset: str, window_ts: int) -> str:
    return f"{asset.lower()}-updown-5m-{window_ts}"


def _parse_token_ids(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(t) for t in raw]
    if isinstance(raw, str):
        return [str(t) for t in json.loads(raw)]
    raise ValueError(f"unexpected clobTokenIds shape: {raw!r}")


def _parse_outcomes(raw) -> list[str]:
    if isinstance(raw, list):
        return [str(o) for o in raw]
    if isinstance(raw, str):
        return [str(o) for o in json.loads(raw)]
    return []


async def fetch_market(client: httpx.AsyncClient, gamma_host: str, slug: str, asset: str) -> Market | None:
    """Fetch a single 5m market by slug from Gamma API. Returns None if missing/closed."""
    r = await client.get(f"{gamma_host}/markets", params={"slug": slug}, timeout=10.0)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    m = data[0] if isinstance(data, list) else data
    if m.get("closed") or m.get("archived"):
        return None

    token_ids = _parse_token_ids(m.get("clobTokenIds") or m.get("clob_token_ids"))
    outcomes = _parse_outcomes(m.get("outcomes"))
    if len(token_ids) != 2:
        return None

    yes_idx = 0
    if outcomes:
        for i, o in enumerate(outcomes):
            if o.strip().lower() in ("yes", "up"):
                yes_idx = i
                break
    no_idx = 1 - yes_idx

    end_iso = m.get("endDate") or m.get("end_date_iso") or m.get("endDateIso")
    end_ts = _iso_to_ts(end_iso) if end_iso else current_window_ts() + WINDOW_S

    return Market(
        slug=slug,
        asset=asset,
        yes_token=token_ids[yes_idx],
        no_token=token_ids[no_idx],
        end_ts=end_ts,
        minimum_order_size=float(m.get("orderMinSize") or m.get("minimum_order_size") or 5.0),
        minimum_tick_size=float(m.get("orderPriceMinTickSize") or m.get("minimum_tick_size") or 0.01),
    )


def _iso_to_ts(iso: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


async def discover(client: httpx.AsyncClient, gamma_host: str, assets: list[str]) -> list[Market]:
    ts = current_window_ts()
    out: list[Market] = []
    for asset in assets:
        try:
            m = await fetch_market(client, gamma_host, slug_for(asset, ts), asset)
        except httpx.HTTPError:
            continue
        if m is not None:
            out.append(m)
    return out
