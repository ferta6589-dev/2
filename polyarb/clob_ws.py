from __future__ import annotations

import asyncio
import json
import time
from typing import AsyncIterator

import structlog
import websockets

log = structlog.get_logger(__name__)

PING_INTERVAL = 20
RECONNECT_DELAYS = (1, 2, 4, 8, 16, 30)


async def stream_market(
    ws_host: str, asset_ids: list[str], stop: asyncio.Event
) -> AsyncIterator[dict]:
    """Yield messages from Polymarket CLOB market channel.

    Reconnects with backoff on failure. The caller is responsible for
    interpreting messages (`book` snapshots, `price_change` diffs).
    """
    attempt = 0
    while not stop.is_set():
        try:
            async with websockets.connect(ws_host, ping_interval=PING_INTERVAL) as ws:
                await ws.send(json.dumps({"type": "market", "assets_ids": asset_ids}))
                attempt = 0
                log.info("ws_connected", assets=asset_ids)
                async for raw in ws:
                    if stop.is_set():
                        break
                    for event in _parse(raw):
                        yield event
        except (websockets.WebSocketException, OSError) as e:
            delay = RECONNECT_DELAYS[min(attempt, len(RECONNECT_DELAYS) - 1)]
            attempt += 1
            log.warning("ws_disconnect", error=str(e), retry_in_s=delay)
            try:
                await asyncio.wait_for(stop.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass


def _parse(raw):
    """Polymarket WS sends both single objects and JSON arrays of events."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return
    if isinstance(data, list):
        yield from data
    else:
        yield data


def now_ms() -> int:
    return int(time.time() * 1000)
