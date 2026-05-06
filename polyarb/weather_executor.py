"""Execution layer for the weather strategy.

* :class:`WeatherPaperExecutor` writes paper buy/sell/resolve records to a
  daily-rotated JSONL log and persists positions in an append-only ledger so
  restarts replay cleanly without double-trading.
* :class:`WeatherLiveExecutor` is a stub raising ``NotImplementedError`` —
  v2 will wire ``py-clob-client`` plus USDC/CTF allowance and signing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import structlog

from .weather_markets import WeatherBucket
from .weather_strategy import EntryOrder, ExitOrder

log = structlog.get_logger("polyarb.weather_executor")


@dataclass
class Position:
    event_slug: str
    bucket_slug: str
    token_yes: str
    qty: float = 0.0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0
    opened_at: float = 0.0
    closed: bool = False

    def apply_buy(self, qty: float, price: float, ts: float) -> None:
        if qty <= 0:
            return
        new_qty = self.qty + qty
        self.avg_cost = (self.avg_cost * self.qty + price * qty) / new_qty if new_qty > 0 else 0.0
        self.qty = new_qty
        if self.opened_at == 0.0:
            self.opened_at = ts

    def apply_sell(self, qty: float, price: float) -> float:
        if qty <= 0 or self.qty <= 0:
            return 0.0
        qty = min(qty, self.qty)
        pnl = (price - self.avg_cost) * qty
        self.qty -= qty
        self.realized_pnl += pnl
        if self.qty <= 1e-9:
            self.qty = 0.0
            self.closed = True
        return pnl

    def apply_resolve(self, payout_per_share: float = 1.0) -> float:
        if self.qty <= 0:
            return 0.0
        pnl = (payout_per_share - self.avg_cost) * self.qty
        self.realized_pnl += pnl
        self.qty = 0.0
        self.closed = True
        return pnl


@dataclass
class _BuyRecord:
    event_slug: str
    bucket_slug: str
    qty: float
    price: float


@dataclass
class _SellRecord:
    event_slug: str
    bucket_slug: str
    qty: float
    price: float
    reason: str


class WeatherExecutor(Protocol):
    async def fill_buy(self, event_slug: str, order: EntryOrder) -> Position: ...
    async def fill_sell(self, event_slug: str, bucket: WeatherBucket, order: ExitOrder) -> float: ...
    async def record_resolve(self, position: Position, payout_per_share: float = 1.0) -> float: ...


class WeatherPaperExecutor:
    """Paper-only executor. Writes JSONL trade log + position ledger."""

    def __init__(self, log_dir: Path, *, prefix: str = "weather_trades", state=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.state = state
        self.positions: dict[tuple[str, str], Position] = {}
        self._fp = None
        self._fp_date: str | None = None
        self._ledger = (self.log_dir / f"{prefix}_positions.jsonl").open("a", encoding="utf-8")
        self._replay_ledger()

    def _replay_ledger(self) -> None:
        path = self.log_dir / f"{self.prefix}_positions.jsonl"
        if not path.exists():
            return
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = (rec["event_slug"], rec["bucket_slug"])
                pos = self.positions.get(key)
                if pos is None:
                    pos = Position(
                        event_slug=rec["event_slug"],
                        bucket_slug=rec["bucket_slug"],
                        token_yes=rec.get("token_yes", ""),
                    )
                    self.positions[key] = pos
                kind = rec.get("kind")
                qty = float(rec.get("qty", 0.0))
                price = float(rec.get("price", 0.0))
                ts = float(rec.get("ts", 0.0))
                if kind == "buy":
                    pos.apply_buy(qty, price, ts)
                elif kind == "sell":
                    pos.apply_sell(qty, price)
                elif kind == "resolve":
                    pos.apply_resolve(price or 1.0)

    def _file_for_today(self):
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        if date != self._fp_date:
            if self._fp is not None:
                self._fp.close()
            path = self.log_dir / f"{self.prefix}_{date}.jsonl"
            self._fp = path.open("a", encoding="utf-8")
            self._fp_date = date
        return self._fp

    def _write_record(self, rec: dict) -> None:
        fp = self._file_for_today()
        line = json.dumps(rec, separators=(",", ":")) + "\n"
        fp.write(line)
        fp.flush()
        self._ledger.write(line)
        self._ledger.flush()

    async def fill_buy(self, event_slug: str, order: EntryOrder) -> Position:
        ts = time.time()
        key = (event_slug, order.bucket.slug)
        pos = self.positions.get(key) or Position(
            event_slug=event_slug,
            bucket_slug=order.bucket.slug,
            token_yes=order.bucket.token_yes,
        )
        pos.apply_buy(order.qty, order.limit_price, ts)
        self.positions[key] = pos
        rec = {
            "ts": ts,
            "kind": "buy",
            "event_slug": event_slug,
            "bucket_slug": order.bucket.slug,
            "token_yes": order.bucket.token_yes,
            "role": order.role,
            "qty": order.qty,
            "price": order.limit_price,
            "cost_basis": pos.avg_cost,
        }
        self._write_record(rec)
        if self.state is not None:
            self.state.record_action(rec)
        log.info(
            "weather_buy",
            event_slug=event_slug,
            bucket=order.bucket.slug,
            qty=round(order.qty, 2),
            price=order.limit_price,
        )
        return pos

    async def fill_sell(self, event_slug: str, bucket: WeatherBucket, order: ExitOrder) -> float:
        ts = time.time()
        key = (event_slug, bucket.slug)
        pos = self.positions.get(key)
        if pos is None or pos.qty <= 0:
            return 0.0
        realized = pos.apply_sell(order.qty, order.limit_price)
        rec = {
            "ts": ts,
            "kind": "sell",
            "event_slug": event_slug,
            "bucket_slug": bucket.slug,
            "token_yes": bucket.token_yes,
            "qty": order.qty,
            "price": order.limit_price,
            "realized_pnl": realized,
            "reason": order.reason,
        }
        self._write_record(rec)
        if self.state is not None:
            self.state.record_action(rec)
        log.info(
            "weather_sell",
            event_slug=event_slug,
            bucket=bucket.slug,
            qty=round(order.qty, 2),
            price=order.limit_price,
            pnl=round(realized, 4),
            reason=order.reason,
        )
        return realized

    async def record_resolve(self, position: Position, payout_per_share: float = 1.0) -> float:
        if position.qty <= 0:
            return 0.0
        ts = time.time()
        realized = position.apply_resolve(payout_per_share)
        rec = {
            "ts": ts,
            "kind": "resolve",
            "event_slug": position.event_slug,
            "bucket_slug": position.bucket_slug,
            "token_yes": position.token_yes,
            "qty": 0.0,
            "price": payout_per_share,
            "realized_pnl": realized,
        }
        self._write_record(rec)
        if self.state is not None:
            self.state.record_action(rec)
        log.info(
            "weather_resolve",
            event_slug=position.event_slug,
            bucket=position.bucket_slug,
            payout=payout_per_share,
            pnl=round(realized, 4),
        )
        return realized

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
        if self._ledger is not None:
            self._ledger.close()


class WeatherLiveExecutor:
    """Live execution stub. v2 will wire ``py-clob-client``.

    Required env: ``POLYGON_PRIVATE_KEY``, ``POLYGON_PROXY_ADDRESS``. Until
    those are wired, every call raises ``NotImplementedError`` so live mode
    cannot accidentally trade against the real exchange.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "live weather execution not wired; v2: py-clob-client + USDC/CTF allowance + signing"
        )

    async def fill_buy(self, event_slug: str, order: EntryOrder) -> Position:  # pragma: no cover
        raise NotImplementedError

    async def fill_sell(self, event_slug: str, bucket: WeatherBucket, order: ExitOrder) -> float:  # pragma: no cover
        raise NotImplementedError

    async def record_resolve(self, position: Position, payout_per_share: float = 1.0) -> float:  # pragma: no cover
        raise NotImplementedError
