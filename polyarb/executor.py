from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import structlog

from .arb import Opportunity

log = structlog.get_logger(__name__)


class Executor(Protocol):
    def fill(self, opp: Opportunity) -> None: ...


class PaperExecutor:
    """Logs would-be arbitrage trades as JSONL — no on-chain activity."""

    def __init__(self, log_dir: Path, state=None):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._fp = None
        self._fp_date: str | None = None
        self.state = state

    def _file_for_today(self):
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        if date != self._fp_date:
            if self._fp is not None:
                self._fp.close()
            path = self.log_dir / f"paper_trades_{date}.jsonl"
            self._fp = path.open("a", encoding="utf-8")
            self._fp_date = date
        return self._fp

    def fill(self, opp: Opportunity) -> None:
        ts = time.time()
        record = {"ts": ts, **asdict(opp)}
        fp = self._file_for_today()
        fp.write(json.dumps(record, separators=(",", ":")) + "\n")
        fp.flush()
        if self.state is not None:
            self.state.record_opportunity(opp, ts)
        log.info(
            "paper_arb",
            slug=opp.slug,
            edge_bps=round(opp.edge_bps, 1),
            size=opp.size,
            profit_usd=round(opp.profit_usd, 4),
            settle_in_s=round(opp.settle_in_s, 1),
        )

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None
