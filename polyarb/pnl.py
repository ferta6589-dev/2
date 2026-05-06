"""Aggregate paper-trade PnL by reading the JSONL logs.

Run as ``python -m polyarb.pnl`` to print a one-shot summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def summarize(log_dir: Path) -> dict:
    files = sorted(log_dir.glob("paper_trades_*.jsonl"))
    n = 0
    total_profit = 0.0
    total_cost = 0.0
    total_size = 0.0
    edge_sum = 0.0
    by_slug: dict[str, int] = {}

    for f in files:
        with f.open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n += 1
                total_profit += float(rec.get("profit_usd", 0))
                total_cost += float(rec.get("cost_usd", 0))
                total_size += float(rec.get("size", 0))
                edge_sum += float(rec.get("edge_bps", 0))
                slug = rec.get("slug", "?")
                by_slug[slug] = by_slug.get(slug, 0) + 1

    avg_edge = edge_sum / n if n else 0.0
    roi_bps = (total_profit / total_cost) * 10_000 if total_cost else 0.0
    return {
        "opportunities": n,
        "unique_markets": len(by_slug),
        "total_size_shares": total_size,
        "total_cost_usd": round(total_cost, 4),
        "total_profit_usd": round(total_profit, 4),
        "avg_edge_bps": round(avg_edge, 2),
        "blended_roi_bps": round(roi_bps, 2),
    }


def main(argv: list[str]) -> int:
    log_dir = Path(argv[1]) if len(argv) > 1 else Path("./logs")
    summary = summarize(log_dir)
    for k, v in summary.items():
        print(f"{k:>20s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
