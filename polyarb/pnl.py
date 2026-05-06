"""Aggregate paper-trade PnL by reading the JSONL logs.

Run as ``python -m polyarb.pnl`` for the crypto arb summary, or
``python -m polyarb.pnl --prefix weather_trades`` for the weather strategy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def summarize(log_dir: Path, prefix: str = "paper_trades") -> dict:
    files = sorted(log_dir.glob(f"{prefix}_2*.jsonl"))
    if prefix == "paper_trades":
        return _summarize_arb(files)
    return _summarize_weather(files)


def _summarize_arb(files: list[Path]) -> dict:
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


def _summarize_weather(files: list[Path]) -> dict:
    buys = 0
    sells = 0
    resolves = 0
    total_buy_cost = 0.0
    total_realized = 0.0
    total_resolved = 0.0
    by_event: dict[str, dict[str, float]] = {}

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
                kind = rec.get("kind")
                event = rec.get("event_slug", "?")
                bucket = rec.get("bucket_slug", "?")
                qty = float(rec.get("qty", 0))
                price = float(rec.get("price", 0))
                pnl = float(rec.get("realized_pnl", 0))
                slot = by_event.setdefault(event, {"realized": 0.0, "cost": 0.0})
                if kind == "buy":
                    buys += 1
                    total_buy_cost += qty * price
                    slot["cost"] += qty * price
                elif kind == "sell":
                    sells += 1
                    total_realized += pnl
                    slot["realized"] += pnl
                elif kind == "resolve":
                    resolves += 1
                    total_resolved += pnl
                    slot["realized"] += pnl

    grand_total = total_realized + total_resolved
    return {
        "buys": buys,
        "sells": sells,
        "resolves": resolves,
        "events_traded": len(by_event),
        "total_buy_cost_usd": round(total_buy_cost, 4),
        "realized_from_sells_usd": round(total_realized, 4),
        "realized_from_resolves_usd": round(total_resolved, 4),
        "total_realized_usd": round(grand_total, 4),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser("polyarb.pnl")
    parser.add_argument("log_dir", nargs="?", default="./logs")
    parser.add_argument("--prefix", default="paper_trades")
    args = parser.parse_args(argv[1:])
    summary = summarize(Path(args.log_dir), prefix=args.prefix)
    for k, v in summary.items():
        print(f"{k:>28s}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
