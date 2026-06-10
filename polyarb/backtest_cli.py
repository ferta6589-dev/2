"""CLI for the 2-bucket backtest.

Examples:

    # No network — Monte-Carlo validation of the gate:
    python -m polyarb.backtest_cli --montecarlo

    # Historical backtest for all cities (needs network / Open-Meteo archive):
    python -m polyarb.backtest_cli --historical --days 120

    # One city:
    python -m polyarb.backtest_cli --historical --city MOSCOW --days 365
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import date, timedelta

from .backtest import (
    CITY_COORDS,
    monte_carlo,
    run_historical,
    sweep_calibration,
    sweep_gates,
)


def _montecarlo() -> int:
    print("=== Monte-Carlo validation (no network) ===\n")
    r = monte_carlo(max_sigma_c=1.0, min_pair_prob=0.92)
    print(f"Robust gate (σ≤1.0, prob≥0.92), calib=1.15:")
    print(f"  coverage   {r.coverage:6.1%}")
    print(f"  hit-rate   {r.hit_rate:6.1%}\n")

    print("Sensitivity to ensemble miscalibration (robust gate):")
    print(f"  {'calib':>6} {'coverage':>9} {'hit-rate':>9}")
    for f in (1.0, 1.1, 1.15, 1.25, 1.4):
        rr = monte_carlo(max_sigma_c=1.0, min_pair_prob=0.92, calibration_factor=f)
        print(f"  {f:>6.2f} {rr.coverage:>8.1%} {rr.hit_rate:>9.1%}")
    print()

    print("Gate sweep (calib=1.15):")
    print(f"  {'σ':>4} {'prob':>5} {'coverage':>9} {'hit-rate':>9}")
    for rr in sweep_gates():
        print(f"  {rr.max_sigma_c:>4.1f} {rr.min_pair_prob:>5.2f} "
              f"{rr.coverage:>8.1%} {rr.hit_rate:>9.1%}")
    return 0


async def _historical(cities: list[str], days: int, sigma: float, prob: float) -> int:
    import httpx

    end = date.today() - timedelta(days=3)  # leave room for archive finalisation
    start = end - timedelta(days=days)
    print(f"=== Historical backtest {start} → {end} ===")
    print(f"gate: σ≤{sigma}, prob≥{prob}\n")

    totals = {"days": 0, "traded": 0, "hit": 0, "pnl": 0.0}
    async with httpx.AsyncClient() as client:
        for city in cities:
            bt = await run_historical(
                client, city, start, end,
                max_sigma_c=sigma, min_pair_prob=prob,
            )
            if bt.forecasts and bt.forecasts[0][0] == "ERROR":
                print(f"  {city:<10} — archive unreachable (network blocked?)")
                continue
            print(f"  {city:<10} days={bt.n_days:>4} traded={bt.n_traded:>4} "
                  f"({bt.coverage:>5.1%})  hit={bt.hit_rate:>6.1%}  "
                  f"pnl=${bt.pnl_usd:>8.2f}")
            totals["days"] += bt.n_days
            totals["traded"] += bt.n_traded
            totals["hit"] += bt.n_hit
            totals["pnl"] += bt.pnl_usd

    if totals["traded"]:
        print(f"\n  {'TOTAL':<10} days={totals['days']:>4} "
              f"traded={totals['traded']:>4} "
              f"({totals['traded']/max(totals['days'],1):>5.1%})  "
              f"hit={totals['hit']/totals['traded']:>6.1%}  "
              f"pnl=${totals['pnl']:>8.2f}")
    else:
        print("\n  No trades executed (network blocked or no qualifying events).")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser("polyarb.backtest")
    p.add_argument("--montecarlo", action="store_true", help="Run MC validation (no network)")
    p.add_argument("--historical", action="store_true", help="Run historical backtest (needs network)")
    p.add_argument("--city", default=None, help="Single city (default: all)")
    p.add_argument("--days", type=int, default=120, help="Lookback window in days")
    p.add_argument("--sigma", type=float, default=1.0, help="Max σ gate (°C)")
    p.add_argument("--prob", type=float, default=0.92, help="Min combined pair probability")
    args = p.parse_args(argv)

    if args.historical:
        cities = [args.city.upper()] if args.city else list(CITY_COORDS.keys())
        return asyncio.run(_historical(cities, args.days, args.sigma, args.prob))
    # default to montecarlo
    return _montecarlo()


if __name__ == "__main__":
    raise SystemExit(main())
