"""Leverage stress on the *winning* swing strategies (not the scalping losers)."""
import numpy as np
import pandas as pd
from backtest import run_backtest, BTConfig
from swing_strategies import (
    sma_cross, supertrend, trend_filter_long_only,
    dual_momentum, weekend_effect, turtle,
)


def load_4h():
    df = pd.read_parquet("btc_1m_full.parquet")
    df = df[df["dt"] >= "2015-01-01"].reset_index(drop=True)
    s = df.set_index("dt")
    return s.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()


def main():
    df = load_4h()
    full = df[(df["dt"] >= "2015-01-01") & (df["dt"] <= "2019-12-31")].reset_index(drop=True)
    oos = df[(df["dt"] >= "2019-01-01") & (df["dt"] <= "2019-12-31")].reset_index(drop=True)

    candidates = [
        ("DualMom_30_90", lambda d: dual_momentum(d, 30, 90)),
        ("Supertrend_10_3_LO", lambda d: supertrend(d, 10, 3.0, long_only=True)),
        ("EMA200_filter", trend_filter_long_only),
        ("Weekend_Effect", weekend_effect),
        ("Turtle_20_10_LO", lambda d: turtle(d, 20, 10, long_only=True)),
        ("BuyAndHold", lambda d: np.ones(len(d), dtype=np.int8)),
    ]

    print("=" * 80)
    print("LEVERAGE STRESS — full 2015-2019 (5 years)")
    print("=" * 80)
    rows = []
    for L in [1, 2, 3, 5, 10]:
        for name, fn in candidates:
            sig = fn(full)
            cfg = BTConfig(leverage=L)
            res = run_backtest(full, sig, cfg)
            m = res.metrics
            rows.append({
                "L": L, "strategy": name,
                "trades": m["n_trades"],
                "ret_%": round(m["total_return_pct"], 0),
                "cagr_%": round(m["cagr_pct"], 1),
                "sharpe": round(m["sharpe"], 2),
                "max_dd_%": round(m["max_dd_pct"], 1),
                "liqs": m["liquidations"],
                "final_equity": round(m["equity_final"], 2),
            })
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    t.to_csv("leverage_winners.csv", index=False)

    # Yearly breakdown at moderate leverage 3x
    print("\n" + "=" * 80)
    print("YEAR-BY-YEAR at LEVERAGE 3x (moderate risk)")
    print("=" * 80)
    rows = []
    for y in range(2015, 2020):
        d = df[(df["dt"] >= f"{y}-01-01") & (df["dt"] <= f"{y}-12-31")].reset_index(drop=True)
        for name, fn in candidates:
            sig = fn(d)
            cfg = BTConfig(leverage=3.0)
            res = run_backtest(d, sig, cfg)
            m = res.metrics
            rows.append({
                "year": y, "strategy": name,
                "ret_%": round(m["total_return_pct"], 1),
                "sharpe": round(m["sharpe"], 2),
                "max_dd_%": round(m["max_dd_pct"], 1),
                "liqs": m["liquidations"],
            })
    t = pd.DataFrame(rows)
    pivot_ret = t.pivot(index="strategy", columns="year", values="ret_%")
    pivot_dd = t.pivot(index="strategy", columns="year", values="max_dd_%")
    print("\nReturns by year (%):")
    print(pivot_ret.to_string())
    print("\nMax drawdown by year (%):")
    print(pivot_dd.to_string())
    t.to_csv("leverage_3x_yearly.csv", index=False)


if __name__ == "__main__":
    main()
