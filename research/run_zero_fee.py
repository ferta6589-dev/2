"""Sanity check: do the strategies have ANY edge at all, ignoring fees?
If even with zero fees they don't beat buy-and-hold, the signals are pure noise.
"""
import numpy as np
import pandas as pd
from backtest import run_backtest, BTConfig
from strategies import STRATEGIES


def load_5m():
    df = pd.read_parquet("btc_1m_full.parquet")
    df = df[df["dt"] >= "2017-01-01"].reset_index(drop=True)
    s = df.set_index("dt")
    out = s.resample("5min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return out[(out["dt"] >= "2017-01-01") & (out["dt"] <= "2019-12-31")].reset_index(drop=True)


def main():
    df5 = load_5m()
    print(f"5m bars 2017-2019: {len(df5):,}")

    print("\n=== ZERO FEE / ZERO SLIPPAGE (theoretical edge only) ===")
    cfg = BTConfig(fee=0.0, slippage=0.0, leverage=1.0, allow_short=True)
    rows = []
    for name, fn in STRATEGIES.items():
        sig = fn(df5)
        res = run_backtest(df5, sig, cfg, name=name)
        m = res.metrics
        rows.append({
            "strategy": name,
            "trades": m["n_trades"],
            "tot_ret_%": round(m["total_return_pct"], 1),
            "cagr_%": round(m["cagr_pct"], 1),
            "sharpe": round(m["sharpe"], 2),
            "win_%": round(m["win_rate_pct"], 1),
            "avg_trade_%": round(m["avg_trade_pct"], 4),
            "pf": round(m["profit_factor"], 2) if np.isfinite(m["profit_factor"]) else 999,
            "max_dd_%": round(m["max_dd_pct"], 1),
        })
    t = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print(t.to_string(index=False))
    t.to_csv("results_zero_fee_5m.csv", index=False)

    print("\n=== REALISTIC (Bybit taker 0.055% + 2bp slip each side) ===")
    cfg2 = BTConfig(leverage=1.0, allow_short=True)
    rows2 = []
    for name, fn in STRATEGIES.items():
        sig = fn(df5)
        res = run_backtest(df5, sig, cfg2, name=name)
        m = res.metrics
        rows2.append({
            "strategy": name,
            "trades": m["n_trades"],
            "tot_ret_%": round(m["total_return_pct"], 1),
            "sharpe": round(m["sharpe"], 2),
            "avg_trade_%": round(m["avg_trade_pct"], 4),
            "fee_bleed_%": round(m["n_trades"] * (2 * 0.00055 + 2 * 0.0002) * 100, 1),
        })
    print(pd.DataFrame(rows2).sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
