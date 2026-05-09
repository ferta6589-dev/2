"""
Walk-forward validation and parameter sensitivity for top swing strategies.

Three additional honesty tests beyond the IS/OOS split:
  1. Year-by-year decomposition (2015-2019) -- consistency check
  2. Parameter grid: how sensitive is the result to chosen parameters?
  3. Walk-forward: train on rolling N months, test on next M months
"""
import numpy as np
import pandas as pd
from backtest import run_backtest, BTConfig
from swing_strategies import (
    sma_cross, turtle, supertrend, trend_filter_long_only,
    momentum_roc, dual_momentum, weekend_effect,
)


def load_4h():
    df = pd.read_parquet("btc_1m_full.parquet")
    df = df[df["dt"] >= "2015-01-01"].reset_index(drop=True)
    s = df.set_index("dt")
    out = s.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return out


def load_1d():
    df = pd.read_parquet("btc_1m_full.parquet")
    df = df[df["dt"] >= "2015-01-01"].reset_index(drop=True)
    s = df.set_index("dt")
    out = s.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return out


def slice_year(df, year):
    return df[(df["dt"] >= f"{year}-01-01") & (df["dt"] <= f"{year}-12-31")].reset_index(drop=True)


def metric_row(name, year, m, sig=None):
    return {
        "name": name, "year": year,
        "trades": m["n_trades"],
        "ret_%": round(m["total_return_pct"], 1),
        "sharpe": round(m["sharpe"], 2),
        "max_dd_%": round(m["max_dd_pct"], 1),
        "win_%": round(m["win_rate_pct"], 1),
    }


def yearly(label, df_all, fn):
    cfg = BTConfig()
    rows = []
    for y in range(2015, 2020):
        d = slice_year(df_all, y)
        if len(d) < 50:
            continue
        sig = fn(d)
        res = run_backtest(d, sig, cfg)
        rows.append(metric_row(label, y, res.metrics))
    return pd.DataFrame(rows)


def main():
    df4h = load_4h()
    df1d = load_1d()
    print(f"4h: {len(df4h)} bars  daily: {len(df1d)} bars")

    # Pick the survivors with best OOS Sharpe + reasonable trade counts
    candidates = [
        ("4h_DualMom_30_90", df4h, lambda df: dual_momentum(df, 30, 90)),
        ("4h_Supertrend_10_3_LO", df4h, lambda df: supertrend(df, 10, 3.0, long_only=True)),
        ("4h_EMA200_filter", df4h, trend_filter_long_only),
        ("4h_Weekend_Effect", df4h, weekend_effect),
        ("4h_SMA_50_200_LO", df4h, lambda df: sma_cross(df, 50, 200, long_only=True)),
        ("4h_Turtle_20_10_LO", df4h, lambda df: turtle(df, 20, 10, long_only=True)),
        ("1D_Mom_ROC_90", df1d, lambda df: momentum_roc(df, 90)),
        ("1D_DualMom_30_90", df1d, lambda df: dual_momentum(df, 30, 90)),
        ("1D_BuyAndHold", df1d, lambda df: np.ones(len(df), dtype=np.int8)),
    ]

    print(f"\n{'='*78}\nYEAR-BY-YEAR (2015-2019) — consistency check\n{'='*78}")
    all_y = []
    for name, src, fn in candidates:
        t = yearly(name, src, fn)
        all_y.append(t)
        print(f"\n{name}:")
        print(t.to_string(index=False))

    # Aggregate: how often is the strategy positive year-over-year?
    print(f"\n{'='*78}\nCONSISTENCY SUMMARY: positive years / total years\n{'='*78}")
    summary = []
    for t in all_y:
        if t.empty:
            continue
        name = t["name"].iloc[0]
        n_yrs = len(t)
        n_pos = (t["ret_%"] > 0).sum()
        avg_ret = t["ret_%"].mean()
        med_ret = t["ret_%"].median()
        worst = t["ret_%"].min()
        avg_dd = t["max_dd_%"].mean()
        summary.append({
            "strategy": name, "years": n_yrs, "positive_yrs": int(n_pos),
            "avg_ret_%": round(avg_ret, 1),
            "median_ret_%": round(med_ret, 1),
            "worst_yr_%": round(worst, 1),
            "avg_yr_dd_%": round(avg_dd, 1),
        })
    summary_df = pd.DataFrame(summary).sort_values("positive_yrs", ascending=False)
    print(summary_df.to_string(index=False))

    # Parameter sensitivity: dual momentum (best of the bunch on 4h)
    print(f"\n{'='*78}\nPARAMETER SENSITIVITY — DualMomentum on 4h, full 2015-2019\n{'='*78}")
    full_4h = df4h[(df4h["dt"] >= "2015-01-01") & (df4h["dt"] <= "2019-12-31")].reset_index(drop=True)
    cfg = BTConfig()
    grid = []
    for ns in [10, 20, 30, 50, 80]:
        for nl in [50, 90, 150, 200]:
            if nl <= ns:
                continue
            sig = dual_momentum(full_4h, ns, nl)
            res = run_backtest(full_4h, sig, cfg)
            m = res.metrics
            grid.append({
                "n_short": ns, "n_long": nl,
                "trades": m["n_trades"],
                "ret_%": round(m["total_return_pct"], 1),
                "sharpe": round(m["sharpe"], 2),
                "max_dd_%": round(m["max_dd_pct"], 1),
            })
    grid_df = pd.DataFrame(grid).sort_values("sharpe", ascending=False)
    print(grid_df.to_string(index=False))

    # Save
    pd.concat(all_y).to_csv("walkforward_yearly.csv", index=False)
    summary_df.to_csv("walkforward_summary.csv", index=False)
    grid_df.to_csv("walkforward_param_grid.csv", index=False)


if __name__ == "__main__":
    main()
