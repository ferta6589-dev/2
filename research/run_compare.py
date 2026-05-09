"""
Compare all scalping strategies on BTC/USD 1m data.

Train/test split:
  IS (in-sample): 2017-01-01 .. 2018-12-31
  OOS (out-of-sample): 2019-01-01 .. 2019-12-31

We resample to 5m for primary scalping evaluation (1m has too much noise &
fees dominate for slow indicators); also report 1m for the fastest strategies.
"""
import sys
import time
import numpy as np
import pandas as pd

from backtest import run_backtest, BTConfig
from strategies import STRATEGIES


def load(parquet_path="btc_1m_full.parquet"):
    df = pd.read_parquet(parquet_path)
    df = df[["dt", "open", "high", "low", "close", "volume"]].copy()
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    s = df.set_index("dt")
    out = s.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna()
    out = out.reset_index()
    return out


def split(df: pd.DataFrame, is_start, is_end, oos_start, oos_end):
    is_df = df[(df["dt"] >= is_start) & (df["dt"] <= is_end)].reset_index(drop=True)
    oos_df = df[(df["dt"] >= oos_start) & (df["dt"] <= oos_end)].reset_index(drop=True)
    return is_df, oos_df


def evaluate(df: pd.DataFrame, name: str, fn, cfg: BTConfig):
    sig = fn(df)
    res = run_backtest(df, sig, cfg, name=name)
    return res


def fmt_row(name: str, m: dict) -> dict:
    return {
        "strategy": name,
        "trades": m.get("n_trades", 0),
        "tot_ret_%": round(m.get("total_return_pct", 0), 2),
        "cagr_%": round(m.get("cagr_pct", 0), 2),
        "sharpe": round(m.get("sharpe", 0), 2),
        "sortino": round(m.get("sortino", 0), 2),
        "max_dd_%": round(m.get("max_dd_pct", 0), 2),
        "win_%": round(m.get("win_rate_pct", 0), 1),
        "avg_trade_%": round(m.get("avg_trade_pct", 0), 4),
        "pf": round(m.get("profit_factor", 0), 2) if np.isfinite(m.get("profit_factor", 0)) else 999,
        "kelly": round(m.get("kelly_frac", 0), 3),
        "liqs": m.get("liquidations", 0),
    }


def main():
    print("Loading 1m BTC/USD data...")
    df = load()
    print(f"Rows: {len(df):,}  range: {df['dt'].min()} -> {df['dt'].max()}")

    # Filter to dense periods (drop pre-2015 sparse data)
    df = df[df["dt"] >= "2015-01-01"].reset_index(drop=True)
    print(f"After 2015 filter: {len(df):,} rows")

    # Build 5m and 15m bars for slower indicators
    print("Resampling to 5m and 15m...")
    df5 = resample(df, "5min")
    df15 = resample(df, "15min")
    print(f"5m bars: {len(df5):,}    15m bars: {len(df15):,}")

    # Time splits
    IS = ("2017-01-01", "2018-12-31")
    OOS = ("2019-01-01", "2019-12-31")

    is5, oos5 = split(df5, *IS, *OOS)
    is15, oos15 = split(df15, *IS, *OOS)

    print(f"\n5m IS bars: {len(is5):,}   OOS bars: {len(oos5):,}")
    print(f"15m IS bars: {len(is15):,}   OOS bars: {len(oos15):,}")

    base_cfg = BTConfig(leverage=1.0, allow_short=True)

    # ---- Pass 1: 5m timeframe, leverage 1, no SL/TP, in-sample ----
    print("\n" + "=" * 78)
    print("PASS 1 -- 5m, leverage 1x, no SL/TP, IN-SAMPLE 2017-2018")
    print("=" * 78)
    rows = []
    for name, fn in STRATEGIES.items():
        t0 = time.time()
        res = evaluate(is5, name, fn, base_cfg)
        dt = time.time() - t0
        rows.append(fmt_row(name, res.metrics))
        print(f"  {name:18s}  done in {dt:.1f}s  trades={res.metrics['n_trades']}")
    is_table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\n", is_table.to_string(index=False))

    # ---- Pass 2: 5m timeframe, OOS 2019 ----
    print("\n" + "=" * 78)
    print("PASS 2 -- 5m, leverage 1x, no SL/TP, OUT-OF-SAMPLE 2019")
    print("=" * 78)
    rows = []
    for name, fn in STRATEGIES.items():
        res = evaluate(oos5, name, fn, base_cfg)
        rows.append(fmt_row(name, res.metrics))
    oos_table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\n", oos_table.to_string(index=False))

    # ---- Pass 3: 15m timeframe, IS+OOS combined ----
    full_15 = df15[(df15["dt"] >= "2017-01-01") & (df15["dt"] <= "2019-12-31")].reset_index(drop=True)
    print("\n" + "=" * 78)
    print("PASS 3 -- 15m, leverage 1x, no SL/TP, FULL 2017-2019")
    print("=" * 78)
    rows = []
    for name, fn in STRATEGIES.items():
        res = evaluate(full_15, name, fn, base_cfg)
        rows.append(fmt_row(name, res.metrics))
    full_table = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    print("\n", full_table.to_string(index=False))

    # ---- Pass 4: pick top-3 from IS by Sharpe, evaluate them with leverage stress ----
    print("\n" + "=" * 78)
    print("PASS 4 -- LEVERAGE STRESS on top-3 IS picks (5m, OOS 2019)")
    print("=" * 78)
    top3 = is_table.head(3)["strategy"].tolist()
    print(f"Top-3 IS by Sharpe: {top3}")
    lev_rows = []
    for L in [1, 3, 5, 10, 20]:
        for name in top3:
            cfg = BTConfig(leverage=L, allow_short=True)
            res = evaluate(oos5, f"{name}_x{L}", STRATEGIES[name], cfg)
            row = fmt_row(f"{name}_x{L}", res.metrics)
            row["leverage"] = L
            lev_rows.append(row)
    lev_table = pd.DataFrame(lev_rows)
    print("\n", lev_table.to_string(index=False))

    # ---- Save tables ----
    is_table.to_csv("results_is_5m.csv", index=False)
    oos_table.to_csv("results_oos_5m.csv", index=False)
    full_table.to_csv("results_full_15m.csv", index=False)
    lev_table.to_csv("results_leverage_stress.csv", index=False)
    print("\nSaved CSVs: results_is_5m.csv, results_oos_5m.csv, results_full_15m.csv, results_leverage_stress.csv")


if __name__ == "__main__":
    main()
