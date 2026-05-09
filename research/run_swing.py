"""
Multi-timeframe swing/position strategy comparison on BTC/USD 2017-2019.

Tests on 1h, 4h, daily timeframes. Realistic Bybit costs.
IS: 2017-2018, OOS: 2019.
"""
import time
import numpy as np
import pandas as pd

from backtest import run_backtest, BTConfig
from swing_strategies import SWING_STRATEGIES


def load_1m():
    df = pd.read_parquet("btc_1m_full.parquet")
    return df[df["dt"] >= "2017-01-01"][["dt", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def resample(df, rule):
    s = df.set_index("dt")
    out = s.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()
    return out


def split(df, is_start, is_end, oos_start, oos_end):
    is_df = df[(df["dt"] >= is_start) & (df["dt"] <= is_end)].reset_index(drop=True)
    oos_df = df[(df["dt"] >= oos_start) & (df["dt"] <= oos_end)].reset_index(drop=True)
    return is_df, oos_df


def fmt_row(name: str, m: dict, tf: str = ""):
    pf = m.get("profit_factor", 0)
    return {
        "tf": tf, "strategy": name,
        "trades": m.get("n_trades", 0),
        "tot_ret_%": round(m.get("total_return_pct", 0), 1),
        "cagr_%": round(m.get("cagr_pct", 0), 1),
        "sharpe": round(m.get("sharpe", 0), 2),
        "sortino": round(m.get("sortino", 0), 2),
        "max_dd_%": round(m.get("max_dd_pct", 0), 1),
        "win_%": round(m.get("win_rate_pct", 0), 1),
        "avg_trade_%": round(m.get("avg_trade_pct", 0), 3),
        "pf": round(pf, 2) if np.isfinite(pf) else 999.0,
        "exposure": "",  # filled below
    }


def exposure_pct(sig: np.ndarray) -> float:
    """Fraction of bars not flat."""
    return float((sig != 0).mean() * 100)


def run_pass(df, label, cfg):
    rows = []
    for name, fn in SWING_STRATEGIES.items():
        try:
            sig = fn(df)
            res = run_backtest(df, sig, cfg, name=name)
            row = fmt_row(name, res.metrics, label)
            row["exposure"] = round(exposure_pct(sig), 1)
            rows.append(row)
        except Exception as e:
            print(f"  ! {name} failed: {e}")
    return pd.DataFrame(rows)


def main():
    print("Loading...")
    df = load_1m()
    df1h = resample(df, "1h")
    df4h = resample(df, "4h")
    dfd = resample(df, "1D")
    print(f"1h: {len(df1h):,}   4h: {len(df4h):,}   1D: {len(dfd):,}")

    IS = ("2017-01-01", "2018-12-31")
    OOS = ("2019-01-01", "2019-12-31")

    cfg = BTConfig(leverage=1.0, allow_short=True)

    all_results = {}
    for tf_label, tf_df in [("1h", df1h), ("4h", df4h), ("1D", dfd)]:
        is_df, oos_df = split(tf_df, *IS, *OOS)
        print(f"\n{'='*78}\n{tf_label}: IS={len(is_df)} bars, OOS={len(oos_df)} bars\n{'='*78}")
        is_t = run_pass(is_df, f"{tf_label}_IS", cfg).sort_values("sharpe", ascending=False)
        oos_t = run_pass(oos_df, f"{tf_label}_OOS", cfg).sort_values("sharpe", ascending=False)
        print(f"\n--- {tf_label} IN-SAMPLE 2017-2018 ---")
        print(is_t.to_string(index=False))
        print(f"\n--- {tf_label} OUT-OF-SAMPLE 2019 ---")
        print(oos_t.to_string(index=False))
        all_results[tf_label] = {"IS": is_t, "OOS": oos_t}

    # Combined IS+OOS leaderboard, ranked by OOS Sharpe-after-cost AND positive expectancy
    print(f"\n{'='*78}\nROBUSTNESS: strategies positive on BOTH IS and OOS (any TF)\n{'='*78}")
    survivors = []
    for tf, d in all_results.items():
        is_pos = d["IS"][d["IS"]["tot_ret_%"] > 0]
        oos_pos = d["OOS"][d["OOS"]["tot_ret_%"] > 0]
        common = set(is_pos["strategy"]) & set(oos_pos["strategy"])
        for s in common:
            ir = is_pos[is_pos["strategy"] == s].iloc[0]
            orw = oos_pos[oos_pos["strategy"] == s].iloc[0]
            survivors.append({
                "tf": tf, "strategy": s,
                "IS_ret_%": ir["tot_ret_%"], "IS_sharpe": ir["sharpe"], "IS_dd": ir["max_dd_%"],
                "OOS_ret_%": orw["tot_ret_%"], "OOS_sharpe": orw["sharpe"], "OOS_dd": orw["max_dd_%"],
                "trades_total": ir["trades"] + orw["trades"],
            })
    surv = pd.DataFrame(survivors).sort_values("OOS_sharpe", ascending=False) if survivors else pd.DataFrame()
    print(surv.to_string(index=False) if not surv.empty else "  (none)")

    # Save
    pieces = []
    for tf, d in all_results.items():
        d["IS"]["split"] = "IS"
        d["OOS"]["split"] = "OOS"
        pieces.extend([d["IS"], d["OOS"]])
    pd.concat(pieces).to_csv("swing_results_all.csv", index=False)
    if not surv.empty:
        surv.to_csv("swing_survivors.csv", index=False)
    print("\nSaved swing_results_all.csv and swing_survivors.csv")


if __name__ == "__main__":
    main()
