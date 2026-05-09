"""
Unified ranking of EVERY strategy tested across all prior experiments.

Pulls together: scalping (5m), intraday (15m/1h, taker+maker), swing (1h/4h/daily),
prop challenges, walk-forward consistency.

Produces one leaderboard ranked by multiple criteria:
  - Profitability:  total return, CAGR
  - Risk-adjusted:  Sharpe, Sortino
  - Robustness:     positive on IS+OOS, year-by-year consistency
  - Practical:      survived realistic costs, reasonable trade frequency
  - Prop:           pass rate on challenge rules

Then picks ONE single winner with full reasoning.
"""
import pandas as pd
import numpy as np
from pathlib import Path

D = Path("/home/user/2/research")


def safe_read(path, **kw):
    try:
        return pd.read_csv(D / path, **kw)
    except Exception:
        return pd.DataFrame()


def annotate(df, source, tf, costs, split):
    if df.empty:
        return df
    df = df.copy()
    df["source"] = source
    df["timeframe"] = tf
    df["costs"] = costs
    df["split"] = split
    return df


print("Loading all result CSVs...")
frames = []

# 5m scalping (these have very few survivors)
for fn, split, tf in [
    ("results_is_5m.csv", "IS", "5m"),
    ("results_oos_5m.csv", "OOS", "5m"),
    ("results_full_15m.csv", "full", "15m"),
]:
    t = safe_read(fn)
    frames.append(annotate(t, "scalp", tf, "taker", split))

# 15m and 1h intraday
for tf in ["15m", "1h"]:
    for cost in ["taker", "maker"]:
        for split in ["IS", "OOS"]:
            t = safe_read(f"intraday_{tf}_{cost}_{split}.csv")
            frames.append(annotate(t, "intraday", tf, cost, split))

# Swing all
sw = safe_read("swing_results_all.csv")
if not sw.empty:
    # Original "tf" column has values like "4h_IS"/"4h_OOS"; split out clean timeframe
    sw["timeframe"] = sw["tf"].str.split("_").str[0]
    if "split" not in sw.columns or sw["split"].isna().all():
        sw["split"] = sw["tf"].str.split("_").str[1]
    sw = sw.drop(columns=["tf"])
    sw["source"] = "swing"
    sw["costs"] = "taker"
    frames.append(sw)

all_results = pd.concat([f for f in frames if not f.empty], ignore_index=True, sort=False)
print(f"Combined rows: {len(all_results):,}")

# Standardize column names
rename_map = {
    "tot_ret_%": "ret_pct",
    "ret_%": "ret_pct",
    "cagr_%": "cagr_pct",
    "max_dd_%": "max_dd_pct",
    "win_%": "win_rate_pct",
    "trades": "n_trades",
    "trades/yr": "trades_per_year",
}
for k, v in rename_map.items():
    if k in all_results.columns and v not in all_results.columns:
        all_results = all_results.rename(columns={k: v})

# Drop incomplete rows
need = ["strategy", "ret_pct", "sharpe", "max_dd_pct"]
all_results = all_results.dropna(subset=need)
print(f"After dropna: {len(all_results):,}")

# Make sure dtypes are numeric where expected
for c in ["ret_pct", "cagr_pct", "sharpe", "max_dd_pct", "win_rate_pct", "n_trades", "trades_per_year"]:
    if c in all_results.columns:
        all_results[c] = pd.to_numeric(all_results[c], errors="coerce")

# Strategies that survived (positive return) under realistic costs (taker)
realistic = all_results[
    (all_results["costs"].fillna("taker") == "taker")
    & (all_results["ret_pct"] > 0)
].copy()
print(f"Profitable under realistic costs: {len(realistic):,}")

# ---------- RANKING 1: pure profitability (best total return on any one window) ----------
print("\n" + "=" * 110)
print("RANKING 1 — PURE PROFITABILITY (best total return on a single test window, taker costs)")
print("=" * 110)
top_profit = realistic.sort_values("ret_pct", ascending=False).head(15)
cols_show = ["strategy", "timeframe", "split", "n_trades", "ret_pct", "cagr_pct",
             "sharpe", "max_dd_pct"]
cols_show = [c for c in cols_show if c in top_profit.columns]
print(top_profit[cols_show].to_string(index=False))

# ---------- RANKING 2: risk-adjusted ----------
print("\n" + "=" * 110)
print("RANKING 2 — RISK-ADJUSTED (best Sharpe, taker costs, profitable only)")
print("=" * 110)
top_sharpe = realistic.sort_values("sharpe", ascending=False).head(15)
print(top_sharpe[cols_show].to_string(index=False))

# ---------- RANKING 3: robustness — positive on BOTH IS and OOS ----------
print("\n" + "=" * 110)
print("RANKING 3 — ROBUSTNESS (positive on BOTH IS and OOS, taker, same TF)")
print("=" * 110)
# Build (strategy, timeframe) keys present in both IS and OOS with positive returns
def has_split(df, split_substring, ret_pos=True):
    sub = df[df["split"].str.contains(split_substring, case=False, na=False)]
    if ret_pos:
        sub = sub[sub["ret_pct"] > 0]
    return set(zip(sub["strategy"], sub["timeframe"]))

is_pos = has_split(realistic, "IS")
oos_pos = has_split(realistic, "OOS")
both = is_pos & oos_pos
print(f"Strategies positive on BOTH IS and OOS (taker): {len(both)}")

robust_rows = []
for strat, tf in sorted(both):
    sub = realistic[(realistic["strategy"] == strat) & (realistic["timeframe"] == tf)]
    is_row = sub[sub["split"].str.contains("IS", case=False, na=False)].iloc[0] if not sub[sub["split"].str.contains("IS", case=False, na=False)].empty else None
    oos_row = sub[sub["split"].str.contains("OOS", case=False, na=False)].iloc[0] if not sub[sub["split"].str.contains("OOS", case=False, na=False)].empty else None
    if is_row is None or oos_row is None:
        continue
    robust_rows.append({
        "strategy": strat,
        "tf": tf,
        "IS_ret_%": round(is_row.get("ret_pct", 0), 1),
        "IS_sharpe": round(is_row.get("sharpe", 0), 2),
        "IS_dd_%": round(is_row.get("max_dd_pct", 0), 1),
        "OOS_ret_%": round(oos_row.get("ret_pct", 0), 1),
        "OOS_sharpe": round(oos_row.get("sharpe", 0), 2),
        "OOS_dd_%": round(oos_row.get("max_dd_pct", 0), 1),
        "min_sharpe": round(min(is_row.get("sharpe", 0), oos_row.get("sharpe", 0)), 2),
    })
robust_df = pd.DataFrame(robust_rows).sort_values("min_sharpe", ascending=False)
print(robust_df.head(20).to_string(index=False))

# ---------- RANKING 4: year-by-year consistency ----------
print("\n" + "=" * 110)
print("RANKING 4 — YEAR-BY-YEAR CONSISTENCY (positive years / 5)")
print("=" * 110)
wf = safe_read("walkforward_summary.csv")
if not wf.empty:
    wf = wf.sort_values(["positive_yrs", "avg_ret_%"], ascending=[False, False])
    print(wf.to_string(index=False))

# ---------- RANKING 5: prop challenge pass rate ----------
print("\n" + "=" * 110)
print("RANKING 5 — PROP CHALLENGE PASS RATE (raw signal, all rules x leverages)")
print("=" * 110)
prop = safe_read("prop_results_all.csv")
if not prop.empty:
    print(prop.sort_values("pass_%", ascending=False).head(15).to_string(index=False))

# ---------- COMPOSITE SCORE: combine multiple metrics ----------
print("\n" + "=" * 110)
print("FINAL COMPOSITE RANKING")
print("=" * 110)
print("Score = z(min OOS-IS Sharpe) + z(min OOS-IS return) - z(max(|DD|))")
print("(only candidates that are positive on BOTH IS and OOS, taker costs)")

if not robust_df.empty:
    def z(x):
        x = pd.Series(x)
        sd = x.std(ddof=0)
        return (x - x.mean()) / (sd if sd > 0 else 1.0)
    rd = robust_df.copy()
    rd["min_ret"] = rd[["IS_ret_%", "OOS_ret_%"]].min(axis=1)
    rd["max_abs_dd"] = rd[["IS_dd_%", "OOS_dd_%"]].abs().max(axis=1)
    rd["score"] = z(rd["min_sharpe"]) * 1.5 + z(rd["min_ret"]) * 0.7 - z(rd["max_abs_dd"]) * 0.8
    rd = rd.sort_values("score", ascending=False)
    print(rd.head(15).to_string(index=False))
    rd.to_csv(D / "FINAL_RANKING.csv", index=False)
    print("\nSaved FINAL_RANKING.csv")

    # ---------- Pick the single winner ----------
    print("\n" + "=" * 110)
    print("THE WINNER")
    print("=" * 110)
    winner = rd.iloc[0]
    print(f"\nStrategy: {winner['strategy']}  ({winner['tf']})")
    print(f"  IS:  ret {winner['IS_ret_%']}%   Sharpe {winner['IS_sharpe']}   DD {winner['IS_dd_%']}%")
    print(f"  OOS: ret {winner['OOS_ret_%']}%   Sharpe {winner['OOS_sharpe']}   DD {winner['OOS_dd_%']}%")
    print(f"  Composite score: {winner['score']:.2f}")

    # Cross-check against year-by-year + prop pass rate
    if not wf.empty:
        m = wf[wf["strategy"].str.contains(winner["strategy"].split("_", 1)[-1], case=False, na=False, regex=False)]
        if not m.empty:
            print(f"\n  Year-by-year:")
            print(m.to_string(index=False))
    if not prop.empty:
        # Match by name pattern
        matches = prop[prop["strategy"].str.contains(winner["strategy"].split("_", 1)[-1], case=False, na=False, regex=False)]
        if not matches.empty:
            print(f"\n  Prop pass rates (sorted):")
            print(matches.sort_values("pass_%", ascending=False).head(10).to_string(index=False))
