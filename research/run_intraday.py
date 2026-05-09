"""
Honest intraday backtest: 15m / 1h, comparing
  (a) realistic Bybit taker fees
  (b) maker-only execution (negative fee = rebate)
on selective intraday strategies + the swing winners.

Goal: find a strategy that trades multiple times per week ("daily activity"
the user wants) but actually makes money after costs.
"""
import numpy as np
import pandas as pd
from backtest import run_backtest, BTConfig
from intraday_strategies import SELECTIVE_STRATEGIES
from swing_strategies import (
    sma_cross, supertrend, trend_filter_long_only,
    dual_momentum, weekend_effect, turtle, momentum_roc,
)

# Bybit perp maker fee = 0.01% (with VIP tiers can go to -0.005% rebate).
# Use 0.01% maker as conservative; user-config can lower.
MAKER_FEE = 0.0001  # 0.01% per side
TAKER_FEE = 0.00055  # 0.055% per side


def load_full():
    df = pd.read_parquet("btc_1m_full.parquet")
    return df[df["dt"] >= "2015-01-01"][["dt", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def resample(df, rule):
    s = df.set_index("dt")
    return s.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()


def slice_range(df, start, end):
    return df[(df["dt"] >= start) & (df["dt"] <= end)].reset_index(drop=True)


def trades_per_year(m, span_years):
    return round(m["n_trades"] / max(0.01, span_years), 1)


def fmt(name, m, span_years):
    pf = m.get("profit_factor", 0)
    return {
        "strategy": name,
        "trades": m["n_trades"],
        "trades/yr": trades_per_year(m, span_years),
        "ret_%": round(m["total_return_pct"], 1),
        "cagr_%": round(m["cagr_pct"], 1),
        "sharpe": round(m["sharpe"], 2),
        "max_dd_%": round(m["max_dd_pct"], 1),
        "win_%": round(m["win_rate_pct"], 1),
        "avg_trade_%": round(m["avg_trade_pct"], 3),
        "pf": round(pf, 2) if np.isfinite(pf) else 999.0,
    }


def run_pass(df, strategies, fee, slippage, label):
    cfg = BTConfig(fee=fee, slippage=slippage, leverage=1.0)
    rows = []
    span_years = (pd.Timestamp(df["dt"].iloc[-1]) - pd.Timestamp(df["dt"].iloc[0])).total_seconds() / (365.25 * 86400)
    for name, fn in strategies.items():
        try:
            sig = fn(df)
            res = run_backtest(df, sig, cfg, name=name)
            rows.append(fmt(name, res.metrics, span_years))
        except Exception as e:
            print(f"  ! {name}: {e}")
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False)


def main():
    print("Loading data...")
    df = load_full()
    df15 = resample(df, "15min")
    df1h = resample(df, "1h")
    print(f"15m: {len(df15):,}  1h: {len(df1h):,}")

    swing_subset = {
        "sw_DualMom_30_90": lambda d: dual_momentum(d, 30, 90),
        "sw_Supertrend_LO": lambda d: supertrend(d, 10, 3.0, long_only=True),
        "sw_EMA200_filter": trend_filter_long_only,
        "sw_Weekend": weekend_effect,
        "sw_Turtle_LO": lambda d: turtle(d, 20, 10, long_only=True),
        "sw_BuyAndHold": lambda d: np.ones(len(d), dtype=np.int8),
    }
    all_strats = {**SELECTIVE_STRATEGIES, **swing_subset}

    IS_RANGE = ("2015-01-01", "2018-12-31")
    OOS_RANGE = ("2019-01-01", "2019-12-31")

    for tf_label, df_tf in [("15m", df15), ("1h", df1h)]:
        is_df = slice_range(df_tf, *IS_RANGE)
        oos_df = slice_range(df_tf, *OOS_RANGE)
        print(f"\n{'#'*80}\n{tf_label} TIMEFRAME  (IS bars={len(is_df):,}  OOS bars={len(oos_df):,})\n{'#'*80}")

        # Pass A: realistic taker fees
        print(f"\n=== {tf_label} | TAKER FEE 0.055% (realistic, what you actually pay) ===")
        is_t = run_pass(is_df, all_strats, fee=TAKER_FEE, slippage=0.0002, label="taker_IS")
        oos_t = run_pass(oos_df, all_strats, fee=TAKER_FEE, slippage=0.0002, label="taker_OOS")
        print("\n-- IS 2015-2018 --")
        print(is_t.head(15).to_string(index=False))
        print("\n-- OOS 2019 --")
        print(oos_t.head(15).to_string(index=False))

        # Pass B: maker-only fees (limit orders only, much lower cost)
        print(f"\n=== {tf_label} | MAKER FEE 0.01% + 1bp slip (limit-order execution) ===")
        is_m = run_pass(is_df, all_strats, fee=MAKER_FEE, slippage=0.0001, label="maker_IS")
        oos_m = run_pass(oos_df, all_strats, fee=MAKER_FEE, slippage=0.0001, label="maker_OOS")
        print("\n-- IS 2015-2018 (maker) --")
        print(is_m.head(15).to_string(index=False))
        print("\n-- OOS 2019 (maker) --")
        print(oos_m.head(15).to_string(index=False))

        # Save tables
        for name, t in [("taker_IS", is_t), ("taker_OOS", oos_t),
                        ("maker_IS", is_m), ("maker_OOS", oos_m)]:
            t.to_csv(f"intraday_{tf_label}_{name}.csv", index=False)

        # Cross-validation: which survive on BOTH IS and OOS, both fee models?
        for fee_label, is_table, oos_table in [
            ("TAKER", is_t, oos_t), ("MAKER", is_m, oos_m)
        ]:
            ispos = is_table[is_table["ret_%"] > 0]["strategy"].tolist()
            oospos = oos_table[oos_table["ret_%"] > 0]["strategy"].tolist()
            common = set(ispos) & set(oospos)
            print(f"\n  {tf_label} | {fee_label} | survived IS+OOS positive: {sorted(common)}")


if __name__ == "__main__":
    main()
