"""
Compare RAW strategies vs PROP-WRAPPED versions on prop challenges.

The wrapper adds:
  - ATR-based position sizing (each trade risks fixed % of account)
  - Per-trade SL at 1.5*ATR, TP at 3*ATR
  - Daily loss circuit-breaker (-3%) -> stop trading until next day
  - Overall drawdown circuit-breaker (-6%) -> stop the challenge
  - Profit-target lock (+10%) -> stop trading after target hit

For each strategy + rule-set, slide a 30-day window through 2015-2019 and
count pass-rate.
"""
import numpy as np
import pandas as pd
from prop_strategy import run_prop_backtest, PropStrategyConfig
from prop_challenge import PROP_RULES
from swing_strategies import (
    sma_cross, supertrend, trend_filter_long_only,
    dual_momentum, weekend_effect, turtle,
)
from intraday_strategies import (
    opening_range_breakout, vol_filtered_donchian, session_momentum,
)


def load_resample(rule):
    df = pd.read_parquet("btc_1m_full.parquet")
    df = df[df["dt"] >= "2015-01-01"].reset_index(drop=True)
    s = df.set_index("dt")
    return s.resample(rule).agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna().reset_index()


def slide_prop(df, signals, rules, base_cfg, window_days=30, step_days=5):
    """Slide a window through df, run prop_backtest on each window, evaluate pass."""
    dt = pd.to_datetime(df["dt"])
    delta = (dt.iloc[1] - dt.iloc[0]).total_seconds()
    bars_per_day = max(1, int(86400 / delta))
    win_bars = bars_per_day * window_days
    step_bars = bars_per_day * step_days

    results = []
    for start in range(0, len(df) - win_bars, step_bars):
        end = start + win_bars
        sub_df = df.iloc[start:end].reset_index(drop=True)
        sub_sig = signals[start:end]
        cfg = PropStrategyConfig(
            fee=base_cfg["fee"], slippage=base_cfg["slippage"],
            risk_per_trade_pct=base_cfg["risk_per_trade"],
            sl_atr_mult=base_cfg["sl_mult"], tp_atr_mult=base_cfg["tp_mult"],
            daily_stop_pct=rules.max_daily_loss * 0.6,  # conservative: stop at 60% of limit
            overall_stop_pct=rules.max_total_dd * 0.7,  # stop at 70% of limit
            max_leverage=base_cfg["max_lev"],
            profit_target_pct=rules.profit_target,
        )
        res = run_prop_backtest(sub_df, sub_sig, cfg)
        # Evaluate against rules
        eq = res["equity"]
        eq_norm = eq / eq[0]
        # final pnl
        pnl = (eq_norm[-1] - 1.0) * 100
        max_dd = ((eq_norm / np.maximum.accumulate(eq_norm)) - 1).min() * 100
        passed = res["target_hit"]
        # determine fail reason
        if passed:
            reason = ""
        elif res["overall_locked"]:
            reason = "total_dd"
        else:
            reason = "target_not_hit"
        # also check daily loss in eq_norm sequence (intraday simulation)
        results.append({
            "passed": passed, "reason": reason,
            "final_pnl_%": round(pnl, 2),
            "max_dd_%": round(max_dd, 2),
            "n_trades": len(res["trades"]),
        })
    return pd.DataFrame(results)


def main():
    df4h = load_resample("4h")
    df1h = load_resample("1h")

    # Build signals for our top strategies
    sig_catalog = [
        ("4h_DualMom_30_90", df4h, dual_momentum(df4h, 30, 90)),
        ("4h_EMA200_filter", df4h, trend_filter_long_only(df4h)),
        ("4h_Supertrend_LO", df4h, supertrend(df4h, 10, 3.0, long_only=True)),
        ("1h_OR_Breakout_LO", df1h, opening_range_breakout(df1h, 4, 16, True)),
        ("1h_VolFilt_Donchian", df1h, vol_filtered_donchian(df1h, 20, 14, 200, 0.6, False)),
        ("1h_Session_US", df1h, session_momentum(df1h, 20, 50, 200, ((13, 21),))),
    ]

    base_cfgs = [
        {"label": "Conservative_0.5pct_lev3", "fee": 0.0001, "slippage": 0.0001,
         "risk_per_trade": 0.005, "sl_mult": 1.5, "tp_mult": 3.0, "max_lev": 3.0},
        {"label": "Moderate_1pct_lev5",       "fee": 0.0001, "slippage": 0.0001,
         "risk_per_trade": 0.01,  "sl_mult": 1.5, "tp_mult": 3.0, "max_lev": 5.0},
        {"label": "Aggressive_2pct_lev10",    "fee": 0.0001, "slippage": 0.0001,
         "risk_per_trade": 0.02,  "sl_mult": 1.5, "tp_mult": 3.0, "max_lev": 10.0},
    ]

    print("=" * 100)
    print("PROP-WRAPPED RESULTS (with ATR sizing + daily/total circuit-breakers)")
    print("=" * 100)

    summary = []
    for rule_key, rules in PROP_RULES.items():
        print(f"\n{'-'*100}")
        print(f"{rules.name}: target +{rules.profit_target*100:.0f}%, "
              f"daily -{rules.max_daily_loss*100:.0f}%, total -{rules.max_total_dd*100:.0f}%")
        print(f"{'-'*100}")
        for cfg in base_cfgs:
            print(f"\nSizing: {cfg['label']}")
            rows = []
            for label, df_tf, sig in sig_catalog:
                try:
                    res = slide_prop(df_tf, sig, rules, cfg)
                    n = len(res)
                    pas = res["passed"].sum()
                    failt = ((~res["passed"]) & (res["reason"] == "total_dd")).sum()
                    failngt = ((~res["passed"]) & (res["reason"] == "target_not_hit")).sum()
                    med_dd = round(res["max_dd_%"].median(), 2)
                    med_pnl = round(res["final_pnl_%"].median(), 2)
                    avg_trades = round(res["n_trades"].mean(), 1)
                    rows.append({
                        "strategy": label,
                        "windows": n, "pass_%": round(100*pas/n, 1),
                        "fail_dd_%": round(100*failt/n, 1),
                        "fail_target_%": round(100*failngt/n, 1),
                        "median_pnl_%": med_pnl, "median_dd_%": med_dd,
                        "avg_trades": avg_trades,
                    })
                    summary.append({**rows[-1], "rule": rules.name, "sizing": cfg["label"]})
                except Exception as e:
                    print(f"  ! {label}: {e}")
            t = pd.DataFrame(rows).sort_values("pass_%", ascending=False)
            print(t.to_string(index=False))

    # Save
    pd.DataFrame(summary).to_csv("prop_wrapped_results.csv", index=False)

    # Top combos overall
    print("\n" + "=" * 100)
    print("TOP COMBINATIONS BY PASS RATE (across rule sets)")
    print("=" * 100)
    full = pd.DataFrame(summary).sort_values("pass_%", ascending=False)
    print(full.head(25).to_string(index=False))

    # Best per rule set
    print("\n" + "=" * 100)
    print("BEST PROP-WRAPPED COMBO PER RULE SET")
    print("=" * 100)
    for rule_key, rules in PROP_RULES.items():
        sub = full[full["rule"] == rules.name].head(3)
        print(f"\n{rules.name}:")
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
