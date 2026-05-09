"""
Evaluate prop-firm pass-rate on candidate strategies.

For each (strategy, leverage, prop rule set) combination, slide a 30-day
window through 2015-2019 BTC data and count pass / fail-reason.

This is the right metric for prop challenges, NOT Sharpe.
"""
import numpy as np
import pandas as pd
from backtest import BTConfig
from prop_challenge import (
    PROP_RULES, evaluate_strategy_on_prop, summarize, simulate_challenge,
)
from swing_strategies import (
    sma_cross, supertrend, trend_filter_long_only,
    dual_momentum, weekend_effect, turtle, momentum_roc,
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


def main():
    df4h = load_resample("4h")
    df1h = load_resample("1h")
    df15 = load_resample("15min")
    print(f"4h: {len(df4h):,}  1h: {len(df1h):,}  15m: {len(df15):,}")

    # Strategy catalog: (label, df, signal_fn)
    strategies = [
        ("4h_DualMom_30_90", df4h, lambda d: dual_momentum(d, 30, 90)),
        ("4h_Supertrend_LO", df4h, lambda d: supertrend(d, 10, 3.0, long_only=True)),
        ("4h_EMA200_filter", df4h, trend_filter_long_only),
        ("4h_Weekend", df4h, weekend_effect),
        ("4h_Turtle_LO", df4h, lambda d: turtle(d, 20, 10, long_only=True)),
        ("1h_OR_Breakout_LO", df1h, lambda d: opening_range_breakout(d, 4, 16, True)),
        ("1h_OR_Breakout_LS", df1h, lambda d: opening_range_breakout(d, 4, 16, False)),
        ("1h_VolFilt_Donchian", df1h, lambda d: vol_filtered_donchian(d, 20, 14, 200, 0.6, False)),
        ("1h_VolFilt_Donchian_LO", df1h, lambda d: vol_filtered_donchian(d, 20, 14, 200, 0.6, True)),
        ("1h_Session_US", df1h, lambda d: session_momentum(d, 20, 50, 200, ((13, 21),))),
        ("1h_Weekend", df1h, weekend_effect),
        ("1h_BuyAndHold", df1h, lambda d: np.ones(len(d), dtype=np.int8)),
        ("4h_BuyAndHold", df4h, lambda d: np.ones(len(d), dtype=np.int8)),
    ]

    cfg_taker = BTConfig(fee=0.00055, slippage=0.0002, allow_short=True)

    print("=" * 90)
    print("PROP CHALLENGE PASS RATE — TAKER FEES (slide 30d window through 2015-2019)")
    print("=" * 90)

    summaries = []
    for rule_key, rules in PROP_RULES.items():
        print(f"\n{'─'*90}")
        print(f"RULE SET: {rules.name}  "
              f"(target +{rules.profit_target*100:.0f}%, daily -{rules.max_daily_loss*100:.0f}%, "
              f"total -{rules.max_total_dd*100:.0f}%, "
              f"days={rules.days_limit if rules.days_limit else 'unlimited'})")
        print(f"{'─'*90}")
        for L in [1, 2, 3, 5]:
            print(f"\nLeverage {L}x:")
            rule_rows = []
            for label, df_tf, fn in strategies:
                try:
                    res = evaluate_strategy_on_prop(df_tf, fn, rules, cfg_taker, leverage=L)
                    s = summarize(res, label, rules)
                    if s:
                        s["leverage"] = L
                        rule_rows.append(s)
                except Exception as e:
                    print(f"  ! {label}: {e}")
            t = pd.DataFrame(rule_rows).sort_values("pass_%", ascending=False)
            print(t[["strategy", "leverage", "windows", "pass_%",
                     "fail_daily_%", "fail_total_%", "fail_target_%",
                     "median_days_to_pass", "median_max_dd_%"]].to_string(index=False))
            summaries.extend(rule_rows)

    full = pd.DataFrame(summaries)
    full.to_csv("prop_results_all.csv", index=False)

    # Best of best
    print("\n" + "=" * 90)
    print("TOP 10 STRATEGIES BY PASS RATE (all rules + leverage combos)")
    print("=" * 90)
    print(full.sort_values("pass_%", ascending=False).head(20).to_string(index=False))

    # Per-rule best strategy
    print("\n" + "=" * 90)
    print("BEST STRATEGY PER RULE SET")
    print("=" * 90)
    for rule_key, rules in PROP_RULES.items():
        sub = full[full["rule_set"] == rules.name].sort_values("pass_%", ascending=False)
        if not sub.empty:
            top = sub.head(3)
            print(f"\n{rules.name}:")
            print(top.to_string(index=False))


if __name__ == "__main__":
    main()
