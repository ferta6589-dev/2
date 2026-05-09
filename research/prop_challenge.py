"""
Prop firm challenge simulator.

Typical crypto prop firms (HyroTrader, FundedNext Crypto, Apex Crypto, BFG):
  - Profit target: +8% to +10% of starting balance
  - Max daily loss: -4% to -5% of starting balance (resets each UTC day)
  - Max overall drawdown: -6% to -10% of starting balance (HWM-based or static)
  - Time limit: 30 days for one-step, no time limit for two-step (sometimes)
  - Some require min trading days (e.g. 5 days with at least 1 trade)
  - Account scaling: pass -> trade firm capital, ~80% profit split

For each candidate strategy + rule set, we slide a window through 2015-2019
historical data and count: pass / fail-DD / fail-target-not-hit / fail-time.

Key insight: this is a probability-of-success problem, not max-Sharpe.
A strategy that has 65% historical win-rate but Sharpe 0.8 may pass MORE OFTEN
than a Sharpe 2 strategy that has occasional -8% days.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from backtest import run_backtest, BTConfig


@dataclass
class PropRules:
    name: str
    profit_target: float       # 0.10 = +10%
    max_daily_loss: float      # 0.05 = -5% from day-start balance (or HWM)
    max_total_dd: float        # 0.10 = -10% from peak/start
    days_limit: int            # 30, 60, or 0 for no limit
    min_trading_days: int = 0  # 0 if no minimum
    daily_loss_basis: str = "start"  # "start" = % of starting balance / "hwm" = % of equity highwater
    overall_dd_basis: str = "start"  # "start" or "hwm"


PROP_RULES = {
    "HyroTrader_25k_1step": PropRules(
        name="HyroTrader 25k 1-step",
        profit_target=0.10, max_daily_loss=0.05, max_total_dd=0.06,
        days_limit=0, min_trading_days=5,
        daily_loss_basis="start", overall_dd_basis="start",
    ),
    "FundedNext_Crypto_2step": PropRules(
        name="FundedNext Crypto 2-step P1",
        profit_target=0.08, max_daily_loss=0.05, max_total_dd=0.10,
        days_limit=0, min_trading_days=0,
        daily_loss_basis="start", overall_dd_basis="start",
    ),
    "Apex_strict": PropRules(
        name="Apex-style strict",
        profit_target=0.08, max_daily_loss=0.03, max_total_dd=0.05,
        days_limit=30, min_trading_days=0,
        daily_loss_basis="start", overall_dd_basis="start",
    ),
    "BFG_lenient": PropRules(
        name="BFG-style lenient",
        profit_target=0.10, max_daily_loss=0.05, max_total_dd=0.10,
        days_limit=30, min_trading_days=5,
        daily_loss_basis="hwm", overall_dd_basis="hwm",
    ),
}


@dataclass
class ChallengeResult:
    pass_: bool
    fail_reason: str  # "" if passed, else "daily_dd" / "total_dd" / "target_not_hit" / "min_days_not_met"
    days_used: int
    final_pnl_pct: float
    max_dd_pct: float
    worst_day_pct: float
    n_trades: int


def simulate_challenge(equity: np.ndarray, dt: np.ndarray, rules: PropRules,
                       n_trades_window: int = 0) -> ChallengeResult:
    """Simulate one prop challenge using an equity curve.

    equity: equity curve normalized so equity[0] = 1.0
    dt: datetimes corresponding to equity samples
    """
    if len(equity) < 2:
        return ChallengeResult(False, "no_data", 0, 0, 0, 0, 0)

    # Bucket by UTC day
    days = pd.to_datetime(dt).normalize()
    unique_days = pd.unique(days)
    days_used = 0
    trading_days = 0

    cur_eq = 1.0
    peak = 1.0
    daily_start = 1.0  # equity at start of current day
    daily_basis = 1.0 if rules.daily_loss_basis == "start" else 1.0
    overall_basis = 1.0 if rules.overall_dd_basis == "start" else 1.0

    worst_day_pct = 0.0  # most-negative single-day drawdown observed
    max_dd_pct = 0.0

    # Find indexes for each day
    day_index = {}
    for i, d in enumerate(days):
        day_index.setdefault(d, []).append(i)

    last_eq_for_day = 1.0
    for d_idx, day in enumerate(unique_days):
        if rules.days_limit > 0 and days_used >= rules.days_limit:
            break
        idxs = day_index[day]
        # equity at the start of the day = equity[idxs[0]]
        daily_start = equity[idxs[0]] if d_idx > 0 else equity[idxs[0]]
        # check intraday
        for i in idxs:
            cur_eq = equity[i]
            peak = max(peak, cur_eq)
            # update bases
            if rules.daily_loss_basis == "hwm":
                daily_basis = peak
            else:
                daily_basis = 1.0  # start of challenge
            if rules.overall_dd_basis == "hwm":
                overall_basis = peak
            else:
                overall_basis = 1.0

            # Daily loss check (current equity vs daily_start basis)
            day_loss_pct = (cur_eq - daily_start) / daily_start
            if rules.daily_loss_basis == "hwm":
                # In HWM mode, daily limit is from intraday peak
                # but most prop firms still anchor at day start; keep simple
                day_loss_pct = (cur_eq - daily_start) / daily_start
            if day_loss_pct < -rules.max_daily_loss:
                return ChallengeResult(
                    pass_=False, fail_reason="daily_dd",
                    days_used=days_used + 1, final_pnl_pct=(cur_eq - 1) * 100,
                    max_dd_pct=max_dd_pct * 100,
                    worst_day_pct=min(worst_day_pct, day_loss_pct) * 100,
                    n_trades=n_trades_window,
                )

            # Overall DD check
            dd = (cur_eq - overall_basis) / overall_basis
            max_dd_pct = min(max_dd_pct, dd)
            if dd < -rules.max_total_dd:
                return ChallengeResult(
                    pass_=False, fail_reason="total_dd",
                    days_used=days_used + 1, final_pnl_pct=(cur_eq - 1) * 100,
                    max_dd_pct=max_dd_pct * 100,
                    worst_day_pct=worst_day_pct * 100,
                    n_trades=n_trades_window,
                )

            # Profit target check (after min trading days)
            if (cur_eq - 1.0) >= rules.profit_target and trading_days >= rules.min_trading_days:
                return ChallengeResult(
                    pass_=True, fail_reason="",
                    days_used=days_used + 1, final_pnl_pct=(cur_eq - 1) * 100,
                    max_dd_pct=max_dd_pct * 100,
                    worst_day_pct=worst_day_pct * 100,
                    n_trades=n_trades_window,
                )

        # End-of-day bookkeeping
        last_eq_for_day = cur_eq
        if (cur_eq != daily_start):
            trading_days += 1
        end_of_day_pct = (last_eq_for_day - daily_start) / daily_start
        worst_day_pct = min(worst_day_pct, end_of_day_pct)
        days_used += 1

    # Reached end of window without passing
    if (cur_eq - 1.0) >= rules.profit_target and trading_days >= rules.min_trading_days:
        return ChallengeResult(True, "", days_used, (cur_eq - 1) * 100,
                               max_dd_pct * 100, worst_day_pct * 100, n_trades_window)
    fail = "target_not_hit"
    if trading_days < rules.min_trading_days and rules.min_trading_days > 0:
        fail = "min_days_not_met"
    return ChallengeResult(False, fail, days_used, (cur_eq - 1) * 100,
                           max_dd_pct * 100, worst_day_pct * 100, n_trades_window)


def evaluate_strategy_on_prop(df: pd.DataFrame, signal_fn, rules: PropRules,
                              cfg: BTConfig, leverage: float = 1.0,
                              window_bars: int = None,
                              step_bars: int = None,
                              risk_per_trade: float = None) -> pd.DataFrame:
    """Slide a challenge-window through the data and run prop sim on each.
    Returns DataFrame with one row per simulated challenge."""
    cfg2 = BTConfig(fee=cfg.fee, slippage=cfg.slippage, leverage=leverage,
                     allow_short=cfg.allow_short)

    # Run full backtest once to get equity curve at strategy's normal sizing
    sig = signal_fn(df)
    full_res = run_backtest(df, sig, cfg2)
    equity = full_res.equity
    dt = df["dt"].to_numpy()

    if window_bars is None:
        # 30 days at the bar timeframe -- estimate from first two timestamps
        delta = (pd.Timestamp(dt[1]) - pd.Timestamp(dt[0])).total_seconds()
        bars_per_day = max(1, int(86400 / delta))
        window_bars = bars_per_day * 30
    if step_bars is None:
        step_bars = window_bars // 6  # slide by 5 days

    results = []
    n = len(equity)
    for start in range(0, n - window_bars, step_bars):
        end = start + window_bars
        eq_window = equity[start:end] / equity[start]  # normalize to start=1.0
        dt_window = dt[start:end]
        # count trades in window
        if full_res.trades is not None and len(full_res.trades) > 0:
            n_tr = ((full_res.trades["entry_idx"] >= start) &
                    (full_res.trades["entry_idx"] < end)).sum()
        else:
            n_tr = 0
        cr = simulate_challenge(eq_window, dt_window, rules, int(n_tr))
        results.append({
            "start": pd.Timestamp(dt_window[0]),
            "passed": cr.pass_,
            "fail_reason": cr.fail_reason,
            "days_used": cr.days_used,
            "final_pnl_%": round(cr.final_pnl_pct, 2),
            "max_dd_%": round(cr.max_dd_pct, 2),
            "worst_day_%": round(cr.worst_day_pct, 2),
            "n_trades": cr.n_trades,
        })
    return pd.DataFrame(results)


def summarize(results: pd.DataFrame, name: str, rules: PropRules) -> dict:
    if results.empty:
        return {}
    total = len(results)
    passed = results["passed"].sum()
    fail_dd = ((~results["passed"]) & (results["fail_reason"] == "daily_dd")).sum()
    fail_total = ((~results["passed"]) & (results["fail_reason"] == "total_dd")).sum()
    fail_tgt = ((~results["passed"]) & (results["fail_reason"] == "target_not_hit")).sum()
    return {
        "strategy": name,
        "rule_set": rules.name,
        "windows": total,
        "pass_%": round(100 * passed / total, 1),
        "fail_daily_%": round(100 * fail_dd / total, 1),
        "fail_total_%": round(100 * fail_total / total, 1),
        "fail_target_%": round(100 * fail_tgt / total, 1),
        "median_days_to_pass": int(results.loc[results["passed"], "days_used"].median()) if passed else 0,
        "median_max_dd_%": round(results["max_dd_%"].median(), 2),
    }
