"""
Prop-firm-optimized strategy wrapper.

Wraps a base signal with hard risk controls that match prop challenge rules:
  1. Per-trade stop-loss in ATR units, sized so a stop = X% of account.
  2. Daily loss circuit breaker: stop trading for the day after losing Y%.
  3. Overall drawdown circuit breaker: stop the challenge after Z%.
  4. Position sizing scales DOWN as drawdown grows (anti-martingale).

This is what actually makes prop strategies pass: not bigger Sharpe, but
hard-coded survival rules.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

from backtest import compute_metrics
from strategies import atr


@dataclass
class PropStrategyConfig:
    fee: float = 0.0001  # maker
    slippage: float = 0.0001
    risk_per_trade_pct: float = 0.005  # 0.5% account risk per trade
    sl_atr_mult: float = 1.5            # stop = 1.5 * ATR(14)
    tp_atr_mult: float = 3.0            # take-profit at 3 * ATR
    daily_stop_pct: float = 0.03        # stop trading after -3% on the day
    overall_stop_pct: float = 0.06      # halt challenge at -6% from start
    max_leverage: float = 10.0          # cap effective leverage even if size says more
    profit_target_pct: float = 0.10     # +10% target -> stop trading


def run_prop_backtest(df: pd.DataFrame, signals: np.ndarray,
                      cfg: PropStrategyConfig) -> dict:
    """Position-sized backtest with prop circuit breakers.

    Returns dict with: equity curve, trades list, metrics, hit_target flag,
    hit_daily flag, hit_overall flag.
    """
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    dt = pd.to_datetime(df["dt"]).to_numpy()
    a = atr(h, l, c, 14)

    n = len(df)
    sig = signals.astype(np.int8, copy=False)

    equity = np.empty(n, dtype=np.float64)
    equity[0] = 1.0
    realized = 1.0
    in_pos = 0
    entry_idx = -1
    entry_px = 0.0
    sl_px = 0.0
    tp_px = 0.0
    pos_notional = 0.0  # notional at entry as fraction of account
    daily_start = 1.0
    cur_day = pd.Timestamp(dt[0]).normalize()
    daily_locked = False
    overall_locked = False
    target_hit = False
    trades = []

    for i in range(1, n):
        # Day boundary: reset daily loss tracker
        ts = pd.Timestamp(dt[i]).normalize()
        if ts != cur_day:
            cur_day = ts
            daily_start = realized
            daily_locked = False

        # Already shut down by prop rules?
        if overall_locked or target_hit:
            equity[i] = equity[i - 1]
            continue

        # ---- Step 1: intrabar SL/TP for open position
        exit_now = False
        exit_px = c[i]
        exit_reason = "hold"
        if in_pos != 0:
            if in_pos > 0:
                if l[i] <= sl_px:
                    exit_now, exit_px, exit_reason = True, sl_px, "sl"
                elif h[i] >= tp_px:
                    exit_now, exit_px, exit_reason = True, tp_px, "tp"
            else:
                if h[i] >= sl_px:
                    exit_now, exit_px, exit_reason = True, sl_px, "sl"
                elif l[i] <= tp_px:
                    exit_now, exit_px, exit_reason = True, tp_px, "tp"

        # signal flip
        s_prev = sig[i - 1]
        flip = (in_pos > 0 and s_prev <= 0) or (in_pos < 0 and s_prev >= 0)
        if in_pos != 0 and not exit_now and flip:
            exit_now, exit_px, exit_reason = True, o[i], "flip"

        if in_pos != 0 and exit_now:
            # PnL = (exit/entry - 1) * direction * pos_notional - 2*fee*pos_notional
            ret_px = (exit_px / entry_px - 1.0) * (1 if in_pos > 0 else -1)
            net = ret_px * pos_notional - (2 * cfg.fee + 2 * cfg.slippage) * pos_notional
            realized = realized * (1 + net)
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "entry_px": entry_px, "exit_px": exit_px,
                "side": "L" if in_pos > 0 else "S",
                "ret_pct": net * 100, "reason": exit_reason,
                "notional_x": pos_notional,
            })
            in_pos = 0

        # ---- Daily loss circuit-breaker check (after each fill, also intrabar)
        cur_eq = realized
        # crude intrabar mark for currently open position
        if in_pos != 0:
            urlz = (c[i] / entry_px - 1.0) * (1 if in_pos > 0 else -1)
            cur_eq = realized * (1 + urlz * pos_notional - cfg.fee * pos_notional)
        equity[i] = cur_eq

        # Daily loss locked?
        if not daily_locked and (cur_eq / daily_start - 1.0) <= -cfg.daily_stop_pct:
            daily_locked = True
            # Force-close any open position
            if in_pos != 0:
                ret_px = (c[i] / entry_px - 1.0) * (1 if in_pos > 0 else -1)
                net = ret_px * pos_notional - (2 * cfg.fee + 2 * cfg.slippage) * pos_notional
                realized = realized * (1 + net)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_px": entry_px, "exit_px": c[i],
                    "side": "L" if in_pos > 0 else "S",
                    "ret_pct": net * 100, "reason": "daily_lock",
                    "notional_x": pos_notional,
                })
                in_pos = 0
            equity[i] = realized

        # Overall stop?
        if not overall_locked and (realized - 1.0) <= -cfg.overall_stop_pct:
            overall_locked = True
            if in_pos != 0:
                ret_px = (c[i] / entry_px - 1.0) * (1 if in_pos > 0 else -1)
                net = ret_px * pos_notional - (2 * cfg.fee + 2 * cfg.slippage) * pos_notional
                realized = realized * (1 + net)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_px": entry_px, "exit_px": c[i],
                    "side": "L" if in_pos > 0 else "S",
                    "ret_pct": net * 100, "reason": "overall_lock",
                    "notional_x": pos_notional,
                })
                in_pos = 0
            equity[i] = realized
            continue

        # Target hit?
        if (realized - 1.0) >= cfg.profit_target_pct:
            target_hit = True
            if in_pos != 0:
                ret_px = (c[i] / entry_px - 1.0) * (1 if in_pos > 0 else -1)
                net = ret_px * pos_notional - (2 * cfg.fee + 2 * cfg.slippage) * pos_notional
                realized = realized * (1 + net)
                trades.append({
                    "entry_idx": entry_idx, "exit_idx": i,
                    "entry_px": entry_px, "exit_px": c[i],
                    "side": "L" if in_pos > 0 else "S",
                    "ret_pct": net * 100, "reason": "target",
                    "notional_x": pos_notional,
                })
                in_pos = 0
            equity[i] = realized
            continue

        # ---- Open new position based on signal (only if not locked, no open pos)
        if in_pos == 0 and not daily_locked and not overall_locked and not target_hit:
            if s_prev != 0 and a[i] > 0:
                # Position size: risk_per_trade / (sl_atr_mult * ATR / price) -> notional multiple
                stop_dist_pct = cfg.sl_atr_mult * a[i] / c[i]
                if stop_dist_pct > 1e-6:
                    notional_x = cfg.risk_per_trade_pct / stop_dist_pct
                    notional_x = min(notional_x, cfg.max_leverage)
                    in_pos = int(s_prev)
                    entry_idx = i
                    entry_px = o[i]
                    pos_notional = notional_x
                    if in_pos > 0:
                        sl_px = entry_px * (1 - cfg.sl_atr_mult * a[i] / entry_px)
                        tp_px = entry_px * (1 + cfg.tp_atr_mult * a[i] / entry_px)
                    else:
                        sl_px = entry_px * (1 + cfg.sl_atr_mult * a[i] / entry_px)
                        tp_px = entry_px * (1 - cfg.tp_atr_mult * a[i] / entry_px)

    # Close any remaining position at last bar
    if in_pos != 0:
        ret_px = (c[-1] / entry_px - 1.0) * (1 if in_pos > 0 else -1)
        net = ret_px * pos_notional - (2 * cfg.fee + 2 * cfg.slippage) * pos_notional
        realized = realized * (1 + net)
        equity[-1] = realized

    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(equity, trades_df, df["dt"].to_numpy())
    return {
        "equity": equity,
        "trades": trades_df,
        "metrics": metrics,
        "target_hit": target_hit,
        "overall_locked": overall_locked,
        "final_pnl_pct": (realized - 1.0) * 100,
    }
