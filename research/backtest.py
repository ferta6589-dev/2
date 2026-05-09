"""
Honest vectorized backtest harness for BTC/USD scalping strategies.

Realism principles enforced:
- Signals computed at bar close are executed at NEXT bar's open (no look-ahead).
- Per-trade cost = taker fee (round-trip) + slippage (one tick worth on each side).
- Single-position mode (one signal at a time, can be long/short/flat).
- Stop-loss / take-profit checked intra-bar using high/low; conservative tie-break:
  if both stop and target are hit in the same bar, assume STOP fires (worst case).
- Leverage applies as a position-size multiplier on returns; liquidation is
  modeled as: if drawdown of OPEN trade exceeds 1/leverage minus maintenance
  margin buffer, position is force-closed at the loss threshold (worst-case).

Equity bookkeeping:
- Realized equity is updated only when a trade closes.
- equity[i] = realized_equity * (1 + unrealized_pnl_pct) for display.
- prev realized state, NOT MTM, is what gets compounded by next trade — this
  prevents the geometric explosion bug from reinvesting MTM on top of MTM.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

# Bybit USDT perp taker fee = 0.055%, maker = 0.02%. Use taker for honesty.
TAKER_FEE = 0.00055
SLIPPAGE = 0.0002  # 2 bps each side -- conservative for liquid hours
MAINT_MARGIN = 0.005  # 0.5% maintenance margin buffer before liquidation


@dataclass
class BTConfig:
    fee: float = TAKER_FEE
    slippage: float = SLIPPAGE
    leverage: float = 1.0
    sl_pct: float = 0.0  # 0 = disabled
    tp_pct: float = 0.0
    timeout_bars: int = 0
    allow_short: bool = True


@dataclass
class BTResult:
    name: str
    metrics: dict = field(default_factory=dict)
    equity: np.ndarray | None = None
    trades: pd.DataFrame | None = None


def _per_trade_cost(cfg: BTConfig) -> float:
    """Round-trip cost per unit notional (entry + exit fees + slippage both sides)."""
    return 2 * cfg.fee + 2 * cfg.slippage


def run_backtest(df: pd.DataFrame, signals: np.ndarray, cfg: BTConfig, name: str = "") -> BTResult:
    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    n = len(df)
    sig = signals.astype(np.int8, copy=False)

    cost = _per_trade_cost(cfg)
    L = cfg.leverage
    liq_pct = max(0.0, 1.0 / L - MAINT_MARGIN) if L > 1 else 1.0

    in_pos = 0
    entry_idx = -1
    entry_px = 0.0
    bars_in_pos = 0

    realized = 1.0  # realized equity (compounds only on closed trades)
    equity = np.empty(n, dtype=np.float64)
    equity[0] = 1.0
    trades = []

    for i in range(1, n):
        # ---- Step 1: evaluate intrabar SL/TP/liq/timeout (uses bar i high/low)
        exit_now = False
        exit_px = c[i]
        exit_reason = "hold"
        if in_pos != 0:
            bars_in_pos += 1
            if in_pos > 0:
                sl_px = entry_px * (1 - cfg.sl_pct) if cfg.sl_pct > 0 else -np.inf
                tp_px = entry_px * (1 + cfg.tp_pct) if cfg.tp_pct > 0 else np.inf
                liq_px = entry_px * (1 - liq_pct) if L > 1 else -np.inf
                if l[i] <= liq_px:
                    exit_now, exit_px, exit_reason = True, liq_px, "liq"
                elif l[i] <= sl_px:
                    exit_now, exit_px, exit_reason = True, sl_px, "sl"
                elif h[i] >= tp_px:
                    exit_now, exit_px, exit_reason = True, tp_px, "tp"
            else:
                sl_px = entry_px * (1 + cfg.sl_pct) if cfg.sl_pct > 0 else np.inf
                tp_px = entry_px * (1 - cfg.tp_pct) if cfg.tp_pct > 0 else -np.inf
                liq_px = entry_px * (1 + liq_pct) if L > 1 else np.inf
                if h[i] >= liq_px:
                    exit_now, exit_px, exit_reason = True, liq_px, "liq"
                elif h[i] >= sl_px:
                    exit_now, exit_px, exit_reason = True, sl_px, "sl"
                elif l[i] <= tp_px:
                    exit_now, exit_px, exit_reason = True, tp_px, "tp"
            if not exit_now and cfg.timeout_bars > 0 and bars_in_pos >= cfg.timeout_bars:
                exit_now, exit_px, exit_reason = True, o[i], "timeout"

        # ---- Step 2: signal-driven flip (signal at bar i-1 acts at bar i open)
        s_prev = sig[i - 1]
        signal_flip = (in_pos > 0 and s_prev <= 0) or (in_pos < 0 and s_prev >= 0)
        if in_pos != 0 and not exit_now and signal_flip:
            exit_now, exit_px, exit_reason = True, o[i], "flip"

        # ---- Step 3: realize PnL on exit
        if in_pos != 0 and exit_now:
            ret = (exit_px / entry_px - 1.0) * (1 if in_pos > 0 else -1)
            net_ret = ret * L - cost  # leverage scales pnl, cost paid round-trip
            realized = realized * (1 + net_ret)
            trades.append({
                "entry_idx": entry_idx, "exit_idx": i,
                "entry_px": entry_px, "exit_px": exit_px,
                "side": "L" if in_pos > 0 else "S",
                "bars": bars_in_pos, "ret_pct": net_ret * 100,
                "reason": exit_reason,
            })
            in_pos = 0
            bars_in_pos = 0

        # ---- Step 4: open new position from signal
        if in_pos == 0 and realized > 0:  # don't trade after blowup
            target_side = 0
            if s_prev == 1:
                target_side = 1
            elif s_prev == -1 and cfg.allow_short:
                target_side = -1
            if target_side != 0:
                in_pos = target_side
                entry_idx = i
                entry_px = o[i]
                bars_in_pos = 0

        # ---- Step 5: write display equity (realized * MTM of any open position)
        if in_pos != 0 and entry_px > 0:
            urlz = (c[i] / entry_px - 1.0) * (1 if in_pos > 0 else -1) * L - cost
            equity[i] = max(0.0, realized * (1 + urlz))
        else:
            equity[i] = realized

    # close any open position at last bar
    if in_pos != 0:
        ret = (c[-1] / entry_px - 1.0) * (1 if in_pos > 0 else -1)
        net_ret = ret * L - cost
        realized = realized * (1 + net_ret)
        equity[-1] = max(0.0, realized)
        trades.append({
            "entry_idx": entry_idx, "exit_idx": n - 1,
            "entry_px": entry_px, "exit_px": c[-1],
            "side": "L" if in_pos > 0 else "S",
            "bars": bars_in_pos, "ret_pct": net_ret * 100,
            "reason": "eod",
        })

    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(equity, trades_df, df["dt"].to_numpy())
    return BTResult(name=name, metrics=metrics, equity=equity, trades=trades_df)


def compute_metrics(equity: np.ndarray, trades: pd.DataFrame, dt) -> dict:
    n = len(equity)
    if n < 2:
        return {}
    # use only positive equity to avoid div-zero noise from blown accounts
    eq = np.where(equity <= 0, 1e-12, equity)
    rets = np.diff(eq) / eq[:-1]
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
    rets = np.clip(rets, -0.99, 10.0)
    bars_per_year_5m = 365 * 24 * 12  # if 5m bars
    # Estimate bars/year from dt array
    span_sec = (pd.Timestamp(dt[-1]) - pd.Timestamp(dt[0])).total_seconds()
    bars_per_year = max(1, n / max(1e-9, span_sec) * (365.25 * 86400))
    mu = rets.mean()
    sd = rets.std(ddof=0)
    sharpe = (mu / sd) * np.sqrt(bars_per_year) if sd > 0 else 0.0
    downside = rets[rets < 0]
    dsd = downside.std(ddof=0) if len(downside) > 1 else 0.0
    sortino = (mu / dsd) * np.sqrt(bars_per_year) if dsd > 0 else 0.0
    peak = np.maximum.accumulate(eq)
    dd = eq / peak - 1
    max_dd = dd.min()
    total_return = eq[-1] / eq[0] - 1
    span_years = span_sec / (365.25 * 86400)
    cagr = (eq[-1] / eq[0]) ** (1 / span_years) - 1 if span_years > 0 and eq[-1] > 0 else -1.0

    nt = len(trades)
    if nt > 0:
        wr = (trades["ret_pct"] > 0).mean()
        avg_trade = trades["ret_pct"].mean()
        gross_w = trades.loc[trades["ret_pct"] > 0, "ret_pct"].sum()
        gross_l = -trades.loc[trades["ret_pct"] < 0, "ret_pct"].sum()
        pf = gross_w / gross_l if gross_l > 0 else float("inf")
        wins = trades.loc[trades["ret_pct"] > 0, "ret_pct"]
        losses = trades.loc[trades["ret_pct"] < 0, "ret_pct"]
        avg_w = wins.mean() if len(wins) else 0.0
        avg_l = -losses.mean() if len(losses) else 0.0
        if avg_l > 0:
            R = avg_w / avg_l
            kelly = wr - (1 - wr) / R
        else:
            kelly = 0.0
        liq_count = (trades["reason"] == "liq").sum() if "reason" in trades.columns else 0
    else:
        wr = avg_trade = pf = kelly = 0.0
        liq_count = 0

    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd_pct": max_dd * 100,
        "win_rate_pct": wr * 100,
        "n_trades": nt,
        "avg_trade_pct": avg_trade,
        "profit_factor": pf,
        "kelly_frac": kelly,
        "equity_final": eq[-1],
        "liquidations": int(liq_count),
        "span_years": round(span_years, 2),
    }
