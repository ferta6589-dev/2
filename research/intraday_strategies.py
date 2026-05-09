"""
Selective intraday strategies: trade frequently (multiple times per week),
but only fire when specific setups are met. Designed for 15m / 1h bars.

Each function returns int8 array {-1, 0, +1} computed at bar close.
Held positions are visible across multiple bars (not just the entry bar).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategies import ema, rsi, atr


# ------------------ ATR breakout with trend filter (selective) ----------------

def atr_breakout(df: pd.DataFrame, atr_n: int = 14, k: float = 1.0,
                 trend_n: int = 100, hold_bars: int = 8, long_only: bool = True) -> np.ndarray:
    """Enter long if close > prior_close + k * ATR AND price > EMA(trend_n);
    enter short on mirror condition. Hold for hold_bars OR until reverse signal."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    a = atr(h, l, c, atr_n)
    e = ema(c, trend_n)
    sig = np.zeros(len(df), dtype=np.int8)
    pos, bars_held = 0, 0
    for i in range(1, len(df)):
        if pos != 0:
            bars_held += 1
            if bars_held >= hold_bars:
                pos, bars_held = 0, 0
        if pos == 0:
            up_break = c[i] > c[i - 1] + k * a[i - 1]
            dn_break = c[i] < c[i - 1] - k * a[i - 1]
            if up_break and c[i] > e[i]:
                pos, bars_held = 1, 0
            elif dn_break and c[i] < e[i] and not long_only:
                pos, bars_held = -1, 0
        sig[i] = pos
    return sig


def opening_range_breakout(df: pd.DataFrame, n_open_bars: int = 4,
                           hold_bars: int = 16, long_only: bool = True) -> np.ndarray:
    """Each UTC day: build the opening range from the first N bars.
    Long when price breaks above the opening high; short when below opening low.
    Exit at end of day or after hold_bars or on opposite break.
    Trades roughly every day — exactly what the user wants."""
    dt = pd.to_datetime(df["dt"])
    day = dt.dt.date.to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    pos, bars_held = 0, 0
    cur_day = None
    or_high, or_low, bar_in_day = -np.inf, np.inf, 0
    for i in range(len(df)):
        if day[i] != cur_day:
            # new day: close any open position, reset OR
            pos, bars_held = 0, 0
            cur_day = day[i]
            or_high, or_low, bar_in_day = -np.inf, np.inf, 0
        bar_in_day += 1
        if bar_in_day <= n_open_bars:
            or_high = max(or_high, h[i])
            or_low = min(or_low, l[i])
            sig[i] = 0
            continue
        if pos != 0:
            bars_held += 1
            # exit on hold_bars or reverse break
            if bars_held >= hold_bars:
                pos, bars_held = 0, 0
            elif pos == 1 and c[i] < or_low:
                pos, bars_held = 0, 0
            elif pos == -1 and c[i] > or_high:
                pos, bars_held = 0, 0
        if pos == 0:
            if c[i] > or_high:
                pos, bars_held = 1, 0
            elif c[i] < or_low and not long_only:
                pos, bars_held = -1, 0
        sig[i] = pos
    return sig


def session_momentum(df: pd.DataFrame, ema_fast: int = 20, ema_slow: int = 50,
                     trend_n: int = 200, sessions=((13, 21),)) -> np.ndarray:
    """Trend-follow on EMA cross BUT only enter during specified UTC hours.
    Outside sessions: hold existing position but don't open new."""
    dt = pd.to_datetime(df["dt"])
    hr = dt.dt.hour.to_numpy()
    c = df["close"].to_numpy()
    f = ema(c, ema_fast)
    s = ema(c, ema_slow)
    e = ema(c, trend_n)
    in_session = np.zeros(len(df), dtype=bool)
    for lo, hi in sessions:
        in_session |= (hr >= lo) & (hr <= hi)
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        # Always exit if cross reverses
        if pos == 1 and f[i] < s[i]:
            pos = 0
        elif pos == -1 and f[i] > s[i]:
            pos = 0
        # Enter only in session
        if pos == 0 and in_session[i]:
            if f[i] > s[i] and c[i] > e[i]:
                pos = 1
            elif f[i] < s[i] and c[i] < e[i]:
                pos = -1
        sig[i] = pos
    return sig


def vol_filtered_donchian(df: pd.DataFrame, n: int = 20, atr_n: int = 14,
                          atr_quantile_lookback: int = 200,
                          atr_quantile_threshold: float = 0.6,
                          long_only: bool = True) -> np.ndarray:
    """Donchian breakout but ONLY when current ATR is in upper quantile of recent ATRs.
    Idea: avoid breakouts in low-volatility chop where they whipsaw."""
    h = pd.Series(df["high"]).rolling(n).max().shift(1).to_numpy()
    l = pd.Series(df["low"]).rolling(n).min().shift(1).to_numpy()
    c = df["close"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    a = atr(hi, lo, c, atr_n)
    a_thresh = pd.Series(a).rolling(atr_quantile_lookback).quantile(atr_quantile_threshold).to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if np.isnan(h[i]) or np.isnan(a_thresh[i]):
            continue
        vol_ok = a[i] >= a_thresh[i]
        if pos == 0 and vol_ok:
            if c[i] > h[i]:
                pos = 1
            elif c[i] < l[i] and not long_only:
                pos = -1
        elif pos == 1 and c[i] < l[i]:
            pos = 0
        elif pos == -1 and c[i] > h[i]:
            pos = 0
        sig[i] = pos
    return sig


def bollinger_pullback(df: pd.DataFrame, n: int = 20, k: float = 2.0,
                       trend_n: int = 200) -> np.ndarray:
    """Trend-following pullback: in uptrend (close > EMA200), buy when price touches
    lower Bollinger band; sell when price reaches MA. Mirror for shorts."""
    cs = pd.Series(df["close"])
    ma = cs.rolling(n).mean()
    sd = cs.rolling(n).std(ddof=0)
    upper = (ma + k * sd).to_numpy()
    lower = (ma - k * sd).to_numpy()
    mav = ma.to_numpy()
    c = df["close"].to_numpy()
    e = ema(c, trend_n)
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if np.isnan(mav[i]):
            continue
        if pos == 0:
            if c[i] <= lower[i] and c[i] > e[i]:
                pos = 1
            elif c[i] >= upper[i] and c[i] < e[i]:
                pos = -1
        elif pos == 1:
            if c[i] >= mav[i] or c[i] < e[i]:
                pos = 0
        elif pos == -1:
            if c[i] <= mav[i] or c[i] > e[i]:
                pos = 0
        sig[i] = pos
    return sig


def macd_with_trend_filter(df: pd.DataFrame, f: int = 12, s: int = 26,
                           sig_n: int = 9, trend_n: int = 200) -> np.ndarray:
    """MACD signal cross filtered by EMA200 regime: long-only above EMA200,
    short-only below. Cuts the false signals scalping MACD got."""
    c = df["close"].to_numpy()
    macd = ema(c, f) - ema(c, s)
    sig_line = ema(macd, sig_n)
    e = ema(c, trend_n)
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if c[i] > e[i]:  # uptrend regime
            if macd[i] > sig_line[i]:
                pos = 1
            else:
                pos = 0
        else:
            if macd[i] < sig_line[i]:
                pos = -1
            else:
                pos = 0
        sig[i] = pos
    return sig


def breakout_with_pullback_entry(df: pd.DataFrame, n: int = 20,
                                  pullback_atr: float = 0.5,
                                  hold_bars: int = 24,
                                  long_only: bool = True) -> np.ndarray:
    """Wait for a Donchian-N break; then look for a small pullback within ATR
    before entering. Reduces 'chasing the top' bias."""
    h = pd.Series(df["high"]).rolling(n).max().shift(1).to_numpy()
    l = pd.Series(df["low"]).rolling(n).min().shift(1).to_numpy()
    c = df["close"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    a = atr(hi, lo, c, 14)
    sig = np.zeros(len(df), dtype=np.int8)
    pos, bars_held = 0, 0
    armed_long, armed_short = False, False
    arm_level_long, arm_level_short = 0.0, 0.0
    for i in range(1, len(df)):
        if np.isnan(h[i]):
            continue
        if pos != 0:
            bars_held += 1
            if bars_held >= hold_bars:
                pos, bars_held = 0, 0
            elif pos == 1 and c[i] < l[i]:
                pos, bars_held = 0, 0
            elif pos == -1 and c[i] > h[i]:
                pos, bars_held = 0, 0
        if pos == 0:
            # Arm long after a break
            if c[i] > h[i]:
                armed_long = True
                arm_level_long = c[i]
                armed_short = False
            elif c[i] < l[i] and not long_only:
                armed_short = True
                arm_level_short = c[i]
                armed_long = False
            # Trigger on pullback
            if armed_long and c[i] <= arm_level_long - pullback_atr * a[i]:
                pos, bars_held = 1, 0
                armed_long = False
            elif armed_short and c[i] >= arm_level_short + pullback_atr * a[i]:
                pos, bars_held = -1, 0
                armed_short = False
        sig[i] = pos
    return sig


SELECTIVE_STRATEGIES = {
    "BuyAndHold": lambda df: np.ones(len(df), dtype=np.int8),
    "ATR_BO_LO": lambda df: atr_breakout(df, 14, 1.0, 100, 8, True),
    "ATR_BO_LS": lambda df: atr_breakout(df, 14, 1.0, 100, 8, False),
    "OR_Breakout_LO": lambda df: opening_range_breakout(df, 4, 16, True),
    "OR_Breakout_LS": lambda df: opening_range_breakout(df, 4, 16, False),
    "Session_Mom_US": lambda df: session_momentum(df, 20, 50, 200, ((13, 21),)),
    "Session_Mom_All": lambda df: session_momentum(df, 20, 50, 200, ((0, 23),)),
    "VolFilt_Donchian": lambda df: vol_filtered_donchian(df, 20, 14, 200, 0.6, False),
    "VolFilt_Donchian_LO": lambda df: vol_filtered_donchian(df, 20, 14, 200, 0.6, True),
    "Boll_Pullback": bollinger_pullback,
    "MACD_Trend_Filter": macd_with_trend_filter,
    "BO_Pullback_Entry": breakout_with_pullback_entry,
}
