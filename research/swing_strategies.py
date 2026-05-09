"""
Swing & position strategies for higher timeframes (1h, 4h, daily).

Goal: signals that hold for hours-to-weeks, so fee bleed is minimal.
Each function returns int8 array of {-1, 0, +1} computed at bar close.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from strategies import ema, rsi, atr


# ------------------ trend following ----------------------------

def sma_cross(df: pd.DataFrame, fast: int = 50, slow: int = 200, long_only: bool = False) -> np.ndarray:
    """Golden-cross / death-cross. Classic but slow."""
    c = pd.Series(df["close"])
    f = c.rolling(fast).mean()
    s = c.rolling(slow).mean()
    sig = np.zeros(len(df), dtype=np.int8)
    fv, sv = f.to_numpy(), s.to_numpy()
    mask = ~(np.isnan(fv) | np.isnan(sv))
    if long_only:
        sig[mask & (fv > sv)] = 1
    else:
        sig[mask & (fv > sv)] = 1
        sig[mask & (fv <= sv)] = -1
    return sig


def turtle(df: pd.DataFrame, n_in: int = 20, n_out: int = 10, long_only: bool = False) -> np.ndarray:
    """Turtle Trader: enter on N-bar high/low breakout, exit on opposite N-bar break.
    Position holds across many bars - very low trade count."""
    h = pd.Series(df["high"])
    l = pd.Series(df["low"])
    c = df["close"].to_numpy()
    high_in = h.rolling(n_in).max().shift(1).to_numpy()
    low_in = l.rolling(n_in).min().shift(1).to_numpy()
    high_out = h.rolling(n_out).max().shift(1).to_numpy()
    low_out = l.rolling(n_out).min().shift(1).to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(len(df)):
        if np.isnan(high_in[i]):
            continue
        if pos == 0:
            if c[i] > high_in[i]:
                pos = 1
            elif c[i] < low_in[i] and not long_only:
                pos = -1
        elif pos == 1:
            if c[i] < low_out[i]:
                pos = -1 if not long_only else 0
        elif pos == -1:
            if c[i] > high_out[i]:
                pos = 1
        sig[i] = pos
    return sig


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0, long_only: bool = False) -> np.ndarray:
    """ATR-based trend follower (Supertrend). Flips when price closes through trailing band."""
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    a = atr(h, l, c, n)
    hl2 = (h + l) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    final_up = upper.copy()
    final_dn = lower.copy()
    sig = np.zeros(len(c), dtype=np.int8)
    direction = 1
    for i in range(1, len(c)):
        final_up[i] = min(upper[i], final_up[i - 1]) if c[i - 1] <= final_up[i - 1] else upper[i]
        final_dn[i] = max(lower[i], final_dn[i - 1]) if c[i - 1] >= final_dn[i - 1] else lower[i]
        if direction == 1 and c[i] < final_dn[i - 1]:
            direction = -1
        elif direction == -1 and c[i] > final_up[i - 1]:
            direction = 1
        if long_only:
            sig[i] = 1 if direction == 1 else 0
        else:
            sig[i] = direction
    return sig


def trend_filter_long_only(df: pd.DataFrame, ema_n: int = 200) -> np.ndarray:
    """Brutally simple: long when above EMA200, flat below. Captures bull markets, dodges crashes."""
    c = df["close"].to_numpy()
    e = ema(c, ema_n)
    return (c > e).astype(np.int8)


# ------------------ mean reversion (slower) ----------------------------

def buy_the_dip(df: pd.DataFrame, rsi_n: int = 2, ema_n: int = 200, exit_rsi: int = 70) -> np.ndarray:
    """Connors-style: long when RSI(2) < 5 AND price > EMA200; exit when RSI > exit_rsi.
    A famous setup that's been documented for two decades on stocks; let's see on BTC."""
    c = df["close"].to_numpy()
    r = rsi(c, rsi_n)
    e = ema(c, ema_n)
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if pos == 0:
            if r[i] < 5 and c[i] > e[i]:
                pos = 1
        else:
            if r[i] > exit_rsi:
                pos = 0
        sig[i] = pos
    return sig


def rsi_extremes(df: pd.DataFrame, n: int = 14, lo: float = 30, hi: float = 70) -> np.ndarray:
    """Long when RSI < lo, short when > hi, exit when crossing 50."""
    r = rsi(df["close"].to_numpy(), n)
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if pos == 0:
            if r[i] < lo:
                pos = 1
            elif r[i] > hi:
                pos = -1
        elif pos == 1 and r[i] > 50:
            pos = 0
        elif pos == -1 and r[i] < 50:
            pos = 0
        sig[i] = pos
    return sig


# ------------------ volatility breakout ----------------------------

def larry_williams_vbo(df: pd.DataFrame, k: float = 0.5, long_only: bool = True) -> np.ndarray:
    """Larry Williams' volatility breakout: enter long if today's price exceeds
    yesterday's close + k * yesterday's range; close end of bar.
    Implemented as bar-bar swing here (held one bar): we approximate by signaling
    long when (close > prev_close + k * prev_range).
    Held one bar: signal flips back to 0 next bar."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    for i in range(1, len(df)):
        rng = h[i - 1] - l[i - 1]
        if c[i] > c[i - 1] + k * rng:
            sig[i] = 1
        elif not long_only and c[i] < c[i - 1] - k * rng:
            sig[i] = -1
        # else 0
    return sig


def keltner_breakout(df: pd.DataFrame, n: int = 20, k: float = 2.0, long_only: bool = False) -> np.ndarray:
    """Keltner channel breakout, position-style (hold until opposite break)."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    e = ema(c, n)
    a = atr(h, l, c, n)
    up = e + k * a
    dn = e - k * a
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if c[i] > up[i]:
            pos = 1
        elif c[i] < dn[i]:
            pos = -1 if not long_only else 0
        sig[i] = pos
    return sig


# ------------------ momentum ----------------------------

def momentum_roc(df: pd.DataFrame, n: int = 90, long_only: bool = False) -> np.ndarray:
    """Long when N-bar rate-of-change is positive (i.e. price higher than N bars ago).
    Cross-sectional momentum is well-documented in equities; absolute momentum on BTC?"""
    c = pd.Series(df["close"])
    roc = (c / c.shift(n) - 1).to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    if long_only:
        sig[roc > 0] = 1
    else:
        sig[roc > 0] = 1
        sig[roc < 0] = -1
    return sig


def dual_momentum(df: pd.DataFrame, n_short: int = 30, n_long: int = 90) -> np.ndarray:
    """Long only when BOTH short and long-window ROC are positive (regime filter)."""
    c = pd.Series(df["close"])
    rs = (c / c.shift(n_short) - 1).to_numpy()
    rl = (c / c.shift(n_long) - 1).to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    sig[(rs > 0) & (rl > 0)] = 1
    return sig


# ------------------ time-based / structural ----------------------------

def weekend_effect(df: pd.DataFrame) -> np.ndarray:
    """Long only over weekends (Fri close to Mon close) -- folk-tale BTC pattern."""
    dt = pd.to_datetime(df["dt"])
    sig = np.zeros(len(df), dtype=np.int8)
    # Long when day-of-week >= Friday (4) or <= Sunday (6)
    dow = dt.dt.dayofweek.to_numpy()
    sig[(dow >= 4) | (dow == 0)] = 1  # Fri-Sat-Sun + Mon
    return sig


def us_session_only(df: pd.DataFrame, ema_n: int = 50) -> np.ndarray:
    """Trend-follow but only during US trading hours (13-21 UTC)."""
    dt = pd.to_datetime(df["dt"])
    hr = dt.dt.hour.to_numpy()
    c = df["close"].to_numpy()
    e = ema(c, ema_n)
    sig = np.zeros(len(df), dtype=np.int8)
    in_session = (hr >= 13) & (hr <= 21)
    sig[in_session & (c > e)] = 1
    sig[in_session & (c < e)] = -1
    return sig


# ------------------ ensemble / multi-rule ----------------------------

def trend_plus_dip(df: pd.DataFrame, ema_n: int = 200, rsi_n: int = 14, rsi_buy: float = 35) -> np.ndarray:
    """Long only when in uptrend (above EMA200) AND we get a pullback (RSI < 35).
    Holds until RSI > 65 or price falls below EMA200. Long-only."""
    c = df["close"].to_numpy()
    e = ema(c, ema_n)
    r = rsi(c, rsi_n)
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if pos == 0:
            if c[i] > e[i] and r[i] < rsi_buy:
                pos = 1
        else:
            if r[i] > 65 or c[i] < e[i]:
                pos = 0
        sig[i] = pos
    return sig


def channel_with_volume(df: pd.DataFrame, n: int = 20, vol_mult: float = 1.5) -> np.ndarray:
    """Donchian breakout confirmed by above-average volume."""
    h = pd.Series(df["high"]).rolling(n).max().shift(1).to_numpy()
    l = pd.Series(df["low"]).rolling(n).min().shift(1).to_numpy()
    v = pd.Series(df["volume"])
    avg_v = v.rolling(n).mean().to_numpy()
    cv = df["close"].to_numpy()
    vv = v.to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if np.isnan(h[i]) or np.isnan(avg_v[i]):
            sig[i] = pos
            continue
        if pos == 0:
            if cv[i] > h[i] and vv[i] > vol_mult * avg_v[i]:
                pos = 1
            elif cv[i] < l[i] and vv[i] > vol_mult * avg_v[i]:
                pos = -1
        else:
            # exit on Donchian-10 opposite break
            if pos == 1 and cv[i] < l[i]:
                pos = 0
            elif pos == -1 and cv[i] > h[i]:
                pos = 0
        sig[i] = pos
    return sig


SWING_STRATEGIES = {
    "BuyAndHold": lambda df: np.ones(len(df), dtype=np.int8),
    "SMA_50_200": lambda df: sma_cross(df, 50, 200),
    "SMA_50_200_LO": lambda df: sma_cross(df, 50, 200, long_only=True),
    "Turtle_20_10": lambda df: turtle(df, 20, 10),
    "Turtle_20_10_LO": lambda df: turtle(df, 20, 10, long_only=True),
    "Turtle_55_20": lambda df: turtle(df, 55, 20),
    "Supertrend_10_3": lambda df: supertrend(df, 10, 3.0),
    "Supertrend_10_3_LO": lambda df: supertrend(df, 10, 3.0, long_only=True),
    "EMA200_filter": trend_filter_long_only,
    "BuyTheDip_Connors": buy_the_dip,
    "RSI_30_70": rsi_extremes,
    "LarryWilliams_VBO": larry_williams_vbo,
    "Keltner_BO_LO": lambda df: keltner_breakout(df, 20, 2.0, long_only=True),
    "Mom_ROC_90": lambda df: momentum_roc(df, 90),
    "Mom_ROC_90_LO": lambda df: momentum_roc(df, 90, long_only=True),
    "DualMomentum_30_90": dual_momentum,
    "Weekend_Effect": weekend_effect,
    "USSession_EMA50": us_session_only,
    "Trend_Plus_Dip": trend_plus_dip,
    "Donchian_Volume": channel_with_volume,
}
