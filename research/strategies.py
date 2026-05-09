"""
Library of scalping/short-term strategies.

Each strategy returns a numpy int8 array of {-1, 0, +1} signals,
one entry per bar, computed from data available AT close of each bar.
The backtest engine handles the next-bar-open execution lag.

All strategies use only past data (rolling/shift) -- no look-ahead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


# ------------------------ helper indicators ---------------------------

def ema(x: np.ndarray, n: int) -> np.ndarray:
    a = 2 / (n + 1)
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def rsi(close: np.ndarray, n: int = 14) -> np.ndarray:
    diff = np.diff(close, prepend=close[0])
    up = np.where(diff > 0, diff, 0.0)
    dn = np.where(diff < 0, -diff, 0.0)
    # Wilder's smoothing
    avg_u = np.zeros_like(close)
    avg_d = np.zeros_like(close)
    avg_u[:n] = up[:n].mean()
    avg_d[:n] = dn[:n].mean()
    for i in range(n, len(close)):
        avg_u[i] = (avg_u[i - 1] * (n - 1) + up[i]) / n
        avg_d[i] = (avg_d[i - 1] * (n - 1) + dn[i]) / n
    rs = np.where(avg_d == 0, 0, avg_u / np.where(avg_d == 0, 1, avg_d))
    return 100 - 100 / (1 + rs)


def atr(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    out = np.zeros_like(c)
    out[:n] = tr[:n].mean()
    for i in range(n, len(c)):
        out[i] = (out[i - 1] * (n - 1) + tr[i]) / n
    return out


def rolling_z(x: np.ndarray, n: int) -> np.ndarray:
    s = pd.Series(x)
    mu = s.rolling(n).mean()
    sd = s.rolling(n).std(ddof=0)
    z = (s - mu) / sd.replace(0, np.nan)
    return z.fillna(0).to_numpy()


# ------------------------ strategies ---------------------------------

def s1_ema_cross(df: pd.DataFrame, fast: int = 9, slow: int = 21) -> np.ndarray:
    """Classic trend-following: long when fast EMA > slow EMA, short otherwise.
    Scalping variant on 1m: entries are frequent, edge depends on trend persistence."""
    c = df["close"].to_numpy()
    f = ema(c, fast)
    s = ema(c, slow)
    sig = np.where(f > s, 1, -1).astype(np.int8)
    return sig


def s2_rsi_meanrev(df: pd.DataFrame, n: int = 14, low: float = 25, high: float = 75) -> np.ndarray:
    """RSI mean reversion: long when RSI crosses up from <low, short crosses down from >high.
    Position is held until RSI returns to neutral 45-55 zone."""
    c = df["close"].to_numpy()
    r = rsi(c, n)
    sig = np.zeros(len(c), dtype=np.int8)
    pos = 0
    for i in range(1, len(c)):
        if pos == 0:
            if r[i - 1] < low and r[i] >= low:
                pos = 1
            elif r[i - 1] > high and r[i] <= high:
                pos = -1
        else:
            if 45 <= r[i] <= 55:
                pos = 0
        sig[i] = pos
    return sig


def s3_donchian_breakout(df: pd.DataFrame, n: int = 20) -> np.ndarray:
    """Donchian channel breakout: long when close exceeds N-bar high, short when below N-bar low."""
    h = pd.Series(df["high"]).rolling(n).max().shift(1)
    l = pd.Series(df["low"]).rolling(n).min().shift(1)
    c = df["close"]
    sig = np.zeros(len(df), dtype=np.int8)
    sig[(c > h).fillna(False).to_numpy()] = 1
    sig[(c < l).fillna(False).to_numpy()] = -1
    return sig


def s4_bollinger_revert(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> np.ndarray:
    """Bollinger band mean-reversion: short when close pierces upper band, long when pierces lower.
    Exit when price crosses back through the SMA. Classic overextension scalp."""
    c = pd.Series(df["close"])
    ma = c.rolling(n).mean()
    sd = c.rolling(n).std(ddof=0)
    up = ma + k * sd
    dn = ma - k * sd
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    cv = c.to_numpy()
    mav = ma.to_numpy()
    upv = up.to_numpy()
    dnv = dn.to_numpy()
    for i in range(1, len(df)):
        if np.isnan(mav[i]):
            sig[i] = 0
            continue
        if pos == 0:
            if cv[i] >= upv[i]:
                pos = -1
            elif cv[i] <= dnv[i]:
                pos = 1
        else:
            if pos == 1 and cv[i] >= mav[i]:
                pos = 0
            elif pos == -1 and cv[i] <= mav[i]:
                pos = 0
        sig[i] = pos
    return sig


def s5_keltner_squeeze(df: pd.DataFrame, n: int = 20, k_atr: float = 1.5) -> np.ndarray:
    """Keltner channel breakout with ATR-based bands. Long > upper, short < lower."""
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    e = ema(c, n)
    a = atr(h, l, c, n)
    up = e + k_atr * a
    dn = e - k_atr * a
    sig = np.where(c > up, 1, np.where(c < dn, -1, 0)).astype(np.int8)
    return sig


def s6_zscore_revert(df: pd.DataFrame, n: int = 30, z_in: float = 2.0, z_out: float = 0.3) -> np.ndarray:
    """Z-score mean reversion: short when zscore > z_in, long when zscore < -z_in.
    Exit when |z| < z_out."""
    z = rolling_z(df["close"].to_numpy(), n)
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if pos == 0:
            if z[i] >= z_in:
                pos = -1
            elif z[i] <= -z_in:
                pos = 1
        else:
            if abs(z[i]) <= z_out:
                pos = 0
            elif pos == 1 and z[i] >= z_in * 0.8:
                pos = 0
            elif pos == -1 and z[i] <= -z_in * 0.8:
                pos = 0
        sig[i] = pos
    return sig


def s7_macd_micro(df: pd.DataFrame, f: int = 12, s: int = 26, sig_n: int = 9) -> np.ndarray:
    """MACD signal-line cross. Long when MACD > signal, short otherwise."""
    c = df["close"].to_numpy()
    macd = ema(c, f) - ema(c, s)
    sig_line = ema(macd, sig_n)
    return np.where(macd > sig_line, 1, -1).astype(np.int8)


def s8_vwap_revert(df: pd.DataFrame, n: int = 60, k: float = 1.5) -> np.ndarray:
    """Rolling-VWAP fade: short when price > VWAP + k*sigma, long when < VWAP - k*sigma.
    Treated as a scalping mean-reversion around volume-weighted average price."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vp = tp * df["volume"]
    rv = df["volume"].rolling(n).sum().replace(0, np.nan)
    vwap = vp.rolling(n).sum() / rv
    sd = (tp - vwap).rolling(n).std(ddof=0)
    up = vwap + k * sd
    dn = vwap - k * sd
    c = df["close"].to_numpy()
    upv = up.to_numpy()
    dnv = dn.to_numpy()
    vwv = vwap.to_numpy()
    sig = np.zeros(len(df), dtype=np.int8)
    pos = 0
    for i in range(1, len(df)):
        if np.isnan(vwv[i]):
            continue
        if pos == 0:
            if c[i] >= upv[i]:
                pos = -1
            elif c[i] <= dnv[i]:
                pos = 1
        else:
            if pos == 1 and c[i] >= vwv[i]:
                pos = 0
            elif pos == -1 and c[i] <= vwv[i]:
                pos = 0
        sig[i] = pos
    return sig


def s9_breakout_with_trend(df: pd.DataFrame, n_break: int = 20, n_trend: int = 200) -> np.ndarray:
    """Donchian breakout filtered by EMA200 trend (only longs in uptrend, only shorts in downtrend)."""
    h = pd.Series(df["high"]).rolling(n_break).max().shift(1).to_numpy()
    l = pd.Series(df["low"]).rolling(n_break).min().shift(1).to_numpy()
    c = df["close"].to_numpy()
    e = ema(c, n_trend)
    sig = np.zeros(len(df), dtype=np.int8)
    long_ok = c > e
    short_ok = c < e
    sig[(c > h) & long_ok] = 1
    sig[(c < l) & short_ok] = -1
    return sig


def s10_buy_and_hold(df: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df), dtype=np.int8)


STRATEGIES = {
    "BuyAndHold": s10_buy_and_hold,
    "EMA_9_21": s1_ema_cross,
    "RSI_MR_25_75": s2_rsi_meanrev,
    "Donchian_20": s3_donchian_breakout,
    "Bollinger_MR": s4_bollinger_revert,
    "Keltner_20_1.5": s5_keltner_squeeze,
    "ZScore_30_2": s6_zscore_revert,
    "MACD_12_26_9": s7_macd_micro,
    "VWAP_Fade_60": s8_vwap_revert,
    "DonchianTrend": s9_breakout_with_trend,
}
