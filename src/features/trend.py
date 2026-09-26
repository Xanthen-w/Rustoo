"""Trend features: EMA, price/EMA, EMA slope, crossovers. All causal
(pandas `.ewm()` / `.rolling()` are backward-looking by construction)."""
from __future__ import annotations

import pandas as pd


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def price_over_ema(close: pd.Series, span: int) -> pd.Series:
    return close / ema(close, span) - 1.0


def ema_slope(close: pd.Series, span: int, slope_lookback: int = 1) -> pd.Series:
    """Fractional change of the EMA itself over `slope_lookback` bars —
    positive means the trend line is rising."""
    e = ema(close, span)
    return e.pct_change(periods=slope_lookback)


def ema_crossover_signal(close: pd.Series, fast_span: int, slow_span: int) -> pd.Series:
    """+1 when fast EMA > slow EMA (uptrend), -1 otherwise. NaN until both
    EMAs are defined."""
    fast = ema(close, fast_span)
    slow = ema(close, slow_span)
    signal = (fast > slow).astype(float) * 2 - 1
    signal[fast.isna() | slow.isna()] = float("nan")
    return signal


def trend_state(close: pd.Series, span: int, band: float = 0.0) -> pd.Series:
    """1.0 while in an uptrend, 0.0 otherwise, with hysteresis: switches on
    when close > EMA * (1 + band), off when close < EMA * (1 - band), and
    otherwise keeps its previous state — so a price hovering around the EMA
    doesn't flip the position every bar. 0.0 until the EMA is warmed up.
    Causal: the state at t depends only on closes <= t."""
    e = ema(close, span)
    state = pd.Series(float("nan"), index=close.index)
    state[close > e * (1.0 + band)] = 1.0
    state[close < e * (1.0 - band)] = 0.0
    state[e.isna()] = 0.0
    return state.ffill().fillna(0.0)
