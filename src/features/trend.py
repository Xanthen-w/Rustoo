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
