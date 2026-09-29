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


def efficiency_ratio(close: pd.Series, window: int, step: int = 24) -> pd.Series:
    """Kaufman's efficiency ratio over `window` bars: |net move| divided by
    the path length travelled in `step`-bar hops (daily hops on hourly bars,
    so hour-to-hour noise doesn't swamp it). Near 1 = a clean directional
    move; near 0 = back-and-forth chop. Causal (trailing windows only)."""
    net = (close - close.shift(window)).abs()
    path = (close - close.shift(step)).abs().rolling(window, min_periods=window).sum() / step
    return net / path.where(path > 0)


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


def confirmed_trend_state(close: pd.Series, span: int, band: float, confirm: pd.Series, threshold: float) -> pd.Series:
    """trend_state whose *entries* also require `confirm >= threshold`
    (e.g. an efficiency ratio: only enter trends that are moving cleanly).
    Exits are unchanged (close < EMA * (1 - band)), so risk control is the
    same. With threshold <= 0 (and a defined confirm) it equals trend_state.
    Causal as long as `confirm` is."""
    e = ema(close, span)
    state = pd.Series(float("nan"), index=close.index)
    state[(close > e * (1.0 + band)) & (confirm >= threshold)] = 1.0
    state[close < e * (1.0 - band)] = 0.0
    state[e.isna()] = 0.0
    return state.ffill().fillna(0.0)
