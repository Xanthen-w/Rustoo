"""Volatility features: rolling std, ATR, Parkinson, percentile/regime.
All rolling — causal by construction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def realized_vol(close: pd.Series, lookback: int, annualize_periods_per_year: float | None = None) -> pd.Series:
    ret = close.pct_change()
    vol = ret.rolling(lookback, min_periods=lookback).std()
    if annualize_periods_per_year:
        vol = vol * np.sqrt(annualize_periods_per_year)
    return vol


def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(lookback, min_periods=lookback).mean()


def parkinson_volatility(high: pd.Series, low: pd.Series, lookback: int) -> pd.Series:
    """Parkinson (1980) high-low range volatility estimator."""
    log_hl_sq = (np.log(high / low)) ** 2
    factor = 1.0 / (4.0 * np.log(2.0))
    return np.sqrt(factor * log_hl_sq.rolling(lookback, min_periods=lookback).mean())


def volatility_percentile(vol: pd.Series, lookback: int) -> pd.Series:
    """Where the current vol reading sits in its own trailing history, in
    [0, 1]. Uses only data up to and including t."""

    def _pct_rank(window: np.ndarray) -> float:
        current = window[-1]
        return float((window <= current).mean())

    return vol.rolling(lookback, min_periods=lookback).apply(_pct_rank, raw=True)


def volatility_regime(vol_percentile: pd.Series, high_threshold: float = 0.8, low_threshold: float = 0.2) -> pd.Series:
    """Categorical regime label from a volatility percentile series."""
    regime = pd.Series("NORMAL", index=vol_percentile.index, dtype=object)
    regime[vol_percentile >= high_threshold] = "HIGH_VOL"
    regime[vol_percentile <= low_threshold] = "LOW_VOL"
    regime[vol_percentile.isna()] = None
    return regime
