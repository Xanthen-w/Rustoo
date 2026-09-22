"""Volume features. Causal rolling stats only."""
from __future__ import annotations

import pandas as pd


def volume_zscore(volume: pd.Series, lookback: int) -> pd.Series:
    mean = volume.rolling(lookback, min_periods=lookback).mean()
    std = volume.rolling(lookback, min_periods=lookback).std()
    return (volume - mean) / std


def volume_change(volume: pd.Series, lookback: int) -> pd.Series:
    return volume.pct_change(periods=lookback)


def price_volume_interaction(close: pd.Series, volume: pd.Series, lookback: int) -> pd.Series:
    """Correlation between price returns and volume changes over a rolling
    window — a crude proxy for whether moves are volume-confirmed."""
    ret = close.pct_change()
    vol_chg = volume.pct_change()
    return ret.rolling(lookback, min_periods=lookback).corr(vol_chg)
