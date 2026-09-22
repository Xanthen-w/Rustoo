"""Momentum features. All functions are causal: the value at index t depends
only on data at or before t (rolling windows / backward pct_change), never on
future bars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def returns(close: pd.Series, lookback: int) -> pd.Series:
    """Simple return over `lookback` bars ending at t: close[t]/close[t-lookback] - 1."""
    return close.pct_change(periods=lookback)


def log_returns(close: pd.Series, lookback: int) -> pd.Series:
    return np.log(close / close.shift(lookback))


def multi_lookback_momentum(close: pd.Series, lookbacks: list[int]) -> pd.DataFrame:
    """One momentum column per lookback, named `mom_{lookback}`."""
    return pd.DataFrame({f"mom_{lb}": returns(close, lb) for lb in lookbacks})


def momentum_score(close: pd.Series, lookbacks: list[int], weights: list[float] | None = None) -> pd.Series:
    """Equal- or custom-weighted average of z-scored momentum across several
    lookbacks — a single composite momentum signal per asset per bar."""
    mat = multi_lookback_momentum(close, lookbacks)
    z = (mat - mat.rolling(max(lookbacks), min_periods=max(lookbacks)).mean()) / mat.rolling(
        max(lookbacks), min_periods=max(lookbacks)
    ).std()
    if weights is None:
        return z.mean(axis=1)
    w = pd.Series(weights, index=mat.columns)
    return (z * w).sum(axis=1) / w.sum()
