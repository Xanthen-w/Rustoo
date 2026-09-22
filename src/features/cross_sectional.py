"""Cross-sectional features: rank/z-score across assets at each timestamp.

Inputs here are "wide" DataFrames: index = timestamp, columns = symbols. Each
row is ranked independently, using only that row's own values, so there is no
leakage across time — only across the (already-causal) per-asset feature
values at that same timestamp.
"""
from __future__ import annotations

import pandas as pd


def cross_sectional_rank(metric_wide: pd.DataFrame, pct: bool = True) -> pd.DataFrame:
    """Rank each row (timestamp) across columns (symbols). 1.0 = highest."""
    return metric_wide.rank(axis=1, pct=pct)


def cross_sectional_zscore(metric_wide: pd.DataFrame) -> pd.DataFrame:
    row_mean = metric_wide.mean(axis=1)
    row_std = metric_wide.std(axis=1)
    return metric_wide.sub(row_mean, axis=0).div(row_std, axis=0)


def relative_strength(returns_wide: pd.DataFrame) -> pd.DataFrame:
    """Each asset's return minus the row's cross-sectional mean return —
    positive means it outperformed the universe at that timestamp."""
    return returns_wide.sub(returns_wide.mean(axis=1), axis=0)


def top_k_mask(metric_wide: pd.DataFrame, k: int) -> pd.DataFrame:
    """Boolean mask, True for the top-k assets by `metric_wide` in each row.
    NaNs never qualify as top-k."""
    ranks = metric_wide.rank(axis=1, ascending=False, method="first")
    return ranks <= k
