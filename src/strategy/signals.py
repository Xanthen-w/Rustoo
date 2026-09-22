"""Baseline strategies (Phase 8). Every strategy is a plain function taking a
"wide" close-price DataFrame (index=timestamp, columns=symbols) plus whatever
params it needs, and returning a target-weights DataFrame of the same shape:
weights in [0, 1] per asset, long-only, not necessarily summing to 1 (the
remainder is implicitly cash).

All strategies are causal: at row t they use only `close.loc[:t]`. This is
enforced structurally by only ever calling rolling/ewm/pct_change (backward
looking) on the full series — never `.shift(-n)` — and is checked by
tests/test_lookahead.py.

No strategy here shorts or uses leverage, matching the competition rules and
the "long-only" default in PortfolioConstraints.
"""
from __future__ import annotations

import pandas as pd

from src.features import cross_sectional as xs
from src.features import momentum as mom
from src.features import trend as trend_feat
from src.features import volatility as vol_feat


def buy_and_hold(close_wide: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """100% in one asset from the first bar it has a price, 0 elsewhere."""
    weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    has_price = close_wide[symbol].notna()
    weights.loc[has_price, symbol] = 1.0
    return weights


def equal_weight(close_wide: pd.DataFrame) -> pd.DataFrame:
    """Equal weight across every symbol that has a price at t, rebalanced
    every bar."""
    has_price = close_wide.notna()
    n_active = has_price.sum(axis=1).replace(0, pd.NA)
    weights = has_price.div(n_active, axis=0).fillna(0.0)
    return weights


def trend_following(close_wide: pd.DataFrame, fast_span: int = 20, slow_span: int = 60) -> pd.DataFrame:
    """Long an asset (equal-weighted among "in-trend" assets) only while its
    fast EMA is above its slow EMA, flat otherwise."""
    signals = {}
    for symbol in close_wide.columns:
        signals[symbol] = trend_feat.ema_crossover_signal(close_wide[symbol], fast_span, slow_span)
    signal_df = pd.DataFrame(signals)
    in_trend = signal_df > 0
    n_active = in_trend.sum(axis=1).replace(0, pd.NA)
    weights = in_trend.div(n_active, axis=0).fillna(0.0)
    return weights


def single_asset_momentum(close_wide: pd.DataFrame, symbol: str, lookback: int = 96) -> pd.DataFrame:
    """Long the asset when its own trailing momentum is positive, flat
    otherwise. Binary (0 or 1), single symbol."""
    weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    momentum = mom.returns(close_wide[symbol], lookback)
    weights.loc[momentum > 0, symbol] = 1.0
    return weights


def cross_sectional_momentum(close_wide: pd.DataFrame, lookback: int = 96, top_k: int = 3) -> pd.DataFrame:
    """Equal-weight the top-k assets by trailing momentum at each bar; flat
    on everything else. Never shorts the bottom-k (long-only)."""
    momentum_wide = pd.DataFrame(
        {symbol: mom.returns(close_wide[symbol], lookback) for symbol in close_wide.columns}
    )
    mask = xs.top_k_mask(momentum_wide, top_k) & (momentum_wide > 0)
    n_active = mask.sum(axis=1).replace(0, pd.NA)
    weights = mask.div(n_active, axis=0).fillna(0.0)
    return weights


def mean_reversion(close_wide: pd.DataFrame, lookback: int = 12, z_entry: float = 1.5) -> pd.DataFrame:
    """Long an asset when its short-term return z-score is below -z_entry
    (oversold), scaled down as it reverts back toward the mean. Never shorts
    an "overbought" asset — long-only, so overbought just means flat."""
    weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    for symbol in close_wide.columns:
        ret = mom.returns(close_wide[symbol], lookback)
        z = (ret - ret.rolling(lookback, min_periods=lookback).mean()) / ret.rolling(
            lookback, min_periods=lookback
        ).std()
        oversold = (-z).clip(lower=0.0) / z_entry
        weights[symbol] = oversold.where(z < -z_entry, 0.0).clip(upper=1.0)
    return weights


def volatility_filtered_momentum(
    close_wide: pd.DataFrame,
    momentum_lookback: int = 96,
    vol_lookback: int = 96,
    vol_percentile_cutoff: float = 0.8,
) -> pd.DataFrame:
    """Cross-sectional momentum, but an asset is excluded whenever its own
    realized-vol percentile is above `vol_percentile_cutoff` (i.e. skip
    assets currently in a high-volatility regime, regardless of how strong
    their momentum looks)."""
    momentum_wide = pd.DataFrame(
        {symbol: mom.returns(close_wide[symbol], momentum_lookback) for symbol in close_wide.columns}
    )
    vol_ok = pd.DataFrame(index=close_wide.index, columns=close_wide.columns, dtype=bool)
    for symbol in close_wide.columns:
        realized = vol_feat.realized_vol(close_wide[symbol], vol_lookback)
        pctile = vol_feat.volatility_percentile(realized, vol_lookback)
        vol_ok[symbol] = pctile < vol_percentile_cutoff

    positive_momentum = momentum_wide > 0
    eligible = positive_momentum & vol_ok
    n_active = eligible.sum(axis=1).replace(0, pd.NA)
    weights = eligible.div(n_active, axis=0).fillna(0.0)
    return weights
