"""Portfolio construction: turns raw per-asset signal scores into normalized,
constrained target weights. Deliberately independent from the alpha/signal
layer (src/strategy/signals.py) — any strategy's raw scores can be fed
through this same construction so strategies stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PortfolioConstraints:
    max_asset_weight: float = 0.35
    max_gross_exposure: float = 1.0
    min_trade_threshold: float = 0.02
    cash_weight_floor: float = 0.0
    long_only: bool = True  # no leverage/shorting per competition rules


def volatility_scaled_weights(scores: pd.Series, volatility: pd.Series) -> pd.Series:
    """raw_weight_i = signal_i / volatility_i, per Phase 12. Assets with zero
    or missing volatility are excluded rather than divided-by-zero."""
    vol = volatility.replace(0.0, pd.NA)
    raw = scores / vol
    return raw.dropna()


def normalize_weights(raw_weights: pd.Series, constraints: PortfolioConstraints) -> pd.Series:
    """Clip to per-asset max, scale down to respect gross exposure, and
    (if long_only) drop negative raw weights entirely rather than shorting."""
    weights = raw_weights.copy()

    if constraints.long_only:
        weights = weights.clip(lower=0.0)

    weights = weights.clip(upper=constraints.max_asset_weight)

    gross = weights.abs().sum()
    max_gross = constraints.max_gross_exposure * (1.0 - constraints.cash_weight_floor)
    if gross > max_gross and gross > 0:
        weights = weights * (max_gross / gross)

    return weights


def apply_rebalance_threshold(
    target_weights: pd.Series, current_weights: pd.Series, threshold: float
) -> pd.Series:
    """Turnover control (Phase 14): keep the *current* weight for any asset
    whose target barely moved, instead of generating a trade for noise."""
    current = current_weights.reindex(target_weights.index, fill_value=0.0)
    delta = (target_weights - current).abs()
    final = target_weights.where(delta >= threshold, current)
    return final


def construct_portfolio(
    scores: pd.Series,
    volatility: pd.Series,
    current_weights: pd.Series,
    constraints: PortfolioConstraints,
) -> pd.Series:
    """Full pipeline: score -> vol-scale -> normalize/constrain -> hysteresis."""
    raw = volatility_scaled_weights(scores, volatility)
    target = normalize_weights(raw, constraints)
    target = target.reindex(current_weights.index.union(target.index), fill_value=0.0)
    return apply_rebalance_threshold(target, current_weights, constraints.min_trade_threshold)
