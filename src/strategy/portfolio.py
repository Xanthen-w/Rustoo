"""Portfolio construction: turns raw per-asset signal scores into normalized,
constrained target weights. Deliberately independent from the alpha/signal
layer (src/strategy/signals.py) — any strategy's raw scores can be fed
through this same construction so strategies stay comparable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
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
    vol = volatility.where(volatility > 0)
    raw = scores / vol
    return raw.dropna()


def normalize_weights(raw_weights: pd.Series, constraints: PortfolioConstraints) -> pd.Series:
    """Turn raw weights into constrained target weights.

    1. If long_only, drop negative raw weights (flat, never short).
    2. If gross exposure exceeds the budget (max_gross_exposure less the
       cash floor), scale everything down proportionally to fit it. Raw
       weights under budget are kept as-is — a weak signal stays small.
    3. Cap each asset at max_asset_weight. When step 2 scaled down (the
       signal wanted the full budget), the excess above the cap is
       redistributed pro rata over the uncapped assets; whatever can't be
       placed without breaching a cap stays in cash.

    Scaling happens before capping so relative sizes survive: raw weights
    like signal/volatility are often >> 1, and capping those first would
    flatten every asset to the same max weight.
    """
    weights = raw_weights.astype(float).copy()

    if constraints.long_only:
        weights = weights.clip(lower=0.0)

    budget = constraints.max_gross_exposure * (1.0 - constraints.cash_weight_floor)
    gross = weights.abs().sum()
    fill_budget = gross > budget and gross > 0
    if fill_budget:
        weights = weights * (budget / gross)

    cap = constraints.max_asset_weight
    if not fill_budget:
        return weights.clip(lower=-cap, upper=cap)

    # Water-filling: repeatedly pin assets at the cap and hand their excess
    # to the rest, in proportion to their current weights.
    values = weights.to_numpy(copy=True)
    capped = np.zeros(len(values), dtype=bool)
    for _ in range(len(values)):
        over = (np.abs(values) > cap + 1e-12) & ~capped
        if not over.any():
            break
        capped |= over
        values[capped] = np.sign(values[capped]) * cap
        remaining = budget - np.abs(values[capped]).sum()
        free_total = np.abs(values[~capped]).sum()
        if remaining <= 0 or free_total <= 0:
            break
        values[~capped] *= remaining / free_total
    return pd.Series(values, index=weights.index)


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
