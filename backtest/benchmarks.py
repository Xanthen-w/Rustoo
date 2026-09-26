"""Benchmarks and baselines, run through the same engine and costs as the
strategy on the same bars (synchronized periods).

- Buy-and-hold of single assets and of an equal-weight basket (bought once,
  never rebalanced: targets follow the basket's drifting weights), and cash.
- Random-entry baseline: the null hypothesis for a timing strategy. For each
  seed it copies the strategy's exposure profile — per asset, its average
  weight while "on" and while "off", and how often it switches — but draws
  the switch times at random. If the strategy doesn't beat most random seeds,
  its timing signal isn't doing measurable work.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Benchmarks are passive: targets follow the drifting weights of a basket
# bought once (below), and this small band keeps float-level differences
# between those targets and the actual holdings from generating trades.
BENCHMARK_BAND = 0.01


def buy_and_hold_weights(close: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """Target weights of a basket bought in equal value on the first bar
    where every asset has a price, then never rebalanced: each asset's
    weight drifts with its price. Zero before that bar."""
    w = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    prices = close[assets].ffill()
    start = prices.dropna().index
    if len(start) == 0:
        return w
    start = start[0]
    growth = prices.loc[start:] / prices.loc[start]
    w.loc[start:, assets] = growth.div(growth.sum(axis=1), axis=0).to_numpy()
    return w


def exposure_profile(weights: pd.Series) -> dict:
    """Split one asset's weight path into 'on' (above half its max) and
    'off', and measure the levels and the switching rate."""
    w = weights.fillna(0.0)
    if w.max() <= 0:
        return {"on_weight": 0.0, "off_weight": 0.0, "switch_prob": 0.0, "on_share": 0.0}
    on = w > 0.5 * w.max()
    switches = int((on != on.shift()).iloc[1:].sum())
    return {
        "on_weight": float(w[on].mean()),
        "off_weight": float(w[~on].mean()) if (~on).any() else 0.0,
        "switch_prob": switches / max(len(w) - 1, 1),
        "on_share": float(on.mean()),
    }


def random_entry_weights(close: pd.DataFrame, strategy_weights: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random on/off timing with the strategy's own exposure profile per
    asset (see module docstring). Causal by construction: the state path is
    drawn independently of prices."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for asset in strategy_weights.columns[(strategy_weights != 0).any()]:
        prof = exposure_profile(strategy_weights[asset])
        n = len(close)
        flips = rng.random(n) < prof["switch_prob"]
        state = np.empty(n, dtype=bool)
        state[0] = rng.random() < prof["on_share"]
        for i in range(1, n):
            state[i] = state[i - 1] ^ flips[i]
        out[asset] = np.where(state, prof["on_weight"], prof["off_weight"])
    gross = out.sum(axis=1)
    return out.div(gross.where(gross > 1.0, 1.0), axis=0).where(close.notna(), 0.0)
