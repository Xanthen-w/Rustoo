import numpy as np
import pandas as pd
import pytest

from src.strategy.portfolio import PortfolioConstraints, normalize_weights, volatility_scaled_weights
from src.strategy import signals

CONSTRAINTS = PortfolioConstraints(max_asset_weight=0.35, max_gross_exposure=1.0, cash_weight_floor=0.0)


def test_large_raw_weights_keep_their_ordering():
    # signal/vol raw weights are typically >> 1; capping before scaling used
    # to flatten these to [1/3, 1/3, 1/3].
    w = normalize_weights(pd.Series([50.0, 10.0, 2.0, 1.0], index=list("abcd")), CONSTRAINTS)
    assert w.sum() == pytest.approx(1.0)
    assert (w <= 0.35 + 1e-12).all()
    assert w["a"] == pytest.approx(0.35)
    assert w["b"] > w["c"] > w["d"] > 0


def test_excess_left_in_cash_when_too_few_assets_to_fill_budget():
    w = normalize_weights(pd.Series([5.0, 5.0], index=list("ab")), CONSTRAINTS)
    assert w.tolist() == pytest.approx([0.35, 0.35])


def test_small_raw_weights_are_not_inflated():
    w = normalize_weights(pd.Series([0.1, 0.05], index=list("ab")), CONSTRAINTS)
    assert w.tolist() == pytest.approx([0.1, 0.05])


def test_long_only_drops_negative_weights():
    w = normalize_weights(pd.Series([3.0, -2.0, 1.0], index=list("abc")), CONSTRAINTS)
    assert w["b"] == 0.0
    # [3, 0, 1] -> scaled [0.75, 0, 0.25] -> a capped at 0.35, its excess
    # moves to c (0.65), which is then capped too; the rest stays in cash.
    assert w.tolist() == pytest.approx([0.35, 0.0, 0.35])


def test_cash_floor_reduces_budget():
    constraints = PortfolioConstraints(max_asset_weight=1.0, max_gross_exposure=1.0, cash_weight_floor=0.2)
    w = normalize_weights(pd.Series([3.0, 1.0], index=list("ab")), constraints)
    assert w.sum() == pytest.approx(0.8)


def test_zero_volatility_assets_excluded():
    raw = volatility_scaled_weights(pd.Series([1.0, 1.0], index=list("ab")), pd.Series([0.0, 0.5], index=list("ab")))
    assert raw.index.tolist() == ["b"]


def test_mean_reversion_never_levered_and_sizes_by_intensity():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=400, freq="h")
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0, 0.02, 400)) for s in "ABCDEFGH"}, index=idx)
    w = signals.mean_reversion(close, lookback=12, z_entry=1.5)
    assert (w.sum(axis=1) <= 1.0 + 1e-12).all()
    assert (w >= 0).all().all()
    # Not a binary 0/1 signal any more.
    assert len(np.unique(np.round(w.to_numpy()[w.to_numpy() > 0], 6))) > 2
