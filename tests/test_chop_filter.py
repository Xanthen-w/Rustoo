"""Efficiency ratio and the choppy-market entry filter."""
import numpy as np
import pandas as pd
import pytest

from src.features.trend import confirmed_trend_state, efficiency_ratio, trend_state
from src.strategy import signals


def hours(n):
    return pd.date_range("2025-01-01", periods=n, freq="h")


def test_efficiency_ratio_extremes():
    n = 24 * 40
    straight = pd.Series(np.linspace(100, 200, n), index=hours(n))
    er = efficiency_ratio(straight, 24 * 10)
    assert er.dropna().between(0.999, 1.001).all()
    zigzag = pd.Series(100 + 10 * np.sign(np.sin(np.arange(n) * 2 * np.pi / 48)), index=hours(n))
    assert efficiency_ratio(zigzag, 24 * 10).dropna().max() < 0.2
    assert efficiency_ratio(straight, 24 * 10).iloc[: 24 * 10].isna().all()  # warm-up


def test_confirmed_state_blocks_entry_but_not_exit():
    n = 400
    close = pd.Series(np.r_[np.full(100, 100.0), np.full(150, 110.0), np.full(150, 90.0)], index=hours(n))
    ok = pd.Series(1.0, index=close.index)
    blocked = pd.Series(0.0, index=close.index)
    base = trend_state(close, 20, 0.02)
    assert confirmed_trend_state(close, 20, 0.02, ok, 0.5).equals(base)
    assert (confirmed_trend_state(close, 20, 0.02, blocked, 0.5) == 0).all()  # never enters
    # entered while confirmed, then exits on price even though confirm drops
    confirm = pd.Series(np.r_[np.ones(200), np.zeros(200)], index=close.index)
    st = confirmed_trend_state(close, 20, 0.02, confirm, 0.5)
    assert st.iloc[150] == 1.0 and st.iloc[-1] == 0.0


def test_zero_threshold_reproduces_live_strategy():
    rng = np.random.default_rng(0)
    n = 24 * 60
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for s in ["BTC/USD", "ETH/USD"]},
                         index=hours(n))
    common = dict(trend_span=96, band=0.03, vol_lookback=72, target_vol=0.5, min_exposure=0.15)
    pd.testing.assert_frame_equal(signals.trend_vol_chop_filter(close, er_window=240, er_min=0.0, **common),
                                  signals.trend_vol_target(close, **common))


def test_filter_reduces_entries_in_chop():
    rng = np.random.default_rng(1)
    n = 24 * 120
    chop = 100 * (1 + 0.06 * np.sin(np.arange(n) * 2 * np.pi / (24 * 12))) * (1 + rng.normal(0, 0.003, n))
    close = pd.DataFrame({"BTC/USD": chop}, index=hours(n))
    common = dict(assets=("BTC/USD",), trend_span=96, band=0.01, vol_lookback=72, target_vol=0.5, min_exposure=0.15)
    live = signals.trend_vol_target(close, **common)["BTC/USD"]
    filt = signals.trend_vol_chop_filter(close, er_window=24 * 10, er_min=0.5, **common)["BTC/USD"]
    flips = lambda w: int((w > w.max() * 0.5).astype(int).diff().abs().sum())  # noqa: E731
    assert flips(filt) < flips(live)
