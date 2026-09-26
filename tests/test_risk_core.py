"""Drawdown state machine, trend hysteresis, the trend_vol_target core, and
the engine's risk-overlay hook."""
import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import BacktestEngine
from src.features.trend import trend_state
from src.risk.drawdown import DrawdownRiskManager, RiskState
from src.strategy import signals

FREE = CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0)


# -- drawdown state machine ------------------------------------------------------

def test_escalates_immediately_through_states():
    rm = DrawdownRiskManager(cooldown_bars=10)
    rm.reset(100.0)
    assert rm.update(101.0) == 1.0 and rm.state is RiskState.NORMAL
    assert rm.update(95.0) == 0.75 and rm.state is RiskState.CAUTION  # -5.9% from 101
    assert rm.update(90.0) == 0.5 and rm.state is RiskState.DEFENSIVE
    assert rm.update(80.0) == 0.0 and rm.state is RiskState.EMERGENCY


def test_can_jump_straight_to_emergency():
    rm = DrawdownRiskManager()
    rm.reset(100.0)
    assert rm.update(70.0) == 0.0


def test_recovers_one_level_per_cooldown_even_when_flat_in_cash():
    rm = DrawdownRiskManager(cooldown_bars=3)
    rm.reset(100.0)
    rm.update(75.0)  # EMERGENCY, in cash: equity can't recover by itself
    states = []
    for _ in range(9):
        rm.update(75.0)
        states.append(rm.state)
    assert states[2] is RiskState.DEFENSIVE
    assert states[5] is RiskState.CAUTION
    assert states[8] is RiskState.NORMAL


def test_re_escalates_after_rearming_if_losses_continue():
    rm = DrawdownRiskManager(cooldown_bars=2)
    rm.reset(100.0)
    rm.update(89.0)  # DEFENSIVE
    rm.update(89.0)
    rm.update(89.0)  # cooldown -> CAUTION, peak re-armed at 89
    assert rm.state is RiskState.CAUTION
    rm.update(80.0)  # -10.1% from the re-armed peak
    assert rm.state is RiskState.DEFENSIVE


@pytest.mark.parametrize(
    "kwargs",
    [dict(caution=0.1, defensive=0.05), dict(caution_exposure=0.3, defensive_exposure=0.6), dict(cooldown_bars=0)],
)
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        DrawdownRiskManager(**kwargs)


# -- trend hysteresis --------------------------------------------------------------

def test_trend_state_band_prevents_whipsaw():
    idx = pd.date_range("2025-01-01", periods=60, freq="h")
    close = pd.Series(100.0, index=idx)
    close.iloc[30:] = 100.0 + np.tile([0.5, -0.5], 15)  # hovering around the EMA
    no_band = trend_state(close, span=10, band=0.0)
    banded = trend_state(close, span=10, band=0.02)
    assert no_band.iloc[30:].diff().abs().sum() > 5
    assert banded.iloc[30:].diff().abs().sum() == 0


def test_trend_state_switches_on_real_moves():
    idx = pd.date_range("2025-01-01", periods=80, freq="h")
    close = pd.Series(np.r_[np.full(20, 100.0), np.linspace(100, 130, 30), np.linspace(130, 90, 30)], index=idx)
    state = trend_state(close, span=10, band=0.01)
    assert state.iloc[45] == 1.0
    assert state.iloc[-1] == 0.0


# -- core strategy ------------------------------------------------------------------

def _prices(seed=0, n=600):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    return pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)) for s in ["BTC/USD", "ETH/USD", "ALT/USD"]}, index=idx)


def test_trend_vol_target_only_holds_its_assets_and_never_levers():
    w = signals.trend_vol_target(_prices(), assets=("BTC/USD", "ETH/USD"), trend_span=48, vol_lookback=48, target_vol=5.0)
    assert (w["ALT/USD"] == 0).all()
    assert (w.sum(axis=1) <= 1.0 + 1e-12).all()
    assert (w >= 0).all().all()


def test_trend_vol_target_scales_with_target_vol():
    close = _prices()
    low = signals.trend_vol_target(close, trend_span=48, vol_lookback=48, target_vol=0.05)
    high = signals.trend_vol_target(close, trend_span=48, vol_lookback=48, target_vol=0.10)
    active = low.sum(axis=1) > 0
    ratio = high.sum(axis=1)[active] / low.sum(axis=1)[active]
    assert ratio.median() == pytest.approx(2.0, rel=0.01)


def test_trend_vol_target_is_causal():
    close = _prices()
    full = signals.trend_vol_target(close, trend_span=48, vol_lookback=48)
    mutated = close.copy()
    mutated.iloc[400:] *= 100.0
    pd.testing.assert_frame_equal(full.iloc[:400], signals.trend_vol_target(mutated, trend_span=48, vol_lookback=48).iloc[:400])


# -- engine overlay ------------------------------------------------------------------

def test_overlay_cuts_exposure_after_drawdown_and_applies_next_bar():
    idx = pd.date_range("2025-01-01", periods=8, freq="h")
    close = pd.DataFrame({"A": [100, 100, 100, 75, 75, 75, 75, 75.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0] * 8}, index=idx)
    overlay = DrawdownRiskManager(cooldown_bars=100)
    result = BacktestEngine(FREE, risk_overlay=overlay).run(close, weights)
    # Fully invested from bar 1; the -25% at bar 3 is taken at full size (no
    # look-ahead), then EMERGENCY moves to cash from bar 4.
    assert result.exposure.iloc[3] == 1.0
    assert result.exposure.iloc[4] == 0.0
    assert result.weights_history["A"].iloc[4:].eq(0.0).all()
    assert result.portfolio_value.iloc[-1] == pytest.approx(75_000.0)


def test_engine_without_overlay_reports_no_exposure():
    idx = pd.date_range("2025-01-01", periods=3, freq="h")
    close = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=idx)
    assert BacktestEngine(FREE).run(close, pd.DataFrame({"A": [0.5] * 3}, index=idx)).exposure is None
