"""1x collateralized shorts in the backtest engine (research only; the live
bot stays long-only until shorts are adopted and tested)."""
import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import BacktestEngine

FREE = CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0)
COSTLY = CostModel(taker_fee=0.001, maker_fee=0.0005, slippage_bps=5.0)


def frame(values, cols=("A",)):
    idx = pd.date_range("2025-01-01", periods=len(values), freq="h")
    return pd.DataFrame(values, index=idx, columns=list(cols))


def test_short_gains_when_price_falls():
    close = frame([100.0, 100.0, 90.0, 80.0])
    w = frame([-1.0] * 4)
    # Held without re-sizing (wide band): a plain 1x short.
    r = BacktestEngine(FREE, allow_short=True, rebalance_threshold=0.5).run(close, w)
    assert r.fills.iloc[0]["side"] == "SELL" and len(r.fills) == 1
    assert r.portfolio_value.iloc[2] == pytest.approx(110_000)  # 10% down -> +10%
    assert r.portfolio_value.iloc[3] == pytest.approx(120_000)
    # Rebalanced every bar to -100% of the (growing) equity: shorts are added.
    r = BacktestEngine(FREE, allow_short=True).run(close, w)
    assert r.portfolio_value.iloc[3] == pytest.approx(110_000 * (1 + 10 / 90))


def test_long_and_short_in_different_assets():
    close = frame([[100.0, 100.0], [100.0, 100.0], [100.0, 110.0]], ("A", "B"))
    w = frame([[0.5, -0.5]] * 3, ("A", "B"))
    r = BacktestEngine(FREE, allow_short=True).run(close, w)
    assert r.portfolio_value.iloc[-1] == pytest.approx(95_000)  # short B lost 10% of 50k


def test_shorts_refused_unless_enabled_and_gross_is_capped():
    close = frame([[1.0, 1.0]] * 3, ("A", "B"))
    with pytest.raises(ValueError, match="shorting is not allowed"):
        BacktestEngine(FREE).run(close, frame([[-0.5, 0.0]] * 3, ("A", "B")))
    with pytest.raises(ValueError, match="leverage"):
        BacktestEngine(FREE, allow_short=True).run(close, frame([[0.6, -0.6]] * 3, ("A", "B")))


@pytest.mark.parametrize("seed", range(5))
def test_reconciliation_and_no_leverage_with_shorts(seed):
    rng = np.random.default_rng(seed)
    n = 200
    close = frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, 3)), axis=0), ("A", "B", "C"))
    raw = rng.uniform(-1, 1, (n, 3)) * (rng.random((n, 3)) > 0.3)
    w = frame(raw / np.maximum(np.abs(raw).sum(axis=1, keepdims=True), 1.0), ("A", "B", "C"))
    engine = BacktestEngine(COSTLY, allow_short=True, rebalance_threshold=0.02, rebalance_hours_utc=(0,))
    with np.errstate(all="raise"):
        r = engine.run(close, w)
    assert r.gross_pnl.sum() - r.total_fees == pytest.approx(r.portfolio_value.iloc[-1] - 1e5, abs=1e-6)
    assert (r.weights_history.abs().sum(axis=1) <= 1.0 + 1e-6).all()


def test_long_only_behaviour_unchanged_by_flag():
    rng = np.random.default_rng(9)
    close = frame(100 * np.cumprod(1 + rng.normal(0, 0.01, (100, 2)), axis=0), ("A", "B"))
    w = frame(rng.dirichlet([1, 1], 100) * 0.9, ("A", "B"))
    a = BacktestEngine(COSTLY, rebalance_threshold=0.03).run(close, w)
    b = BacktestEngine(COSTLY, rebalance_threshold=0.03, allow_short=True).run(close, w)
    pd.testing.assert_series_equal(a.portfolio_value, b.portfolio_value)


def test_long_short_strategy_states():
    from src.strategy import signals
    n = 24 * 60
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    up = np.linspace(100, 160, n // 3)
    down = np.linspace(160, 80, n // 3)
    flat = np.full(n - 2 * (n // 3), 80.0) * (1 + 0.001 * np.sin(np.arange(n - 2 * (n // 3))))
    close = pd.DataFrame({"BTC/USD": np.r_[up, down, flat]}, index=idx)
    close["BTC/USD"] *= 1 + np.random.default_rng(0).normal(0, 0.002, n)
    common = dict(assets=("BTC/USD",), trend_span=48, band=0.03, vol_lookback=48, target_vol=0.5, min_exposure=0.15)
    w = signals.trend_vol_long_short(close, short_entry=0.05, short_scale=0.5, **common)["BTC/USD"]
    assert w.iloc[n // 3 - 5] > 0  # long in the uptrend
    assert w.iloc[2 * (n // 3) - 5] < 0  # short well into the downtrend
    assert (w.abs() <= 1 + 1e-12).all()
    long_only = signals.trend_vol_target(close, **common)
    in_trend = long_only["BTC/USD"] > long_only["BTC/USD"].max() * 0.5
    assert np.allclose(w[in_trend], long_only["BTC/USD"][in_trend])  # identical long side
