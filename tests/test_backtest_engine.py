import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import BacktestEngine


def test_flat_price_zero_signal_preserves_capital_minus_fees():
    dates = pd.date_range("2025-01-01", periods=10, freq="h")
    close = pd.DataFrame({"BTC/USD": [100.0] * 10}, index=dates)
    weights = pd.DataFrame({"BTC/USD": [0.0] * 10}, index=dates)

    engine = BacktestEngine(cost_model=CostModel(taker_fee=0.001, maker_fee=0.0005, slippage_bps=0), initial_capital=1000.0)
    result = engine.run(close, weights)

    assert result.total_fees == 0.0
    assert np.allclose(result.portfolio_value.values, 1000.0)


def test_full_allocation_tracks_price_after_lag():
    dates = pd.date_range("2025-01-01", periods=5, freq="h")
    close = pd.DataFrame({"BTC/USD": [100.0, 110.0, 121.0, 133.1, 146.41]}, index=dates)
    weights = pd.DataFrame({"BTC/USD": [1.0, 1.0, 1.0, 1.0, 1.0]}, index=dates)

    engine = BacktestEngine(cost_model=CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0), initial_capital=1000.0, execution_lag=1)
    result = engine.run(close, weights)

    # weight of 1.0 decided at t=0 only takes effect at t=1 (execution_lag=1),
    # so portfolio_value[0] should still be flat cash (1000), and only start
    # tracking price growth from t=1 onward.
    assert result.portfolio_value.iloc[0] == 1000.0
    growth_from_1 = close["BTC/USD"].iloc[2] / close["BTC/USD"].iloc[1]
    assert np.isclose(result.portfolio_value.iloc[2] / result.portfolio_value.iloc[1], growth_from_1)


def test_fees_reduce_portfolio_value():
    dates = pd.date_range("2025-01-01", periods=3, freq="h")
    close = pd.DataFrame({"BTC/USD": [100.0, 100.0, 100.0]}, index=dates)
    weights = pd.DataFrame({"BTC/USD": [1.0, 1.0, 1.0]}, index=dates)

    engine = BacktestEngine(cost_model=CostModel(taker_fee=0.01, maker_fee=0.01, slippage_bps=0.0), initial_capital=1000.0)
    result = engine.run(close, weights)

    assert result.total_fees > 0.0
    assert result.portfolio_value.iloc[-1] < 1000.0


def test_no_symbol_ever_priced_after_its_nan_start():
    dates = pd.date_range("2025-01-01", periods=4, freq="h")
    close = pd.DataFrame(
        {"BTC/USD": [100.0, 101.0, 102.0, 103.0], "NEW/USD": [np.nan, np.nan, 10.0, 11.0]},
        index=dates,
    )
    weights = pd.DataFrame(
        {"BTC/USD": [0.5, 0.5, 0.5, 0.5], "NEW/USD": [0.5, 0.5, 0.5, 0.5]}, index=dates
    )
    engine = BacktestEngine(cost_model=CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0), initial_capital=1000.0)
    result = engine.run(close, weights)
    # NEW/USD had no price for bars 0-1, so realized weight there must be 0
    assert result.weights_history["NEW/USD"].iloc[:2].eq(0.0).all()


def test_missing_price_bar_is_not_a_loss():
    # A NaN bar means "can't trade this bar", not "position is worth 0".
    dates = pd.date_range("2025-01-01", periods=5, freq="h")
    close = pd.DataFrame({"BTC/USD": [100.0, 100.0, np.nan, 100.0, 100.0]}, index=dates)
    weights = pd.DataFrame({"BTC/USD": [1.0] * 5}, index=dates)
    engine = BacktestEngine(cost_model=CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0), initial_capital=1000.0)
    result = engine.run(close, weights)
    assert np.allclose(result.portfolio_value.values, 1000.0)
    assert result.weights_history["BTC/USD"].iloc[2] == pytest.approx(1.0)
    assert result.trade_notional_history["BTC/USD"].iloc[2] == 0.0


def test_position_frozen_during_gap_keeps_other_targets_within_budget():
    dates = pd.date_range("2025-01-01", periods=4, freq="h")
    close = pd.DataFrame(
        {"A": [100.0, 200.0, np.nan, 200.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=dates
    )
    weights = pd.DataFrame({"A": [0.5] * 4, "B": [0.5] * 4}, index=dates)
    engine = BacktestEngine(cost_model=CostModel(taker_fee=0.001, maker_fee=0.001, slippage_bps=0.0), initial_capital=1000.0)
    result = engine.run(close, weights)
    held = (result.weights_history.sum(axis=1)).to_numpy()
    assert (held <= 1.0 + 1e-9).all()  # never more invested than equity


@pytest.mark.parametrize("taker_fee", [0.001, 0.01])
def test_full_allocation_never_borrows_to_pay_fees(taker_fee):
    dates = pd.date_range("2025-01-01", periods=6, freq="h")
    close = pd.DataFrame({"A": [100.0, 101.0, 99.0, 102.0, 98.0, 100.0], "B": [50.0, 49.0, 51.0, 50.0, 52.0, 51.0]}, index=dates)
    weights = pd.DataFrame({"A": [0.6, 0.4, 0.7, 0.3, 0.5, 0.5], "B": [0.4, 0.6, 0.3, 0.7, 0.5, 0.5]}, index=dates)
    engine = BacktestEngine(cost_model=CostModel(taker_fee=taker_fee, maker_fee=taker_fee, slippage_bps=5.0), initial_capital=1000.0)
    result = engine.run(close, weights)
    invested = result.weights_history.sum(axis=1)
    assert (invested <= 1.0 + 1e-9).all()
    cash = result.portfolio_value * (1.0 - invested)
    assert (cash >= -1e-9).all()


def test_leveraged_target_weights_rejected():
    dates = pd.date_range("2025-01-01", periods=3, freq="h")
    close = pd.DataFrame({"A": [1.0, 1.0, 1.0], "B": [1.0, 1.0, 1.0]}, index=dates)
    weights = pd.DataFrame({"A": [0.8] * 3, "B": [0.8] * 3}, index=dates)
    with pytest.raises(ValueError, match="leverage"):
        BacktestEngine(cost_model=CostModel()).run(close, weights)


def test_negative_target_weights_rejected():
    dates = pd.date_range("2025-01-01", periods=3, freq="h")
    close = pd.DataFrame({"A": [1.0, 1.0, 1.0]}, index=dates)
    weights = pd.DataFrame({"A": [-0.5] * 3}, index=dates)
    with pytest.raises(ValueError, match="shorting"):
        BacktestEngine(cost_model=CostModel()).run(close, weights)


def test_size_orders_with_all_positions_frozen_and_negative_rounding_budget():
    # Real failure from walk-forward: fully invested, every held asset inside
    # the rebalance band, cash 0, budget = -1.5e-11 from float rounding, and a
    # new asset wanting to enter. Used to divide by zero and poison equity.
    from backtest.engine import size_orders

    current = np.array([43955.2370953610, 43955.2370953610, 0.0])
    equity = current.sum() - 1.4551915228366852e-11
    tradable = np.array([False, False, True])
    target = np.array([0.45, 0.45, 0.10])
    with np.errstate(all="raise"):
        desired = size_orders(target, current, tradable, equity, cost_rate=0.0015)
    assert np.isfinite(desired).all()
    assert desired[2] == 0.0  # nothing left to spend
    assert desired[:2].tolist() == current[:2].tolist()


@pytest.mark.parametrize("seed", range(40))
def test_engine_stays_finite_and_unlevered_under_random_targets(seed):
    rng = np.random.default_rng(seed)
    n, m = 120, 6
    dates = pd.date_range("2025-01-01", periods=n, freq="h")
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, m)), axis=0), index=dates, columns=list("ABCDEF"))
    close.iloc[rng.integers(0, n, 10), rng.integers(0, m, 10)] = np.nan  # scattered gaps
    raw = rng.random((n, m)) * (rng.random((n, m)) > 0.4)
    weights = pd.DataFrame(raw / np.maximum(raw.sum(axis=1, keepdims=True), 1.0), index=dates, columns=close.columns)
    engine = BacktestEngine(cost_model=CostModel(taker_fee=0.001, maker_fee=0.001, slippage_bps=5.0),
                            rebalance_threshold=float(rng.choice([0.0, 0.02, 0.05, 0.1])))
    with np.errstate(all="raise"):
        result = engine.run(close, weights)
    assert np.isfinite(result.portfolio_value).all()
    assert (result.weights_history.sum(axis=1) <= 1.0 + 1e-9).all()
