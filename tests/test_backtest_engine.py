import numpy as np
import pandas as pd

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
