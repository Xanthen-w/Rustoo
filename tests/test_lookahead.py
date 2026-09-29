"""Look-ahead bias catchers (Phase 6 requirement): every strategy's output at
time t must be identical regardless of what happens after t. We test this by
computing weights on a full price series, then truncating the tail and
recomputing — the overlapping prefix must match exactly.
"""
import numpy as np
import pandas as pd
import pytest

from src.strategy import signals

np.random.seed(0)
N = 300
DATES = pd.date_range("2025-01-01", periods=N, freq="h")
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]


def _random_walk_prices() -> pd.DataFrame:
    data = {}
    for symbol in SYMBOLS:
        steps = np.random.normal(loc=0.0002, scale=0.01, size=N)
        data[symbol] = 100 * np.cumprod(1 + steps)
    return pd.DataFrame(data, index=DATES)


CLOSE = _random_walk_prices()
CUTOFF = 200  # truncate here; then mutate the future tail wildly


@pytest.mark.parametrize(
    "strategy_fn",
    [
        lambda df: signals.equal_weight(df),
        lambda df: signals.trend_following(df, fast_span=10, slow_span=30),
        lambda df: signals.single_asset_momentum(df, "BTC/USD", lookback=24),
        lambda df: signals.cross_sectional_momentum(df, lookback=24, top_k=2),
        lambda df: signals.mean_reversion(df, lookback=12, z_entry=1.0),
        lambda df: signals.volatility_filtered_momentum(df, momentum_lookback=24, vol_lookback=24),
        lambda df: signals.trend_vol_target(df, assets=("BTC/USD", "ETH/USD"), trend_span=24, vol_lookback=24),
        lambda df: signals.trend_vol_long_short(df, assets=("BTC/USD", "ETH/USD"), trend_span=24, vol_lookback=24,
                                                short_entry=0.02),
        lambda df: signals.trend_vol_chop_filter(df, assets=("BTC/USD", "ETH/USD"), trend_span=24, vol_lookback=24,
                                                 er_window=48, er_min=0.2),
    ],
)
def test_strategy_output_unaffected_by_future_data(strategy_fn):
    full_weights = strategy_fn(CLOSE)

    mutated = CLOSE.copy()
    # Blow up the future tail with an extreme shock — if any strategy is
    # accidentally reading ahead, this will change its past output.
    mutated.iloc[CUTOFF:] = mutated.iloc[CUTOFF:] * 100.0
    mutated_weights = strategy_fn(mutated)

    prefix_full = full_weights.iloc[:CUTOFF]
    prefix_mutated = mutated_weights.iloc[:CUTOFF]

    pd.testing.assert_frame_equal(prefix_full, prefix_mutated)


def test_buy_and_hold_unaffected_by_future_data():
    full_weights = signals.buy_and_hold(CLOSE, "BTC/USD")
    mutated = CLOSE.copy()
    mutated.iloc[CUTOFF:] = mutated.iloc[CUTOFF:] * 100.0
    mutated_weights = signals.buy_and_hold(mutated, "BTC/USD")
    pd.testing.assert_frame_equal(full_weights.iloc[:CUTOFF], mutated_weights.iloc[:CUTOFF])


def test_backtest_engine_execution_lag_enforced():
    from backtest.costs import BASE
    from backtest.engine import BacktestEngine

    with pytest.raises(ValueError):
        BacktestEngine(cost_model=BASE, execution_lag=0)
