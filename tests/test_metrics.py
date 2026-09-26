import numpy as np
import pandas as pd
import pytest

from backtest import metrics


def test_cumulative_return_simple():
    pv = pd.Series([100.0, 110.0, 121.0])
    assert metrics.cumulative_return(pv) == pytest.approx(0.21)


def test_max_drawdown_known_series():
    # 100 -> 120 (peak) -> 90 -> 100: drawdown from 120 to 90 = -25%
    pv = pd.Series([100.0, 120.0, 90.0, 100.0])
    assert metrics.max_drawdown(pv) == pytest.approx(-0.25)


def test_max_drawdown_monotonic_up_is_zero():
    pv = pd.Series([100.0, 105.0, 110.0, 120.0])
    assert metrics.max_drawdown(pv) == pytest.approx(0.0)


def test_sharpe_ratio_zero_for_zero_vol_returns():
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    # zero std -> guarded to return 0.0 rather than raising/inf
    assert metrics.sharpe_ratio(returns, periods_per_year=252) == 0.0


def test_sharpe_ratio_positive_for_positive_mean_returns():
    np.random.seed(1)
    returns = pd.Series(np.random.normal(0.001, 0.01, 500))
    sharpe = metrics.sharpe_ratio(returns, periods_per_year=252)
    assert sharpe > 0


def test_sortino_ignores_upside_volatility():
    # Same mean, but one series has upside outliers only -> should not be
    # penalized by Sortino the way Sharpe would penalize it.
    np.random.seed(2)
    base = np.random.normal(0.001, 0.005, 500)
    upside_shock = base.copy()
    upside_shock[::50] += 0.05  # large positive-only shocks

    returns_base = pd.Series(base)
    returns_shock = pd.Series(upside_shock)

    sortino_base = metrics.sortino_ratio(returns_base, periods_per_year=252)
    sortino_shock = metrics.sortino_ratio(returns_shock, periods_per_year=252)
    sharpe_shock = metrics.sharpe_ratio(returns_shock, periods_per_year=252)

    # Sortino on the shocked (higher upside, same downside) series should be
    # at least as good, while Sharpe is dragged down by the added variance.
    assert sortino_shock >= sortino_base * 0.9
    assert sharpe_shock < sortino_shock


def test_composite_score_matches_formula():
    score = metrics.composite_score(sortino=1.0, sharpe=2.0, calmar=0.5)
    assert score == pytest.approx(0.4 * 1.0 + 0.3 * 2.0 + 0.3 * 0.5)


def test_turnover_zero_when_weights_unchanged():
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5], "B": [0.5, 0.5, 0.5]})
    t = metrics.turnover(weights)
    assert t.iloc[1:].eq(0.0).all()


def test_trade_stats_empty_series():
    stats = metrics.trade_stats(pd.Series(dtype=float))
    assert stats["num_trades"] == 0
    assert stats["win_rate"] == 0.0


def test_sortino_uses_downside_deviation_over_all_periods():
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    # downside deviation = sqrt(mean([0, 0.01^2, 0, 0.02^2])) = sqrt(0.000125)
    expected = returns.mean() / np.sqrt(0.000125) * np.sqrt(252)
    assert metrics.sortino_ratio(returns, periods_per_year=252) == pytest.approx(expected)


def test_sortino_single_losing_period_is_not_zero():
    # The old std-of-negatives version returned 0 with fewer than 2 losses.
    returns = pd.Series([0.01, 0.02, -0.01, 0.015])
    assert metrics.sortino_ratio(returns, periods_per_year=252) > 0
