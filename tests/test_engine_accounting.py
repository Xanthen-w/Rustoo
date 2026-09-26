"""Engine accounting (step 1 of the backtester upgrade): gross vs net
reconciliation, cost decomposition, next-bar-open execution, fills ledger,
and engine-level causality."""
import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import BacktestEngine

FREE = CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0)
COSTLY = CostModel(taker_fee=0.001, maker_fee=0.0005, slippage_bps=5.0, maker_fill_probability=0.3,
                   spread_bps=4.0, impact_coef=0.02, impact_alpha=0.5)


def market(seed=0, n=300, m=4, gaps=True):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01 01:00", periods=n, freq="h", tz="UTC")
    cols = [f"A{j}" for j in range(m)]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (n, m)), axis=0), index=idx, columns=cols)
    open_ = close.shift(1) * (1 + rng.normal(0, 0.001, (n, m)))
    open_.iloc[0] = close.iloc[0]
    volume = pd.DataFrame(rng.uniform(50, 500, (n, m)), index=idx, columns=cols)
    if gaps:
        for _ in range(8):
            i, j = rng.integers(1, n), rng.integers(0, m)
            close.iat[i, j] = np.nan
            open_.iat[i, j] = np.nan
    raw = rng.random((n, m)) * (rng.random((n, m)) > 0.3)
    weights = pd.DataFrame(raw / np.maximum(raw.sum(axis=1, keepdims=True), 1.0), index=idx, columns=cols)
    return close, open_, volume, weights


@pytest.mark.parametrize("execution_price", ["close", "open"])
@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("threshold", [0.0, 0.05])
def test_gross_minus_costs_equals_net(execution_price, seed, threshold):
    close, open_, volume, weights = market(seed)
    engine = BacktestEngine(COSTLY, rebalance_threshold=threshold, execution_price=execution_price,
                            rebalance_hours_utc=(0,))
    r = engine.run(close, weights, open_wide=open_, volume_wide=volume)
    net = r.portfolio_value.iloc[-1] - r.initial_capital
    assert r.gross_pnl.sum() - r.costs.to_numpy().sum() == pytest.approx(net, abs=1e-6)
    assert r.gross_value.iloc[-1] - r.portfolio_value.iloc[-1] == pytest.approx(r.costs.to_numpy().sum(), abs=1e-6)
    # component columns add up to the per-bar total, and the fills ledger to both
    assert np.allclose(r.costs.sum(axis=1), r.fees_per_period)
    assert r.fills["cost"].sum() == pytest.approx(r.total_fees, abs=1e-6)
    for c in ["fee", "spread", "slippage", "impact"]:
        assert r.fills[c].sum() == pytest.approx(r.costs[c].sum(), abs=1e-6)
    assert (r.costs.to_numpy() >= 0).all()


def test_zero_costs_make_gross_and_net_identical():
    close, open_, volume, weights = market(1)
    r = BacktestEngine(FREE, execution_price="open").run(close, weights, open_wide=open_)
    assert np.allclose(r.gross_value, r.portfolio_value)


def test_cost_components_match_their_definitions():
    idx = pd.date_range("2025-01-01", periods=3, freq="h")
    close = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=idx)
    volume = pd.DataFrame({"A": [1_000.0] * 3}, index=idx)  # $100k traded per bar
    weights = pd.DataFrame({"A": [0.5, 0.5, 0.5]}, index=idx)
    cm = CostModel(taker_fee=0.001, maker_fee=0.001, slippage_bps=10, spread_bps=20, impact_coef=0.01, impact_alpha=1.0)
    r = BacktestEngine(cm, initial_capital=100_000).run(close, weights, volume_wide=volume)
    fill = r.fills.iloc[0]
    notional = fill["notional"]
    assert fill["fee"] == pytest.approx(notional * 0.001)
    assert fill["spread"] == pytest.approx(notional * 0.001)  # half of 20 bps
    assert fill["slippage"] == pytest.approx(notional * 0.001)
    # participation of the *intended* $50k order in a $100k bar = 0.5 -> impact 0.5%
    assert fill["participation"] == pytest.approx(0.5)
    assert fill["impact"] == pytest.approx(notional * 0.01 * 0.5)


def test_impact_without_volume_is_refused():
    close, open_, volume, weights = market(2)
    with pytest.raises(ValueError, match="volume"):
        BacktestEngine(COSTLY).run(close, weights)


def test_next_bar_open_execution_fills_at_the_open():
    idx = pd.date_range("2025-01-01", periods=4, freq="h")
    close = pd.DataFrame({"A": [100.0, 110.0, 120.0, 130.0]}, index=idx)
    open_ = pd.DataFrame({"A": [100.0, 101.0, 111.0, 121.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 1.0, 1.0, 1.0]}, index=idx)
    r = BacktestEngine(FREE, execution_price="open", initial_capital=1_000).run(close, weights, open_wide=open_)
    first = r.fills.iloc[0]
    assert first["timestamp"] == idx[1] and first["price"] == 101.0  # signal at bar 0 -> open of bar 1
    assert first["quantity"] == pytest.approx(1_000 / 101.0)
    assert r.portfolio_value.iloc[1] == pytest.approx(1_000 / 101.0 * 110.0)  # marked at bar 1 close


def test_latency_delays_the_fill_by_whole_bars():
    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    close = pd.DataFrame({"A": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=idx)
    weights = pd.DataFrame({"A": [1.0, 0.0, 0.0, 0.0, 0.0]}, index=idx)
    for lag in (1, 2, 3):
        r = BacktestEngine(FREE, execution_lag=lag).run(close, weights)
        assert r.fills.iloc[0]["timestamp"] == idx[lag]


def test_open_mode_requires_opens():
    close, _, _, weights = market(3)
    with pytest.raises(ValueError, match="open_wide"):
        BacktestEngine(FREE, execution_price="open").run(close, weights)


@pytest.mark.parametrize("execution_price", ["close", "open"])
def test_engine_output_up_to_bar_k_ignores_all_later_data(execution_price):
    """Mutating every price, volume and target after bar k must leave the
    equity curve, costs and fills through bar k unchanged."""
    close, open_, volume, weights = market(4, gaps=False)
    k = 150
    engine = BacktestEngine(COSTLY, execution_price=execution_price, rebalance_threshold=0.02)
    base = engine.run(close, weights, open_wide=open_, volume_wide=volume)
    c2, o2, v2, w2 = close.copy(), open_.copy(), volume.copy(), weights.copy()
    c2.iloc[k + 1:] *= 3.0
    o2.iloc[k + 1:] *= 0.2
    v2.iloc[k + 1:] *= 0.01
    w2.iloc[k + 1:] = w2.iloc[k + 1:].iloc[:, ::-1].to_numpy()
    mutated = engine.run(c2, w2, open_wide=o2, volume_wide=v2)
    pd.testing.assert_series_equal(base.portfolio_value.iloc[: k + 1], mutated.portfolio_value.iloc[: k + 1])
    pd.testing.assert_frame_equal(base.costs.iloc[: k + 1], mutated.costs.iloc[: k + 1])
    early = lambda f: f[f["timestamp"] <= close.index[k]].reset_index(drop=True)
    pd.testing.assert_frame_equal(early(base.fills), early(mutated.fills))


def test_simple_close_above_previous_close_strategy_cannot_see_its_fill_bar():
    """Mandatory look-ahead check: 'long if today's close > yesterday's close'.
    The decision at bar t uses closes <= t; the fill happens at bar t+1's open,
    and the size of that fill depends only on prices up to that open."""
    rng = np.random.default_rng(5)
    idx = pd.date_range("2025-01-01", periods=200, freq="h")
    close = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 200))}, index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    weights = (close > close.shift(1)).astype(float)
    r = BacktestEngine(FREE, execution_price="open").run(close, weights, open_wide=open_)
    for _, fill in r.fills.iterrows():
        t = idx.get_loc(fill["timestamp"])
        decided = weights["A"].iloc[t - 1]  # the signal from the previous bar's close
        assert (fill["side"] == "BUY") == (decided == 1.0)
        assert fill["price"] == open_["A"].iloc[t]


def test_no_dust_fills_when_target_is_unchanged():
    idx = pd.date_range("2025-01-01", periods=50, freq="h")
    rng = np.random.default_rng(6)
    close = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))}, index=idx)
    r = BacktestEngine(FREE).run(close, pd.DataFrame({"A": [1.0] * 50}, index=idx))
    assert len(r.fills) == 1  # the entry only; holding 100% needs no rebalancing
