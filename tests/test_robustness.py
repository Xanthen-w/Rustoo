"""Step 3 of the backtester upgrade: robustness analyses."""
import json

import numpy as np
import pandas as pd
import pytest

from backtest import robustness as rb
from backtest.costs import CostModel
from backtest.robustness_report import run_robustness
from src.strategy import signals

BASE = CostModel(taker_fee=0.001, maker_fee=0.0005, slippage_bps=5.0)
PARAMS = {"assets": ("BTC/USD", "ETH/USD"), "trend_span": 48, "vol_lookback": 48, "target_vol": 0.5,
          "band": 0.02, "min_exposure": 0.15}
ENGINE = {"execution_price": "open", "execution_lag": 1, "rebalance_threshold": 0.05, "rebalance_hours_utc": (0,)}


def make_spec(seed=0, n=24 * 90, drift=0.0003):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01 01:00", periods=n, freq="h", tz="UTC")
    cols = ["BTC/USD", "ETH/USD"]
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(drift, 0.008, (n, 2)), axis=0), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.DataFrame(rng.uniform(1e3, 1e4, (n, 2)), index=idx, columns=cols)
    return rb.RunSpec(close, open_, volume, idx[60:], signals.trend_vol_target, dict(PARAMS), BASE, dict(ENGINE))


def test_breakeven_interpolates_zero_crossing():
    t = pd.DataFrame({"bps": [0, 10, 20], "net_return": [0.04, 0.02, -0.02]})
    assert rb.breakeven(t) == pytest.approx(15.0)
    assert rb.breakeven(pd.DataFrame({"bps": [0, 10], "net_return": [0.05, 0.01]})) is None
    # unprofitable before any cost: not a "0 bps breakeven"
    assert np.isnan(rb.breakeven(pd.DataFrame({"bps": [0, 10], "net_return": [-0.01, -0.02]})))


def test_higher_costs_never_help():
    spec = make_spec()
    for knob in ("fee_bps", "slippage_bps", "spread_bps"):
        t = rb.cost_sensitivity(spec, knob, [0, 10, 50])
        assert (np.diff(t.net_return.to_numpy()) < 0).all(), knob
        assert (np.diff(t.total_costs.to_numpy()) > 0).all(), knob


def test_zero_cost_sweep_equals_gross():
    spec = make_spec(1)
    zero = rb.cost_sensitivity(spec, "fee_bps", [0]).iloc[0]
    free = spec.run(cost_model=CostModel(taker_fee=0, maker_fee=0, slippage_bps=5.0))
    assert zero.net_return == pytest.approx(free.portfolio_value.iloc[-1] / 1e5 - 1)


def test_stress_base_row_matches_plain_run_and_scenarios_bite():
    spec = make_spec(2)
    table = rb.stress_test(spec)
    base = spec.run()
    assert table.iloc[0].net_return == pytest.approx(base.portfolio_value.iloc[-1] / 1e5 - 1)
    row = table.set_index("scenario")
    assert row.loc["3x fees", "total_costs"] > row.loc["2x fees", "total_costs"] > row.loc["Base", "total_costs"]
    assert row.loc["Market impact, 10% of liquidity", "total_costs"] > row.loc["Market impact, normal liquidity", "total_costs"]
    assert row.loc["Volatility x1.5 (synthetic prices)", "synthetic"]


def test_latency_scenario_delays_fills():
    spec = make_spec(3)
    base = spec.run()
    late = spec.run(engine_overrides={"execution_lag": 3})
    assert late.fills.timestamp.min() > base.fills.timestamp.min()


def test_scaled_returns_scale_volatility():
    spec = make_spec(4)
    scaled = rb._scaled_returns(spec.close, 1.5)
    ratio = scaled.pct_change().std() / spec.close.pct_change().std()
    assert np.allclose(ratio, 1.5, rtol=1e-6)
    assert np.allclose(scaled.iloc[0], spec.close.iloc[0])


def test_monte_carlo_is_seeded_and_consistent():
    spec = make_spec(5)
    base = spec.run()
    bench = spec.close["BTC/USD"] / spec.close["BTC/USD"].loc[spec.eval_index[0]] * 1e5
    cfg = rb.MonteCarloConfig(simulations=300, seeds=(1, 2), horizon_days=7, block_hours=24)
    a = rb.monte_carlo(base, 8766, cfg, bench.loc[spec.eval_index])
    b = rb.monte_carlo(base, 8766, cfg, bench.loc[spec.eval_index])
    pd.testing.assert_frame_equal(a["simulations"], b["simulations"])
    assert len(a["simulations"]) == 600
    p = a["summary"]["percentiles"]
    assert p["p5"]["total_return"] <= p["p50"]["total_return"] <= p["p95"]["total_return"]
    assert (a["simulations"].max_drawdown <= 0).all()
    assert 0 <= a["summary"]["p_beats_benchmark"] <= 1
    assert len(a["fan"]) == 7 * 24


def test_monte_carlo_with_no_cost_noise_reproduces_path_math():
    """One block covering the whole horizon from a fixed start must equal
    the run's actual equity growth over those bars."""
    spec = make_spec(6)
    base = spec.run()
    prev = base.portfolio_value.shift(1).fillna(1e5)
    r = (base.gross_pnl - base.fees_per_period) / prev
    assert np.allclose((1 + r).cumprod().iloc[-1], base.portfolio_value.iloc[-1] / 1e5)


def test_regimes_label_a_clear_bull_and_bear():
    idx = pd.date_range("2025-01-01", periods=24 * 120, freq="h", tz="UTC")
    up = np.linspace(100, 200, len(idx) // 2)
    down = np.linspace(200, 100, len(idx) - len(idx) // 2)
    close = pd.Series(np.r_[up, down] * (1 + np.random.default_rng(0).normal(0, 0.001, len(idx))), index=idx)
    labels = rb.classify_regimes(close, 8766, rb.RegimeConfig(lookback_days=20, trend_threshold=0.05))
    assert labels.trend.iloc[len(idx) // 2 - 10] == "bull"
    assert labels.trend.iloc[-10] == "bear"
    assert labels.trend.iloc[:100].isna().all()  # not enough history yet


def test_landscape_covers_grid_and_scores_plateau():
    spec = make_spec(7)
    grid = rb.parameter_landscape(spec, "trend_span", [24, 48, 96], "target_vol", [0.3, 0.5], window_days=7)
    assert len(grid) == 6 and {"window_mean_return", "sharpe"} <= set(grid.columns)
    score = rb.plateau_score(grid, "trend_span", "target_vol", "sharpe")
    assert score["best"] >= score["neighbour_mean"] and score["best"] >= score["grid_median"]


def test_random_entry_percentiles_are_fractions():
    res = rb.random_entry_test(make_spec(8), seeds=[0, 1, 2, 3])
    for k in ("return_percentile", "sharpe_percentile", "composite_percentile"):
        assert 0.0 <= res[k] <= 1.0
    assert len(res["random"]) == 4


def test_robustness_report_writes_outputs(tmp_path):
    spec = make_spec(9)
    summary = run_robustness(
        spec, tmp_path, title="t", period_label="synthetic",
        cost_grid_bps={"fee_bps": [0, 10, 50]},
        mc=rb.MonteCarloConfig(simulations=100, seeds=(1,), horizon_days=7, block_hours=24),
        landscapes={"synthetic": (spec, [("trend_span", [24, 48], "target_vol", [0.3, 0.5])])},
        random_entry={"synthetic": spec}, random_seeds=[0, 1],
    )
    for name in ["robustness.html", "summary.json", "cost_sensitivity.csv", "stress.csv", "monte_carlo_per_seed.csv",
                 "regimes.csv", "random_entry_synthetic.csv"]:
        assert (tmp_path / name).exists(), name
    page = (tmp_path / "robustness.html").read_text()
    for anchor in ["costs", "stress", "mc", "regimes", "landscape", "random", "repro"]:
        assert f"id={anchor}" in page
    assert json.loads((tmp_path / "summary.json").read_text())["base"]["net_return"] == pytest.approx(summary["base"]["net_return"])
