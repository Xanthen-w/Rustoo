"""Step 2 of the backtester upgrade: analytics, benchmarks, random-entry
baseline and the HTML report."""
import json

import numpy as np
import pandas as pd
import pytest

from backtest import analytics as an
from backtest.benchmarks import BENCHMARK_BAND, buy_and_hold_weights, exposure_profile, random_entry_weights
from backtest.costs import CostModel
from backtest.engine import BacktestEngine
from backtest.report import ReportConfig, run_report
from src.strategy import signals

COSTLY = CostModel(taker_fee=0.001, maker_fee=0.0005, slippage_bps=5.0, spread_bps=4.0)
FREE = CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0)


def market(seed=0, n=24 * 120, cols=("BTC/USD", "ETH/USD")):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01 01:00", periods=n, freq="h", tz="UTC")
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0.0002, 0.008, (n, len(cols))), axis=0), index=idx, columns=list(cols))
    open_ = close.shift(1).fillna(close.iloc[0])
    volume = pd.DataFrame(rng.uniform(1e3, 1e4, (n, len(cols))), index=idx, columns=list(cols))
    return close, open_, volume


# -- FIFO trades reconcile with the engine ---------------------------------------------

@pytest.mark.parametrize("seed", range(4))
def test_fifo_realized_plus_unrealized_equals_engine_gross_pnl(seed):
    close, open_, _ = market(seed)
    w = signals.trend_vol_target(close, trend_span=48, vol_lookback=48, target_vol=0.5, min_exposure=0.15)
    r = BacktestEngine(COSTLY, execution_price="open", rebalance_threshold=0.05, rebalance_hours_utc=(0,)).run(close, w, open_wide=open_)
    trades, open_pos = an.fifo_trades(r.fills, close.ffill().iloc[-1])
    realized = trades["gross_pnl"].sum() if len(trades) else 0.0
    unrealized = open_pos["unrealized_gross_pnl"].sum() if len(open_pos) else 0.0
    assert realized + unrealized == pytest.approx(r.gross_pnl.sum(), abs=1e-6)
    # every cost lands in exactly one place: closed trades or still-open lots
    open_costs = open_pos["entry_costs"].sum() if len(open_pos) else 0.0
    assert trades["costs"].sum() + open_costs == pytest.approx(r.total_fees, abs=1e-6)
    assert (trades["holding_time"] >= pd.Timedelta(0)).all()


def test_fifo_matches_oldest_lot_first():
    t = pd.date_range("2025-01-01", periods=3, freq="h")
    fills = pd.DataFrame([
        {"timestamp": t[0], "symbol": "A", "side": "BUY", "quantity": 1.0, "price": 10.0, "cost": 0.0},
        {"timestamp": t[1], "symbol": "A", "side": "BUY", "quantity": 1.0, "price": 20.0, "cost": 0.0},
        {"timestamp": t[2], "symbol": "A", "side": "SELL", "quantity": 1.5, "price": 30.0, "cost": 0.0},
    ])
    trades, open_pos = an.fifo_trades(fills, pd.Series({"A": 30.0}))
    assert trades["entry_price"].tolist() == [10.0, 20.0]
    assert trades["quantity"].tolist() == [1.0, 0.5]
    assert trades["gross_pnl"].tolist() == [20.0, 5.0]
    assert open_pos["quantity"].tolist() == [0.5]


# -- metrics on known series ---------------------------------------------------------------

def test_drawdown_episodes_known_path():
    idx = pd.date_range("2025-01-01", periods=8, freq="D")
    eq = pd.Series([100, 110, 99, 110, 120, 90, 100, 105.0], index=idx)
    eps = an.drawdown_episodes(eq)
    assert len(eps) == 2
    deepest = eps.iloc[0]
    assert deepest.depth == pytest.approx(0.25) and deepest.peak == idx[4] and deepest.trough == idx[5]
    assert pd.isna(deepest.recovery)  # never got back above 120
    first = eps.iloc[1]
    assert first.depth == pytest.approx(0.1) and first.recovery == idx[3]
    assert an.drawdown_summary(eq)["open_drawdown"] is True


def test_period_returns_compound_to_total():
    close, _, _ = market(1)
    eq = close["BTC/USD"] / close["BTC/USD"].iloc[0] * 1e5
    months = an.period_returns(eq, "ME")
    assert (1 + months).prod() - 1 == pytest.approx(eq.iloc[-1] / eq.iloc[0] - 1)


def test_var_cvar_ordering_and_sign():
    close, _, _ = market(2)
    s = an.return_risk_stats(close["BTC/USD"] * 1000, 8766)
    assert 0 < s["var_95_daily"] <= s["cvar_95_daily"]
    assert s["var_95_daily"] <= s["var_99_daily"] <= s["cvar_99_daily"]


def test_trade_statistics_concentration():
    trades = pd.DataFrame({"net_pnl": [100.0, -10.0, 5.0, -5.0, 10.0], "holding_time": [pd.Timedelta(hours=1)] * 5})
    st = an.trade_statistics(trades)
    assert st["trades"] == 5 and st["win_rate"] == pytest.approx(0.6)
    assert st["profit_factor"] == pytest.approx(115 / 15)
    assert st["top_1_share_of_realized_pnl"] == pytest.approx(100 / 100)
    assert st["realized_pnl_without_top_1"] == pytest.approx(0.0)
    assert st["max_consecutive_losses"] == 1


# -- benchmarks --------------------------------------------------------------------------------

def test_basket_benchmark_buys_once_and_tracks_passive_holdings():
    close, open_, _ = market(3)
    w = buy_and_hold_weights(close, ["BTC/USD", "ETH/USD"])
    r = BacktestEngine(FREE, execution_price="open", rebalance_threshold=BENCHMARK_BAND).run(close, w, open_wide=open_)
    assert len(r.fills) == 2 and set(r.fills["side"]) == {"BUY"}
    entry = r.fills.set_index("symbol")["price"]
    passive = 50_000 * (close["BTC/USD"] / entry["BTC/USD"] + close["ETH/USD"] / entry["ETH/USD"])
    assert r.portfolio_value.iloc[-1] == pytest.approx(passive.iloc[-1])


def test_random_entry_keeps_exposure_profile_and_is_seeded():
    close, _, _ = market(4)
    strat = signals.trend_vol_target(close, trend_span=48, vol_lookback=48, target_vol=0.5, min_exposure=0.15)
    a, b = random_entry_weights(close, strat, 7), random_entry_weights(close, strat, 7)
    pd.testing.assert_frame_equal(a, b)
    assert not random_entry_weights(close, strat, 8).equals(a)
    assert (a.sum(axis=1) <= 1 + 1e-12).all()
    prof_s, prof_r = exposure_profile(strat["BTC/USD"]), exposure_profile(a["BTC/USD"])
    assert prof_r["switch_prob"] == pytest.approx(prof_s["switch_prob"], rel=0.5)


def test_random_entry_ignores_prices():
    close, _, _ = market(5)
    strat = signals.trend_vol_target(close, trend_span=48, vol_lookback=48)
    shocked = close.copy()
    shocked.iloc[500:] *= 10
    # same exposure profile input -> identical random path regardless of prices
    pd.testing.assert_frame_equal(random_entry_weights(close, strat, 1), random_entry_weights(shocked, strat, 1))


# -- report ------------------------------------------------------------------------------------------

def test_report_writes_consistent_outputs(tmp_path):
    close, open_, volume = market(6)
    cfg = ReportConfig(strategy_name="trend_vol_target", strategy_fn=signals.trend_vol_target,
                       strategy_params={"trend_span": 48, "vol_lookback": 48, "target_vol": 0.5, "min_exposure": 0.15},
                       cost_model=COSTLY, rebalance_threshold=0.05, rebalance_hours_utc=(0,), random_seeds=[0, 1, 2],
                       hypothesis="test hypothesis")
    summary = run_report(cfg, close, open_, volume, close.index[48:], tmp_path)
    for name in ["report.html", "summary.json", "config.json", "equity.csv", "fills.csv", "trades.csv", "drawdowns.csv",
                 "random_entry.csv", "windows_strategy.csv"]:
        assert (tmp_path / name).exists(), name
    page = (tmp_path / "report.html").read_text()
    for anchor in ["grossnet", "drawdowns", "trades", "windows", "benchmarks", "limits", "repro", "test hypothesis"]:
        assert anchor in page
    saved = json.loads((tmp_path / "summary.json").read_text())
    c = saved["costs"]
    assert c["gross_pnl"] - c["total_costs"] == pytest.approx(c["net_pnl"], abs=1e-6)
    equity = pd.read_csv(tmp_path / "equity.csv", index_col=0)
    assert equity["net_equity"].iloc[-1] == pytest.approx(summary["net"]["ending_equity"])
    assert all(v["total_return"] != 0 for v in summary["benchmarks"].values())  # every benchmark actually invested
