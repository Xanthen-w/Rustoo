"""Split / holdout guards, walk-forward fold construction, and the engine's
rebalance threshold."""
import numpy as np
import pandas as pd
import pytest

from backtest.costs import CostModel
from backtest.engine import BacktestEngine
from backtest.splits import DataSplit, HoldoutAccessError, load_split_panel, load_splits
from backtest.walk_forward import expand_grid, make_folds, run_walk_forward
from src.data.binance import pair_to_filename
from src.data.historical import ParquetDataSource

FREE = CostModel(taker_fee=0.0, maker_fee=0.0, slippage_bps=0.0)


def _write(root, pair, index, close):
    root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=index)
    df.index.name = "timestamp"
    df.to_parquet(root / pair_to_filename(pair))


SPLITS_CFG = {
    "splits": {
        "train": {"start": "2025-01-01", "end": "2025-01-05"},
        "validation": {"start": "2025-01-05", "end": "2025-01-08"},
        "test": {"start": "2025-01-08", "end": None},
    }
}


@pytest.fixture
def source(tmp_path):
    idx = pd.date_range("2025-01-01 01:00", "2025-01-10 00:00", freq="1h", tz="UTC")
    rng = np.random.default_rng(0)
    _write(tmp_path, "BTC/USD", idx, 100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))))
    _write(tmp_path, "OLD/USD", idx[:30], np.full(30, 5.0))  # delisted early in train
    return ParquetDataSource(tmp_path)


def test_repo_splits_are_ordered_and_holdout_last():
    splits = load_splits()
    assert splits["train"].end == splits["validation"].start
    assert splits["validation"].end == splits["test"].start


def test_overlapping_splits_rejected():
    bad = {"splits": {"train": {"start": "2025-01-01", "end": "2025-02-01"},
                      "test": {"start": "2025-01-15", "end": None}}}
    with pytest.raises(ValueError, match="overlap"):
        load_splits(bad)


def test_holdout_refused_without_opt_in(source):
    splits = load_splits(SPLITS_CFG)
    with pytest.raises(HoldoutAccessError):
        load_split_panel(source, ["BTC/USD"], splits["test"], resample=None)
    panel = load_split_panel(source, ["BTC/USD"], splits["test"], resample=None, allow_holdout=True)
    assert panel.eval_index[0] >= splits["test"].start


def test_train_panel_never_contains_later_bars(source):
    splits = load_splits(SPLITS_CFG)
    panel = load_split_panel(source, ["BTC/USD"], splits["train"], resample=None)
    assert panel.close.index.max() < splits["train"].end


def test_validation_panel_keeps_warmup_but_scores_only_its_window(source):
    split = load_splits(SPLITS_CFG)["validation"]
    panel = load_split_panel(source, ["BTC/USD"], split, resample=None)
    assert panel.close.index.min() < split.start  # warm-up history available
    assert panel.eval_index.min() >= split.start
    assert panel.eval_index.max() < split.end


def test_pair_delisted_before_split_end_is_dropped(source):
    panel = load_split_panel(source, ["BTC/USD", "OLD/USD"], load_splits(SPLITS_CFG)["train"], resample=None)
    assert "OLD/USD" in panel.dropped_pairs
    assert list(panel.close.columns) == ["BTC/USD"]


def test_rolling_and_anchored_folds():
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-04-01", tz="UTC")
    day = pd.Timedelta(days=1)
    rolling = make_folds(start, end, 30 * day, 10 * day, 10 * day)
    assert len(rolling) == 6
    assert all(f.oos_start == f.is_end and f.oos_end <= end for f in rolling)
    assert rolling[1].is_start == start + 10 * day
    for a, b in zip(rolling, rolling[1:]):
        assert b.oos_start == a.oos_end  # contiguous, non-overlapping OOS windows
    anchored = make_folds(start, end, 30 * day, 10 * day, 10 * day, anchored=True)
    assert all(f.is_start == start for f in anchored)


def test_grid_expansion_skips_invalid_ema_pairs():
    cands = expand_grid("trend_following", {"params": {"fast_span": [10, 50], "slow_span": [20, 50]},
                                            "engine": {"rebalance_threshold": [0.0, 0.1]}})
    # (10,20), (10,50) valid; (50,20), (50,50) invalid -> 2 x 2 thresholds
    assert len(cands) == 4


def test_walk_forward_picks_on_in_sample_only():
    idx = pd.date_range("2025-01-01 01:00", periods=24 * 60, freq="1h", tz="UTC")
    rng = np.random.default_rng(1)
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(idx))) for s in "ABC"}, index=idx)
    cands = expand_grid("cross_sectional_momentum", {"params": {"lookback": [12, 48], "top_k": [1, 2]}})
    folds = make_folds(idx[0], idx[-1] + pd.Timedelta(hours=1), pd.Timedelta(days=20), pd.Timedelta(days=10), pd.Timedelta(days=10))
    result = run_walk_forward(close, cands, folds, FREE, 8766)
    scores = result.candidate_scores
    for _, row in result.folds.iterrows():
        fold_scores = scores[scores.fold == row["fold"]]
        best_is = fold_scores.loc[fold_scores.is_score.idxmax()]
        assert row["chosen"] == best_is["candidate"]
    assert result.oos_equity.index.min() >= folds[0].oos_start
    assert len(result.folds) == len(folds)


def test_rebalance_threshold_suppresses_small_drift_trades():
    idx = pd.date_range("2025-01-01", periods=50, freq="h")
    rng = np.random.default_rng(2)
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0, 0.003, 50)) for s in "AB"}, index=idx)
    weights = pd.DataFrame(0.5, index=idx, columns=["A", "B"])
    every_bar = BacktestEngine(FREE, rebalance_threshold=0.0).run(close, weights)
    banded = BacktestEngine(FREE, rebalance_threshold=0.05).run(close, weights)
    assert (banded.trade_notional_history != 0).sum().sum() < (every_bar.trade_notional_history != 0).sum().sum()
    assert (banded.trade_notional_history.iloc[1] != 0).all()  # initial entry still happens


def test_rebalance_threshold_never_blocks_exit():
    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    close = pd.DataFrame({"A": [100.0] * 5}, index=idx)
    weights = pd.DataFrame({"A": [0.03, 0.03, 0.0, 0.0, 0.0]}, index=idx)
    result = BacktestEngine(FREE, rebalance_threshold=0.02).run(close, weights)
    assert result.weights_history["A"].iloc[1] == pytest.approx(0.03)  # entry >= threshold
    assert result.weights_history["A"].iloc[3] == 0.0  # exit to zero executes despite < threshold
