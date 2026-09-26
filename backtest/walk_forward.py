"""Walk-forward optimisation (WFO).

For each fold: score every candidate (strategy params + engine params) on
the in-sample window, pick the best by the objective, then score that pick
on the following out-of-sample window it has never seen. Stitching the
out-of-sample pieces together gives an honest track record of the whole
select-then-trade procedure, rather than of one parameter set chosen with
hindsight.

Weights for each candidate are computed once over the whole panel and
sliced per window. That is only valid because every strategy is causal
(weights at t use data <= t), which tests/test_lookahead.py enforces; the
panel itself never extends past the split being studied (backtest/splits.py).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest import metrics
from backtest.costs import CostModel
from backtest.engine import BacktestEngine
from src.risk.drawdown import DrawdownRiskManager
from src.strategy.signals import STRATEGIES


def _freeze(value):
    """Grid values can be lists/dicts (asset lists, risk settings); make
    them hashable so candidates can be dict keys."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def build_engine(cost_model: CostModel, engine_params: dict, initial_capital: float = 100_000.0) -> BacktestEngine:
    """`engine_params` may include `risk`: None for no overlay, or a dict
    of DrawdownRiskManager settings."""
    params = dict(engine_params)
    risk = params.pop("risk", None)
    overlay = DrawdownRiskManager(**dict(risk)) if risk else None
    return BacktestEngine(cost_model, initial_capital=initial_capital, risk_overlay=overlay, **params)


@dataclass(frozen=True)
class Fold:
    number: int
    is_start: pd.Timestamp
    is_end: pd.Timestamp  # exclusive
    oos_start: pd.Timestamp
    oos_end: pd.Timestamp  # exclusive


def make_folds(
    start: pd.Timestamp,
    end: pd.Timestamp,
    in_sample: pd.Timedelta,
    out_of_sample: pd.Timedelta,
    step: pd.Timedelta,
    anchored: bool = False,
) -> list[Fold]:
    """Consecutive folds inside [start, end). Only complete out-of-sample
    windows are produced. Rolling: the in-sample window slides by `step`;
    anchored: it always starts at `start` and grows."""
    folds = []
    k = 0
    while True:
        is_start = start if anchored else start + k * step
        is_end = start + in_sample + k * step
        oos_end = is_end + out_of_sample
        if oos_end > end:
            break
        folds.append(Fold(k, is_start, is_end, is_end, oos_end))
        k += 1
    if not folds:
        raise ValueError("split too short for one in-sample + out-of-sample window")
    return folds


@dataclass(frozen=True)
class Candidate:
    strategy: str
    params: tuple  # sorted (key, value) pairs, hashable
    engine_params: tuple

    @property
    def label(self) -> str:
        def fmt(v):
            if isinstance(v, tuple) and v and all(isinstance(x, tuple) and len(x) == 2 for x in v):
                return "{" + ",".join(f"{k}:{fmt(x)}" for k, x in v) + "}"
            if isinstance(v, tuple):
                return "[" + ",".join(map(str, v)) + "]"
            return str(v)
        parts = [f"{k}={fmt(v)}" for k, v in self.params + self.engine_params]
        return f"{self.strategy}({', '.join(parts)})"

    def strategy_kwargs(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.params}

    def engine_kwargs(self) -> dict:
        return {k: (dict(v) if isinstance(v, tuple) and v and isinstance(v[0], tuple) else v) for k, v in self.engine_params}


def expand_grid(strategy: str, spec: dict) -> list[Candidate]:
    if strategy not in STRATEGIES:
        raise KeyError(f"unknown strategy {strategy!r}; known: {sorted(STRATEGIES)}")

    def combos(grid: dict) -> list[tuple]:
        keys = sorted(grid or {})
        return [tuple(zip(keys, map(_freeze, values))) for values in itertools.product(*(grid[k] for k in keys))]

    out = []
    for params in combos(spec.get("params", {})):
        p = dict(params)
        if "fast_span" in p and "slow_span" in p and p["fast_span"] >= p["slow_span"]:
            continue
        for engine_params in combos(spec.get("engine", {})):
            out.append(Candidate(strategy, params, engine_params))
    return out


def evaluate_window(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cost_model: CostModel,
    periods_per_year: float,
    engine_params: dict | None = None,
    initial_capital: float = 100_000.0,
) -> tuple[dict, pd.Series]:
    """Backtest [start, end) starting from cash; returns (summary, equity)."""
    mask = (close.index >= start) & (close.index < end)
    window_close = close[mask].dropna(axis=1, how="all")
    window_weights = weights.loc[window_close.index, window_close.columns]
    engine = build_engine(cost_model, engine_params or {}, initial_capital)
    result = engine.run(window_close, window_weights)
    summary = metrics.summarize(result.portfolio_value, result.weights_history, result.total_fees, periods_per_year)
    summary["avg_trades_per_bar"] = float((result.trade_notional_history != 0).sum(axis=1).mean())
    return summary, result.portfolio_value


@dataclass
class WalkForwardResult:
    folds: pd.DataFrame  # one row per fold: chosen candidate, IS and OOS scores
    candidate_scores: pd.DataFrame  # every candidate x fold, IS and OOS
    oos_equity: pd.Series  # stitched out-of-sample equity of the WFO procedure
    oos_summary: dict = field(default_factory=dict)


def _normalise_weights(weights: pd.DataFrame, close: pd.DataFrame) -> pd.DataFrame:
    """Strategies return weights for the panel's columns; enforce the same
    index/columns and zero-fill (a NaN weight means 'no position')."""
    return weights.reindex(index=close.index, columns=close.columns).fillna(0.0)


def run_walk_forward(
    close: pd.DataFrame,
    candidates: list[Candidate],
    folds: list[Fold],
    cost_model: CostModel,
    periods_per_year: float,
    objective: str = "sortino_ratio",
    initial_capital: float = 100_000.0,
) -> WalkForwardResult:
    weight_cache: dict[tuple, pd.DataFrame] = {}
    rows, fold_rows, oos_pieces = [], [], []

    for fold in folds:
        best: tuple[float, Candidate] | None = None
        fold_scores = {}
        for cand in candidates:
            key = (cand.strategy, cand.params)
            if key not in weight_cache:
                weight_cache[key] = _normalise_weights(STRATEGIES[cand.strategy](close, **cand.strategy_kwargs()), close)
            weights = weight_cache[key]
            engine_params = cand.engine_kwargs()
            is_summary, _ = evaluate_window(close, weights, fold.is_start, fold.is_end, cost_model,
                                            periods_per_year, engine_params, initial_capital)
            oos_summary, oos_equity = evaluate_window(close, weights, fold.oos_start, fold.oos_end, cost_model,
                                                      periods_per_year, engine_params, initial_capital)
            fold_scores[cand] = (is_summary, oos_summary, oos_equity)
            rows.append({
                "fold": fold.number, "candidate": cand.label,
                "is_score": is_summary[objective], "oos_score": oos_summary[objective],
                "is_return": is_summary["cumulative_return"], "oos_return": oos_summary["cumulative_return"],
            })
            score = is_summary[objective]
            if np.isfinite(score) and (best is None or score > best[0]):
                best = (score, cand)

        chosen = best[1] if best else candidates[0]
        is_summary, oos_summary, oos_equity = fold_scores[chosen]
        oos_pieces.append(oos_equity.pct_change().fillna(0.0))
        fold_rows.append({
            "fold": fold.number,
            "in_sample": f"{fold.is_start:%Y-%m-%d} -> {fold.is_end:%Y-%m-%d}",
            "out_of_sample": f"{fold.oos_start:%Y-%m-%d} -> {fold.oos_end:%Y-%m-%d}",
            "chosen": chosen.label,
            "is_score": is_summary[objective],
            "oos_score": oos_summary[objective],
            "oos_return": oos_summary["cumulative_return"],
            "oos_max_drawdown": oos_summary["max_drawdown"],
            "oos_trades_per_bar": oos_summary["avg_trades_per_bar"],
        })

    stitched_returns = pd.concat(oos_pieces)
    oos_equity = initial_capital * (1.0 + stitched_returns).cumprod()
    rets = oos_equity.pct_change().dropna()
    sharpe = metrics.sharpe_ratio(rets, periods_per_year)
    sortino = metrics.sortino_ratio(rets, periods_per_year)
    calmar = metrics.calmar_ratio(oos_equity, periods_per_year)
    oos_summary = {
        "cumulative_return": metrics.cumulative_return(oos_equity),
        "max_drawdown": metrics.max_drawdown(oos_equity),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "composite_score": metrics.composite_score(sortino, sharpe, calmar),
    }
    return WalkForwardResult(pd.DataFrame(fold_rows), pd.DataFrame(rows), oos_equity, oos_summary)
