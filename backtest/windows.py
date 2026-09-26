"""Fixed-horizon window evaluation: the distribution of outcomes a strategy
would have had over every historical window of the competition's length.

The competition scores a single 14-day run starting from cash
(docs/COMPETITION_RULES.md), so the relevant question isn't a strategy's
multi-year Sharpe but "starting on a random day, how likely is a positive
14-day return, what composite score, and how many days with trades?".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import metrics
from backtest.costs import CostModel
from backtest.walk_forward import build_engine


@dataclass(frozen=True)
class Window:
    start: pd.Timestamp
    end: pd.Timestamp  # exclusive


def rolling_windows(
    eval_index: pd.DatetimeIndex,
    length: pd.Timedelta,
    step: pd.Timedelta,
    skip_first: pd.Timedelta = pd.Timedelta(0),
) -> list[Window]:
    """Windows [start, start + length) fully inside the evaluation range,
    starting `skip_first` after its beginning (for indicator warm-up when
    the split is the first data available)."""
    first, last = eval_index[0], eval_index[-1]
    windows = []
    start = first + skip_first
    while start + length <= last + pd.Timedelta(microseconds=1):
        windows.append(Window(start, start + length))
        start += step
    return windows


def evaluate_windows(
    close: pd.DataFrame,
    weights: pd.DataFrame,
    windows: list[Window],
    cost_model: CostModel,
    periods_per_year: float,
    engine_params: dict | None = None,
    initial_capital: float = 100_000.0,
) -> pd.DataFrame:
    """One row per window: the strategy run from cash at the window start."""
    # Only columns the strategy ever holds matter to the engine; dropping the
    # rest makes this ~10x faster on a wide panel without changing results.
    active = weights.columns[(weights != 0).any()]
    close, weights = close[active], weights[active]
    rows = []
    for w in windows:
        mask = (close.index >= w.start) & (close.index < w.end)
        wc, ww = close[mask], weights[mask]
        if wc.empty or len(active) == 0:
            rows.append({"start": w.start, "return": 0.0, "composite_score": 0.0, "sharpe_ratio": 0.0,
                         "sortino_ratio": 0.0, "calmar_ratio": 0.0, "max_drawdown": 0.0,
                         "trading_days": 0, "trades": 0, "fees": 0.0})
            continue
        result = build_engine(cost_model, engine_params or {}, initial_capital).run(wc, ww)
        s = metrics.summarize(result.portfolio_value, result.weights_history, result.total_fees, periods_per_year)
        traded = (result.trade_notional_history != 0).any(axis=1)
        rows.append({
            "start": w.start,
            "return": s["cumulative_return"],
            "composite_score": s["composite_score"],
            "sharpe_ratio": s["sharpe_ratio"],
            "sortino_ratio": s["sortino_ratio"],
            "calmar_ratio": s["calmar_ratio"],
            "max_drawdown": s["max_drawdown"],
            "trading_days": int(pd.Series(traded[traded].index.date).nunique()),
            "trades": int((result.trade_notional_history != 0).to_numpy().sum()),
            "fees": s["total_fees"],
        })
    return pd.DataFrame(rows)


def summarize_windows(per_window: pd.DataFrame, min_trading_days: int = 8) -> dict:
    r = per_window["return"]
    return {
        "windows": len(per_window),
        "median_return": float(r.median()),
        "p_positive": float((r > 0).mean()),
        "p10_return": float(r.quantile(0.10)),
        "p90_return": float(r.quantile(0.90)),
        "median_composite": float(per_window["composite_score"].median()),
        "median_max_dd": float(per_window["max_drawdown"].median()),
        "median_trading_days": float(per_window["trading_days"].median()),
        f"p_days_ge_{min_trading_days}": float((per_window["trading_days"] >= min_trading_days).mean()),
        "median_trades": float(per_window["trades"].median()),
    }


def summarize_benchmark(per_window: pd.DataFrame, benchmark: pd.DataFrame) -> dict:
    """How often the strategy beats the benchmark's return in the same window."""
    joined = per_window.set_index("start")["return"].to_frame("s").join(benchmark.set_index("start")["return"].to_frame("b"))
    return {"p_beats_benchmark": float((joined.s > joined.b).mean()) if len(joined) else np.nan}
