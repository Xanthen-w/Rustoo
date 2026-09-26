"""Chronological, multi-asset backtest engine (Phase 6).

Design choice made explicit to avoid look-ahead bias: the caller supplies a
`target_weights` DataFrame already computed causally (e.g. from
src/strategy/signals.py, which only uses `close.loc[:t]` at each row t). The
engine then applies an `execution_lag` (default 1 bar) so that the weight
decided using information through bar t is only ever filled using the price
at bar `t + execution_lag`, never bar t's own price. This models the real
latency between "signal computed" and "order actually executed" and is the
single most common source of accidental look-ahead in naive backtests.

Nothing here uses `close_wide` values beyond the bar being executed at any
point in the loop.

Accounting rules:

- Long-only, no leverage: target weights must be >= 0 and each row must sum
  to at most `max_gross_exposure` (default 1.0). Violations raise instead of
  being silently simulated as borrowing.
- A missing (NaN) price means the asset can't be traded on that bar. An
  existing position is carried unchanged and marked at its last known price
  — a data gap is not a loss.
- Trades are sized so that cash never goes negative after fees: when a
  target would spend more than the available equity once costs are
  included, all buys are scaled down proportionally.
- `risk_overlay` (optional, e.g. src/risk/drawdown.py::DrawdownRiskManager):
  multiplies each bar's target weights by an exposure in [0, 1]. It is
  updated with each bar's *post-trade* equity and its answer applies from
  the next bar, so sizing never uses the price the trade executes at.
- `rebalance_hours_utc`: bars whose (UTC close-time) hour is listed are
  rebalanced exactly to target, ignoring the band — a scheduled daily
  rebalance, mirroring what the live bot does.
- `rebalance_threshold` (turnover control): an asset is only traded when
  |target weight - current weight| >= the threshold; below it the position
  is left to drift. A target of zero is always executed in full, so exits
  are never blocked by the band.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import CostModel

_WEIGHT_TOLERANCE = 1e-9


def size_orders(
    target_weights: np.ndarray,
    current_value: np.ndarray,
    tradable: np.ndarray,
    equity: float,
    cost_rate: float,
) -> np.ndarray:
    """Desired post-trade position values for one bar.

    Untradable assets (no price, or inside the rebalance band) keep their
    current value. Tradable assets go to target * equity, unless that plus
    fees would spend more than the equity not tied up in frozen positions —
    then all tradable targets are scaled down together so cash never goes
    negative.
    """
    frozen_value = current_value[~tradable].sum()
    # Clamped: with every position frozen and cash at ~0 this can come out a
    # hair below zero from float rounding.
    budget = max(equity - frozen_value, 0.0)
    base = np.where(tradable, target_weights, 0.0) * equity
    desired = np.where(tradable, base, current_value)
    fee = cost_rate * np.abs(desired - current_value).sum()
    spend = desired[tradable].sum()
    if spend <= 0 or spend + fee <= budget:
        return desired

    # Fees are a small fraction of notional, so this fixed-point iteration
    # converges in a handful of steps.
    for _ in range(20):
        scale = max(budget - fee, 0.0) / spend
        desired = np.where(tradable, base * scale, current_value)
        new_fee = cost_rate * np.abs(desired - current_value).sum()
        converged = abs(new_fee - fee) < 1e-12 * max(equity, 1.0)
        fee = new_fee
        if converged:
            break
    # Absorb the last rounding error so cash can't dip below zero.
    overshoot = desired[tradable].sum() + fee - budget
    total = desired[tradable].sum()
    if overshoot > 0 and total > 0:
        desired = np.where(tradable, desired * max(1.0 - overshoot / total, 0.0), desired)
    return desired


@dataclass
class BacktestResult:
    portfolio_value: pd.Series
    weights_history: pd.DataFrame
    trade_notional_history: pd.DataFrame
    fees_per_period: pd.Series
    total_fees: float = field(init=False)
    exposure: pd.Series | None = None  # risk-overlay multiplier applied at each bar

    def __post_init__(self):
        self.total_fees = float(self.fees_per_period.sum())


class BacktestEngine:
    def __init__(
        self,
        cost_model: CostModel,
        initial_capital: float = 100_000.0,
        execution_lag: int = 1,
        max_gross_exposure: float = 1.0,
        rebalance_threshold: float = 0.0,
        risk_overlay=None,
        rebalance_hours_utc: tuple = (),
    ):
        if execution_lag < 1:
            raise ValueError(
                "execution_lag must be >= 1: executing on the same bar a "
                "signal was computed from is a look-ahead risk."
            )
        if not 0 < max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]: leverage is not allowed")
        if not 0.0 <= rebalance_threshold < 1.0:
            raise ValueError("rebalance_threshold must be in [0, 1)")
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.execution_lag = execution_lag
        self.max_gross_exposure = max_gross_exposure
        self.rebalance_threshold = rebalance_threshold
        self.risk_overlay = risk_overlay
        self.rebalance_hours_utc = frozenset(int(h) for h in rebalance_hours_utc)
        if any(not 0 <= h < 24 for h in self.rebalance_hours_utc):
            raise ValueError("rebalance_hours_utc must be hours in [0, 24)")

    def _validate_weights(self, target_weights: pd.DataFrame) -> None:
        values = target_weights.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("target_weights contains infinite values")
        if (np.nan_to_num(values) < -_WEIGHT_TOLERANCE).any():
            raise ValueError("target_weights has negative weights: shorting is not allowed")
        gross = np.nansum(values, axis=1)
        if (gross > self.max_gross_exposure + _WEIGHT_TOLERANCE).any():
            worst = target_weights.index[int(np.argmax(gross))]
            raise ValueError(
                f"target_weights row sums exceed max_gross_exposure="
                f"{self.max_gross_exposure} (max {gross.max():.4f} at {worst}): "
                f"leverage is not allowed"
            )

    def run(self, close_wide: pd.DataFrame, target_weights: pd.DataFrame) -> BacktestResult:
        if not close_wide.index.equals(target_weights.index):
            raise ValueError("close_wide and target_weights must share the same index")
        if not close_wide.columns.equals(target_weights.columns):
            raise ValueError("close_wide and target_weights must share the same columns")
        if not close_wide.index.is_monotonic_increasing:
            raise ValueError("close_wide index must be chronologically sorted")
        self._validate_weights(target_weights)

        # Shift weights forward by execution_lag: the weight computed at row
        # t is only actually applied at row t + execution_lag.
        applied = np.nan_to_num(target_weights.shift(self.execution_lag).to_numpy(dtype=float))
        prices = close_wide.to_numpy(dtype=float)
        n, m = prices.shape
        rate = self.cost_model.cost_rate

        cash = float(self.initial_capital)
        quantities = np.zeros(m)
        last_price = np.full(m, np.nan)

        portfolio_values = np.empty(n)
        fees = np.zeros(n)
        realized_weights = np.zeros((n, m))
        trade_notional = np.zeros((n, m))
        exposures = np.ones(n)
        exposure = 1.0
        if self.risk_overlay is not None:
            self.risk_overlay.reset(cash)
        index_utc = close_wide.index.tz_convert("UTC") if close_wide.index.tz is not None else close_wide.index
        scheduled = np.isin(index_utc.hour, list(self.rebalance_hours_utc)) if self.rebalance_hours_utc else np.zeros(n, bool)

        for i in range(n):
            p = prices[i]
            tradable = np.isfinite(p) & (p > 0)
            last_price = np.where(tradable, p, last_price)
            # Positions are marked at the last known price; an asset that has
            # never had a price can only have a zero position.
            mark = np.nan_to_num(last_price)

            current_value = quantities * mark
            equity = cash + current_value.sum()
            target_row = applied[i] * exposure
            exposures[i] = exposure

            if self.rebalance_threshold > 0 and equity > 0 and not scheduled[i]:
                # Inside the band: leave the position alone this bar (treated
                # exactly like an untradable asset below). Exits always trade.
                drift = np.abs(target_row - current_value / equity)
                tradable = tradable & ((drift >= self.rebalance_threshold) | (target_row == 0))

            desired = size_orders(target_row, current_value, tradable, equity, rate)

            delta = np.where(tradable, desired - current_value, 0.0)
            delta[np.abs(delta) < 1e-12] = 0.0
            traded = delta != 0.0
            quantities = np.where(traded, desired / np.where(tradable, p, 1.0), quantities)
            fee = rate * np.abs(delta).sum()

            cash = equity - (quantities * mark).sum() - fee
            post_value = cash + (quantities * mark).sum()
            if not np.isfinite(post_value):
                raise RuntimeError(f"non-finite portfolio value at {close_wide.index[i]}; engine state is corrupt")
            if self.risk_overlay is not None:
                exposure = float(self.risk_overlay.update(post_value))

            portfolio_values[i] = post_value
            fees[i] = fee
            trade_notional[i] = delta
            if post_value > 0:
                realized_weights[i] = quantities * mark / post_value

        index, columns = close_wide.index, close_wide.columns
        return BacktestResult(
            portfolio_value=pd.Series(portfolio_values, index=index, name="portfolio_value"),
            weights_history=pd.DataFrame(realized_weights, index=index, columns=columns),
            trade_notional_history=pd.DataFrame(trade_notional, index=index, columns=columns),
            fees_per_period=pd.Series(fees, index=index, name="fees"),
            exposure=pd.Series(exposures, index=index, name="exposure") if self.risk_overlay is not None else None,
        )
