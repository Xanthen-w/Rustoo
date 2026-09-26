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
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import CostModel

_WEIGHT_TOLERANCE = 1e-9


@dataclass
class BacktestResult:
    portfolio_value: pd.Series
    weights_history: pd.DataFrame
    trade_notional_history: pd.DataFrame
    fees_per_period: pd.Series
    total_fees: float = field(init=False)

    def __post_init__(self):
        self.total_fees = float(self.fees_per_period.sum())


class BacktestEngine:
    def __init__(
        self,
        cost_model: CostModel,
        initial_capital: float = 100_000.0,
        execution_lag: int = 1,
        max_gross_exposure: float = 1.0,
    ):
        if execution_lag < 1:
            raise ValueError(
                "execution_lag must be >= 1: executing on the same bar a "
                "signal was computed from is a look-ahead risk."
            )
        if not 0 < max_gross_exposure <= 1.0:
            raise ValueError("max_gross_exposure must be in (0, 1]: leverage is not allowed")
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.execution_lag = execution_lag
        self.max_gross_exposure = max_gross_exposure

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

        for i in range(n):
            p = prices[i]
            tradable = np.isfinite(p) & (p > 0)
            last_price = np.where(tradable, p, last_price)
            # Positions are marked at the last known price; an asset that has
            # never had a price can only have a zero position.
            mark = np.nan_to_num(last_price)

            current_value = quantities * mark
            equity = cash + current_value.sum()

            # Positions in untradable assets are frozen; only the rest of
            # equity is available for the tradable targets.
            frozen_value = current_value[~tradable].sum()
            target = np.where(tradable, applied[i], 0.0)
            desired = np.where(tradable, target * equity, current_value)
            fee = rate * np.abs(desired - current_value).sum()

            budget = equity - frozen_value
            spend = desired[tradable].sum()
            if spend > 0 and spend + fee > budget:
                # Scale buys down until spend + fees fits the budget. Fees are
                # a small fraction of notional, so this fixed-point iteration
                # converges in a handful of steps.
                base = target * equity
                scale = 1.0
                for _ in range(20):
                    scale = max(budget - fee, 0.0) / spend
                    desired = np.where(tradable, base * scale, current_value)
                    new_fee = rate * np.abs(desired - current_value).sum()
                    if abs(new_fee - fee) < 1e-12 * max(equity, 1.0):
                        fee = new_fee
                        break
                    fee = new_fee
                # Guard the last rounding step so cash can't dip below zero.
                overshoot = desired[tradable].sum() + fee - budget
                if overshoot > 0:
                    desired = np.where(tradable, desired * (1 - overshoot / desired[tradable].sum()), desired)
                    fee = rate * np.abs(desired - current_value).sum()

            delta = np.where(tradable, desired - current_value, 0.0)
            delta[np.abs(delta) < 1e-12] = 0.0
            traded = delta != 0.0
            quantities = np.where(traded, desired / np.where(tradable, p, 1.0), quantities)
            fee = rate * np.abs(delta).sum()

            cash = equity - (quantities * mark).sum() - fee
            post_value = cash + (quantities * mark).sum()

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
        )
