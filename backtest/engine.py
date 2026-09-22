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
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.costs import CostModel


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
    ):
        if execution_lag < 1:
            raise ValueError(
                "execution_lag must be >= 1: executing on the same bar a "
                "signal was computed from is a look-ahead risk."
            )
        self.cost_model = cost_model
        self.initial_capital = initial_capital
        self.execution_lag = execution_lag

    def run(self, close_wide: pd.DataFrame, target_weights: pd.DataFrame) -> BacktestResult:
        if not close_wide.index.equals(target_weights.index):
            raise ValueError("close_wide and target_weights must share the same index")
        if not close_wide.index.is_monotonic_increasing:
            raise ValueError("close_wide index must be chronologically sorted")

        symbols = close_wide.columns
        # Shift weights forward by execution_lag: the weight computed at row
        # t is only actually applied at row t + execution_lag.
        applied_weights = target_weights.shift(self.execution_lag)

        n = len(close_wide)
        cash = self.initial_capital
        quantities = pd.Series(0.0, index=symbols)

        portfolio_values = np.full(n, np.nan)
        fees = np.zeros(n)
        realized_weights = pd.DataFrame(0.0, index=close_wide.index, columns=symbols)
        trade_notional = pd.DataFrame(0.0, index=close_wide.index, columns=symbols)

        for i in range(n):
            prices = close_wide.iloc[i]
            valid = prices.notna()

            position_value = (quantities * prices.fillna(0.0))
            pre_trade_value = cash + position_value.sum()

            target = applied_weights.iloc[i].fillna(0.0)
            target = target.where(valid, 0.0)  # can't hold what has no price

            desired_value = target * pre_trade_value
            current_value = position_value
            delta_value = desired_value - current_value

            period_fee = 0.0
            for symbol in symbols:
                if not valid[symbol]:
                    continue
                trade = delta_value[symbol]
                if abs(trade) < 1e-12:
                    continue
                cost = self.cost_model.trade_cost(abs(trade))
                period_fee += cost
                quantities[symbol] = desired_value[symbol] / prices[symbol]
                trade_notional.loc[close_wide.index[i], symbol] = trade

            cash = pre_trade_value - desired_value.where(valid, current_value).sum() - period_fee
            post_trade_position_value = (quantities * prices.fillna(0.0)).sum()
            portfolio_value_t = cash + post_trade_position_value

            portfolio_values[i] = portfolio_value_t
            fees[i] = period_fee
            if portfolio_value_t > 0:
                realized_weights.iloc[i] = (quantities * prices.fillna(0.0) / portfolio_value_t).fillna(0.0)

        return BacktestResult(
            portfolio_value=pd.Series(portfolio_values, index=close_wide.index, name="portfolio_value"),
            weights_history=realized_weights,
            trade_notional_history=trade_notional,
            fees_per_period=pd.Series(fees, index=close_wide.index, name="fees"),
        )
