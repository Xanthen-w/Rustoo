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

Execution timing (`execution_price`), made explicit:

- "close" (default): the order from bar t's signal fills at the close of
  bar t + execution_lag.
- "open": it fills at the *open* of bar t + execution_lag. With lag 1 that
  is the first price after the signal bar closes — what the live bot does,
  since it trades minutes after each hourly bar closes. Requires `open_wide`.

Gross vs net P&L. Each bar's mark-to-market P&L of the positions actually
held is recorded (`gross_pnl`: from the previous mark to the execution
price with the old holdings, then to the close with the new holdings),
separately from trading costs (`costs`: fee / spread / slippage / impact,
see backtest/costs.py). The accounting identity

    final equity - initial capital = sum(gross_pnl) - sum(costs)

holds to floating-point precision and is enforced by tests. The gross
curve is the *same trades* without costs deducted (not a re-run with zero
costs, which would trade differently).

Accounting rules:

- No leverage: each row's gross exposure (sum of |weights|) must be at most
  `max_gross_exposure` (default 1.0). Long-only by default: negative
  weights raise unless `allow_short=True`, in which case a negative weight
  is a 1x short whose notional is covered by collateral (Roostoo's
  /v6/short_open model; fees as for spot orders). Violations raise instead
  of being silently simulated as borrowing.
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
    cost_rate,
) -> np.ndarray:
    """Desired post-trade position values for one bar.

    Untradable assets (no price, or inside the rebalance band) keep their
    current value. Tradable assets go to target * equity, unless that plus
    fees would spend more than the equity not tied up in frozen positions —
    then all tradable targets are scaled down together so cash never goes
    negative. `cost_rate` is a scalar or a per-asset array (market impact
    differs by asset).

    Exposure is measured gross (sum of absolute values): a short uses its
    notional as collateral, so longs + short collateral + fees must fit in
    equity (no leverage). For long-only books this is the plain sum.
    """
    frozen_value = np.abs(current_value[~tradable]).sum()
    # Clamped: with every position frozen and cash at ~0 this can come out a
    # hair below zero from float rounding.
    budget = max(equity - frozen_value, 0.0)
    base = np.where(tradable, target_weights, 0.0) * equity
    desired = np.where(tradable, base, current_value)
    fee = (cost_rate * np.abs(desired - current_value)).sum()
    spend = np.abs(desired[tradable]).sum()
    if spend <= 0 or spend + fee <= budget:
        return desired

    # Fees are a small fraction of notional, so this fixed-point iteration
    # converges in a handful of steps.
    for _ in range(20):
        scale = max(budget - fee, 0.0) / spend
        desired = np.where(tradable, base * scale, current_value)
        new_fee = (cost_rate * np.abs(desired - current_value)).sum()
        converged = abs(new_fee - fee) < 1e-12 * max(equity, 1.0)
        fee = new_fee
        if converged:
            break
    # Absorb the last rounding error so cash can't dip below zero.
    overshoot = np.abs(desired[tradable]).sum() + fee - budget
    total = np.abs(desired[tradable]).sum()
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
    gross_pnl: pd.Series | None = None  # mark-to-market P&L per bar, before costs
    costs: pd.DataFrame | None = None  # per-bar fee / spread / slippage / impact
    fills: pd.DataFrame | None = None  # one row per asset traded per bar
    initial_capital: float = 100_000.0

    def __post_init__(self):
        self.total_fees = float(self.fees_per_period.sum())

    @property
    def gross_value(self) -> pd.Series:
        """Equity of the same trades with no costs deducted."""
        return (self.initial_capital + self.gross_pnl.cumsum()).rename("gross_value")

    @property
    def net_pnl(self) -> float:
        return float(self.portfolio_value.iloc[-1] - self.initial_capital)

    def cost_totals(self) -> dict:
        return {c: float(self.costs[c].sum()) for c in self.costs.columns}


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
        execution_price: str = "close",
        allow_short: bool = False,
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
        if execution_price not in {"close", "open"}:
            raise ValueError("execution_price must be 'close' or 'open'")
        self.execution_price = execution_price
        self.allow_short = allow_short
        self.rebalance_hours_utc = frozenset(int(h) for h in rebalance_hours_utc)
        if any(not 0 <= h < 24 for h in self.rebalance_hours_utc):
            raise ValueError("rebalance_hours_utc must be hours in [0, 24)")

    def _validate_weights(self, target_weights: pd.DataFrame) -> None:
        values = target_weights.to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("target_weights contains infinite values")
        if not self.allow_short and (np.nan_to_num(values) < -_WEIGHT_TOLERANCE).any():
            raise ValueError("target_weights has negative weights: shorting is not allowed "
                             "(pass allow_short=True to model 1x collateralized shorts)")
        gross = np.nansum(np.abs(values), axis=1)
        if (gross > self.max_gross_exposure + _WEIGHT_TOLERANCE).any():
            worst = target_weights.index[int(np.argmax(gross))]
            raise ValueError(
                f"target_weights row sums exceed max_gross_exposure="
                f"{self.max_gross_exposure} (max {gross.max():.4f} at {worst}): "
                f"leverage is not allowed"
            )

    def run(
        self,
        close_wide: pd.DataFrame,
        target_weights: pd.DataFrame,
        open_wide: pd.DataFrame | None = None,
        volume_wide: pd.DataFrame | None = None,
    ) -> BacktestResult:
        """`open_wide` is required for execution_price="open"; `volume_wide`
        (base-asset volume per bar) enables participation reporting and is
        required when the cost model uses market impact."""
        if not close_wide.index.equals(target_weights.index):
            raise ValueError("close_wide and target_weights must share the same index")
        if not close_wide.columns.equals(target_weights.columns):
            raise ValueError("close_wide and target_weights must share the same columns")
        if not close_wide.index.is_monotonic_increasing:
            raise ValueError("close_wide index must be chronologically sorted")
        for name, frame in (("open_wide", open_wide), ("volume_wide", volume_wide)):
            if frame is not None and not (frame.index.equals(close_wide.index) and frame.columns.equals(close_wide.columns)):
                raise ValueError(f"{name} must have the same index and columns as close_wide")
        if self.execution_price == "open" and open_wide is None:
            raise ValueError("execution_price='open' requires open_wide")
        if self.cost_model.uses_impact and volume_wide is None:
            raise ValueError("the cost model uses market impact, which needs volume_wide; "
                             "impact is never modeled without volume data")
        self._validate_weights(target_weights)

        # Shift weights forward by execution_lag: the weight computed at row
        # t is only actually applied at row t + execution_lag.
        applied = np.nan_to_num(target_weights.shift(self.execution_lag).to_numpy(dtype=float))
        closes = close_wide.to_numpy(dtype=float)
        execs = open_wide.to_numpy(dtype=float) if self.execution_price == "open" else closes
        volumes = volume_wide.to_numpy(dtype=float) if volume_wide is not None else None
        n, m = closes.shape
        cm = self.cost_model
        base_rate = cm.cost_rate

        cash = float(self.initial_capital)
        quantities = np.zeros(m)
        last_price = np.full(m, np.nan)

        portfolio_values = np.empty(n)
        total_costs = np.zeros(n)
        cost_parts = np.zeros((n, 4))  # fee, spread, slippage, impact
        gross = np.zeros(n)
        realized_weights = np.zeros((n, m))
        trade_notional = np.zeros((n, m))
        exposures = np.ones(n)
        exposure = 1.0
        fills: list[dict] = []
        if self.risk_overlay is not None:
            self.risk_overlay.reset(cash)
        index = close_wide.index
        index_utc = index.tz_convert("UTC") if index.tz is not None else index
        scheduled = np.isin(index_utc.hour, list(self.rebalance_hours_utc)) if self.rebalance_hours_utc else np.zeros(n, bool)
        columns = list(close_wide.columns)

        for i in range(n):
            pc, pe = closes[i], execs[i]
            close_ok = np.isfinite(pc) & (pc > 0)
            tradable = close_ok & np.isfinite(pe) & (pe > 0)
            # Mark held positions at the execution price (last known price for
            # anything untradable this bar); an asset never priced has no position.
            prev_mark = np.nan_to_num(last_price)
            exec_mark = np.where(tradable, pe, prev_mark)
            gross_pnl = (quantities * (exec_mark - prev_mark)).sum()

            current_value = quantities * exec_mark
            equity = cash + current_value.sum()
            target_row = applied[i] * exposure
            exposures[i] = exposure

            if self.rebalance_threshold > 0 and equity > 0 and not scheduled[i]:
                # Inside the band: leave the position alone this bar (treated
                # exactly like an untradable asset below). Exits always trade.
                drift = np.abs(target_row - current_value / equity)
                tradable = tradable & ((drift >= self.rebalance_threshold) | (target_row == 0))

            if volumes is not None:
                bar_notional = np.nan_to_num(volumes[i]) * np.where(tradable, pe, 0.0)
                intended = np.abs(np.where(tradable, target_row * equity - current_value, 0.0))
                participation = np.divide(intended, bar_notional, out=np.full(m, np.nan), where=bar_notional > 0)
            else:
                participation = np.full(m, np.nan)
            impact_rate = cm.impact_rate(np.nan_to_num(participation, nan=0.0)) if cm.uses_impact else np.zeros(m)
            if cm.uses_impact and np.any(tradable & (np.nan_to_num(volumes[i]) <= 0) & (np.abs(target_row * equity - current_value) > 1e-9)):
                # A trade in a bar with zero recorded volume: impact can't be
                # modeled from data, so refuse rather than assume zero.
                raise ValueError(f"market impact needs positive volume; zero volume at {index[i]}")
            rates = base_rate + impact_rate

            desired = size_orders(target_row, current_value, tradable, equity, rates)

            delta = np.where(tradable, desired - current_value, 0.0)
            # Float residue from re-deriving values (~1e-11 at $100k) is not a
            # trade; without this, dust fills pollute trade counts and days.
            delta[np.abs(delta) < 1e-9 * max(equity, 1.0)] = 0.0
            traded = delta != 0.0
            quantities_new = np.where(traded, desired / np.where(tradable, pe, 1.0), quantities)
            abs_delta = np.abs(delta)
            parts = np.stack([cm.fee_rate * abs_delta, cm.spread_rate * abs_delta,
                              cm.slippage_rate * abs_delta, impact_rate * abs_delta])
            bar_cost = parts.sum()

            cash = equity - (quantities_new * exec_mark).sum() - bar_cost
            quantities = quantities_new
            last_price = np.where(close_ok, pc, last_price)
            close_mark = np.nan_to_num(last_price)
            gross_pnl += (quantities * (close_mark - exec_mark)).sum()
            post_value = cash + (quantities * close_mark).sum()
            if not np.isfinite(post_value):
                raise RuntimeError(f"non-finite portfolio value at {index[i]}; engine state is corrupt")
            if self.risk_overlay is not None:
                exposure = float(self.risk_overlay.update(post_value))

            portfolio_values[i] = post_value
            total_costs[i] = bar_cost
            cost_parts[i] = parts.sum(axis=1)
            gross[i] = gross_pnl
            trade_notional[i] = delta
            if post_value > 0:
                realized_weights[i] = quantities * close_mark / post_value
            for j in np.flatnonzero(traded):
                fills.append({
                    "timestamp": index[i], "symbol": columns[j], "side": "BUY" if delta[j] > 0 else "SELL",
                    "quantity": abs(delta[j]) / pe[j], "price": pe[j], "notional": abs_delta[j],
                    "fee": parts[0, j], "spread": parts[1, j], "slippage": parts[2, j], "impact": parts[3, j],
                    "cost": parts[:, j].sum(), "participation": participation[j],
                    "scheduled": bool(scheduled[i]),
                })

        fill_columns = ["timestamp", "symbol", "side", "quantity", "price", "notional", "fee", "spread",
                        "slippage", "impact", "cost", "participation", "scheduled"]
        return BacktestResult(
            portfolio_value=pd.Series(portfolio_values, index=index, name="portfolio_value"),
            weights_history=pd.DataFrame(realized_weights, index=index, columns=close_wide.columns),
            trade_notional_history=pd.DataFrame(trade_notional, index=index, columns=close_wide.columns),
            fees_per_period=pd.Series(total_costs, index=index, name="fees"),
            exposure=pd.Series(exposures, index=index, name="exposure") if self.risk_overlay is not None else None,
            gross_pnl=pd.Series(gross, index=index, name="gross_pnl"),
            costs=pd.DataFrame(cost_parts, index=index, columns=["fee", "spread", "slippage", "impact"]),
            fills=pd.DataFrame(fills, columns=fill_columns),
            initial_capital=float(self.initial_capital),
        )
