"""Performance, risk, drawdown and trade analytics on a BacktestResult.

Everything here is computed from what the engine recorded — the equity
curves, per-bar costs and the fills ledger — so it reconciles with the
engine by construction and is tested to do so.

Trades: the strategies rebalance continuously rather than taking discrete
round trips, so "trades" are built from fills by FIFO lot matching: each
sale closes the oldest open quantity of that asset. A trade record is one
sale (or the part of it matched to one buy), with its entry/exit prices,
holding time, gross P&L and the costs of both legs. Positions still open at
the end are reported separately as unrealized.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest import metrics

SECONDS_PER_YEAR = 365.25 * 24 * 3600


def periods_per_year(index: pd.DatetimeIndex) -> float:
    step = index.to_series().diff().median()
    return SECONDS_PER_YEAR / step.total_seconds()


# -- returns and risk ------------------------------------------------------------

def _daily(equity: pd.Series) -> pd.Series:
    return equity.resample("1D").last().dropna()


def return_risk_stats(equity: pd.Series, ppy: float) -> dict:
    """Return and risk metrics of one equity curve. VaR/CVaR are historical
    (empirical quantiles) on *daily* returns, reported as positive losses."""
    rets = metrics.periodic_returns(equity)
    daily = _daily(equity).pct_change().dropna()
    weekly = equity.resample("1W").last().dropna().pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).total_seconds() / SECONDS_PER_YEAR
    total = metrics.cumulative_return(equity)
    downside = metrics.downside_deviation(rets) * np.sqrt(ppy)
    gains, losses = rets[rets > 0].sum(), -rets[rets < 0].sum()

    def var_cvar(level: float) -> tuple[float, float]:
        if len(daily) < 2:
            return float("nan"), float("nan")
        q = daily.quantile(1 - level)
        tail = daily[daily <= q]
        return float(-q), float(-tail.mean())

    var95, cvar95 = var_cvar(0.95)
    var99, cvar99 = var_cvar(0.99)
    sharpe = metrics.sharpe_ratio(rets, ppy)
    sortino = metrics.sortino_ratio(rets, ppy)
    calmar = metrics.calmar_ratio(equity, ppy)
    return {
        "total_return": total,
        "cagr": float((1 + total) ** (1 / years) - 1) if years > 0 and total > -1 else float("nan"),
        "annualized_volatility": metrics.annualized_volatility(rets, ppy),
        "downside_volatility": float(downside),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "composite": metrics.composite_score(sortino, sharpe, calmar),
        "omega": float(gains / losses) if losses > 0 else float("inf"),
        "max_drawdown": metrics.max_drawdown(equity),
        "var_95_daily": var95,
        "cvar_95_daily": cvar95,
        "var_99_daily": var99,
        "cvar_99_daily": cvar99,
        "skewness": float(rets.skew()),
        "excess_kurtosis": float(rets.kurt()),
        "worst_day": float(daily.min()) if len(daily) else float("nan"),
        "best_day": float(daily.max()) if len(daily) else float("nan"),
        "worst_week": float(weekly.min()) if len(weekly) else float("nan"),
        "ending_equity": float(equity.iloc[-1]),
        "total_pnl": float(equity.iloc[-1] - equity.iloc[0]),
    }


def drawdown_series(equity: pd.Series) -> pd.Series:
    return (equity / equity.cummax() - 1.0).rename("drawdown")


def drawdown_episodes(equity: pd.Series, min_depth: float = 0.0) -> pd.DataFrame:
    """Every peak -> trough -> recovery episode deeper than `min_depth`
    (a positive fraction). Recovery is NaT for a drawdown still open at the
    end."""
    dd = drawdown_series(equity)
    rows, in_dd, peak_t = [], False, equity.index[0]
    for t, value in dd.items():
        if value < 0 and not in_dd:
            in_dd, start = True, peak_t
        if value == 0:
            if in_dd:
                window = dd.loc[start:t]
                trough_t = window.idxmin()
                rows.append((start, trough_t, t, -window.min()))
                in_dd = False
            peak_t = t
    if in_dd:
        window = dd.loc[start:]
        rows.append((start, window.idxmin(), pd.NaT, -window.min()))
    out = pd.DataFrame(rows, columns=["peak", "trough", "recovery", "depth"])
    out = out[out.depth > min_depth].sort_values("depth", ascending=False).reset_index(drop=True)
    out["time_to_trough"] = out.trough - out.peak
    out["time_to_recovery"] = out.recovery - out.trough
    out["duration"] = out.recovery - out.peak
    return out


def drawdown_summary(equity: pd.Series, significant: float = 0.05) -> dict:
    eps = drawdown_episodes(equity)
    rec = eps.time_to_recovery.dropna()
    top = eps.iloc[0] if len(eps) else None
    return {
        "max_drawdown": float(-top.depth) if top is not None else 0.0,
        "max_dd_peak": top.peak if top is not None else None,
        "max_dd_trough": top.trough if top is not None else None,
        "max_dd_recovery": top.recovery if top is not None else None,
        "average_drawdown": float(-eps.depth.mean()) if len(eps) else 0.0,
        "significant_drawdowns": int((eps.depth >= significant).sum()),
        "significant_threshold": significant,
        "max_recovery_time": rec.max() if len(rec) else None,
        "median_recovery_time": rec.median() if len(rec) else None,
        "longest_drawdown": eps.duration.max() if len(eps) else None,
        "open_drawdown": bool(len(eps) and eps.recovery.isna().any()),
    }


def period_returns(equity: pd.Series, freq: str) -> pd.Series:
    """Compounded returns per calendar period ('ME' month, 'QE' quarter,
    'YE' year), including the first partial period."""
    ends = equity.resample(freq).last().dropna()
    starts = pd.concat([pd.Series([equity.iloc[0]], index=[ends.index[0]]), ends.iloc[:-1]]).to_numpy()
    return pd.Series(ends.to_numpy() / starts - 1, index=ends.index, name=f"return_{freq}")


def monthly_table(equity: pd.Series) -> pd.DataFrame:
    """Year x month grid of monthly returns (for the heatmap)."""
    m = period_returns(equity, "ME")
    return m.to_frame("r").assign(year=m.index.year, month=m.index.month).pivot(index="year", columns="month", values="r")


def rolling_metrics(equity: pd.Series, window_bars: int, ppy: float) -> pd.DataFrame:
    rets = equity.pct_change()
    roll = rets.rolling(window_bars, min_periods=window_bars)
    return pd.DataFrame({
        "rolling_sharpe": roll.mean() / roll.std() * np.sqrt(ppy),
        "rolling_volatility": roll.std() * np.sqrt(ppy),
        "rolling_return": equity / equity.shift(window_bars) - 1,
    })


def consistency(equity: pd.Series) -> dict:
    out = {}
    for label, freq in (("months", "ME"), ("quarters", "QE"), ("years", "YE")):
        r = period_returns(equity, freq)
        out[f"profitable_{label}"] = float((r > 0).mean()) if len(r) else float("nan")
        out[f"n_{label}"] = int(len(r))
    return out


# -- exposure and turnover ----------------------------------------------------------

def exposure_stats(result, ppy: float) -> dict:
    gross = result.weights_history.sum(axis=1)
    traded = result.fills["notional"].sum() if len(result.fills) else 0.0
    years = len(result.portfolio_value) / ppy
    avg_equity = float(result.portfolio_value.mean())
    return {
        "average_exposure": float(gross.mean()),
        "time_in_market": float((gross > 1e-6).mean()),
        "max_exposure": float(gross.max()),
        "traded_notional": float(traded),
        "annual_turnover": float(traded / avg_equity / years) if years > 0 and avg_equity > 0 else float("nan"),
        "fills": int(len(result.fills)),
        "trading_days": int(pd.to_datetime(result.fills["timestamp"]).dt.date.nunique()) if len(result.fills) else 0,
        "max_participation": float(result.fills["participation"].max()) if len(result.fills) and result.fills["participation"].notna().any() else float("nan"),
    }


def cost_summary(result) -> dict:
    totals = result.cost_totals()
    total = sum(totals.values())
    gross = float(result.gross_pnl.sum())
    return {
        **{f"total_{k}": v for k, v in totals.items()},
        "total_costs": total,
        "gross_pnl": gross,
        "net_pnl": result.net_pnl,
        "cost_drag_pct_of_capital": total / result.initial_capital,
        "cost_share_of_gross_pnl": total / gross if gross > 0 else float("nan"),
    }


# -- trades (FIFO lots from fills) -------------------------------------------------

@dataclass
class _Lot:
    qty: float
    price: float
    time: pd.Timestamp
    cost_per_unit: float


def fifo_trades(fills: pd.DataFrame, last_prices: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(closed trades, open positions) from the fills ledger by FIFO matching.
    Buy-leg costs are allocated to the quantity eventually sold."""
    lots: dict[str, deque[_Lot]] = {}
    closed = []
    for f in fills.sort_values("timestamp", kind="stable").itertuples(index=False):
        book = lots.setdefault(f.symbol, deque())
        if f.side == "BUY":
            book.append(_Lot(f.quantity, f.price, f.timestamp, f.cost / f.quantity))
            continue
        remaining, sell_cost_per_unit = f.quantity, f.cost / f.quantity
        while remaining > 1e-12 and book:
            lot = book[0]
            q = min(lot.qty, remaining)
            gross = q * (f.price - lot.price)
            costs = q * (lot.cost_per_unit + sell_cost_per_unit)
            closed.append({
                "symbol": f.symbol, "entry_time": lot.time, "exit_time": f.timestamp, "quantity": q,
                "entry_price": lot.price, "exit_price": f.price, "gross_pnl": gross, "costs": costs,
                "net_pnl": gross - costs, "return_pct": (f.price / lot.price - 1) * 100,
                "holding_time": f.timestamp - lot.time,
            })
            lot.qty -= q
            remaining -= q
            if lot.qty <= 1e-12:
                book.popleft()
        if remaining > 1e-9 * max(f.quantity, 1.0):
            raise ValueError(f"sell of {f.symbol} exceeds recorded holdings by {remaining}")
    open_rows = []
    for symbol, book in lots.items():
        for lot in book:
            if lot.qty <= 1e-12:
                continue
            last = float(last_prices.get(symbol, np.nan))
            open_rows.append({"symbol": symbol, "entry_time": lot.time, "quantity": lot.qty, "entry_price": lot.price,
                              "last_price": last, "unrealized_gross_pnl": lot.qty * (last - lot.price),
                              "entry_costs": lot.qty * lot.cost_per_unit})
    return pd.DataFrame(closed), pd.DataFrame(open_rows)


def trade_statistics(trades: pd.DataFrame, top_n: tuple = (1, 5, 10)) -> dict:
    if trades.empty:
        return {"trades": 0}
    pnl = trades["net_pnl"]
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    signs = np.sign(pnl.to_numpy())

    def longest_run(sign: int) -> int:
        best = run = 0
        for s in signs:
            run = run + 1 if s == sign else 0
            best = max(best, run)
        return best

    total = pnl.sum()
    ordered = pnl.sort_values(ascending=False)
    out = {
        "trades": int(len(trades)),
        "win_rate": float((pnl > 0).mean()),
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "expectancy": float(pnl.mean()),
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) else float("inf"),
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan"),
        "median_holding_time": trades["holding_time"].median(),
        "average_holding_time": trades["holding_time"].mean(),
        "max_consecutive_wins": longest_run(1),
        "max_consecutive_losses": longest_run(-1),
        "total_realized_net_pnl": float(total),
    }
    for n in top_n:
        out[f"top_{n}_share_of_realized_pnl"] = float(ordered.head(n).sum() / total) if total != 0 else float("nan")
        out[f"realized_pnl_without_top_{n}"] = float(total - ordered.head(n).sum())
        out[f"worst_{n}_sum"] = float(ordered.tail(n).sum())
    return out


# -- statistical context --------------------------------------------------------------

def statistical_context(equity: pd.Series, ppy: float, horizon_days: int = 14) -> dict:
    """Sample-size facts so readers can judge significance themselves. The
    Sharpe standard error uses the iid approximation (Lo, 2002) and is
    optimistic when returns are autocorrelated."""
    rets = metrics.periodic_returns(equity)
    n = len(rets)
    days = (equity.index[-1] - equity.index[0]).total_seconds() / 86400
    sr_period = rets.mean() / rets.std() if rets.std() > 0 else 0.0
    se_annual = np.sqrt((1 + 0.5 * sr_period ** 2) / n) * np.sqrt(ppy) if n > 1 else float("nan")
    sharpe = metrics.sharpe_ratio(rets, ppy)
    return {
        "bars": int(n),
        "days": float(days),
        "independent_14d_periods": float(days / horizon_days),
        "sharpe": sharpe,
        "sharpe_standard_error": float(se_annual),
        "sharpe_95ci_low": float(sharpe - 1.96 * se_annual),
        "sharpe_95ci_high": float(sharpe + 1.96 * se_annual),
    }
