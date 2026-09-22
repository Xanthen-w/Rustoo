"""Performance metrics (Phase 19). Kept deliberately explicit about
annualization assumptions since that's the easiest place for silent bugs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def cumulative_return(portfolio_value: pd.Series) -> float:
    return float(portfolio_value.iloc[-1] / portfolio_value.iloc[0] - 1.0)


def periodic_returns(portfolio_value: pd.Series) -> pd.Series:
    return portfolio_value.pct_change().dropna()


def annualized_return(portfolio_value: pd.Series, periods_per_year: float) -> float:
    total_return = portfolio_value.iloc[-1] / portfolio_value.iloc[0]
    n_periods = len(portfolio_value) - 1
    if n_periods <= 0:
        return 0.0
    return float(total_return ** (periods_per_year / n_periods) - 1.0)


def annualized_volatility(returns: pd.Series, periods_per_year: float) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(returns: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return 0.0
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / periods_per_year
    downside = excess[excess < 0]
    downside_std = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if not downside_std or np.isnan(downside_std):
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(periods_per_year))


def max_drawdown(portfolio_value: pd.Series) -> float:
    """Returned as a negative fraction, e.g. -0.23 for a 23% drawdown."""
    running_max = portfolio_value.cummax()
    drawdown = portfolio_value / running_max - 1.0
    return float(drawdown.min())


def calmar_ratio(portfolio_value: pd.Series, periods_per_year: float) -> float:
    mdd = max_drawdown(portfolio_value)
    if mdd == 0:
        return 0.0
    return float(annualized_return(portfolio_value, periods_per_year) / abs(mdd))


def turnover(weights_history: pd.DataFrame) -> pd.Series:
    """Per-period turnover: sum of absolute weight changes across assets."""
    return weights_history.diff().abs().sum(axis=1)


def composite_score(sortino: float, sharpe: float, calmar: float) -> float:
    """The hackathon's stated finalist scoring formula."""
    return 0.4 * sortino + 0.3 * sharpe + 0.3 * calmar


def trade_stats(trade_pnls: pd.Series) -> dict:
    """Win rate / avg win / avg loss / profit factor / count from a series of
    realized per-trade PnL values (one entry per closed trade)."""
    if len(trade_pnls) == 0:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    gross_profit = wins.sum()
    gross_loss = -losses.sum()
    return {
        "num_trades": int(len(trade_pnls)),
        "win_rate": float(len(wins) / len(trade_pnls)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0,
    }


def summarize(
    portfolio_value: pd.Series,
    weights_history: pd.DataFrame,
    total_fees: float,
    periods_per_year: float,
) -> dict:
    rets = periodic_returns(portfolio_value)
    sharpe = sharpe_ratio(rets, periods_per_year)
    sortino = sortino_ratio(rets, periods_per_year)
    calmar = calmar_ratio(portfolio_value, periods_per_year)
    return {
        "cumulative_return": cumulative_return(portfolio_value),
        "annualized_return": annualized_return(portfolio_value, periods_per_year),
        "annualized_volatility": annualized_volatility(rets, periods_per_year),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_drawdown(portfolio_value),
        "calmar_ratio": calmar,
        "composite_score": composite_score(sortino, sharpe, calmar),
        "avg_turnover": float(turnover(weights_history).mean()),
        "total_fees": total_fees,
    }
