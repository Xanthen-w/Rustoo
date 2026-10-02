"""Strategies drawn from a quant-research reading list, vs the live strategy.

    python scripts/reading_list_research.py

The list (about 4,000 links, mostly equities/bonds/options research) was
screened for ideas usable here: spot crypto, OHLCV data only, 0.1% taker
fee, a 14-day horizon and at least one trade a day. Each rule below is the
published one, adapted only where noted:

  xs_momentum    Sparkline, "Crypto Factor Investing": long the top third of
                 liquid coins by 14-day return, a seventh of the book
                 re-formed each day (7-day holding). Long-only and
                 long/short.
  low_variance   Lee & Wang, "Variance Decomposition and Cryptocurrency
                 Return Prediction": coins with high realized variance earn
                 less the following week. Long the lowest-variance third
                 (hourly returns here; the paper uses 5-minute).
  anchoring      Jia et al., "Psychological Anchoring Effect and Cross
                 Section of Cryptocurrency Returns": long the third of coins
                 nearest their 52-week high.
  ibs            Pandey & Joshi, "Using Internal Bar Strength ...": IBS =
                 (close - low) / (high - low) on daily bars; long for one
                 day after IBS < 0.2 (BTC and ETH), or long the lowest-IBS
                 coins of the basket each day.
  mad            Avramov et al., "Market Timing with Moving Average
                 Distance": long while the 21-day average is above the
                 200-day average.
  bollinger      "When Bollinger Meets Edgeworth": contrarian, long below the
                 lower 20-bar band until the price is back at the middle
                 (plain bands; the paper's skew/kurtosis adjustment omitted).
  multi_trend    Man, "In Crypto We Trend": 50/200-day moving-average
                 crossover on ten coins, equal risk. Long-only and
                 long/short.

Universe for the multi-coin rules: the 20 (or 10) pairs with the highest
train-period traded value among pairs with full train history.

Same execution policy as live: next-bar-open fills, base costs, 5% band,
exact rebalances every 6h. Protocol: compare on train; validation is a
check; holdout unused. A rule that needs a long warm-up (200-day average,
52-week high) is scored only on windows where it has a signal, next to the
live strategy on those same windows.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import analytics as an  # noqa: E402
from backtest.costs import SCENARIOS  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402
from backtest.splits import load_split_fields, load_split_panel, load_splits  # noqa: E402
from backtest.windows import evaluate_windows, rolling_windows  # noqa: E402
from scripts.universe_comparison import liquidity_ranking  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402
from src.strategy.signals import buy_and_hold, trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BTC, ETH = "BTC/USD", "ETH/USD"
MAJORS = (BTC, ETH)


def daily(frame: pd.DataFrame, how: str) -> pd.DataFrame:
    """Hourly bars (labelled by close time) -> UTC-day bars, same labelling."""
    return frame.resample("1D", closed="right", label="right").agg(how)


def to_hourly(daily_weights: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """A daily target holds from the day's close until the next one."""
    return daily_weights.reindex(index, method="ffill").fillna(0.0)


def tercile_book(score: pd.DataFrame, hold_days: int, long_short: bool) -> pd.DataFrame:
    """Each day: equal weight in the top third by `score` (minus the bottom
    third if long_short, half the capital per side); the target is the
    average of the last `hold_days` daily books."""
    rank = score.rank(axis=1, pct=True)
    top = (rank > 2 / 3).astype(float)
    book = top.div(top.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    if long_short:
        bottom = (rank <= 1 / 3).astype(float)
        book = 0.5 * book - 0.5 * bottom.div(bottom.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    book[score.notna().sum(axis=1) < 6] = 0.0  # too few ranked coins to form thirds
    return book.rolling(hold_days, min_periods=1).mean()


def xs_momentum(dclose: pd.DataFrame, long_short: bool) -> pd.DataFrame:
    return tercile_book(dclose.pct_change(14), 7, long_short)


def low_variance(close: pd.DataFrame, dclose: pd.DataFrame, long_short: bool) -> pd.DataFrame:
    realized = (np.log(close).diff() ** 2).rolling(168, min_periods=168).sum()
    return tercile_book(-realized.reindex(dclose.index), 7, long_short)


def anchoring(dclose: pd.DataFrame, long_short: bool) -> pd.DataFrame:
    return tercile_book(dclose / dclose.rolling(365, min_periods=365).max(), 7, long_short)


def ibs(dclose: pd.DataFrame, dhigh: pd.DataFrame, dlow: pd.DataFrame) -> pd.DataFrame:
    return (dclose - dlow) / (dhigh - dlow).replace(0, np.nan)


def ibs_threshold(ibs_: pd.DataFrame, assets: tuple, threshold: float = 0.2) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=ibs_.index, columns=ibs_.columns)
    for a in assets:
        weights[a] = (ibs_[a] < threshold).astype(float) / len(assets)
    return weights


def ibs_lowest(ibs_: pd.DataFrame, n: int) -> pd.DataFrame:
    """Long the n lowest-IBS coins of the basket for one day."""
    rank = ibs_.rank(axis=1, method="first")
    return (rank <= n).astype(float) / n


def mad(dclose: pd.DataFrame, assets: tuple) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=dclose.index, columns=dclose.columns)
    for a in assets:
        ratio = dclose[a].rolling(21).mean() / dclose[a].rolling(200, min_periods=200).mean()
        weights[a] = (ratio > 1).astype(float) / len(assets)
    return weights


def bollinger_contrarian(close: pd.DataFrame, assets: tuple, window: int = 20, width: float = 2.0) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in assets:
        mid = close[a].rolling(window).mean()
        lower = mid - width * close[a].rolling(window).std()
        state = pd.Series(np.nan, index=close.index)
        state[close[a] < lower] = 1.0
        state[close[a] >= mid] = 0.0
        weights[a] = state.ffill().fillna(0.0) / len(assets)
    return weights


def multi_trend(dclose: pd.DataFrame, long_short: bool) -> pd.DataFrame:
    """50/200-day crossover per coin, sized by inverse 30-day volatility,
    gross exposure scaled to at most 100%."""
    fast, slow = dclose.rolling(50).mean(), dclose.rolling(200, min_periods=200).mean()
    direction = np.sign(fast - slow)
    if not long_short:
        direction = direction.clip(lower=0.0)
    inv_vol = 1.0 / dclose.pct_change().rolling(30).std()
    raw = (direction * inv_vol).where(slow.notna(), 0.0).fillna(0.0)
    # Equal risk per coin: each coin's share of the book is its inverse vol
    # over the sum across coins that have a signal at all.
    budget = inv_vol.where(slow.notna()).sum(axis=1).replace(0, np.nan)
    return raw.div(budget, axis=0).fillna(0.0)


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    live = dict(cfg["trend_vol_target"])
    live["assets"] = tuple(live["assets"])
    policy = cfg["execution_policy"]
    engine_kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
                 "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"]), "allow_short": True}
    costs = SCENARIOS["base"]
    pd.set_option("display.width", 340)
    pd.set_option("display.max_columns", 40)

    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    splits = load_splits()
    ranked = liquidity_ranking(source, splits["train"])
    top20, top10 = ranked[:20], ranked[:10]
    print("universe (top 20 by train traded value):", ", ".join(p.split("/")[0] for p in top20))

    tables = []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, top20, splits[split_name], resample="1h")
        fields = load_split_fields(source, panel, ("open", "high", "low"), resample="1h")
        close, open_ = panel.close, fields["open"]
        cols20 = [p for p in top20 if p in close.columns]
        cols10 = [p for p in top10 if p in close.columns]
        idx = panel.eval_index[live["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))
        dclose, dhigh, dlow = daily(close, "last"), daily(fields["high"], "max"), daily(fields["low"], "min")
        ibs_ = ibs(dclose, dhigh, dlow)

        def wide(daily_weights: pd.DataFrame) -> pd.DataFrame:
            return to_hourly(daily_weights, close.index).reindex(columns=close.columns).fillna(0.0)

        equal10 = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        equal10[cols10] = close[cols10].notna().astype(float) / len(cols10)
        variants = {
            "BTC buy-and-hold": buy_and_hold(close, BTC),
            "live (trend_vol_target)": trend_vol_target(close, **live),
            "top-10 equal weight, held": equal10,
            "xs momentum 14d, long top third": wide(xs_momentum(dclose[cols20], False)),
            "xs momentum 14d, long/short": wide(xs_momentum(dclose[cols20], True)),
            "low variance, long bottom third": wide(low_variance(close[cols20], dclose[cols20], False)),
            "low variance, long/short": wide(low_variance(close[cols20], dclose[cols20], True)),
            "anchoring (52w high), long top third": wide(anchoring(dclose[cols20], False)),
            "IBS < 0.2, BTC+ETH, 1 day": wide(ibs_threshold(ibs_, MAJORS)),
            "IBS lowest 2 of top 10, 1 day": wide(ibs_lowest(ibs_[cols10], 2)),
            "MAD 21/200d, BTC+ETH": wide(mad(dclose, MAJORS)),
            "Bollinger contrarian, daily bars": wide(bollinger_contrarian(dclose, MAJORS)),
            "Bollinger contrarian, hourly bars": bollinger_contrarian(close, MAJORS),
            "multi-coin trend 50/200d, long only": wide(multi_trend(dclose[cols10], False)),
            "multi-coin trend 50/200d, long/short": wide(multi_trend(dclose[cols10], True)),
        }

        per_window, rows = {}, []
        for name, w in variants.items():
            w = w.reindex(index=close.index, columns=close.columns).fillna(0.0)
            result = BacktestEngine(costs, **engine_kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            win = evaluate_windows(close, w, windows, costs, ppy, engine_kw, open_wide=open_).set_index("start")
            per_window[name] = win
            # Windows that start once the rule has ever held a position.
            held = w.loc[idx].abs().sum(axis=1) > 0
            first = held.idxmax() if held.any() else idx[-1]
            scored = win[win.index >= first]
            ret = scored["return"]
            s = an.return_risk_stats(result.portfolio_value.loc[first:], ppy) if held.any() else {}
            rows.append({
                "variant": name, "windows": len(scored),
                "mean_pct": 100 * ret.mean(), "median_pct": 100 * ret.median(), "p_pos_pct": 100 * (ret > 0).mean(),
                "p10_pct": 100 * ret.quantile(0.1), "worst_pct": 100 * ret.min(), "best_pct": 100 * ret.max(),
                "p_gt10_pct": 100 * (ret > 0.10).mean(), "p_lt_m10_pct": 100 * (ret < -0.10).mean(),
                "p_days_ge8_pct": 100 * (scored["trading_days"] >= 8).mean(), "med_trades": scored["trades"].median(),
                "mean_fees": scored["fees"].mean(), "time_invested_pct": 100 * held.loc[first:].mean(),
                "cont_return_pct": 100 * s.get("total_return", np.nan), "cont_max_dd_pct": 100 * s.get("max_drawdown", np.nan),
            })
            print(f"  {split_name} {name} done", flush=True)
        live_win, btc_win = per_window["live (trend_vol_target)"]["return"], per_window["BTC buy-and-hold"]["return"]
        for row in rows:
            mine = per_window[row["variant"]]["return"].iloc[-row["windows"]:] if row["windows"] else live_win.iloc[:0]
            row["live_mean_same_windows_pct"] = 100 * live_win.loc[mine.index].mean()
            row["btc_mean_same_windows_pct"] = 100 * btc_win.loc[mine.index].mean()
            row["p_beats_live_pct"] = 100 * (mine > live_win.loc[mine.index]).mean()
        table = pd.DataFrame(rows).assign(split=split_name)
        tables.append(table)
        print(f"\n=== {split_name}: 14-day windows (returns in %)")
        print(table.drop(columns="split").set_index("variant").round(1).to_string())

    out = REPO_ROOT / "research" / "experiments" / "reading_list.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(out, index=False)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
