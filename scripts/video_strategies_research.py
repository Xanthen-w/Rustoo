"""Two published strategies (neurotrader, YouTube) vs the live strategy.

    python scripts/video_strategies_research.py

1. Modified Donchian breakout (video ncJKep-6vU8): long when the hourly
   close breaks above the highest close of the previous `lookback - 1`
   bars, reversed to short when it breaks below the lowest; always in the
   market. Published lookback: 72 (3 days).
2. Intermarket difference (video n2mY86S01fg): cmma = (close - SMA(lookback))
   / (ATR(168) * sqrt(lookback)) on ETH and BTC; indicator = ETH - BTC. Long
   ETH when it crosses above +threshold, short below -threshold, flat again
   when it returns to zero. Published: lookback 24, threshold 0.25, and
   results WITHOUT transaction costs.

Each is run as published (long/short) and long/flat (the live bot does not
short). Everything else matches the live policy: next-bar-open fills, base
costs (0.1% taker + 5 bps slippage), 5% band, exact rebalances every 6h.

Protocol: train shows the parameter sweeps; validation is a check of the
published parameters; holdout unused. Headline numbers are the distribution
over every 14-day window, the competition's horizon.
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
from src.data.historical import ParquetDataSource  # noqa: E402
from src.strategy.signals import buy_and_hold, trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BTC, ETH = "BTC/USD", "ETH/USD"
DONCHIAN_LOOKBACKS = (12, 24, 48, 72, 120, 168, 336, 672)
DONCHIAN_TRAIN_BEST = 672  # best mean 14-day return on train, both modes (2026-10-02)
IMD_LOOKBACKS = (12, 24, 48, 96, 168)
IMD_THRESHOLDS = (0.1, 0.25, 0.5, 0.75)


def donchian_signal(close: pd.Series, lookback: int) -> pd.Series:
    """+1 after an upward close breakout, -1 after a downward one, 0 before
    the first. The channel is the previous lookback - 1 closes (lagged one
    bar), as in the published code."""
    upper = close.shift(1).rolling(lookback - 1).max()
    lower = close.shift(1).rolling(lookback - 1).min()
    sig = pd.Series(np.nan, index=close.index)
    sig[close > upper] = 1.0
    sig[close < lower] = -1.0
    return sig.ffill().fillna(0.0)


def cmma(close: pd.Series, high: pd.Series, low: pd.Series, lookback: int, atr_lookback: int = 168) -> pd.Series:
    """Close minus moving average, in units of ATR * sqrt(lookback)."""
    prev = close.shift(1)
    true_range = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = true_range.ewm(alpha=1.0 / atr_lookback, min_periods=atr_lookback, adjust=False).mean()
    return (close - close.rolling(lookback).mean()) / (atr * lookback ** 0.5)


def threshold_revert_signal(indicator: pd.Series, threshold: float) -> pd.Series:
    """+1 above threshold, -1 below -threshold, held until the indicator
    is back at or through zero."""
    out = np.zeros(len(indicator))
    position = 0.0
    for i, value in enumerate(indicator.to_numpy()):
        if value > threshold:
            position = 1.0
        elif value < -threshold:
            position = -1.0
        if (position == 1.0 and value <= 0) or (position == -1.0 and value >= 0):
            position = 0.0
        out[i] = position
    return pd.Series(out, index=indicator.index)


def to_weights(close: pd.DataFrame, signals: dict[str, pd.Series], long_short: bool) -> pd.DataFrame:
    """Signals in {-1, 0, 1} per asset -> equal capital share per asset."""
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for asset, sig in signals.items():
        weights[asset] = (sig if long_short else sig.clip(lower=0.0)) / len(signals)
    return weights


def donchian_weights(close: pd.DataFrame, assets: tuple, lookback: int, long_short: bool) -> pd.DataFrame:
    return to_weights(close, {a: donchian_signal(close[a], lookback) for a in assets}, long_short)


def intermarket_signal(close, high, low, lookback: int, threshold: float) -> pd.Series:
    diff = cmma(close[ETH], high[ETH], low[ETH], lookback) - cmma(close[BTC], high[BTC], low[BTC], lookback)
    return threshold_revert_signal(diff, threshold)


def profit_factor(signal: pd.Series, close: pd.Series, idx: pd.DatetimeIndex, cost: float) -> dict:
    """The videos' own measure: signal x next-bar log return, with `cost`
    charged per unit of signal change. Returns profit factor and trades/yr."""
    ret = signal * np.log(close).diff().shift(-1) - cost * signal.diff().abs().fillna(0.0)
    ret = ret.loc[idx].dropna()
    gains, losses = ret[ret > 0].sum(), -ret[ret < 0].sum()
    years = (idx[-1] - idx[0]).total_seconds() / (365.25 * 86400)
    flips = (signal.loc[idx].diff().abs() > 0).sum()
    return {"pf": gains / losses if losses else float("nan"), "log_return": ret.sum(), "signal_changes_per_year": flips / years}


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    live = dict(cfg["trend_vol_target"])
    live["assets"] = tuple(live["assets"])
    policy = cfg["execution_policy"]
    engine_kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
                 "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"]), "allow_short": True}
    costs = SCENARIOS["base"]

    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    splits = load_splits()
    rows, sweeps = [], []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, [BTC, ETH], splits[split_name], resample="1h")
        fields = load_split_fields(source, panel, ("open", "high", "low"), resample="1h")
        close, open_, high, low = panel.close, fields["open"], fields["high"], fields["low"]
        idx = panel.eval_index[live["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))

        imd = intermarket_signal(close, high, low, 24, 0.25)
        variants = {
            "BTC buy-and-hold": buy_and_hold(close, BTC),
            "live (trend_vol_target)": trend_vol_target(close, **live),
            "Donchian 72 BTC+ETH long/short": donchian_weights(close, (BTC, ETH), 72, True),
            "Donchian 72 BTC+ETH long/flat": donchian_weights(close, (BTC, ETH), 72, False),
            "Donchian 72 BTC long/short": donchian_weights(close, (BTC,), 72, True),
            "Donchian 72 ETH long/short": donchian_weights(close, (ETH,), 72, True),
            "Intermarket 24/0.25 ETH long/short": to_weights(close, {ETH: imd}, True),
            "Intermarket 24/0.25 ETH long/flat": to_weights(close, {ETH: imd}, False),
        }
        # Parameter sweep as full 14-day-window runs on train; validation
        # only checks the published lookback and the train-best one.
        for lb in DONCHIAN_LOOKBACKS if split_name == "train" else (DONCHIAN_TRAIN_BEST,):
            if lb != 72:
                variants[f"Donchian {lb} BTC+ETH long/short"] = donchian_weights(close, (BTC, ETH), lb, True)
                variants[f"Donchian {lb} BTC+ETH long/flat"] = donchian_weights(close, (BTC, ETH), lb, False)

        per_window = {}
        for name, w in variants.items():
            w = w.reindex(index=close.index, columns=close.columns).fillna(0.0)
            r = BacktestEngine(costs, **engine_kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            win = evaluate_windows(close, w, windows, costs, ppy, engine_kw, open_wide=open_)
            per_window[name] = win.set_index("start")
            s = an.return_risk_stats(r.portfolio_value, ppy)
            ret = win["return"]
            rows.append({
                "split": split_name, "variant": name,
                "mean_pct": 100 * ret.mean(), "median_pct": 100 * ret.median(), "p_pos_pct": 100 * (ret > 0).mean(),
                "p10_pct": 100 * ret.quantile(0.1), "p90_pct": 100 * ret.quantile(0.9),
                "worst_pct": 100 * ret.min(), "best_pct": 100 * ret.max(),
                "p_gt10_pct": 100 * (ret > 0.10).mean(), "p_lt_m10_pct": 100 * (ret < -0.10).mean(),
                "med_days": win["trading_days"].median(), "p_days_ge8_pct": 100 * (win["trading_days"] >= 8).mean(),
                "med_trades": win["trades"].median(), "mean_fees": win["fees"].mean(),
                "med_composite": win["composite_score"].median(),
                "cont_return_pct": 100 * s["total_return"], "cont_max_dd_pct": 100 * s["max_drawdown"],
                "time_short_pct": 100 * (w.loc[idx] < -1e-9).any(axis=1).mean(),
            })
            print(f"  {split_name} {name} done", flush=True)
        for row in rows:
            if row["split"] == split_name:
                mine = per_window[row["variant"]]["return"]
                row["p_beats_btc_pct"] = 100 * (mine > per_window["BTC buy-and-hold"]["return"]).mean()
                row["p_beats_live_pct"] = 100 * (mine > per_window["live (trend_vol_target)"]["return"]).mean()

        # The videos' own measure (profit factor of signal x next log return),
        # without costs as published and with our cost per unit of turnover.
        for lb in DONCHIAN_LOOKBACKS:
            for asset in (BTC, ETH):
                sig = donchian_signal(close[asset], lb)
                for label, c in (("gross", 0.0), ("net", costs.cost_rate)):
                    sweeps.append({"split": split_name, "strategy": f"donchian {asset[:3]}", "lookback": lb, "threshold": np.nan,
                                   "costs": label, **profit_factor(sig, close[asset], idx, c)})
        for lb in IMD_LOOKBACKS:
            for th in IMD_THRESHOLDS:
                sig = intermarket_signal(close, high, low, lb, th)
                for label, c in (("gross", 0.0), ("net", costs.cost_rate)):
                    sweeps.append({"split": split_name, "strategy": "intermarket ETH", "lookback": lb, "threshold": th,
                                   "costs": label, **profit_factor(sig, close[ETH], idx, c)})

    table, sweep = pd.DataFrame(rows), pd.DataFrame(sweeps)
    out = REPO_ROOT / "research" / "experiments" / "video_strategies.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    sweep.to_csv(out.with_name("video_strategies_sweep.csv"), index=False)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    for split_name in ("train", "validation"):
        print(f"\n=== {split_name}: 14-day windows (returns in %)")
        print(table[table.split == split_name].drop(columns="split").set_index("variant").round(1).to_string())
        s = sweep[sweep.split == split_name]
        d = s[s.strategy.str.startswith("donchian")]
        print(f"\n--- {split_name}: Donchian profit factor by lookback (long/short, per-bar)")
        print(d.pivot_table(index="lookback", columns=["strategy", "costs"], values="pf").round(3).to_string())
        print(f"signal changes per year: {d[d.costs == 'net'].groupby('lookback').signal_changes_per_year.mean().round(0).to_dict()}")
        i = s[s.strategy == "intermarket ETH"]
        for label in ("gross", "net"):
            print(f"\n--- {split_name}: intermarket profit factor, {label} (rows lookback, cols threshold)")
            print(i[i.costs == label].pivot(index="lookback", columns="threshold", values="pf").round(3).to_string())
        print("signal changes per year:")
        print(i[i.costs == "net"].pivot(index="lookback", columns="threshold", values="signal_changes_per_year").round(0).to_string())
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
