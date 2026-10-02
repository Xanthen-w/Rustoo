"""Deeper look at the 4-week Donchian breakout (follow-up to
scripts/video_strategies_research.py, where lookback 672 was the best of
eight on train).

    python scripts/donchian_deep_dive.py

Asks, on train and validation (holdout unused):
  A. Is 672 a plateau or a spike? Neighbouring lookbacks, 2 to 8 weeks.
  B. Where does the result come from? Long leg, short leg, each coin,
     half-size shorts, volatility-target sizing.
  C. Does it survive pessimistic costs?
  D. How many independent bets is the evidence? Every position episode.
  E. When does it win or lose against the live strategy? By what BTC did in
     the same 14-day window, and by trailing regime.
  F. How far has price moved against a short (collateral is gone at +100%)?

Same execution policy as live: next-bar-open fills, base costs, 5% band,
exact rebalances every 6h.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import analytics as an  # noqa: E402
from backtest import robustness as rb  # noqa: E402
from backtest.costs import SCENARIOS  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402
from backtest.splits import load_split_fields, load_split_panel, load_splits  # noqa: E402
from backtest.windows import evaluate_windows, rolling_windows  # noqa: E402
from scripts.short_research import worst_adverse_move  # noqa: E402
from scripts.video_strategies_research import donchian_signal, to_weights  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402
from src.features import volatility as vol_feat  # noqa: E402
from src.strategy.signals import buy_and_hold, trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BTC, ETH = "BTC/USD", "ETH/USD"
LOOKBACK = 672
NEIGHBOURS = (336, 504, 588, 672, 756, 840, 1008, 1344)
WARMUP_BARS = max(NEIGHBOURS)


def donchian(close: pd.DataFrame, lookback: int = LOOKBACK, assets: tuple = (BTC, ETH), long_scale: float = 1.0,
             short_scale: float = 1.0) -> pd.DataFrame:
    """Equal capital per asset; long and short legs scaled separately."""
    signals = {}
    for a in assets:
        s = donchian_signal(close[a], lookback)
        signals[a] = s.clip(lower=0.0) * long_scale + s.clip(upper=0.0) * short_scale
    return to_weights(close, signals, long_short=True)


def donchian_vol_sized(close: pd.DataFrame, live: dict, lookback: int = LOOKBACK) -> pd.DataFrame:
    """Donchian direction with the live strategy's volatility-target sizing."""
    ppy = an.periods_per_year(close.index)
    weights = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for a in (BTC, ETH):
        vol = vol_feat.realized_vol(close[a], live["vol_lookback"], annualize_periods_per_year=ppy)
        size = (live["target_vol"] / 2 / vol).clip(upper=1.0)
        weights[a] = (donchian_signal(close[a], lookback) * size).fillna(0.0)
    gross = weights.abs().sum(axis=1)
    return weights.div(gross.where(gross > 1.0, 1.0), axis=0)


def episodes(close: pd.Series, signal: pd.Series, idx: pd.DatetimeIndex, cost: float) -> pd.DataFrame:
    """One row per stretch of constant position inside `idx`: price return
    in the position's direction, net of one entry and one exit."""
    sig, px = signal.loc[idx], close.loc[idx]
    group = (sig != sig.shift()).cumsum()
    rows = []
    for _, block in sig.groupby(group):
        side = block.iloc[0]
        if side == 0:
            continue
        entry, exit_ = px.loc[block.index[0]], px.loc[block.index[-1]]
        rows.append({"side": "long" if side > 0 else "short", "start": block.index[0], "days": len(block) / 24,
                     "return_pct": 100 * (side * (exit_ / entry - 1) - 2 * cost)})
    return pd.DataFrame(rows)


def window_row(name: str, win: pd.DataFrame, live_win: pd.DataFrame, btc_win: pd.DataFrame) -> dict:
    ret = win["return"]
    return {"variant": name, "mean_pct": 100 * ret.mean(), "median_pct": 100 * ret.median(),
            "p_pos_pct": 100 * (ret > 0).mean(), "p10_pct": 100 * ret.quantile(0.1), "p90_pct": 100 * ret.quantile(0.9),
            "worst_pct": 100 * ret.min(), "best_pct": 100 * ret.max(),
            "p_gt10_pct": 100 * (ret > 0.10).mean(), "p_lt_m10_pct": 100 * (ret < -0.10).mean(),
            "p_days_ge8_pct": 100 * (win["trading_days"] >= 8).mean(), "mean_fees": win["fees"].mean(),
            "p_beats_btc_pct": 100 * (ret.to_numpy() > btc_win["return"].to_numpy()).mean(),
            "p_beats_live_pct": 100 * (ret.to_numpy() > live_win["return"].to_numpy()).mean()}


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    live = dict(cfg["trend_vol_target"])
    live["assets"] = tuple(live["assets"])
    policy = cfg["execution_policy"]
    engine_kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
                 "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"]), "allow_short": True}
    base_costs = SCENARIOS["base"]
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)

    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    splits = load_splits()
    tables = []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, [BTC, ETH], splits[split_name], resample="1h")
        open_ = load_split_fields(source, panel, ("open",), resample="1h")["open"]
        close = panel.close
        idx = panel.eval_index[WARMUP_BARS:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))

        def run(weights, costs=base_costs):
            w = weights.reindex(index=close.index, columns=close.columns).fillna(0.0)
            result = BacktestEngine(costs, **engine_kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            return w, result, evaluate_windows(close, w, windows, costs, ppy, engine_kw, open_wide=open_)

        _, _, btc_win = run(buy_and_hold(close, BTC))
        _, live_res, live_win = run(trend_vol_target(close, **live))
        variants = {f"Donchian {lb} long/short": donchian(close, lb) for lb in NEIGHBOURS}
        variants.update({
            "672 long leg only": donchian(close, short_scale=0.0),
            "672 short leg only": donchian(close, long_scale=0.0),
            "672 half-size shorts": donchian(close, short_scale=0.5),
            "672 BTC only long/short": donchian(close, assets=(BTC,)),
            "672 ETH only long/short": donchian(close, assets=(ETH,)),
            "672 long/short, vol-target sizing": donchian_vol_sized(close, live),
        })
        rows = [window_row("BTC buy-and-hold", btc_win, live_win, btc_win), window_row("live (trend_vol_target)", live_win, live_win, btc_win)]
        results = {}
        for name, weights in variants.items():
            w, result, win = run(weights)
            results[name] = (w, result, win)
            s = an.return_risk_stats(result.portfolio_value, ppy)
            rows.append({**window_row(name, win, live_win, btc_win), "cont_return_pct": 100 * s["total_return"],
                         "cont_max_dd_pct": 100 * s["max_drawdown"]})
            print(f"  {split_name} {name} done", flush=True)
        _, result, win = run(variants["Donchian 672 long/short"], SCENARIOS["pessimistic"])
        s = an.return_risk_stats(result.portfolio_value, ppy)
        rows.append({**window_row("Donchian 672 long/short, pessimistic costs", win, live_win, btc_win),
                     "cont_return_pct": 100 * s["total_return"], "cont_max_dd_pct": 100 * s["max_drawdown"]})
        table = pd.DataFrame(rows).assign(split=split_name)
        tables.append(table)
        print(f"\n=== {split_name}: 14-day windows ({len(windows)} windows, returns in %)")
        print(table.drop(columns="split").set_index("variant").round(1).to_string())

        w, result, win = results["Donchian 672 long/short"]
        print(f"\n--- {split_name}: every position episode, Donchian 672 (net of entry and exit cost)")
        for a in (BTC, ETH):
            ep = episodes(close[a], donchian_signal(close[a], LOOKBACK), idx, base_costs.cost_rate)
            print(a, f"{len(ep)} episodes, {(ep.return_pct > 0).sum()} winners")
            print(ep.assign(start=ep.start.dt.strftime("%Y-%m-%d")).round(1).to_string(index=False))
        print(f"worst move against a short: {100 * max(worst_adverse_move(close[a].loc[idx], w[a].loc[idx]) for a in (BTC, ETH)):.1f}%")

        # By what BTC did in the same window: where the two strategies differ.
        joined = pd.DataFrame({"btc": btc_win["return"], "live": live_win["return"], "donchian": win["return"]})
        bucket = pd.cut(joined.btc, [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf],
                        labels=["BTC < -10%", "-10% to -3%", "-3% to +3%", "+3% to +10%", "BTC > +10%"])
        by = joined.groupby(bucket, observed=True).agg(windows=("btc", "size"), btc=("btc", "mean"), live=("live", "mean"),
                                                         donchian=("donchian", "mean"), donchian_worst=("donchian", "min"))
        by[["btc", "live", "donchian", "donchian_worst"]] *= 100
        print(f"\n--- {split_name}: mean 14-day return (%) by BTC's return in the same window")
        print(by.round(1).to_string())

        labels = rb.classify_regimes(close[BTC].ffill(), ppy, rb.RegimeConfig()).loc[idx]
        btc_equity = close[BTC].loc[idx] / close[BTC].loc[idx].iloc[0] * 1e5
        print(f"\n--- {split_name}: return (%) by trailing 30-day BTC regime")
        for name, res in (("live", live_res), ("donchian 672", result)):
            reg = rb.regime_breakdown(res, btc_equity, labels, ppy)
            reg = reg[reg.dimension == "trend"].set_index("regime")
            print(name, (reg["strategy_return"] * 100).round(1).to_dict(), "| BTC", (reg["benchmark_return"] * 100).round(1).to_dict(),
                  "| share of time", (reg["share_of_bars"] * 100).round(0).to_dict())

    out = REPO_ROOT / "research" / "experiments" / "donchian_deep_dive.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(out, index=False)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
