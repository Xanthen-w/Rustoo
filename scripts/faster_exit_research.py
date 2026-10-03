"""Would a faster exit help? The live strategy vs versions that leave a
falling coin sooner.

    python scripts/faster_exit_research.py

Two families, everything else (sizing, 5% floor, caps, execution) as live:
  faster trend line    the whole rule on a shorter EMA (20, 10, 7, 3 days),
                       with the live 3% band and a tighter 1% band.
  fast-exit gate       the live 40-day entry, plus: drop to the floor while
                       the close is more than `band` below a short EMA, back
                       in once it recovers above it.

Protocol: compare on train, check on validation, holdout unused. Last, the
same rules on the live bot's first day on the server (Binance hourly
closes from its public API) to see what each would have done in that dip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import analytics as an  # noqa: E402
from backtest.costs import SCENARIOS  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402
from backtest.splits import load_split_fields, load_split_panel, load_splits  # noqa: E402
from backtest.windows import evaluate_windows, rolling_windows  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402
from src.data.live_history import BinanceKlineFeed  # noqa: E402
from src.features import trend as trend_feat  # noqa: E402
from src.features import volatility as vol_feat  # noqa: E402
from src.strategy.signals import _bars_per_year, buy_and_hold, trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BTC, ETH = "BTC/USD", "ETH/USD"


def fast_exit_gate(close_wide: pd.DataFrame, live: dict, exit_span: int, exit_band: float) -> pd.DataFrame:
    """trend_vol_target whose 'in trend' also requires the close not to be
    more than exit_band below a short EMA (hysteresis: back in above it)."""
    weights = pd.DataFrame(0.0, index=close_wide.index, columns=close_wide.columns)
    assets = [a for a in live["assets"] if a in close_wide.columns]
    bpy = _bars_per_year(close_wide.index)
    for asset in assets:
        close = close_wide[asset]
        slow = trend_feat.trend_state(close, live["trend_span"], live["band"])
        e = trend_feat.ema(close, exit_span)
        ok = pd.Series(float("nan"), index=close.index)
        ok[close < e * (1.0 - exit_band)] = 0.0
        ok[close > e] = 1.0
        ok = ok.ffill().fillna(1.0)
        in_trend = slow * ok
        vol = vol_feat.realized_vol(close, live["vol_lookback"], annualize_periods_per_year=bpy)
        size = (live["target_vol"] / len(assets) / vol).clip(upper=1.0)
        weights[asset] = ((in_trend + (1.0 - in_trend) * live["min_exposure"]) * size).fillna(0.0)
    gross = weights.sum(axis=1)
    return weights.div(gross.where(gross > 1.0, 1.0), axis=0)


def variants(close: pd.DataFrame, live: dict) -> dict[str, pd.DataFrame]:
    out = {"BTC buy-and-hold": buy_and_hold(close, BTC), "live: 40d EMA, 3% band": trend_vol_target(close, **live)}
    for days in (20, 10, 7, 3):
        for band in (0.03, 0.01):
            out[f"{days}d EMA, {band:.0%} band"] = trend_vol_target(close, **{**live, "trend_span": days * 24, "band": band})
    for days in (7, 3, 1):
        for band in (0.02, 0.04):
            out[f"live entry + exit {band:.0%} below {days}d EMA"] = fast_exit_gate(close, live, days * 24, band)
    return out


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    live = dict(cfg["trend_vol_target"])
    live["assets"] = tuple(live["assets"])
    policy = cfg["execution_policy"]
    engine_kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
                 "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"])}
    costs = SCENARIOS["base"]
    pd.set_option("display.width", 300)

    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    splits = load_splits()
    tables = []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, [BTC, ETH], splits[split_name], resample="1h")
        open_ = load_split_fields(source, panel, ("open",), resample="1h")["open"]
        close = panel.close
        idx = panel.eval_index[live["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))
        rows, live_ret = [], None
        for name, w in variants(close, live).items():
            w = w.reindex(index=close.index, columns=close.columns).fillna(0.0)
            win = evaluate_windows(close, w, windows, costs, ppy, engine_kw, open_wide=open_)
            r = BacktestEngine(costs, **engine_kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            s = an.return_risk_stats(r.portfolio_value, ppy)
            ret = win["return"]
            if name.startswith("live"):
                live_ret = ret
            rows.append({"variant": name, "mean_pct": 100 * ret.mean(), "median_pct": 100 * ret.median(),
                         "p10_pct": 100 * ret.quantile(0.1), "worst_pct": 100 * ret.min(), "best_pct": 100 * ret.max(),
                         "p_gt10_pct": 100 * (ret > 0.10).mean(), "p_lt_m5_pct": 100 * (ret < -0.05).mean(),
                         "p_lt_m10_pct": 100 * (ret < -0.10).mean(), "mean_fees": win["fees"].mean(),
                         "cont_return_pct": 100 * s["total_return"], "cont_max_dd_pct": 100 * s["max_drawdown"],
                         "_ret": ret})
            print(f"  {split_name} {name} done", flush=True)
        for row in rows:
            row["p_beats_live_pct"] = 100 * (row.pop("_ret").to_numpy() > live_ret.to_numpy()).mean()
        table = pd.DataFrame(rows).assign(split=split_name)
        tables.append(table)
        print(f"\n=== {split_name}: 14-day windows ({len(windows)}), returns in %")
        print(table.drop(columns="split").set_index("variant").round(1).to_string())

    # The live bot's first day on the server: what would each rule have held?
    recent = BinanceKlineFeed("1h").close_panel([BTC, ETH], 2000)
    start = pd.Timestamp("2026-10-02 14:00", tz="UTC")
    print(f"\n=== since the bot started on the server ({start:%Y-%m-%d %H:%M} -> {recent.index[-1]:%Y-%m-%d %H:%M} UTC)")
    print("price change:", {a: f"{100 * (recent[a].iloc[-1] / recent[a].asof(start) - 1):+.1f}%" for a in recent.columns})
    for name, w in variants(recent, live).items():
        w = w.loc[start:]
        r = (w.shift(1) * recent.pct_change().loc[start:]).sum(axis=1)
        print(f"  {name:40s} invested now {100 * w.iloc[-1].sum():5.1f}% | min invested {100 * w.sum(axis=1).min():5.1f}% "
              f"| return before fees {100 * ((1 + r).prod() - 1):+.2f}%")

    out = REPO_ROOT / "research" / "experiments" / "faster_exit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(out, index=False)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
