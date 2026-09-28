"""Does adding a short leg help? Live long-only strategy vs trend_vol_long_short.

    python scripts/short_research.py

Everything except the short leg stays at the live settings (coins, trend,
sizing, 15% floor when flat, 5% band, exact rebalances every 6h, next-bar-open
fills, base costs; shorts cost the same 0.1% taker fee on open and close as
Roostoo's short endpoints charge). Grid: short_scale {0.5, 1.0} x
short_entry {3%, 6%, 10%} below the 40-day EMA, covering back at the EMA.

Protocol: choose on train; validation is a one-time check; holdout unused.
Also reports the time spent short, regime breakdown (by BTC's trailing
30-day return) and the worst adverse move against any short episode (a
short's collateral is wiped out if the price doubles).
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
from src.data.historical import ParquetDataSource  # noqa: E402
from src.strategy.signals import trend_vol_long_short, trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def worst_adverse_move(close: pd.Series, weight: pd.Series) -> float:
    """Largest rise of the price above the entry price of any short
    episode (0.5 = +50%); a short's collateral is gone at +100%."""
    short = weight < -1e-9
    worst, entry = 0.0, None
    for t, is_short in short.items():
        price = close.get(t)
        if is_short and entry is None:
            entry = price
        elif is_short and entry is not None and np.isfinite(price):
            worst = max(worst, price / entry - 1)
        elif not is_short:
            entry = None
    return worst


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    base = dict(cfg["trend_vol_target"])
    base["assets"] = tuple(base["assets"])
    policy = cfg["execution_policy"]
    engine_kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
                 "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"]), "allow_short": True}
    variants = {"long-only (live)": (trend_vol_target, base)}
    for scale in (0.5, 1.0):
        for entry in (0.03, 0.06, 0.10):
            variants[f"short x{scale} below -{entry:.0%}"] = (
                trend_vol_long_short, {**base, "short_scale": scale, "short_entry": entry, "short_exit": 0.0})

    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    splits = load_splits()
    rows, regimes = [], []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, list(base["assets"]), splits[split_name], resample="1h")
        open_ = load_split_fields(source, panel, ("open",), resample="1h")["open"]
        close = panel.close
        idx = panel.eval_index[base["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))
        labels = rb.classify_regimes(close["BTC/USD"].ffill(), ppy, rb.RegimeConfig()).loc[idx]
        btc_equity = close["BTC/USD"].loc[idx] / close["BTC/USD"].loc[idx].iloc[0] * 1e5
        for name, (fn, params) in variants.items():
            w = fn(close, **params).reindex(index=close.index, columns=close.columns).fillna(0.0)
            r = BacktestEngine(SCENARIOS["base"], **engine_kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            win = evaluate_windows(close, w, windows, SCENARIOS["base"], ppy, engine_kw, open_wide=open_)
            s = an.return_risk_stats(r.portfolio_value, ppy)
            wl = w.loc[idx]
            rows.append({
                "split": split_name, "variant": name,
                "time_short_pct": 100 * (wl < -1e-9).any(axis=1).mean(),
                "win_mean_pct": 100 * win["return"].mean(), "win_median_pct": 100 * win["return"].median(),
                "win_p_pos_pct": 100 * (win["return"] > 0).mean(), "win_p10_pct": 100 * win["return"].quantile(0.1),
                "win_worst_pct": 100 * win["return"].min(),
                "cont_return_pct": 100 * s["total_return"], "cont_sharpe": s["sharpe"], "cont_sortino": s["sortino"],
                "cont_max_dd_pct": 100 * s["max_drawdown"], "cont_composite": s["composite"], "costs": r.total_fees,
                "worst_move_against_short_pct": 100 * max(worst_adverse_move(close[a].loc[idx], wl[a]) for a in base["assets"]),
            })
            reg = rb.regime_breakdown(r, btc_equity, labels, ppy)
            reg = reg[reg.dimension == "trend"][["regime", "share_of_bars", "strategy_return", "benchmark_return"]]
            regimes.append(reg.assign(split=split_name, variant=name))
            print(f"  {split_name} {name} done", flush=True)

    table = pd.DataFrame(rows)
    out = REPO_ROOT / "research" / "experiments" / "short_research.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    reg = pd.concat(regimes)
    reg.to_csv(out.with_name("short_research_regimes.csv"), index=False)
    pd.set_option("display.width", 260)
    for split_name in ("train", "validation"):
        print(f"\n=== {split_name}")
        print(table[table.split == split_name].drop(columns="split").set_index("variant").round(2).to_string())
        r = reg[reg.split == split_name]
        piv = r.pivot(index="variant", columns="regime", values="strategy_return").mul(100).round(1)
        bench = r.drop_duplicates("regime").set_index("regime")["benchmark_return"].mul(100).round(1)
        share = r.drop_duplicates("regime").set_index("regime")["share_of_bars"].mul(100).round(0)
        print(f"regime returns (%) — BTC: {bench.to_dict()}, share of bars: {share.to_dict()}")
        print(piv.loc[list(variants)].to_string())
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
