"""How does the number of scheduled exact rebalances per day change trading
activity, costs and returns? Everything else stays at the live settings.

    python scripts/rebalance_frequency.py

Schedules: once a day (live: 00:00 UTC), 2x, 4x, 6x and 24x (every hour).
Reported on train and validation, on 14-day windows from cash and as one
continuous run: fills per day, days with a fill, fill sizes, costs, returns.
The holdout is not used.
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
from src.strategy.signals import trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES = {
    "1x/day (live)": (0,),
    "2x/day": (0, 12),
    "4x/day": (0, 6, 12, 18),
    "6x/day": (0, 4, 8, 12, 16, 20),
    "24x/day": tuple(range(24)),
}


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    params = dict(cfg["trend_vol_target"])
    params["assets"] = tuple(params["assets"])
    policy = cfg["execution_policy"]
    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    splits = load_splits()
    rows = []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, list(params["assets"]), splits[split_name], resample="1h")
        open_ = load_split_fields(source, panel, ("open",), resample="1h")["open"]
        close = panel.close
        idx = panel.eval_index[params["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        w = trend_vol_target(close, **params).reindex(index=close.index, columns=close.columns).fillna(0.0)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))
        for label, hours in SCHEDULES.items():
            kw = {"execution_price": policy.get("execution_price", "open"),
                  "rebalance_threshold": policy["rebalance_threshold"], "rebalance_hours_utc": hours}
            r = BacktestEngine(SCENARIOS["base"], **kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            win = evaluate_windows(close, w, windows, SCENARIOS["base"], ppy, kw, open_wide=open_)
            s = an.return_risk_stats(r.portfolio_value, ppy)
            f = r.fills.assign(day=pd.to_datetime(r.fills.timestamp).dt.date)
            days = idx.normalize().nunique()
            rows.append({
                "split": split_name, "schedule": label,
                "fills_per_day": len(f) / days, "days_with_fill_pct": 100 * f.day.nunique() / days,
                "days_with_fill_ge_10usd_pct": 100 * f[f.notional >= 10].day.nunique() / days,
                "median_fill_usd": f.notional.median(),
                "costs_usd": r.total_fees, "cont_return_pct": 100 * s["total_return"], "cont_sharpe": s["sharpe"],
                "win_mean_pct": 100 * win["return"].mean(), "win_p10_pct": 100 * win["return"].quantile(0.1),
                "win_days_ge_8_pct": 100 * (win.trading_days >= 8).mean(), "win_median_trades": win.trades.median(),
            })
            print(f"  {split_name} {label} done", flush=True)
    table = pd.DataFrame(rows)
    out = REPO_ROOT / "research" / "experiments" / "rebalance_frequency.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    for split_name in ("train", "validation"):
        print(f"\n=== {split_name}")
        print(table[table.split == split_name].drop(columns="split").set_index("schedule").round(2).to_string())
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
