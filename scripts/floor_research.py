"""How big does the out-of-trend floor (min_exposure) need to be?

    python scripts/floor_research.py

The 15% floor was added (2026-09-27) only to guarantee daily trading under
the undefined "enough trades each day" rule. The organizers have since
clarified that an active day is one with at least 1 trade (docs/
COMPETITION_RULES.md). The loss attribution showed the floor losing $17.5k
on train and $6.8k on validation. This compares floors 0/2/5/10% with the
live 15%, all else at the live settings.

Pre-registered adoption rule (written before running): adopt the SMALLEST
floor that
  1. has >= 8 days with a fill of at least $2 (Roostoo's $1 minimum plus a
     margin) in 100% of rolling 14-day windows, on train AND validation;
  2. beats live on TRAIN in mean 14-day return AND composite;
  3. is not worse than live on VALIDATION in mean 14-day return and
     continuous return.
Otherwise keep 15%. Trade sizes are reported for judgement. Holdout unused.
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
FLOORS = [0.15, 0.10, 0.05, 0.02, 0.0]
MIN_FILL_USD = 2.0


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    base = dict(cfg["trend_vol_target"])
    base["assets"] = tuple(base["assets"])
    policy = cfg["execution_policy"]
    kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
          "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"])}
    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    rows = []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, list(base["assets"]), load_splits()[split_name], resample="1h")
        open_ = load_split_fields(source, panel, ("open",), resample="1h")["open"]
        close = panel.close
        idx = panel.eval_index[base["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))
        all_days = pd.Index(sorted(set(idx.date)))
        for floor in FLOORS:
            w = trend_vol_target(close, **{**base, "min_exposure": floor}).reindex(index=close.index, columns=close.columns).fillna(0.0)
            r = BacktestEngine(SCENARIOS["base"], **kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            win = evaluate_windows(close, w, windows, SCENARIOS["base"], ppy, kw, open_wide=open_)
            s = an.return_risk_stats(r.portfolio_value, ppy)
            f = r.fills.assign(day=pd.to_datetime(r.fills.timestamp).dt.date)
            real = f[f.notional >= MIN_FILL_USD]
            active = pd.Series(all_days.isin(real.day.unique()), index=all_days).astype(int)
            rolling_days = active.rolling(14).sum().dropna()
            daily_max = f.groupby("day").notional.max().reindex(all_days).fillna(0.0)
            rows.append({
                "split": split_name, "floor_pct": 100 * floor,
                "days_with_fill_ge_2usd_pct": 100 * active.mean(),
                "worst_14d_window_active_days": int(rolling_days.min()),
                "windows_ge_8_days_pct": 100 * (rolling_days >= 8).mean(),
                "median_fill_usd": f.notional.median(), "median_daily_largest_fill_usd": daily_max.median(),
                "p10_daily_largest_fill_usd": daily_max.quantile(0.1),
                "win_mean_pct": 100 * win["return"].mean(), "win_p10_pct": 100 * win["return"].quantile(0.1),
                "win_worst_pct": 100 * win["return"].min(),
                "cont_return_pct": 100 * s["total_return"], "cont_sharpe": s["sharpe"],
                "cont_max_dd_pct": 100 * s["max_drawdown"], "cont_composite": s["composite"], "costs": r.total_fees,
            })
            print(f"  {split_name} floor {floor:.0%} done", flush=True)
    t = pd.DataFrame(rows)
    out = REPO_ROOT / "research" / "experiments" / "floor_research.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    pd.set_option("display.width", 260)
    for split_name in ("train", "validation"):
        print(f"\n=== {split_name}")
        print(t[t.split == split_name].drop(columns="split").set_index("floor_pct").round(2).to_string())

    tr = t[t.split == "train"].set_index("floor_pct")
    va = t[t.split == "validation"].set_index("floor_pct")
    live = 15.0
    verdict = {}
    for fl in sorted(tr.index):
        if fl == live:
            continue
        checks = {
            "activity": tr.loc[fl, "windows_ge_8_days_pct"] == 100 and va.loc[fl, "windows_ge_8_days_pct"] == 100,
            "train": tr.loc[fl, "win_mean_pct"] > tr.loc[live, "win_mean_pct"] and tr.loc[fl, "cont_composite"] > tr.loc[live, "cont_composite"],
            "validation": va.loc[fl, "win_mean_pct"] >= va.loc[live, "win_mean_pct"] and va.loc[fl, "cont_return_pct"] >= va.loc[live, "cont_return_pct"],
        }
        verdict[fl] = checks
        print(f"floor {fl:>4.0f}%: " + ", ".join(f"{k} {'PASS' if v else 'fail'}" for k, v in checks.items()))
    passing = [fl for fl, c in verdict.items() if all(c.values())]
    print("\nadopt:", f"{min(passing):.0f}% floor (smallest passing)" if passing else "none -> keep the live 15% floor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
