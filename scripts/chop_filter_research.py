"""Choppy-market entry filter: does requiring a clean move before entering
a trend (Kaufman efficiency ratio >= er_min) fix the whipsaw leak?

    python scripts/chop_filter_research.py

Live strategy vs trend_vol_chop_filter over er_window {10, 20, 30} days x
er_min {0.2, 0.3, 0.4}; everything else at the live settings (6-hourly
rebalances, band, base costs, next-bar-open fills).

Pre-registered adoption rule (fixed before running): a variant is adopted
only if it beats the live strategy on TRAIN in both mean 14-day return and
composite score, AND is not worse than live on VALIDATION in mean 14-day
return and continuous return. The holdout is not used.
"""
from __future__ import annotations

import sys
from pathlib import Path

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
from src.strategy.signals import trend_vol_chop_filter, trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def episodes(weights: pd.Series) -> pd.DataFrame:
    """Contiguous full-size (above half of max) long episodes."""
    on = weights > 0.5 * weights.max() if weights.max() > 0 else weights > 1e9
    run = (on != on.shift()).cumsum()
    rows = [(g.index[0], g.index[-1]) for _, g in on.groupby(run) if g.iloc[0]]
    return pd.DataFrame(rows, columns=["start", "end"])


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    base = dict(cfg["trend_vol_target"])
    base["assets"] = tuple(base["assets"])
    policy = cfg["execution_policy"]
    kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
          "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"])}
    variants = {"live (no filter)": (trend_vol_target, base)}
    for days in (10, 20, 30):
        for er_min in (0.2, 0.3, 0.4):
            variants[f"ER {days}d >= {er_min}"] = (trend_vol_chop_filter, {**base, "er_window": days * 24, "er_min": er_min})

    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    rows, reg_rows = [], []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, list(base["assets"]), load_splits()[split_name], resample="1h")
        open_ = load_split_fields(source, panel, ("open",), resample="1h")["open"]
        close = panel.close
        idx = panel.eval_index[base["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(idx)
        windows = rolling_windows(idx, pd.Timedelta(days=14), pd.Timedelta(days=1))
        labels = rb.classify_regimes(close["BTC/USD"].ffill(), ppy, rb.RegimeConfig()).loc[idx]
        btc_eq = close["BTC/USD"].loc[idx] / close["BTC/USD"].loc[idx].iloc[0] * 1e5
        for name, (fn, params) in variants.items():
            w = fn(close, **params).reindex(index=close.index, columns=close.columns).fillna(0.0)
            r = BacktestEngine(SCENARIOS["base"], **kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
            win = evaluate_windows(close, w, windows, SCENARIOS["base"], ppy, kw, open_wide=open_)
            s = an.return_risk_stats(r.portfolio_value, ppy)
            eq_prev = r.portfolio_value.shift(1).fillna(r.initial_capital)
            contrib = r.weights_history.shift(1).fillna(0.0).mul(close.loc[idx].pct_change().fillna(0.0)).mul(eq_prev, axis=0)
            n_eps, n_whip, whip_pnl = 0, 0, 0.0
            for a in base["assets"]:
                e = episodes(w.loc[idx, a])
                n_eps += len(e)
                for ep in e.itertuples():
                    if (ep.end - ep.start) < pd.Timedelta(days=14):
                        n_whip += 1
                        whip_pnl += contrib[a].loc[ep.start:ep.end].sum()
            rows.append({"split": split_name, "variant": name,
                         "win_mean_pct": 100 * win["return"].mean(), "win_p10_pct": 100 * win["return"].quantile(0.1),
                         "win_worst_pct": 100 * win["return"].min(), "win_days_ge_8_pct": 100 * (win.trading_days >= 8).mean(),
                         "cont_return_pct": 100 * s["total_return"], "cont_sharpe": s["sharpe"],
                         "cont_max_dd_pct": 100 * s["max_drawdown"], "cont_composite": s["composite"], "costs": r.total_fees,
                         "trend_entries": n_eps, "whipsaws_lt_14d": n_whip, "whipsaw_pnl": whip_pnl})
            reg = rb.regime_breakdown(r, btc_eq, labels, ppy)
            reg_rows.append(reg[reg.dimension == "trend"].assign(split=split_name, variant=name))
            print(f"  {split_name} {name} done", flush=True)

    t = pd.DataFrame(rows)
    out = REPO_ROOT / "research" / "experiments" / "chop_filter_research.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out, index=False)
    regs = pd.concat(reg_rows)
    pd.set_option("display.width", 260)
    for split_name in ("train", "validation"):
        print(f"\n=== {split_name}")
        print(t[t.split == split_name].drop(columns="split").set_index("variant").round(2).to_string())
        r = regs[regs.split == split_name]
        print("regime returns (%):")
        print(r.pivot(index="variant", columns="regime", values="strategy_return").mul(100).round(1).loc[list(variants)].to_string())

    # Apply the pre-registered rule.
    tr = t[t.split == "train"].set_index("variant")
    va = t[t.split == "validation"].set_index("variant")
    live = "live (no filter)"
    passed = [v for v in variants if v != live
              and tr.loc[v, "win_mean_pct"] > tr.loc[live, "win_mean_pct"] and tr.loc[v, "cont_composite"] > tr.loc[live, "cont_composite"]
              and va.loc[v, "win_mean_pct"] >= va.loc[live, "win_mean_pct"] and va.loc[v, "cont_return_pct"] >= va.loc[live, "cont_return_pct"]]
    print("\nvariants passing the pre-registered rule:", passed or "none -> keep the live strategy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
