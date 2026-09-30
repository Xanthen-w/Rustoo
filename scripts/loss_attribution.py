"""Where does the live strategy make and lose money?

    python scripts/loss_attribution.py

Runs the live configuration on train and validation and splits P&L by coin,
by strategy state (in trend at full size vs the 15% floor), by market
regime (BTC's trailing 30-day return), by individual trend episode (entry to
exit of the trend filter), and by cost type.

Per-bar P&L per coin is computed from the quantities actually held: fills
happen at each bar's open, so the gap from the previous close to the open
is earned by the previous quantity and the move from the open to the close
by the new quantity. This reproduces the engine's gross P&L exactly (the
residual printed should be ~0). The first version multiplied the previous
bar's *weight* by the close-to-close return, which charged each exit bar's
move to the old full-size position and wrongly attributed ~$14k of train
losses to the 15% floor. Bars are labelled with the trend state in force
(decided at the previous close), for the same reason.
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
from src.data.historical import ParquetDataSource  # noqa: E402
from src.features.trend import trend_state  # noqa: E402
from src.strategy.signals import trend_vol_target  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    p = dict(cfg["trend_vol_target"])
    p["assets"] = tuple(p["assets"])
    policy = cfg["execution_policy"]
    kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
          "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"])}
    source = ParquetDataSource(REPO_ROOT / "data" / "binance" / "5m")
    pd.set_option("display.width", 220)
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, list(p["assets"]), load_splits()[split_name], resample="1h")
        open_ = load_split_fields(source, panel, ("open",), resample="1h")["open"]
        close = panel.close
        idx = panel.eval_index[p["trend_span"]:] if split_name == "train" else panel.eval_index
        w = trend_vol_target(close, **p).reindex(index=close.index, columns=close.columns).fillna(0.0)
        r = BacktestEngine(SCENARIOS["base"], **kw).run(close.loc[idx], w.loc[idx], open_wide=open_.loc[idx])
        ppy = an.periods_per_year(idx)

        c, o = close.loc[idx], open_.loc[idx]
        qty = r.weights_history.mul(r.portfolio_value, axis=0) / c.ffill()  # quantity held after each bar's trade
        prev_q, prev_c = qty.shift(1).fillna(0.0), c.ffill().shift(1)
        contrib = (prev_q * (o - prev_c)).fillna(0.0) + (qty * (c.ffill() - o)).fillna(0.0)  # $ per coin per bar
        # The state *in force* for a bar is the one decided at the previous
        # close (fills follow at the next open). Labelling a bar with its own
        # close's state would put the drop that triggers an exit — earned by
        # the full-size position — into the "out of trend" bucket.
        state = pd.DataFrame({a: trend_state(close[a], p["trend_span"], p["band"]).shift(1).loc[idx] for a in p["assets"]})
        labels = rb.classify_regimes(close["BTC/USD"].ffill(), ppy, rb.RegimeConfig()).loc[idx]["trend"]

        net = r.portfolio_value.iloc[-1] - r.initial_capital
        costs = r.total_fees
        explained = contrib.to_numpy().sum() - costs
        print(f"\n################ {split_name}: {idx[0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d}")
        print(f"net P&L ${net:,.0f} = market P&L ${contrib.to_numpy().sum():,.0f} - costs ${costs:,.0f} "
              f"(residual ${net - explained:,.0f})")

        rows = []
        for a in p["assets"]:
            for st, label in ((1.0, "in trend (full size)"), (0.0, f"out of trend ({p['min_exposure']:.0%} floor)")):
                m = state[a] == st
                rows.append({"coin": a.split("/")[0], "state": label, "share_of_time_%": 100 * m.mean(),
                             "avg_weight_%": 100 * r.weights_history[a][m].mean(), "P&L_$": contrib[a][m].sum()})
        print("\n-- P&L by coin and state")
        print(pd.DataFrame(rows).round(1).to_string(index=False))

        print("\n-- P&L by market regime (both coins, before costs)")
        reg = contrib.sum(axis=1).groupby(labels).sum()
        share = labels.value_counts(normalize=True) * 100
        print(pd.DataFrame({"share_of_time_%": share, "P&L_$": reg}).round(0).to_string())

        eps = []
        for a in p["assets"]:
            s = state[a]
            run_id = (s != s.shift()).cumsum()
            for _, g in s.groupby(run_id):
                if g.iloc[0] != 1.0:
                    continue
                start, end = g.index[0], g.index[-1]
                eps.append({"coin": a.split("/")[0], "start": start, "end": end,
                            "days": (end - start).total_seconds() / 86400,
                            "price_move_%": 100 * (close[a].loc[end] / close[a].loc[start] - 1),
                            "P&L_$": contrib[a].loc[start:end].sum()})
        eps = pd.DataFrame(eps)
        if len(eps):
            short = eps[eps.days < 14]
            print(f"\n-- trend episodes (full-size long): {len(eps)} total, {int((eps['P&L_$'] > 0).sum())} profitable")
            print(f"   lasting < 14 days (whipsaws): {len(short)} episodes, P&L ${short['P&L_$'].sum():,.0f}")
            print(f"   lasting >= 14 days:           {len(eps) - len(short)} episodes, P&L ${eps[eps.days >= 14]['P&L_$'].sum():,.0f}")
            print("   worst 5:")
            print(eps.nsmallest(5, "P&L_$").assign(start=lambda d: d.start.dt.strftime("%Y-%m-%d"),
                                                   end=lambda d: d.end.dt.strftime("%Y-%m-%d")).round(1).to_string(index=False))
            print("   best 3:")
            print(eps.nlargest(3, "P&L_$").assign(start=lambda d: d.start.dt.strftime("%Y-%m-%d"),
                                                  end=lambda d: d.end.dt.strftime("%Y-%m-%d")).round(1).to_string(index=False))

        f = r.fills
        big = f.notional > 0.05 * r.initial_capital
        print(f"\n-- costs: ${costs:,.0f} total = {100 * costs / r.initial_capital:.2f}% of capital")
        print(f"   signal changes (fills > $5k): {int(big.sum())} fills, ${f.cost[big].sum():,.0f}")
        print(f"   routine rebalances (<= $5k):  {int((~big).sum())} fills, ${f.cost[~big].sum():,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
