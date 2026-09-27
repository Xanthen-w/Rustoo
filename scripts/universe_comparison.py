"""Which coins should the live strategy trade? Compares the live
trend_vol_target settings across pre-specified coin sets.

    python scripts/universe_comparison.py                 # train (selection) + validation (confirmation)

Coin sets are fixed up front: BTC only, BTC+ETH (live), BTC+ETH+TRX,
BTC+TRX, and the top 5 / top 10 coins by *train-period* traded value
(among coins with full train history). Each set is evaluated on every
14-day window from cash (the competition horizon) and as one continuous
run. Protocol: pick on train; validation is a one-time check. The holdout
is not used.
"""
from __future__ import annotations

import argparse
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


def liquidity_ranking(source: ParquetDataSource, split) -> list[str]:
    """Pairs with full train history, ranked by traded value (quote volume) over the train split."""
    totals = {}
    for pair in source.available_pairs():
        df = source.load(pair, split.start, split.end - pd.Timedelta(microseconds=1))
        if df.empty or df.index[0] > split.start + pd.Timedelta(days=2) or "quote_volume" not in df:
            continue
        totals[pair] = float(df["quote_volume"].sum())
    return sorted(totals, key=totals.get, reverse=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "research" / "experiments" / "universe_comparison.csv")
    args = parser.parse_args()

    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    base = dict(cfg["trend_vol_target"])
    policy = cfg["execution_policy"]
    engine_kw = {"execution_price": policy.get("execution_price", "open"), "rebalance_threshold": policy["rebalance_threshold"],
                 "rebalance_hours_utc": tuple(policy["rebalance_hours_utc"])}
    source = ParquetDataSource(args.data)
    splits = load_splits()
    ranked = liquidity_ranking(source, splits["train"])
    print("train-period liquidity ranking:", ", ".join(p.split("/")[0] for p in ranked[:12]))
    sets = {
        "BTC": ["BTC/USD"],
        "BTC+ETH (live)": ["BTC/USD", "ETH/USD"],
        "BTC+ETH+TRX": ["BTC/USD", "ETH/USD", "TRX/USD"],
        "BTC+TRX": ["BTC/USD", "TRX/USD"],
        "Top 5 by liquidity": ranked[:5],
        "Top 10 by liquidity": ranked[:10],
    }
    for name, s in sets.items():
        print(f"  {name}: {', '.join(p.split('/')[0] for p in s)}")

    pairs = sorted({p for s in sets.values() for p in s})
    rows = []
    for split_name in ("train", "validation"):
        panel = load_split_panel(source, pairs, splits[split_name], resample="1h")
        fields = load_split_fields(source, panel, ("open",), resample="1h")
        close, open_ = panel.close, fields["open"]
        eval_index = panel.eval_index[base["trend_span"]:] if split_name == "train" else panel.eval_index
        ppy = an.periods_per_year(eval_index)
        windows = rolling_windows(eval_index, pd.Timedelta(days=14), pd.Timedelta(days=1))
        for name, assets in sets.items():
            params = {**base, "assets": tuple(assets)}
            w = trend_vol_target(close, **params).reindex(index=close.index, columns=close.columns).fillna(0.0)
            win = evaluate_windows(close, w, windows, SCENARIOS["base"], ppy, engine_kw, open_wide=open_)
            r = BacktestEngine(SCENARIOS["base"], **engine_kw).run(close.loc[eval_index], w.loc[eval_index], open_wide=open_.loc[eval_index])
            s = an.return_risk_stats(r.portfolio_value, ppy)
            rows.append({
                "split": split_name, "coins": name, "n": len(assets),
                "win_mean": win["return"].mean(), "win_median": win["return"].median(), "win_p_pos": (win["return"] > 0).mean(),
                "win_p10": win["return"].quantile(0.1), "win_worst": win["return"].min(),
                "win_days_ge_8": (win["trading_days"] >= 8).mean(),
                "cont_return": s["total_return"], "cont_sharpe": s["sharpe"], "cont_max_dd": s["max_drawdown"],
                "cont_composite": s["composite"], "costs": r.total_fees,
            })
            print(f"  {split_name:10s} {name:22s} done", flush=True)
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    pd.set_option("display.width", 250)
    for split_name in ("train", "validation"):
        t = table[table.split == split_name].drop(columns="split").set_index("coins")
        pct = ["win_mean", "win_median", "win_p_pos", "win_p10", "win_worst", "win_days_ge_8", "cont_return", "cont_max_dd"]
        t[pct] = t[pct] * 100
        print(f"\n=== {split_name} (percent where applicable)")
        print(t.round(2).to_string())
    print(f"\nsaved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
