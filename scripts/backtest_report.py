"""Full backtest report for the live strategy (or any registered strategy).

    python scripts/backtest_report.py                          # live strategy, train split
    python scripts/backtest_report.py --split validation
    python scripts/backtest_report.py --split test --use-holdout
    python scripts/backtest_report.py --costs pessimistic --seeds 50

Writes research/experiments/reports/<name>/report.html (self-contained,
open it in a browser) plus summary.json, config.json, equity.csv,
fills.csv, trades.csv, open_positions.csv, drawdowns.csv,
random_entry.csv and the 14-day window tables.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.costs import SCENARIOS  # noqa: E402
from backtest.registry import RunRegistry  # noqa: E402
from backtest.report import ReportConfig, run_report  # noqa: E402
from backtest.splits import DataSplit, load_split_fields, load_split_panel, load_splits  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402
from src.strategy.signals import STRATEGIES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    strategy_cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    policy = strategy_cfg.get("execution_policy", {})
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test", "custom"])
    parser.add_argument("--start", help="With --split custom: first day (UTC), e.g. 2025-03-01.")
    parser.add_argument("--end", help="With --split custom: day after the last day (UTC, exclusive).")
    parser.add_argument("--use-holdout", action="store_true", help="Required for --split test, or a custom range reaching into it.")
    parser.add_argument("--strategy", default="trend_vol_target", choices=sorted(STRATEGIES))
    parser.add_argument("--params", type=json.loads, default=None,
                        help="JSON strategy params (default: the live config in config/strategy.yaml).")
    parser.add_argument("--costs", default="base", choices=sorted(SCENARIOS), help="Cost preset; the flags below override parts of it.")
    parser.add_argument("--fee-bps", type=float, default=None, help="Taker fee in bps (maker = half).")
    parser.add_argument("--maker-share", type=float, default=None, help="Assumed share of maker fills, 0-1.")
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--spread-bps", type=float, default=None, help="Full modeled spread; half is paid per trade.")
    parser.add_argument("--execution-price", default=policy.get("execution_price", "open"), choices=["open", "close"])
    parser.add_argument("--latency-bars", type=int, default=1, help="Bars between signal and fill (>= 1).")
    parser.add_argument("--rebalance-threshold", type=float, default=None, help="Weight band (default: live config for trend_vol_target, else 0).")
    parser.add_argument("--rebalance-hours", type=int, nargs="*", default=None, help="UTC hours of exact rebalances (default: live config).")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--seeds", type=int, default=20, help="Random-entry baseline seeds.")
    parser.add_argument("--benchmarks", nargs="*", default=["BTC/USD", "ETH/USD"])
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.params is None:
        params = dict(strategy_cfg[args.strategy]) if args.strategy in strategy_cfg else {}
    else:
        params = args.params
    if "assets" in params:
        params["assets"] = tuple(params["assets"])
    source = ParquetDataSource(args.data)
    available = set(source.available_pairs())
    benchmarks = [b for b in args.benchmarks if b in available]
    if len(benchmarks) < len(args.benchmarks):
        print(f"note: benchmarks not in this data source were dropped: {sorted(set(args.benchmarks) - set(benchmarks))}")
    needed = set(params.get("assets", ())) | ({params["symbol"]} if "symbol" in params else set())
    if needed - available:
        print(f"error: {sorted(needed - available)} not in data source {args.data}")
        return 2
    pairs = sorted(needed | set(benchmarks))

    splits = load_splits()
    if args.split == "custom":
        if not (args.start and args.end):
            print("error: --split custom needs --start and --end")
            return 2
        split = DataSplit("custom", pd.Timestamp(args.start, tz="UTC"), pd.Timestamp(args.end, tz="UTC"))
        holdout = splits["test"]
        if split.end > holdout.start and not args.use_holdout:
            print(f"error: this range reaches into the holdout (from {holdout.start:%Y-%m-%d}); pass --use-holdout if that's intended")
            return 2
    else:
        split = splits[args.split]
    bar = policy.get("bar", "1h")
    panel = load_split_panel(source, pairs, split, resample=bar, allow_holdout=args.use_holdout)
    fields = load_split_fields(source, panel, ("open", "volume"), resample=bar)
    eval_index = panel.eval_index
    warm = max([v for k, v in params.items() if k.endswith(("span", "lookback")) and isinstance(v, int)] or [0])
    if panel.close.index[0] >= split.start:
        # The period starts at the beginning of the data: skip the indicator
        # warm-up (longest lookback) so the strategy isn't scored while it's
        # still forced into cash.
        eval_index = eval_index[warm:]
    if len(eval_index) < 24:
        print(f"error: only {len(eval_index)} bars left to evaluate; the strategy needs {warm} bars of history "
              f"before the first scored bar and the data has {len(panel.close)}. Use a longer dataset or shorter lookbacks.")
        return 2

    base_costs = SCENARIOS[args.costs]
    overrides = {}
    if args.fee_bps is not None:
        overrides.update(taker_fee=args.fee_bps / 1e4, maker_fee=args.fee_bps / 2e4)
    if args.maker_share is not None:
        overrides["maker_fill_probability"] = args.maker_share
    if args.slippage_bps is not None:
        overrides["slippage_bps"] = args.slippage_bps
    if args.spread_bps is not None:
        overrides["spread_bps"] = args.spread_bps
    cost_model = replace(base_costs, **overrides)
    live = args.strategy == "trend_vol_target"
    threshold = args.rebalance_threshold if args.rebalance_threshold is not None else (
        float(policy.get("rebalance_threshold", 0.0)) if live else 0.0)
    hours = tuple(args.rebalance_hours) if args.rebalance_hours is not None else (
        tuple(policy.get("rebalance_hours_utc", [])) if live else ())

    label = f"{split.name} {split.start:%Y-%m-%d}→{split.end:%Y-%m-%d}" if split.name == "custom" else f"{split.name} split"
    name = args.out_dir or REPO_ROOT / "research" / "experiments" / "reports" / (
        f"{args.strategy}_{split.name}_{args.costs}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    cfg = ReportConfig(
        strategy_name=args.strategy, strategy_fn=STRATEGIES[args.strategy], strategy_params=params,
        cost_model=cost_model, execution_price=args.execution_price, execution_lag=args.latency_bars,
        rebalance_threshold=threshold, rebalance_hours_utc=hours, initial_capital=args.capital,
        benchmark_assets=benchmarks or [pairs[0]], random_seeds=list(range(args.seeds)),
        period_label=f"{label}, {args.costs} costs" + (" (customized)" if overrides else ""),
        data_source=f"{args.data} (resampled to {bar})",
        hypothesis=strategy_cfg.get("strategy_notes", {}).get("hypothesis", "") if live and args.params is None else "",
    )
    summary = run_report(cfg, panel.close, fields["open"], fields["volume"], eval_index, Path(name))
    n = summary["net"]
    print(f"{args.strategy} on {label} ({summary['period']['start'][:10]} -> {summary['period']['end'][:10]}): "
          f"net {n['total_return']:+.2%}, gross {summary['gross']['total_return']:+.2%}, Sharpe {n['sharpe']:.2f}, "
          f"max DD {n['max_drawdown']:.2%}, costs ${summary['costs']['total_costs']:,.0f}")
    rep = summary["reproducibility"]
    run_id = RunRegistry().register(
        "backtest", [str(Path(__file__).relative_to(REPO_ROOT))] + sys.argv[1:],
        {**rep["config"], "split": split.name, "start": str(split.start), "end": str(split.end), "cost_scenario": args.costs},
        rep["data_hash_close"], name,
        {"net_return": n["total_return"], "gross_return": summary["gross"]["total_return"], "sharpe": n["sharpe"],
         "max_drawdown": n["max_drawdown"], "composite": n["composite"], "total_costs": summary["costs"]["total_costs"]})
    print(f"report: {Path(name) / 'report.html'}")
    print(f"run id: {run_id}  (python scripts/runs.py show {run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
