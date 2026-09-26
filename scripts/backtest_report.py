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
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.costs import SCENARIOS  # noqa: E402
from backtest.registry import RunRegistry  # noqa: E402
from backtest.report import ReportConfig, run_report  # noqa: E402
from backtest.splits import load_split_fields, load_split_panel, load_splits  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402
from src.strategy.signals import STRATEGIES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    strategy_cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    policy = strategy_cfg.get("execution_policy", {})
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--use-holdout", action="store_true", help="Required for --split test.")
    parser.add_argument("--strategy", default="trend_vol_target", choices=sorted(STRATEGIES))
    parser.add_argument("--params", type=json.loads, default=None,
                        help="JSON strategy params (default: the live config in config/strategy.yaml).")
    parser.add_argument("--costs", default="base", choices=sorted(SCENARIOS))
    parser.add_argument("--execution-price", default=policy.get("execution_price", "open"), choices=["open", "close"])
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
    pairs = sorted(set(params.get("assets", ())) | set(args.benchmarks) | ({params["symbol"]} if "symbol" in params else set()))

    source = ParquetDataSource(args.data)
    split = load_splits()[args.split]
    panel = load_split_panel(source, pairs, split, resample=policy.get("bar", "1h"), allow_holdout=args.use_holdout)
    fields = load_split_fields(source, panel, ("open", "volume"), resample=policy.get("bar", "1h"))
    eval_index = panel.eval_index
    if split.name == "train":
        # The train split starts at the beginning of the data: skip the
        # indicator warm-up (longest lookback) so the strategy isn't scored
        # while it's still forced into cash.
        warm = max([v for k, v in params.items() if k.endswith(("span", "lookback")) and isinstance(v, int)] or [0])
        eval_index = eval_index[warm:]

    name = args.out_dir or REPO_ROOT / "research" / "experiments" / "reports" / (
        f"{args.strategy}_{split.name}_{args.costs}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    cfg = ReportConfig(
        strategy_name=args.strategy, strategy_fn=STRATEGIES[args.strategy], strategy_params=params,
        cost_model=SCENARIOS[args.costs], execution_price=args.execution_price,
        rebalance_threshold=float(policy.get("rebalance_threshold", 0.0)) if args.strategy == "trend_vol_target" else 0.0,
        rebalance_hours_utc=tuple(policy.get("rebalance_hours_utc", [])) if args.strategy == "trend_vol_target" else (),
        benchmark_assets=args.benchmarks, random_seeds=list(range(args.seeds)),
        period_label=f"{split.name} split, {args.costs} costs",
        hypothesis=strategy_cfg.get("strategy_notes", {}).get("hypothesis", "") if args.strategy == "trend_vol_target" else "",
    )
    summary = run_report(cfg, panel.close, fields["open"], fields["volume"], eval_index, Path(name))
    n = summary["net"]
    print(f"{args.strategy} on {split.name} ({summary['period']['start'][:10]} -> {summary['period']['end'][:10]}): "
          f"net {n['total_return']:+.2%}, gross {summary['gross']['total_return']:+.2%}, Sharpe {n['sharpe']:.2f}, "
          f"max DD {n['max_drawdown']:.2%}, costs ${summary['costs']['total_costs']:,.0f}")
    rep = summary["reproducibility"]
    run_id = RunRegistry().register(
        "backtest", [str(Path(__file__).relative_to(REPO_ROOT))] + sys.argv[1:],
        {**rep["config"], "split": split.name, "cost_scenario": args.costs}, rep["data_hash_close"], name,
        {"net_return": n["total_return"], "gross_return": summary["gross"]["total_return"], "sharpe": n["sharpe"],
         "max_drawdown": n["max_drawdown"], "composite": n["composite"], "total_costs": summary["costs"]["total_costs"]})
    print(f"report: {Path(name) / 'report.html'}")
    print(f"run id: {run_id}  (python scripts/runs.py show {run_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
