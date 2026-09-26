"""Robustness report for the live strategy (or any registered strategy).

    python scripts/robustness_report.py                        # evaluate on validation; landscapes on train + validation
    python scripts/robustness_report.py --split test --use-holdout
    python scripts/robustness_report.py --mc-sims 5000 --mc-seeds 42 123 2026 9999 --cost-sigma 0.25

Writes research/experiments/robustness/<run>/robustness.html (self-contained)
plus CSVs: cost_sensitivity, stress, monte_carlo_*, regimes, landscape_*,
random_entry_*, and summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import robustness as rb  # noqa: E402
from backtest.costs import SCENARIOS  # noqa: E402
from backtest.robustness_report import run_robustness  # noqa: E402
from backtest.splits import load_split_fields, load_split_panel, load_splits  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402
from src.strategy.signals import STRATEGIES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

LANDSCAPE_GRIDS = [
    ("trend_span", [240, 480, 720, 960, 1440, 2400], "target_vol", [0.3, 0.4, 0.5, 0.7, 1.0]),
    ("band", [0.0, 0.01, 0.03, 0.05, 0.08], "min_exposure", [0.0, 0.1, 0.15, 0.25, 0.4]),
]


def make_spec(source, split, pairs, params, cost_model, engine_kwargs, bar, allow_holdout, warm_bars):
    panel = load_split_panel(source, pairs, split, resample=bar, allow_holdout=allow_holdout)
    f = load_split_fields(source, panel, ("open", "volume"), resample=bar)
    eval_index = panel.eval_index
    if panel.close.index[0] >= split.start:  # split starts at the data start: skip indicator warm-up
        eval_index = eval_index[warm_bars:]
    return rb.RunSpec(panel.close, f["open"], f["volume"], eval_index, STRATEGIES["trend_vol_target"], params,
                      cost_model, engine_kwargs)


def main() -> int:
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    policy = cfg.get("execution_policy", {})
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--use-holdout", action="store_true", help="Required to touch the test split.")
    parser.add_argument("--params", type=json.loads, default=None, help="JSON override of the live strategy params.")
    parser.add_argument("--costs", default="base", choices=sorted(SCENARIOS))
    parser.add_argument("--mc-sims", type=int, default=2000)
    parser.add_argument("--mc-seeds", type=int, nargs="*", default=[42, 123, 2026, 9999])
    parser.add_argument("--cost-sigma", type=float, default=0.25, help="Monte Carlo per-path cost multiplier (lognormal sigma).")
    parser.add_argument("--random-seeds", type=int, default=30)
    parser.add_argument("--no-landscape", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    params = dict(cfg["trend_vol_target"]) if args.params is None else args.params
    params["assets"] = tuple(params.get("assets", ("BTC/USD", "ETH/USD")))
    pairs = sorted(set(params["assets"]) | {"BTC/USD"})
    engine_kwargs = {"execution_price": policy.get("execution_price", "open"), "execution_lag": 1,
                     "rebalance_threshold": float(policy.get("rebalance_threshold", 0.0)),
                     "rebalance_hours_utc": tuple(policy.get("rebalance_hours_utc", []))}
    bar = policy.get("bar", "1h")
    source = ParquetDataSource(args.data)
    splits = load_splits()
    warm = max(max(g[1]) for g in LANDSCAPE_GRIDS if g[0].endswith("span"))
    make = lambda name: make_spec(source, splits[name], pairs, params, SCENARIOS[args.costs], engine_kwargs, bar,
                                  name == "test" and args.use_holdout, warm)

    spec = make(args.split)
    landscapes = None if args.no_landscape else {
        name: (make(name), LANDSCAPE_GRIDS) for name in ("train", "validation") if name == "train" or args.split != "train"}
    random_splits = ["train", "validation"] + (["test"] if args.use_holdout else [])
    out = args.out_dir or REPO_ROOT / "research" / "experiments" / "robustness" / (
        f"{args.split}_{args.costs}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
    summary = run_robustness(
        spec, Path(out), title="trend_vol_target robustness", period_label=f"{args.split} split, {args.costs} costs",
        mc=rb.MonteCarloConfig(simulations=args.mc_sims, seeds=tuple(args.mc_seeds), cost_sigma=args.cost_sigma),
        landscapes=landscapes, random_entry={name: (spec if name == args.split else make(name)) for name in random_splits},
        random_seeds=list(range(args.random_seeds)),
    )
    m = summary["monte_carlo"]
    print(f"base net {summary['base']['net_return']:+.2%} | worst stress {summary['worst_stress']['scenario']} "
          f"{summary['worst_stress']['net_return']:+.2%} | breakeven (NaN = negative even at zero cost) {summary['breakeven_bps']}")
    print(f"MC 14d: P(loss) {m['p_loss']:.1%}, p5 {m['percentiles']['p5']['total_return']:+.2%}, "
          f"median {m['percentiles']['p50']['total_return']:+.2%}, P(beat BTC) {m.get('p_beats_benchmark', float('nan')):.1%}")
    print("random entry:", summary["random_entry"])
    print(f"report: {Path(out) / 'robustness.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
