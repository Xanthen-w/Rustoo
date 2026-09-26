"""Walk-forward optimisation of the strategy grids in config/research.yaml.

    python scripts/walk_forward.py                                    # every grid, on the train split
    python scripts/walk_forward.py --strategies trend_following cross_sectional_momentum
    python scripts/walk_forward.py --split validation                 # confirm on validation

For each strategy: per fold, choose the grid point with the best in-sample
objective, then trade it on the next out-of-sample window. The stitched
out-of-sample record is the honest estimate of that strategy family. Also
prints how much in-sample scores overstate out-of-sample ones, and which
parameter sets get picked — unstable picks mean the edge is noise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.costs import SCENARIOS  # noqa: E402
from backtest.splits import load_research_config, load_split_panel, load_splits  # noqa: E402
from backtest.walk_forward import expand_grid, make_folds, run_walk_forward  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def main() -> int:
    config = load_research_config()
    wf = config["walk_forward"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    parser.add_argument("--pairs", nargs="*", help="Default: every pair with a data file.")
    parser.add_argument("--strategies", nargs="*", help="Default: every grid in config/research.yaml.")
    parser.add_argument("--split", default=wf.get("split", "train"), choices=["train", "validation", "test"])
    parser.add_argument("--use-holdout", action="store_true", help="Required for --split test.")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "research" / "experiments" / "walk_forward")
    args = parser.parse_args()

    source = ParquetDataSource(args.data)
    pairs = args.pairs or source.available_pairs()
    split = load_splits(config)[args.split]
    panel = load_split_panel(source, pairs, split, resample=wf.get("resample", "1h"), allow_holdout=args.use_holdout)
    close = panel.close
    eval_index = panel.eval_index
    bar_seconds = eval_index.to_series().diff().median().total_seconds()
    periods_per_year = SECONDS_PER_YEAR / bar_seconds

    folds = make_folds(
        eval_index[0], eval_index[-1] + pd.Timedelta(seconds=bar_seconds),
        pd.Timedelta(days=wf["in_sample_days"]), pd.Timedelta(days=wf["out_of_sample_days"]),
        pd.Timedelta(days=wf["step_days"]), anchored=wf.get("anchored", False),
    )
    cost_model = SCENARIOS[wf.get("cost_scenario", "base")]
    objective = wf.get("objective", "composite_score")
    print(f"split={split.name}: {len(close.columns)} pairs, {len(eval_index)} bars, {len(folds)} folds "
          f"({wf['in_sample_days']}d in-sample / {wf['out_of_sample_days']}d out-of-sample, "
          f"{'anchored' if wf.get('anchored') else 'rolling'}), objective={objective}, costs={wf.get('cost_scenario', 'base')}")
    if panel.dropped_pairs:
        print("Excluded: " + ", ".join(f"{p} ({why})" for p, why in sorted(panel.dropped_pairs.items())))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 90)

    overview = []
    for strategy in args.strategies or list(config["grids"]):
        candidates = expand_grid(strategy, config["grids"][strategy])
        result = run_walk_forward(close, candidates, folds, cost_model, periods_per_year, objective)
        scores = result.candidate_scores
        print(f"\n##### {strategy}: {len(candidates)} candidates")
        print(result.folds.drop(columns=["in_sample"]).round(3).to_string(index=False))
        picks = result.folds["chosen"].value_counts()
        print(f"most-picked: {picks.index[0]} ({picks.iloc[0]}/{len(folds)} folds)")
        s = result.oos_summary
        print(f"stitched out-of-sample: return={s['cumulative_return']:.1%} maxDD={s['max_drawdown']:.1%} "
              f"sharpe={s['sharpe_ratio']:.2f} sortino={s['sortino_ratio']:.2f} composite={s['composite_score']:.3f}")

        result.folds.to_csv(args.out_dir / f"{split.name}_{strategy}_folds.csv", index=False)
        scores.to_csv(args.out_dir / f"{split.name}_{strategy}_candidates.csv", index=False)
        overview.append({
            "strategy": strategy,
            "candidates": len(candidates),
            "mean_is_score_of_picks": result.folds["is_score"].mean(),
            "mean_oos_score_of_picks": result.folds["oos_score"].mean(),
            "distinct_picks": result.folds["chosen"].nunique(),
            "oos_return": s["cumulative_return"],
            "oos_max_drawdown": s["max_drawdown"],
            "oos_composite": s["composite_score"],
        })

    print("\n##### overview (out-of-sample = stitched walk-forward record)")
    print(pd.DataFrame(overview).set_index("strategy").sort_values("oos_composite", ascending=False).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
