"""Distribution of competition-length (14-day) outcomes for each candidate.

    python scripts/window_analysis.py                     # train split, grids in config/research.yaml:window_analysis
    python scripts/window_analysis.py --split validation --candidates "trend_vol_target(...)"  # confirm finalists

Every window starts from cash, like the competition. Reports per candidate:
median / 10th / 90th percentile return, share of windows with a positive
return (screen 2 ranks by return), median composite score (screen 3), and
days with at least one trade (the rules require >= 8 active days).
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.costs import SCENARIOS  # noqa: E402
from backtest.splits import load_research_config, load_split_panel, load_splits  # noqa: E402
from backtest.walk_forward import _normalise_weights, expand_grid  # noqa: E402
from backtest.windows import evaluate_windows, rolling_windows, summarize_benchmark, summarize_windows  # noqa: E402
from src.data.historical import ParquetDataSource  # noqa: E402
from src.strategy.signals import STRATEGIES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def main() -> int:
    config = load_research_config()
    wa = config["window_analysis"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--use-holdout", action="store_true", help="Required for --split test.")
    parser.add_argument("--candidates", nargs="*", help="Only evaluate candidates whose label contains one of these strings.")
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "research" / "experiments" / "windows")
    args = parser.parse_args()

    source = ParquetDataSource(args.data)
    split = load_splits(config)[args.split]
    panel = load_split_panel(source, source.available_pairs(), split, resample=wa.get("resample", "1h"),
                             allow_holdout=args.use_holdout)
    close, eval_index = panel.close, panel.eval_index
    bar_seconds = eval_index.to_series().diff().median().total_seconds()
    periods_per_year = SECONDS_PER_YEAR / bar_seconds

    # Warm-up only needs skipping when the split starts at the beginning of the data.
    skip = pd.Timedelta(days=wa.get("warmup_days", 0)) if close.index[0] >= split.start else pd.Timedelta(0)
    windows = rolling_windows(eval_index, pd.Timedelta(days=wa.get("window_days", 14)),
                              pd.Timedelta(days=wa.get("step_days", 1)), skip)
    cost_model = SCENARIOS[wa.get("cost_scenario", "base")]
    min_days = wa.get("min_trading_days", 8)
    print(f"split={split.name}: {len(windows)} windows of {wa.get('window_days', 14)}d "
          f"({windows[0].start:%Y-%m-%d} -> {windows[-1].end:%Y-%m-%d}), costs={wa.get('cost_scenario', 'base')}")

    candidates = [c for strategy, spec in wa["grids"].items() for c in expand_grid(strategy, spec)]
    if args.candidates:
        candidates = [c for c in candidates if any(s in c.label for s in args.candidates)]
    print(f"{len(candidates)} candidates")

    benchmark = None
    rows, weight_cache = [], {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for cand in candidates:
        key = (cand.strategy, cand.params)
        if key not in weight_cache:
            weight_cache[key] = _normalise_weights(STRATEGIES[cand.strategy](close, **cand.strategy_kwargs()), close)
        per_window = evaluate_windows(close, weight_cache[key], windows, cost_model, periods_per_year, cand.engine_kwargs())
        if benchmark is None and cand.strategy == "buy_and_hold":
            benchmark = per_window
        rows.append({"candidate": cand.label, **summarize_windows(per_window, min_days),
                     **(summarize_benchmark(per_window, benchmark) if benchmark is not None else {})})
        label_id = hashlib.sha1(cand.label.encode()).hexdigest()[:10]
        per_window.assign(candidate=cand.label).to_csv(args.out_dir / f"{split.name}_{label_id}.csv", index=False)

    table = pd.DataFrame(rows).set_index("candidate")
    table.to_csv(args.out_dir / f"{split.name}_summary.csv")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_colwidth", 110)
    table.index = table.index.str.replace("vol_lookback=720, ", "").str.replace("rebalance_threshold=", "thr=")
    for key, title in [("median_return", "by median 14-day return (screen 2)"),
                       ("median_composite", "by median 14-day composite (screen 3)")]:
        print(f"\n=== top {args.top} {title}")
        print(table.sort_values(key, ascending=False).head(args.top).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
