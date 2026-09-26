"""Run every baseline strategy on historical data under each cost scenario
and rank them by the competition's composite score.

    python scripts/run_baselines.py                           # all pairs in data/binance/5m, hourly bars
    python scripts/run_baselines.py --resample 4h --start 2025-06-01
    python scripts/run_baselines.py --pairs BTC/USD ETH/USD SOL/USD --out research/experiments/baselines.csv

Strategy parameters come from config/strategy.yaml and are in *bars* of the
chosen (resampled) interval.

`trade_limit_breach` is the share of bars in which a strategy wanted more
trades than the competition's 1-trade-per-minute limit allows within one bar
(e.g. 60 trades per 1h bar). The engine doesn't enforce that limit, so a
strategy with a high breach rate is not executable as backtested.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest import metrics  # noqa: E402
from backtest.costs import SCENARIOS  # noqa: E402
from backtest.engine import BacktestEngine  # noqa: E402
from src.data.historical import ParquetDataSource, load_panel  # noqa: E402
from src.strategy import signals  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SECONDS_PER_YEAR = 365.25 * 24 * 3600


def build_strategies(params: dict, benchmark: str) -> dict:
    mom = params.get("momentum", {})
    lookbacks = mom.get("lookback_periods", [24, 96, 288])
    mid_lookback = lookbacks[len(lookbacks) // 2]
    fast, slow = mom.get("ema_fast", 20), mom.get("ema_slow", 60)
    top_k = params.get("cross_sectional", {}).get("top_k", 3)
    mr = params.get("mean_reversion", {})
    vol_lookback = params.get("volatility", {}).get("lookback_periods", 96)
    return {
        f"buy_and_hold[{benchmark}]": lambda c: signals.buy_and_hold(c, benchmark),
        "equal_weight": signals.equal_weight,
        f"trend_following[{fast}/{slow}]": lambda c: signals.trend_following(c, fast, slow),
        f"single_asset_momentum[{benchmark},{mid_lookback}]": lambda c: signals.single_asset_momentum(c, benchmark, mid_lookback),
        f"cross_sectional_momentum[{mid_lookback},top{top_k}]": lambda c: signals.cross_sectional_momentum(c, mid_lookback, top_k),
        f"mean_reversion[{mr.get('lookback_periods', 12)},z{mr.get('z_entry', 1.5)}]": lambda c: signals.mean_reversion(
            c, mr.get("lookback_periods", 12), mr.get("z_entry", 1.5)
        ),
        f"vol_filtered_momentum[{mid_lookback}]": lambda c: signals.volatility_filtered_momentum(c, mid_lookback, vol_lookback),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    parser.add_argument("--pairs", nargs="*", help="Default: every pair with a data file.")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default="2100-01-01")
    parser.add_argument("--resample", default="1h", help="Bar size to trade on, e.g. 5min, 1h, 4h. 'none' keeps source bars.")
    parser.add_argument("--benchmark", default="BTC/USD")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--strategy-config", type=Path, default=REPO_ROOT / "config" / "strategy.yaml")
    parser.add_argument("--out", type=Path, help="Optional CSV path for the full results table.")
    args = parser.parse_args()

    source = ParquetDataSource(args.data)
    pairs = args.pairs or source.available_pairs()
    if not pairs:
        print(f"No data files in {args.data}. Run scripts/download_binance_history.py first.")
        return 1
    if args.benchmark not in pairs:
        pairs = [args.benchmark, *pairs]

    resample = None if args.resample.lower() == "none" else args.resample
    close = load_panel(source, pairs, args.start, args.end, resample=resample)
    close = close.dropna(how="all")
    bar_seconds = close.index.to_series().diff().median().total_seconds()
    periods_per_year = SECONDS_PER_YEAR / bar_seconds
    max_trades_per_bar = max(int(bar_seconds // 60), 1)

    print(
        f"{len(close.columns)} pairs, {len(close)} bars of {pd.Timedelta(seconds=bar_seconds)} "
        f"({close.index[0]:%Y-%m-%d} -> {close.index[-1]:%Y-%m-%d}), periods/year={periods_per_year:.0f}"
    )

    params = yaml.safe_load(args.strategy_config.read_text()) or {}
    rows = []
    for name, fn in build_strategies(params, args.benchmark).items():
        weights = fn(close)
        for scenario, cost_model in SCENARIOS.items():
            result = BacktestEngine(cost_model, initial_capital=args.capital).run(close, weights)
            summary = metrics.summarize(result.portfolio_value, result.weights_history, result.total_fees, periods_per_year)
            trades_per_bar = (result.trade_notional_history != 0).sum(axis=1)
            rows.append(
                {
                    "strategy": name,
                    "costs": scenario,
                    **summary,
                    "avg_trades_per_bar": float(trades_per_bar.mean()),
                    "trade_limit_breach": float((trades_per_bar > max_trades_per_bar).mean()),
                }
            )

    table = pd.DataFrame(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"Full results written to {args.out}")

    show = ["composite_score", "sortino_ratio", "sharpe_ratio", "calmar_ratio", "cumulative_return",
            "max_drawdown", "avg_turnover", "total_fees", "avg_trades_per_bar", "trade_limit_breach"]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 20)
    for scenario in SCENARIOS:
        part = table[table.costs == scenario].set_index("strategy")[show].sort_values("composite_score", ascending=False)
        print(f"\n=== {scenario} costs ===")
        print(part.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
