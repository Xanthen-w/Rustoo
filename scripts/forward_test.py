"""Frozen forward testing: evaluate the live strategy only on data that did
not exist when its configuration was frozen.

    python scripts/forward_test.py freeze --name live-2026-10     # snapshot config; commit the file it writes
    python scripts/forward_test.py evaluate --name live-2026-10   # later, after downloading newer data

`freeze` writes research/forward/<name>.json: strategy parameters, execution
policy, cost assumptions, the git commit, the freeze time, and the last bar
of data available at that moment, plus a hash of all of it. It refuses to
overwrite an existing freeze or to freeze uncommitted code.

`evaluate` checks the hash, then backtests the *frozen* configuration
(current config files are ignored; there are no parameter overrides) on
bars strictly after the freeze's data cut-off, with earlier bars used only
for indicator warm-up. The report is labelled FORWARD. The freeze file's
first git commit is shown as evidence of when it was fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.costs import SCENARIOS, CostModel  # noqa: E402
from backtest.registry import RunRegistry, git_state  # noqa: E402
from backtest.report import ReportConfig, run_report  # noqa: E402
from src.data.historical import ParquetDataSource, load_panel  # noqa: E402
from src.strategy.signals import STRATEGIES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
FREEZE_DIR = REPO_ROOT / "research" / "forward"


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _hash(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "freeze_hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def _data_end(source: ParquetDataSource, pairs: list[str]) -> pd.Timestamp:
    return min(source.load(p, "2000-01-01", "2100-01-01").index.max() for p in pairs)


def freeze(args) -> int:
    path = FREEZE_DIR / f"{args.name}.json"
    if path.exists():
        print(f"{path} already exists; a freeze is immutable. Choose a new --name.")
        return 1
    commit, dirty = git_state()
    if dirty and not args.allow_dirty:
        print("The working tree has uncommitted changes to tracked files; commit first so the freeze points at exact code.")
        return 1
    cfg = yaml.safe_load((REPO_ROOT / "config" / "strategy.yaml").read_text())
    params = dict(cfg["trend_vol_target"])
    source = ParquetDataSource(args.data)
    data_end = _data_end(source, list(params["assets"]))
    record = {
        "name": args.name, "frozen_at": datetime.now(timezone.utc).isoformat(), "git_commit": commit, "git_dirty": dirty,
        "strategy": "trend_vol_target", "params": params, "execution_policy": cfg.get("execution_policy", {}),
        "cost_scenario": args.costs, "costs": asdict(SCENARIOS[args.costs]), "benchmarks": ["BTC/USD", "ETH/USD"],
        "data_source": _rel(args.data), "data_end": str(data_end),
        "hypothesis": cfg.get("strategy_notes", {}).get("hypothesis", ""),
    }
    record["freeze_hash"] = _hash(record)
    FREEZE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str) + "\n")
    print(f"frozen: {_rel(path)} (data cut-off {data_end}, commit {commit[:8]})")
    print("Commit this file now; its commit time is the evidence of when the configuration was fixed.")
    return 0


def evaluate(args) -> int:
    path = FREEZE_DIR / f"{args.name}.json"
    record = json.loads(path.read_text())
    if _hash(record) != record["freeze_hash"]:
        print(f"{path} has been modified since it was frozen (hash mismatch); refusing to evaluate.")
        return 1
    log = subprocess.run(["git", "log", "--format=%h %cI", "--", _rel(path)], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip().splitlines()
    evidence = f"first committed {log[-1]}" if log else "NOT COMMITTED: the freeze time isn't verifiable from git history"
    params = dict(record["params"])
    params["assets"] = tuple(params["assets"])
    policy = record["execution_policy"]
    pairs = sorted(set(params["assets"]) | set(record["benchmarks"]))
    source = ParquetDataSource(args.data)
    cutoff = pd.Timestamp(record["data_end"])
    bar = policy.get("bar", "1h")
    close = load_panel(source, pairs, "2000-01-01", "2100-01-01", resample=bar).dropna(how="all")
    open_ = load_panel(source, pairs, "2000-01-01", "2100-01-01", field="open", resample=bar).reindex_like(close)
    volume = load_panel(source, pairs, "2000-01-01", "2100-01-01", field="volume", resample=bar).reindex_like(close)
    forward = close.index[close.index > cutoff]
    days = (forward[-1] - forward[0]).total_seconds() / 86400 if len(forward) > 1 else 0
    if days < args.min_days:
        print(f"only {days:.1f} days of data after the freeze cut-off {cutoff}; need at least {args.min_days}. "
              "Download newer data first (scripts/download_binance_history.py).")
        return 1
    out = args.out_dir or REPO_ROOT / "research" / "experiments" / "forward" / f"{args.name}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    cfg = ReportConfig(
        strategy_name=record["strategy"], strategy_fn=STRATEGIES[record["strategy"]], strategy_params=params,
        cost_model=CostModel(**record["costs"]), execution_price=policy.get("execution_price", "open"),
        rebalance_threshold=float(policy.get("rebalance_threshold", 0.0)),
        rebalance_hours_utc=tuple(policy.get("rebalance_hours_utc", [])), benchmark_assets=record["benchmarks"],
        period_label=f"FORWARD test of frozen config '{record['name']}' (frozen {record['frozen_at'][:16]} UTC, {evidence})",
        hypothesis=record.get("hypothesis", ""),
    )
    summary = run_report(cfg, close, open_, volume, forward, Path(out))
    n = summary["net"]
    run_id = RunRegistry().register(
        "forward", [str(Path(__file__).relative_to(REPO_ROOT))] + sys.argv[1:],
        {"freeze": record["name"], "freeze_hash": record["freeze_hash"], "data_end": record["data_end"]},
        summary["reproducibility"]["data_hash_close"], out,
        {"net_return": n["total_return"], "sharpe": n["sharpe"], "max_drawdown": n["max_drawdown"], "composite": n["composite"]})
    print(f"FORWARD {forward[0]} -> {forward[-1]} ({days:.1f} days): net {n['total_return']:+.2%}, Sharpe {n['sharpe']:.2f}, "
          f"max DD {n['max_drawdown']:.2%}; freeze {evidence}")
    print(f"report: {Path(out) / 'report.html'}\nrun id: {run_id}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("freeze", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--name", required=True)
        p.add_argument("--data", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    sub.choices["freeze"].add_argument("--costs", default="base", choices=sorted(SCENARIOS))
    sub.choices["freeze"].add_argument("--allow-dirty", action="store_true")
    sub.choices["evaluate"].add_argument("--min-days", type=float, default=3.0)
    sub.choices["evaluate"].add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    return freeze(args) if args.cmd == "freeze" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
