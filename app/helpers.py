"""Pure helpers for the local research app (no Streamlit imports here, so
they're unit-testable). The app never re-implements research logic: every
action becomes a command line for one of the scripts in scripts/, run as a
subprocess. That way each run from the browser is recorded in the run
registry with a command that reproduces it from the terminal."""
from __future__ import annotations

import inspect
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


# -- data sources ------------------------------------------------------------------

def data_sources() -> dict[str, Path]:
    """Label -> directory for every folder of per-pair parquet files."""
    out = {}
    binance = DATA_ROOT / "binance" / "5m"
    if any(binance.glob("*.parquet")):
        out["Binance 5m (2y, all Roostoo pairs)"] = binance
    imported = DATA_ROOT / "imported"
    if imported.exists():
        for d in sorted(p for p in imported.iterdir() if p.is_dir() and any(p.glob("*.parquet"))):
            out[f"Imported: {d.name}"] = d
    return out


def pairs_in(source_dir: Path) -> list[str]:
    return sorted(p.stem.replace("-", "/", 1) for p in Path(source_dir).glob("*.parquet"))


def source_span(source_dir: Path, pair: str) -> tuple[pd.Timestamp, pd.Timestamp, int]:
    idx = pd.read_parquet(Path(source_dir) / f"{pair.replace('/', '-')}.parquet", columns=["close"]).index
    return idx.min(), idx.max(), len(idx)


# -- strategy parameters -----------------------------------------------------------------

def strategy_defaults(fn, live_config: dict | None = None) -> dict:
    """Keyword defaults of a strategy function (skipping the price panel),
    overlaid with the live config when given."""
    params = {name: p.default for name, p in list(inspect.signature(fn).parameters.items())[1:]
              if p.default is not inspect.Parameter.empty}
    for name, p in list(inspect.signature(fn).parameters.items())[1:]:
        if p.default is inspect.Parameter.empty:
            params[name] = None  # required (e.g. symbol)
    if live_config:
        params.update({k: v for k, v in live_config.items() if k in params})
    return params


# -- command builders ------------------------------------------------------------------------

def backtest_argv(o: dict) -> list[str]:
    """Options from the Backtest page -> scripts/backtest_report.py argv."""
    argv = ["scripts/backtest_report.py", "--data", str(o["data"]), "--strategy", o["strategy"],
            "--params", json.dumps(o["params"]), "--costs", o.get("costs", "base"),
            "--execution-price", o.get("execution_price", "open"), "--latency-bars", str(o.get("latency_bars", 1)),
            "--rebalance-threshold", str(o.get("rebalance_threshold", 0.0)),
            "--capital", str(o.get("capital", 100_000.0)), "--seeds", str(o.get("seeds", 20))]
    argv += ["--rebalance-hours", *map(str, o.get("rebalance_hours", []))]
    if o.get("benchmarks"):
        argv += ["--benchmarks", *o["benchmarks"]]
    split = o.get("split", "validation")
    argv += ["--split", split]
    if split == "custom":
        argv += ["--start", str(o["start"]), "--end", str(o["end"])]
    if o.get("use_holdout"):
        argv.append("--use-holdout")
    for key, flag in (("fee_bps", "--fee-bps"), ("maker_share", "--maker-share"), ("slippage_bps", "--slippage-bps"),
                      ("spread_bps", "--spread-bps")):
        if o.get(key) is not None:
            argv += [flag, str(o[key])]
    return argv


def robustness_argv(o: dict) -> list[str]:
    argv = ["scripts/robustness_report.py", "--data", str(o["data"]), "--split", o.get("split", "validation"),
            "--costs", o.get("costs", "base"), "--mc-sims", str(o.get("mc_sims", 2000)),
            "--mc-seeds", *map(str, o.get("mc_seeds", [42, 123, 2026, 9999])),
            "--cost-sigma", str(o.get("cost_sigma", 0.25)), "--random-seeds", str(o.get("random_seeds", 30))]
    if o.get("params") is not None:
        argv += ["--params", json.dumps(o["params"])]
    if not o.get("landscape", False):
        argv.append("--no-landscape")
    if o.get("use_holdout"):
        argv.append("--use-holdout")
    return argv


def import_argv(files: list[Path], o: dict) -> list[str]:
    argv = ["scripts/import_data.py", *map(str, files), "--source", o.get("source", "uploads"),
            "--labelled-by", o.get("labelled_by", "start"), "--mode", o.get("mode", "warn")]
    if o.get("tz"):
        argv += ["--tz", o["tz"]]
    if o.get("symbol"):
        argv += ["--symbol", o["symbol"]]
    if o.get("mapping"):
        argv += ["--map", *[f"{k}={v}" for k, v in o["mapping"].items()]]
    if o.get("reference"):
        argv += ["--reference", str(o["reference"])]
    if o.get("not_crypto"):
        argv.append("--not-crypto")
    return argv


def display_command(argv: list[str]) -> str:
    """Shell-pasteable form of an argv (for showing users what ran)."""
    def q(a: str) -> str:
        return a if re.fullmatch(r"[\w./:=+-]+", a) else "'" + a.replace("'", "'\\''") + "'"
    return ".venv/bin/python " + " ".join(q(a) for a in argv)


def stream(argv: list[str], on_line=None) -> tuple[int, str]:
    """Run a repo script with this interpreter; call on_line(line) as output
    arrives. Returns (exit code, full output)."""
    proc = subprocess.Popen([sys.executable, "-u", *argv], cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env={**__import__("os").environ, "PYTHONWARNINGS": "ignore"})
    lines = []
    for line in proc.stdout:
        lines.append(line)
        if on_line:
            on_line(line)
    return proc.wait(), "".join(lines)


def parse_outputs(output: str) -> dict:
    """Pull the report path and run id out of a script's output."""
    found = {}
    m = re.search(r"^report: (.+)$", output, re.M)
    if m:
        found["report"] = (REPO_ROOT / m.group(1).strip()).resolve() if not m.group(1).startswith("/") else Path(m.group(1).strip())
    m = re.search(r"^run id: (\S+)", output, re.M)
    if m:
        found["run_id"] = m.group(1)
    return found


# -- paper bot ----------------------------------------------------------------------------------

def bot_snapshot(db: Path = DATA_ROOT / "state" / "bot-paper.sqlite3") -> dict | None:
    """Read-only view of the bot's audit database."""
    if not db.exists():
        return None
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        q = lambda sql: pd.read_sql_query(sql, conn)  # noqa: E731
        equity = q("SELECT ts, equity, cash, positions FROM equity ORDER BY id")
        equity["ts"] = pd.to_datetime(equity["ts"], utc=True, format="ISO8601")
        return {
            "equity": equity,
            "decisions": q("SELECT ts, bar_close, equity, targets, current, scheduled, planned_orders, note FROM decisions ORDER BY id DESC LIMIT 48"),
            "orders": q("SELECT ts, mode, pair, side, quantity, est_price, fee, status, reason, error FROM orders ORDER BY id DESC"),
            "api": q("SELECT path, COUNT(*) AS calls, SUM(success) AS ok, COUNT(*) - SUM(success) AS failed, "
                     "ROUND(AVG(elapsed_ms)) AS avg_ms FROM api_calls GROUP BY path"),
            "days": q("SELECT DISTINCT substr(ts, 1, 10) AS day FROM orders WHERE status IN ('FILLED','PARTIALLY_FILLED','SIMULATED') ORDER BY 1"),
            "wallet": json.loads((conn.execute("SELECT value FROM kv WHERE key='paper_wallet'").fetchone() or ["null"])[0]),
        }
    finally:
        conn.close()
