"""Experiment registry: every report run gets a run ID and a record of
exactly how it was produced — the command line, configuration, data hash,
git commit (and whether the working tree was dirty), library versions, the
output directory and its headline metrics. `scripts/runs.py reproduce` uses
the record to re-run it and check the numbers match.

Stored in research/experiments/registry.sqlite3 (gitignored, like the
outputs it points to).
"""
from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = REPO_ROOT / "research" / "experiments" / "registry.sqlite3"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, kind TEXT, created_at TEXT, command TEXT, config TEXT, data_hash TEXT,
    git_commit TEXT, git_dirty INTEGER, versions TEXT, output_dir TEXT, metrics TEXT, notes TEXT
);
"""


def git_state() -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
        return commit or "unknown", dirty
    except Exception:
        return "unknown", True


def config_hash(config: dict, data_hash: str = "") -> str:
    blob = json.dumps(config, sort_keys=True, default=str) + data_hash
    return hashlib.sha256(blob.encode()).hexdigest()


def flatten_numbers(obj, prefix: str = "") -> dict[str, float]:
    """Numeric leaves of a nested summary, keyed by path (for comparisons)."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("reproducibility", "generated_at"):
                continue
            out.update(flatten_numbers(v, f"{prefix}{k}."))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        out[prefix.rstrip(".")] = float(obj)
    return out


def compare_summaries(a: dict, b: dict, rtol: float = 1e-9, atol: float = 1e-9) -> list[tuple[str, float, float]]:
    """Paths whose numbers differ between two summaries."""
    fa, fb = flatten_numbers(a), flatten_numbers(b)
    diffs = []
    for key in sorted(set(fa) | set(fb)):
        x, y = fa.get(key, np.nan), fb.get(key, np.nan)
        if np.isnan(x) and np.isnan(y):
            continue
        if not np.isclose(x, y, rtol=rtol, atol=atol, equal_nan=True):
            diffs.append((key, x, y))
    return diffs


class RunRegistry:
    def __init__(self, path: Path | str = DEFAULT_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.executescript(SCHEMA)

    def register(self, kind: str, command: list[str], config: dict, data_hash: str, output_dir: Path | str,
                 metrics: dict, notes: str = "") -> str:
        now = datetime.now(timezone.utc)
        commit, dirty = git_state()
        run_id = f"{kind}-{now:%Y%m%dT%H%M%SZ}-{config_hash(config, data_hash)[:8]}"
        versions = {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}
        self._conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, kind, now.isoformat(), json.dumps(command), json.dumps(config, default=str), data_hash,
             commit, int(dirty), json.dumps(versions), str(output_dir), json.dumps(metrics, default=str), notes),
        )
        self._conn.commit()
        return run_id

    def _rows(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self._conn.execute(sql, params)
        cols = [c[0] for c in cur.description]
        out = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            for k in ("command", "config", "versions", "metrics"):
                d[k] = json.loads(d[k]) if d[k] else None
            d["git_dirty"] = bool(d["git_dirty"])
            out.append(d)
        return out

    def get(self, run_id: str) -> dict:
        rows = self._rows("SELECT * FROM runs WHERE run_id = ? OR run_id LIKE ?", (run_id, f"%{run_id}%"))
        if not rows:
            raise KeyError(f"no run matching {run_id!r}")
        if len(rows) > 1:
            raise KeyError(f"{run_id!r} matches {len(rows)} runs; be more specific")
        return rows[0]

    def list(self, limit: int = 50, kind: str | None = None) -> list[dict]:
        if kind:
            return self._rows("SELECT * FROM runs WHERE kind = ? ORDER BY created_at DESC LIMIT ?", (kind, limit))
        return self._rows("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,))

    def delete(self, run_id: str) -> None:
        run = self.get(run_id)
        self._conn.execute("DELETE FROM runs WHERE run_id = ?", (run["run_id"],))
        self._conn.commit()
