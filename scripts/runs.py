"""Experiment history: list, inspect, compare and reproduce registered runs.

    python scripts/runs.py list
    python scripts/runs.py show <run-id>
    python scripts/runs.py compare <run-id-a> <run-id-b>
    python scripts/runs.py reproduce <run-id>      # re-run it and check every number matches
    python scripts/runs.py delete <run-id>

Run IDs may be abbreviated to any unique substring.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backtest.registry import RunRegistry, compare_summaries, git_state  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def _summary(output_dir: str) -> dict:
    path = Path(output_dir) / "summary.json"
    if not path.exists():
        raise SystemExit(f"summary not found at {path} (outputs deleted?)")
    return json.loads(path.read_text())


def _with_out_dir(command: list[str], out_dir: str) -> list[str]:
    cmd = list(command)
    if "--out-dir" in cmd:
        cmd[cmd.index("--out-dir") + 1] = out_dir
    else:
        cmd += ["--out-dir", out_dir]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_list = sub.add_parser("list")
    p_list.add_argument("--kind")
    p_list.add_argument("--limit", type=int, default=30)
    sub.add_parser("show").add_argument("run_id")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("a")
    p_cmp.add_argument("b")
    sub.add_parser("reproduce").add_argument("run_id")
    sub.add_parser("delete").add_argument("run_id")
    args = parser.parse_args()
    reg = RunRegistry()

    if args.cmd == "list":
        for r in reg.list(args.limit, args.kind):
            m = r["metrics"] or {}
            head = ", ".join(f"{k}={v:+.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(m.items())[:3])
            print(f"{r['run_id']:48s} {r['git_commit'][:8]}{'*' if r['git_dirty'] else ' '} {head}")
        print("(* = run from uncommitted code)")
    elif args.cmd == "show":
        r = reg.get(args.run_id)
        print(json.dumps(r, indent=2, default=str))
    elif args.cmd == "compare":
        a, b = reg.get(args.a), reg.get(args.b)
        cfg_a, cfg_b = a["config"], b["config"]
        print("config differences:")
        for k in sorted(set(cfg_a) | set(cfg_b)):
            if cfg_a.get(k) != cfg_b.get(k):
                print(f"  {k}: {cfg_a.get(k)!r}  ->  {cfg_b.get(k)!r}")
        print(f"data hash: {a['data_hash']} vs {b['data_hash']}; commits {a['git_commit'][:8]} vs {b['git_commit'][:8]}")
        diffs = compare_summaries(_summary(a["output_dir"]), _summary(b["output_dir"]))
        print(f"{len(diffs)} differing numbers" + (":" if diffs else ""))
        for key, x, y in diffs[:40]:
            print(f"  {key}: {x:.6g} -> {y:.6g}")
    elif args.cmd == "reproduce":
        r = reg.get(args.run_id)
        commit, dirty = git_state()
        if commit != r["git_commit"] or r["git_dirty"] or dirty:
            print(f"note: run was made at {r['git_commit'][:8]}{' (dirty)' if r['git_dirty'] else ''}, "
                  f"code now at {commit[:8]}{' (dirty)' if dirty else ''}; differences may come from code changes")
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable] + _with_out_dir(r["command"], str(Path(tmp) / "rerun"))
            print("running:", " ".join(cmd[1:]))
            proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
            if proc.returncode != 0:
                print(proc.stdout[-2000:], proc.stderr[-2000:])
                return 1
            new = json.loads((Path(tmp) / "rerun" / "summary.json").read_text())
            new_hash = new.get("reproducibility", {}).get("data_hash_close")
        diffs = compare_summaries(_summary(r["output_dir"]), new)
        if new_hash and new_hash != r["data_hash"]:
            print(f"data changed since the run: hash {r['data_hash']} -> {new_hash}")
        if diffs:
            print(f"NOT REPRODUCED: {len(diffs)} numbers differ")
            for key, x, y in diffs[:40]:
                print(f"  {key}: {x:.10g} -> {y:.10g}")
            return 1
        print(f"REPRODUCED: every number in {r['run_id']} matches")
    elif args.cmd == "delete":
        reg.delete(args.run_id)
        print("deleted (output files left in place)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
