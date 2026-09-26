"""Run registry, summary comparison, and the frozen forward-test workflow."""
import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.registry import RunRegistry, compare_summaries, flatten_numbers

REPO = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_registry_roundtrip_and_prefix_lookup(tmp_path):
    reg = RunRegistry(tmp_path / "r.sqlite3")
    a = reg.register("backtest", ["scripts/x.py", "--split", "train"], {"p": 1}, "h1", tmp_path, {"net_return": 0.1})
    b = reg.register("robustness", ["scripts/y.py"], {"p": 2}, "h2", tmp_path, {"net_return": -0.1})
    assert reg.get(a)["config"] == {"p": 1} and reg.get(a)["command"][1] == "--split"
    assert reg.get(a.split("-")[-1])["run_id"] == a  # unique suffix
    assert [r["run_id"] for r in reg.list(kind="robustness")] == [b]
    with pytest.raises(KeyError):
        reg.get("nope")
    reg.delete(b)
    assert len(reg.list()) == 1


def test_run_id_depends_on_config_and_data():
    from backtest.registry import config_hash
    assert config_hash({"a": 1}, "x") != config_hash({"a": 2}, "x") != config_hash({"a": 1}, "y")
    assert config_hash({"a": 1, "b": 2}, "x") == config_hash({"b": 2, "a": 1}, "x")


def test_compare_summaries_ignores_reproducibility_and_finds_changes():
    a = {"net": {"sharpe": 1.0, "ret": 0.1}, "reproducibility": {"generated_at": "t1", "x": 1.0}, "label": "s"}
    b = {"net": {"sharpe": 1.0, "ret": 0.1 + 1e-15}, "reproducibility": {"generated_at": "t2", "x": 2.0}, "label": "s"}
    assert compare_summaries(a, b) == []
    c = {"net": {"sharpe": 1.2, "ret": 0.1}}
    assert [d[0] for d in compare_summaries(a, c)] == ["net.sharpe"]
    assert flatten_numbers({"a": {"b": 1, "c": True}}) == {"a.b": 1.0}


@pytest.fixture
def forward_env(tmp_path, monkeypatch):
    ft = load_script("forward_test")
    monkeypatch.setattr(ft, "FREEZE_DIR", tmp_path / "forward")
    monkeypatch.setattr(ft, "RunRegistry", lambda: RunRegistry(tmp_path / "registry.sqlite3"))  # keep the real one clean
    data = tmp_path / "data"
    data.mkdir()
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01 00:05", periods=12 * 24 * 90, freq="5min", tz="UTC")
    for pair, base in (("BTC/USD", 80_000.0), ("ETH/USD", 3_000.0)):
        close = base * np.cumprod(1 + rng.normal(0, 0.001, len(idx)))
        df = pd.DataFrame({"open": close, "high": close * 1.001, "low": close * 0.999, "close": close, "volume": 10.0}, index=idx)
        df.index.name = "timestamp"
        df.iloc[: 12 * 24 * 80].to_parquet(data / f"{pair.replace('/', '-')}.parquet")  # data known at freeze time
        df.to_parquet(tmp_path / f"full_{pair.replace('/', '-')}.parquet")
    return ft, tmp_path, data


def test_freeze_is_immutable_and_tamper_evident(forward_env):
    ft, tmp, data = forward_env
    args = Namespace(name="t1", data=data, costs="base", allow_dirty=True)
    assert ft.freeze(args) == 0
    assert ft.freeze(args) == 1  # can't overwrite
    path = tmp / "forward" / "t1.json"
    record = json.loads(path.read_text())
    assert record["data_end"].startswith("2026-03-22") and record["freeze_hash"] == ft._hash(record)
    record["params"]["target_vol"] = 0.9  # tamper
    path.write_text(json.dumps(record))
    assert ft.evaluate(Namespace(name="t1", data=data, min_days=3, out_dir=tmp / "out")) == 1


def test_forward_evaluation_uses_only_post_freeze_bars(forward_env):
    ft, tmp, data = forward_env
    assert ft.freeze(Namespace(name="t2", data=data, costs="base", allow_dirty=True)) == 0
    # nothing new yet -> refused
    assert ft.evaluate(Namespace(name="t2", data=data, min_days=3, out_dir=tmp / "o1")) == 1
    # "download" newer data, then evaluate
    for f in tmp.glob("full_*.parquet"):
        f.replace(data / f.name.replace("full_", ""))
    assert ft.evaluate(Namespace(name="t2", data=data, min_days=3, out_dir=tmp / "o2")) == 0
    summary = json.loads((tmp / "o2" / "summary.json").read_text())
    cutoff = pd.Timestamp(json.loads((tmp / "forward" / "t2.json").read_text())["data_end"])
    assert pd.Timestamp(summary["period"]["start"]) > cutoff
    assert "FORWARD" in summary["period"]["label"]
