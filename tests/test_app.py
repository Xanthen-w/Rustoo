"""Local research app: command builders (pure) and page rendering."""
import json
from pathlib import Path

import pytest

from app import helpers as h
from src.strategy.signals import STRATEGIES


def test_backtest_argv_maps_every_option():
    argv = h.backtest_argv({"data": Path("data/binance/5m"), "strategy": "trend_vol_target",
                            "params": {"assets": ["BTC/USD"], "trend_span": 480}, "costs": "base", "fee_bps": 20.0,
                            "slippage_bps": None, "execution_price": "open", "latency_bars": 2, "rebalance_threshold": 0.05,
                            "rebalance_hours": [0, 12], "capital": 50_000, "seeds": 5, "benchmarks": ["BTC/USD"],
                            "split": "custom", "start": "2025-03-01", "end": "2025-05-01", "use_holdout": False})
    i = argv.index
    assert argv[0] == "scripts/backtest_report.py"
    assert json.loads(argv[i("--params") + 1]) == {"assets": ["BTC/USD"], "trend_span": 480}
    assert argv[i("--fee-bps") + 1] == "20.0" and "--slippage-bps" not in argv
    assert argv[i("--rebalance-hours") + 1: i("--rebalance-hours") + 3] == ["0", "12"]
    assert argv[i("--split") + 1] == "custom" and argv[i("--start") + 1] == "2025-03-01"
    assert "--use-holdout" not in argv


FULL_BACKTEST = {"data": "data/binance/5m", "strategy": "trend_vol_target", "params": {"assets": ["BTC/USD"]},
                 "costs": "base", "fee_bps": 20.0, "maker_share": 0.2, "slippage_bps": 3.0, "spread_bps": 4.0,
                 "execution_price": "close", "latency_bars": 2, "rebalance_threshold": 0.05, "rebalance_hours": [0],
                 "capital": 50_000, "seeds": 5, "benchmarks": ["BTC/USD"], "split": "custom", "start": "2025-03-01",
                 "end": "2025-05-01", "use_holdout": True}


@pytest.mark.parametrize("script,argv", [
    ("backtest_report", h.backtest_argv(FULL_BACKTEST)),
    ("robustness_report", h.robustness_argv({"data": "d", "split": "test", "use_holdout": True, "mc_seeds": [1, 2],
                                             "landscape": True, "params": {"assets": ["BTC/USD"]}})),
    ("import_data", h.import_argv([Path("a.xlsx"), Path("b.csv")], {"source": "s", "tz": "UTC", "symbol": "BTC/USD",
                                                                     "mapping": {"close": "Last"}, "reference": "ref",
                                                                     "not_crypto": True, "mode": "remove"})),
])
def test_app_commands_are_accepted_by_the_real_scripts(script, argv, monkeypatch):
    """Parse the exact argv the app builds with each script's own argparse
    parser (then stop before doing any work): an unknown or malformed flag
    fails here instead of in the browser."""
    import argparse
    import importlib.util

    class Parsed(Exception):
        pass

    real_parse = argparse.ArgumentParser.parse_args

    def parse_then_stop(self, args=None, namespace=None):
        raise Parsed(real_parse(self, argv[1:]))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_then_stop)
    spec = importlib.util.spec_from_file_location(script, h.REPO_ROOT / "scripts" / f"{script}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with pytest.raises(Parsed) as parsed:
        mod.main()
    ns = parsed.value.args[0]
    if script == "backtest_report":
        assert (ns.split, ns.fee_bps, ns.latency_bars, ns.rebalance_hours, ns.use_holdout) == ("custom", 20.0, 2, [0], True)
    if script == "import_data":
        assert ns.map == ["close=Last"] and ns.mode == "remove"


def test_robustness_and_import_argv():
    r = h.robustness_argv({"data": "d", "split": "validation", "mc_seeds": [1, 2], "landscape": False, "params": None})
    assert "--no-landscape" in r and r[r.index("--mc-seeds") + 1: r.index("--mc-seeds") + 3] == ["1", "2"]
    assert "--params" not in r
    imp = h.import_argv([Path("a.xlsx")], {"source": "bb", "tz": "Asia/Kolkata", "mapping": {"close": "PX_LAST"}})
    assert imp[:2] == ["scripts/import_data.py", "a.xlsx"] and "close=PX_LAST" in imp


def test_parse_outputs_and_display_command():
    found = h.parse_outputs("x\nreport: /tmp/r/report.html\nrun id: backtest-1-abc  (python ...)\n")
    assert found == {"report": Path("/tmp/r/report.html"), "run_id": "backtest-1-abc"}
    shown = h.display_command(["scripts/x.py", "--params", '{"a": 1}'])
    assert shown.endswith("""--params '{"a": 1}'""")


def test_strategy_defaults_overlay_live_config():
    d = h.strategy_defaults(STRATEGIES["trend_vol_target"], {"trend_span": 960, "unknown": 1})
    assert d["trend_span"] == 960 and "unknown" not in d and "min_exposure" in d
    assert h.strategy_defaults(STRATEGIES["buy_and_hold"])["symbol"] is None  # required param


def test_every_page_renders():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    for page in ["Overview", "Data import", "Backtest", "Robustness", "Run history", "Paper bot"]:
        at = AppTest.from_file(str(h.REPO_ROOT / "app" / "app.py"), default_timeout=60)
        at.run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
