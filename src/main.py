"""Bot entry point.

    python -m src.main              # run continuously (paper mode unless live trading is enabled)
    python -m src.main --once       # a single iteration, then exit (smoke test)
    python -m src.main --status     # print the audit-trail summary and exit

Mode: orders go to Roostoo only when APP_ENV=live AND LIVE_TRADING=true
(and API credentials are set). Otherwise the bot runs against a simulated
PaperBroker wallet — same data, same decisions, no orders sent.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.bot.logging_setup import setup_logging  # noqa: E402
from src.bot.runner import BotConfig, TradingBot  # noqa: E402
from src.bot.store import BotStore  # noqa: E402
from src.config.settings import load_settings  # noqa: E402
from src.data.live_history import BinanceKlineFeed  # noqa: E402
from src.data.universe import Universe  # noqa: E402
from src.execution.broker import LiveBroker, PaperBroker  # noqa: E402
from src.execution.client import build_clients_from_settings  # noqa: E402

logger = logging.getLogger("src.main")


def status(store: BotStore) -> None:
    print(json.dumps({
        "mode_paper_wallet": store.get(PaperBroker.WALLET_KEY),
        "decisions": store.count("decisions"),
        "orders": store.count("orders"),
        "api_calls": store.count("api_calls"),
        "equity_snapshots": store.count("equity"),
        "active_trading_days": store.filled_order_days(),
        "last_decision_bar": store.get("last_decision_bar"),
        "last_orders": store.orders(limit=5),
    }, indent=2, default=str))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--once", action="store_true", help="Run one iteration and exit.")
    parser.add_argument("--status", action="store_true", help="Print the audit summary and exit.")
    parser.add_argument("--db", type=Path, default=None, help="SQLite path (default data/state/bot-<mode>.sqlite3).")
    parser.add_argument("--strategy-config", type=Path, default=REPO_ROOT / "config" / "strategy.yaml")
    args = parser.parse_args()

    settings = load_settings()
    live = settings.is_live_trading_enabled
    mode = "live" if live else "paper"
    setup_logging(REPO_ROOT / settings.raw.get("logging", {}).get("dir", "logs"),
                  settings.raw.get("logging", {}).get("level", "INFO"))
    # Separate databases per mode, so paper runs never mix into the live audit trail.
    store = BotStore(args.db or REPO_ROOT / "data" / "state" / f"bot-{mode}.sqlite3")
    if args.status:
        status(store)
        return 0

    public, private = build_clients_from_settings(settings, on_request=store.record_api_call)
    if live:
        settings.assert_credentials_present()
        broker = LiveBroker(private, settings.execution.min_seconds_between_orders)
    else:
        broker = PaperBroker(store, fee_rate=settings.costs.taker_fee)
    universe_cfg = yaml.safe_load((REPO_ROOT / "config" / "universe.yaml").read_text()) or {}
    universe = Universe.from_exchange_info(public, universe_cfg.get("whitelist"), universe_cfg.get("blacklist"))

    strategy_cfg = yaml.safe_load(args.strategy_config.read_text()) or {}
    config = BotConfig.from_strategy_yaml(strategy_cfg, fee_rate=settings.costs.taker_fee,
                                          stop_file=REPO_ROOT / "STOP")
    bot = TradingBot(config, public, broker, BinanceKlineFeed("1h"), universe, store)
    logger.info("starting", extra={"mode": mode, "assets": config.assets, "params": config.strategy_params})
    if args.once:
        result = bot.tick()
        print(json.dumps({k: v for k, v in result.items() if k != "orders"} | {"orders": len(result.get("orders", []))},
                         default=str))
        return 0
    bot.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
