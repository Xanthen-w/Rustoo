"""The trading loop.

Every `poll_seconds`:
  1. read Roostoo tickers and the account balance; snapshot equity every
     `snapshot_minutes`;
  2. once per closed hourly bar (a few minutes after it closes, so the
     signal source has finalized it): fetch recent Binance hourly closes,
     compute the strategy's target weights on them — exactly the function
     the backtests ran — and plan + execute the orders that move the
     account toward those targets.

Execution policy mirrors the backtest engine: trade an asset only when its
weight drifts past `rebalance_threshold`, except at the scheduled exact
rebalances: every hourly bar whose UTC hour is in `rebalance_hours_utc`. If a UTC day reaches
`activity_fallback_hour_utc` with no filled order, one exact rebalance is
forced — the competition requires trades on >= 8 days
(docs/COMPETITION_RULES.md).

Kill switch: while a file named by `stop_file` exists, decisions are still
computed and logged but no orders are sent.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.bot.store import BotStore, utc_now
from src.data.market_data import tickers_from_response
from src.execution.portfolio import parse_wallet, plan_orders
from src.strategy.signals import STRATEGIES

logger = logging.getLogger(__name__)

FILLED_STATUSES = {"FILLED", "PARTIALLY_FILLED", "SIMULATED"}


@dataclass
class BotConfig:
    strategy: str = "trend_vol_target"
    strategy_params: dict = field(default_factory=dict)
    assets: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    history_bars: int = 2000
    rebalance_threshold: float = 0.05
    rebalance_hours_utc: tuple = (0,)
    activity_fallback_hour_utc: int | None = 12
    decision_delay_minutes: int = 2
    max_bar_staleness_hours: float = 2.0
    poll_seconds: float = 60.0
    snapshot_minutes: int = 15
    fee_rate: float = 0.001
    cash_buffer: float = 0.005
    stop_file: Path = Path("STOP")

    @classmethod
    def from_strategy_yaml(cls, cfg: dict, **overrides) -> "BotConfig":
        params = dict(cfg.get("trend_vol_target", {}))
        policy = cfg.get("execution_policy", {})
        assets = list(params.get("assets", ["BTC/USD", "ETH/USD"]))
        params["assets"] = tuple(assets)
        return cls(
            strategy="trend_vol_target",
            strategy_params=params,
            assets=assets,
            rebalance_threshold=float(policy.get("rebalance_threshold", 0.05)),
            rebalance_hours_utc=tuple(policy.get("rebalance_hours_utc", [0])),
            **overrides,
        )


class TradingBot:
    def __init__(self, config: BotConfig, public_client, broker, feed, universe, store: BotStore,
                 clock=utc_now, sleep=time.sleep):
        self.config = config
        self.public = public_client
        self.broker = broker
        self.feed = feed
        self.universe = universe
        self.store = store
        self._clock = clock
        self._sleep = sleep
        self._running = True
        for pair in config.assets:
            if not universe.is_tradable(pair):
                raise ValueError(f"{pair} is not tradable on Roostoo")

    # -- helpers -------------------------------------------------------------

    def _now(self) -> pd.Timestamp:
        ts = pd.Timestamp(self._clock())
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    def _account_and_tickers(self):
        tickers = tickers_from_response(self.public.get_ticker())
        account = parse_wallet(self.broker.balance())
        return account, tickers

    def _traded_today(self, now: pd.Timestamp) -> bool:
        return now.strftime("%Y-%m-%d") in set(self.store.filled_order_days())

    def _maybe_snapshot(self, now: pd.Timestamp, account, tickers) -> float:
        equity = account.equity(tickers)
        last = self.store.get("last_snapshot")
        if last is None or now - pd.Timestamp(last) >= pd.Timedelta(minutes=self.config.snapshot_minutes):
            positions = {p: round(v, 2) for p, v in account.position_values(tickers).items()}
            self.store.record_equity(equity, account.cash, positions, ts=now.to_pydatetime())
            self.store.set("last_snapshot", now.isoformat())
        return equity

    def _due_bar(self, now: pd.Timestamp) -> pd.Timestamp | None:
        """The latest closed hourly bar, if it hasn't been acted on yet and
        the decision delay has passed."""
        bar = now.floor("1h")
        if now - bar < pd.Timedelta(minutes=self.config.decision_delay_minutes):
            bar -= pd.Timedelta(hours=1)
        last = self.store.get("last_decision_bar")
        if last is not None and pd.Timestamp(last) >= bar:
            return None
        return bar

    def target_weights(self, now: pd.Timestamp) -> tuple[dict[str, float], pd.Timestamp]:
        closes = self.feed.close_panel(self.config.assets, self.config.history_bars, now)
        weights = STRATEGIES[self.config.strategy](closes, **self.config.strategy_params)
        last = weights.iloc[-1]
        return {pair: float(last.get(pair, 0.0)) for pair in self.config.assets}, closes.index[-1]

    # -- one iteration ---------------------------------------------------------

    def tick(self) -> dict:
        now = self._now()
        account, tickers = self._account_and_tickers()
        equity = self._maybe_snapshot(now, account, tickers)
        bar = self._due_bar(now)
        if bar is None:
            return {"action": "idle", "equity": equity}

        targets, last_bar = self.target_weights(now)
        if now - last_bar > pd.Timedelta(hours=self.config.max_bar_staleness_hours):
            note = f"signal data stale (last bar {last_bar}); no trading"
            logger.error(note)
            self.store.record_decision(bar_close=bar, equity=equity, targets=targets, current={}, scheduled=False,
                                       planned_orders=0, note=note, ts=now.to_pydatetime())
            self.store.set("last_decision_bar", bar.isoformat())
            return {"action": "stale", "equity": equity}

        today = bar.strftime("%Y-%m-%d")
        # Each hourly bar is decided at most once (last_decision_bar), so every
        # bar at a configured hour is its own scheduled rebalance.
        scheduled = bar.hour in self.config.rebalance_hours_utc
        fallback = (not scheduled and self.config.activity_fallback_hour_utc is not None
                    and bar.hour >= self.config.activity_fallback_hour_utc
                    and self.store.get("last_fallback_date") != today and not self._traded_today(now))
        exact = scheduled or fallback

        orders, info = plan_orders(targets, account, tickers, self.universe,
                                   rebalance_threshold=self.config.rebalance_threshold, scheduled=exact,
                                   fee_rate=self.config.fee_rate, cash_buffer=self.config.cash_buffer)
        current = {p: round(v / equity, 4) if equity else 0.0 for p, v in account.position_values(tickers).items()}
        kill = Path(self.config.stop_file).exists()
        note = "; ".join(filter(None, [
            "scheduled rebalance" if scheduled else "",
            "activity fallback rebalance" if fallback else "",
            f"KILL SWITCH ({self.config.stop_file}) - orders not sent" if kill and orders else "",
            f"skipped: {info['skipped']}" if info.get("skipped") else "",
        ]))
        self.store.record_decision(bar_close=bar, equity=equity, targets={k: round(v, 4) for k, v in targets.items()},
                                   current=current, scheduled=exact, planned_orders=len(orders), note=note,
                                   ts=now.to_pydatetime())
        logger.info("decision", extra={"bar": str(bar), "targets": targets, "current": current,
                                       "orders": len(orders), "note": note})

        results = []
        if not kill:
            for order in orders:
                record = self.broker.execute(order)
                self.store.record_order(record, ts=self._now().to_pydatetime())
                results.append(record)
                log = logger.info if record.get("status") in FILLED_STATUSES else logger.error
                log("order", extra={k: record.get(k) for k in ("pair", "side", "quantity", "status", "avg_price", "fee", "error")})

        self.store.set("last_decision_bar", bar.isoformat())
        if scheduled:
            self.store.set("last_scheduled_bar", bar.isoformat())
        if fallback:
            self.store.set("last_fallback_date", today)
        return {"action": "decided", "equity": equity, "orders": results, "scheduled": scheduled,
                "fallback": fallback, "killed": kill}

    # -- loop ----------------------------------------------------------------------

    def stop(self, *_):
        logger.info("shutdown requested")
        self._running = False

    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        logger.info("bot started", extra={"mode": getattr(self.broker, "mode", "?"), "assets": self.config.assets})
        while self._running:
            started = time.monotonic()
            try:
                self.tick()
            except Exception:
                # Never die on a transient failure: log it and try again next
                # poll. The audit trail keeps the failed API calls.
                logger.exception("tick failed")
            elapsed = time.monotonic() - started
            if self._running:
                self._sleep(max(self.config.poll_seconds - elapsed, 1.0))
        logger.info("bot stopped")
