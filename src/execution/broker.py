"""Order execution: LiveBroker (Roostoo) and PaperBroker (simulated).

Both expose the same interface to the bot:
    balance() -> dict            # GET /v3/balance-shaped response
    execute(order) -> dict       # one order-record dict for BotStore.record_order

LiveBroker only works with a PrivateTradingClient built with live trading
enabled (APP_ENV=live and LIVE_TRADING=true); otherwise the client itself
refuses. Order outcomes are always read from Roostoo's response — fill
quantity, average price, commission and maker/taker role — never assumed.
"""
from __future__ import annotations

import logging
import time

from src.execution.client import (
    PrivateTradingClient,
    RoostooAPIError,
    RoostooError,
    RoostooOrderStateUnknownError,
    RoostooSafetyError,
)
from src.execution.portfolio import QUOTE, PlannedOrder

logger = logging.getLogger(__name__)


def _order_record(order: PlannedOrder, mode: str, **fields) -> dict:
    return {
        "mode": mode, "pair": order.pair, "side": order.side, "quantity": order.quantity,
        "est_price": order.est_price, "reason": order.reason, **fields,
    }


def _from_order_detail(order: PlannedOrder, mode: str, detail: dict, raw) -> dict:
    return _order_record(
        order, mode,
        status=detail.get("Status", "UNKNOWN"),
        order_id=detail.get("OrderID"),
        filled_qty=float(detail.get("FilledQuantity", 0.0) or 0.0),
        avg_price=float(detail.get("FilledAverPrice", 0.0) or 0.0),
        fee=float(detail.get("CommissionChargeValue", 0.0) or 0.0),
        role=detail.get("Role"),
        raw=raw,
    )


class LiveBroker:
    mode = "live"

    def __init__(self, client: PrivateTradingClient, min_seconds_between_orders: float = 60.0,
                 sleep=time.sleep, monotonic=time.monotonic):
        if not client.live_trading_enabled:
            raise RoostooSafetyError("LiveBroker requires a client with live trading enabled")
        self.client = client
        self.min_gap = min_seconds_between_orders
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_order_at: float | None = None

    def balance(self) -> dict:
        return self.client.get_balance()

    def _wait_for_slot(self) -> None:
        if self._last_order_at is None:
            return
        wait = self.min_gap - (self._monotonic() - self._last_order_at)
        if wait > 0:
            logger.info("waiting for order throttle", extra={"seconds": round(wait, 1)})
            self._sleep(wait + 0.5)

    def _reconcile(self, order: PlannedOrder, sent_at_ms: int) -> dict:
        """After an order's outcome was lost in transit, look for it among
        the pair's recent orders instead of blindly resubmitting."""
        try:
            recent = self.client.query_order(pair=order.pair, limit=10).get("OrderMatched", [])
        except RoostooError as exc:
            return _order_record(order, self.mode, status="UNKNOWN", error=f"reconcile failed: {exc}")
        for detail in recent:
            if (detail.get("Side") == order.side
                    and abs(float(detail.get("Quantity", 0) or 0) - order.quantity) < 1e-12
                    and int(detail.get("CreateTimestamp", 0) or 0) >= sent_at_ms - 120_000):
                logger.warning("order outcome recovered by reconciliation", extra={"pair": order.pair})
                return _from_order_detail(order, self.mode, detail, {"reconciled": detail})
        return _order_record(order, self.mode, status="NOT_FOUND",
                             error="order outcome unknown and no matching recent order; assumed not placed")

    def execute(self, order: PlannedOrder) -> dict:
        self._wait_for_slot()
        sent_at_ms = int(time.time() * 1000)
        try:
            response = self.client.place_order(order.pair, order.side, "MARKET", order.quantity)
            return _from_order_detail(order, self.mode, response.get("OrderDetail", {}) or {}, response)
        except RoostooOrderStateUnknownError as exc:
            logger.error("order outcome unknown; reconciling", extra={"pair": order.pair, "error": str(exc)})
            return self._reconcile(order, sent_at_ms)
        except RoostooAPIError as exc:
            return _order_record(order, self.mode, status="REJECTED", error=exc.err_msg, raw=exc.response)
        except RoostooSafetyError as exc:
            return _order_record(order, self.mode, status="BLOCKED", error=str(exc))
        except RoostooError as exc:
            return _order_record(order, self.mode, status="FAILED", error=str(exc))
        finally:
            self._last_order_at = self._monotonic()


class PaperBroker:
    """Simulated account for dry runs: fills every order at its estimated
    price (bid for sells, ask for buys) with the taker fee, and keeps the
    wallet in the bot store so it survives restarts."""

    mode = "paper"
    WALLET_KEY = "paper_wallet"

    def __init__(self, store, initial_cash: float = 100_000.0, fee_rate: float = 0.001):
        self.store = store
        self.fee_rate = fee_rate
        if store.get(self.WALLET_KEY) is None:
            store.set(self.WALLET_KEY, {QUOTE: initial_cash})

    def _wallet(self) -> dict[str, float]:
        return dict(self.store.get(self.WALLET_KEY))

    def balance(self) -> dict:
        return {"Success": True, "Wallet": {c: {"Free": q, "Lock": 0.0} for c, q in self._wallet().items()}}

    def execute(self, order: PlannedOrder) -> dict:
        wallet = self._wallet()
        coin = order.pair.split("/")[0]
        notional = order.quantity * order.est_price
        fee = notional * self.fee_rate
        if order.side == "BUY":
            if wallet.get(QUOTE, 0.0) < notional + fee - 1e-9:
                return _order_record(order, self.mode, status="REJECTED", error="insufficient USD (paper)")
            wallet[QUOTE] = wallet.get(QUOTE, 0.0) - notional - fee
            wallet[coin] = wallet.get(coin, 0.0) + order.quantity
        else:
            if wallet.get(coin, 0.0) < order.quantity - 1e-12:
                return _order_record(order, self.mode, status="REJECTED", error=f"insufficient {coin} (paper)")
            wallet[coin] = wallet.get(coin, 0.0) - order.quantity
            wallet[QUOTE] = wallet.get(QUOTE, 0.0) + notional - fee
        self.store.set(self.WALLET_KEY, wallet)
        return _order_record(order, self.mode, status="SIMULATED", filled_qty=order.quantity,
                             avg_price=order.est_price, fee=fee, role="TAKER")
