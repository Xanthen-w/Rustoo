"""Account state and order planning for the live bot.

`plan_orders` is a pure function — targets + wallet + prices + exchange
rules in, a list of orders out — so the logic that decides what to trade is
fully testable without the network. It applies the same execution policy as
the backtest engine (backtest/engine.py): skip assets whose weight drift is
inside `rebalance_threshold` unless this is a scheduled rebalance, and never
block an exit to zero.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.data.market_data import Ticker

logger = logging.getLogger(__name__)

QUOTE = "USD"


@dataclass
class AccountSnapshot:
    cash_free: float
    cash_locked: float
    free: dict[str, float] = field(default_factory=dict)  # coin -> free qty
    locked: dict[str, float] = field(default_factory=dict)

    @property
    def cash(self) -> float:
        return self.cash_free + self.cash_locked

    def quantity(self, coin: str) -> float:
        return self.free.get(coin, 0.0) + self.locked.get(coin, 0.0)

    def coins(self) -> list[str]:
        return sorted(c for c in set(self.free) | set(self.locked) if self.quantity(c) > 0)

    def position_values(self, tickers: dict[str, Ticker]) -> dict[str, float]:
        """Pair -> market value at LastPrice. Coins without a USD ticker are
        reported and excluded (they can't be valued or traded)."""
        values = {}
        for coin in self.coins():
            pair = f"{coin}/{QUOTE}"
            if pair in tickers:
                values[pair] = self.quantity(coin) * tickers[pair].price
            else:
                logger.warning("holding without a USD ticker; excluded from equity", extra={"coin": coin})
        return values

    def equity(self, tickers: dict[str, Ticker]) -> float:
        return self.cash + sum(self.position_values(tickers).values())


def parse_wallet(balance_response: dict) -> AccountSnapshot:
    """GET /v3/balance -> AccountSnapshot. Accepts `Wallet` (documented) or
    `SpotWallet` (newer deployments); missing numbers mean 0 per the docs."""
    wallet = balance_response.get("Wallet") or balance_response.get("SpotWallet") or {}
    free, locked = {}, {}
    for coin, amounts in wallet.items():
        f = float((amounts or {}).get("Free", 0.0) or 0.0)
        lk = float((amounts or {}).get("Lock", 0.0) or 0.0)
        if coin == QUOTE:
            continue
        free[coin], locked[coin] = f, lk
    usd = wallet.get(QUOTE, {}) or {}
    return AccountSnapshot(
        cash_free=float(usd.get("Free", 0.0) or 0.0),
        cash_locked=float(usd.get("Lock", 0.0) or 0.0),
        free=free,
        locked=locked,
    )


@dataclass(frozen=True)
class PlannedOrder:
    pair: str
    side: str  # BUY / SELL
    quantity: float
    est_price: float
    reason: str

    @property
    def notional(self) -> float:
        return self.quantity * self.est_price


def _sell_price(t: Ticker) -> float:
    return t.bid if t.bid > 0 else t.price


def _buy_price(t: Ticker) -> float:
    return t.ask if t.ask > 0 else t.price


def plan_orders(
    targets: dict[str, float],
    account: AccountSnapshot,
    tickers: dict[str, Ticker],
    rules,
    *,
    rebalance_threshold: float = 0.0,
    scheduled: bool = False,
    fee_rate: float = 0.001,
    cash_buffer: float = 0.005,
) -> tuple[list[PlannedOrder], dict]:
    """Orders that move the account toward `targets` (pair -> weight of
    equity, long-only, sum <= 1). Held assets missing from `targets` are
    sold. `rules` must provide `rule_for(pair) -> TradingRule`.

    Returns (orders, info) with sells before buys; `info` explains skips.
    """
    if any(w < 0 for w in targets.values()) or sum(targets.values()) > 1.0 + 1e-9:
        raise ValueError("targets must be long-only and sum to at most 1")
    equity = account.equity(tickers)
    info: dict = {"equity": equity, "skipped": {}}
    if equity <= 0:
        return [], info

    values = account.position_values(tickers)
    pairs = sorted(set(targets) | set(values))
    sells: list[PlannedOrder] = []
    buy_wants: dict[str, float] = {}

    for pair in pairs:
        if pair not in tickers:
            info["skipped"][pair] = "no ticker"
            continue
        target_w = float(targets.get(pair, 0.0))
        current_value = values.get(pair, 0.0)
        drift = target_w - current_value / equity
        exit_to_zero = target_w == 0.0 and current_value > 0
        if not scheduled and not exit_to_zero and abs(drift) < rebalance_threshold:
            info["skipped"][pair] = f"inside band ({drift:+.4f})"
            continue
        delta_value = target_w * equity - current_value
        if delta_value < 0:
            coin = pair.split("/")[0]
            price = _sell_price(tickers[pair])
            qty = account.free.get(coin, 0.0) if exit_to_zero else min(-delta_value / price, account.free.get(coin, 0.0))
            sells.append(("exit" if exit_to_zero else "rebalance", pair, qty, price))
        elif delta_value > 0:
            buy_wants[pair] = delta_value

    orders: list[PlannedOrder] = []
    proceeds = 0.0
    for reason, pair, qty, price in sells:
        rule = rules.rule_for(pair)
        qty = rule.round_quantity(qty)
        if qty <= 0 or not rule.meets_min_notional(price, qty):
            info["skipped"][pair] = f"sell below MiniOrder ({qty * price:.2f})"
            continue
        orders.append(PlannedOrder(pair, "SELL", qty, price, reason))
        proceeds += qty * price * (1 - fee_rate)

    budget = max(account.cash_free + proceeds - cash_buffer * equity, 0.0)
    total_wanted = sum(buy_wants.values())
    scale = min(1.0, budget / (total_wanted * (1 + fee_rate))) if total_wanted > 0 else 0.0
    if 0 < scale < 1:
        info["buy_scale"] = scale
    for pair, notional in sorted(buy_wants.items()):
        rule = rules.rule_for(pair)
        price = _buy_price(tickers[pair])
        qty = rule.round_quantity(notional * scale / price)
        if qty <= 0 or not rule.meets_min_notional(price, qty):
            info["skipped"][pair] = f"buy below MiniOrder ({qty * price:.2f})"
            continue
        orders.append(PlannedOrder(pair, "BUY", qty, price, "rebalance"))
    return orders, info
