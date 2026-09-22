"""Tradable universe, derived from /v3/exchangeInfo — never hard-coded.

Also owns precision rounding and minimum-notional validation, since both are
per-pair facts that only exchangeInfo knows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.execution.client import PublicMarketDataClient


@dataclass(frozen=True)
class TradingRule:
    pair: str
    coin: str
    unit: str
    can_trade: bool
    price_precision: int
    amount_precision: int
    min_order_notional: float

    def round_price(self, price: float) -> float:
        factor = 10 ** self.price_precision
        return math.floor(price * factor) / factor

    def round_quantity(self, quantity: float) -> float:
        factor = 10 ** self.amount_precision
        return math.floor(quantity * factor) / factor

    def meets_min_notional(self, price: float, quantity: float) -> bool:
        return price * quantity >= self.min_order_notional

    def clamp_order(self, price: float, quantity: float) -> tuple[float, float]:
        """Round to exchange precision. Raises if the rounded order would
        fall below the minimum notional (caller should skip the order in
        that case, not silently bump the size up)."""
        p = self.round_price(price)
        q = self.round_quantity(quantity)
        if not self.meets_min_notional(p, q):
            raise ValueError(
                f"{self.pair}: order notional {p * q:.6f} below MiniOrder "
                f"{self.min_order_notional} after rounding to exchange precision"
            )
        return p, q


class Universe:
    def __init__(
        self,
        rules: dict[str, TradingRule],
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
    ):
        self._rules = rules
        self._whitelist = set(whitelist or [])
        self._blacklist = set(blacklist or [])

    @classmethod
    def from_exchange_info(
        cls,
        client: PublicMarketDataClient,
        whitelist: list[str] | None = None,
        blacklist: list[str] | None = None,
    ) -> "Universe":
        info = client.get_exchange_info()
        rules: dict[str, TradingRule] = {}
        for pair, meta in info.get("TradePairs", {}).items():
            rules[pair] = TradingRule(
                pair=pair,
                coin=meta["Coin"],
                unit=meta["Unit"],
                can_trade=bool(meta.get("CanTrade", False)),
                price_precision=int(meta["PricePrecision"]),
                amount_precision=int(meta["AmountPrecision"]),
                min_order_notional=float(meta["MiniOrder"]),
            )
        return cls(rules, whitelist=whitelist, blacklist=blacklist)

    def tradable_pairs(self) -> list[str]:
        pairs = [p for p, r in self._rules.items() if r.can_trade]
        if self._whitelist:
            pairs = [p for p in pairs if p in self._whitelist]
        pairs = [p for p in pairs if p not in self._blacklist]
        return sorted(pairs)

    def rule_for(self, pair: str) -> TradingRule:
        if pair not in self._rules:
            raise KeyError(f"{pair} is not listed in exchangeInfo")
        return self._rules[pair]

    def is_tradable(self, pair: str) -> bool:
        return pair in self.tradable_pairs()
