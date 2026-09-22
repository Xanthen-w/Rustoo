import pytest

from src.data.universe import TradingRule, Universe

BTC_RULE = TradingRule(
    pair="BTC/USD",
    coin="BTC",
    unit="USD",
    can_trade=True,
    price_precision=2,
    amount_precision=6,
    min_order_notional=1.0,
)


def test_round_price_truncates_to_precision():
    assert BTC_RULE.round_price(12345.6789) == 12345.67


def test_round_quantity_truncates_to_precision():
    assert BTC_RULE.round_quantity(0.1234567) == 0.123456


def test_clamp_order_below_min_notional_raises():
    tiny_rule = TradingRule(
        pair="X/USD", coin="X", unit="USD", can_trade=True,
        price_precision=2, amount_precision=2, min_order_notional=10.0,
    )
    with pytest.raises(ValueError):
        tiny_rule.clamp_order(price=1.0, quantity=0.5)


def test_clamp_order_within_min_notional_ok():
    price, qty = BTC_RULE.clamp_order(price=50000.123, quantity=0.0001234)
    assert price == 50000.12
    assert qty == 0.000123


def test_universe_filters_untradable_and_blacklisted():
    rules = {
        "BTC/USD": BTC_RULE,
        "ETH/USD": TradingRule("ETH/USD", "ETH", "USD", True, 2, 4, 1.0),
        "DEAD/USD": TradingRule("DEAD/USD", "DEAD", "USD", False, 2, 2, 1.0),
    }
    universe = Universe(rules, blacklist=["ETH/USD"])
    assert universe.tradable_pairs() == ["BTC/USD"]
    assert universe.is_tradable("DEAD/USD") is False


def test_universe_whitelist_intersects():
    rules = {
        "BTC/USD": BTC_RULE,
        "ETH/USD": TradingRule("ETH/USD", "ETH", "USD", True, 2, 4, 1.0),
    }
    universe = Universe(rules, whitelist=["BTC/USD"])
    assert universe.tradable_pairs() == ["BTC/USD"]
