"""Live-trading path, offline: wallet parsing, order planning, brokers,
the kline feed, and the bot loop driven by fakes and a fake clock."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.bot.runner import BotConfig, TradingBot
from src.bot.store import BotStore
from src.data.live_history import BinanceKlineFeed
from src.data.market_data import Ticker
from src.data.universe import TradingRule, Universe
from src.execution.broker import LiveBroker, PaperBroker
from src.execution.client import RoostooAPIError, RoostooOrderStateUnknownError, RoostooSafetyError
from src.execution.portfolio import AccountSnapshot, PlannedOrder, parse_wallet, plan_orders
from src.strategy import signals

T0 = datetime(2026, 10, 4, 0, 0, tzinfo=timezone.utc)


def ticker(pair, price, spread=0.0):
    return Ticker(T0, pair, price, price * (1 - spread), price * (1 + spread), 0.0)


RULES = Universe({
    "BTC/USD": TradingRule("BTC/USD", "BTC", "USD", True, 2, 5, 1.0),
    "ETH/USD": TradingRule("ETH/USD", "ETH", "USD", True, 2, 4, 1.0),
    "DOGE/USD": TradingRule("DOGE/USD", "DOGE", "USD", True, 5, 0, 1.0),
})
TICKERS = {"BTC/USD": ticker("BTC/USD", 100_000.0), "ETH/USD": ticker("ETH/USD", 4_000.0),
           "DOGE/USD": ticker("DOGE/USD", 0.2)}


# -- wallet ------------------------------------------------------------------------

def test_parse_wallet_accepts_wallet_or_spotwallet_and_missing_zeros():
    for key in ("Wallet", "SpotWallet"):
        acct = parse_wallet({"Success": True, key: {"USD": {"Free": 50_000, "Lock": 100}, "BTC": {"Free": 0.25}}})
        assert acct.cash == 50_100
        assert acct.quantity("BTC") == 0.25
        assert acct.equity(TICKERS) == pytest.approx(50_100 + 25_000)


# -- planning --------------------------------------------------------------------------

def test_plan_from_cash_buys_targets_within_budget():
    acct = AccountSnapshot(cash_free=100_000, cash_locked=0)
    orders, info = plan_orders({"BTC/USD": 0.5, "ETH/USD": 0.5}, acct, TICKERS, RULES, fee_rate=0.001, cash_buffer=0.005)
    assert [o.side for o in orders] == ["BUY", "BUY"]
    spend = sum(o.notional * 1.001 for o in orders)
    assert spend <= 100_000 * 0.995 + 1e-6
    assert info["buy_scale"] < 1


def test_sells_come_first_and_fund_buys():
    acct = AccountSnapshot(cash_free=0.0, cash_locked=0, free={"BTC": 1.0})
    orders, _ = plan_orders({"BTC/USD": 0.3, "ETH/USD": 0.6}, acct, TICKERS, RULES)
    assert orders[0].side == "SELL" and orders[0].pair == "BTC/USD"
    assert orders[1].side == "BUY" and orders[1].pair == "ETH/USD"
    assert orders[1].notional < orders[0].notional  # buy funded from sell proceeds


def test_band_skips_small_drift_unless_scheduled():
    acct = AccountSnapshot(cash_free=48_000, cash_locked=0, free={"BTC": 0.52})  # 52% BTC
    orders, info = plan_orders({"BTC/USD": 0.5}, acct, TICKERS, RULES, rebalance_threshold=0.05)
    assert orders == [] and "inside band" in info["skipped"]["BTC/USD"]
    orders, _ = plan_orders({"BTC/USD": 0.5}, acct, TICKERS, RULES, rebalance_threshold=0.05, scheduled=True)
    assert len(orders) == 1 and orders[0].side == "SELL"


def test_untargeted_holdings_are_exited_even_inside_band():
    acct = AccountSnapshot(cash_free=99_000, cash_locked=0, free={"DOGE": 5_000.0})  # 1% DOGE
    orders, _ = plan_orders({"BTC/USD": 0.0}, acct, TICKERS, RULES, rebalance_threshold=0.05)
    assert [(o.pair, o.side, o.quantity, o.reason) for o in orders] == [("DOGE/USD", "SELL", 5000.0, "exit")]


def test_quantities_rounded_down_and_min_notional_respected():
    acct = AccountSnapshot(cash_free=100_000, cash_locked=0)
    orders, _ = plan_orders({"BTC/USD": 0.123456789}, acct, TICKERS, RULES, cash_buffer=0.0, fee_rate=0.0)
    assert orders[0].quantity == 0.12345  # 5 decimals, truncated
    tiny = AccountSnapshot(cash_free=100_000, cash_locked=0)
    orders, info = plan_orders({"BTC/USD": 0.000005}, tiny, TICKERS, RULES, cash_buffer=0.0)
    assert orders == [] and "MiniOrder" in info["skipped"]["BTC/USD"]


def test_plan_rejects_levered_targets():
    with pytest.raises(ValueError):
        plan_orders({"BTC/USD": 0.7, "ETH/USD": 0.7}, AccountSnapshot(1000, 0), TICKERS, RULES)


# -- brokers --------------------------------------------------------------------------------

def test_paper_broker_fills_with_fee_and_persists():
    store = BotStore(":memory:")
    broker = PaperBroker(store, initial_cash=10_000, fee_rate=0.001)
    rec = broker.execute(PlannedOrder("ETH/USD", "BUY", 1.0, 4_000.0, "rebalance"))
    assert rec["status"] == "SIMULATED" and rec["fee"] == pytest.approx(4.0)
    wallet = parse_wallet(PaperBroker(store).balance())  # new instance, same store
    assert wallet.cash == pytest.approx(10_000 - 4_004)
    assert wallet.quantity("ETH") == 1.0
    assert broker.execute(PlannedOrder("ETH/USD", "SELL", 2.0, 4_000.0, "x"))["status"] == "REJECTED"


class FakePrivateClient:
    def __init__(self, outcome, recent=None, live=True):
        self.outcome, self.recent, self.live_trading_enabled = outcome, recent or [], live
        self.placed = []

    def place_order(self, pair, side, order_type, quantity):
        self.placed.append((pair, side, order_type, quantity))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome

    def query_order(self, pair=None, limit=None):
        return {"OrderMatched": self.recent}

    def get_balance(self):
        return {"Wallet": {"USD": {"Free": 1.0}}}


ORDER = PlannedOrder("BTC/USD", "BUY", 0.1, 100_000.0, "rebalance")


def test_live_broker_requires_live_client():
    with pytest.raises(RoostooSafetyError):
        LiveBroker(FakePrivateClient({}, live=False))


def test_live_broker_records_real_fill_details():
    detail = {"OrderID": 81, "Status": "FILLED", "Role": "TAKER", "FilledQuantity": 0.1,
              "FilledAverPrice": 100_050.0, "CommissionChargeValue": 10.005}
    client = FakePrivateClient({"Success": True, "OrderDetail": detail})
    rec = LiveBroker(client, sleep=lambda s: None).execute(ORDER)
    assert client.placed == [("BTC/USD", "BUY", "MARKET", 0.1)]
    assert (rec["status"], rec["order_id"], rec["avg_price"], rec["fee"], rec["role"]) == ("FILLED", 81, 100_050.0, 10.005, "TAKER")


def test_live_broker_reconciles_unknown_outcome_without_resubmitting():
    import time as _t
    found = {"OrderID": 9, "Status": "FILLED", "Side": "BUY", "Quantity": 0.1, "FilledQuantity": 0.1,
             "FilledAverPrice": 1.0, "CreateTimestamp": int(_t.time() * 1000)}
    client = FakePrivateClient(RoostooOrderStateUnknownError("timeout"), recent=[found])
    rec = LiveBroker(client, sleep=lambda s: None).execute(ORDER)
    assert len(client.placed) == 1
    assert rec["status"] == "FILLED" and rec["order_id"] == 9
    client = FakePrivateClient(RoostooOrderStateUnknownError("timeout"), recent=[])
    assert LiveBroker(client, sleep=lambda s: None).execute(ORDER)["status"] == "NOT_FOUND"


def test_live_broker_records_rejection():
    client = FakePrivateClient(RoostooAPIError("insufficient balance", {"Success": False}))
    rec = LiveBroker(client, sleep=lambda s: None).execute(ORDER)
    assert rec["status"] == "REJECTED" and rec["error"] == "insufficient balance"


def test_live_broker_waits_out_the_throttle():
    waits, clock = [], [0.0]
    detail = {"Status": "FILLED"}
    broker = LiveBroker(FakePrivateClient({"OrderDetail": detail}), min_seconds_between_orders=60,
                        sleep=waits.append, monotonic=lambda: clock[0])
    broker.execute(ORDER)
    clock[0] = 10.0
    broker.execute(ORDER)
    assert waits and waits[0] == pytest.approx(50.5)


# -- kline feed -------------------------------------------------------------------------------

class FakeKlineSession:
    """Serves hourly klines up to `now` (the last one still open)."""

    def __init__(self, now, total=2500):
        end = pd.Timestamp(now).floor("1h")
        self.opens = pd.date_range(end=end, periods=total, freq="1h", tz="UTC")
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        opens = self.opens
        if "endTime" in params:
            opens = opens[opens <= pd.Timestamp(params["endTime"], unit="ms", tz="UTC")]
        opens = opens[-params["limit"]:]
        rows = [[int(t.timestamp() * 1000), "1", "1", "1", str(100 + i), "0"] for i, t in enumerate(opens)]
        resp = type("R", (), {})()
        resp.raise_for_status = lambda: None
        resp.json = lambda: rows
        return resp


def test_feed_pages_backwards_and_drops_open_bar():
    now = pd.Timestamp("2026-10-04 10:30", tz="UTC")
    feed = BinanceKlineFeed("1h", session=FakeKlineSession(now))
    closes = feed.closes("BTC/USD", 1500, now)
    assert len(closes) == 1500
    assert closes.index[-1] == pd.Timestamp("2026-10-04 10:00", tz="UTC")  # 09:00-10:00 bar, closed
    assert closes.index.is_monotonic_increasing and not closes.index.has_duplicates
    assert feed.session.calls == 2


# -- bot loop -----------------------------------------------------------------------------------

class FakePublic:
    def __init__(self, prices):
        self.prices = prices

    def get_ticker(self, pair=None):
        return {"ServerTime": int(T0.timestamp() * 1000),
                "Data": {p: {"LastPrice": px, "MaxBid": px, "MinAsk": px} for p, px in self.prices.items()}}


class FakeFeed:
    """Uptrending closes ending at the bar before `now`."""

    def close_panel(self, pairs, bars, now):
        idx = pd.date_range(end=pd.Timestamp(now).floor("1h"), periods=bars, freq="1h", tz="UTC")
        rng = np.random.default_rng(0)
        return pd.DataFrame({p: 100 * np.cumprod(1 + 0.0005 + rng.normal(0, 0.004, bars)) for p in pairs}, index=idx)


class Clock:
    def __init__(self, t):
        self.t = pd.Timestamp(t)

    def __call__(self):
        return self.t.to_pydatetime()


def make_bot(tmp_path, clock, **cfg):
    store = BotStore(":memory:")
    config = BotConfig(strategy_params={"assets": ("BTC/USD", "ETH/USD"), "trend_span": 96, "vol_lookback": 72,
                                        "target_vol": 0.5, "band": 0.02, "min_exposure": 0.15},
                       history_bars=400, stop_file=tmp_path / "STOP", **cfg)
    bot = TradingBot(config, FakePublic({"BTC/USD": 100_000.0, "ETH/USD": 4_000.0}), PaperBroker(store),
                     FakeFeed(), RULES, store, clock=clock)
    return bot, store


def test_bot_decides_once_per_bar_and_trades(tmp_path):
    clock = Clock("2026-10-04 10:05")
    bot, store = make_bot(tmp_path, clock)
    first = bot.tick()
    assert first["action"] == "decided" and len(first["orders"]) == 2
    assert all(o["status"] == "SIMULATED" for o in first["orders"])
    clock.t = pd.Timestamp("2026-10-04 10:40")
    assert bot.tick()["action"] == "idle"  # same bar
    clock.t = pd.Timestamp("2026-10-04 11:03")
    assert bot.tick()["action"] == "decided"
    assert store.count("decisions") == 2 and store.count("orders") == 2


def test_bot_targets_match_the_backtested_strategy(tmp_path):
    clock = Clock("2026-10-04 10:05")
    bot, _ = make_bot(tmp_path, clock)
    targets, _ = bot.target_weights(pd.Timestamp(clock()))
    expected = signals.trend_vol_target(FakeFeed().close_panel(["BTC/USD", "ETH/USD"], 400, clock()),
                                        **bot.config.strategy_params).iloc[-1]
    assert targets == pytest.approx(expected.to_dict())


def test_kill_switch_logs_decision_but_sends_nothing(tmp_path):
    (tmp_path / "STOP").touch()
    bot, store = make_bot(tmp_path, Clock("2026-10-04 10:05"))
    result = bot.tick()
    assert result["killed"] and result["orders"] == []
    assert store.count("orders") == 0 and store.count("decisions") == 1


def test_scheduled_rebalance_once_per_day_at_midnight(tmp_path):
    clock = Clock("2026-10-04 00:04")
    bot, store = make_bot(tmp_path, clock)
    assert bot.tick()["scheduled"] is True
    assert store.get("last_scheduled_date") == "2026-10-04"
    clock.t = pd.Timestamp("2026-10-04 01:04")
    assert bot.tick()["scheduled"] is False


def test_activity_fallback_when_no_trade_by_noon(tmp_path):
    clock = Clock("2026-10-04 12:05")
    bot, store = make_bot(tmp_path, clock, activity_fallback_hour_utc=12, rebalance_hours_utc=())
    result = bot.tick()
    assert result["fallback"] is True
    clock.t = pd.Timestamp("2026-10-04 13:05")
    assert bot.tick()["fallback"] is False  # at most once a day


def test_stale_signal_data_blocks_trading(tmp_path):
    class StaleFeed(FakeFeed):
        def close_panel(self, pairs, bars, now):
            return super().close_panel(pairs, bars, pd.Timestamp(now) - pd.Timedelta(hours=6))

    clock = Clock("2026-10-04 10:05")
    bot, store = make_bot(tmp_path, clock)
    bot.feed = StaleFeed()
    assert bot.tick()["action"] == "stale"
    assert store.count("orders") == 0


def test_bot_refuses_untradable_asset(tmp_path):
    store = BotStore(":memory:")
    with pytest.raises(ValueError):
        TradingBot(BotConfig(assets=["NOPE/USD"]), FakePublic({}), PaperBroker(store), FakeFeed(), RULES, store)


def test_request_audit_hook_records_every_call():
    from tests.test_client_safety import FakeResponse, FakeSession, make_client
    store = BotStore(":memory:")
    client = make_client(FakeSession([FakeResponse(body={"Success": False, "ErrMsg": "bad"})]))
    client._on_request = store.record_api_call
    with pytest.raises(RoostooAPIError):
        client.get_balance()
    assert store.count("api_calls") >= 1


def test_fully_invested_scheduled_rebalance_still_trades():
    """Regression, from the paper run of 2026-09-27: fully invested with the
    0.5% cash buffer, the 00:00 UTC rebalance planned no orders because both
    holdings sat just below their (unscaled) targets and every buy was
    blocked by the buffer. Wallet and targets are the ones from that run."""
    tickers = {"BTC/USD": ticker("BTC/USD", 84_560.0), "ETH/USD": ticker("ETH/USD", 2_695.0)}
    acct = AccountSnapshot(cash_free=500.14, cash_locked=0.0, free={"BTC": 0.6761, "ETH": 15.7877})
    targets = {"BTC/USD": 0.573, "ETH/USD": 0.427}
    orders, info = plan_orders(targets, acct, tickers, RULES, rebalance_threshold=0.05, scheduled=True,
                               fee_rate=0.001, cash_buffer=0.005)
    assert info["target_scale"] == pytest.approx(0.995)
    assert {o.side for o in orders} == {"BUY", "SELL"}  # moves weight from the overweight asset to the other
    assert all(o.notional >= 1.0 for o in orders)  # above Roostoo's MiniOrder
    # cash after the trades stays at (about) the buffer: never overspent
    proceeds = sum(o.notional * (1 - 0.001) for o in orders if o.side == "SELL")
    spend = sum(o.notional * (1 + 0.001) for o in orders if o.side == "BUY")
    equity = acct.equity(tickers)
    assert acct.cash_free + proceeds - spend >= 0.005 * equity - 1.0
    # the same wallet outside the scheduled hour stays inside the band: no trades
    assert plan_orders(targets, acct, tickers, RULES, rebalance_threshold=0.05, cash_buffer=0.005)[0] == []


def test_targets_below_investable_are_not_scaled():
    acct = AccountSnapshot(cash_free=100_000, cash_locked=0)
    _, info = plan_orders({"BTC/USD": 0.3, "ETH/USD": 0.3}, acct, TICKERS, RULES, cash_buffer=0.005)
    assert "target_scale" not in info


def test_bot_trades_on_its_own_over_several_fully_invested_days(tmp_path):
    """End to end: from cash, with scheduled rebalances only, the bot must
    fill at least one order every UTC day while fully invested."""
    clock = Clock("2026-10-04 00:04")
    bot, store = make_bot(tmp_path, clock, activity_fallback_hour_utc=None)
    # Realistically small daily moves (~0.3%): too small to push either
    # holding above its unscaled target, which is exactly when the old
    # planner stopped trading.
    prices = [(100_000.0, 4_000.0), (100_300.0, 3_990.0), (100_050.0, 4_012.0), (100_350.0, 4_001.0)]
    for day, (btc, eth) in enumerate(prices):
        bot.public = FakePublic({"BTC/USD": btc, "ETH/USD": eth})
        clock.t = pd.Timestamp("2026-10-04 00:04") + pd.Timedelta(days=day)
        bot.tick()
    assert len(store.filled_order_days()) == len(prices)
