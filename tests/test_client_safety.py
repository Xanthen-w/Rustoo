"""Client-side safety guards on the Roostoo client, exercised against a fake
HTTP session — nothing here touches the network."""
from types import SimpleNamespace

import pytest
import requests

from src.config.settings import CostsConfig, ExecutionConfig, RoostooConfig, Settings
from src.execution.client import (
    PrivateTradingClient,
    RoostooOrderStateUnknownError,
    RoostooSafetyError,
    _RetryPolicy,
    build_clients_from_settings,
    format_decimal,
)

SERVER_TIME_MS = 1_700_000_000_000


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body if body is not None else {"Success": True}
        self.text = str(self._body)

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._body


class FakeSession:
    """Answers /v3/serverTime itself; every other request pops the next
    queued outcome (a FakeResponse, or an exception to raise)."""

    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.calls = []

    def request(self, method, url, params=None, data=None, headers=None, timeout=None):
        if url.endswith("/v3/serverTime"):
            return FakeResponse(body={"ServerTime": SERVER_TIME_MS})
        self.calls.append(SimpleNamespace(method=method, url=url, params=params, data=data, headers=headers))
        outcome = self.outcomes.pop(0) if self.outcomes else FakeResponse()
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def make_client(session, **kwargs):
    kwargs.setdefault("live_trading_enabled", True)
    return PrivateTradingClient(
        "key",
        "secret",
        session=session,
        retry_policy=_RetryPolicy(max_retries=3, backoff_base_seconds=0.0),
        **kwargs,
    )


def make_settings(app_env, live_flag, **execution):
    execution_cfg = dict(min_seconds_between_orders=60, allow_cancel_all_without_filter=False, allow_shorting=False)
    execution_cfg.update(execution)
    return Settings(
        app_env=app_env,
        live_trading_flag=live_flag,
        api_key="key",
        api_secret="secret",
        roostoo=RoostooConfig("https://example.invalid", 10, 4, 0.5, 20, 60_000),
        execution=ExecutionConfig(**execution_cfg),
        costs=CostsConfig(0.001, 0.0005, 5, 0.001, 0.001),
        raw={},
    )


# -- live-trading gate -------------------------------------------------------

@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.place_order("BTC/USD", "BUY", "MARKET", 0.01),
        lambda c: c.cancel_order(order_id=1),
    ],
)
def test_state_changing_calls_refused_when_live_trading_disabled(call):
    session = FakeSession()
    client = make_client(session, live_trading_enabled=False)
    with pytest.raises(RoostooSafetyError):
        call(client)
    assert session.calls == []


def test_read_only_calls_allowed_when_live_trading_disabled():
    session = FakeSession([FakeResponse(body={"Success": True, "Wallet": {}})])
    client = make_client(session, live_trading_enabled=False)
    assert client.get_balance()["Success"] is True


@pytest.mark.parametrize(
    "app_env,live_flag,expected",
    [
        ("development", True, False),
        ("paper", True, False),
        ("live", False, False),
        ("live", True, True),
    ],
)
def test_factory_applies_settings_double_gate(app_env, live_flag, expected):
    _, private = build_clients_from_settings(make_settings(app_env, live_flag))
    assert private.live_trading_enabled is expected


def test_factory_gated_client_refuses_orders_in_development():
    _, private = build_clients_from_settings(make_settings("development", True))
    with pytest.raises(RoostooSafetyError):
        private.place_order("BTC/USD", "BUY", "MARKET", 0.01)


# -- shorting / cancel-all config --------------------------------------------

def test_short_open_refused_unless_shorting_allowed():
    session = FakeSession()
    client = make_client(session, allow_shorting=False)
    with pytest.raises(RoostooSafetyError):
        client.short_open("BTC/USD", collateral=100.0)
    with pytest.raises(RoostooSafetyError):
        client.short_close("BTC/USD")
    assert session.calls == []


def test_cancel_all_needs_both_call_flag_and_config():
    session = FakeSession()
    with pytest.raises(RoostooSafetyError):
        make_client(session, allow_cancel_all=False).cancel_order(allow_cancel_all=True)
    with pytest.raises(RoostooSafetyError):
        make_client(session, allow_cancel_all=True).cancel_order()
    assert session.calls == []

    make_client(session, allow_cancel_all=True).cancel_order(allow_cancel_all=True)
    assert len(session.calls) == 1


# -- no retry on order-creating calls ----------------------------------------

def test_place_order_not_retried_on_timeout():
    session = FakeSession([requests.exceptions.ReadTimeout("timed out")])
    client = make_client(session)
    with pytest.raises(RoostooOrderStateUnknownError):
        client.place_order("BTC/USD", "BUY", "MARKET", 0.01)
    assert len(session.calls) == 1


@pytest.mark.parametrize("status", [500, 502, 504])
def test_place_order_not_retried_on_5xx(status):
    session = FakeSession([FakeResponse(status_code=status)])
    client = make_client(session)
    with pytest.raises(RoostooOrderStateUnknownError):
        client.place_order("BTC/USD", "BUY", "MARKET", 0.01)
    assert len(session.calls) == 1


def test_read_only_calls_still_retried():
    session = FakeSession(
        [requests.exceptions.ConnectionError("reset"), FakeResponse(body={"Success": True, "Wallet": {}})]
    )
    client = make_client(session)
    assert client.get_balance()["Success"] is True
    assert len(session.calls) == 2


def test_order_rate_limit_still_enforced():
    session = FakeSession()
    client = make_client(session)
    client.place_order("BTC/USD", "BUY", "MARKET", 0.01)
    with pytest.raises(RoostooSafetyError):
        client.place_order("BTC/USD", "BUY", "MARKET", 0.01)
    assert len(session.calls) == 1


# -- number formatting ---------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [
        (0.00001, "0.00001"),
        (1e-8, "0.00000001"),
        (0.1 + 0.2, "0.30000000000000004"),  # exact shortest repr, no sci notation
        (2000, "2000"),
        (2000.0, "2000"),
        (12345.67, "12345.67"),
        (1.5e7, "15000000"),
    ],
)
def test_format_decimal_never_uses_scientific_notation(value, expected):
    assert format_decimal(value) == expected


def test_place_order_sends_plain_decimal_quantity():
    session = FakeSession()
    make_client(session).place_order("BTC/USD", "BUY", "LIMIT", 0.00001, price=65000.5)
    body = session.calls[0].data
    assert "quantity=0.00001" in body
    assert "price=65000.5" in body
    assert "e-" not in body


@pytest.mark.parametrize("qty", [0, -1.0, float("nan"), float("inf")])
def test_place_order_rejects_invalid_quantity(qty):
    session = FakeSession()
    with pytest.raises(ValueError):
        make_client(session).place_order("BTC/USD", "BUY", "MARKET", qty)
    assert session.calls == []


# -- clock sync ----------------------------------------------------------------

def test_signed_timestamp_uses_server_clock_offset():
    session = FakeSession([FakeResponse(body={"Success": True, "Wallet": {}})])
    client = make_client(session)
    client.get_balance()
    query = session.calls[0].url.split("?", 1)[1]
    timestamp = int(dict(kv.split("=") for kv in query.split("&"))["timestamp"])
    # The fake server clock is fixed in 2023, far from local time; the signed
    # timestamp must track the server, not the local wall clock.
    assert abs(timestamp - SERVER_TIME_MS) < 5_000
