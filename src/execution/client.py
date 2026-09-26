"""Roostoo REST API client.

Split into `PublicMarketDataClient` (no credentials needed) and
`PrivateTradingClient` (signed, requires API key/secret) per docs/API_NOTES.md.
Strategy code must never import this module directly — it should go through
`src.data.roostoo_data` so signal generation stays decoupled from the
transport layer.

Every endpoint here is a direct 1:1 mapping to something documented in
docs/API_NOTES.md. Nothing is implemented speculatively.
"""
from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import requests

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------

class RoostooError(Exception):
    """Base for all Roostoo client errors."""


class RoostooNetworkError(RoostooError):
    """Transport-level failure (timeout, connection refused, DNS, ...)."""


class RoostooHTTPError(RoostooError):
    """Non-2xx HTTP status that persisted through retries."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"HTTP {status_code}: {body[:500]}")
        self.status_code = status_code
        self.body = body


class RoostooAPIError(RoostooError):
    """HTTP 200 but the API's own `Success` field is false.

    Roostoo returns HTTP 200 for logical failures too (e.g. "no order
    matched", "insufficient balance") — see docs/API_NOTES.md. Callers that
    need to distinguish a benign "empty" result from a real error should
    catch this and inspect `err_msg`.
    """

    def __init__(self, err_msg: str, response: dict[str, Any]):
        super().__init__(err_msg)
        self.err_msg = err_msg
        self.response = response


class RoostooSafetyError(RoostooError):
    """Raised when a call is blocked by a client-side safety guard."""


class RoostooOrderStateUnknownError(RoostooNetworkError):
    """An order-creating request failed in a way that leaves its outcome
    unknown (e.g. timeout after the request may have reached the server).

    Order-creating calls are never retried automatically — a blind retry
    could place the same order twice. The caller must reconcile via
    `query_order` / `get_balance` before trying again.
    """


# Messages documented as normal "nothing found" outcomes, not real errors.
_BENIGN_EMPTY_MESSAGES = {
    "no pending order under this account",
    "no order matched",
}


# --------------------------------------------------------------------------
# Signing
# --------------------------------------------------------------------------

def build_total_params(payload: dict[str, Any]) -> str:
    """sortParamsByKey, join with '=' then '&', per docs/API_NOTES.md."""
    return "&".join(f"{k}={payload[k]}" for k in sorted(payload.keys()))


def sign(secret: str, total_params: str) -> str:
    return hmac.new(
        secret.encode("utf-8"), total_params.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def format_decimal(value: float | int | str) -> str:
    """Render a number as a plain decimal string for the API.

    `str(float)` is not safe here: `str(0.00001)` is `'1e-05'`, which the
    exchange won't parse as a quantity. Goes through the shortest round-trip
    repr so no binary float noise is introduced.
    """
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"cannot send non-finite number {value!r} to the API")
    d = Decimal(repr(value)) if isinstance(value, float) else Decimal(str(value))
    text = format(d, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text not in {"", "-0"} else "0"


def _require_positive(name: str, value: float) -> None:
    if not (isinstance(value, (int, float)) and math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")


# --------------------------------------------------------------------------
# Shared HTTP plumbing (retry + backoff + structured logging)
# --------------------------------------------------------------------------

_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass
class _RetryPolicy:
    max_retries: int = 4
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 20.0

    def delay(self, attempt: int) -> float:
        return min(self.backoff_base_seconds * (2 ** attempt), self.backoff_max_seconds)


class _BaseClient:
    def __init__(
        self,
        base_url: str = "https://mock-api.roostoo.com",
        timeout_seconds: float = 10.0,
        retry_policy: _RetryPolicy | None = None,
        session: requests.Session | None = None,
        clock_skew_tolerance_ms: int = 60_000,
        clock_resync_seconds: float = 300.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or _RetryPolicy()
        self.session = session or requests.Session()
        self._clock_skew_tolerance_ms = clock_skew_tolerance_ms
        self._clock_resync_seconds = clock_resync_seconds
        self._clock_offset_ms = 0
        self._last_clock_sync: float | None = None

    # -- clock sync ------------------------------------------------------------
    # The server rejects any request whose timestamp is more than
    # clock_skew_tolerance_ms from its own clock (docs/API_NOTES.md), so
    # timestamps are taken from local time corrected by the measured offset
    # to /v3/serverTime, re-measured every clock_resync_seconds.

    def sync_clock(self) -> int:
        """Measure and store (server - local) clock offset in ms."""
        sent_ms = time.time() * 1000
        server_ms = int(self._request("GET", "/v3/serverTime")["ServerTime"])
        received_ms = time.time() * 1000
        offset = int(round(server_ms - (sent_ms + received_ms) / 2))
        self._clock_offset_ms = offset
        self._last_clock_sync = time.monotonic()
        if abs(offset) > self._clock_skew_tolerance_ms / 2:
            logger.warning(
                "local clock drift vs roostoo server is large; timestamps are "
                "being corrected, but the host clock should be fixed",
                extra={"offset_ms": offset, "tolerance_ms": self._clock_skew_tolerance_ms},
            )
        return offset

    def _timestamp_ms(self) -> str:
        if (
            self._last_clock_sync is None
            or time.monotonic() - self._last_clock_sync > self._clock_resync_seconds
        ):
            self.sync_clock()
        return str(int(time.time() * 1000) + self._clock_offset_ms)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: str | None = None,
        headers: dict[str, str] | None = None,
        retry: bool = True,
    ) -> dict[str, Any]:
        """`retry=False` is for non-idempotent calls (anything that creates
        an order or position): no automatic retry, and a transport failure
        raises RoostooOrderStateUnknownError, since the request may have
        been executed server-side even though we never saw the response."""
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        max_retries = self.retry_policy.max_retries if retry else 0

        for attempt in range(max_retries + 1):
            try:
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    data=data,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
            except requests.exceptions.RequestException as exc:
                last_exc = exc
                logger.warning(
                    "roostoo request network error",
                    extra={"endpoint": path, "attempt": attempt, "error": str(exc)},
                )
                if not retry:
                    raise RoostooOrderStateUnknownError(
                        f"{path} failed in transit ({exc}); the order may or may "
                        f"not have been placed — reconcile before retrying"
                    ) from exc
                if attempt < max_retries:
                    time.sleep(self.retry_policy.delay(attempt))
                    continue
                raise RoostooNetworkError(str(exc)) from exc

            if resp.status_code in _RETRYABLE_STATUS and attempt < max_retries:
                logger.warning(
                    "roostoo request retryable HTTP status",
                    extra={"endpoint": path, "attempt": attempt, "status": resp.status_code},
                )
                time.sleep(self.retry_policy.delay(attempt))
                continue

            if not retry and resp.status_code >= 500:
                raise RoostooOrderStateUnknownError(
                    f"{path} returned HTTP {resp.status_code}; the order may or "
                    f"may not have been placed — reconcile before retrying"
                )

            if not resp.ok:
                logger.error(
                    "roostoo request failed",
                    extra={"endpoint": path, "status": resp.status_code, "body": resp.text[:500]},
                )
                raise RoostooHTTPError(resp.status_code, resp.text)

            try:
                body = resp.json()
            except ValueError as exc:
                raise RoostooError(f"non-JSON response from {path}: {resp.text[:200]}") from exc

            if isinstance(body, dict) and body.get("Success") is False:
                err_msg = body.get("ErrMsg", "")
                if err_msg not in _BENIGN_EMPTY_MESSAGES:
                    logger.info(
                        "roostoo API returned Success=false",
                        extra={"endpoint": path, "err_msg": err_msg},
                    )
                raise RoostooAPIError(err_msg, body)

            return body

        # Unreachable, but keeps type-checkers happy.
        raise RoostooNetworkError(str(last_exc))


# --------------------------------------------------------------------------
# Public (unauthenticated / RCL_TSCheck) endpoints
# --------------------------------------------------------------------------

class PublicMarketDataClient(_BaseClient):
    """No API key required. Covers RCL_NoVerification and RCL_TSCheck GETs."""

    def get_server_time(self) -> int:
        return self._request("GET", "/v3/serverTime")["ServerTime"]

    def get_exchange_info(self) -> dict[str, Any]:
        return self._request("GET", "/v3/exchangeInfo")

    def get_ticker(self, pair: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"timestamp": self._timestamp_ms()}
        if pair is not None:
            params["pair"] = pair
        return self._request("GET", "/v3/ticker", params=params)


# --------------------------------------------------------------------------
# Private (RCL_TopLevelCheck / SIGNED) endpoints
# --------------------------------------------------------------------------

class PrivateTradingClient(_BaseClient):
    """Signed endpoints. Read-only calls (balance, pending count, order
    query, short positions) always work. Every call that changes account
    state is refused unless `live_trading_enabled=True` was passed — build
    the client via `build_clients_from_settings` so that flag comes from the
    APP_ENV + LIVE_TRADING double gate in src/config/settings.py, rather
    than being hand-set."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        live_trading_enabled: bool = False,
        allow_shorting: bool = False,
        allow_cancel_all: bool = False,
        min_seconds_between_orders: float = 60.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not api_key or not api_secret:
            raise RoostooSafetyError("api_key and api_secret are both required")
        self._api_key = api_key
        self._api_secret = api_secret
        self._live_trading_enabled = live_trading_enabled is True
        self._allow_shorting = allow_shorting is True
        self._allow_cancel_all = allow_cancel_all is True
        self._min_seconds_between_orders = min_seconds_between_orders
        self._last_order_ts: float | None = None

    @property
    def live_trading_enabled(self) -> bool:
        return self._live_trading_enabled

    def _require_live_trading(self, action: str) -> None:
        if not self._live_trading_enabled:
            raise RoostooSafetyError(
                f"Refusing to {action}: live trading is not enabled on this "
                f"client (requires APP_ENV=live and LIVE_TRADING=true)."
            )

    def _signed_headers_and_body(self, payload: dict[str, Any]) -> tuple[dict[str, str], str]:
        payload = dict(payload)
        payload["timestamp"] = self._timestamp_ms()
        total_params = build_total_params(payload)
        signature = sign(self._api_secret, total_params)
        headers = {"RST-API-KEY": self._api_key, "MSG-SIGNATURE": signature}
        return headers, total_params

    def _signed_get(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers, total_params = self._signed_headers_and_body(payload or {})
        # For GET, totalParams IS the query string — append it verbatim
        # rather than letting requests re-encode a dict, to guarantee
        # byte-for-byte match with what was signed.
        path_with_query = f"{path}?{total_params}" if total_params else path
        return self._request("GET", path_with_query, headers=headers)

    def _signed_post(
        self, path: str, payload: dict[str, Any] | None = None, *, retry: bool = True
    ) -> dict[str, Any]:
        headers, total_params = self._signed_headers_and_body(payload or {})
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._request("POST", path, data=total_params, headers=headers, retry=retry)

    def _enforce_order_rate_limit(self) -> None:
        if self._last_order_ts is None:
            return
        elapsed = time.monotonic() - self._last_order_ts
        if elapsed < self._min_seconds_between_orders:
            raise RoostooSafetyError(
                f"Order rate limit: only {elapsed:.1f}s since last order, "
                f"minimum is {self._min_seconds_between_orders}s (competition "
                f"prohibits HFT)."
            )

    # -- account -----------------------------------------------------------

    def get_balance(self) -> dict[str, Any]:
        return self._signed_get("/v3/balance")

    def get_pending_count(self) -> dict[str, Any]:
        try:
            return self._signed_get("/v3/pending_count")
        except RoostooAPIError as exc:
            if exc.err_msg in _BENIGN_EMPTY_MESSAGES:
                return {"Success": False, "ErrMsg": exc.err_msg, "TotalPending": 0, "OrderPairs": {}}
            raise

    # -- orders --------------------------------------------------------------

    def place_order(
        self,
        pair: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None = None,
    ) -> dict[str, Any]:
        """Place a spot order. `quantity`/`price` should already be rounded
        to the pair's precision (src/data/universe.py::TradingRule). Never
        retried automatically — see RoostooOrderStateUnknownError."""
        side = side.upper()
        order_type = order_type.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if order_type not in {"LIMIT", "MARKET"}:
            raise ValueError(f"type must be LIMIT or MARKET, got {order_type!r}")
        if order_type == "LIMIT" and price is None:
            raise ValueError("price is required for LIMIT orders")
        _require_positive("quantity", quantity)
        if price is not None:
            _require_positive("price", price)

        self._require_live_trading("place an order")
        self._enforce_order_rate_limit()

        payload: dict[str, Any] = {
            "pair": pair,
            "side": side,
            "type": order_type,
            "quantity": format_decimal(quantity),
        }
        if price is not None:
            payload["price"] = format_decimal(price)

        try:
            result = self._signed_post("/v3/place_order", payload, retry=False)
        finally:
            # Even a rejected/errored attempt still consumed a "slot" against
            # the exchange from a rate-limiting perspective, so we throttle
            # our own next attempt regardless of outcome.
            self._last_order_ts = time.monotonic()
        return result

    def query_order(
        self,
        *,
        order_id: str | int | None = None,
        pair: str | None = None,
        offset: int | None = None,
        limit: int | None = None,
        pending_only: bool | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair is not None:
                payload["pair"] = pair
            if offset is not None:
                payload["offset"] = str(offset)
            if limit is not None:
                payload["limit"] = str(limit)
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"

        try:
            return self._signed_post("/v3/query_order", payload)
        except RoostooAPIError as exc:
            if exc.err_msg in _BENIGN_EMPTY_MESSAGES:
                return {"Success": False, "ErrMsg": exc.err_msg, "OrderMatched": []}
            raise

    def cancel_order(
        self,
        *,
        order_id: str | int | None = None,
        pair: str | None = None,
        allow_cancel_all: bool = False,
    ) -> dict[str, Any]:
        """Cancelling with neither `order_id` nor `pair` cancels ALL pending
        orders on the account; that needs both the per-call
        `allow_cancel_all=True` and the client-level config opt-in
        (execution.allow_cancel_all_without_filter)."""
        if order_id is None and pair is None:
            if not (allow_cancel_all and self._allow_cancel_all):
                raise RoostooSafetyError(
                    "cancel_order called with neither order_id nor pair — this "
                    "cancels ALL pending orders on the account. It requires "
                    "allow_cancel_all=True on the call AND "
                    "execution.allow_cancel_all_without_filter: true in config."
                )
        self._require_live_trading("cancel orders")
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair is not None:
            payload["pair"] = pair
        return self._signed_post("/v3/cancel_order", payload)

    # -- shorting (see docs/API_NOTES.md open question #1 before using) ----

    def _require_shorting(self) -> None:
        if not self._allow_shorting:
            raise RoostooSafetyError(
                "Shorting is disabled (execution.allow_shorting: false). See "
                "docs/API_NOTES.md open question #1."
            )

    def short_open(self, pair: str, collateral: float, price: float | None = None) -> dict[str, Any]:
        _require_positive("collateral", collateral)
        if price is not None:
            _require_positive("price", price)
        self._require_shorting()
        self._require_live_trading("open a short")
        self._enforce_order_rate_limit()
        payload: dict[str, Any] = {"pair": pair, "collateral": format_decimal(collateral)}
        if price is not None:
            payload["order_type"] = "LIMIT"
            payload["price"] = format_decimal(price)
        try:
            return self._signed_post("/v6/short_open", payload, retry=False)
        finally:
            self._last_order_ts = time.monotonic()

    def short_close(
        self, pair: str, *, close_qty: float | None = None, close_pct: float | None = None
    ) -> dict[str, Any]:
        if close_qty is not None:
            _require_positive("close_qty", close_qty)
        elif close_pct is not None:
            _require_positive("close_pct", close_pct)
        self._require_shorting()
        self._require_live_trading("close a short")
        self._enforce_order_rate_limit()
        payload: dict[str, Any] = {"pair": pair}
        if close_qty is not None:
            payload["close_qty"] = format_decimal(close_qty)
        elif close_pct is not None:
            payload["close_pct"] = format_decimal(close_pct)
        try:
            # A partial close is not idempotent, so no automatic retry.
            return self._signed_post("/v6/short_close", payload, retry=False)
        finally:
            self._last_order_ts = time.monotonic()

    def get_short_positions(self) -> dict[str, Any]:
        return self._signed_get("/v6/short_positions")


def build_clients_from_settings(settings) -> tuple[PublicMarketDataClient, PrivateTradingClient | None]:
    """Convenience factory. Private client is None if no credentials are set,
    so read-only research/paper flows work without ever touching secrets.

    This is where the settings-level safety gates are applied: the private
    client can only change account state if `settings.is_live_trading_enabled`
    (APP_ENV=live AND LIVE_TRADING=true), and shorting / cancel-all follow
    config/config.yaml."""
    common = dict(
        base_url=settings.roostoo.base_url,
        timeout_seconds=settings.roostoo.request_timeout_seconds,
        clock_skew_tolerance_ms=settings.roostoo.clock_skew_tolerance_ms,
    )

    def retry_policy() -> _RetryPolicy:
        return _RetryPolicy(
            max_retries=settings.roostoo.max_retries,
            backoff_base_seconds=settings.roostoo.backoff_base_seconds,
            backoff_max_seconds=settings.roostoo.backoff_max_seconds,
        )

    public = PublicMarketDataClient(retry_policy=retry_policy(), **common)
    private = None
    if settings.api_key and settings.api_secret:
        private = PrivateTradingClient(
            settings.api_key,
            settings.api_secret,
            live_trading_enabled=settings.is_live_trading_enabled,
            allow_shorting=settings.execution.allow_shorting,
            allow_cancel_all=settings.execution.allow_cancel_all_without_filter,
            min_seconds_between_orders=settings.execution.min_seconds_between_orders,
            retry_policy=retry_policy(),
            **common,
        )
    return public, private
