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
import time
from dataclasses import dataclass
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


def current_timestamp_ms() -> str:
    return str(int(time.time() * 1000))


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
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_policy = retry_policy or _RetryPolicy()
        self.session = session or requests.Session()

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        data: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None

        for attempt in range(self.retry_policy.max_retries + 1):
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
                if attempt < self.retry_policy.max_retries:
                    time.sleep(self.retry_policy.delay(attempt))
                    continue
                raise RoostooNetworkError(str(exc)) from exc

            if resp.status_code in _RETRYABLE_STATUS and attempt < self.retry_policy.max_retries:
                logger.warning(
                    "roostoo request retryable HTTP status",
                    extra={"endpoint": path, "attempt": attempt, "status": resp.status_code},
                )
                time.sleep(self.retry_policy.delay(attempt))
                continue

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
        params: dict[str, Any] = {"timestamp": current_timestamp_ms()}
        if pair is not None:
            params["pair"] = pair
        return self._request("GET", "/v3/ticker", params=params)


# --------------------------------------------------------------------------
# Private (RCL_TopLevelCheck / SIGNED) endpoints
# --------------------------------------------------------------------------

class PrivateTradingClient(_BaseClient):
    def __init__(self, api_key: str, api_secret: str, *, min_seconds_between_orders: float = 60.0, **kwargs):
        super().__init__(**kwargs)
        if not api_key or not api_secret:
            raise RoostooSafetyError("api_key and api_secret are both required")
        self._api_key = api_key
        self._api_secret = api_secret
        self._min_seconds_between_orders = min_seconds_between_orders
        self._last_order_ts: float = 0.0

    def _signed_headers_and_body(self, payload: dict[str, Any]) -> tuple[dict[str, str], str]:
        payload = dict(payload)
        payload["timestamp"] = current_timestamp_ms()
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

    def _signed_post(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        headers, total_params = self._signed_headers_and_body(payload or {})
        headers["Content-Type"] = "application/x-www-form-urlencoded"
        return self._request("POST", path, data=total_params, headers=headers)

    def _enforce_order_rate_limit(self) -> None:
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
        side = side.upper()
        order_type = order_type.upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError(f"side must be BUY or SELL, got {side!r}")
        if order_type not in {"LIMIT", "MARKET"}:
            raise ValueError(f"type must be LIMIT or MARKET, got {order_type!r}")
        if order_type == "LIMIT" and price is None:
            raise ValueError("price is required for LIMIT orders")

        self._enforce_order_rate_limit()

        payload: dict[str, Any] = {
            "pair": pair,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
        }
        if price is not None:
            payload["price"] = str(price)

        try:
            result = self._signed_post("/v3/place_order", payload)
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
        if order_id is None and pair is None and not allow_cancel_all:
            raise RoostooSafetyError(
                "cancel_order called with neither order_id nor pair — this "
                "cancels ALL pending orders on the account. Pass "
                "allow_cancel_all=True if that is really intended."
            )
        payload: dict[str, Any] = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair is not None:
            payload["pair"] = pair
        return self._signed_post("/v3/cancel_order", payload)

    # -- shorting (see docs/API_NOTES.md open question #1 before using) ----

    def short_open(self, pair: str, collateral: float, price: float | None = None) -> dict[str, Any]:
        self._enforce_order_rate_limit()
        payload: dict[str, Any] = {"pair": pair, "collateral": str(collateral)}
        if price is not None:
            payload["order_type"] = "LIMIT"
            payload["price"] = str(price)
        try:
            return self._signed_post("/v6/short_open", payload)
        finally:
            self._last_order_ts = time.monotonic()

    def short_close(
        self, pair: str, *, close_qty: float | None = None, close_pct: float | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"pair": pair}
        if close_qty is not None:
            payload["close_qty"] = str(close_qty)
        elif close_pct is not None:
            payload["close_pct"] = str(close_pct)
        return self._signed_post("/v6/short_close", payload)

    def get_short_positions(self) -> dict[str, Any]:
        return self._signed_get("/v6/short_positions")


def build_clients_from_settings(settings) -> tuple[PublicMarketDataClient, PrivateTradingClient | None]:
    """Convenience factory. Private client is None if no credentials are set,
    so read-only research/paper flows work without ever touching secrets."""
    public = PublicMarketDataClient(
        base_url=settings.roostoo.base_url,
        timeout_seconds=settings.roostoo.request_timeout_seconds,
        retry_policy=_RetryPolicy(
            max_retries=settings.roostoo.max_retries,
            backoff_base_seconds=settings.roostoo.backoff_base_seconds,
            backoff_max_seconds=settings.roostoo.backoff_max_seconds,
        ),
    )
    private = None
    if settings.api_key and settings.api_secret:
        private = PrivateTradingClient(
            settings.api_key,
            settings.api_secret,
            base_url=settings.roostoo.base_url,
            timeout_seconds=settings.roostoo.request_timeout_seconds,
            min_seconds_between_orders=settings.execution.min_seconds_between_orders,
            retry_policy=_RetryPolicy(
                max_retries=settings.roostoo.max_retries,
                backoff_base_seconds=settings.roostoo.backoff_base_seconds,
                backoff_max_seconds=settings.roostoo.backoff_max_seconds,
            ),
        )
    return public, private
