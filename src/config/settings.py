"""Configuration layer: env vars (secrets) + config.yaml (everything else).

The one job this module must never fail at: refusing live trading unless it
was explicitly, unambiguously requested. See `Settings.assert_live_trading_allowed`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]


class LiveTradingNotEnabledError(RuntimeError):
    """Raised when code tries to trade live without explicit opt-in."""


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class RoostooConfig:
    base_url: str
    request_timeout_seconds: float
    max_retries: int
    backoff_base_seconds: float
    backoff_max_seconds: float
    clock_skew_tolerance_ms: int


@dataclass(frozen=True)
class ExecutionConfig:
    min_seconds_between_orders: float
    allow_cancel_all_without_filter: bool
    allow_shorting: bool


@dataclass(frozen=True)
class CostsConfig:
    taker_fee: float
    maker_fee: float
    slippage_bps: float
    short_open_fee: float
    short_close_fee: float


@dataclass(frozen=True)
class Settings:
    app_env: str
    live_trading_flag: bool
    api_key: str | None
    api_secret: str | None
    roostoo: RoostooConfig
    execution: ExecutionConfig
    costs: CostsConfig
    raw: dict[str, Any] = field(repr=False)

    @property
    def is_live_trading_enabled(self) -> bool:
        """Both the environment AND the explicit flag must agree.

        This double gate exists so that a stray LIVE_TRADING=true left in a
        shared .env doesn't cause live orders when someone is just running
        `APP_ENV=development` locally, and conversely so that flipping
        APP_ENV=live alone (e.g. a typo) doesn't start trading without the
        separate explicit flag also being set.
        """
        return self.app_env == "live" and self.live_trading_flag is True

    def assert_live_trading_allowed(self) -> None:
        if not self.is_live_trading_enabled:
            raise LiveTradingNotEnabledError(
                f"Refusing to place a live order: APP_ENV={self.app_env!r}, "
                f"LIVE_TRADING={self.live_trading_flag!r}. Both must be "
                f"'live' and true respectively."
            )

    def assert_credentials_present(self) -> None:
        if not self.api_key or not self.api_secret:
            raise LiveTradingNotEnabledError(
                "ROOSTOO_API_KEY / ROOSTOO_API_SECRET are not set. Copy "
                ".env.example to .env and fill them in."
            )


def load_settings(config_path: Path | None = None) -> Settings:
    config_path = config_path or (REPO_ROOT / "config" / "config.yaml")
    raw = _load_yaml(config_path)

    app_env = os.getenv("APP_ENV", "development").strip().lower()
    live_flag_str = os.getenv("LIVE_TRADING", "false").strip().lower()
    live_flag = live_flag_str == "true"

    roostoo_raw = raw.get("roostoo", {})
    execution_raw = raw.get("execution", {})
    costs_raw = raw.get("costs", {})

    return Settings(
        app_env=app_env,
        live_trading_flag=live_flag,
        api_key=os.getenv("ROOSTOO_API_KEY") or None,
        api_secret=os.getenv("ROOSTOO_API_SECRET") or None,
        roostoo=RoostooConfig(
            base_url=roostoo_raw.get("base_url", "https://mock-api.roostoo.com"),
            request_timeout_seconds=float(roostoo_raw.get("request_timeout_seconds", 10)),
            max_retries=int(roostoo_raw.get("max_retries", 4)),
            backoff_base_seconds=float(roostoo_raw.get("backoff_base_seconds", 0.5)),
            backoff_max_seconds=float(roostoo_raw.get("backoff_max_seconds", 20)),
            clock_skew_tolerance_ms=int(roostoo_raw.get("clock_skew_tolerance_ms", 60000)),
        ),
        execution=ExecutionConfig(
            min_seconds_between_orders=float(execution_raw.get("min_seconds_between_orders", 60)),
            allow_cancel_all_without_filter=bool(execution_raw.get("allow_cancel_all_without_filter", False)),
            allow_shorting=bool(execution_raw.get("allow_shorting", False)),
        ),
        costs=CostsConfig(
            taker_fee=float(costs_raw.get("taker_fee", 0.0010)),
            maker_fee=float(costs_raw.get("maker_fee", 0.0005)),
            slippage_bps=float(costs_raw.get("slippage_bps", 5)),
            short_open_fee=float(costs_raw.get("short_open_fee", 0.0010)),
            short_close_fee=float(costs_raw.get("short_close_fee", 0.0010)),
        ),
        raw=raw,
    )
