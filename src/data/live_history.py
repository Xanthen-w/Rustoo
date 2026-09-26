"""Recent hourly history for the live bot, from Binance's public REST API.

The live strategy needs weeks of hourly closes (a 40-day EMA), and Roostoo
has no history endpoint, so the bot computes signals on the same data the
strategy was backtested on: Binance spot klines, indexed by UTC close time
(src/data/binance.py). Only *closed* bars are returned — the kline still
forming is dropped, so a signal never uses a partial bar. Execution prices
come from Roostoo's own ticker, not from here.
"""
from __future__ import annotations

import logging
import time

import pandas as pd
import requests

from src.data.binance import interval_offset, roostoo_to_binance_symbol

logger = logging.getLogger(__name__)

KLINES_URL = "https://api.binance.com/api/v3/klines"
MAX_LIMIT = 1000


class BinanceKlineFeed:
    def __init__(self, interval: str = "1h", session: requests.Session | None = None,
                 timeout_seconds: float = 15.0, max_retries: int = 3, url: str = KLINES_URL):
        self.interval = interval
        self.bar = interval_offset(interval)
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.url = url

    def _get(self, params: dict) -> list:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(self.url, params=params, timeout=self.timeout_seconds)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == self.max_retries:
                    raise
                logger.warning("binance klines request failed, retrying", extra={"error": str(exc), "attempt": attempt})
                time.sleep(2 ** attempt)
        return []

    def closes(self, pair: str, bars: int, now: pd.Timestamp | None = None) -> pd.Series:
        """The last `bars` closed bars' close prices for a Roostoo pair,
        indexed by UTC close time."""
        now = now or pd.Timestamp.now(tz="UTC")
        symbol = roostoo_to_binance_symbol(pair)
        rows: list = []
        end_ms = None
        remaining = bars + 1  # +1: the newest kline may still be open
        while remaining > 0:
            params = {"symbol": symbol, "interval": self.interval, "limit": min(MAX_LIMIT, remaining)}
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get(params)
            if not batch:
                break
            rows = batch + rows
            remaining -= len(batch)
            end_ms = int(batch[0][0]) - 1  # page backwards from the oldest open time
            if len(batch) < params["limit"]:
                break
        if not rows:
            raise RuntimeError(f"no klines returned for {symbol}")

        open_time = pd.to_datetime([int(r[0]) for r in rows], unit="ms", utc=True)
        close_time = open_time + self.bar
        series = pd.Series([float(r[4]) for r in rows], index=close_time, name=pair)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        series = series[series.index <= now]  # drop the still-forming bar
        return series.iloc[-bars:]

    def close_panel(self, pairs: list[str], bars: int, now: pd.Timestamp | None = None) -> pd.DataFrame:
        now = now or pd.Timestamp.now(tz="UTC")
        return pd.DataFrame({pair: self.closes(pair, bars, now) for pair in pairs}).sort_index()
