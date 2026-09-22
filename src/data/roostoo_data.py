"""Live market-data feed: wraps PublicMarketDataClient and normalizes its
output into src.data.market_data types. Also builds our own OHLCV bars from
repeated ticker polling, since Roostoo has no historical endpoint at all
(see docs/API_NOTES.md) — this is the only way the live bot can ever
accumulate its own history going forward.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from src.data.market_data import Ticker, tickers_from_response
from src.execution.client import PublicMarketDataClient

logger = logging.getLogger(__name__)


class MarketDataFeed:
    """Normalized read-only view over the public Roostoo API."""

    def __init__(self, client: PublicMarketDataClient):
        self._client = client

    def get_ticker(self, pair: str) -> Ticker:
        response = self._client.get_ticker(pair)
        tickers = tickers_from_response(response)
        if pair not in tickers:
            raise KeyError(f"{pair} not present in ticker response")
        return tickers[pair]

    def get_all_tickers(self) -> dict[str, Ticker]:
        response = self._client.get_ticker(pair=None)
        return tickers_from_response(response)


class TickerBarBuilder:
    """Aggregates a stream of `Ticker` snapshots into fixed-interval OHLCV
    bars per symbol, purely from `LastPrice`/`CoinTradeValue`. This is a
    minute-by-minute-poll substitute for a real trade feed — coarser than
    exchange-native OHLCV, but it's the only history the live bot can build
    for itself, since /v3/ticker has no historical lookup.

    Usage: call `add(ticker)` every time you poll, and `flush_completed()`
    periodically to pop off any bars whose interval has closed.
    """

    def __init__(self, interval_seconds: int = 60):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval = interval_seconds
        self._open_bars: dict[str, dict] = {}
        self._completed: dict[str, list[dict]] = defaultdict(list)

    def _bucket_start(self, ts: datetime) -> datetime:
        epoch = ts.timestamp()
        bucket_epoch = epoch - (epoch % self._interval)
        return datetime.fromtimestamp(bucket_epoch, tz=timezone.utc)

    def add(self, ticker: Ticker) -> None:
        bucket = self._bucket_start(ticker.timestamp)
        bar = self._open_bars.get(ticker.symbol)
        if bar is None or bar["bucket"] != bucket:
            if bar is not None:
                self._completed[ticker.symbol].append(bar)
            bar = {
                "bucket": bucket,
                "open": ticker.price,
                "high": ticker.price,
                "low": ticker.price,
                "close": ticker.price,
                "volume": ticker.volume,
            }
            self._open_bars[ticker.symbol] = bar
        else:
            bar["high"] = max(bar["high"], ticker.price)
            bar["low"] = min(bar["low"], ticker.price)
            bar["close"] = ticker.price
            bar["volume"] = ticker.volume  # CoinTradeValue is cumulative, not delta

    def flush_completed(self, symbol: str) -> pd.DataFrame:
        """Pop off and return all bars for `symbol` whose interval has
        closed, as a DataFrame indexed by bucket start timestamp."""
        rows = self._completed.pop(symbol, [])
        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows).set_index("bucket").sort_index()
        return df
