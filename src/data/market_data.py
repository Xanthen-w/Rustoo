"""Normalized market-data structures.

Strategy code depends only on these types, never on raw Roostoo JSON. This is
what lets the strategy layer stay untestable-against-a-live-API-free — you can
construct a `Ticker` or an OHLCV `DataFrame` by hand in a test and feed it to
any strategy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Ticker:
    """A single live snapshot for one pair, normalized from GET /v3/ticker."""

    timestamp: datetime
    symbol: str
    price: float  # LastPrice
    bid: float  # MaxBid
    ask: float  # MinAsk
    volume: float  # CoinTradeValue (base-asset volume traded)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @classmethod
    def from_roostoo(cls, symbol: str, data: dict, server_time_ms: int) -> "Ticker":
        return cls(
            timestamp=datetime.fromtimestamp(server_time_ms / 1000, tz=timezone.utc),
            symbol=symbol,
            price=float(data["LastPrice"]),
            bid=float(data["MaxBid"]),
            ask=float(data["MinAsk"]),
            volume=float(data.get("CoinTradeValue", 0.0)),
        )


def tickers_from_response(response: dict) -> dict[str, Ticker]:
    """Parse the full body of GET /v3/ticker (single-pair or all-pairs)."""
    server_time_ms = int(response["ServerTime"])
    return {
        symbol: Ticker.from_roostoo(symbol, payload, server_time_ms)
        for symbol, payload in response.get("Data", {}).items()
    }


class HistoricalDataSource(ABC):
    """Pluggable interface for historical OHLCV bars used only by research /
    backtesting. Roostoo itself exposes no historical endpoint (see
    docs/API_NOTES.md) so the concrete implementation is intentionally not
    fixed yet — plug in whatever source you actually have (CSV, parquet, a
    self-collected ticker archive, a vendor API) by implementing `load`.
    """

    @abstractmethod
    def load(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Return a DataFrame indexed by UTC timestamp (ascending, no gaps
        required) with columns OHLCV_COLUMNS, containing only bars with
        timestamp in [start, end]. Must not return anything timestamped after
        `end` — callers rely on this for look-ahead safety.
        """

    def load_panel(self, symbols: list[str], start: datetime, end: datetime) -> dict[str, pd.DataFrame]:
        return {symbol: self.load(symbol, start, end) for symbol in symbols}


def validate_ohlcv(df: pd.DataFrame) -> None:
    """Sanity checks a research framework should run on any loaded history
    before backtesting on it, to catch the failure modes Phase 6 calls out."""
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"OHLCV frame missing columns: {missing}")
    if not df.index.is_monotonic_increasing:
        raise ValueError("OHLCV frame index must be sorted ascending (chronological)")
    if df.index.has_duplicates:
        raise ValueError("OHLCV frame index has duplicate timestamps")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLCV frame contains non-positive prices")
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        raise ValueError(f"OHLCV frame has {bad_hl.sum()} bars where high < low")
