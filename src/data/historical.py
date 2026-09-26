"""Concrete historical data sources + panel helpers for research/backtests.

Timestamp convention for every source here: the index is the bar's *close*
time in UTC, i.e. the moment its close price becomes known. A strategy
reading row t therefore only uses information available at time t, and the
backtest engine's execution lag fills it at the next row.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.binance import filename_to_pair, pair_to_filename
from src.data.market_data import OHLCV_COLUMNS, HistoricalDataSource, validate_ohlcv


def _to_utc(ts: datetime | pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _slice(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df[(df.index >= _to_utc(start)) & (df.index <= _to_utc(end))]


class ParquetDataSource(HistoricalDataSource):
    """One parquet file per pair (e.g. `BTC-USD.parquet`), as written by
    src/data/binance.py."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self._cache: dict[str, pd.DataFrame] = {}

    def available_pairs(self) -> list[str]:
        return sorted(filename_to_pair(p.name) for p in self.root.glob("*.parquet"))

    def _read(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            path = self.root / pair_to_filename(symbol)
            if not path.exists():
                raise KeyError(f"no historical data file for {symbol} at {path}")
            df = pd.read_parquet(path)
            for col in OHLCV_COLUMNS:
                if col not in df.columns:
                    df[col] = float("nan")
            validate_ohlcv(df)
            self._cache[symbol] = df
        return self._cache[symbol]

    def load(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(self._read(symbol), start, end)


class BloombergExcelSource(HistoricalDataSource):
    """Bloomberg intraday-bar exports: one .xlsx per coin with columns
    Date/Open/High/Low/Close (no volume), newest row first.

    Verified against Binance on the Sept 2026 exports: timestamps are in the
    terminal's local time (IST, UTC+5:30) and label the bar's *start*, so
    each is shifted to its UTC close time here. Volume is filled with NaN.
    """

    def __init__(
        self,
        files: dict[str, Path | str],
        timezone: str = "Asia/Kolkata",
        bar_length: pd.Timedelta = pd.Timedelta(minutes=5),
        labelled_by: str = "start",
    ):
        if labelled_by not in {"start", "end"}:
            raise ValueError("labelled_by must be 'start' or 'end'")
        self.files = {pair: Path(path) for pair, path in files.items()}
        self.timezone = timezone
        self.bar_length = bar_length
        self.labelled_by = labelled_by
        self._cache: dict[str, pd.DataFrame] = {}

    def available_pairs(self) -> list[str]:
        return sorted(self.files)

    def _read(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._cache:
            if symbol not in self.files:
                raise KeyError(f"no Bloomberg file configured for {symbol}")
            raw = pd.read_excel(self.files[symbol])
            raw.columns = [str(c).strip().lower() for c in raw.columns]
            index = pd.DatetimeIndex(pd.to_datetime(raw["date"]))
            index = index.tz_localize(self.timezone).tz_convert("UTC")
            if self.labelled_by == "start":
                index = index + self.bar_length
            df = raw[["open", "high", "low", "close"]].astype(float)
            df.index = pd.DatetimeIndex(index, name="timestamp")
            df["volume"] = float("nan")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            validate_ohlcv(df)
            self._cache[symbol] = df
        return self._cache[symbol]

    def load(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        return _slice(self._read(symbol), start, end)


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate close-time-labelled bars into coarser ones, still labelled
    by close time (so a 1h bar stamped 10:00 covers (09:00, 10:00]). Bars
    with no underlying data are dropped, not forward-filled."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(rule, label="right", closed="right").agg(agg)
    if "volume" in out.columns and df["volume"].isna().all():
        out["volume"] = float("nan")
    return out.dropna(subset=["close"])


def load_panel(
    source: HistoricalDataSource,
    pairs: list[str],
    start,
    end,
    field: str = "close",
    resample: str | None = None,
) -> pd.DataFrame:
    """Wide frame (index = UTC close time, columns = pairs) of one OHLCV
    field. Pairs are outer-joined on time; a pair with no bar at some time
    (not yet listed, exchange outage) is NaN there, which the backtest
    engine treats as untradable for that bar."""
    columns = {}
    for pair in pairs:
        df = source.load(pair, start, end)
        if resample:
            df = resample_ohlcv(df, resample)
        columns[pair] = df[field]
    panel = pd.DataFrame(columns).sort_index()
    return panel
