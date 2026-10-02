"""Historical OHLCV from Binance's public bulk-data archive
(https://data.binance.vision) — research/backtesting only.

Roostoo has no historical endpoint (docs/API_NOTES.md open question #2), so
backtests use Binance spot klines for the same coins. Roostoo pairs are
quoted in USD, Binance's in USDT; the basis is a few bps (USDT trades within
~0.1% of $1) and is ignored here.

Archives are cached under `cache_dir` exactly as downloaded, so re-running a
download only fetches files not already on disk. Output is one parquet file
per Roostoo pair (see src/data/historical.py::ParquetDataSource), indexed by
bar *close* time in UTC — the moment the bar's close price is known.
"""
from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ARCHIVE_BASE_URL = "https://data.binance.vision/data/spot"

KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore",
]

_INTERVAL_OFFSETS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "30m": pd.Timedelta(minutes=30),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


def roostoo_to_binance_symbol(pair: str, quote: str = "USDT") -> str:
    """'BTC/USD' -> 'BTCUSDT'."""
    coin, _unit = pair.split("/")
    return f"{coin}{quote}"


def interval_offset(interval: str) -> pd.Timedelta:
    if interval not in _INTERVAL_OFFSETS:
        raise ValueError(f"unsupported interval {interval!r}; expected one of {sorted(_INTERVAL_OFFSETS)}")
    return _INTERVAL_OFFSETS[interval]


def parse_kline_csv(raw: bytes, interval: str) -> pd.DataFrame:
    """Parse one archive CSV into OHLCV indexed by UTC bar close time.

    Handles both archive formats: older files carry millisecond timestamps,
    spot files from 2025 onward carry microseconds; some files have a header
    row, most don't.
    """
    df = pd.read_csv(io.BytesIO(raw), header=None, names=KLINE_COLUMNS, dtype=str)
    if len(df) and not df.iloc[0]["open_time"].strip().isdigit():
        df = df.iloc[1:]  # header row
    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "quote_volume"])

    open_time = df["open_time"].astype("int64")
    # ms timestamps are 13 digits for this era; µs are 16.
    unit = "us" if open_time.iloc[0] > 10**14 else "ms"
    index = pd.to_datetime(open_time, unit=unit, utc=True) + interval_offset(interval)

    out = df[["open", "high", "low", "close", "volume", "quote_volume"]].astype(float)
    out.index = pd.DatetimeIndex(index, name="timestamp")
    return out


@dataclass
class DownloadReport:
    pair: str
    symbol: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    files_downloaded: int
    files_cached: int
    files_missing: int
    output: Path | None


class BinanceArchiveDownloader:
    def __init__(
        self,
        cache_dir: Path,
        output_dir: Path,
        interval: str = "5m",
        session: requests.Session | None = None,
        timeout_seconds: float = 60.0,
    ):
        interval_offset(interval)  # validate early
        self.cache_dir = Path(cache_dir)
        self.output_dir = Path(output_dir)
        self.interval = interval
        self.session = session or requests.Session()
        self.timeout_seconds = timeout_seconds

    def _monthly_item(self, symbol: str, month: date) -> tuple[str, Path]:
        name = f"{symbol}-{self.interval}-{month:%Y-%m}.zip"
        return (f"{ARCHIVE_BASE_URL}/monthly/klines/{symbol}/{self.interval}/{name}",
                self.cache_dir / symbol / self.interval / "monthly" / name)

    def _daily_items(self, symbol: str, first: date, last: date) -> list[tuple[str, Path]]:
        items: list[tuple[str, Path]] = []
        day = first
        while day <= last:
            name = f"{symbol}-{self.interval}-{day:%Y-%m-%d}.zip"
            items.append(
                (f"{ARCHIVE_BASE_URL}/daily/klines/{symbol}/{self.interval}/{name}",
                 self.cache_dir / symbol / self.interval / "daily" / name)
            )
            day += timedelta(days=1)
        return items

    def _archive_urls(self, symbol: str, start: date, end: date) -> list[tuple[str, Path]]:
        """Monthly archives for every whole month in [start, end] that has
        ended before `end`'s month, daily archives for the remainder."""
        items: list[tuple[str, Path]] = []
        month = date(start.year, start.month, 1)
        end_month = date(end.year, end.month, 1)
        while month < end_month:
            items.append(self._monthly_item(symbol, month))
            month = date(month.year + (month.month == 12), month.month % 12 + 1, 1)
        return items + self._daily_items(symbol, max(end_month, start), end)

    def _fetch(self, url: str, path: Path) -> str:
        """Returns 'cached', 'downloaded' or 'missing' (404: the symbol
        didn't trade that period, or the archive isn't published yet)."""
        if path.exists():
            return "cached"
        resp = self.session.get(url, timeout=self.timeout_seconds)
        if resp.status_code == 404:
            return "missing"
        resp.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(resp.content)
        tmp.replace(path)
        return "downloaded"

    def download_pair(self, pair: str, start: date, end: date) -> DownloadReport:
        symbol = roostoo_to_binance_symbol(pair)
        counts = {"cached": 0, "downloaded": 0, "missing": 0}
        frames = []
        items = self._archive_urls(symbol, start, end)
        # Binance publishes a month's archive a few days into the next month.
        # Until then the month just ended is only available as daily files.
        last_month_end = date(end.year, end.month, 1) - timedelta(days=1)
        last_month = self._monthly_item(symbol, date(last_month_end.year, last_month_end.month, 1))
        while items:
            url, path = items.pop(0)
            status = self._fetch(url, path)
            if status == "missing" and (url, path) == last_month:
                items = self._daily_items(symbol, max(last_month_end.replace(day=1), start), last_month_end) + items
                continue
            counts[status] += 1
            if status == "missing":
                continue
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    frames.append(parse_kline_csv(zf.read(member), self.interval))

        frames = [f for f in frames if not f.empty]
        if not frames:
            logger.warning("no binance data", extra={"pair": pair, "symbol": symbol})
            return DownloadReport(pair, symbol, 0, None, None, counts["downloaded"], counts["cached"], counts["missing"], None)

        df = pd.concat(frames).sort_index()
        df = df[~df.index.duplicated(keep="last")]
        lo = pd.Timestamp(start, tz="UTC")
        hi = pd.Timestamp(end + timedelta(days=1), tz="UTC")
        df = df[(df.index > lo) & (df.index <= hi)]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output = self.output_dir / pair_to_filename(pair)
        df.to_parquet(output)
        return DownloadReport(
            pair, symbol, len(df), df.index[0], df.index[-1],
            counts["downloaded"], counts["cached"], counts["missing"], output,
        )


def pair_to_filename(pair: str) -> str:
    """'BTC/USD' -> 'BTC-USD.parquet'."""
    return pair.replace("/", "-") + ".parquet"


def filename_to_pair(name: str) -> str:
    return Path(name).stem.replace("-", "/", 1)
