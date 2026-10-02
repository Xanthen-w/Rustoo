"""Historical data layer: Binance archive parsing, the parquet and Bloomberg
sources, resampling and panel building. Offline — fixtures are built in
memory / tmp_path."""
import io
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.data.binance import (
    BinanceArchiveDownloader,
    filename_to_pair,
    pair_to_filename,
    parse_kline_csv,
    roostoo_to_binance_symbol,
)
from src.data.historical import BloombergExcelSource, ParquetDataSource, load_panel, resample_ohlcv


def _kline_row(open_time: int, price: float) -> str:
    return f"{open_time},{price},{price + 1},{price - 1},{price + 0.5},10,{open_time + 299999},1000,5,4,400,0"


MS_2024 = int(pd.Timestamp("2024-12-31 23:50", tz="UTC").timestamp() * 1000)


def test_symbol_and_filename_mapping():
    assert roostoo_to_binance_symbol("BTC/USD") == "BTCUSDT"
    assert pair_to_filename("1000CHEEMS/USD") == "1000CHEEMS-USD.parquet"
    assert filename_to_pair("1000CHEEMS-USD.parquet") == "1000CHEEMS/USD"


def test_parse_millisecond_klines_labels_bar_by_close_time():
    raw = "\n".join([_kline_row(MS_2024, 100.0), _kline_row(MS_2024 + 300_000, 101.0)]).encode()
    df = parse_kline_csv(raw, "5m")
    assert list(df.index) == [pd.Timestamp("2024-12-31 23:55", tz="UTC"), pd.Timestamp("2025-01-01 00:00", tz="UTC")]
    assert df["close"].tolist() == [100.5, 101.5]


def test_parse_microsecond_klines_with_header():
    us = MS_2024 * 1000
    header = ",".join(["open_time", "open", "high", "low", "close", "volume", "close_time",
                       "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
    raw = "\n".join([header, _kline_row(us, 100.0)]).encode()
    df = parse_kline_csv(raw, "5m")
    assert df.index[0] == pd.Timestamp("2024-12-31 23:55", tz="UTC")


class _ArchiveSession:
    """Serves a zip for every monthly/daily archive URL that has an entry in
    `available`, 404 otherwise."""

    def __init__(self, available: dict[str, str]):
        self.available = available
        self.requested = []

    def get(self, url, timeout=None):
        self.requested.append(url)
        name = url.rsplit("/", 1)[1]
        resp = type("R", (), {})()
        if name not in self.available:
            resp.status_code = 404
            return resp
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name.replace(".zip", ".csv"), self.available[name])
        resp.status_code = 200
        resp.content = buf.getvalue()
        resp.raise_for_status = lambda: None
        return resp


def test_downloader_merges_monthly_and_daily_and_caches(tmp_path):
    dec = int(pd.Timestamp("2024-12-31 23:55", tz="UTC").timestamp() * 1000)
    jan = int(pd.Timestamp("2025-01-01 00:00", tz="UTC").timestamp() * 1_000_000)  # µs era
    session = _ArchiveSession({
        "BTCUSDT-5m-2024-12.zip": _kline_row(dec, 100.0),
        "BTCUSDT-5m-2025-01-01.zip": _kline_row(jan, 101.0),
    })
    dl = BinanceArchiveDownloader(tmp_path / "cache", tmp_path / "out", "5m", session=session)
    report = dl.download_pair("BTC/USD", date(2024, 12, 1), date(2025, 1, 2))

    assert report.rows == 2
    assert report.files_missing == 1  # 2025-01-02 daily file not published
    df = pd.read_parquet(tmp_path / "out" / "BTC-USD.parquet")
    assert df.index.to_series().diff().dropna().eq(pd.Timedelta("5min")).all()

    session.requested.clear()
    report = BinanceArchiveDownloader(tmp_path / "cache", tmp_path / "out", "5m", session=session).download_pair(
        "BTC/USD", date(2024, 12, 1), date(2025, 1, 2)
    )
    assert report.files_cached == 2
    assert all("2025-01-02" in url for url in session.requested)  # only the missing file is retried


def test_downloader_uses_daily_files_while_last_months_archive_is_unpublished(tmp_path):
    dec = int(pd.Timestamp("2024-12-31 23:55", tz="UTC").timestamp() * 1000)
    jan = int(pd.Timestamp("2025-01-01 00:00", tz="UTC").timestamp() * 1_000_000)
    session = _ArchiveSession({
        "BTCUSDT-5m-2024-12-31.zip": _kline_row(dec, 100.0),  # no 2024-12 monthly archive yet
        "BTCUSDT-5m-2025-01-01.zip": _kline_row(jan, 101.0),
    })
    dl = BinanceArchiveDownloader(tmp_path / "cache", tmp_path / "out", "5m", session=session)
    report = dl.download_pair("BTC/USD", date(2024, 11, 15), date(2025, 1, 1))

    assert report.rows == 2
    assert report.files_downloaded == 2
    # Only the month just ended falls back to daily files: an older missing
    # month means the symbol wasn't trading, not a late archive.
    assert not any("2024-11-" in url for url in session.requested)
    assert sum("2024-12-" in url for url in session.requested) == 31


def _write_parquet(root, pair, index, close):
    root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=index)
    df.index.name = "timestamp"
    df.to_parquet(root / pair_to_filename(pair))


def test_parquet_source_slices_inclusively_and_never_past_end(tmp_path):
    idx = pd.date_range("2025-01-01 00:05", periods=12, freq="5min", tz="UTC")
    _write_parquet(tmp_path, "BTC/USD", idx, np.arange(1.0, 13.0))
    src = ParquetDataSource(tmp_path)
    assert src.available_pairs() == ["BTC/USD"]
    df = src.load("BTC/USD", "2025-01-01 00:10", "2025-01-01 00:30")
    assert df.index[0] == pd.Timestamp("2025-01-01 00:10", tz="UTC")
    assert df.index[-1] == pd.Timestamp("2025-01-01 00:30", tz="UTC")
    with pytest.raises(KeyError):
        src.load("ETH/USD", "2025-01-01", "2025-01-02")


def test_resample_uses_close_time_labels():
    idx = pd.date_range("2025-01-01 00:05", periods=24, freq="5min", tz="UTC")  # (00:00, 02:00]
    df = pd.DataFrame({"open": np.arange(24.0) + 1, "high": 100.0, "low": 1.0, "close": np.arange(24.0) + 1, "volume": 1.0}, index=idx)
    hourly = resample_ohlcv(df, "1h")
    assert list(hourly.index) == [pd.Timestamp("2025-01-01 01:00", tz="UTC"), pd.Timestamp("2025-01-01 02:00", tz="UTC")]
    assert hourly["open"].tolist() == [1.0, 13.0]
    assert hourly["close"].tolist() == [12.0, 24.0]  # the 01:00 bar's close is the 00:55-01:00 bar's close
    assert hourly["volume"].tolist() == [12.0, 12.0]


def test_panel_outer_joins_pairs_with_different_histories(tmp_path):
    idx = pd.date_range("2025-01-01 00:05", periods=6, freq="5min", tz="UTC")
    _write_parquet(tmp_path, "BTC/USD", idx, np.full(6, 100.0))
    _write_parquet(tmp_path, "NEW/USD", idx[3:], np.full(3, 5.0))
    panel = load_panel(ParquetDataSource(tmp_path), ["BTC/USD", "NEW/USD"], "2025-01-01", "2025-01-02")
    assert panel.shape == (6, 2)
    assert panel["NEW/USD"].iloc[:3].isna().all()


def test_bloomberg_source_converts_ist_bar_start_to_utc_close(tmp_path):
    path = tmp_path / "BTC OHLC.xlsx"
    # Newest-first, IST, labelled by bar start — as the real exports are.
    pd.DataFrame(
        {
            "Date": [pd.Timestamp("2026-09-08 02:35"), pd.Timestamp("2026-09-08 02:30")],
            "Open": [101.0, 100.0], "High": [102.0, 101.0], "Low": [100.0, 99.0], "Close": [101.5, 100.5],
        }
    ).to_excel(path, index=False)
    df = BloombergExcelSource({"BTC/USD": path}).load("BTC/USD", "2026-01-01", "2027-01-01")
    # 02:30 IST bar start = 21:00 UTC previous day; closes at 21:05 UTC.
    assert list(df.index) == [pd.Timestamp("2026-09-07 21:05", tz="UTC"), pd.Timestamp("2026-09-07 21:10", tz="UTC")]
    assert df["close"].tolist() == [100.5, 101.5]
    assert df["volume"].isna().all()
