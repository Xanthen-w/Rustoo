"""Excel/CSV importer: reading, column detection, timestamps, validation,
handling modes, duplicate content and reference checks. Synthetic files only."""
import numpy as np
import pandas as pd
import pytest

from src.data.historical import ParquetDataSource
from src.data.importer import (
    ColumnMappingError,
    detect_columns,
    flag_duplicate_contents,
    guess_symbol,
    import_file,
    parse_timestamps,
    read_table,
    validate,
)


def ohlc(n=600, start="2026-09-01 00:00", freq="5min", seed=0, base=100.0):
    rng = np.random.default_rng(seed)
    t = pd.date_range(start, periods=n, freq=freq)
    close = base * np.cumprod(1 + rng.normal(0, 0.002, n))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.001, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.001, n))
    return pd.DataFrame({"Date": t, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": rng.uniform(1, 10, n)})


def canonical(df):
    return df.rename(columns=str.lower).set_index("date").rename_axis("timestamp").tz_localize("UTC")


# -- reading and detection ---------------------------------------------------------------

def test_excel_with_title_rows_and_newest_first(tmp_path):
    path = tmp_path / "BTC export.xlsx"
    df = ohlc().iloc[::-1]
    with pd.ExcelWriter(path) as w:
        pd.DataFrame([["Bloomberg intraday bars"], ["XBTUSD Curncy"]]).to_excel(w, header=False, index=False)
        df.to_excel(w, startrow=3, index=False)
    table = read_table(path)
    assert list(table.columns)[:5] == ["Date", "Open", "High", "Low", "Close"]
    [res] = import_file(path, tmp_path / "out", tz="UTC", labelled_by="start")
    assert res.symbol == "BTC/USD" and res.status == "imported"
    assert any("sorted" in n for n in res.notes)
    out = pd.read_parquet(res.output)
    assert out.index.is_monotonic_increasing
    assert out.index[0] == pd.Timestamp("2026-09-01 00:05", tz="UTC")  # start label -> close label


@pytest.mark.parametrize("columns", [
    ["Date_Time", "OpenPrice", "HighPrice", "LowPrice", "ClosePrice", "Vol"],
    ["timestamp", "open", "high", "low", "close", "volume"],
    ["Datetime", "Open Price", "High", "Low", "PX_LAST", "PX_VOLUME"],
])
def test_aliases_are_detected(columns):
    det = detect_columns(columns)
    assert det.missing == [] and det.ambiguous == {}
    assert set(det.mapping) >= {"timestamp", "open", "high", "low", "close", "volume"}


def test_ambiguous_columns_refuse_to_guess(tmp_path):
    df = ohlc()
    df["Last"] = df["Close"]  # both Close and Last map to close
    path = tmp_path / "ETH.csv"
    df.to_csv(path, index=False)
    [res] = import_file(path, tmp_path / "out")
    assert res.status == "rejected" and "ambiguous" in res.issues[0]["message"]
    [res] = import_file(path, tmp_path / "out", mapping={"close": "Close"}, tz="UTC")
    assert res.status == "imported"


def test_missing_required_column_is_explained(tmp_path):
    path = tmp_path / "SOL.csv"
    ohlc().drop(columns="Low").to_csv(path, index=False)
    [res] = import_file(path, tmp_path / "out")
    assert res.status == "rejected" and "low" in res.issues[0]["message"]


def test_epoch_timestamps_and_timezone():
    ms = pd.Series([1_700_000_000_000, 1_700_000_300_000])
    idx, notes = parse_timestamps(ms, None)
    assert idx[0] == pd.Timestamp(1_700_000_000_000, unit="ms", tz="UTC") and "epoch ms" in notes[0]
    naive = pd.Series(pd.to_datetime(["2026-09-08 02:30"]))
    idx, _ = parse_timestamps(naive, "Asia/Kolkata")
    assert idx[0] == pd.Timestamp("2026-09-07 21:00", tz="UTC")
    _, notes = parse_timestamps(naive, None)
    assert "assumed UTC" in notes[0]


def test_symbol_guessing():
    from pathlib import Path
    assert guess_symbol(Path("dogecoin ohlc.xlsx")) == "DOGE/USD"
    assert guess_symbol(Path("TRON ohlc.xlsx")) == "TRX/USD"
    assert guess_symbol(Path("prices.xlsx")) is None


def test_combined_multi_symbol_file_is_split(tmp_path):
    a, b = ohlc(seed=1).assign(Ticker="BTC/USD"), ohlc(seed=2, base=5).assign(Ticker="ETH/USD")
    path = tmp_path / "combined.csv"
    pd.concat([a, b]).sample(frac=1, random_state=0).to_csv(path, index=False)
    res = import_file(path, tmp_path / "out", tz="UTC")
    assert {r.symbol for r in res} == {"BTC/USD", "ETH/USD"}
    src = ParquetDataSource(tmp_path / "out")
    btc = src.load("BTC/USD", "2026-01-01", "2027-01-01")
    assert np.allclose(btc["close"].to_numpy(), a["Close"].to_numpy())


# -- validation ---------------------------------------------------------------------------------

def test_validation_flags_each_problem():
    df = canonical(ohlc(200))
    df.iloc[10, df.columns.get_loc("high")] = df.iloc[10]["low"] * 0.5  # high < low
    df.iloc[20, df.columns.get_loc("close")] = -1  # non-positive
    df.iloc[30, df.columns.get_loc("open")] = np.nan  # missing
    dup = df.iloc[[40]]
    df = pd.concat([df, dup, dup.assign(close=dup.close * 1.01)]).sort_index()
    codes = {i.code for i in validate(df)}
    assert {"high_below_low", "non_positive_price", "missing_prices", "duplicate_rows", "duplicate_timestamps",
            "ohlc_envelope"} <= codes


def test_weekday_only_series_is_flagged_not_24_7():
    df = canonical(ohlc(12 * 24 * 21, freq="5min"))
    weekdays = df[df.index.dayofweek < 5]
    assert "not_24_7" in {i.code for i in validate(weekdays)}
    assert "not_24_7" not in {i.code for i in validate(df)}


def test_gaps_and_extreme_moves_are_warnings():
    df = canonical(ohlc(300))
    df = df.drop(df.index[100:150])
    df.iloc[200, df.columns.get_loc("close")] *= 1.5
    issues = {i.code: i for i in validate(df)}
    assert issues["gaps"].severity == "warning"
    assert "extreme_moves" in issues


# -- modes -----------------------------------------------------------------------------------------

def _dirty_file(tmp_path):
    df = ohlc(300)
    df.loc[50, "High"] = df.loc[50, "Low"] * 0.5
    df = pd.concat([df, df.iloc[[60]]])  # exact duplicate row
    path = tmp_path / "XRP.csv"
    df.to_csv(path, index=False)
    return path


@pytest.mark.parametrize("mode,status,removed", [("reject", "rejected", 0), ("warn", "rejected", 0),
                                                 ("repair", "rejected", 0), ("remove", "imported_with_warnings", 2)])
def test_handling_modes(tmp_path, mode, status, removed):
    [res] = import_file(_dirty_file(tmp_path), tmp_path / "out", tz="UTC", mode=mode)
    assert res.status == status and res.rows_removed == removed
    assert (res.output is not None) == (status != "rejected")


def test_repair_fixes_only_exact_duplicates(tmp_path):
    df = pd.concat([ohlc(300), ohlc(300).iloc[[60]]])
    path = tmp_path / "LTC.csv"
    df.to_csv(path, index=False)
    [res] = import_file(path, tmp_path / "out", tz="UTC", mode="repair")
    assert res.status == "imported_with_warnings" and res.rows_removed == 1


# -- cross-file checks ------------------------------------------------------------------------------

def test_duplicate_content_keeps_reference_confirmed_copy(tmp_path):
    df = ohlc(400, seed=3)
    df.to_csv(tmp_path / "XRP.csv", index=False)
    df.to_csv(tmp_path / "USDC.csv", index=False)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref = canonical(df)
    ref.index = ref.index + pd.Timedelta(minutes=5)
    ref.to_parquet(ref_dir / "XRP-USD.parquet")
    reference = ParquetDataSource(ref_dir)
    results = [r for f in ("XRP.csv", "USDC.csv") for r in import_file(tmp_path / f, tmp_path / "out", tz="UTC", reference=reference)]
    flag_duplicate_contents(results)
    status = {r.symbol: r.status for r in results}
    assert status == {"XRP/USD": "imported_with_warnings", "USDC/USD": "rejected"}
    assert not (tmp_path / "out" / "USDC-USD.parquet").exists()


def test_reference_mismatch_rejects_wrong_instrument(tmp_path):
    df = ohlc(400, seed=4, base=0.83)  # "BNB" at 0.83
    df.to_csv(tmp_path / "BNB.csv", index=False)
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    real = canonical(ohlc(400, seed=5, base=600.0))
    real.index = real.index + pd.Timedelta(minutes=5)
    real.to_parquet(ref_dir / "BNB-USD.parquet")
    [res] = import_file(tmp_path / "BNB.csv", tmp_path / "out", tz="UTC", reference=ParquetDataSource(ref_dir))
    assert res.status == "rejected"
    assert "reference_price_mismatch" in {i["code"] for i in res.issues}
