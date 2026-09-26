"""Import OHLC(V) data from Excel/CSV files into the canonical format.

Canonical format (same as every other source in this repo): a DataFrame
indexed by bar *close* time in UTC with columns open, high, low, close,
volume (NaN when the file has no volume — never fabricated), saved as one
parquet file per pair that src/data/historical.py::ParquetDataSource reads.

Pipeline: read table (finding the header row, even below title rows) ->
detect columns from aliases (refusing to guess when ambiguous) -> parse
timestamps (datetime, text or epoch s/ms/us; source timezone; bar-start
or bar-end labels) -> validate -> apply the handling mode -> save + report.

Handling modes (what happens to rows that fail validation):
- reject: any error-level issue -> nothing is imported
- warn:   import as-is if there are no error-level issues; warnings attached
- remove: drop the offending rows (duplicate timestamps, missing/invalid
          prices), then import; every removal is counted in the report
- repair: only objectively safe fixes (drop exact duplicate rows); anything
          else still blocks like `warn`
Sorting into chronological order is always applied and always reported.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.binance import pair_to_filename

CANONICAL = ("timestamp", "open", "high", "low", "close", "volume", "symbol")
REQUIRED = ("timestamp", "open", "high", "low", "close")

ALIASES = {
    "timestamp": {"timestamp", "date", "datetime", "time", "datetimeutc", "dates", "opentime", "closetime", "bartime", "period"},
    "open": {"open", "openprice", "o", "pxopen", "opening", "first"},
    "high": {"high", "highprice", "h", "pxhigh", "max"},
    "low": {"low", "lowprice", "l", "pxlow", "min"},
    "close": {"close", "closeprice", "c", "last", "lastprice", "pxlast", "closing", "price"},
    "volume": {"volume", "vol", "pxvolume", "basevolume", "qty", "quantity", "volumebase"},
    "symbol": {"symbol", "ticker", "pair", "asset", "coin", "instrument", "security"},
}

# File-name / ticker words -> Roostoo pair (extend as needed).
SYMBOL_WORDS = {
    "btc": "BTC/USD", "bitcoin": "BTC/USD", "xbt": "BTC/USD", "eth": "ETH/USD", "ethereum": "ETH/USD",
    "sol": "SOL/USD", "solana": "SOL/USD", "bnb": "BNB/USD", "xrp": "XRP/USD", "ripple": "XRP/USD",
    "trx": "TRX/USD", "tron": "TRX/USD", "doge": "DOGE/USD", "dogecoin": "DOGE/USD", "zec": "ZEC/USD",
    "zcash": "ZEC/USD", "xmr": "XMR/USD", "monero": "XMR/USD", "ada": "ADA/USD", "cardano": "ADA/USD",
    "usdt": "USDT/USD", "tether": "USDT/USD", "usdc": "USDC/USD", "ltc": "LTC/USD", "litecoin": "LTC/USD",
    "link": "LINK/USD", "avax": "AVAX/USD", "dot": "DOT/USD", "sui": "SUI/USD",
}


class ImportError_(ValueError):
    """Raised when a file can't be imported as requested."""


class ColumnMappingError(ImportError_):
    """Column detection was ambiguous or incomplete; pass an explicit mapping."""


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -- reading --------------------------------------------------------------------------

def read_table(path: Path, sheet: str | int | None = None, max_header_scan: int = 25) -> pd.DataFrame:
    """Read a .csv/.xlsx/.xls file, locating the header row: the first row
    (within `max_header_scan`) where at least 3 cells are known aliases."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(path, header=None, dtype=object)
    elif suffix in (".xlsx", ".xlsm", ".xls"):
        try:
            raw = pd.read_excel(path, header=None, sheet_name=0 if sheet is None else sheet, dtype=object)
        except ImportError as exc:  # .xls needs xlrd
            raise ImportError_(f"{path.name}: reading {suffix} needs an extra package ({exc}). "
                               "Re-save the file as .xlsx or .csv, or `pip install xlrd`.") from exc
    else:
        raise ImportError_(f"{path.name}: unsupported file type {suffix} (use .csv, .xlsx or .xls)")
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise ImportError_(f"{path.name}: no data")
    known = set().union(*ALIASES.values())
    for i in range(min(max_header_scan, len(raw))):
        cells = [_norm(c) for c in raw.iloc[i].tolist() if pd.notna(c)]
        if sum(c in known for c in cells) >= 3:
            table = raw.iloc[i + 1:].copy()
            table.columns = [str(c).strip() if pd.notna(c) else f"unnamed_{j}" for j, c in enumerate(raw.iloc[i])]
            return table.reset_index(drop=True)
    raise ColumnMappingError(f"{path.name}: couldn't find a header row with recognizable column names "
                             f"(looked at the first {max_header_scan} rows)")


@dataclass
class ColumnDetection:
    mapping: dict  # canonical -> source column
    ambiguous: dict  # canonical -> [candidate columns]
    unmapped: list  # source columns not used
    missing: list  # required canonical columns not found


def detect_columns(columns: list[str], explicit: dict | None = None) -> ColumnDetection:
    """Map source columns onto canonical names by alias. `explicit`
    ({canonical: source}) overrides detection."""
    explicit = dict(explicit or {})
    candidates: dict[str, list[str]] = {k: [] for k in ALIASES}
    for col in columns:
        n = _norm(col)
        for canon, names in ALIASES.items():
            if n in names:
                candidates[canon].append(col)
    mapping, ambiguous = {}, {}
    for canon, cands in candidates.items():
        if canon in explicit:
            continue
        if len(cands) == 1:
            mapping[canon] = cands[0]
        elif len(cands) > 1:
            ambiguous[canon] = cands
    for canon, src in explicit.items():
        if canon not in ALIASES:
            raise ColumnMappingError(f"unknown canonical column {canon!r}; expected one of {sorted(ALIASES)}")
        if src not in columns:
            raise ColumnMappingError(f"mapped column {src!r} (for {canon}) not in file; columns are {columns}")
        mapping[canon] = src
    missing = [c for c in REQUIRED if c not in mapping and c not in ambiguous]
    used = set(mapping.values())
    return ColumnDetection(mapping, ambiguous, [c for c in columns if c not in used], missing)


def guess_symbol(path: Path) -> str | None:
    words = re.findall(r"[a-z0-9]+", path.stem.lower())
    hits = {SYMBOL_WORDS[w] for w in words if w in SYMBOL_WORDS}
    return hits.pop() if len(hits) == 1 else None


def parse_timestamps(values: pd.Series, tz: str | None) -> tuple[pd.DatetimeIndex, list[str]]:
    """Datetimes, date strings or epoch numbers (s/ms/us by magnitude) ->
    UTC DatetimeIndex. Naive values are localized to `tz` (UTC with a
    warning when `tz` is None)."""
    notes = []
    is_datetime = pd.api.types.is_datetime64_any_dtype(values) or values.map(
        lambda v: isinstance(v, (pd.Timestamp, datetime, np.datetime64))).any()
    # Only genuine numbers are epochs: pd.to_numeric would also turn datetimes
    # into nanosecond integers, silently skipping the timezone conversion.
    numeric = pd.to_numeric(values, errors="coerce") if not is_datetime else pd.Series(dtype=float)
    if len(numeric) and numeric.notna().all():
        mag = numeric.abs().median()
        unit = "us" if mag > 1e14 else "ms" if mag > 1e11 else "s"
        idx = pd.to_datetime(numeric.astype("int64"), unit=unit, utc=True)
        notes.append(f"timestamps parsed as epoch {unit}")
        return pd.DatetimeIndex(idx), notes
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce", utc=False))
    if idx.tz is None:
        if tz is None:
            notes.append("timestamps have no timezone and none was given: assumed UTC")
            idx = idx.tz_localize("UTC")
        else:
            idx = idx.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
            notes.append(f"timestamps localized from {tz}")
    return idx.tz_convert("UTC"), notes


# -- validation ---------------------------------------------------------------------------

@dataclass
class Issue:
    severity: str  # error | warning | info
    code: str
    message: str
    rows: int = 0


def infer_interval(index: pd.DatetimeIndex) -> pd.Timedelta | None:
    if len(index) < 3:
        return None
    return pd.Series(index).diff().dropna().mode().iloc[0]


def timeframe_label(interval: pd.Timedelta | None) -> str:
    if interval is None:
        return "unknown"
    minutes = interval.total_seconds() / 60
    for label, m in (("1m", 1), ("5m", 5), ("15m", 15), ("30m", 30), ("1h", 60), ("4h", 240), ("1d", 1440)):
        if abs(minutes - m) < 1e-9:
            return label
    return f"{minutes:g}min"


def validate(df: pd.DataFrame, *, crypto_24_7: bool = True, extreme_move: float = 0.2) -> list[Issue]:
    """Checks on a frame with a UTC timestamp index and OHLC(V) columns
    (already sorted). Nothing is modified here."""
    issues: list[Issue] = []
    n = len(df)
    if n == 0:
        return [Issue("error", "empty", "no rows")]
    bad_ts = df.index.isna().sum()
    if bad_ts:
        issues.append(Issue("error", "bad_timestamp", f"{bad_ts} rows have unparseable timestamps", int(bad_ts)))
    flat_rows = df.reset_index()
    dup_rows = int(flat_rows.duplicated().sum())  # extra copies of identical rows
    if dup_rows:
        issues.append(Issue("error", "duplicate_rows", f"{dup_rows} exact duplicate rows", dup_rows))
    distinct = flat_rows.drop_duplicates()
    conflicts = int(distinct["timestamp"].duplicated(keep=False).sum())  # same time, different values
    if conflicts:
        issues.append(Issue("error", "duplicate_timestamps", f"{conflicts} rows share a timestamp with different values", conflicts))
    prices = df[["open", "high", "low", "close"]]
    missing = prices.isna().any(axis=1).sum()
    if missing:
        issues.append(Issue("error", "missing_prices", f"{missing} rows with missing OHLC values", int(missing)))
    nonpos = (prices <= 0).any(axis=1).sum()
    if nonpos:
        issues.append(Issue("error", "non_positive_price", f"{nonpos} rows with zero or negative prices", int(nonpos)))
    hl = (df.high < df.low).sum()
    if hl:
        issues.append(Issue("error", "high_below_low", f"{hl} rows with high < low", int(hl)))
    envelope = ((df.high < df[["open", "close"]].max(axis=1)) | (df.low > df[["open", "close"]].min(axis=1))).sum()
    if envelope:
        issues.append(Issue("error", "ohlc_envelope", f"{envelope} rows where open/close lie outside [low, high]", int(envelope)))
    if "volume" in df and df.volume.notna().any():
        negv = (df.volume < 0).sum()
        if negv:
            issues.append(Issue("error", "negative_volume", f"{negv} rows with negative volume", int(negv)))
    else:
        issues.append(Issue("info", "no_volume", "file has no volume column: volume-based analysis (participation, market impact) unavailable"))

    interval = infer_interval(df.index)
    if interval is not None:
        diffs = pd.Series(df.index).diff().dropna()
        irregular = (diffs != interval).sum()
        gaps = diffs[diffs > interval]
        if len(gaps):
            issues.append(Issue("warning", "gaps", f"{len(gaps)} gaps (largest {gaps.max()}); "
                                f"{int((gaps / interval).sum() - len(gaps))} bars missing vs a regular {timeframe_label(interval)} grid",
                                int(len(gaps))))
        if irregular and irregular != len(gaps):
            issues.append(Issue("warning", "irregular_interval", f"{irregular - len(gaps)} intervals shorter than the dominant {interval}"))
        if crypto_24_7 and interval < pd.Timedelta(days=1):
            weekend = df.index.dayofweek >= 5
            span_weeks = (df.index[-1] - df.index[0]) / pd.Timedelta(days=7)
            if span_weeks >= 1 and weekend.mean() < 0.05:
                issues.append(Issue("error", "not_24_7", "almost no weekend bars although crypto trades 24/7: "
                                    "this looks like a different (exchange-hours) instrument"))
    rets = df.close.pct_change().abs()
    extreme = (rets > extreme_move).sum()
    if extreme:
        issues.append(Issue("warning", "extreme_moves", f"{extreme} bar-to-bar close moves above {extreme_move:.0%}", int(extreme)))
    flat = (prices.nunique(axis=1) == 1).mean()
    if flat > 0.5:
        issues.append(Issue("warning", "mostly_flat", f"{flat:.0%} of bars have open = high = low = close (stablecoin or illiquid?)"))
    return issues


def reference_check(df: pd.DataFrame, reference_close: pd.Series, max_median_bps: float = 50.0,
                    min_return_corr: float = 0.5) -> list[Issue]:
    """Compare against another source for the same pair on overlapping
    bars: catches wrong-ticker exports and timezone/label mistakes."""
    common = df.index.intersection(reference_close.index)
    if len(common) < 50:
        return [Issue("info", "reference_no_overlap", f"only {len(common)} bars overlap the reference; not compared")]
    a, b = df.close.loc[common], reference_close.loc[common]
    diff_bps = float(((a / b - 1).abs() * 1e4).median())
    corr = float(np.log(a).diff().corr(np.log(b).diff()))
    issues = [Issue("info", "reference", f"vs reference: median |close diff| {diff_bps:.1f} bps, return correlation {corr:.3f} "
                    f"over {len(common)} bars")]
    if diff_bps > max_median_bps:
        issues.append(Issue("error", "reference_price_mismatch", f"median close differs from the reference by {diff_bps:.0f} bps: "
                            "wrong instrument, wrong currency, or wrong timezone/labels?"))
    elif corr < min_return_corr:
        issues.append(Issue("warning", "reference_low_correlation", f"return correlation with the reference is only {corr:.2f}"))
    return issues


# -- import --------------------------------------------------------------------------------------

@dataclass
class ImportResult:
    file: str
    symbol: str | None
    status: str  # imported | imported_with_warnings | rejected
    timeframe: str = "unknown"
    start: str | None = None
    end: str | None = None
    bars: int = 0
    rows_removed: int = 0
    file_hash: str = ""
    content_hash: str = ""
    columns: dict = field(default_factory=dict)
    issues: list = field(default_factory=list)
    output: str | None = None
    notes: list = field(default_factory=list)

    @property
    def symbol_mark(self) -> str:
        return {"imported": "✓", "imported_with_warnings": "⚠", "rejected": "✕"}[self.status]


def _content_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df[["open", "high", "low", "close"]].round(10), index=False)
                          .to_numpy().tobytes()).hexdigest()[:16]


def normalize(path: Path, *, symbol: str | None = None, mapping: dict | None = None, tz: str | None = None,
              labelled_by: str = "start", bar: pd.Timedelta | None = None, sheet=None) -> tuple[dict[str, pd.DataFrame], list[str], dict]:
    """Read + map + parse into canonical frames per symbol (not yet
    validated). Returns (frames, notes, column mapping)."""
    if labelled_by not in ("start", "end"):
        raise ImportError_("labelled_by must be 'start' or 'end'")
    table = read_table(path, sheet)
    det = detect_columns(list(table.columns), mapping)
    if det.ambiguous:
        raise ColumnMappingError(f"{Path(path).name}: ambiguous columns {det.ambiguous}; pass an explicit mapping, "
                                 f"e.g. --map close={next(iter(det.ambiguous.values()))[0]}")
    if det.missing:
        raise ColumnMappingError(f"{Path(path).name}: required columns not found: {det.missing}. "
                                 f"Columns in file: {list(table.columns)}. Pass --map canonical=column.")
    idx, notes = parse_timestamps(table[det.mapping["timestamp"]], tz)
    frame = pd.DataFrame({c: pd.to_numeric(table[det.mapping[c]], errors="coerce").to_numpy()
                          for c in ("open", "high", "low", "close")}, index=idx)
    frame["volume"] = pd.to_numeric(table[det.mapping["volume"]], errors="coerce").to_numpy() if "volume" in det.mapping else np.nan
    if not frame.index.is_monotonic_increasing:
        notes.append("rows were not in chronological order: sorted")
    frame = frame.sort_index(kind="stable")
    interval = bar or infer_interval(frame.index.dropna())
    if labelled_by == "start":
        if interval is None:
            raise ImportError_("can't infer the bar length to convert start labels to close times; pass bar=...")
        frame.index = frame.index + interval
        notes.append(f"bar-start labels shifted by {interval} to close-time labels")
    frame.index.name = "timestamp"

    if "symbol" in det.mapping and symbol is None:
        syms = table[det.mapping["symbol"]].astype(str).str.strip().to_numpy()
        order = np.argsort(idx.to_numpy(), kind="stable")  # align with the sorted frame
        frames = {s: g.drop(columns="__sym") for s, g in frame.assign(__sym=syms[order]).groupby("__sym")}
        notes.append(f"combined file split into {len(frames)} symbols")
    else:
        sym = symbol or guess_symbol(Path(path))
        if sym is None:
            raise ImportError_(f"{Path(path).name}: can't tell which asset this is; pass --symbol (e.g. BTC/USD)")
        if symbol is None:
            notes.append(f"symbol {sym} guessed from the file name")
        frames = {sym: frame}
    return frames, notes, det.mapping


def apply_mode(df: pd.DataFrame, issues: list[Issue], mode: str) -> tuple[pd.DataFrame, int, str]:
    """Returns (frame, rows_removed, status)."""
    if mode not in ("reject", "warn", "remove", "repair"):
        raise ImportError_("mode must be reject, warn, remove or repair")
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    removed = 0
    if mode == "remove":
        before = len(df)
        out = df[~df.index.isna()]
        out = out.reset_index().drop_duplicates().set_index("timestamp")  # identical copies: keep one
        out = out[~out.index.duplicated(keep=False)]  # conflicting values at one time: drop all, can't tell which is right
        p = out[["open", "high", "low", "close"]]
        ok = p.notna().all(axis=1) & (p > 0).all(axis=1) & (out.high >= out.low) \
            & (out.high >= out[["open", "close"]].max(axis=1)) & (out.low <= out[["open", "close"]].min(axis=1))
        out = out[ok]
        removed = before - len(out)
        fixable = {"bad_timestamp", "duplicate_rows", "duplicate_timestamps", "missing_prices", "non_positive_price",
                   "high_below_low", "ohlc_envelope"}
        remaining = [e for e in errors if e.code not in fixable]
        return (out, removed, "rejected") if remaining else (out, removed, "imported_with_warnings" if (warnings or removed) else "imported")
    if mode == "repair":
        before = len(df)
        out = df.reset_index().drop_duplicates().set_index("timestamp")
        removed = before - len(out)
        remaining = [e for e in errors if e.code != "duplicate_rows"]
        if remaining:
            return df, 0, "rejected"
        return out, removed, "imported_with_warnings" if (warnings or removed) else "imported"
    if errors:
        return df, 0, "rejected"
    return df, 0, "imported_with_warnings" if warnings else "imported"


def import_file(path: Path, out_dir: Path, *, symbol: str | None = None, mapping: dict | None = None,
                tz: str | None = None, labelled_by: str = "start", mode: str = "warn", sheet=None,
                reference=None, crypto_24_7: bool = True) -> list[ImportResult]:
    """Import one file (one or several symbols). `reference`, if given, is a
    HistoricalDataSource used for a cross-source check."""
    path = Path(path)
    fhash = file_sha256(path)
    try:
        frames, notes, cols = normalize(path, symbol=symbol, mapping=mapping, tz=tz, labelled_by=labelled_by, sheet=sheet)
    except ImportError_ as exc:
        return [ImportResult(path.name, symbol, "rejected", file_hash=fhash,
                             issues=[asdict(Issue("error", "unreadable", str(exc)))])]
    results = []
    for sym, df in frames.items():
        issues = validate(df, crypto_24_7=crypto_24_7)
        if reference is not None:
            try:
                ref = reference.load(sym, df.index.min(), df.index.max())["close"]
                issues += reference_check(df, ref)
            except KeyError:
                issues.append(Issue("info", "reference_missing", f"no reference data for {sym}"))
        clean, removed, status = apply_mode(df, issues, mode)
        interval = infer_interval(clean.index) if len(clean) else None
        res = ImportResult(path.name, sym, status, timeframe_label(interval),
                           str(clean.index.min()) if len(clean) else None, str(clean.index.max()) if len(clean) else None,
                           len(clean), removed, fhash, _content_hash(df) if len(df) else "", cols,
                           [asdict(i) for i in issues], None, notes)
        if status != "rejected":
            out_dir.mkdir(parents=True, exist_ok=True)
            target = out_dir / pair_to_filename(sym)
            clean[["open", "high", "low", "close", "volume"]].to_parquet(target)
            res.output = str(target)
        results.append(res)
    return results


def _reference_confirmed(r: ImportResult) -> bool:
    codes = {i["code"] for i in r.issues}
    return "reference" in codes and not codes & {"reference_price_mismatch", "reference_low_correlation"}


def flag_duplicate_contents(results: list[ImportResult]) -> None:
    """Different files/symbols with identical price content (e.g. one
    export saved under two names): at most one can be genuine. If exactly
    one of them is confirmed by the reference source it's kept (with a
    warning); every other copy is rejected."""
    by_hash: dict[str, list[ImportResult]] = {}
    for r in results:
        if r.content_hash:
            by_hash.setdefault(r.content_hash, []).append(r)
    for group in by_hash.values():
        if len({r.symbol for r in group}) <= 1:
            continue
        names = ", ".join(f"{r.file} ({r.symbol})" for r in group)
        confirmed = [r for r in group if _reference_confirmed(r)]
        keeper = confirmed[0] if len(confirmed) == 1 else None
        for r in group:
            if r is keeper:
                r.issues.append(asdict(Issue("warning", "duplicate_content",
                                             f"identical prices in {names}; this one matches the reference source, so it's kept")))
                if r.status == "imported":
                    r.status = "imported_with_warnings"
                continue
            reason = (f"identical prices in {names}; {keeper.file} matches the reference, so this is the copy"
                      if keeper else f"identical prices in {names}: at most one can be genuine and none is confirmed by a reference")
            r.issues.append(asdict(Issue("error", "duplicate_content", reason)))
            if r.status != "rejected":
                if r.output:
                    Path(r.output).unlink(missing_ok=True)
                r.status, r.output = "rejected", None


def write_library(results: list[ImportResult], library_path: Path, source: str, settings: dict) -> None:
    """Append/replace entries in the dataset library index (JSON)."""
    library_path.parent.mkdir(parents=True, exist_ok=True)
    lib = json.loads(library_path.read_text()) if library_path.exists() else {"datasets": []}
    keep = [d for d in lib["datasets"] if not any(d["file"] == r.file and d["symbol"] == r.symbol for r in results)]
    now = datetime.now(timezone.utc).isoformat()
    for r in results:
        keep.append({**asdict(r), "source": source, "imported_at": now, "settings": settings})
    lib["datasets"] = sorted(keep, key=lambda d: (d["source"], str(d["symbol"]), d["file"]))
    library_path.write_text(json.dumps(lib, indent=2, default=str))
