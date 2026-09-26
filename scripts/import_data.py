"""Import Excel/CSV OHLC(V) files into the canonical parquet format.

    python scripts/import_data.py data/raw/bloomberg/*.xlsx --source bloomberg --tz Asia/Kolkata --labelled-by start \\
        --reference data/binance/5m
    python scripts/import_data.py prices.csv --symbol SOL/USD --map timestamp=Date_Time close=ClosePrice --mode remove

Columns are detected from common aliases (Date/Datetime/Time, Open/Open Price,
Close/Last/PX_LAST, Volume/Vol, ...); ambiguous or missing columns stop the
import with a message saying which --map to pass. Each file is validated
(duplicates, gaps, OHLC consistency, non-24/7 trading, extreme moves,
identical content across files, and optionally a cross-check against a
reference source) and handled per --mode: reject | warn | remove | repair.

Output: data/imported/<source>/<PAIR>.parquet (readable by the backtester),
data/imported/<source>/validation_report.json, and the dataset library
data/imported/library.json. data/ is gitignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.historical import ParquetDataSource  # noqa: E402
from src.data.importer import flag_duplicate_contents, import_file, write_library  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--source", default="imported", help="Name for this dataset source (output subfolder).")
    parser.add_argument("--symbol", help="Pair for a single-asset file, e.g. BTC/USD (otherwise guessed from the file name).")
    parser.add_argument("--map", nargs="*", default=[], metavar="CANONICAL=COLUMN",
                        help="Explicit column mapping, e.g. timestamp=Date close=PX_LAST.")
    parser.add_argument("--tz", default=None, help="Timezone of naive timestamps, e.g. Asia/Kolkata (default: assume UTC, with a warning).")
    parser.add_argument("--labelled-by", default="start", choices=["start", "end"], help="Whether each row's time is its bar's start or end.")
    parser.add_argument("--mode", default="warn", choices=["reject", "warn", "remove", "repair"])
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index (default: first).")
    parser.add_argument("--reference", type=Path, default=None, help="Parquet source directory to cross-check prices against.")
    parser.add_argument("--not-crypto", action="store_true", help="Don't require 24/7 trading.")
    parser.add_argument("--out-root", type=Path, default=REPO_ROOT / "data" / "imported")
    args = parser.parse_args()

    mapping = dict(m.split("=", 1) for m in args.map)
    sheet = int(args.sheet) if args.sheet and args.sheet.isdigit() else args.sheet
    reference = ParquetDataSource(args.reference) if args.reference else None
    out_dir = args.out_root / args.source
    results = []
    for f in args.files:
        results += import_file(f, out_dir, symbol=args.symbol, mapping=mapping, tz=args.tz, labelled_by=args.labelled_by,
                               mode=args.mode, sheet=sheet, reference=reference, crypto_24_7=not args.not_crypto)
    flag_duplicate_contents(results)

    settings = {"tz": args.tz, "labelled_by": args.labelled_by, "mode": args.mode, "mapping": mapping,
                "reference": str(args.reference) if args.reference else None}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation_report.json").write_text(json.dumps([asdict(r) for r in results], indent=2, default=str))
    write_library(results, args.out_root / "library.json", args.source, settings)

    for r in results:
        span = f"{r.start[:16]} -> {r.end[:16]}" if r.start else ""
        print(f"{r.symbol_mark} {r.file:32s} {str(r.symbol):10s} {r.timeframe:5s} {r.bars:>7d} bars  {span}  [{r.status}]")
        for issue in r.issues:
            if issue["severity"] in ("error", "warning") or issue["code"] == "reference":
                print(f"      {issue['severity']:7s} {issue['code']}: {issue['message']}")
    ok = sum(r.status != "rejected" for r in results)
    print(f"\n{ok}/{len(results)} imported into {out_dir}; report: {out_dir / 'validation_report.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
