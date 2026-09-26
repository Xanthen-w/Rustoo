"""Cross-check two historical sources on their overlapping bars — used to
validate the Binance backtest data against the Bloomberg exports.

    python scripts/compare_sources.py      # Bloomberg files in data/raw/bloomberg vs data/binance/5m

Reports, per pair: overlapping bars, median / p99 absolute close difference
in bps, and correlation of bar returns. Healthy agreement is a few bps and a
return correlation near 1; a timezone or bar-labelling mistake shows up as
correlation near 0.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.historical import BloombergExcelSource, ParquetDataSource  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

# Bloomberg export filename (lowercased) keyword -> Roostoo pair. Only files
# verified to contain the named coin are listed; the Sept 2026 "BNB" export
# held a weekday-only non-crypto instrument and "usdc" was a copy of XRP.
BLOOMBERG_FILES = {
    "btc": "BTC/USD",
    "eth": "ETH/USD",
    "sol": "SOL/USD",
    "xrp": "XRP/USD",
    "tron": "TRX/USD",
    "dogecoin": "DOGE/USD",
    "zcash": "ZEC/USD",
}


def discover_bloomberg_files(folder: Path) -> dict[str, Path]:
    files = {}
    for path in sorted(folder.glob("*.xlsx")):
        first_word = path.stem.lower().split()[0]
        if first_word in BLOOMBERG_FILES:
            files[BLOOMBERG_FILES[first_word]] = path
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bloomberg-dir", type=Path, default=REPO_ROOT / "data" / "raw" / "bloomberg")
    parser.add_argument("--binance-dir", type=Path, default=REPO_ROOT / "data" / "binance" / "5m")
    args = parser.parse_args()

    bloomberg = BloombergExcelSource(discover_bloomberg_files(args.bloomberg_dir))
    binance = ParquetDataSource(args.binance_dir)
    rows = []
    for pair in bloomberg.available_pairs():
        a = bloomberg.load(pair, "2000-01-01", "2100-01-01").close
        try:
            b = binance.load(pair, a.index[0], a.index[-1]).close
        except KeyError:
            rows.append({"pair": pair, "overlap_bars": 0})
            continue
        common = a.index.intersection(b.index)
        diff_bps = (a[common] / b[common] - 1).abs() * 1e4
        ra, rb = np.log(a[common]).diff(), np.log(b[common]).diff()
        rows.append(
            {
                "pair": pair,
                "overlap_bars": len(common),
                "median_diff_bps": float(diff_bps.median()),
                "p99_diff_bps": float(diff_bps.quantile(0.99)),
                "return_corr": float(ra.corr(rb)),
            }
        )
    print(pd.DataFrame(rows).set_index("pair").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
