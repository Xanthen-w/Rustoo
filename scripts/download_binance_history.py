"""Download Binance spot klines for Roostoo's tradable pairs into data/.

    python scripts/download_binance_history.py                      # all Roostoo pairs, 2y of 5m
    python scripts/download_binance_history.py --pairs BTC/USD ETH/USD --start 2025-01-01

Raw archives are cached in data/raw/binance/ (re-runs only fetch new files);
per-pair parquet files go to data/binance/<interval>/. data/ is gitignored —
never commit market data.
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.binance import BinanceArchiveDownloader  # noqa: E402
from src.data.universe import Universe  # noqa: E402
from src.execution.client import PublicMarketDataClient  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    yesterday = date.today() - timedelta(days=1)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", nargs="*", help="Roostoo pairs, e.g. BTC/USD. Default: every tradable pair on Roostoo.")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", type=date.fromisoformat, default=yesterday - timedelta(days=730))
    parser.add_argument("--end", type=date.fromisoformat, default=yesterday)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    pairs = args.pairs or Universe.from_exchange_info(PublicMarketDataClient()).tradable_pairs()
    cache_dir = args.data_dir / "raw" / "binance"
    output_dir = args.data_dir / "binance" / args.interval
    print(f"Downloading {len(pairs)} pairs, {args.interval} bars, {args.start} -> {args.end} into {output_dir}", flush=True)

    def run(pair: str):
        # One downloader (and HTTP session) per task: requests.Session is
        # not guaranteed thread-safe.
        return BinanceArchiveDownloader(cache_dir, output_dir, args.interval).download_pair(pair, args.start, args.end)

    reports, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, pair): pair for pair in pairs}
        for future in as_completed(futures):
            pair = futures[future]
            try:
                report = future.result()
            except Exception as exc:  # report and keep going with other pairs
                failures.append((pair, exc))
                print(f"  FAILED {pair}: {exc}")
                continue
            reports.append(report)
            span = f"{report.first:%Y-%m-%d} -> {report.last:%Y-%m-%d}" if report.rows else "no data"
            print(f"  {pair:14s} {report.rows:>8d} bars  {span}  "
                  f"(new {report.files_downloaded}, cached {report.files_cached}, missing {report.files_missing})",
                  flush=True)

    empty = sorted(r.pair for r in reports if r.rows == 0)
    print(f"\nDone: {sum(1 for r in reports if r.rows)} pairs with data, {len(empty)} without, {len(failures)} failed.")
    if empty:
        print("No Binance data for:", ", ".join(empty))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
