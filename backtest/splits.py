"""Chronological train / validation / test splits (config/research.yaml).

The holdout guarantee is structural, not a convention: `load_split_panel`
truncates the loaded data at the requested split's end, so a train or
validation run never has holdout bars in memory, and asking for the test
split raises unless `allow_holdout=True` is passed explicitly.

Each split is evaluated with *warm-up*: the panel starts at the beginning of
the data, so indicators at the split's first bar are computed from genuinely
earlier bars, and only the evaluation window [start, end) is scored.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

from src.data.historical import load_panel
from src.data.market_data import HistoricalDataSource

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESEARCH_CONFIG = REPO_ROOT / "config" / "research.yaml"
HOLDOUT_SPLIT = "test"


class HoldoutAccessError(RuntimeError):
    """Raised when code asks for the holdout split without explicit opt-in."""


@dataclass(frozen=True)
class DataSplit:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp | None  # exclusive; None = through end of data

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        keep = index >= self.start
        if self.end is not None:
            keep &= index < self.end
        return pd.Series(keep, index=index)


def _utc(value) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def load_research_config(path: Path | None = None) -> dict:
    with open(path or DEFAULT_RESEARCH_CONFIG) as f:
        return yaml.safe_load(f) or {}


def load_splits(config: dict | None = None) -> dict[str, DataSplit]:
    config = config if config is not None else load_research_config()
    splits = {
        name: DataSplit(name, _utc(spec["start"]), _utc(spec.get("end")))
        for name, spec in config["splits"].items()
    }
    ordered = sorted(splits.values(), key=lambda s: s.start)
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier.end is None or earlier.end > later.start:
            raise ValueError(f"splits {earlier.name!r} and {later.name!r} overlap")
    if HOLDOUT_SPLIT in splits and ordered[-1].name != HOLDOUT_SPLIT:
        raise ValueError("the holdout split must be the most recent one")
    return splits


@dataclass
class SplitPanel:
    """Close prices from the start of the data through the split's end
    (warm-up included), plus the split's evaluation window."""

    close: pd.DataFrame
    split: DataSplit
    dropped_pairs: dict[str, str]

    @property
    def eval_index(self) -> pd.DatetimeIndex:
        return self.close.index[self.split.mask(self.close.index).to_numpy()]


def load_split_panel(
    source: HistoricalDataSource,
    pairs: list[str],
    split: DataSplit,
    *,
    resample: str | None = "1h",
    allow_holdout: bool = False,
    max_staleness_days: float = 2.0,
    data_start: str = "2000-01-01",
) -> SplitPanel:
    if split.name == HOLDOUT_SPLIT and not allow_holdout:
        raise HoldoutAccessError(
            "Refusing to load the holdout (test) split. It must only be used once, "
            "for the final evaluation of an already-chosen strategy; pass "
            "allow_holdout=True (scripts: --use-holdout) if that is what this is."
        )
    # Load strictly before the split end so no later bar is ever in memory.
    end = split.end - pd.Timedelta(microseconds=1) if split.end is not None else "2100-01-01"
    close = load_panel(source, pairs, data_start, end, resample=resample).dropna(how="all")
    if split.end is not None:
        close = close[close.index < split.end]

    dropped: dict[str, str] = {}
    in_window = close[split.mask(close.index).to_numpy()]
    if in_window.empty:
        raise ValueError(f"no data inside split {split.name!r}")
    for pair in close.columns:
        series = in_window[pair].dropna()
        if series.empty:
            dropped[pair] = "no data in split"
        elif series.index[-1] < in_window.index[-1] - pd.Timedelta(days=max_staleness_days):
            # Delisted from the source exchange mid-split; the engine would
            # carry a position in it frozen at its last price forever.
            dropped[pair] = f"data ends {series.index[-1]:%Y-%m-%d}"
    close = close.drop(columns=list(dropped))
    return SplitPanel(close=close, split=split, dropped_pairs=dropped)


def load_split_fields(
    source: HistoricalDataSource,
    panel: SplitPanel,
    fields: tuple[str, ...] = ("open", "volume"),
    *,
    resample: str | None = "1h",
    data_start: str = "2000-01-01",
) -> dict[str, pd.DataFrame]:
    """Other OHLCV fields aligned to an already-loaded SplitPanel (same bars,
    same pairs, same truncation at the split end)."""
    split = panel.split
    end = split.end - pd.Timedelta(microseconds=1) if split.end is not None else "2100-01-01"
    out = {}
    for field in fields:
        frame = load_panel(source, list(panel.close.columns), data_start, end, field=field, resample=resample)
        out[field] = frame.reindex(index=panel.close.index, columns=panel.close.columns)
    return out
