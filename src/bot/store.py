"""Persistent audit trail + bot state (SQLite, stdlib only).

The competition screens on trade-log integrity, and the organizers ask
teams to record every trade, performance, and the success/failure of every
API request (docs/COMPETITION_RULES.md). Everything the bot decides or does
lands here, and survives restarts:

- api_calls: one row per HTTP attempt to Roostoo
- decisions: one row per hourly decision (targets, whether scheduled)
- orders:    every planned order and its outcome (fill, fee, role, error)
- equity:    periodic portfolio value snapshots
- kv:        small bot state (e.g. date of the last scheduled rebalance)

The database lives under data/ (gitignored).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_calls (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, method TEXT, path TEXT,
    http_status INTEGER, success INTEGER, error TEXT, elapsed_ms REAL
);
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, bar_close TEXT, equity REAL,
    targets TEXT, current TEXT, scheduled INTEGER, planned_orders INTEGER, note TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, mode TEXT, pair TEXT, side TEXT,
    quantity REAL, est_price REAL, reason TEXT, status TEXT, order_id TEXT,
    filled_qty REAL, avg_price REAL, fee REAL, role TEXT, error TEXT, raw TEXT
);
CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, equity REAL, cash REAL, positions TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders(ts);
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts);
"""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(ts: datetime | None) -> str:
    return (ts or utc_now()).astimezone(timezone.utc).isoformat()


class BotStore:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL") if str(path) != ":memory:" else None
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def _exec(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    # -- writes --------------------------------------------------------------

    def record_api_call(self, call: dict, ts: datetime | None = None) -> None:
        self._exec(
            "INSERT INTO api_calls (ts, method, path, http_status, success, error, elapsed_ms) VALUES (?,?,?,?,?,?,?)",
            (_iso(ts), call.get("method"), call.get("path"), call.get("http_status"),
             int(bool(call.get("success"))), call.get("error", ""), call.get("elapsed_ms")),
        )

    def record_decision(self, *, bar_close, equity: float, targets: dict, current: dict, scheduled: bool,
                        planned_orders: int, note: str = "", ts: datetime | None = None) -> None:
        self._exec(
            "INSERT INTO decisions (ts, bar_close, equity, targets, current, scheduled, planned_orders, note) VALUES (?,?,?,?,?,?,?,?)",
            (_iso(ts), str(bar_close), equity, json.dumps(targets), json.dumps(current), int(scheduled), planned_orders, note),
        )

    def record_order(self, record: dict, ts: datetime | None = None) -> None:
        self._exec(
            "INSERT INTO orders (ts, mode, pair, side, quantity, est_price, reason, status, order_id, filled_qty,"
            " avg_price, fee, role, error, raw) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (_iso(ts), record.get("mode"), record.get("pair"), record.get("side"), record.get("quantity"),
             record.get("est_price"), record.get("reason"), record.get("status"),
             None if record.get("order_id") is None else str(record.get("order_id")),
             record.get("filled_qty"), record.get("avg_price"), record.get("fee"), record.get("role"),
             record.get("error"), json.dumps(record.get("raw")) if record.get("raw") is not None else None),
        )

    def record_equity(self, equity: float, cash: float, positions: dict, ts: datetime | None = None) -> None:
        self._exec("INSERT INTO equity (ts, equity, cash, positions) VALUES (?,?,?,?)",
                   (_iso(ts), equity, cash, json.dumps(positions)))

    def set(self, key: str, value) -> None:
        self._exec("INSERT INTO kv (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, json.dumps(value)))

    # -- reads ---------------------------------------------------------------

    def get(self, key: str, default=None):
        row = self._exec("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def orders(self, limit: int = 1000) -> list[dict]:
        cur = self._exec("SELECT * FROM orders ORDER BY id DESC LIMIT ?", (limit,))
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def count(self, table: str) -> int:
        if table not in {"api_calls", "decisions", "orders", "equity"}:
            raise ValueError(table)
        return self._exec(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def filled_order_days(self) -> list[str]:
        """UTC dates with at least one filled order — the competition's
        'active trading days'."""
        rows = self._exec(
            "SELECT DISTINCT substr(ts, 1, 10) FROM orders WHERE status IN ('FILLED', 'PARTIALLY_FILLED', 'SIMULATED') ORDER BY 1"
        ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
