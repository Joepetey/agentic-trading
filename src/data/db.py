"""SQLite database initialisation and connection management."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = _PROJECT_ROOT / "db" / "market.db"

# ── PRAGMAs ───────────────────────────────────────────────────────────

_PRAGMAS = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -200000;
"""

# ── Schema ────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS symbols (
  symbol         TEXT PRIMARY KEY,
  asset_class    TEXT NOT NULL DEFAULT 'us_equity',
  exchange       TEXT,
  active         INTEGER NOT NULL DEFAULT 1,
  created_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  updated_at     TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS bars (
  symbol        TEXT NOT NULL,
  timeframe     TEXT NOT NULL,
  ts            TEXT NOT NULL,
  open          REAL NOT NULL,
  high          REAL NOT NULL,
  low           REAL NOT NULL,
  close         REAL NOT NULL,
  volume        INTEGER,
  trade_count   INTEGER,
  vwap          REAL,
  source        TEXT NOT NULL DEFAULT 'alpaca',
  ingest_run_id TEXT,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  PRIMARY KEY (symbol, timeframe, ts),
  FOREIGN KEY (symbol) REFERENCES symbols(symbol)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS bars_tf_ts_idx
  ON bars(timeframe, ts);

CREATE INDEX IF NOT EXISTS bars_symbol_tf_ts_desc_idx
  ON bars(symbol, timeframe, ts DESC);

CREATE TABLE IF NOT EXISTS ingest_state (
  symbol           TEXT NOT NULL,
  timeframe        TEXT NOT NULL,
  last_ingested_ts TEXT,
  last_attempt_ts  TEXT,
  last_success_ts  TEXT,
  status           TEXT NOT NULL DEFAULT 'ok',
  last_error       TEXT,
  updated_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  PRIMARY KEY (symbol, timeframe),
  FOREIGN KEY (symbol) REFERENCES symbols(symbol)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS ingest_state_status_idx
  ON ingest_state(status);

CREATE TABLE IF NOT EXISTS data_gaps (
  gap_id        TEXT PRIMARY KEY,
  symbol        TEXT NOT NULL,
  timeframe     TEXT NOT NULL,
  gap_start     TEXT NOT NULL,
  gap_end       TEXT NOT NULL,
  expected_bars INTEGER,
  status        TEXT NOT NULL DEFAULT 'open',
  detected_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  resolved_at   TEXT,
  repair_run_id TEXT,
  FOREIGN KEY (symbol) REFERENCES symbols(symbol)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS data_gaps_status_idx
  ON data_gaps(status);

CREATE TABLE IF NOT EXISTS ingest_runs (
  run_id       TEXT PRIMARY KEY,
  started_at   TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  finished_at  TEXT,
  status       TEXT NOT NULL DEFAULT 'running',
  symbols      TEXT,
  timeframes   TEXT,
  bars_written INTEGER DEFAULT 0,
  error        TEXT
) WITHOUT ROWID;
"""

# ── Public API ────────────────────────────────────────────────────────


def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    """Open a connection with PRAGMAs applied and schema ensured."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    conn.executescript(_PRAGMAS)
    conn.executescript(_SCHEMA)

    logger.debug("db_connected", path=str(path))
    return conn
