"""PortfolioIntent persistence — write intents to SQLite for audit/replay."""

from __future__ import annotations

import json
import sqlite3
import uuid

import structlog

from src.orchestrator.models import PortfolioIntent

logger = structlog.get_logger(__name__)

# ── Schema extension ─────────────────────────────────────────────────

INTENT_SCHEMA = """
CREATE TABLE IF NOT EXISTS portfolio_intents (
    intent_id        TEXT PRIMARY KEY,
    as_of_ts         TEXT NOT NULL,
    strategy_run_id  TEXT,
    sizing_method    TEXT NOT NULL,
    portfolio_equity REAL NOT NULL,
    portfolio_cash   REAL NOT NULL,
    universe_included TEXT NOT NULL,
    universe_excluded TEXT,
    signals_used     TEXT,
    signals_dropped  TEXT,
    targets          TEXT NOT NULL,
    trade_allowed    INTEGER NOT NULL DEFAULT 1,
    elapsed_ms       REAL,
    explain          TEXT NOT NULL DEFAULT '',
    created_at       TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS intents_as_of_idx
    ON portfolio_intents(as_of_ts DESC);
"""


def ensure_intent_schema(conn: sqlite3.Connection) -> None:
    """Create the portfolio_intents table if it doesn't exist."""
    conn.executescript(INTENT_SCHEMA)


def generate_intent_id() -> str:
    """Generate a UUID4 hex string for intent identification."""
    return uuid.uuid4().hex


def write_intent(conn: sqlite3.Connection, intent: PortfolioIntent) -> None:
    """Persist a PortfolioIntent to SQLite."""
    conn.execute(
        """INSERT OR REPLACE INTO portfolio_intents
           (intent_id, as_of_ts, strategy_run_id, sizing_method,
            portfolio_equity, portfolio_cash,
            universe_included, universe_excluded,
            signals_used, signals_dropped, targets,
            trade_allowed, elapsed_ms, explain)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            intent.intent_id,
            intent.as_of_ts.isoformat(),
            intent.strategy_run_id,
            intent.sizing_method.value,
            intent.portfolio_state.equity,
            intent.portfolio_state.cash,
            json.dumps(list(intent.universe.included)),
            json.dumps([e.model_dump() for e in intent.universe.excluded]) if intent.universe.excluded else None,
            json.dumps([m.model_dump() for m in intent.signals_used]) if intent.signals_used else None,
            json.dumps([d.model_dump() for d in intent.signals_dropped]) if intent.signals_dropped else None,
            json.dumps([t.model_dump() for t in intent.targets]),
            int(intent.trade_allowed),
            intent.elapsed_ms,
            intent.explain,
        ),
    )
    conn.commit()
    logger.info(
        "intent_written",
        intent_id=intent.intent_id,
        as_of_ts=intent.as_of_ts.isoformat(),
        targets=len(intent.targets),
    )


def get_latest_intent(conn: sqlite3.Connection) -> dict | None:
    """Return the most recent portfolio_intent as a dict, or None."""
    row = conn.execute(
        "SELECT * FROM portfolio_intents ORDER BY as_of_ts DESC LIMIT 1"
    ).fetchone()
    return dict(row) if row else None


def get_intent(conn: sqlite3.Connection, intent_id: str) -> dict | None:
    """Return a specific portfolio_intent by ID."""
    row = conn.execute(
        "SELECT * FROM portfolio_intents WHERE intent_id = ?",
        (intent_id,),
    ).fetchone()
    return dict(row) if row else None
