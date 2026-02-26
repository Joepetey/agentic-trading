"""Signal persistence — write and read strategy signals to/from SQLite."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from src.strategies.signal import Signal

logger = structlog.get_logger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalise_ts(ts: datetime | str) -> str:
    if isinstance(ts, datetime):
        return ts.isoformat()
    return ts


# ── Signal persistence ───────────────────────────────────────────────

_INSERT_SIGNAL_SQL = """
    INSERT OR REPLACE INTO signals
        (ts, symbol, strategy_id, strategy_version, params_hash,
         side, strength, confidence, horizon_bars,
         entry_type, entry_price_hint, stop_price, take_profit_price,
         time_stop_bars, invalidate, tags, explain, run_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def write_signals(
    conn: sqlite3.Connection,
    signals: list[Signal],
    *,
    run_id: str | None = None,
    eval_ts: datetime,
    params_hashes: dict[str, str],
) -> int:
    """Persist signals to the ``signals`` table.

    Args:
        conn:          SQLite connection.
        signals:       Signal objects from strategy execution.
        run_id:        Optional strategy_runs.run_id for audit linkage.
        eval_ts:       The evaluation timestamp (now_ts from runner).
        params_hashes: Mapping of strategy_id → params_hash.

    Returns:
        Number of rows written.
    """
    if not signals:
        return 0

    ts_str = _normalise_ts(eval_ts)
    rows: list[tuple[Any, ...]] = []
    for sig in signals:
        invalidate_json = (
            json.dumps([ic.model_dump() for ic in sig.invalidate])
            if sig.invalidate
            else None
        )
        tags_json = json.dumps(list(sig.tags)) if sig.tags else None

        rows.append((
            ts_str,
            sig.symbol,
            sig.strategy_id,
            "unknown",  # placeholder — overridden below
            params_hashes.get(sig.strategy_id, ""),
            sig.side.value,
            sig.strength,
            sig.confidence,
            sig.horizon_bars,
            sig.entry.value,
            sig.entry_price_hint,
            sig.stop_price,
            sig.take_profit_price,
            sig.time_stop_bars,
            invalidate_json,
            tags_json,
            sig.explain,
            run_id,
        ))

    before = conn.total_changes
    conn.executemany(_INSERT_SIGNAL_SQL, rows)
    written = conn.total_changes - before
    conn.commit()

    logger.info(
        "signals_written",
        count=written,
        total=len(signals),
        run_id=run_id,
    )
    return written


def write_signals_from_result(
    conn: sqlite3.Connection,
    signals: list[Signal],
    *,
    run_id: str | None = None,
    eval_ts: datetime,
    strategies: list[Any],
) -> int:
    """Convenience wrapper that builds params_hashes + version from strategy objects.

    Args:
        conn:       SQLite connection.
        signals:    Signal objects from strategy execution.
        run_id:     Optional strategy_runs.run_id.
        eval_ts:    The evaluation timestamp.
        strategies: Strategy instances that were run (for version + hash lookup).

    Returns:
        Number of rows written.
    """
    if not signals:
        return 0

    # Build lookups from strategy objects.
    version_map: dict[str, str] = {s.strategy_id: s.version for s in strategies}
    hash_map: dict[str, str] = {s.strategy_id: s.params_hash for s in strategies}

    ts_str = _normalise_ts(eval_ts)
    rows: list[tuple[Any, ...]] = []
    for sig in signals:
        invalidate_json = (
            json.dumps([ic.model_dump() for ic in sig.invalidate])
            if sig.invalidate
            else None
        )
        tags_json = json.dumps(list(sig.tags)) if sig.tags else None

        rows.append((
            ts_str,
            sig.symbol,
            sig.strategy_id,
            version_map.get(sig.strategy_id, "unknown"),
            hash_map.get(sig.strategy_id, ""),
            sig.side.value,
            sig.strength,
            sig.confidence,
            sig.horizon_bars,
            sig.entry.value,
            sig.entry_price_hint,
            sig.stop_price,
            sig.take_profit_price,
            sig.time_stop_bars,
            invalidate_json,
            tags_json,
            sig.explain,
            run_id,
        ))

    before = conn.total_changes
    conn.executemany(_INSERT_SIGNAL_SQL, rows)
    written = conn.total_changes - before
    conn.commit()

    logger.info(
        "signals_written",
        count=written,
        total=len(signals),
        run_id=run_id,
    )
    return written


# ── Run lifecycle ────────────────────────────────────────────────────


def create_run(
    conn: sqlite3.Connection,
    eval_ts: datetime,
    strategies: list[str],
    universe_size: int,
) -> str:
    """Insert a ``strategy_runs`` record with ``status='running'``.

    Returns the generated run_id (UUID4 hex).
    """
    run_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO strategy_runs "
        "(run_id, eval_ts, strategies, universe_size) "
        "VALUES (?, ?, ?, ?)",
        (run_id, _normalise_ts(eval_ts), ",".join(strategies), universe_size),
    )
    conn.commit()
    logger.info("strategy_run_created", run_id=run_id, eval_ts=eval_ts.isoformat())
    return run_id


def complete_run(
    conn: sqlite3.Connection,
    run_id: str,
    *,
    signals_written: int,
    errors: int,
    elapsed_ms: float,
    error: str | None = None,
) -> None:
    """Update a ``strategy_runs`` record to ``done`` (or ``failed``)."""
    status = "failed" if error else "done"
    conn.execute(
        "UPDATE strategy_runs SET "
        "finished_at = ?, status = ?, signals_written = ?, "
        "errors = ?, elapsed_ms = ?, error = ? "
        "WHERE run_id = ?",
        (_utcnow_iso(), status, signals_written, errors, elapsed_ms, error, run_id),
    )
    conn.commit()
    logger.info(
        "strategy_run_complete",
        run_id=run_id,
        status=status,
        signals_written=signals_written,
        errors=errors,
    )


# ── Query helpers ────────────────────────────────────────────────────


def get_signals(
    conn: sqlite3.Connection,
    strategy_id: str,
    *,
    start_ts: datetime | str | None = None,
    end_ts: datetime | str | None = None,
) -> list[dict[str, Any]]:
    """Read signals for a strategy, optionally filtered by time range.

    Returns list of dicts with all signal fields.
    """
    clauses = ["strategy_id = ?"]
    params: list[Any] = [strategy_id]

    if start_ts is not None:
        clauses.append("ts >= ?")
        params.append(_normalise_ts(start_ts))
    if end_ts is not None:
        clauses.append("ts <= ?")
        params.append(_normalise_ts(end_ts))

    where = " AND ".join(clauses)
    rows = conn.execute(
        f"SELECT * FROM signals WHERE {where} ORDER BY ts, symbol",
        params,
    ).fetchall()

    return [dict(row) for row in rows]


def get_latest_signals(
    conn: sqlite3.Connection,
    strategy_id: str,
) -> list[dict[str, Any]]:
    """Return signals from the most recent eval_ts for a strategy.

    Returns empty list if no signals exist.
    """
    row = conn.execute(
        "SELECT ts FROM signals WHERE strategy_id = ? ORDER BY ts DESC LIMIT 1",
        (strategy_id,),
    ).fetchone()

    if row is None:
        return []

    latest_ts = row["ts"]
    rows = conn.execute(
        "SELECT * FROM signals WHERE strategy_id = ? AND ts = ? ORDER BY symbol",
        (strategy_id, latest_ts),
    ).fetchall()

    return [dict(row) for row in rows]
