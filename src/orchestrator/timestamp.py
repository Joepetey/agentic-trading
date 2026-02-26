"""Evaluation timestamp resolution — bar-close-aligned eval_ts with freshness checks."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import structlog
from pydantic import BaseModel, ConfigDict

logger = structlog.get_logger(__name__)


class EvalTimestampResult(BaseModel):
    """Result of resolving the evaluation timestamp from bar data."""

    model_config = ConfigDict(frozen=True)

    eval_ts: datetime
    fresh_symbols: tuple[str, ...] = ()
    stale_symbols: tuple[str, ...] = ()
    missing_symbols: tuple[str, ...] = ()


def resolve_eval_ts(
    conn: sqlite3.Connection,
    symbols: list[str],
    timeframe: str,
    max_staleness_minutes: int | None = None,
) -> EvalTimestampResult:
    """Resolve the bar-close-aligned evaluation timestamp.

    Algorithm:
    1. For each symbol, query the latest bar timestamp.
    2. Symbols with no bars → missing_symbols.
    3. If max_staleness_minutes is set, symbols whose latest bar is
       more than that many minutes behind the *freshest* symbol → stale.
    4. eval_ts = min(latest_ts) across all fresh symbols.
    5. If no fresh symbols, eval_ts = max of any available ts, or now(UTC).

    Staleness is relative to the most recent bar across all symbols
    (not wall-clock time) so backtesting and replay still work correctly.

    Args:
        conn:                   SQLite connection with bars table.
        symbols:                Symbols to resolve across.
        timeframe:              Timeframe to query (e.g. "1Day").
        max_staleness_minutes:  If set, symbols older than this relative
                                to the freshest bar are marked stale.

    Returns:
        EvalTimestampResult with eval_ts and classification of symbols.
    """
    if not symbols:
        now = datetime.now(timezone.utc)
        logger.warning("resolve_eval_ts_empty_universe")
        return EvalTimestampResult(eval_ts=now)

    # Query latest bar timestamp per symbol
    latest_map: dict[str, datetime] = {}
    missing: list[str] = []

    for sym in symbols:
        row = conn.execute(
            "SELECT MAX(ts) AS max_ts FROM bars WHERE symbol = ? AND timeframe = ?",
            (sym, timeframe),
        ).fetchone()

        max_ts = row["max_ts"] if row and row["max_ts"] else None
        if max_ts is None:
            missing.append(sym)
        else:
            latest_map[sym] = datetime.fromisoformat(max_ts)

    # If no symbol has data at all, fallback to now(UTC)
    if not latest_map:
        now = datetime.now(timezone.utc)
        logger.warning(
            "resolve_eval_ts_no_data",
            missing=len(missing),
            fallback=now.isoformat(),
        )
        return EvalTimestampResult(
            eval_ts=now,
            missing_symbols=tuple(missing),
        )

    # Determine freshest bar across all symbols
    freshest_ts = max(latest_map.values())

    # Classify fresh vs stale
    fresh: list[str] = []
    stale: list[str] = []

    if max_staleness_minutes is not None:
        for sym, ts in latest_map.items():
            delta_minutes = (freshest_ts - ts).total_seconds() / 60
            if delta_minutes > max_staleness_minutes:
                stale.append(sym)
            else:
                fresh.append(sym)
    else:
        # No staleness check — all symbols with data are fresh
        fresh = list(latest_map.keys())

    # Resolve eval_ts
    if fresh:
        # eval_ts = min of latest timestamps across fresh symbols
        eval_ts = min(latest_map[sym] for sym in fresh)
    else:
        # All symbols stale — use the freshest available as fallback
        eval_ts = freshest_ts

    logger.info(
        "eval_ts_resolved",
        eval_ts=eval_ts.isoformat(),
        fresh=len(fresh),
        stale=len(stale),
        missing=len(missing),
        timeframe=timeframe,
    )

    return EvalTimestampResult(
        eval_ts=eval_ts,
        fresh_symbols=tuple(fresh),
        stale_symbols=tuple(stale),
        missing_symbols=tuple(missing),
    )
