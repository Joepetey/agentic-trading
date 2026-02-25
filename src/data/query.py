"""Read-only data access layer — the only way strategies should read market data."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import datetime, timedelta, timezone

from src.core.errors import StaleDataError


# ── Bar dataclass ─────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class Bar:
    """Immutable representation of a single OHLCV bar."""

    symbol: str
    timeframe: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int | None
    trade_count: int | None
    vwap: float | None


# ── Helpers ───────────────────────────────────────────────────────────


def _row_to_bar(row: sqlite3.Row) -> Bar:
    return Bar(
        symbol=row["symbol"],
        timeframe=row["timeframe"],
        ts=datetime.fromisoformat(row["ts"]),
        open=row["open"],
        high=row["high"],
        low=row["low"],
        close=row["close"],
        volume=row["volume"],
        trade_count=row["trade_count"],
        vwap=row["vwap"],
    )


def _normalise_ts(ts: datetime | str) -> str:
    """Accept a datetime or ISO-8601 string; return the ISO-8601 string."""
    if isinstance(ts, datetime):
        return ts.isoformat()
    return ts


# ── Query functions ───────────────────────────────────────────────────


def get_latest(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    *,
    max_staleness: timedelta | None = None,
) -> Bar:
    """Return the most recent bar for a (symbol, timeframe).

    Raises ``LookupError`` if no bars exist.
    Raises ``StaleDataError`` if the bar is older than *max_staleness*.
    """
    row = conn.execute(
        "SELECT * FROM bars "
        "WHERE symbol = ? AND timeframe = ? "
        "ORDER BY ts DESC LIMIT 1",
        (symbol, timeframe),
    ).fetchone()

    if row is None:
        raise LookupError(f"No bars for {symbol}/{timeframe}")

    bar = _row_to_bar(row)

    if max_staleness is not None:
        cutoff = datetime.now(timezone.utc) - max_staleness
        bar_ts = bar.ts if bar.ts.tzinfo else bar.ts.replace(tzinfo=timezone.utc)
        if bar_ts < cutoff:
            raise StaleDataError(
                f"{symbol}/{timeframe} latest bar {bar.ts.isoformat()} "
                f"is older than {max_staleness}"
            )

    return bar


def get_window(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    end_ts: datetime | str,
    lookback_bars: int,
) -> list[Bar]:
    """Return the last *lookback_bars* bars ending at or before *end_ts*.

    Results are sorted in ascending ts order (monotonic).
    Returns ``[]`` if no bars match.
    """
    rows = conn.execute(
        "SELECT * FROM bars "
        "WHERE symbol = ? AND timeframe = ? AND ts <= ? "
        "ORDER BY ts DESC LIMIT ?",
        (symbol, timeframe, _normalise_ts(end_ts), lookback_bars),
    ).fetchall()

    return [_row_to_bar(r) for r in reversed(rows)]


def get_range(
    conn: sqlite3.Connection,
    symbols: list[str],
    timeframe: str,
    start_ts: datetime | str,
    end_ts: datetime | str,
) -> list[Bar]:
    """Return bars for multiple symbols in a time range.

    Results are sorted by ``(symbol, ts)`` ascending.
    """
    placeholders = ",".join("?" for _ in symbols)
    rows = conn.execute(
        f"SELECT * FROM bars "
        f"WHERE symbol IN ({placeholders}) AND timeframe = ? "
        f"AND ts >= ? AND ts <= ? "
        f"ORDER BY symbol, ts",
        [*symbols, timeframe, _normalise_ts(start_ts), _normalise_ts(end_ts)],
    ).fetchall()

    return [_row_to_bar(r) for r in rows]
