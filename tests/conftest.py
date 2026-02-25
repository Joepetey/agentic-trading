"""Shared test fixtures for strategy tests."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.data.db import _PRAGMAS, _SCHEMA
from src.data.query import Bar


@pytest.fixture
def conn() -> sqlite3.Connection:
    """In-memory SQLite with full schema applied."""
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(_PRAGMAS)
    c.executescript(_SCHEMA)
    c.commit()
    return c


def insert_bars(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    closes: list[float],
    *,
    start: datetime | None = None,
    interval: timedelta = timedelta(days=1),
) -> list[Bar]:
    """Insert synthetic bars with given close prices.

    Open/high/low derived from close for simplicity.
    Ensures the symbol exists in the symbols table.
    Returns the list of Bar objects inserted.
    """
    if start is None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)

    # Ensure symbol exists
    conn.execute(
        "INSERT OR IGNORE INTO symbols (symbol) VALUES (?)",
        (symbol,),
    )

    bars: list[Bar] = []
    for i, close in enumerate(closes):
        ts = start + interval * i
        bar = Bar(
            symbol=symbol,
            timeframe=timeframe,
            ts=ts,
            open=close * 0.999,
            high=close * 1.005,
            low=close * 0.995,
            close=close,
            volume=1000,
            trade_count=100,
            vwap=close,
        )
        conn.execute(
            "INSERT INTO bars "
            "(symbol, timeframe, ts, open, high, low, close, volume, trade_count, vwap) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                bar.symbol,
                bar.timeframe,
                bar.ts.isoformat(),
                bar.open,
                bar.high,
                bar.low,
                bar.close,
                bar.volume,
                bar.trade_count,
                bar.vwap,
            ),
        )
        bars.append(bar)

    conn.commit()
    return bars
