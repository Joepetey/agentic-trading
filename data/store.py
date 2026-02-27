from __future__ import annotations

from datetime import date, datetime

from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from core.types import Bar, IntradayBar
from storage.db import get_session
from storage.models import BarRow, IntradayBarRow


def upsert_bars(bars: list[Bar]) -> int:
    """Insert bars into SQLite, skipping duplicates on (ts, symbol). Returns count written."""
    if not bars:
        return 0

    session = get_session()
    try:
        count = 0
        for bar in bars:
            stmt = sqlite_insert(BarRow).values(
                ts=bar.ts,
                symbol=bar.symbol,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["ts", "symbol"],
            )
            result = session.execute(stmt)
            count += result.rowcount
        session.commit()
        return count
    finally:
        session.close()


def load_bars(symbol: str, start: date, end: date) -> list[Bar]:
    """Load bars from SQLite for a symbol in [start, end], sorted ascending by ts."""
    session = get_session()
    try:
        start_dt = datetime(start.year, start.month, start.day)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59)
        rows = (
            session.query(BarRow)
            .filter(
                BarRow.symbol == symbol,
                BarRow.ts >= start_dt,
                BarRow.ts <= end_dt,
            )
            .order_by(BarRow.ts.asc())
            .all()
        )
        return [
            Bar(
                ts=row.ts,
                symbol=row.symbol,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            for row in rows
        ]
    finally:
        session.close()


def upsert_intraday_bars(bars: list[IntradayBar], timeframe: str = "5Min") -> int:
    """Insert intraday bars into SQLite, skipping duplicates. Returns count written."""
    if not bars:
        return 0

    session = get_session()
    try:
        count = 0
        for bar in bars:
            stmt = sqlite_insert(IntradayBarRow).values(
                ts=bar.ts,
                symbol=bar.symbol,
                timeframe=timeframe,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            stmt = stmt.on_conflict_do_nothing(
                index_elements=["ts", "symbol", "timeframe"],
            )
            result = session.execute(stmt)
            count += result.rowcount
        session.commit()
        return count
    finally:
        session.close()


def load_intraday_bars(
    symbol: str, start: date, end: date, timeframe: str = "5Min",
) -> list[IntradayBar]:
    """Load intraday bars from SQLite, sorted ascending by ts."""
    session = get_session()
    try:
        start_dt = datetime(start.year, start.month, start.day)
        end_dt = datetime(end.year, end.month, end.day, 23, 59, 59)
        rows = (
            session.query(IntradayBarRow)
            .filter(
                IntradayBarRow.symbol == symbol,
                IntradayBarRow.timeframe == timeframe,
                IntradayBarRow.ts >= start_dt,
                IntradayBarRow.ts <= end_dt,
            )
            .order_by(IntradayBarRow.ts.asc())
            .all()
        )
        return [
            IntradayBar(
                ts=row.ts,
                symbol=row.symbol,
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                volume=row.volume,
            )
            for row in rows
        ]
    finally:
        session.close()
