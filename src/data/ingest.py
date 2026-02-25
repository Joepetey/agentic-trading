"""Ingestion pipeline — universe sync, historical backfill, and incremental update."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pandas as pd
import structlog
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.alpaca.client import AlpacaDataClient
from src.core.errors import AlpacaAuthError, AlpacaError, ConfigError

logger = structlog.get_logger(__name__)

# ── Timeframe mapping ─────────────────────────────────────────────────

_TF_MAP: dict[str, TimeFrame] = {
    "1Min": TimeFrame(1, TimeFrameUnit.Minute),
    "5Min": TimeFrame(5, TimeFrameUnit.Minute),
    "15Min": TimeFrame(15, TimeFrameUnit.Minute),
    "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
    "1Day": TimeFrame(1, TimeFrameUnit.Day),
    "1Week": TimeFrame(1, TimeFrameUnit.Week),
}

# Chunk sizes (calendar days) per timeframe for backfill requests.
_CHUNK_DAYS: dict[str, int] = {
    "1Day": 365,
    "1Week": 730,
    "1Hour": 90,
    "15Min": 60,
    "5Min": 30,
    "1Min": 7,
}
_DEFAULT_CHUNK_DAYS = 90

_FEED_MAP: dict[str, DataFeed] = {
    "iex": DataFeed.IEX,
    "sip": DataFeed.SIP,
}

_INSERT_BAR_SQL = """
    INSERT OR IGNORE INTO bars
        (symbol, timeframe, ts, open, high, low, close,
         volume, trade_count, vwap, source, ingest_run_id)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


# ── Helpers ───────────────────────────────────────────────────────────


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def parse_timeframe(tf_str: str) -> TimeFrame:
    """Convert a config timeframe string to an Alpaca ``TimeFrame``."""
    if tf_str not in _TF_MAP:
        raise ConfigError(f"Unknown timeframe: {tf_str!r}. Known: {list(_TF_MAP)}")
    return _TF_MAP[tf_str]


def _time_chunks(
    start: datetime, end: datetime, chunk_days: int
) -> list[tuple[datetime, datetime]]:
    """Split ``[start, end)`` into consecutive windows of *chunk_days*."""
    chunks: list[tuple[datetime, datetime]] = []
    cursor = start
    delta = timedelta(days=chunk_days)
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _df_to_rows(
    df: pd.DataFrame, timeframe_str: str, run_id: str
) -> list[tuple]:
    """Flatten an alpaca bars DataFrame into insert-ready tuples."""
    if df.empty:
        return []
    rows: list[tuple] = []
    df_reset = df.reset_index()
    for _, r in df_reset.iterrows():
        ts = r["timestamp"]
        ts_str = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
        rows.append((
            r["symbol"],
            timeframe_str,
            ts_str,
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            int(r["volume"]) if pd.notna(r.get("volume")) else None,
            int(r["trade_count"]) if pd.notna(r.get("trade_count")) else None,
            float(r["vwap"]) if pd.notna(r.get("vwap")) else None,
            "alpaca",
            run_id,
        ))
    return rows


def _write_bars(conn: sqlite3.Connection, rows: list[tuple]) -> int:
    """Batch ``INSERT OR IGNORE`` into bars; return count of new rows."""
    if not rows:
        return 0
    before = conn.total_changes
    conn.executemany(_INSERT_BAR_SQL, rows)
    return conn.total_changes - before


def _upsert_ingest_state(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    *,
    status: str = "ok",
    last_ingested_ts: str | None = None,
    attempt_ts: str | None = None,
    success_ts: str | None = None,
    error: str | None = None,
) -> None:
    """Insert or update the ``ingest_state`` watermark for a (symbol, timeframe) pair."""
    now = _utcnow_iso()
    existing = conn.execute(
        "SELECT 1 FROM ingest_state WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
    ).fetchone()

    if existing:
        conn.execute(
            "UPDATE ingest_state SET "
            "  last_ingested_ts = COALESCE(?, last_ingested_ts), "
            "  last_attempt_ts  = COALESCE(?, last_attempt_ts), "
            "  last_success_ts  = COALESCE(?, last_success_ts), "
            "  status     = ?, "
            "  last_error = ?, "
            "  updated_at = ? "
            "WHERE symbol = ? AND timeframe = ?",
            (last_ingested_ts, attempt_ts, success_ts, status, error, now,
             symbol, timeframe),
        )
    else:
        conn.execute(
            "INSERT INTO ingest_state "
            "  (symbol, timeframe, last_ingested_ts, last_attempt_ts, "
            "   last_success_ts, status, last_error, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (symbol, timeframe, last_ingested_ts, attempt_ts, success_ts,
             status, error, now),
        )
    conn.commit()


# ── Job 1: Universe bootstrap ────────────────────────────────────────


def sync_universe(
    conn: sqlite3.Connection,
    config_symbols: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """Reconcile the ``symbols`` table with the config universe.

    Returns (added, reactivated, deactivated) symbol lists.
    """
    log = logger.bind(job="sync_universe")
    now = _utcnow_iso()
    config_set = {s.upper() for s in config_symbols}

    rows = conn.execute("SELECT symbol, active FROM symbols").fetchall()
    db_map = {r["symbol"]: r["active"] for r in rows}

    added: list[str] = []
    reactivated: list[str] = []
    deactivated: list[str] = []

    with conn:
        # Insert new symbols
        for sym in sorted(config_set - set(db_map)):
            conn.execute(
                "INSERT INTO symbols (symbol, active, created_at, updated_at) "
                "VALUES (?, 1, ?, ?)",
                (sym, now, now),
            )
            added.append(sym)

        # Reactivate symbols back in config
        for sym in sorted(config_set & set(db_map)):
            if db_map[sym] == 0:
                conn.execute(
                    "UPDATE symbols SET active = 1, updated_at = ? WHERE symbol = ?",
                    (now, sym),
                )
                reactivated.append(sym)

        # Deactivate symbols removed from config
        for sym in sorted(set(db_map) - config_set):
            if db_map[sym] == 1:
                conn.execute(
                    "UPDATE symbols SET active = 0, updated_at = ? WHERE symbol = ?",
                    (now, sym),
                )
                deactivated.append(sym)

    log.info(
        "universe_synced",
        added=added,
        reactivated=reactivated,
        deactivated=deactivated,
    )
    return added, reactivated, deactivated


# ── Job 2: Historical backfill ────────────────────────────────────────


def backfill_bars(
    conn: sqlite3.Connection,
    client: AlpacaDataClient,
    symbols: list[str],
    timeframes: list[str],
    lookback: dict[str, int],
    feed: str = "iex",
) -> str:
    """Backfill historical bars and update watermarks.

    Returns the ``run_id`` UUID for this ingest run.
    """
    log = logger.bind(job="backfill")
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    now_iso = _utcnow_iso()
    total_written = 0

    data_feed = _FEED_MAP.get(feed.lower())
    if data_feed is None:
        raise ConfigError(f"Unknown data feed: {feed!r}. Known: {list(_FEED_MAP)}")

    # Record run start
    conn.execute(
        "INSERT INTO ingest_runs (run_id, started_at, status, symbols, timeframes) "
        "VALUES (?, ?, 'running', ?, ?)",
        (run_id, now_iso, ",".join(symbols), ",".join(timeframes)),
    )
    conn.commit()

    try:
        for tf_str in timeframes:
            tf = parse_timeframe(tf_str)
            days = lookback.get(tf_str, 730)
            chunk_days = _CHUNK_DAYS.get(tf_str, _DEFAULT_CHUNK_DAYS)
            start = now - timedelta(days=days)

            for symbol in symbols:
                sym_log = log.bind(symbol=symbol, timeframe=tf_str)
                sym_written = 0
                max_ts: str | None = None

                _upsert_ingest_state(
                    conn, symbol, tf_str, status="backfilling", attempt_ts=now_iso,
                )

                chunks = _time_chunks(start, now, chunk_days)
                sym_log.info("backfill_start", chunks=len(chunks), lookback_days=days)

                try:
                    for chunk_start, chunk_end in chunks:
                        request = StockBarsRequest(
                            symbol_or_symbols=symbol,
                            timeframe=tf,
                            start=chunk_start,
                            end=chunk_end,
                            feed=data_feed,
                            adjustment=Adjustment.ALL,
                        )
                        result = client.get_stock_bars(request)
                        df = result.df

                        rows = _df_to_rows(df, tf_str, run_id)
                        written = 0
                        if rows:
                            with conn:
                                written = _write_bars(conn, rows)
                            sym_written += written

                            chunk_max = max(r[2] for r in rows)
                            if max_ts is None or chunk_max > max_ts:
                                max_ts = chunk_max

                        sym_log.debug(
                            "chunk_done",
                            chunk_start=chunk_start.isoformat(),
                            chunk_end=chunk_end.isoformat(),
                            bars_in_chunk=len(rows),
                            written=written,
                        )

                    _upsert_ingest_state(
                        conn, symbol, tf_str,
                        status="ok",
                        last_ingested_ts=max_ts,
                        success_ts=_utcnow_iso(),
                    )
                    total_written += sym_written
                    sym_log.info("backfill_symbol_done", bars_written=sym_written)

                except AlpacaAuthError:
                    raise
                except AlpacaError as exc:
                    sym_log.error("backfill_symbol_failed", error=str(exc))
                    _upsert_ingest_state(
                        conn, symbol, tf_str,
                        status="error",
                        error=str(exc),
                    )

        # Mark run done
        conn.execute(
            "UPDATE ingest_runs SET status = 'done', finished_at = ?, bars_written = ? "
            "WHERE run_id = ?",
            (_utcnow_iso(), total_written, run_id),
        )
        conn.commit()
        log.info("backfill_complete", run_id=run_id, total_bars_written=total_written)

    except Exception as exc:
        conn.execute(
            "UPDATE ingest_runs SET status = 'failed', finished_at = ?, error = ? "
            "WHERE run_id = ?",
            (_utcnow_iso(), str(exc), run_id),
        )
        conn.commit()
        log.error("backfill_failed", run_id=run_id, error=str(exc))
        raise

    return run_id


# ── Job 3: Incremental update ────────────────────────────────────────


def incremental_update(
    conn: sqlite3.Connection,
    client: AlpacaDataClient,
    symbols: list[str],
    timeframes: list[str],
    feed: str = "iex",
) -> str:
    """Fetch bars since each symbol's watermark and update the DB.

    Batches all symbols into a single API call per timeframe using
    ``min(last_ingested_ts)`` as the start.  ``INSERT OR IGNORE``
    handles overlap for symbols whose watermark is further ahead.

    Returns the ``run_id`` UUID for this ingest run.
    """
    log = logger.bind(job="incremental")
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    now_iso = _utcnow_iso()
    total_written = 0

    data_feed = _FEED_MAP.get(feed.lower())
    if data_feed is None:
        raise ConfigError(f"Unknown data feed: {feed!r}. Known: {list(_FEED_MAP)}")

    conn.execute(
        "INSERT INTO ingest_runs (run_id, started_at, status, symbols, timeframes) "
        "VALUES (?, ?, 'running', ?, ?)",
        (run_id, now_iso, ",".join(symbols), ",".join(timeframes)),
    )
    conn.commit()

    try:
        for tf_str in timeframes:
            tf = parse_timeframe(tf_str)
            tf_log = log.bind(timeframe=tf_str)

            # Read watermarks for all symbols in this timeframe
            placeholders = ",".join("?" for _ in symbols)
            rows = conn.execute(
                f"SELECT symbol, last_ingested_ts FROM ingest_state "
                f"WHERE timeframe = ? AND symbol IN ({placeholders})",
                [tf_str, *symbols],
            ).fetchall()
            watermarks = {r["symbol"]: r["last_ingested_ts"] for r in rows}

            # Earliest watermark across all symbols; fall back to 1 day ago
            ts_values = [v for v in watermarks.values() if v]
            if ts_values:
                start = datetime.fromisoformat(min(ts_values))
            else:
                start = now - timedelta(days=1)

            tf_log.info(
                "incremental_start",
                symbols=len(symbols),
                start=start.isoformat(),
            )

            # Single batched API call for all symbols
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbols,
                    timeframe=tf,
                    start=start,
                    end=now,
                    feed=data_feed,
                    adjustment=Adjustment.ALL,
                )
                result = client.get_stock_bars(request)
                df = result.df
            except AlpacaAuthError:
                raise
            except AlpacaError as exc:
                tf_log.error("incremental_fetch_failed", error=str(exc))
                for sym in symbols:
                    _upsert_ingest_state(
                        conn, sym, tf_str,
                        status="error",
                        attempt_ts=now_iso,
                        error=str(exc),
                    )
                continue

            if df.empty:
                tf_log.info("incremental_no_new_bars")
                for sym in symbols:
                    _upsert_ingest_state(
                        conn, sym, tf_str,
                        status="ok",
                        attempt_ts=now_iso,
                        success_ts=now_iso,
                    )
                continue

            # Process per symbol
            df_reset = df.reset_index()
            for sym in symbols:
                sym_log = tf_log.bind(symbol=sym)
                sym_df = df_reset[df_reset["symbol"] == sym]

                if sym_df.empty:
                    _upsert_ingest_state(
                        conn, sym, tf_str,
                        status="ok",
                        attempt_ts=now_iso,
                        success_ts=now_iso,
                    )
                    continue

                try:
                    # Re-index so _df_to_rows works (expects symbol+timestamp index)
                    rows_data = _df_to_rows(
                        sym_df.set_index(["symbol", "timestamp"]),
                        tf_str,
                        run_id,
                    )
                    written = 0
                    if rows_data:
                        with conn:
                            written = _write_bars(conn, rows_data)
                        total_written += written

                    max_ts = max(r[2] for r in rows_data) if rows_data else None
                    _upsert_ingest_state(
                        conn, sym, tf_str,
                        status="ok",
                        last_ingested_ts=max_ts,
                        attempt_ts=now_iso,
                        success_ts=now_iso,
                    )
                    sym_log.debug("incremental_symbol_done", written=written)

                except Exception as exc:
                    sym_log.error("incremental_symbol_failed", error=str(exc))
                    _upsert_ingest_state(
                        conn, sym, tf_str,
                        status="error",
                        attempt_ts=now_iso,
                        error=str(exc),
                    )

            tf_log.info("incremental_timeframe_done")

        conn.execute(
            "UPDATE ingest_runs SET status = 'done', finished_at = ?, bars_written = ? "
            "WHERE run_id = ?",
            (_utcnow_iso(), total_written, run_id),
        )
        conn.commit()
        log.info("incremental_complete", run_id=run_id, total_bars_written=total_written)

    except Exception as exc:
        conn.execute(
            "UPDATE ingest_runs SET status = 'failed', finished_at = ?, error = ? "
            "WHERE run_id = ?",
            (_utcnow_iso(), str(exc), run_id),
        )
        conn.commit()
        log.error("incremental_failed", run_id=run_id, error=str(exc))
        raise

    return run_id
