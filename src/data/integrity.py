"""Data integrity — monotonic checks, gap detection, and gap repair."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import structlog
from alpaca.data.enums import Adjustment
from alpaca.data.requests import StockBarsRequest

from src.alpaca.client import AlpacaDataClient
from src.core.errors import AlpacaAuthError, AlpacaError, ConfigError
from src.data.ingest import (
    _FEED_MAP,
    _df_to_rows,
    _utcnow_iso,
    _write_bars,
    parse_timeframe,
)

logger = structlog.get_logger(__name__)

# ── Gap detection thresholds ──────────────────────────────────────────
#
# For daily bars: flag when consecutive bars are more than 4 calendar
# days apart (weekends = 3 days, so this catches holidays > 1 extra day).
#
# For intraday bars: flag when consecutive bars within the same trading
# session have a gap > 2x the expected interval.  Gaps >= 16 hours
# (960 minutes) are overnight / weekend — not data gaps.

_MAX_GAP_DAILY_DAYS = 4

_OVERNIGHT_THRESHOLD_MINUTES = 960  # ~16 hours

# Expected interval in minutes per intraday timeframe.
_INTERVAL_MINUTES: dict[str, int] = {
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
    "1Hour": 60,
}

_GAP_MULTIPLIER = 2  # flag if gap > multiplier * interval


# ── Monotonic check ──────────────────────────────────────────────────


def check_monotonic(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
) -> list[tuple[str, str]]:
    """Verify bar timestamps are strictly increasing.

    Returns a list of ``(ts_a, ts_b)`` pairs where ``ts_b <= ts_a``.
    Should always be empty given the PK ordering guarantee, but serves
    as a data-integrity sanity check.
    """
    rows = conn.execute(
        "SELECT ts FROM bars "
        "WHERE symbol = ? AND timeframe = ? "
        "ORDER BY ts",
        (symbol, timeframe),
    ).fetchall()

    violations: list[tuple[str, str]] = []
    for i in range(1, len(rows)):
        if rows[i]["ts"] <= rows[i - 1]["ts"]:
            violations.append((rows[i - 1]["ts"], rows[i]["ts"]))

    return violations


# ── Gap detection ────────────────────────────────────────────────────


def _is_intraday_gap(gap_minutes: float, timeframe: str) -> bool:
    """Return True if the gap represents missing bars within a session."""
    interval = _INTERVAL_MINUTES.get(timeframe)
    if interval is None:
        return False
    return (
        gap_minutes > interval * _GAP_MULTIPLIER
        and gap_minutes < _OVERNIGHT_THRESHOLD_MINUTES
    )


def _estimate_missing_bars(gap_minutes: float, timeframe: str) -> int:
    interval = _INTERVAL_MINUTES.get(timeframe)
    if interval is None:
        # Daily — rough estimate
        return max(1, int(gap_minutes / (24 * 60)) - 1)
    return max(1, int(gap_minutes / interval) - 1)


def detect_gaps(
    conn: sqlite3.Connection,
    symbols: list[str],
    timeframes: list[str],
) -> list[dict]:
    """Scan bars for gaps and record them in ``data_gaps``.

    Returns a list of newly inserted gap dicts.
    """
    log = logger.bind(job="detect_gaps")
    new_gaps: list[dict] = []

    for tf_str in timeframes:
        is_daily = tf_str in ("1Day", "1Week")

        for symbol in symbols:
            sym_log = log.bind(symbol=symbol, timeframe=tf_str)

            # Monotonic check first
            violations = check_monotonic(conn, symbol, tf_str)
            if violations:
                sym_log.warning("monotonic_violations", count=len(violations))

            rows = conn.execute(
                "SELECT ts FROM bars "
                "WHERE symbol = ? AND timeframe = ? "
                "ORDER BY ts",
                (symbol, tf_str),
            ).fetchall()

            if len(rows) < 2:
                continue

            for i in range(1, len(rows)):
                ts_a = datetime.fromisoformat(rows[i - 1]["ts"])
                ts_b = datetime.fromisoformat(rows[i]["ts"])
                gap_minutes = (ts_b - ts_a).total_seconds() / 60

                is_gap = False
                if is_daily:
                    gap_days = gap_minutes / (24 * 60)
                    is_gap = gap_days > _MAX_GAP_DAILY_DAYS
                else:
                    is_gap = _is_intraday_gap(gap_minutes, tf_str)

                if not is_gap:
                    continue

                gap_id = str(uuid.uuid4())
                gap_start = rows[i - 1]["ts"]
                gap_end = rows[i]["ts"]
                expected = _estimate_missing_bars(gap_minutes, tf_str)

                # Avoid duplicate gap entries
                existing = conn.execute(
                    "SELECT 1 FROM data_gaps "
                    "WHERE symbol = ? AND timeframe = ? AND gap_start = ?",
                    (symbol, tf_str, gap_start),
                ).fetchone()

                if existing:
                    continue

                conn.execute(
                    "INSERT INTO data_gaps "
                    "  (gap_id, symbol, timeframe, gap_start, gap_end, expected_bars) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (gap_id, symbol, tf_str, gap_start, gap_end, expected),
                )
                conn.commit()

                gap_info = {
                    "gap_id": gap_id,
                    "symbol": symbol,
                    "timeframe": tf_str,
                    "gap_start": gap_start,
                    "gap_end": gap_end,
                    "expected_bars": expected,
                }
                new_gaps.append(gap_info)
                sym_log.info("gap_detected", **gap_info)

    log.info("gap_detection_complete", new_gaps=len(new_gaps))
    return new_gaps


# ── Gap repair ───────────────────────────────────────────────────────


def repair_gaps(
    conn: sqlite3.Connection,
    client: AlpacaDataClient,
    feed: str = "iex",
) -> str:
    """Re-fetch bars for all open gaps and mark them closed or ignored.

    Returns the ``run_id`` UUID for this repair run.
    """
    log = logger.bind(job="repair_gaps")
    run_id = str(uuid.uuid4())
    now_iso = _utcnow_iso()
    total_written = 0

    data_feed = _FEED_MAP.get(feed.lower())
    if data_feed is None:
        raise ConfigError(f"Unknown data feed: {feed!r}. Known: {list(_FEED_MAP)}")

    open_gaps = conn.execute(
        "SELECT * FROM data_gaps WHERE status = 'open' ORDER BY symbol, timeframe, gap_start"
    ).fetchall()

    if not open_gaps:
        log.info("no_open_gaps")
        return run_id

    # Collect symbols for the ingest_runs record
    gap_symbols = sorted({g["symbol"] for g in open_gaps})
    gap_timeframes = sorted({g["timeframe"] for g in open_gaps})

    conn.execute(
        "INSERT INTO ingest_runs (run_id, started_at, status, symbols, timeframes) "
        "VALUES (?, ?, 'running', ?, ?)",
        (run_id, now_iso, ",".join(gap_symbols), ",".join(gap_timeframes)),
    )
    conn.commit()

    try:
        for gap in open_gaps:
            gap_log = log.bind(
                gap_id=gap["gap_id"],
                symbol=gap["symbol"],
                timeframe=gap["timeframe"],
            )

            try:
                tf = parse_timeframe(gap["timeframe"])
                start = datetime.fromisoformat(gap["gap_start"])
                end = datetime.fromisoformat(gap["gap_end"])

                request = StockBarsRequest(
                    symbol_or_symbols=gap["symbol"],
                    timeframe=tf,
                    start=start,
                    end=end,
                    feed=data_feed,
                    adjustment=Adjustment.ALL,
                )
                result = client.get_stock_bars(request)
                df = result.df

                if df.empty:
                    # No data available — this was a non-trading interval
                    conn.execute(
                        "UPDATE data_gaps SET status = 'ignored', resolved_at = ? "
                        "WHERE gap_id = ?",
                        (now_iso, gap["gap_id"]),
                    )
                    conn.commit()
                    gap_log.info("gap_ignored", reason="no data returned")
                    continue

                rows = _df_to_rows(df, gap["timeframe"], run_id)
                written = 0
                if rows:
                    with conn:
                        written = _write_bars(conn, rows)
                    total_written += written

                conn.execute(
                    "UPDATE data_gaps SET status = 'closed', resolved_at = ?, "
                    "repair_run_id = ? WHERE gap_id = ?",
                    (_utcnow_iso(), run_id, gap["gap_id"]),
                )
                conn.commit()
                gap_log.info("gap_repaired", bars_written=written)

            except AlpacaAuthError:
                raise
            except (AlpacaError, Exception) as exc:
                gap_log.error("gap_repair_failed", error=str(exc))
                continue

        conn.execute(
            "UPDATE ingest_runs SET status = 'done', finished_at = ?, bars_written = ? "
            "WHERE run_id = ?",
            (_utcnow_iso(), total_written, run_id),
        )
        conn.commit()
        log.info("repair_complete", run_id=run_id, total_bars_written=total_written)

    except Exception as exc:
        conn.execute(
            "UPDATE ingest_runs SET status = 'failed', finished_at = ?, error = ? "
            "WHERE run_id = ?",
            (_utcnow_iso(), str(exc), run_id),
        )
        conn.commit()
        log.error("repair_failed", run_id=run_id, error=str(exc))
        raise

    return run_id
