"""Tests for signal persistence (src/data/signals.py)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from src.data.signals import (
    complete_run,
    create_run,
    get_latest_signals,
    get_signals,
    write_signals,
    write_signals_from_result,
)
from src.strategies.signal import (
    CompareOp,
    EntryType,
    InvalidateCondition,
    PriceField,
    Side,
    Signal,
)


# ── Helpers ──────────────────────────────────────────────────────────

EVAL_TS = datetime(2025, 1, 15, 14, 30, tzinfo=timezone.utc)
EVAL_TS_2 = datetime(2025, 1, 16, 14, 30, tzinfo=timezone.utc)


def _make_signal(
    *,
    strategy_id: str = "test_strat",
    symbol: str = "AAPL",
    side: Side = Side.LONG,
    strength: float = 0.7,
    confidence: float = 0.8,
    horizon_bars: int = 5,
    entry: EntryType = EntryType.MARKET,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
    tags: tuple[str, ...] = (),
    explain: str = "test signal",
    invalidate: tuple[InvalidateCondition, ...] = (),
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        strength=strength,
        confidence=confidence,
        horizon_bars=horizon_bars,
        entry=entry,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
        tags=tags,
        explain=explain,
        invalidate=invalidate,
    )


def _ensure_symbol(conn: sqlite3.Connection, symbol: str) -> None:
    conn.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (symbol,))
    conn.commit()


# ── Tests ────────────────────────────────────────────────────────────


class TestWriteAndReadSignals:
    def test_round_trip(self, conn: sqlite3.Connection) -> None:
        """Write signals and read them back."""
        _ensure_symbol(conn, "AAPL")
        sig = _make_signal()
        written = write_signals(
            conn,
            [sig],
            eval_ts=EVAL_TS,
            params_hashes={"test_strat": "abc123"},
        )
        assert written == 1

        rows = get_signals(conn, "test_strat")
        assert len(rows) == 1
        row = rows[0]
        assert row["symbol"] == "AAPL"
        assert row["strategy_id"] == "test_strat"
        assert row["side"] == "long"
        assert row["strength"] == 0.7
        assert row["confidence"] == 0.8
        assert row["horizon_bars"] == 5
        assert row["entry_type"] == "market"
        assert row["explain"] == "test signal"
        assert row["params_hash"] == "abc123"

    def test_multiple_signals(self, conn: sqlite3.Connection) -> None:
        """Write multiple signals for different symbols."""
        _ensure_symbol(conn, "AAPL")
        _ensure_symbol(conn, "MSFT")
        signals = [
            _make_signal(symbol="AAPL", strength=0.8),
            _make_signal(symbol="MSFT", strength=0.5),
        ]
        written = write_signals(
            conn,
            signals,
            eval_ts=EVAL_TS,
            params_hashes={"test_strat": "abc123"},
        )
        assert written == 2
        rows = get_signals(conn, "test_strat")
        assert len(rows) == 2

    def test_empty_signals(self, conn: sqlite3.Connection) -> None:
        """Writing empty list returns 0."""
        assert write_signals(
            conn, [], eval_ts=EVAL_TS, params_hashes={},
        ) == 0

    def test_replace_on_conflict(self, conn: sqlite3.Connection) -> None:
        """Same PK → overwrite (INSERT OR REPLACE)."""
        _ensure_symbol(conn, "AAPL")
        sig1 = _make_signal(strength=0.5, explain="first")
        write_signals(
            conn, [sig1], eval_ts=EVAL_TS, params_hashes={"test_strat": "v1"},
        )

        sig2 = _make_signal(strength=0.9, explain="second")
        write_signals(
            conn, [sig2], eval_ts=EVAL_TS, params_hashes={"test_strat": "v1"},
        )

        rows = get_signals(conn, "test_strat")
        assert len(rows) == 1
        assert rows[0]["strength"] == 0.9
        assert rows[0]["explain"] == "second"

    def test_prices_persisted(self, conn: sqlite3.Connection) -> None:
        """Stop and take-profit prices survive round-trip."""
        _ensure_symbol(conn, "AAPL")
        sig = _make_signal(stop_price=95.5, take_profit_price=110.0)
        write_signals(
            conn, [sig], eval_ts=EVAL_TS, params_hashes={"test_strat": "h"},
        )
        row = get_signals(conn, "test_strat")[0]
        assert row["stop_price"] == 95.5
        assert row["take_profit_price"] == 110.0

    def test_params_hash_stored(self, conn: sqlite3.Connection) -> None:
        """params_hash is persisted and queryable."""
        _ensure_symbol(conn, "AAPL")
        sig = _make_signal()
        write_signals(
            conn, [sig], eval_ts=EVAL_TS, params_hashes={"test_strat": "deadbeef01234567"},
        )
        row = get_signals(conn, "test_strat")[0]
        assert row["params_hash"] == "deadbeef01234567"


class TestTagsAndInvalidateSerialization:
    def test_tags_json_round_trip(self, conn: sqlite3.Connection) -> None:
        """Tags are serialized as JSON and can be deserialized."""
        _ensure_symbol(conn, "AAPL")
        sig = _make_signal(tags=("momentum", "breakout", "high_vol"))
        write_signals(
            conn, [sig], eval_ts=EVAL_TS, params_hashes={"test_strat": "h"},
        )
        row = get_signals(conn, "test_strat")[0]
        tags = json.loads(row["tags"])
        assert tags == ["momentum", "breakout", "high_vol"]

    def test_empty_tags_stored_as_null(self, conn: sqlite3.Connection) -> None:
        """Empty tags tuple → NULL in DB."""
        _ensure_symbol(conn, "AAPL")
        sig = _make_signal(tags=())
        write_signals(
            conn, [sig], eval_ts=EVAL_TS, params_hashes={"test_strat": "h"},
        )
        row = get_signals(conn, "test_strat")[0]
        assert row["tags"] is None

    def test_invalidate_json_round_trip(self, conn: sqlite3.Connection) -> None:
        """InvalidateConditions are serialized as JSON array."""
        _ensure_symbol(conn, "AAPL")
        ic = InvalidateCondition(field=PriceField.CLOSE, op=CompareOp.LT, value=90.0)
        sig = _make_signal(invalidate=(ic,))
        write_signals(
            conn, [sig], eval_ts=EVAL_TS, params_hashes={"test_strat": "h"},
        )
        row = get_signals(conn, "test_strat")[0]
        conditions = json.loads(row["invalidate"])
        assert len(conditions) == 1
        assert conditions[0]["field"] == "close"
        assert conditions[0]["op"] == "lt"
        assert conditions[0]["value"] == 90.0

    def test_empty_invalidate_stored_as_null(self, conn: sqlite3.Connection) -> None:
        _ensure_symbol(conn, "AAPL")
        sig = _make_signal(invalidate=())
        write_signals(
            conn, [sig], eval_ts=EVAL_TS, params_hashes={"test_strat": "h"},
        )
        row = get_signals(conn, "test_strat")[0]
        assert row["invalidate"] is None


class TestGetSignals:
    def test_filter_by_time_range(self, conn: sqlite3.Connection) -> None:
        """get_signals respects start_ts and end_ts filters."""
        _ensure_symbol(conn, "AAPL")
        write_signals(
            conn,
            [_make_signal()],
            eval_ts=EVAL_TS,
            params_hashes={"test_strat": "h"},
        )
        write_signals(
            conn,
            [_make_signal(strength=0.9)],
            eval_ts=EVAL_TS_2,
            params_hashes={"test_strat": "h"},
        )

        # Only first day
        rows = get_signals(conn, "test_strat", end_ts=EVAL_TS)
        assert len(rows) == 1
        assert rows[0]["strength"] == 0.7

        # Only second day
        rows = get_signals(conn, "test_strat", start_ts=EVAL_TS_2)
        assert len(rows) == 1
        assert rows[0]["strength"] == 0.9

        # Both
        rows = get_signals(conn, "test_strat", start_ts=EVAL_TS, end_ts=EVAL_TS_2)
        assert len(rows) == 2

    def test_filter_by_strategy_id(self, conn: sqlite3.Connection) -> None:
        """get_signals only returns signals for the specified strategy."""
        _ensure_symbol(conn, "AAPL")
        write_signals(
            conn,
            [_make_signal(strategy_id="strat_a")],
            eval_ts=EVAL_TS,
            params_hashes={"strat_a": "h"},
        )
        write_signals(
            conn,
            [_make_signal(strategy_id="strat_b")],
            eval_ts=EVAL_TS,
            params_hashes={"strat_b": "h"},
        )
        rows = get_signals(conn, "strat_a")
        assert len(rows) == 1
        assert rows[0]["strategy_id"] == "strat_a"


class TestGetLatestSignals:
    def test_returns_most_recent(self, conn: sqlite3.Connection) -> None:
        """get_latest_signals returns only the most recent eval_ts."""
        _ensure_symbol(conn, "AAPL")
        _ensure_symbol(conn, "MSFT")
        write_signals(
            conn,
            [_make_signal(symbol="AAPL", strength=0.5)],
            eval_ts=EVAL_TS,
            params_hashes={"test_strat": "h"},
        )
        write_signals(
            conn,
            [
                _make_signal(symbol="AAPL", strength=0.9),
                _make_signal(symbol="MSFT", strength=0.6),
            ],
            eval_ts=EVAL_TS_2,
            params_hashes={"test_strat": "h"},
        )

        rows = get_latest_signals(conn, "test_strat")
        assert len(rows) == 2
        symbols = {r["symbol"] for r in rows}
        assert symbols == {"AAPL", "MSFT"}
        # Should be from EVAL_TS_2
        for r in rows:
            assert r["ts"] == EVAL_TS_2.isoformat()

    def test_empty_when_no_signals(self, conn: sqlite3.Connection) -> None:
        assert get_latest_signals(conn, "nonexistent") == []


class TestRunLifecycle:
    def test_create_and_complete_run(self, conn: sqlite3.Connection) -> None:
        """Full run lifecycle: create → complete."""
        run_id = create_run(
            conn,
            eval_ts=EVAL_TS,
            strategies=["strat_a", "strat_b"],
            universe_size=10,
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 32  # UUID4 hex

        # Verify running state
        row = conn.execute(
            "SELECT * FROM strategy_runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        assert row["status"] == "running"
        assert row["eval_ts"] == EVAL_TS.isoformat()
        assert row["strategies"] == "strat_a,strat_b"
        assert row["universe_size"] == 10

        # Complete
        complete_run(
            conn,
            run_id,
            signals_written=5,
            errors=1,
            elapsed_ms=123.45,
        )
        row = conn.execute(
            "SELECT * FROM strategy_runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        assert row["status"] == "done"
        assert row["signals_written"] == 5
        assert row["errors"] == 1
        assert row["elapsed_ms"] == 123.45
        assert row["finished_at"] is not None
        assert row["error"] is None

    def test_complete_run_with_error(self, conn: sqlite3.Connection) -> None:
        """Run that fails stores the error message."""
        run_id = create_run(
            conn, eval_ts=EVAL_TS, strategies=["s1"], universe_size=5,
        )
        complete_run(
            conn,
            run_id,
            signals_written=0,
            errors=1,
            elapsed_ms=50.0,
            error="database locked",
        )
        row = conn.execute(
            "SELECT * FROM strategy_runs WHERE run_id = ?", (run_id,),
        ).fetchone()
        assert row["status"] == "failed"
        assert row["error"] == "database locked"

    def test_signals_linked_to_run(self, conn: sqlite3.Connection) -> None:
        """Signals store the run_id for audit linkage."""
        _ensure_symbol(conn, "AAPL")
        run_id = create_run(
            conn, eval_ts=EVAL_TS, strategies=["test_strat"], universe_size=1,
        )
        write_signals(
            conn,
            [_make_signal()],
            eval_ts=EVAL_TS,
            run_id=run_id,
            params_hashes={"test_strat": "h"},
        )
        row = get_signals(conn, "test_strat")[0]
        assert row["run_id"] == run_id


class TestWriteSignalsFromResult:
    def test_version_and_hash_from_strategy(self, conn: sqlite3.Connection) -> None:
        """write_signals_from_result extracts version and hash from strategy objects."""
        from tests.strategies.test_runner import _AlwaysLong

        _ensure_symbol(conn, "AAPL")

        strat = _AlwaysLong()
        sig = _make_signal(strategy_id="always_long")
        written = write_signals_from_result(
            conn,
            [sig],
            eval_ts=EVAL_TS,
            strategies=[strat],
        )
        assert written == 1

        row = get_signals(conn, "always_long")[0]
        assert row["strategy_version"] == "1.0.0"
        assert row["params_hash"] == strat.params_hash
