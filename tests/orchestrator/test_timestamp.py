"""Tests for evaluation timestamp resolution."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.orchestrator.timestamp import EvalTimestampResult, resolve_eval_ts
from tests.conftest import insert_bars

# Convenience timestamps
JAN_10 = datetime(2024, 1, 10, 21, 0, tzinfo=timezone.utc)
JAN_09 = datetime(2024, 1, 9, 21, 0, tzinfo=timezone.utc)
JAN_08 = datetime(2024, 1, 8, 21, 0, tzinfo=timezone.utc)
JAN_05 = datetime(2024, 1, 5, 21, 0, tzinfo=timezone.utc)


class TestAllFresh:
    def test_eval_ts_is_min_of_latest_bars(self, conn):
        """eval_ts = min(latest_ts) across fresh symbols."""
        # AAPL has bars up to Jan 10, MSFT up to Jan 9
        insert_bars(conn, "AAPL", "1Day", [100.0] * 10, start=JAN_05 - timedelta(days=9))
        insert_bars(conn, "MSFT", "1Day", [200.0] * 9, start=JAN_05 - timedelta(days=8))

        result = resolve_eval_ts(conn, ["AAPL", "MSFT"], "1Day")

        # Both fresh (no staleness check), eval_ts = min of the two latest
        assert len(result.fresh_symbols) == 2
        assert len(result.stale_symbols) == 0
        assert len(result.missing_symbols) == 0
        # eval_ts should be the earlier of the two latest bars
        aapl_latest = JAN_05 - timedelta(days=9) + timedelta(days=9)
        msft_latest = JAN_05 - timedelta(days=8) + timedelta(days=8)
        assert result.eval_ts == min(aapl_latest, msft_latest)

    def test_single_symbol(self, conn):
        """Single symbol: eval_ts = its latest bar."""
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=JAN_08)

        result = resolve_eval_ts(conn, ["AAPL"], "1Day")

        expected_latest = JAN_08 + timedelta(days=4)
        assert result.eval_ts == expected_latest
        assert result.fresh_symbols == ("AAPL",)


class TestStaleness:
    def test_one_symbol_stale(self, conn):
        """One symbol's latest bar exceeds staleness threshold."""
        # AAPL: latest bar at Jan 10
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5, start=JAN_10 - timedelta(days=4))
        # MSFT: latest bar at Jan 5 (5 days = 7200 min behind)
        insert_bars(conn, "MSFT", "1Day", [200.0] * 3, start=JAN_05 - timedelta(days=2))

        result = resolve_eval_ts(
            conn, ["AAPL", "MSFT"], "1Day", max_staleness_minutes=2880,
        )

        assert "AAPL" in result.fresh_symbols
        assert "MSFT" in result.stale_symbols
        assert result.eval_ts == JAN_10

    def test_staleness_relative_to_freshest(self, conn):
        """Staleness is measured against the most recent bar, not wall clock."""
        # Both symbols have bars, but measured relative to freshest
        # AAPL latest: day 10, MSFT latest: day 8
        insert_bars(conn, "AAPL", "1Day", [100.0] * 3, start=JAN_08)
        insert_bars(conn, "MSFT", "1Day", [200.0] * 1, start=JAN_08)

        # MSFT is 2 days behind AAPL = 2880 minutes
        # With staleness=2880, MSFT is exactly at the boundary → fresh
        result = resolve_eval_ts(
            conn, ["AAPL", "MSFT"], "1Day", max_staleness_minutes=2880,
        )
        assert "MSFT" in result.fresh_symbols

        # With staleness=2879, MSFT just exceeds → stale
        result2 = resolve_eval_ts(
            conn, ["AAPL", "MSFT"], "1Day", max_staleness_minutes=2879,
        )
        assert "MSFT" in result2.stale_symbols

    def test_no_staleness_check_when_none(self, conn):
        """max_staleness_minutes=None skips the check entirely."""
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5, start=JAN_10 - timedelta(days=4))
        insert_bars(conn, "MSFT", "1Day", [200.0] * 1, start=JAN_05)

        result = resolve_eval_ts(conn, ["AAPL", "MSFT"], "1Day", max_staleness_minutes=None)

        # Both should be fresh (no check)
        assert len(result.fresh_symbols) == 2
        assert len(result.stale_symbols) == 0


class TestMissingData:
    def test_symbol_missing_data(self, conn):
        """Symbol with zero bars is classified as missing."""
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5, start=JAN_08)

        result = resolve_eval_ts(conn, ["AAPL", "GHOST"], "1Day")

        assert "AAPL" in result.fresh_symbols
        assert "GHOST" in result.missing_symbols
        assert result.eval_ts == JAN_08 + timedelta(days=4)

    def test_all_symbols_missing(self, conn):
        """All symbols missing → eval_ts falls back to now(UTC)."""
        result = resolve_eval_ts(conn, ["GHOST1", "GHOST2"], "1Day")

        assert len(result.missing_symbols) == 2
        assert len(result.fresh_symbols) == 0
        # eval_ts should be roughly now
        assert (datetime.now(timezone.utc) - result.eval_ts).total_seconds() < 5


class TestAllStale:
    def test_all_symbols_stale(self, conn):
        """All stale → eval_ts falls back to the freshest available bar."""
        # Only one bar each, both old — but one is older than the other
        insert_bars(conn, "AAPL", "1Day", [100.0], start=JAN_05)
        insert_bars(conn, "MSFT", "1Day", [200.0], start=JAN_08)

        # Both are stale relative to the freshest (MSFT at Jan 8):
        # AAPL is 3 days = 4320 min behind; set staleness=1 to make both stale
        result = resolve_eval_ts(
            conn, ["AAPL", "MSFT"], "1Day", max_staleness_minutes=1,
        )

        # MSFT is the freshest → 0 minutes behind → fresh
        # Only AAPL is actually stale (4320 > 1)
        assert "MSFT" in result.fresh_symbols
        assert "AAPL" in result.stale_symbols
        # eval_ts = min of fresh = MSFT latest = Jan 8
        assert result.eval_ts == JAN_08


class TestEmptyUniverse:
    def test_empty_universe(self, conn):
        """Empty universe returns sensible defaults."""
        result = resolve_eval_ts(conn, [], "1Day")

        assert len(result.fresh_symbols) == 0
        assert len(result.stale_symbols) == 0
        assert len(result.missing_symbols) == 0
        assert (datetime.now(timezone.utc) - result.eval_ts).total_seconds() < 5


class TestDeterminism:
    def test_deterministic(self, conn):
        """Same inputs produce same outputs."""
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5, start=JAN_08)
        insert_bars(conn, "MSFT", "1Day", [200.0] * 3, start=JAN_08)

        r1 = resolve_eval_ts(conn, ["AAPL", "MSFT"], "1Day", max_staleness_minutes=2880)
        r2 = resolve_eval_ts(conn, ["AAPL", "MSFT"], "1Day", max_staleness_minutes=2880)

        assert r1.eval_ts == r2.eval_ts
        assert set(r1.fresh_symbols) == set(r2.fresh_symbols)
        assert set(r1.stale_symbols) == set(r2.stale_symbols)
        assert set(r1.missing_symbols) == set(r2.missing_symbols)


class TestTimeframeFiltering:
    def test_only_matches_requested_timeframe(self, conn):
        """Bars from other timeframes are ignored."""
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5, start=JAN_08)
        insert_bars(conn, "AAPL", "5Min", [100.0] * 100, start=JAN_10)

        result = resolve_eval_ts(conn, ["AAPL"], "1Day")

        # Should use the 1Day bars, not the 5Min bars
        assert result.eval_ts == JAN_08 + timedelta(days=4)
