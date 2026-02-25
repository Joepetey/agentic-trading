"""Tests for DataAccess, Constraints, and StrategyContext."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.strategies.context import Constraints, DataAccess, StrategyContext
from tests.conftest import insert_bars


# ── DataAccess ────────────────────────────────────────────────────────


class TestDataAccess:
    def test_get_window_respects_as_of(self, conn: sqlite3.Connection) -> None:
        closes = [100.0 + i for i in range(30)]
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", closes, start=start)

        # as_of at day 15 — should only see first 15 bars
        as_of = start + timedelta(days=14)
        dao = DataAccess(conn, as_of)
        bars = dao.get_window("AAPL", "1Day", 50)
        assert len(bars) == 15
        assert all(b.ts <= as_of for b in bars)

    def test_get_range_respects_as_of(self, conn: sqlite3.Connection) -> None:
        closes = [100.0 + i for i in range(30)]
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", closes, start=start)

        as_of = start + timedelta(days=9)
        dao = DataAccess(conn, as_of)
        bars = dao.get_range(["AAPL"], "1Day", start)
        assert len(bars) == 10
        assert all(b.ts <= as_of for b in bars)

    def test_get_latest(self, conn: sqlite3.Connection) -> None:
        closes = [100.0, 101.0, 102.0]
        insert_bars(conn, "AAPL", "1Day", closes)

        dao = DataAccess(conn, datetime(2025, 1, 1, tzinfo=timezone.utc))
        bar = dao.get_latest("AAPL", "1Day")
        assert bar.close == 102.0

    def test_get_latest_no_data_raises(self, conn: sqlite3.Connection) -> None:
        # Insert the symbol but no bars
        conn.execute("INSERT INTO symbols (symbol) VALUES ('EMPTY')")
        conn.commit()

        dao = DataAccess(conn, datetime(2025, 1, 1, tzinfo=timezone.utc))
        with pytest.raises(LookupError):
            dao.get_latest("EMPTY", "1Day")

    def test_as_of_property(self, conn: sqlite3.Connection) -> None:
        ts = datetime(2024, 6, 15, tzinfo=timezone.utc)
        dao = DataAccess(conn, ts)
        assert dao.as_of == ts


# ── Prefetch + cache ─────────────────────────────────────────────────


class TestPrefetch:
    def test_prefetch_populates_cache(self, conn: sqlite3.Connection) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", [100.0 + i for i in range(60)], start=start)
        insert_bars(conn, "MSFT", "1Day", [200.0 + i for i in range(60)], start=start)

        as_of = start + timedelta(days=59)
        dao = DataAccess(conn, as_of)
        dao.prefetch(["AAPL", "MSFT"], "1Day", lookback_bars=20)

        # get_window should serve from cache
        aapl = dao.get_window("AAPL", "1Day", 20)
        assert len(aapl) == 20
        assert aapl[-1].close == 159.0  # 100 + 59

        msft = dao.get_window("MSFT", "1Day", 20)
        assert len(msft) == 20
        assert msft[-1].close == 259.0

    def test_get_window_cache_trims_correctly(self, conn: sqlite3.Connection) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", [100.0 + i for i in range(50)], start=start)

        as_of = start + timedelta(days=49)
        dao = DataAccess(conn, as_of)
        dao.prefetch(["AAPL"], "1Day", lookback_bars=50)

        # Ask for fewer bars than were cached
        bars = dao.get_window("AAPL", "1Day", 5)
        assert len(bars) == 5
        # Should be the last 5
        assert bars[0].close == 145.0
        assert bars[-1].close == 149.0

    def test_get_window_cache_miss_falls_through(self, conn: sqlite3.Connection) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", [100.0 + i for i in range(30)], start=start)

        as_of = start + timedelta(days=29)
        dao = DataAccess(conn, as_of)
        # No prefetch — should fall through to SQLite
        bars = dao.get_window("AAPL", "1Day", 10)
        assert len(bars) == 10

    def test_get_latest_from_cache(self, conn: sqlite3.Connection) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", [100.0, 101.0, 102.0], start=start)

        as_of = start + timedelta(days=2)
        dao = DataAccess(conn, as_of)
        dao.prefetch(["AAPL"], "1Day", lookback_bars=10)

        bar = dao.get_latest("AAPL", "1Day")
        assert bar.close == 102.0

    def test_get_latest_cache_miss_falls_through(self, conn: sqlite3.Connection) -> None:
        insert_bars(conn, "AAPL", "1Day", [100.0, 101.0])

        dao = DataAccess(conn, datetime(2025, 1, 1, tzinfo=timezone.utc))
        # No prefetch
        bar = dao.get_latest("AAPL", "1Day")
        assert bar.close == 101.0

    def test_get_universe_windows(self, conn: sqlite3.Connection) -> None:
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", [100.0 + i for i in range(20)], start=start)
        insert_bars(conn, "MSFT", "1Day", [200.0 + i for i in range(20)], start=start)

        as_of = start + timedelta(days=19)
        dao = DataAccess(conn, as_of)
        dao.prefetch(["AAPL", "MSFT"], "1Day", lookback_bars=20)

        windows = dao.get_universe_windows("1Day", lookback_bars=10)
        assert set(windows.keys()) == {"AAPL", "MSFT"}
        assert len(windows["AAPL"]) == 10
        assert len(windows["MSFT"]) == 10
        # Should be the last 10 bars
        assert windows["AAPL"][-1].close == 119.0
        assert windows["MSFT"][-1].close == 219.0

    def test_get_universe_windows_empty_without_prefetch(self, conn: sqlite3.Connection) -> None:
        dao = DataAccess(conn, datetime(2025, 1, 1, tzinfo=timezone.utc))
        windows = dao.get_universe_windows("1Day", lookback_bars=10)
        assert windows == {}

    def test_prefetch_chunking(self, conn: sqlite3.Connection) -> None:
        """Verify prefetch works with more symbols than _CHUNK_SIZE."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        symbols = [f"SYM{i:03d}" for i in range(75)]
        for sym in symbols:
            insert_bars(conn, sym, "1Day", [100.0, 101.0, 102.0], start=start)

        as_of = start + timedelta(days=2)
        dao = DataAccess(conn, as_of)
        dao.prefetch(symbols, "1Day", lookback_bars=3)

        for sym in symbols:
            bars = dao.get_window(sym, "1Day", 3)
            assert len(bars) == 3, f"Expected 3 bars for {sym}, got {len(bars)}"

    def test_prefetch_respects_as_of(self, conn: sqlite3.Connection) -> None:
        """Prefetched cache should not include bars after as_of."""
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", [100.0 + i for i in range(30)], start=start)

        # as_of at day 10 — cache should only have 11 bars (days 0..10)
        as_of = start + timedelta(days=10)
        dao = DataAccess(conn, as_of)
        dao.prefetch(["AAPL"], "1Day", lookback_bars=50)

        bars = dao.get_window("AAPL", "1Day", 50)
        assert len(bars) == 11
        assert all(b.ts <= as_of for b in bars)


# ── Constraints ───────────────────────────────────────────────────────


class TestConstraints:
    def test_defaults(self) -> None:
        c = Constraints()
        assert c.max_names is None
        assert c.min_avg_volume is None
        assert c.min_price is None
        assert c.extras == {}

    def test_with_values(self) -> None:
        c = Constraints(max_names=5, min_price=10.0, extras={"custom": True})
        assert c.max_names == 5
        assert c.min_price == 10.0
        assert c.extras["custom"] is True

    def test_frozen(self) -> None:
        c = Constraints()
        with pytest.raises(Exception):
            c.max_names = 10  # type: ignore[misc]


# ── StrategyContext ───────────────────────────────────────────────────


class TestStrategyContext:
    def test_construction(self, conn: sqlite3.Connection) -> None:
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        dao = DataAccess(conn, now)
        ctx = StrategyContext(
            now_ts=now,
            universe=("AAPL", "MSFT"),
            timeframe="1Day",
            data=dao,
            config={"lookback": 20},
        )
        assert ctx.now_ts == now
        assert ctx.universe == ("AAPL", "MSFT")
        assert ctx.timeframe == "1Day"
        assert ctx.config["lookback"] == 20
        assert isinstance(ctx.constraints, Constraints)
        assert ctx.feature_version is None

    def test_frozen(self, conn: sqlite3.Connection) -> None:
        now = datetime(2024, 6, 15, tzinfo=timezone.utc)
        ctx = StrategyContext(
            now_ts=now,
            universe=("AAPL",),
            timeframe="1Day",
            data=DataAccess(conn, now),
        )
        with pytest.raises(Exception):
            ctx.timeframe = "5Min"  # type: ignore[misc]
