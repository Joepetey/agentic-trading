"""Tests for the TQQQ Weekly Cycle strategy."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from src.strategies.context import DataAccess, StrategyContext
from src.strategies.signal import EntryType, Side
from src.strategies.tqqq_weekly import TQQQWeekly
from tests.conftest import insert_bars

# ── Helpers ──────────────────────────────────────────────────────────

# Monday 2024-01-08 is a known Monday (ISO week 2024-W02).
MON = datetime(2024, 1, 8, tzinfo=timezone.utc)
TUE = MON + timedelta(days=1)
WED = MON + timedelta(days=2)
THU = MON + timedelta(days=3)
FRI = MON + timedelta(days=4)

SYMBOL = "TQQQ"
TF = "1Day"


def _make_ctx(
    conn: sqlite3.Connection,
    now_ts: datetime,
    universe: tuple[str, ...] = (SYMBOL,),
) -> StrategyContext:
    dao = DataAccess(conn, now_ts)
    dao.prefetch(list(universe), TF, 10)
    return StrategyContext(
        now_ts=now_ts,
        universe=universe,
        timeframe=TF,
        data=dao,
    )


def _insert_week_bars(
    conn: sqlite3.Connection,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start: datetime = MON,
) -> None:
    """Insert daily bars with explicit OHLC for the test week."""
    conn.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (SYMBOL,))
    for i, (o, h, l, c) in enumerate(zip(opens, highs, lows, closes)):
        ts = start + timedelta(days=i)
        conn.execute(
            "INSERT INTO bars "
            "(symbol, timeframe, ts, open, high, low, close, volume, trade_count, vwap) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (SYMBOL, TF, ts.isoformat(), o, h, l, c, 1000, 100, c),
        )
    conn.commit()


# ── Tests ────────────────────────────────────────────────────────────


class TestEntry:
    """Rule 1: weekly entry at market open."""

    def test_entry_signal_no_bars_this_week(self, conn: sqlite3.Connection) -> None:
        """No bars in the current week -> emit LONG MARKET entry."""
        prior_mon = MON - timedelta(weeks=1)
        insert_bars(conn, SYMBOL, TF, [50.0, 51.0, 52.0, 51.5, 52.5], start=prior_mon)

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, MON)
        signals = strat.run(ctx)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.side == Side.LONG
        assert sig.entry == EntryType.MARKET
        assert "entry" in sig.tags

    def test_symbol_not_in_universe(self, conn: sqlite3.Connection) -> None:
        """Symbol missing from universe -> no signals."""
        insert_bars(conn, SYMBOL, TF, [50.0], start=MON)

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, MON, universe=("AAPL",))
        assert strat.run(ctx) == []


class TestNormalMode:
    """Rule 2: Type-A profit target in NORMAL mode."""

    def test_hold_normal_mode(self, conn: sqlite3.Connection) -> None:
        """Day 1 close >= entry -> LONG hold with Type-A target."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[102.0],
            lows=[99.0],
            closes=[101.0],  # close > entry -> no weakness
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, TUE)
        signals = strat.run(ctx)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.side == Side.LONG
        assert "mode:normal" in sig.tags
        expected_target = round(entry_open * (1 + 0.081), 4)
        assert sig.take_profit_price == expected_target

    def test_hold_includes_stop_price(self, conn: sqlite3.Connection) -> None:
        """Hold signal includes the stop-loss level."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[102.0],
            lows=[99.0],
            closes=[101.0],
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, TUE)
        sig = strat.run(ctx)[0]

        expected_stop = round(entry_open * (1 + (-0.015)), 4)
        assert sig.stop_price == expected_stop


class TestWeakness:
    """Rule 3 (Carlos mod): weakness detection and +2.5% target."""

    def test_weakness_detection(self, conn: sqlite3.Connection) -> None:
        """Day 1 close < entry -> mode=WEAKNESS, target switches to Carlos."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[101.0],
            lows=[98.0],
            closes=[99.0],  # close < entry -> weakness
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, TUE)
        signals = strat.run(ctx)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.side == Side.LONG
        assert "mode:weakness" in sig.tags
        expected_target = round(entry_open * (1 + 0.025), 4)
        assert sig.take_profit_price == expected_target

    def test_weakness_persists_midweek(self, conn: sqlite3.Connection) -> None:
        """Weakness from day 1 carries through to later days."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open, 99.5],
            highs=[101.0, 100.5],
            lows=[98.0, 99.0],
            closes=[99.0, 100.0],  # Day 1 close < entry -> weakness
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, WED)
        sig = strat.run(ctx)[0]
        assert "mode:weakness" in sig.tags


class TestStopLoss:
    """Rule 4 (Carlos mod): close-based stop trigger."""

    def test_stop_trigger_at_latest_close(self, conn: sqlite3.Connection) -> None:
        """Latest close <= entry*(1 - 1.3%) -> FLAT stop exit signal."""
        entry_open = 100.0
        stop_trigger = entry_open * (1 - 0.013)  # 98.7
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[101.0],
            lows=[98.0],
            closes=[stop_trigger - 0.1],
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, TUE)
        signals = strat.run(ctx)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.side == Side.FLAT
        assert sig.entry == EntryType.MARKET
        assert "stop" in sig.tags

    def test_stop_trigger_on_later_day(self, conn: sqlite3.Connection) -> None:
        """Stop triggered on day 2 close."""
        entry_open = 100.0
        stop_trigger = entry_open * (1 - 0.013)
        _insert_week_bars(
            conn,
            opens=[entry_open, 99.0],
            highs=[101.0, 99.5],
            lows=[98.0, 97.0],
            closes=[99.5, stop_trigger - 0.5],  # Day 2 close triggers
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, WED)
        signals = strat.run(ctx)

        assert len(signals) == 1
        assert signals[0].side == Side.FLAT
        assert "stop" in signals[0].tags

    def test_stop_exit_method_stop_order(self, conn: sqlite3.Connection) -> None:
        """stop_exit_method='stop_order' emits EntryType.STOP."""
        entry_open = 100.0
        stop_trigger = entry_open * (1 - 0.013)
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[101.0],
            lows=[97.0],
            closes=[stop_trigger - 0.5],
        )

        strat = TQQQWeekly(stop_exit_method="stop_order")
        ctx = _make_ctx(conn, TUE)
        signals = strat.run(ctx)

        assert len(signals) == 1
        assert signals[0].entry == EntryType.STOP
        assert signals[0].stop_price is not None

    def test_close_above_trigger_no_stop(self, conn: sqlite3.Connection) -> None:
        """Close above stop trigger level -> no stop signal."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[101.0],
            lows=[98.0],
            closes=[99.5],  # Above -1.3% threshold (98.7)
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, TUE)
        sig = strat.run(ctx)[0]
        assert sig.side == Side.LONG  # Hold, not stop


class TestEndOfWeek:
    """Rule 5: end-of-week exit."""

    def test_end_of_week_exit_friday(self, conn: sqlite3.Connection) -> None:
        """Friday bar, no other triggers -> FLAT end-of-week."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open, 101.0, 102.0, 101.5, 102.0],
            highs=[102.0, 103.0, 103.0, 103.0, 103.0],
            lows=[99.5, 100.0, 101.0, 100.5, 101.0],
            closes=[101.0, 102.0, 102.5, 102.0, 102.5],
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, FRI)
        signals = strat.run(ctx)

        assert len(signals) == 1
        sig = signals[0]
        assert sig.side == Side.FLAT
        assert "end_of_week" in sig.tags

    def test_end_of_week_with_five_bars(self, conn: sqlite3.Connection) -> None:
        """5 bars in week triggers end-of-week via bar-count check."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open, 101.0, 102.0, 101.5, 102.0],
            highs=[102.0, 103.0, 103.0, 103.0, 103.0],
            lows=[99.5, 100.0, 101.0, 100.5, 101.0],
            closes=[101.0, 102.0, 102.5, 102.0, 102.5],
        )

        strat = TQQQWeekly()
        # Saturday: not Friday (weekday != 4), but all 5 bars visible.
        sat = FRI + timedelta(days=1)
        ctx = _make_ctx(conn, sat)
        signals = strat.run(ctx)
        assert len(signals) == 1
        assert signals[0].side == Side.FLAT

    def test_stop_overrides_eow(self, conn: sqlite3.Connection) -> None:
        """Stop triggered on Friday close -> stop signal, not EOW."""
        entry_open = 100.0
        stop_trigger = entry_open * (1 - 0.013)
        _insert_week_bars(
            conn,
            opens=[entry_open, 101.0, 100.0, 99.0, 98.0],
            highs=[102.0, 102.0, 101.0, 100.0, 99.0],
            lows=[99.0, 100.0, 99.0, 98.0, 96.0],
            closes=[101.0, 101.0, 100.0, 99.0, stop_trigger - 0.5],
        )

        strat = TQQQWeekly()
        ctx = _make_ctx(conn, FRI)
        signals = strat.run(ctx)

        assert len(signals) == 1
        assert "stop" in signals[0].tags


class TestParams:
    """Configuration and metadata."""

    def test_params_and_hash(self) -> None:
        """Verify params dict and deterministic hash."""
        strat = TQQQWeekly()
        p = strat.params()
        assert p["symbol"] == "TQQQ"
        assert p["profit_target_a"] == 0.081
        assert p["profit_target_carlos"] == 0.025
        assert p["stop_trigger_close"] == -0.013
        assert p["stop_exit"] == -0.015
        assert p["stop_exit_method"] == "moo"

        strat2 = TQQQWeekly()
        assert strat.params_hash == strat2.params_hash

    def test_custom_params(self, conn: sqlite3.Connection) -> None:
        """Non-default config values flow through correctly."""
        strat = TQQQWeekly(
            symbol="TQQQ",
            profit_target_a=0.07,
            profit_target_carlos=0.03,
            stop_trigger_close=-0.02,
            stop_exit=-0.025,
        )
        assert strat.params()["profit_target_a"] == 0.07
        assert strat.strategy_id == "tqqq_weekly"
        assert strat.version == "1.0.0"
        assert strat.required_timeframes() == ["1Day"]

    def test_different_params_different_hash(self) -> None:
        """Changing a param changes the hash."""
        a = TQQQWeekly(profit_target_a=0.081)
        b = TQQQWeekly(profit_target_a=0.07)
        assert a.params_hash != b.params_hash


class TestFullWeekWalkthrough:
    """Deterministic full-week evaluation at each day boundary.

    Uses a known week of bars and asserts exact signal properties
    for every evaluation timestamp from Monday through Friday.
    """

    def _setup_week(self, conn: sqlite3.Connection) -> None:
        """Insert a normal-mode week: Mon open 100, close > open each day."""
        # Prior-week lookback data.
        prior_mon = MON - timedelta(weeks=1)
        insert_bars(conn, SYMBOL, TF, [95.0, 96.0, 97.0, 98.0, 99.0], start=prior_mon)
        # This week: steady rally, no weakness, no stop trigger.
        _insert_week_bars(
            conn,
            opens=[100.0, 101.5, 102.5, 103.0, 103.5],
            highs=[102.0, 103.0, 104.0, 104.5, 105.0],
            lows=[99.5, 101.0, 102.0, 102.5, 103.0],
            closes=[101.5, 102.5, 103.0, 103.5, 104.0],
        )

    def test_monday_entry(self, conn: sqlite3.Connection) -> None:
        """Monday with bar → hold (Mon bar already exists)."""
        self._setup_week(conn)
        sig = TQQQWeekly().run(_make_ctx(conn, MON))[0]
        # Monday close=101.5 > open=100 → NORMAL mode hold
        assert sig.side == Side.LONG
        assert "mode:normal" in sig.tags
        assert sig.horizon_bars == 4  # 5 - 1 bar = 4 remaining

    def test_tuesday_hold(self, conn: sqlite3.Connection) -> None:
        self._setup_week(conn)
        sig = TQQQWeekly().run(_make_ctx(conn, TUE))[0]
        assert sig.side == Side.LONG
        assert sig.horizon_bars == 3
        # Target based on Mon open=100 at +8.1%
        assert sig.take_profit_price == round(100.0 * 1.081, 4)

    def test_wednesday_hold(self, conn: sqlite3.Connection) -> None:
        self._setup_week(conn)
        sig = TQQQWeekly().run(_make_ctx(conn, WED))[0]
        assert sig.side == Side.LONG
        assert sig.horizon_bars == 2

    def test_thursday_hold(self, conn: sqlite3.Connection) -> None:
        self._setup_week(conn)
        sig = TQQQWeekly().run(_make_ctx(conn, THU))[0]
        assert sig.side == Side.LONG
        assert sig.horizon_bars == 1

    def test_friday_exit(self, conn: sqlite3.Connection) -> None:
        self._setup_week(conn)
        sig = TQQQWeekly().run(_make_ctx(conn, FRI))[0]
        assert sig.side == Side.FLAT
        assert "end_of_week" in sig.tags
        assert sig.entry == EntryType.MARKET
        # Explain should reference entry price
        assert "$100.00" in sig.explain

    def test_horizon_decreases_monotonically(self, conn: sqlite3.Connection) -> None:
        """horizon_bars should decrease each day Mon→Thu (Fri is exit)."""
        self._setup_week(conn)
        strat = TQQQWeekly()
        horizons = []
        for day_ts in [MON, TUE, WED, THU]:
            sig = strat.run(_make_ctx(conn, day_ts))[0]
            horizons.append(sig.horizon_bars)
        assert horizons == [4, 3, 2, 1]

    def test_stop_and_target_stable_throughout_week(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Stop and take-profit levels stay constant (anchored to Mon open)."""
        self._setup_week(conn)
        strat = TQQQWeekly()
        expected_stop = round(100.0 * (1 - 0.015), 4)
        expected_tp = round(100.0 * 1.081, 4)
        for day_ts in [MON, TUE, WED, THU]:
            sig = strat.run(_make_ctx(conn, day_ts))[0]
            assert sig.stop_price == expected_stop
            assert sig.take_profit_price == expected_tp


class TestEdgeCases:
    """Boundary conditions and edge cases."""

    def test_no_bars_at_all(self, conn: sqlite3.Connection) -> None:
        """Empty database → empty return (no crash)."""
        # Symbol exists but no bars.
        conn.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (SYMBOL,))
        conn.commit()
        strat = TQQQWeekly()
        ctx = _make_ctx(conn, MON)
        assert strat.run(ctx) == []

    def test_close_equals_entry_exactly(self, conn: sqlite3.Connection) -> None:
        """Day-1 close == entry open → NOT weakness (strictly < is required)."""
        entry_open = 100.0
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[101.0],
            lows=[99.0],
            closes=[entry_open],  # exactly equal → NORMAL
        )
        sig = TQQQWeekly().run(_make_ctx(conn, TUE))[0]
        assert "mode:normal" in sig.tags

    def test_close_equals_stop_trigger_exactly(self, conn: sqlite3.Connection) -> None:
        """Close == stop_trigger_level → triggers stop (uses <=)."""
        entry_open = 100.0
        stop_trigger = entry_open * (1 - 0.013)  # 98.7 exactly
        _insert_week_bars(
            conn,
            opens=[entry_open],
            highs=[101.0],
            lows=[97.0],
            closes=[stop_trigger],
        )
        sig = TQQQWeekly().run(_make_ctx(conn, TUE))[0]
        assert sig.side == Side.FLAT
        assert "stop" in sig.tags

    def test_custom_symbol(self, conn: sqlite3.Connection) -> None:
        """Strategy configured for a different symbol ignores TQQQ."""
        insert_bars(conn, "SQQQ", TF, [50.0, 51.0], start=MON - timedelta(weeks=1))
        conn.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", ("SQQQ",))
        conn.commit()

        strat = TQQQWeekly(symbol="SQQQ")
        ctx = _make_ctx(conn, MON, universe=("SQQQ",))
        signals = strat.run(ctx)
        assert len(signals) == 1
        assert signals[0].symbol == "SQQQ"
