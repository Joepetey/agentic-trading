"""Property tests — invariants that must hold for every Signal ever emitted.

These tests run TQQQWeekly (the only concrete strategy) against a variety
of market scenarios and assert universal properties on every output signal.
"""

from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.strategies.context import DataAccess, StrategyContext
from src.strategies.signal import Side, Signal
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


def _insert_week_bars(
    conn: sqlite3.Connection,
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    start: datetime = MON,
) -> None:
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


def _collect_signals(conn: sqlite3.Connection, now_ts: datetime) -> list[Signal]:
    """Run the default TQQQWeekly and return its signals."""
    strat = TQQQWeekly()
    ctx = _make_ctx(conn, now_ts)
    return strat.run(ctx)


# ── Scenario data ───────────────────────────────────────────────────
# Each scenario is (opens, highs, lows, closes, eval_day) and
# represents a different market condition.

SCENARIOS: list[tuple[str, list[float], list[float], list[float], list[float], datetime]] = [
    # (label, opens, highs, lows, closes, eval_ts)
    (
        "normal_hold_day2",
        [100.0],
        [102.0],
        [99.0],
        [101.0],
        TUE,
    ),
    (
        "weakness_day2",
        [100.0],
        [101.0],
        [98.0],
        [99.0],
        TUE,
    ),
    (
        "stop_triggered",
        [100.0],
        [101.0],
        [97.0],
        [98.0],
        TUE,
    ),
    (
        "midweek_hold",
        [100.0, 101.0, 102.0],
        [102.0, 103.0, 104.0],
        [99.0, 100.0, 101.0],
        [101.0, 102.0, 103.0],
        THU,
    ),
    (
        "end_of_week_friday",
        [100.0, 101.0, 102.0, 101.5, 102.0],
        [102.0, 103.0, 103.0, 103.0, 103.0],
        [99.5, 100.0, 101.0, 100.5, 101.0],
        [101.0, 102.0, 102.5, 102.0, 102.5],
        FRI,
    ),
    (
        "weakness_midweek",
        [100.0, 99.5, 99.0],
        [101.0, 100.5, 100.0],
        [98.0, 98.5, 98.0],
        [99.0, 99.5, 99.0],
        THU,
    ),
    (
        "entry_no_bars",
        [],  # No bars this week — prior week only
        [],
        [],
        [],
        MON,
    ),
]


# ── Property tests ──────────────────────────────────────────────────


class TestSignalProperties:
    """Universal properties that must hold for every signal emitted."""

    @pytest.fixture(params=[s[0] for s in SCENARIOS])
    def scenario_signals(
        self, request: pytest.FixtureRequest, conn: sqlite3.Connection,
    ) -> list[Signal]:
        """Run TQQQWeekly for each scenario and return signals."""
        label = request.param
        for s in SCENARIOS:
            if s[0] == label:
                _, opens, highs, lows, closes, eval_ts = s
                break

        # Insert prior-week bars so strategy always has lookback data.
        prior_mon = MON - timedelta(weeks=1)
        insert_bars(conn, SYMBOL, TF, [50.0, 51.0, 52.0, 51.5, 52.5], start=prior_mon)

        if opens:
            _insert_week_bars(conn, opens, highs, lows, closes)

        return _collect_signals(conn, eval_ts)

    def test_no_nan_strength(self, scenario_signals: list[Signal]) -> None:
        """strength must never be NaN."""
        for sig in scenario_signals:
            assert not math.isnan(sig.strength), f"NaN strength in {sig}"

    def test_no_nan_confidence(self, scenario_signals: list[Signal]) -> None:
        """confidence must never be NaN."""
        for sig in scenario_signals:
            assert not math.isnan(sig.confidence), f"NaN confidence in {sig}"

    def test_strength_in_range(self, scenario_signals: list[Signal]) -> None:
        """strength must be in [-1, +1]."""
        for sig in scenario_signals:
            assert -1.0 <= sig.strength <= 1.0

    def test_confidence_in_range(self, scenario_signals: list[Signal]) -> None:
        """confidence must be in [0, 1]."""
        for sig in scenario_signals:
            assert 0.0 <= sig.confidence <= 1.0

    def test_horizon_bars_positive(self, scenario_signals: list[Signal]) -> None:
        """horizon_bars must be > 0."""
        for sig in scenario_signals:
            assert sig.horizon_bars > 0

    def test_strategy_id_matches(self, scenario_signals: list[Signal]) -> None:
        """All signals must have strategy_id == 'tqqq_weekly'."""
        for sig in scenario_signals:
            assert sig.strategy_id == "tqqq_weekly"

    def test_symbol_matches(self, scenario_signals: list[Signal]) -> None:
        """All signals must target the configured symbol."""
        for sig in scenario_signals:
            assert sig.symbol == SYMBOL

    def test_no_nan_prices(self, scenario_signals: list[Signal]) -> None:
        """Numeric price fields must not be NaN."""
        for sig in scenario_signals:
            if sig.stop_price is not None:
                assert not math.isnan(sig.stop_price)
            if sig.take_profit_price is not None:
                assert not math.isnan(sig.take_profit_price)
            if sig.entry_price_hint is not None:
                assert not math.isnan(sig.entry_price_hint)

    def test_stop_below_take_profit_for_longs(
        self, scenario_signals: list[Signal],
    ) -> None:
        """For LONG signals, stop_price < take_profit_price when both are set."""
        for sig in scenario_signals:
            if sig.side == Side.LONG and sig.stop_price and sig.take_profit_price:
                assert sig.stop_price < sig.take_profit_price, (
                    f"LONG signal has stop ({sig.stop_price}) >= "
                    f"take_profit ({sig.take_profit_price})"
                )

    def test_explain_non_empty_for_non_flat(
        self, scenario_signals: list[Signal],
    ) -> None:
        """Non-FLAT signals must have a non-empty explain string.

        FLAT exit signals should also have explain, but the property
        specifically targets actionable (LONG/SHORT) intent.
        """
        for sig in scenario_signals:
            assert sig.explain, f"Empty explain on {sig.side} signal for {sig.symbol}"

    def test_max_signals_per_run(self, scenario_signals: list[Signal]) -> None:
        """A single strategy must not emit more than len(universe) signals."""
        # TQQQWeekly targets one symbol, so max 1 signal.
        assert len(scenario_signals) <= 1

    def test_tags_are_strings(self, scenario_signals: list[Signal]) -> None:
        """Every tag must be a non-empty string."""
        for sig in scenario_signals:
            for tag in sig.tags:
                assert isinstance(tag, str)
                assert tag, "empty tag"

    def test_time_stop_bars_valid(self, scenario_signals: list[Signal]) -> None:
        """time_stop_bars, if set, must be > 0."""
        for sig in scenario_signals:
            if sig.time_stop_bars is not None:
                assert sig.time_stop_bars > 0

    def test_signal_is_serializable(self, scenario_signals: list[Signal]) -> None:
        """Every signal must be serializable via model_dump."""
        for sig in scenario_signals:
            d = sig.model_dump()
            assert isinstance(d, dict)
            assert d["strategy_id"] == sig.strategy_id
