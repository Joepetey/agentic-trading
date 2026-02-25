"""Tests for run_strategies() and RunResult."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

from src.core.errors import InsufficientDataError
from src.strategies.base import Strategy
from src.strategies.context import Constraints, StrategyContext
from src.strategies.runner import RunResult, StrategyRunError, run_strategies
from src.strategies.signal import Side, Signal
from tests.conftest import insert_bars


# ── Stub strategies ──────────────────────────────────────────────────


class _AlwaysLong(Strategy):
    """Emits LONG for every symbol in the universe."""

    @property
    def strategy_id(self) -> str:
        return "always_long"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=sym,
                side=Side.LONG,
                strength=0.7,
                confidence=0.8,
                horizon_bars=5,
                explain="always long",
            )
            for sym in ctx.universe
        ]


class _AlwaysFlat(Strategy):
    """Emits nothing (no signals)."""

    @property
    def strategy_id(self) -> str:
        return "always_flat"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return []


class _Exploding(Strategy):
    """Always raises an exception."""

    @property
    def strategy_id(self) -> str:
        return "exploding"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def run(self, ctx: StrategyContext) -> list[Signal]:
        raise InsufficientDataError("boom")


class _BadReturnType(Strategy):
    """Returns wrong type from run()."""

    @property
    def strategy_id(self) -> str:
        return "bad_return"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return "not a list"  # type: ignore[return-value]


class _WrongStrategyId(Strategy):
    """Returns a signal with mismatched strategy_id."""

    @property
    def strategy_id(self) -> str:
        return "wrong_id"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return [
            Signal(
                strategy_id="someone_else",
                symbol="AAPL",
                side=Side.LONG,
                strength=0.5,
                confidence=0.5,
                horizon_bars=5,
            )
        ]


# ── Tests ─────────────────────────────────────────────────────────────


class TestRunStrategies:
    def test_happy_path(self, conn: sqlite3.Connection) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_AlwaysLong()],
            universe=["AAPL", "MSFT"],
            conn=conn,
            now_ts=now,
        )
        assert isinstance(result, RunResult)
        assert len(result.signals) == 2
        assert len(result.errors) == 0
        assert result.strategies_run == 1
        assert all(s.side == Side.LONG for s in result.signals)

    def test_multiple_strategies(self, conn: sqlite3.Connection) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_AlwaysLong(), _AlwaysFlat()],
            universe=["AAPL"],
            conn=conn,
            now_ts=now,
        )
        assert len(result.signals) == 1  # AlwaysLong emits 1, AlwaysFlat emits 0
        assert result.strategies_run == 2

    def test_error_isolation(self, conn: sqlite3.Connection) -> None:
        """One strategy blowing up must not prevent others from running."""
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_Exploding(), _AlwaysLong()],
            universe=["AAPL"],
            conn=conn,
            now_ts=now,
        )
        # AlwaysLong should still produce its signal
        assert len(result.signals) == 1
        assert result.signals[0].strategy_id == "always_long"
        # Exploding should be recorded as an error
        assert len(result.errors) == 1
        assert result.errors[0].strategy_id == "exploding"
        assert result.errors[0].error_type == "InsufficientDataError"

    def test_bad_return_type_caught(self, conn: sqlite3.Connection) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_BadReturnType()],
            universe=["AAPL"],
            conn=conn,
            now_ts=now,
        )
        assert len(result.signals) == 0
        assert len(result.errors) == 1
        assert "expected list" in result.errors[0].error_message

    def test_wrong_strategy_id_caught(self, conn: sqlite3.Connection) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_WrongStrategyId()],
            universe=["AAPL"],
            conn=conn,
            now_ts=now,
        )
        assert len(result.signals) == 0
        assert len(result.errors) == 1
        assert "does not match" in result.errors[0].error_message

    def test_signals_sorted_by_strength(self, conn: sqlite3.Connection) -> None:
        """Verify signals come out sorted (highest |strength| first)."""

        class _Mixed(Strategy):
            @property
            def strategy_id(self) -> str:
                return "mixed"

            @property
            def version(self) -> str:
                return "1.0.0"

            def required_timeframes(self) -> list[str]:
                return ["1Day"]

            def required_lookback_bars(self) -> int:
                return 1

            def run(self, ctx: StrategyContext) -> list[Signal]:
                return [
                    Signal(
                        strategy_id="mixed",
                        symbol="WEAK",
                        side=Side.LONG,
                        strength=0.1,
                        confidence=0.5,
                        horizon_bars=5,
                    ),
                    Signal(
                        strategy_id="mixed",
                        symbol="STRONG",
                        side=Side.SHORT,
                        strength=-0.9,
                        confidence=0.9,
                        horizon_bars=5,
                    ),
                ]

        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_Mixed()],
            universe=["X"],  # universe doesn't matter for this strat
            conn=conn,
            now_ts=now,
        )
        assert result.signals[0].symbol == "STRONG"
        assert result.signals[1].symbol == "WEAK"

    def test_elapsed_ms_populated(self, conn: sqlite3.Connection) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_AlwaysFlat()],
            universe=["AAPL"],
            conn=conn,
            now_ts=now,
        )
        assert result.elapsed_ms >= 0

    def test_config_map_passed_to_context(self, conn: sqlite3.Connection) -> None:
        """Verify per-strategy config reaches the StrategyContext."""

        class _ConfigReader(Strategy):
            @property
            def strategy_id(self) -> str:
                return "config_reader"

            @property
            def version(self) -> str:
                return "1.0.0"

            def required_timeframes(self) -> list[str]:
                return ["1Day"]

            def required_lookback_bars(self) -> int:
                return 1

            def run(self, ctx: StrategyContext) -> list[Signal]:
                lookback = ctx.config.get("lookback", 0)
                return [
                    Signal(
                        strategy_id=self.strategy_id,
                        symbol=ctx.universe[0],
                        side=Side.FLAT,
                        strength=0.0,
                        confidence=0.0,
                        horizon_bars=1,
                        explain=f"lookback={lookback}",
                    )
                ]

        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[_ConfigReader()],
            universe=["AAPL"],
            conn=conn,
            now_ts=now,
            config_map={"config_reader": {"lookback": 42}},
        )
        assert result.signals[0].explain == "lookback=42"

    def test_empty_strategies_list(self, conn: sqlite3.Connection) -> None:
        now = datetime(2025, 1, 1, tzinfo=timezone.utc)
        result = run_strategies(
            strategies=[],
            universe=["AAPL"],
            conn=conn,
            now_ts=now,
        )
        assert result.signals == []
        assert result.errors == []
        assert result.strategies_run == 0


class TestStrategyRunError:
    def test_model_dump(self) -> None:
        err = StrategyRunError(
            strategy_id="s1",
            version="1.0.0",
            error_type="ValueError",
            error_message="oops",
        )
        d = err.model_dump()
        assert d["strategy_id"] == "s1"
        assert d["error_type"] == "ValueError"
