"""Tests for Strategy ABC."""

from __future__ import annotations

import pytest

from src.strategies.base import Strategy
from src.strategies.context import StrategyContext
from src.strategies.signal import Signal


# ── ABC enforcement ───────────────────────────────────────────────────


class TestStrategyABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_incomplete_subclass_raises(self) -> None:
        class Incomplete(Strategy):
            @property
            def strategy_id(self) -> str:
                return "incomplete"

            @property
            def version(self) -> str:
                return "1.0.0"

            # Missing: required_timeframes, required_lookback_bars, run

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_works(self) -> None:
        class Complete(Strategy):
            @property
            def strategy_id(self) -> str:
                return "complete"

            @property
            def version(self) -> str:
                return "1.0.0"

            def required_timeframes(self) -> list[str]:
                return ["1Day"]

            def required_lookback_bars(self) -> int:
                return 10

            def run(self, ctx: StrategyContext) -> list[Signal]:
                return []

        strat = Complete()
        assert strat.strategy_id == "complete"
        assert strat.version == "1.0.0"
        assert strat.required_timeframes() == ["1Day"]
        assert strat.required_lookback_bars() == 10

    def test_healthcheck_defaults_to_true(self) -> None:
        class Minimal(Strategy):
            @property
            def strategy_id(self) -> str:
                return "minimal"

            @property
            def version(self) -> str:
                return "1.0.0"

            def required_timeframes(self) -> list[str]:
                return ["1Day"]

            def required_lookback_bars(self) -> int:
                return 5

            def run(self, ctx: StrategyContext) -> list[Signal]:
                return []

        strat = Minimal()
        # healthcheck should work without a context (returns True by default)
        assert strat.healthcheck(None) is True  # type: ignore[arg-type]
