"""Tests for Strategy ABC."""

from __future__ import annotations

from typing import Any

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

            # Missing: required_timeframes, required_lookback_bars, params, run

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

            def params(self) -> dict[str, Any]:
                return {"lookback": 10}

            def run(self, ctx: StrategyContext) -> list[Signal]:
                return []

        strat = Complete()
        assert strat.strategy_id == "complete"
        assert strat.version == "1.0.0"
        assert strat.required_timeframes() == ["1Day"]
        assert strat.required_lookback_bars() == 10
        assert strat.params() == {"lookback": 10}

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

            def params(self) -> dict[str, Any]:
                return {}

            def run(self, ctx: StrategyContext) -> list[Signal]:
                return []

        strat = Minimal()
        assert strat.healthcheck(None) is True  # type: ignore[arg-type]


# ── params_hash ───────────────────────────────────────────────────────


class TestParamsHash:
    def _make_strategy(
        self, sid: str = "test", ver: str = "1.0.0", p: dict | None = None,
    ) -> Strategy:
        _p = p or {}

        class S(Strategy):
            @property
            def strategy_id(self) -> str:
                return sid

            @property
            def version(self) -> str:
                return ver

            def required_timeframes(self) -> list[str]:
                return ["1Day"]

            def required_lookback_bars(self) -> int:
                return 1

            def params(self) -> dict[str, Any]:
                return _p

            def run(self, ctx: StrategyContext) -> list[Signal]:
                return []

        return S()

    def test_deterministic(self) -> None:
        s1 = self._make_strategy(p={"fast": 10, "slow": 50})
        s2 = self._make_strategy(p={"fast": 10, "slow": 50})
        assert s1.params_hash == s2.params_hash

    def test_different_params_different_hash(self) -> None:
        s1 = self._make_strategy(p={"fast": 10, "slow": 50})
        s2 = self._make_strategy(p={"fast": 10, "slow": 100})
        assert s1.params_hash != s2.params_hash

    def test_different_version_different_hash(self) -> None:
        s1 = self._make_strategy(ver="1.0.0", p={"x": 1})
        s2 = self._make_strategy(ver="2.0.0", p={"x": 1})
        assert s1.params_hash != s2.params_hash

    def test_different_id_different_hash(self) -> None:
        s1 = self._make_strategy(sid="alpha", p={"x": 1})
        s2 = self._make_strategy(sid="beta", p={"x": 1})
        assert s1.params_hash != s2.params_hash

    def test_hash_is_hex_string(self) -> None:
        s = self._make_strategy()
        assert isinstance(s.params_hash, str)
        assert len(s.params_hash) == 16
        int(s.params_hash, 16)  # valid hex
