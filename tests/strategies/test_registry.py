"""Tests for StrategyRegistry."""

from __future__ import annotations

import pytest

from src.core.errors import ConfigError
from src.strategies.base import Strategy
from src.strategies.context import StrategyContext
from src.strategies.registry import StrategyRegistry
from src.strategies.signal import Signal


# ── Helpers ───────────────────────────────────────────────────────────


class _DummyStrategy(Strategy):
    def __init__(self, sid: str = "dummy", ver: str = "1.0.0") -> None:
        self._sid = sid
        self._ver = ver

    @property
    def strategy_id(self) -> str:
        return self._sid

    @property
    def version(self) -> str:
        return self._ver

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 10

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return []


# ── Tests ─────────────────────────────────────────────────────────────


class TestStrategyRegistry:
    def test_register_and_get(self) -> None:
        reg = StrategyRegistry()
        strat = _DummyStrategy("my_strat")
        reg.register(strat)
        assert reg.get("my_strat") is strat

    def test_duplicate_raises_config_error(self) -> None:
        reg = StrategyRegistry()
        reg.register(_DummyStrategy("dup"))
        with pytest.raises(ConfigError, match="already registered"):
            reg.register(_DummyStrategy("dup"))

    def test_get_missing_raises_key_error(self) -> None:
        reg = StrategyRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.get("nonexistent")

    def test_non_strategy_raises_type_error(self) -> None:
        reg = StrategyRegistry()
        with pytest.raises(TypeError, match="Expected Strategy"):
            reg.register("not a strategy")  # type: ignore[arg-type]

    def test_names(self) -> None:
        reg = StrategyRegistry()
        reg.register(_DummyStrategy("alpha"))
        reg.register(_DummyStrategy("beta"))
        assert set(reg.names()) == {"alpha", "beta"}

    def test_all(self) -> None:
        reg = StrategyRegistry()
        s1 = _DummyStrategy("s1")
        s2 = _DummyStrategy("s2")
        reg.register(s1)
        reg.register(s2)
        assert set(reg.all()) == {s1, s2}

    def test_len(self) -> None:
        reg = StrategyRegistry()
        assert len(reg) == 0
        reg.register(_DummyStrategy("x"))
        assert len(reg) == 1

    def test_contains(self) -> None:
        reg = StrategyRegistry()
        reg.register(_DummyStrategy("present"))
        assert "present" in reg
        assert "absent" not in reg
