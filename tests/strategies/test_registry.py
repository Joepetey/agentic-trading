"""Tests for StrategyRegistry."""

from __future__ import annotations

from typing import Any

import pytest

from src.core.config import StrategyConfig, StrategyEntry
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

    def params(self) -> dict[str, Any]:
        return {}

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return []


class _ConfigurableStrategy(Strategy):
    """Strategy that accepts constructor params — for build_from_config tests."""

    def __init__(self, fast_period: int = 10, slow_period: int = 50, **_: Any) -> None:
        self._fast = fast_period
        self._slow = slow_period

    @property
    def strategy_id(self) -> str:
        return "configurable"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return self._slow

    def params(self) -> dict[str, Any]:
        return {"fast_period": self._fast, "slow_period": self._slow}

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return []


# ── Instance registration ────────────────────────────────────────────


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


# ── Class (factory) registration + build_from_config ─────────────────


class TestBuildFromConfig:
    def test_register_class_and_build(self) -> None:
        reg = StrategyRegistry()
        reg.register_class("configurable", _ConfigurableStrategy)

        cfg = StrategyConfig(
            enabled=["configurable"],
            entries={"configurable": StrategyEntry(fast_period=5, slow_period=20)},
        )
        reg.build_from_config(cfg)

        strat = reg.get("configurable")
        assert strat.params() == {"fast_period": 5, "slow_period": 20}
        assert strat.required_lookback_bars() == 20

    def test_build_uses_defaults_when_no_entry(self) -> None:
        reg = StrategyRegistry()
        reg.register_class("configurable", _ConfigurableStrategy)

        cfg = StrategyConfig(enabled=["configurable"], entries={})
        reg.build_from_config(cfg)

        strat = reg.get("configurable")
        assert strat.params() == {"fast_period": 10, "slow_period": 50}

    def test_build_missing_factory_raises(self) -> None:
        reg = StrategyRegistry()
        cfg = StrategyConfig(enabled=["nonexistent"], entries={})
        with pytest.raises(ConfigError, match="no registered factory"):
            reg.build_from_config(cfg)

    def test_duplicate_factory_raises(self) -> None:
        reg = StrategyRegistry()
        reg.register_class("x", _ConfigurableStrategy)
        with pytest.raises(ConfigError, match="already registered"):
            reg.register_class("x", _ConfigurableStrategy)

    def test_enabled_order_preserved(self) -> None:
        reg = StrategyRegistry()
        reg.register_class("alpha", _DummyStrategy)
        reg.register_class("beta", _DummyStrategy)

        cfg = StrategyConfig(
            enabled=["beta", "alpha"],
            entries={
                "alpha": StrategyEntry(),
                "beta": StrategyEntry(),
            },
        )

        # _DummyStrategy takes sid as first arg; factory call uses **params
        # so we need classes that produce the right strategy_id.
        # Use direct registration instead for this test.
        reg2 = StrategyRegistry()
        s_beta = _DummyStrategy("beta")
        s_alpha = _DummyStrategy("alpha")
        reg2.register(s_beta)
        reg2.register(s_alpha)

        ordered = reg2.enabled(cfg)
        assert [s.strategy_id for s in ordered] == ["beta", "alpha"]


# ── StrategyEntry ─────────────────────────────────────────────────────


class TestStrategyEntry:
    def test_strategy_params_excludes_constraints(self) -> None:
        entry = StrategyEntry(
            fast_period=10, slow_period=50, max_names=5, min_price=10.0,
        )
        params = entry.strategy_params()
        assert params == {"fast_period": 10, "slow_period": 50}
        assert "max_names" not in params
        assert "min_price" not in params

    def test_constraint_overrides(self) -> None:
        entry = StrategyEntry(max_names=5, min_avg_volume=100000)
        overrides = entry.constraint_overrides()
        assert overrides == {"max_names": 5, "min_avg_volume": 100000}

    def test_constraint_overrides_empty_when_all_none(self) -> None:
        entry = StrategyEntry(fast_period=10)
        assert entry.constraint_overrides() == {}

    def test_strategy_params_with_extra_fields(self) -> None:
        entry = StrategyEntry(lookback=20, entry_z=2.0, timeframe="1Day")
        params = entry.strategy_params()
        assert params == {"lookback": 20, "entry_z": 2.0, "timeframe": "1Day"}
