"""Strategy registry — discovery, validation, and lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from src.core.errors import ConfigError

if TYPE_CHECKING:
    from src.strategies.base import Strategy

logger = structlog.get_logger(__name__)


class StrategyRegistry:
    """Central registry of available strategies.

    Enforces name uniqueness and type safety at registration time.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}

    def register(self, strategy: Strategy) -> None:
        """Register a strategy instance.  Raises on conflict or bad type."""
        from src.strategies.base import Strategy as StrategyBase

        if not isinstance(strategy, StrategyBase):
            raise TypeError(
                f"Expected Strategy subclass, got {type(strategy).__name__}"
            )

        key = strategy.strategy_id
        if key in self._strategies:
            existing = self._strategies[key]
            raise ConfigError(
                f"Strategy {key!r} already registered "
                f"(version {existing.version}).  "
                f"Cannot register version {strategy.version}."
            )

        self._strategies[key] = strategy
        logger.info(
            "strategy_registered",
            strategy_id=key,
            version=strategy.version,
            timeframes=strategy.required_timeframes(),
            lookback_bars=strategy.required_lookback_bars(),
        )

    def get(self, name: str) -> Strategy:
        """Look up by strategy_id.  Raises ``KeyError`` if missing."""
        if name not in self._strategies:
            raise KeyError(
                f"Strategy {name!r} not registered.  "
                f"Available: {list(self._strategies)}"
            )
        return self._strategies[name]

    def all(self) -> list[Strategy]:
        return list(self._strategies.values())

    def names(self) -> list[str]:
        return list(self._strategies.keys())

    def __len__(self) -> int:
        return len(self._strategies)

    def __contains__(self, name: str) -> bool:
        return name in self._strategies


# ── Module-level default ──────────────────────────────────────────────

_default_registry = StrategyRegistry()


def register(strategy: Strategy) -> Strategy:
    """Register with the default registry.  Returns the strategy."""
    _default_registry.register(strategy)
    return strategy


def get_default_registry() -> StrategyRegistry:
    return _default_registry
