"""Strategy registry — discovery, validation, config-driven instantiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from src.core.errors import ConfigError

if TYPE_CHECKING:
    from src.core.config import StrategyConfig
    from src.strategies.base import Strategy

logger = structlog.get_logger(__name__)


class StrategyRegistry:
    """Central registry of available strategies.

    Supports two workflows:

    1. **Manual** — instantiate a strategy and call :meth:`register`.
    2. **Config-driven** — register factory classes with
       :meth:`register_class`, then call :meth:`build_from_config`
       to instantiate and register every enabled strategy.
    """

    def __init__(self) -> None:
        self._strategies: dict[str, Strategy] = {}
        self._factories: dict[str, type[Strategy]] = {}

    # ── instance registration ─────────────────────────────────────────

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
            params_hash=strategy.params_hash,
            timeframes=strategy.required_timeframes(),
            lookback_bars=strategy.required_lookback_bars(),
        )

    # ── class (factory) registration ──────────────────────────────────

    def register_class(self, name: str, cls: type[Strategy]) -> None:
        """Register a strategy class for config-driven instantiation.

        Args:
            name: The strategy_id this class produces.
            cls:  The Strategy subclass.  Will be called with
                  ``cls(**params)`` during :meth:`build_from_config`.
        """
        if name in self._factories:
            raise ConfigError(f"Factory for {name!r} already registered")
        self._factories[name] = cls
        logger.debug("strategy_class_registered", strategy_id=name, cls=cls.__name__)

    def build_from_config(self, config: StrategyConfig) -> None:
        """Instantiate and register every enabled strategy from config.

        For each name in ``config.enabled``:

        1. Look up the factory class registered via :meth:`register_class`.
        2. Extract ``strategy_params()`` from the matching
           ``config.entries[name]`` (or ``{}`` if no entry).
        3. Instantiate ``cls(**params)``.
        4. Register the instance.

        Raises ``ConfigError`` if an enabled strategy has no registered
        factory class.
        """
        for name in config.enabled:
            if name not in self._factories:
                raise ConfigError(
                    f"Strategy {name!r} is enabled in config but has no "
                    f"registered factory class.  Available: {list(self._factories)}"
                )

            cls = self._factories[name]
            entry = config.entries.get(name)
            params = entry.strategy_params() if entry else {}

            instance = cls(**params)
            self.register(instance)

    # ── lookup ────────────────────────────────────────────────────────

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

    def enabled(self, config: StrategyConfig) -> list[Strategy]:
        """Return registered strategies in the order listed in ``config.enabled``."""
        return [self._strategies[n] for n in config.enabled if n in self._strategies]

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


def register_class(name: str, cls: type[Strategy]) -> type[Strategy]:
    """Register a factory class with the default registry.  Returns the class."""
    _default_registry.register_class(name, cls)
    return cls


def get_default_registry() -> StrategyRegistry:
    return _default_registry
