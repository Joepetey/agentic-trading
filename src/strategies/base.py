"""Strategy base class — the contract every strategy must fulfil."""

from __future__ import annotations

import abc
import re

from src.strategies.context import StrategyContext
from src.strategies.signal import Signal

_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")


class Strategy(abc.ABC):
    """Abstract base for all strategies.

    Strategies are **stateless**.  All market state lives in the DB;
    the runner provides a ``StrategyContext`` with a time-bounded DAO.

    Subclasses must implement the four abstract members below.
    ``healthcheck`` is optional (defaults to ``True``).
    """

    @property
    @abc.abstractmethod
    def strategy_id(self) -> str:
        """Unique identifier (e.g. ``'sma_crossover'``)."""
        ...

    @property
    @abc.abstractmethod
    def version(self) -> str:
        """Semantic version string (``X.Y.Z``)."""
        ...

    @abc.abstractmethod
    def required_timeframes(self) -> list[str]:
        """Timeframes this strategy needs (e.g. ``['1Day', '5Min']``)."""
        ...

    @abc.abstractmethod
    def required_lookback_bars(self) -> int:
        """Minimum number of bars the strategy needs to produce a signal."""
        ...

    @abc.abstractmethod
    def run(self, ctx: StrategyContext) -> list[Signal]:
        """Evaluate the universe and return 0..N signals.

        Must read data **only** via ``ctx.data``.
        Must never call Alpaca or access external state.
        """
        ...

    def healthcheck(self, ctx: StrategyContext) -> bool:
        """Optional liveness probe.  Override to add custom checks."""
        return True

    # ── Validation helpers ────────────────────────────────────────────

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Validate metadata at class-definition time where possible."""
        super().__init_subclass__(**kwargs)

        # Skip validation for abstract subclasses (intermediate ABCs).
        if abc.ABC in cls.__bases__:
            return

        # If version is a plain attribute (not a property), validate eagerly.
        version_attr = cls.__dict__.get("version")
        if isinstance(version_attr, str) and not _SEMVER_RE.match(version_attr):
            raise ValueError(
                f"Strategy {cls.__name__}.version must be semver (X.Y.Z), "
                f"got {version_attr!r}"
            )
