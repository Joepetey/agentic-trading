"""Strategies — signal types, base class, registry, and runner."""

from src.strategies.base import Strategy
from src.strategies.context import Constraints, DataAccess, StrategyContext
from src.strategies.registry import StrategyRegistry, get_default_registry, register
from src.strategies.runner import RunResult, StrategyRunError, run_strategies
from src.strategies.signal import (
    CompareOp,
    EntryType,
    InvalidateCondition,
    PriceField,
    Side,
    Signal,
)

__all__ = [
    "CompareOp",
    "Constraints",
    "DataAccess",
    "EntryType",
    "InvalidateCondition",
    "PriceField",
    "RunResult",
    "Side",
    "Signal",
    "Strategy",
    "StrategyContext",
    "StrategyRegistry",
    "StrategyRunError",
    "get_default_registry",
    "register",
    "run_strategies",
]
