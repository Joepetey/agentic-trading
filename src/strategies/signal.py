"""Signal — the typed output of every strategy."""

from __future__ import annotations

import functools
import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ─────────────────────────────────────────────────────────────


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class EntryType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PriceField(str, Enum):
    CLOSE = "close"
    HIGH = "high"
    LOW = "low"
    OPEN = "open"
    VWAP = "vwap"


class CompareOp(str, Enum):
    GT = "gt"
    LT = "lt"
    GTE = "gte"
    LTE = "lte"


# ── InvalidateCondition ──────────────────────────────────────────────


class InvalidateCondition(BaseModel):
    """Machine-checkable condition under which a signal is invalidated."""

    model_config = ConfigDict(frozen=True)

    field: PriceField
    op: CompareOp
    value: float


# ── Signal ───────────────────────────────────────────────────────────


@functools.total_ordering
class Signal(BaseModel):
    """Immutable, comparable, serialisable strategy output.

    Strategies return 0..N Signal objects.  Each represents an *intent*
    (not a position size).  Sizing belongs to the risk / execution layer.
    """

    model_config = ConfigDict(frozen=True)

    # ── Core fields (set by strategy) ────────────────────────────────
    strategy_id: str
    symbol: str
    side: Side
    strength: float = Field(description="Conviction direction+magnitude, clamped to [-1, +1]")
    confidence: float = Field(description="Meta-confidence in the signal, clamped to [0, 1]")
    horizon_bars: int = Field(gt=0, description="Expected holding period in bars")
    entry: EntryType = EntryType.MARKET
    entry_price_hint: float | None = None
    stop_price: float | None = None
    take_profit_price: float | None = None
    time_stop_bars: int | None = Field(default=None, gt=0)
    invalidate: tuple[InvalidateCondition, ...] = ()
    tags: tuple[str, ...] = ()
    explain: str = ""

    # ── Metadata (stamped by runner, not set by strategies) ────────
    signal_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Unique identifier for this signal instance",
    )
    cycle_id: str = Field(
        default="",
        description="Orchestrator intent_id linking this signal to a decision cycle",
    )
    strategy_version: str = Field(
        default="",
        description="Strategy semver, stamped by the runner after execution",
    )
    params_hash: str = Field(
        default="",
        description="Strategy params hash for reproducibility, stamped by runner",
    )
    alpha_net: float | None = Field(
        default=None,
        description="Net alpha after calibration and cost, stamped by normalize step",
    )

    # ── validators ────────────────────────────────────────────────────

    @field_validator("strength", mode="before")
    @classmethod
    def _clamp_strength(cls, v: Any) -> float:
        return max(-1.0, min(1.0, float(v)))

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))

    # ── ordering ──────────────────────────────────────────────────────
    # Sort by: highest |strength| first, then highest confidence, then symbol ASC.

    def _sort_key(self) -> tuple[float, float, str]:
        return (-abs(self.strength), -self.confidence, self.symbol)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return NotImplemented
        return self._sort_key() == other._sort_key()

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Signal):
            return NotImplemented
        return self._sort_key() < other._sort_key()

    def __hash__(self) -> int:
        return hash((self.strategy_id, self.symbol, self.side, self.strength, self.confidence))
