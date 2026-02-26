"""Orchestrator data models — frozen Pydantic types for the decision pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ── Enums ────────────────────────────────────────────────────────────


class ExclusionReason(str, Enum):
    """Why a symbol was excluded from the candidate universe."""

    BELOW_MIN_PRICE = "below_min_price"
    BELOW_MIN_VOLUME = "below_min_volume"
    INSUFFICIENT_DATA = "insufficient_data"
    MANUALLY_EXCLUDED = "manually_excluded"
    MAX_NAMES_EXCEEDED = "max_names_exceeded"
    DATA_TOO_STALE = "data_too_stale"


class DropReason(str, Enum):
    """Why a signal was dropped during deconfliction."""

    FLAT_SIGNAL = "flat_signal"
    CONFLICTING_SIDES = "conflicting_sides"
    BELOW_CONFIDENCE_THRESHOLD = "below_confidence_threshold"
    ZERO_STRENGTH = "zero_strength"
    SYMBOL_EXCLUDED = "symbol_excluded"


class SizingMethod(str, Enum):
    """Position sizing algorithm."""

    EQUAL_WEIGHT = "equal_weight"
    SIGNAL_WEIGHTED = "signal_weighted"


# ── Portfolio State (input) ──────────────────────────────────────────


class PositionSnapshot(BaseModel):
    """A single position as of a point in time."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    qty: float  # signed: positive=long, negative=short
    market_value: float  # current mark-to-market
    avg_entry_price: float
    unrealized_pnl: float


class OpenOrderSnapshot(BaseModel):
    """An open order as of a point in time."""

    model_config = ConfigDict(frozen=True)

    order_id: str
    symbol: str
    side: str  # "buy" | "sell"
    qty: float
    order_type: str  # "market" | "limit" | "stop" | "stop_limit"
    limit_price: float | None = None
    stop_price: float | None = None


class PortfolioState(BaseModel):
    """Snapshot of current holdings, cash, and open orders.

    Passed IN to the orchestrator.  Phase 3 never calls Alpaca;
    this is built externally (Phase 4 or a coordinator).
    """

    model_config = ConfigDict(frozen=True)

    as_of_ts: datetime
    equity: float  # total account equity
    cash: float  # available cash
    buying_power: float
    positions: tuple[PositionSnapshot, ...] = ()
    open_orders: tuple[OpenOrderSnapshot, ...] = ()

    @property
    def position_map(self) -> dict[str, PositionSnapshot]:
        """Lookup positions by symbol."""
        return {p.symbol: p for p in self.positions}

    @property
    def total_exposure(self) -> float:
        """Sum of absolute market values of all positions."""
        return sum(abs(p.market_value) for p in self.positions)

    @property
    def exposure_pct(self) -> float:
        """Current exposure as a fraction of equity."""
        if self.equity <= 0:
            return 0.0
        return self.total_exposure / self.equity


# ── Universe audit ───────────────────────────────────────────────────


class SymbolExclusion(BaseModel):
    """Record of a symbol excluded from the candidate universe."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    reason: ExclusionReason
    detail: str = ""


class UniverseResult(BaseModel):
    """Result of universe filtering: included symbols + exclusion audit trail."""

    model_config = ConfigDict(frozen=True)

    included: tuple[str, ...]
    excluded: tuple[SymbolExclusion, ...] = ()


# ── Signal deconfliction audit ───────────────────────────────────────


class SignalContribution(BaseModel):
    """One strategy's contribution to a merged target."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    side: str
    strength: float
    confidence: float
    weight: float  # weight used in aggregation
    horizon_bars: int


class DroppedSignal(BaseModel):
    """Record of a signal dropped during deconfliction."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    symbol: str
    side: str
    strength: float
    confidence: float
    reason: DropReason
    detail: str = ""


class MergedSignal(BaseModel):
    """Output of deconfliction: one aggregated signal per symbol."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    side: str  # "long" | "short" | "flat"
    agg_strength: float = Field(
        description="Weighted-average strength, clamped to [-1, +1]"
    )
    agg_confidence: float = Field(
        description="Weighted-average confidence, clamped to [0, 1]"
    )
    horizon_bars: int = Field(description="Min horizon across contributors")
    stop_hint: float | None = None  # tightest stop from any contributor
    tp_hint: float | None = None  # nearest TP from any contributor
    contributions: tuple[SignalContribution, ...] = ()

    @field_validator("agg_strength", mode="before")
    @classmethod
    def _clamp_strength(cls, v: Any) -> float:
        return max(-1.0, min(1.0, float(v)))

    @field_validator("agg_confidence", mode="before")
    @classmethod
    def _clamp_confidence(cls, v: Any) -> float:
        return max(0.0, min(1.0, float(v)))


# ── Target Position ─────────────────────────────────────────────────


class TargetPosition(BaseModel):
    """Desired holding for a single symbol — output of the sizing step."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    target_notional: float  # $ amount (signed: +ve=long, -ve=short)
    target_pct: float  # fraction of portfolio equity
    confidence: float  # from MergedSignal
    horizon_bars: int
    stop_hint: float | None = None
    tp_hint: float | None = None
    provenance: tuple[SignalContribution, ...] = ()
    explain: str = ""


# ── Portfolio Intent (the final artifact) ────────────────────────────


class PortfolioIntent(BaseModel):
    """The complete output of one orchestration cycle.

    This is the handoff artifact to Phase 4 (execution / risk gate).
    It is a pure computation result — no side effects.
    """

    model_config = ConfigDict(frozen=True)

    intent_id: str = Field(description="UUID4 hex for audit linkage")
    as_of_ts: datetime
    portfolio_state: PortfolioState
    universe: UniverseResult
    signals_used: tuple[MergedSignal, ...] = ()
    signals_dropped: tuple[DroppedSignal, ...] = ()
    targets: tuple[TargetPosition, ...] = ()
    sizing_method: SizingMethod = SizingMethod.SIGNAL_WEIGHTED
    strategy_run_id: str | None = None
    trade_allowed: bool = True
    elapsed_ms: float = 0.0
    explain: str = ""
