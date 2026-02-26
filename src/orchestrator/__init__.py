"""Orchestrator — decision cycle from signals to portfolio intent."""

from src.orchestrator.deconflict import deconflict_signals
from src.orchestrator.normalize import normalize_signals
from src.orchestrator.intent_persist import (
    ensure_intent_schema,
    generate_intent_id,
    get_intent,
    get_latest_intent,
    write_intent,
)
from src.orchestrator.models import (
    DroppedSignal,
    DropReason,
    ExclusionReason,
    MergedSignal,
    OpenOrderSnapshot,
    PortfolioIntent,
    PortfolioState,
    PositionSnapshot,
    SignalContribution,
    SizingMethod,
    SymbolExclusion,
    TargetPosition,
    UniverseResult,
)
from src.orchestrator.orchestrate import orchestrate
from src.orchestrator.sizing import compute_targets
from src.orchestrator.timestamp import EvalTimestampResult, resolve_eval_ts
from src.orchestrator.universe import filter_universe
from src.orchestrator.volatility import estimate_volatilities

__all__ = [
    # Main entry point
    "orchestrate",
    # Pipeline steps (usable individually)
    "resolve_eval_ts",
    "EvalTimestampResult",
    "filter_universe",
    "normalize_signals",
    "deconflict_signals",
    "compute_targets",
    "estimate_volatilities",
    # Models
    "DroppedSignal",
    "DropReason",
    "ExclusionReason",
    "MergedSignal",
    "OpenOrderSnapshot",
    "PortfolioIntent",
    "PortfolioState",
    "PositionSnapshot",
    "SignalContribution",
    "SizingMethod",
    "SymbolExclusion",
    "TargetPosition",
    "UniverseResult",
    # Persistence
    "ensure_intent_schema",
    "generate_intent_id",
    "get_intent",
    "get_latest_intent",
    "write_intent",
]
