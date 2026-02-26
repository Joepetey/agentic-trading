"""Position sizing — convert merged signals to target notional allocations."""

from __future__ import annotations

import structlog

from src.core.config import RiskLimits
from src.orchestrator.models import (
    MergedSignal,
    PortfolioState,
    SizingMethod,
    TargetPosition,
)

logger = structlog.get_logger(__name__)


def compute_targets(
    merged_signals: list[MergedSignal],
    portfolio: PortfolioState,
    risk_limits: RiskLimits,
    method: SizingMethod = SizingMethod.SIGNAL_WEIGHTED,
) -> list[TargetPosition]:
    """Convert merged signals into target positions with risk-aware sizing.

    Flow:
    1. Compute raw allocation per signal based on sizing method.
    2. Cap each position to max_position_pct of equity.
    3. Scale all positions down if total exceeds max_portfolio_exposure_pct.
    4. Add FLAT targets for symbols with exit signals (target_notional=0).
    5. Existing positions not referenced by any signal remain unchanged
       (no target emitted — Phase 4 interprets missing target as "hold").

    Args:
        merged_signals:  Output of deconflict step (directional + flat).
        portfolio:       Current portfolio state.
        risk_limits:     Risk configuration.
        method:          Sizing algorithm.

    Returns:
        List of TargetPosition, one per symbol with a signal.
    """
    if not merged_signals:
        return []

    equity = portfolio.equity
    if equity <= 0:
        logger.warning("zero_equity", equity=equity)
        return []

    max_notional = equity * risk_limits.max_portfolio_exposure_pct
    max_per_position = equity * risk_limits.max_position_pct

    # Separate directional from exit signals
    directional = [m for m in merged_signals if m.side != "flat"]
    exits = [m for m in merged_signals if m.side == "flat"]

    # Compute directional targets
    targets: list[TargetPosition] = []

    if method == SizingMethod.EQUAL_WEIGHT:
        targets.extend(_size_equal_weight(directional, equity, max_notional, max_per_position))
    elif method == SizingMethod.SIGNAL_WEIGHTED:
        targets.extend(_size_signal_weighted(directional, equity, max_notional, max_per_position))

    # Add exit targets (notional=0 means "close the position")
    for m in exits:
        targets.append(TargetPosition(
            symbol=m.symbol,
            target_notional=0.0,
            target_pct=0.0,
            confidence=m.agg_confidence,
            horizon_bars=m.horizon_bars,
            stop_hint=m.stop_hint,
            tp_hint=m.tp_hint,
            provenance=m.contributions,
            explain=f"Exit signal: side=flat, agg_strength={m.agg_strength:.3f}",
        ))

    logger.info(
        "targets_computed",
        directional=len(directional),
        exits=len(exits),
        total=len(targets),
        method=method.value,
    )

    return targets


def _size_equal_weight(
    signals: list[MergedSignal],
    equity: float,
    max_total: float,
    max_per_position: float,
) -> list[TargetPosition]:
    """Equal-weight sizing: each signal gets equal share of available capital."""
    if not signals:
        return []

    n = len(signals)
    per_position = min(max_total / n, max_per_position)

    return [_build_target(m, per_position, equity) for m in signals]


def _size_signal_weighted(
    signals: list[MergedSignal],
    equity: float,
    max_total: float,
    max_per_position: float,
) -> list[TargetPosition]:
    """Signal-weighted sizing: allocate proportional to |agg_strength| * agg_confidence.

    Steps:
    1. Raw weight = |agg_strength| * agg_confidence for each signal.
    2. Normalize so weights sum to 1.
    3. Allocate: notional = weight * max_total.
    4. Cap each at max_per_position; redistribute excess equally.
    """
    if not signals:
        return []

    raw_weights = [
        abs(m.agg_alpha) if m.agg_alpha is not None
        else abs(m.agg_strength) * m.agg_confidence
        for m in signals
    ]
    total_raw = sum(raw_weights)

    if total_raw < 1e-12:
        return _size_equal_weight(signals, equity, max_total, max_per_position)

    norm_weights = [w / total_raw for w in raw_weights]

    # First pass: allocate and cap
    notionals = [w * max_total for w in norm_weights]
    excess = 0.0
    uncapped_count = 0

    for i, n in enumerate(notionals):
        if n > max_per_position:
            excess += n - max_per_position
            notionals[i] = max_per_position
        else:
            uncapped_count += 1

    # Redistribute excess to uncapped positions (single pass, conservative)
    if excess > 0 and uncapped_count > 0:
        bonus = excess / uncapped_count
        for i, n in enumerate(notionals):
            if n < max_per_position:
                notionals[i] = min(n + bonus, max_per_position)

    return [_build_target(signals[i], notionals[i], equity) for i in range(len(signals))]


def _build_target(
    merged: MergedSignal,
    notional: float,
    equity: float,
) -> TargetPosition:
    """Construct a TargetPosition from a MergedSignal and computed notional."""
    # Sign the notional: negative for short
    signed_notional = notional if merged.side == "long" else -notional

    return TargetPosition(
        symbol=merged.symbol,
        target_notional=round(signed_notional, 2),
        target_pct=round(signed_notional / equity, 6) if equity > 0 else 0.0,
        confidence=merged.agg_confidence,
        horizon_bars=merged.horizon_bars,
        stop_hint=merged.stop_hint,
        tp_hint=merged.tp_hint,
        provenance=merged.contributions,
        explain=(
            f"side={merged.side}, "
            f"agg_strength={merged.agg_strength:.3f}, "
            f"agg_confidence={merged.agg_confidence:.3f}, "
            f"notional=${abs(signed_notional):,.2f} "
            f"({abs(signed_notional / equity) * 100:.1f}% of equity)"
        ),
    )
