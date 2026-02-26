"""Signal deconfliction — normalize and merge signals per symbol."""

from __future__ import annotations

from collections import defaultdict

import structlog

from src.orchestrator.models import (
    DroppedSignal,
    DropReason,
    MergedSignal,
    SignalContribution,
)
from src.strategies.signal import Side, Signal

logger = structlog.get_logger(__name__)

# Default strategy weight (equal weight across strategies).
_DEFAULT_STRATEGY_WEIGHT = 1.0


def deconflict_signals(
    signals: list[Signal],
    universe: tuple[str, ...],
    strategy_weights: dict[str, float] | None = None,
) -> tuple[list[MergedSignal], list[DroppedSignal]]:
    """Normalize and merge multiple strategy signals per symbol.

    Algorithm per symbol:
    1. Drop signals for symbols not in the filtered universe.
    2. Drop zero-strength signals.
    3. Separate exit intents (FLAT with negative strength) from directional.
    4. If only exits → build FLAT merged signal.
    5. If exits AND directional → weighted-vote contest; exit wins ties.
    6. For directional: weighted side consensus via |strength| * confidence * weight.
    7. Keep same-side signals, drop opposite-side.
    8. Weighted-average merge for final MergedSignal.
    9. Pass through tightest stop, nearest TP from any contributor.

    Args:
        signals:           Validated Signal objects from Phase 2.
        universe:          Filtered universe (signals for excluded symbols dropped).
        strategy_weights:  Optional per-strategy_id weight multiplier.

    Returns:
        Tuple of (merged_signals, dropped_signals).
    """
    strategy_weights = strategy_weights or {}
    universe_set = set(universe)

    by_symbol: dict[str, list[Signal]] = defaultdict(list)
    dropped: list[DroppedSignal] = []

    for sig in signals:
        # Drop signals for symbols not in filtered universe
        if sig.symbol not in universe_set:
            dropped.append(DroppedSignal(
                strategy_id=sig.strategy_id,
                symbol=sig.symbol,
                side=sig.side.value,
                strength=sig.strength,
                confidence=sig.confidence,
                reason=DropReason.SYMBOL_EXCLUDED,
                detail=f"{sig.symbol} not in filtered universe",
            ))
            continue

        # Drop zero-strength signals
        if sig.strength == 0.0:
            dropped.append(DroppedSignal(
                strategy_id=sig.strategy_id,
                symbol=sig.symbol,
                side=sig.side.value,
                strength=sig.strength,
                confidence=sig.confidence,
                reason=DropReason.ZERO_STRENGTH,
            ))
            continue

        by_symbol[sig.symbol].append(sig)

    merged: list[MergedSignal] = []

    for symbol, sigs in sorted(by_symbol.items()):
        result = _merge_symbol_signals(symbol, sigs, strategy_weights, dropped)
        if result is not None:
            merged.append(result)

    # Sort by abs(agg_strength) descending, then symbol
    merged.sort(key=lambda m: (-abs(m.agg_strength), m.symbol))

    logger.info(
        "deconfliction_complete",
        input_signals=len(signals),
        merged_count=len(merged),
        dropped_count=len(dropped),
    )

    return merged, dropped


def _merge_symbol_signals(
    symbol: str,
    sigs: list[Signal],
    strategy_weights: dict[str, float],
    dropped: list[DroppedSignal],
) -> MergedSignal | None:
    """Merge all signals for a single symbol into one MergedSignal."""
    # Separate exit signals (FLAT with negative strength) from directional.
    # Use explicit partitioning because Signal.__eq__ compares by sort key,
    # not full identity — two different signals can compare as "equal".
    exit_sigs: list[Signal] = []
    directional_sigs: list[Signal] = []
    for s in sigs:
        if s.side == Side.FLAT and s.strength < 0:
            exit_sigs.append(s)
        else:
            directional_sigs.append(s)

    # Only exit signals → emit FLAT merged
    if exit_sigs and not directional_sigs:
        return _build_exit_merged(symbol, exit_sigs, strategy_weights)

    # No signals at all
    if not directional_sigs and not exit_sigs:
        return None

    # Both exits and directional → weighted contest
    if exit_sigs and directional_sigs:
        exit_weight = sum(
            abs(s.strength) * s.confidence * strategy_weights.get(s.strategy_id, _DEFAULT_STRATEGY_WEIGHT)
            for s in exit_sigs
        )
        dir_weight = sum(
            abs(s.strength) * s.confidence * strategy_weights.get(s.strategy_id, _DEFAULT_STRATEGY_WEIGHT)
            for s in directional_sigs
        )

        # Exit wins ties (safe default)
        if exit_weight >= dir_weight:
            for s in directional_sigs:
                dropped.append(DroppedSignal(
                    strategy_id=s.strategy_id,
                    symbol=s.symbol,
                    side=s.side.value,
                    strength=s.strength,
                    confidence=s.confidence,
                    reason=DropReason.CONFLICTING_SIDES,
                    detail="Exit signals outweigh directional signals",
                ))
            return _build_exit_merged(symbol, exit_sigs, strategy_weights)
        else:
            for s in exit_sigs:
                dropped.append(DroppedSignal(
                    strategy_id=s.strategy_id,
                    symbol=s.symbol,
                    side=s.side.value,
                    strength=s.strength,
                    confidence=s.confidence,
                    reason=DropReason.CONFLICTING_SIDES,
                    detail="Directional signals outweigh exit signals",
                ))

    # Directional consensus via weighted vote
    net_vote = 0.0
    for s in directional_sigs:
        w = strategy_weights.get(s.strategy_id, _DEFAULT_STRATEGY_WEIGHT)
        vote = abs(s.strength) * s.confidence * w
        if s.side == Side.SHORT:
            net_vote -= vote
        else:
            # LONG or FLAT-with-positive-strength both treated as long direction
            net_vote += vote

    # Perfect cancellation
    if abs(net_vote) < 1e-9:
        for s in directional_sigs:
            dropped.append(DroppedSignal(
                strategy_id=s.strategy_id,
                symbol=s.symbol,
                side=s.side.value,
                strength=s.strength,
                confidence=s.confidence,
                reason=DropReason.CONFLICTING_SIDES,
                detail="Net directional vote is zero",
            ))
        return None

    consensus_side = "long" if net_vote > 0 else "short"

    # Keep same-side, drop opposite-side
    same_side: list[Signal] = []
    for s in directional_sigs:
        sig_side = "short" if s.side == Side.SHORT else "long"
        if sig_side == consensus_side:
            same_side.append(s)
        else:
            dropped.append(DroppedSignal(
                strategy_id=s.strategy_id,
                symbol=s.symbol,
                side=s.side.value,
                strength=s.strength,
                confidence=s.confidence,
                reason=DropReason.CONFLICTING_SIDES,
                detail=f"Opposite to consensus side={consensus_side}",
            ))

    if not same_side:
        return None

    return _build_directional_merged(symbol, consensus_side, same_side, strategy_weights)


def _build_directional_merged(
    symbol: str,
    side: str,
    sigs: list[Signal],
    strategy_weights: dict[str, float],
) -> MergedSignal:
    """Weighted-average merge for same-direction signals."""
    contributions: list[SignalContribution] = []
    total_weight = 0.0
    weighted_strength = 0.0
    weighted_confidence = 0.0
    min_horizon = float("inf")
    tightest_stop: float | None = None
    nearest_tp: float | None = None

    for s in sigs:
        w = strategy_weights.get(s.strategy_id, _DEFAULT_STRATEGY_WEIGHT)
        effective_weight = abs(s.strength) * s.confidence * w
        total_weight += effective_weight
        weighted_strength += s.strength * effective_weight
        weighted_confidence += s.confidence * effective_weight
        min_horizon = min(min_horizon, s.horizon_bars)

        if s.stop_price is not None:
            if tightest_stop is None:
                tightest_stop = s.stop_price
            elif side == "long":
                tightest_stop = max(tightest_stop, s.stop_price)
            else:
                tightest_stop = min(tightest_stop, s.stop_price)

        if s.take_profit_price is not None:
            if nearest_tp is None:
                nearest_tp = s.take_profit_price
            elif side == "long":
                nearest_tp = min(nearest_tp, s.take_profit_price)
            else:
                nearest_tp = max(nearest_tp, s.take_profit_price)

        contributions.append(SignalContribution(
            strategy_id=s.strategy_id,
            side=s.side.value,
            strength=s.strength,
            confidence=s.confidence,
            weight=round(effective_weight, 6),
            horizon_bars=s.horizon_bars,
        ))

    agg_strength = weighted_strength / total_weight if total_weight > 0 else 0.0
    agg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0

    return MergedSignal(
        symbol=symbol,
        side=side,
        agg_strength=agg_strength,
        agg_confidence=agg_confidence,
        horizon_bars=int(min_horizon) if min_horizon != float("inf") else 1,
        stop_hint=tightest_stop,
        tp_hint=nearest_tp,
        contributions=tuple(contributions),
    )


def _build_exit_merged(
    symbol: str,
    sigs: list[Signal],
    strategy_weights: dict[str, float],
) -> MergedSignal:
    """Build a FLAT merged signal from exit signals."""
    contributions: list[SignalContribution] = []
    total_weight = 0.0
    weighted_strength = 0.0
    weighted_confidence = 0.0

    for s in sigs:
        w = strategy_weights.get(s.strategy_id, _DEFAULT_STRATEGY_WEIGHT)
        effective_weight = abs(s.strength) * s.confidence * w
        total_weight += effective_weight
        weighted_strength += s.strength * effective_weight
        weighted_confidence += s.confidence * effective_weight
        contributions.append(SignalContribution(
            strategy_id=s.strategy_id,
            side=s.side.value,
            strength=s.strength,
            confidence=s.confidence,
            weight=round(effective_weight, 6),
            horizon_bars=s.horizon_bars,
        ))

    return MergedSignal(
        symbol=symbol,
        side="flat",
        agg_strength=weighted_strength / total_weight if total_weight > 0 else 0.0,
        agg_confidence=weighted_confidence / total_weight if total_weight > 0 else 0.0,
        horizon_bars=1,
        contributions=tuple(contributions),
    )
