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


# ── Helpers ──────────────────────────────────────────────────────────


def _regime_multiplier(
    sig: Signal,
    regime: str | None,
    regime_weights: dict[str, dict[str, float]],
    strategy_categories: dict[str, str],
) -> float:
    """Regime-based weight multiplier for a signal.

    Lookup order: strategy_id (most specific) → category → 1.0 (default).
    """
    if not regime:
        return 1.0
    weights_for_regime = regime_weights.get(regime, {})
    if not weights_for_regime:
        return 1.0
    # Check by strategy_id first
    if sig.strategy_id in weights_for_regime:
        return weights_for_regime[sig.strategy_id]
    # Fall back to category
    category = strategy_categories.get(sig.strategy_id)
    if category and category in weights_for_regime:
        return weights_for_regime[category]
    return 1.0


def _effective_weight(
    sig: Signal,
    strategy_weights: dict[str, float],
    regime_mult: float = 1.0,
) -> float:
    """Effective voting weight for a signal.

    If ``alpha_net`` is set (normalization ran), use its absolute value.
    Otherwise fall back to the classic ``|strength| * confidence * weight``.
    The regime multiplier is applied on top.
    """
    if sig.alpha_net is not None:
        return abs(sig.alpha_net) * regime_mult
    w = strategy_weights.get(sig.strategy_id, _DEFAULT_STRATEGY_WEIGHT)
    return abs(sig.strength) * sig.confidence * w * regime_mult


def _net_vote_contribution(
    sig: Signal,
    strategy_weights: dict[str, float],
    regime_mult: float = 1.0,
) -> float:
    """Signed vote contribution for directional consensus.

    If ``alpha_net`` is set, use it directly (already signed) scaled by regime.
    Otherwise compute from strength/confidence/weight.
    """
    if sig.alpha_net is not None:
        return sig.alpha_net * regime_mult
    w = strategy_weights.get(sig.strategy_id, _DEFAULT_STRATEGY_WEIGHT)
    vote = abs(sig.strength) * sig.confidence * w * regime_mult
    return -vote if sig.side == Side.SHORT else vote


# ── Public API ───────────────────────────────────────────────────────


def deconflict_signals(
    signals: list[Signal],
    universe: tuple[str, ...],
    strategy_weights: dict[str, float] | None = None,
    *,
    regime: str | None = None,
    regime_weights: dict[str, dict[str, float]] | None = None,
    strategy_categories: dict[str, str] | None = None,
    veto_tags: tuple[str, ...] = ("do_not_trade",),
    min_symbol_alpha: float = 0.0,
) -> tuple[list[MergedSignal], list[DroppedSignal]]:
    """Normalize and merge multiple strategy signals per symbol.

    Algorithm per symbol:
    1. Drop signals for symbols not in the filtered universe.
    2. Drop zero-strength signals (or zero-alpha_net if normalized).
    3. Check for veto tags — if any signal has a veto tag, drop ALL for that symbol.
    4. Separate exit intents (FLAT with negative strength) from directional.
    5. If only exits → build FLAT merged signal.
    6. If exits AND directional → weighted-vote contest; exit wins ties.
    7. For directional: weighted side consensus via effective weight (with regime multiplier).
    8. Keep same-side signals, drop opposite-side.
    9. Merge: agg_alpha = sum of alpha_net, horizon_bars = weighted average.
    10. Pass through tightest stop, nearest TP from any contributor.
    11. Post-merge: discard if abs(agg_alpha) < min_symbol_alpha.

    Args:
        signals:              Validated Signal objects from Phase 2.
        universe:             Filtered universe (signals for excluded symbols dropped).
        strategy_weights:     Optional per-strategy_id weight multiplier.
                              Ignored when signals carry alpha_net (normalization
                              already baked weights in).
        regime:               Current market regime (e.g., "trend", "chop").
        regime_weights:       Maps regime → category/strategy_id → weight multiplier.
        strategy_categories:  Maps strategy_id → category (e.g., "trend", "mean_rev").
        veto_tags:            Signal tags that trigger per-symbol veto.
        min_symbol_alpha:     Post-merge filter: discard if abs(agg_alpha) < threshold.

    Returns:
        Tuple of (merged_signals, dropped_signals).
    """
    strategy_weights = strategy_weights or {}
    regime_weights = regime_weights or {}
    strategy_categories = strategy_categories or {}
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

        # Drop signals where normalization zeroed out alpha (cost ate the alpha)
        if sig.alpha_net is not None and sig.alpha_net == 0.0:
            dropped.append(DroppedSignal(
                strategy_id=sig.strategy_id,
                symbol=sig.symbol,
                side=sig.side.value,
                strength=sig.strength,
                confidence=sig.confidence,
                reason=DropReason.BELOW_COST_THRESHOLD,
                detail="alpha_net=0 after cost subtraction",
            ))
            continue

        # Drop zero-strength signals (un-normalized path)
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
        result = _merge_symbol_signals(
            symbol, sigs, strategy_weights, dropped,
            regime=regime,
            regime_weights=regime_weights,
            strategy_categories=strategy_categories,
            veto_tags=veto_tags,
        )
        if result is not None:
            merged.append(result)

    # Post-merge alpha threshold filter
    if min_symbol_alpha > 0.0:
        filtered: list[MergedSignal] = []
        for m in merged:
            if m.agg_alpha is not None and abs(m.agg_alpha) < min_symbol_alpha:
                for c in m.contributions:
                    dropped.append(DroppedSignal(
                        strategy_id=c.strategy_id,
                        symbol=m.symbol,
                        side=c.side,
                        strength=c.strength,
                        confidence=c.confidence,
                        reason=DropReason.BELOW_ALPHA_THRESHOLD,
                        detail=(
                            f"abs(agg_alpha)={abs(m.agg_alpha):.4f} "
                            f"< threshold={min_symbol_alpha}"
                        ),
                    ))
                continue
            filtered.append(m)
        merged = filtered

    # Sort by abs(agg_alpha) when available, fallback to abs(agg_strength)
    merged.sort(key=lambda m: (
        -(abs(m.agg_alpha) if m.agg_alpha is not None else abs(m.agg_strength)),
        m.symbol,
    ))

    logger.info(
        "deconfliction_complete",
        input_signals=len(signals),
        merged_count=len(merged),
        dropped_count=len(dropped),
    )

    return merged, dropped


# ── Per-symbol merge ─────────────────────────────────────────────────


def _merge_symbol_signals(
    symbol: str,
    sigs: list[Signal],
    strategy_weights: dict[str, float],
    dropped: list[DroppedSignal],
    *,
    regime: str | None,
    regime_weights: dict[str, dict[str, float]],
    strategy_categories: dict[str, str],
    veto_tags: tuple[str, ...],
) -> MergedSignal | None:
    """Merge all signals for a single symbol into one MergedSignal."""
    # ── Veto check ────────────────────────────────────────────────
    veto_set = set(veto_tags)
    if any(veto_set & set(s.tags) for s in sigs):
        for s in sigs:
            dropped.append(DroppedSignal(
                strategy_id=s.strategy_id,
                symbol=s.symbol,
                side=s.side.value,
                strength=s.strength,
                confidence=s.confidence,
                reason=DropReason.VETOED,
                detail="Symbol vetoed by signal tag",
            ))
        return None

    # ── Precompute regime multipliers ─────────────────────────────
    regime_mults = {
        id(s): _regime_multiplier(s, regime, regime_weights, strategy_categories)
        for s in sigs
    }

    # Separate exit signals (FLAT with negative strength) from directional.
    exit_sigs: list[Signal] = []
    directional_sigs: list[Signal] = []
    for s in sigs:
        if s.side == Side.FLAT and s.strength < 0:
            exit_sigs.append(s)
        else:
            directional_sigs.append(s)

    # Only exit signals → emit FLAT merged
    if exit_sigs and not directional_sigs:
        return _build_exit_merged(symbol, exit_sigs, strategy_weights, regime_mults)

    # No signals at all
    if not directional_sigs and not exit_sigs:
        return None

    # Both exits and directional → weighted contest
    if exit_sigs and directional_sigs:
        exit_weight = sum(
            _effective_weight(s, strategy_weights, regime_mults[id(s)])
            for s in exit_sigs
        )
        dir_weight = sum(
            _effective_weight(s, strategy_weights, regime_mults[id(s)])
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
            return _build_exit_merged(symbol, exit_sigs, strategy_weights, regime_mults)
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
    net_vote = sum(
        _net_vote_contribution(s, strategy_weights, regime_mults[id(s)])
        for s in directional_sigs
    )

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

    return _build_directional_merged(
        symbol, consensus_side, same_side, strategy_weights, regime_mults,
    )


# ── Merge builders ───────────────────────────────────────────────────


def _build_directional_merged(
    symbol: str,
    side: str,
    sigs: list[Signal],
    strategy_weights: dict[str, float],
    regime_mults: dict[int, float],
) -> MergedSignal:
    """Weighted-average merge for same-direction signals.

    agg_alpha is the SUM of alpha_net (not weighted average).
    horizon_bars is the weighted average (not min).
    """
    contributions: list[SignalContribution] = []
    total_weight = 0.0
    weighted_strength = 0.0
    weighted_confidence = 0.0
    weighted_horizon = 0.0
    sum_alpha = 0.0
    has_alpha = False
    tightest_stop: float | None = None
    nearest_tp: float | None = None

    for s in sigs:
        rm = regime_mults.get(id(s), 1.0)
        ew = _effective_weight(s, strategy_weights, rm)
        total_weight += ew
        weighted_strength += s.strength * ew
        weighted_confidence += s.confidence * ew
        weighted_horizon += s.horizon_bars * ew

        if s.alpha_net is not None:
            has_alpha = True
            sum_alpha += s.alpha_net

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
            weight=round(ew, 6),
            horizon_bars=s.horizon_bars,
            alpha_net=s.alpha_net,
        ))

    agg_strength = weighted_strength / total_weight if total_weight > 0 else 0.0
    agg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
    agg_alpha = sum_alpha if has_alpha else None
    horizon = max(1, round(weighted_horizon / total_weight)) if total_weight > 0 else 1

    return MergedSignal(
        symbol=symbol,
        side=side,
        agg_strength=agg_strength,
        agg_confidence=agg_confidence,
        agg_alpha=agg_alpha,
        horizon_bars=horizon,
        stop_hint=tightest_stop,
        tp_hint=nearest_tp,
        contributions=tuple(contributions),
    )


def _build_exit_merged(
    symbol: str,
    sigs: list[Signal],
    strategy_weights: dict[str, float],
    regime_mults: dict[int, float],
) -> MergedSignal:
    """Build a FLAT merged signal from exit signals."""
    contributions: list[SignalContribution] = []
    total_weight = 0.0
    weighted_strength = 0.0
    weighted_confidence = 0.0
    sum_alpha = 0.0
    has_alpha = False

    for s in sigs:
        rm = regime_mults.get(id(s), 1.0)
        ew = _effective_weight(s, strategy_weights, rm)
        total_weight += ew
        weighted_strength += s.strength * ew
        weighted_confidence += s.confidence * ew

        if s.alpha_net is not None:
            has_alpha = True
            sum_alpha += s.alpha_net

        contributions.append(SignalContribution(
            strategy_id=s.strategy_id,
            side=s.side.value,
            strength=s.strength,
            confidence=s.confidence,
            weight=round(ew, 6),
            horizon_bars=s.horizon_bars,
            alpha_net=s.alpha_net,
        ))

    agg_alpha = sum_alpha if has_alpha else None

    return MergedSignal(
        symbol=symbol,
        side="flat",
        agg_strength=weighted_strength / total_weight if total_weight > 0 else 0.0,
        agg_confidence=weighted_confidence / total_weight if total_weight > 0 else 0.0,
        agg_alpha=agg_alpha,
        horizon_bars=1,
        contributions=tuple(contributions),
    )
