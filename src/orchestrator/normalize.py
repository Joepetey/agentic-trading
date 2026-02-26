"""Signal normalization — compute a canonical alpha score per signal."""

from __future__ import annotations

import math

import structlog

from src.strategies.signal import Signal

logger = structlog.get_logger(__name__)

# Defaults when no config is provided.
_DEFAULT_WEIGHT = 1.0
_DEFAULT_EDGE_SCALE = 1.0
_BPS_DIVISOR = 10_000.0


def normalize_signals(
    signals: list[Signal],
    timeframe: str,
    *,
    strategy_weights: dict[str, float] | None = None,
    edge_scales: dict[str, float] | None = None,
    cost_bps: dict[str, float] | None = None,
) -> list[Signal]:
    """Compute alpha_net for each signal and return enriched copies.

    Formula per signal::

        raw_alpha  = strength * confidence          # strength is already signed
        calibrated = raw_alpha * weight * edge_scale
        cost       = cost_bps[timeframe] / 10_000
        alpha_net  = calibrated - copysign(cost, calibrated)

    If ``|calibrated| <= cost``, the signal's alpha is consumed by
    transaction costs and ``alpha_net`` is set to ``0.0``.

    This is a **pure function**: input signals are not mutated.

    Args:
        signals:          Validated Signal objects from the strategy runner.
        timeframe:        Primary timeframe for cost lookup.
        strategy_weights: Per-strategy weight multiplier (default 1.0).
        edge_scales:      Per-strategy edge calibration (default 1.0).
        cost_bps:         Round-trip cost per timeframe in basis points.

    Returns:
        New list of signals with ``alpha_net`` stamped via ``model_copy``.
    """
    weights = strategy_weights or {}
    scales = edge_scales or {}
    costs = cost_bps or {}

    cost = costs.get(timeframe, 0.0) / _BPS_DIVISOR

    result: list[Signal] = []
    for sig in signals:
        raw_alpha = sig.strength * sig.confidence
        w = weights.get(sig.strategy_id, _DEFAULT_WEIGHT)
        e = scales.get(sig.strategy_id, _DEFAULT_EDGE_SCALE)
        calibrated = raw_alpha * w * e

        if abs(calibrated) <= cost:
            alpha_net = 0.0
        else:
            alpha_net = calibrated - math.copysign(cost, calibrated)

        result.append(sig.model_copy(update={"alpha_net": alpha_net}))

    logger.info(
        "signals_normalized",
        count=len(signals),
        timeframe=timeframe,
        cost_bps=costs.get(timeframe, 0.0),
        zero_alpha=sum(1 for s in result if s.alpha_net == 0.0),
    )

    return result
