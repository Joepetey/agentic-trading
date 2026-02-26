"""Universe filtering — apply constraints to the raw symbol list."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import structlog

from src.data.query import get_window
from src.orchestrator.models import ExclusionReason, SymbolExclusion, UniverseResult
from src.strategies.context import Constraints

logger = structlog.get_logger(__name__)

# Default lookback for volume/price checks.
_VOLUME_LOOKBACK_BARS = 20


def filter_universe(
    conn: sqlite3.Connection,
    symbols: list[str],
    constraints: Constraints,
    as_of: datetime,
    timeframe: str = "1Day",
) -> UniverseResult:
    """Apply constraints to produce a filtered universe with audit trail.

    Checks performed (in order):
    1. insufficient_data: skip symbols with no bars
    2. min_price: latest close >= threshold
    3. min_avg_volume: 20-day average volume >= threshold
    4. max_names: if more symbols pass than max_names, keep top by volume

    Args:
        conn:        SQLite connection (read-only).
        symbols:     Raw universe from config.
        constraints: Filters to apply.
        as_of:       Point-in-time ceiling for data access.
        timeframe:   Timeframe for price/volume lookups.

    Returns:
        UniverseResult with included symbols and exclusion records.
    """
    included: list[str] = []
    excluded: list[SymbolExclusion] = []
    volume_ranking: list[tuple[str, float]] = []  # (symbol, avg_volume)

    for symbol in sorted(symbols):
        # Check exclude_symbols first (no DB query needed)
        if symbol in constraints.exclude_symbols:
            excluded.append(SymbolExclusion(
                symbol=symbol,
                reason=ExclusionReason.MANUALLY_EXCLUDED,
                detail="In exclude_symbols list",
            ))
            continue

        bars = get_window(conn, symbol, timeframe, as_of, _VOLUME_LOOKBACK_BARS)

        if not bars:
            excluded.append(SymbolExclusion(
                symbol=symbol,
                reason=ExclusionReason.INSUFFICIENT_DATA,
                detail="No bars available",
            ))
            continue

        latest_close = bars[-1].close
        volumes = [b.volume for b in bars if b.volume is not None]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0.0

        # Check min_price
        if constraints.min_price is not None and latest_close < constraints.min_price:
            excluded.append(SymbolExclusion(
                symbol=symbol,
                reason=ExclusionReason.BELOW_MIN_PRICE,
                detail=f"close={latest_close:.2f} < min={constraints.min_price:.2f}",
            ))
            continue

        # Check min_avg_volume
        if constraints.min_avg_volume is not None and avg_volume < constraints.min_avg_volume:
            excluded.append(SymbolExclusion(
                symbol=symbol,
                reason=ExclusionReason.BELOW_MIN_VOLUME,
                detail=f"avg_vol={avg_volume:.0f} < min={constraints.min_avg_volume}",
            ))
            continue

        included.append(symbol)
        volume_ranking.append((symbol, avg_volume))

    # Apply max_names: keep top N by average volume
    if constraints.max_names is not None and len(included) > constraints.max_names:
        volume_ranking.sort(key=lambda x: x[1], reverse=True)
        kept = {s for s, _ in volume_ranking[: constraints.max_names]}
        newly_excluded = [s for s in included if s not in kept]
        for s in newly_excluded:
            excluded.append(SymbolExclusion(
                symbol=s,
                reason=ExclusionReason.MAX_NAMES_EXCEEDED,
                detail=f"Ranked below top {constraints.max_names} by volume",
            ))
        included = [s for s in included if s in kept]

    logger.info(
        "universe_filtered",
        raw_count=len(symbols),
        included_count=len(included),
        excluded_count=len(excluded),
    )

    return UniverseResult(
        included=tuple(sorted(included)),
        excluded=tuple(excluded),
    )
