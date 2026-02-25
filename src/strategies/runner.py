"""Strategy runner — executes N strategies across a symbol universe."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict

from src.core.errors import StrategyError
from src.strategies.base import Strategy
from src.strategies.context import Constraints, DataAccess, StrategyContext
from src.strategies.signal import Signal

logger = structlog.get_logger(__name__)


# ── Result types ──────────────────────────────────────────────────────


class StrategyRunError(BaseModel):
    """Structured record of a single strategy failure."""

    model_config = ConfigDict(frozen=True)

    strategy_id: str
    version: str
    error_type: str
    error_message: str


class RunResult(BaseModel):
    """Aggregate output of a strategy run."""

    model_config = ConfigDict(frozen=True)

    signals: list[Signal]
    errors: list[StrategyRunError]
    elapsed_ms: float
    strategies_run: int


# ── Runner ────────────────────────────────────────────────────────────


def run_strategies(
    strategies: list[Strategy],
    universe: list[str],
    conn: sqlite3.Connection,
    now_ts: datetime,
    config_map: dict[str, dict[str, Any]] | None = None,
    constraints: Constraints | None = None,
) -> RunResult:
    """Run every strategy against the universe, collecting signals and errors.

    Error isolation: a failure in one strategy is logged and recorded
    but does not affect other strategies.

    Execution is sequential — SQLite doesn't benefit from concurrent
    reads on the same connection, and strategy evaluation is CPU-light.

    Args:
        strategies:  Strategy instances to evaluate.
        universe:    Ticker symbols.
        conn:        SQLite connection (read-only usage).
        now_ts:      Point-in-time for evaluation.
        config_map:  Per-strategy config dicts, keyed by strategy_id.
        constraints: Pre-evaluation filters.

    Returns:
        RunResult with signals sorted by descending |strength|.
    """
    config_map = config_map or {}
    constraints = constraints or Constraints()
    universe_tuple = tuple(universe)

    run_log = logger.bind(
        runner="run_strategies",
        strategy_count=len(strategies),
        symbol_count=len(universe),
        as_of=now_ts.isoformat(),
    )
    run_log.info("run_start")

    t0 = time.monotonic()
    signals: list[Signal] = []
    errors: list[StrategyRunError] = []

    for strategy in strategies:
        sid = strategy.strategy_id
        ver = strategy.version
        strat_log = run_log.bind(strategy_id=sid, version=ver)

        dao = DataAccess(conn, now_ts)
        primary_tf = strategy.required_timeframes()[0]

        # Batch-prefetch bar windows for the whole universe in one query.
        dao.prefetch(list(universe_tuple), primary_tf, strategy.required_lookback_bars())

        ctx = StrategyContext(
            now_ts=now_ts,
            universe=universe_tuple,
            timeframe=primary_tf,
            data=dao,
            config=config_map.get(sid, {}),
            constraints=constraints,
        )

        try:
            result = strategy.run(ctx)

            # Validate return type
            if not isinstance(result, list):
                raise StrategyError(
                    f"{sid}.run() returned {type(result).__name__}, expected list"
                )

            for sig in result:
                if not isinstance(sig, Signal):
                    raise StrategyError(
                        f"{sid}.run() emitted {type(sig).__name__}, expected Signal"
                    )
                if sig.strategy_id != sid:
                    raise StrategyError(
                        f"Signal strategy_id {sig.strategy_id!r} "
                        f"does not match {sid!r}"
                    )

            signals.extend(result)
            strat_log.info(
                "strategy_complete",
                signal_count=len(result),
            )

        except Exception as exc:
            error = StrategyRunError(
                strategy_id=sid,
                version=ver,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            errors.append(error)
            strat_log.warning(
                "strategy_failed",
                error_type=error.error_type,
                error_message=error.error_message,
            )

    elapsed_ms = (time.monotonic() - t0) * 1000
    signals.sort()

    result = RunResult(
        signals=signals,
        errors=errors,
        elapsed_ms=round(elapsed_ms, 2),
        strategies_run=len(strategies),
    )

    run_log.info(
        "run_complete",
        signals=len(signals),
        errors=len(errors),
        elapsed_ms=result.elapsed_ms,
    )

    return result
