"""Strategy runner — executes N strategies across a symbol universe."""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict

from src.core.errors import StrategyError
from src.strategies.base import Strategy
from src.strategies.context import Constraints, DataAccess, StrategyContext
from src.strategies.signal import Signal

logger = structlog.get_logger(__name__)

# Defaults
_DEFAULT_MAX_WORKERS = 4
_DEFAULT_TIMEOUT_SECS = 30.0


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


# ── Internals ────────────────────────────────────────────────────────


def _validate_signals(signals: Any, strategy_id: str) -> list[Signal]:
    """Validate the return value from ``Strategy.run()``.

    Raises StrategyError on type mismatches or mismatched strategy_id.
    """
    if not isinstance(signals, list):
        raise StrategyError(
            f"{strategy_id}.run() returned {type(signals).__name__}, expected list"
        )

    for sig in signals:
        if not isinstance(sig, Signal):
            raise StrategyError(
                f"{strategy_id}.run() emitted {type(sig).__name__}, expected Signal"
            )
        if sig.strategy_id != strategy_id:
            raise StrategyError(
                f"Signal strategy_id {sig.strategy_id!r} "
                f"does not match {strategy_id!r}"
            )

    return signals


def _run_one(strategy: Strategy, ctx: StrategyContext) -> list[Signal]:
    """Execute a single strategy and validate its output.

    This function runs inside a worker thread when parallel mode is used.
    """
    result = strategy.run(ctx)
    return _validate_signals(result, strategy.strategy_id)


# ── Runner ────────────────────────────────────────────────────────────


def run_strategies(
    strategies: list[Strategy],
    universe: list[str],
    conn: sqlite3.Connection,
    now_ts: datetime,
    config_map: dict[str, dict[str, Any]] | None = None,
    constraints: Constraints | None = None,
    *,
    max_workers: int = 1,
    strategy_timeout_secs: float | None = _DEFAULT_TIMEOUT_SECS,
    persist: bool = False,
) -> RunResult:
    """Run every strategy against the universe, collecting signals and errors.

    Error isolation: a failure in one strategy is logged and recorded
    but does not affect other strategies.

    Parallelism is by *strategy* (not symbol).  Data prefetch happens
    sequentially on the calling thread (SQLite reads), then each
    strategy's ``run()`` executes in a thread pool.

    When ``max_workers=1`` (the default) execution is fully sequential
    with no thread-pool overhead.

    Args:
        strategies:      Strategy instances to evaluate.
        universe:        Ticker symbols.
        conn:            SQLite connection (read-only usage).
        now_ts:          Point-in-time for evaluation.
        config_map:      Per-strategy config dicts, keyed by strategy_id.
        constraints:     Pre-evaluation filters.
        max_workers:     Thread-pool size.  ``1`` → sequential (no pool).
        strategy_timeout_secs:
            Per-strategy time budget in seconds.  ``None`` → no limit.
            A strategy that exceeds its budget is recorded as a
            ``StrategyRunError`` with ``error_type="TimeoutError"``.
        persist:
            If ``True``, write signals and a strategy_runs record to
            SQLite after execution completes.

    Returns:
        RunResult with signals sorted by descending |strength|.
    """
    config_map = config_map or {}
    constraints = constraints or Constraints()
    universe_tuple = tuple(universe)

    # ── Optional: create run record ─────────────────────────────────
    run_id: str | None = None
    if persist:
        from src.data.signals import create_run

        run_id = create_run(
            conn,
            eval_ts=now_ts,
            strategies=[s.strategy_id for s in strategies],
            universe_size=len(universe),
        )

    run_log = logger.bind(
        runner="run_strategies",
        strategy_count=len(strategies),
        symbol_count=len(universe),
        as_of=now_ts.isoformat(),
        max_workers=max_workers,
        run_id=run_id,
    )
    run_log.info("run_start")

    t0 = time.monotonic()

    # ── Phase 1: prefetch data (sequential, main thread) ────────────
    #
    # SQLite doesn't benefit from concurrent reads on the same
    # connection, so we pull all bar data before fanning out.
    prepared: list[tuple[Strategy, StrategyContext]] = []
    for strategy in strategies:
        dao = DataAccess(conn, now_ts)
        primary_tf = strategy.required_timeframes()[0]
        dao.prefetch(list(universe_tuple), primary_tf, strategy.required_lookback_bars())

        ctx = StrategyContext(
            now_ts=now_ts,
            universe=universe_tuple,
            timeframe=primary_tf,
            data=dao,
            config=config_map.get(strategy.strategy_id, {}),
            constraints=constraints,
        )
        prepared.append((strategy, ctx))

    # ── Phase 2: execute strategies ─────────────────────────────────
    signals: list[Signal] = []
    errors: list[StrategyRunError] = []

    if max_workers <= 1 or len(strategies) <= 1:
        # Fast path: no thread-pool overhead.
        for strategy, ctx in prepared:
            _execute_strategy(strategy, ctx, strategy_timeout_secs, signals, errors, run_log)
    else:
        _execute_parallel(prepared, max_workers, strategy_timeout_secs, signals, errors, run_log)

    # ── Phase 3: aggregate results ──────────────────────────────────
    elapsed_ms = (time.monotonic() - t0) * 1000
    signals.sort()

    result = RunResult(
        signals=signals,
        errors=errors,
        elapsed_ms=round(elapsed_ms, 2),
        strategies_run=len(strategies),
    )

    # ── Phase 4: persist (optional) ─────────────────────────────────
    if persist:
        from src.data.signals import complete_run, write_signals_from_result

        try:
            written = write_signals_from_result(
                conn,
                signals,
                run_id=run_id,
                eval_ts=now_ts,
                strategies=strategies,
            )
            complete_run(
                conn,
                run_id,  # type: ignore[arg-type]
                signals_written=written,
                errors=len(errors),
                elapsed_ms=result.elapsed_ms,
            )
        except Exception as exc:
            run_log.error("persist_failed", error=str(exc))
            complete_run(
                conn,
                run_id,  # type: ignore[arg-type]
                signals_written=0,
                errors=len(errors),
                elapsed_ms=result.elapsed_ms,
                error=str(exc),
            )

    run_log.info(
        "run_complete",
        signals=len(signals),
        errors=len(errors),
        elapsed_ms=result.elapsed_ms,
    )

    return result


def _execute_strategy(
    strategy: Strategy,
    ctx: StrategyContext,
    timeout_secs: float | None,
    signals: list[Signal],
    errors: list[StrategyRunError],
    run_log: Any,
) -> None:
    """Run a single strategy synchronously with optional timeout.

    For the sequential path (max_workers=1), timeout is enforced via a
    single-thread pool so we can still interrupt hung strategies.
    """
    sid = strategy.strategy_id
    ver = strategy.version
    strat_log = run_log.bind(
        strategy_id=sid, version=ver, params_hash=strategy.params_hash,
    )

    try:
        if timeout_secs is not None:
            # Use a one-off thread to enforce the timeout even in
            # sequential mode.
            with ThreadPoolExecutor(max_workers=1) as mini_pool:
                future = mini_pool.submit(_run_one, strategy, ctx)
                result = future.result(timeout=timeout_secs)
        else:
            result = _run_one(strategy, ctx)

        signals.extend(result)
        strat_log.info("strategy_complete", signal_count=len(result))

    except TimeoutError:
        error = StrategyRunError(
            strategy_id=sid,
            version=ver,
            error_type="TimeoutError",
            error_message=f"{sid} exceeded time budget of {timeout_secs}s",
        )
        errors.append(error)
        strat_log.warning("strategy_timeout", timeout_secs=timeout_secs)

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


def _execute_parallel(
    prepared: list[tuple[Strategy, StrategyContext]],
    max_workers: int,
    timeout_secs: float | None,
    signals: list[Signal],
    errors: list[StrategyRunError],
    run_log: Any,
) -> None:
    """Fan out strategy execution across a thread pool."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_strategy: dict[Future[list[Signal]], Strategy] = {}
        for strategy, ctx in prepared:
            future = pool.submit(_run_one, strategy, ctx)
            future_to_strategy[future] = strategy

        for future, strategy in future_to_strategy.items():
            sid = strategy.strategy_id
            ver = strategy.version
            strat_log = run_log.bind(
                strategy_id=sid, version=ver, params_hash=strategy.params_hash,
            )

            try:
                result = future.result(timeout=timeout_secs)
                signals.extend(result)
                strat_log.info("strategy_complete", signal_count=len(result))

            except TimeoutError:
                error = StrategyRunError(
                    strategy_id=sid,
                    version=ver,
                    error_type="TimeoutError",
                    error_message=f"{sid} exceeded time budget of {timeout_secs}s",
                )
                errors.append(error)
                strat_log.warning("strategy_timeout", timeout_secs=timeout_secs)

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
