"""Orchestrate — the top-level decision cycle."""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime
from typing import Any

import structlog

from src.core.config import OrchestratorConfig, RiskLimits
from src.orchestrator.deconflict import deconflict_signals
from src.orchestrator.intent_persist import (
    ensure_intent_schema,
    generate_intent_id,
    write_intent,
)
from src.orchestrator.models import (
    ExclusionReason,
    PortfolioIntent,
    PortfolioState,
    SizingMethod,
    SymbolExclusion,
    UniverseResult,
)
from src.orchestrator.sizing import compute_targets
from src.orchestrator.timestamp import resolve_eval_ts
from src.orchestrator.universe import filter_universe
from src.strategies.base import Strategy
from src.strategies.context import Constraints
from src.strategies.runner import RunResult, run_strategies

logger = structlog.get_logger(__name__)


def orchestrate(
    strategies: list[Strategy],
    universe: list[str],
    conn: sqlite3.Connection,
    portfolio: PortfolioState,
    risk_limits: RiskLimits,
    *,
    now_ts: datetime | None = None,
    orchestrator_config: OrchestratorConfig | None = None,
    constraints: Constraints | None = None,
    config_map: dict[str, dict[str, Any]] | None = None,
    strategy_weights: dict[str, float] | None = None,
    sizing_method: SizingMethod = SizingMethod.SIGNAL_WEIGHTED,
    max_workers: int = 1,
    strategy_timeout_secs: float | None = 30.0,
    persist: bool = False,
) -> PortfolioIntent:
    """Execute one full decision cycle.

    Steps:
        0. Resolve evaluation timestamp (bar-close aligned) + freshness check.
        1. Filter universe using constraints + DAO.
        2. Run strategies (Phase 2) on filtered universe.
        3. Normalize + deconflict signals.
        4. Size positions to produce TargetPositions.
        5. Assemble and return PortfolioIntent.

    This function is pure computation (no Alpaca calls).
    It is deterministic given the same inputs + DB state.

    Args:
        strategies:         Strategy instances to run.
        universe:           Raw symbol list (pre-filtering).
        conn:               SQLite connection.
        portfolio:          Current portfolio snapshot.
        risk_limits:        Risk configuration from Settings.
        now_ts:             Evaluation timestamp override.  If None, auto-resolved
                            from the latest bar timestamps in the DB.
        orchestrator_config: Orchestrator settings (staleness thresholds, etc.).
        constraints:        Universe filtering constraints.
        config_map:         Per-strategy config dicts.
        strategy_weights:   Per-strategy weights for deconfliction.
        sizing_method:      Position sizing algorithm.
        max_workers:        Thread-pool size for strategy execution.
        strategy_timeout_secs: Per-strategy timeout.
        persist:            If True, persist intent + strategy run to SQLite.

    Returns:
        PortfolioIntent — the complete decision plan.
    """
    t0 = time.monotonic()
    intent_id = generate_intent_id()
    constraints = constraints or Constraints()
    orch_cfg = orchestrator_config or OrchestratorConfig()

    # ── Step 0: Resolve evaluation timestamp ──────────────────────────
    stale_exclusions: list[SymbolExclusion] = []

    if now_ts is not None:
        eval_ts = now_ts
    else:
        timeframe = orch_cfg.primary_timeframe
        max_stale_min = orch_cfg.max_staleness.get(timeframe)

        ts_result = resolve_eval_ts(
            conn=conn,
            symbols=universe,
            timeframe=timeframe,
            max_staleness_minutes=max_stale_min,
        )
        eval_ts = ts_result.eval_ts

        # Check if too many symbols are stale/missing → NO_TRADE
        bad_count = len(ts_result.stale_symbols) + len(ts_result.missing_symbols)
        if universe and bad_count / len(universe) > orch_cfg.max_stale_pct:
            elapsed_ms = (time.monotonic() - t0) * 1000
            explain = (
                f"NO_TRADE: {bad_count}/{len(universe)} symbols stale/missing "
                f"({bad_count / len(universe):.0%} > {orch_cfg.max_stale_pct:.0%} threshold). "
                f"Stale: {list(ts_result.stale_symbols)}, "
                f"Missing: {list(ts_result.missing_symbols)}."
            )
            logger.warning(
                "orchestrate_no_trade",
                intent_id=intent_id,
                stale=len(ts_result.stale_symbols),
                missing=len(ts_result.missing_symbols),
                threshold=orch_cfg.max_stale_pct,
            )

            no_trade_intent = PortfolioIntent(
                intent_id=intent_id,
                as_of_ts=eval_ts,
                portfolio_state=portfolio,
                universe=UniverseResult(included=()),
                trade_allowed=False,
                sizing_method=sizing_method,
                elapsed_ms=round(elapsed_ms, 2),
                explain=explain,
            )
            if persist:
                ensure_intent_schema(conn)
                write_intent(conn, no_trade_intent)
            return no_trade_intent

        # Build exclusions for stale/missing symbols
        for sym in ts_result.stale_symbols:
            stale_exclusions.append(SymbolExclusion(
                symbol=sym,
                reason=ExclusionReason.DATA_TOO_STALE,
                detail=f"Latest bar too old for timeframe {timeframe}",
            ))
        for sym in ts_result.missing_symbols:
            stale_exclusions.append(SymbolExclusion(
                symbol=sym,
                reason=ExclusionReason.DATA_TOO_STALE,
                detail=f"No bars found for timeframe {timeframe}",
            ))

        # Remove stale/missing from universe before filtering
        stale_set = set(ts_result.stale_symbols) | set(ts_result.missing_symbols)
        universe = [s for s in universe if s not in stale_set]

    orch_log = logger.bind(
        intent_id=intent_id,
        as_of=eval_ts.isoformat(),
        strategy_count=len(strategies),
        raw_universe=len(universe),
    )
    orch_log.info("orchestrate_start")

    # ── Step 1: Filter universe ──────────────────────────────────────
    universe_result = filter_universe(conn, universe, constraints, as_of=eval_ts)

    # Merge stale exclusions into the universe result
    if stale_exclusions:
        universe_result = UniverseResult(
            included=universe_result.included,
            excluded=tuple(stale_exclusions) + universe_result.excluded,
        )

    filtered_universe = list(universe_result.included)
    orch_log.info(
        "universe_filtered",
        included=len(filtered_universe),
        excluded=len(universe_result.excluded),
    )

    # ── Step 2: Run strategies ───────────────────────────────────────
    run_result: RunResult = run_strategies(
        strategies=strategies,
        universe=filtered_universe,
        conn=conn,
        now_ts=eval_ts,
        config_map=config_map,
        constraints=constraints,
        max_workers=max_workers,
        strategy_timeout_secs=strategy_timeout_secs,
        persist=persist,
    )
    orch_log.info(
        "strategies_executed",
        signals=len(run_result.signals),
        errors=len(run_result.errors),
        elapsed_ms=run_result.elapsed_ms,
    )

    # ── Step 3: Deconflict signals ───────────────────────────────────
    merged_signals, dropped_signals = deconflict_signals(
        signals=run_result.signals,
        universe=universe_result.included,
        strategy_weights=strategy_weights,
    )
    orch_log.info(
        "signals_deconflicted",
        merged=len(merged_signals),
        dropped=len(dropped_signals),
    )

    # ── Step 4: Position sizing ──────────────────────────────────────
    targets = compute_targets(
        merged_signals=merged_signals,
        portfolio=portfolio,
        risk_limits=risk_limits,
        method=sizing_method,
    )
    orch_log.info(
        "targets_computed",
        target_count=len(targets),
        sizing_method=sizing_method.value,
    )

    # ── Step 5: Assemble PortfolioIntent ─────────────────────────────
    elapsed_ms = (time.monotonic() - t0) * 1000

    explain_parts = [
        f"Cycle at {eval_ts.isoformat()}.",
        f"Universe: {len(filtered_universe)}/{len(universe)} symbols after filtering.",
        f"Strategies: {run_result.strategies_run} run, {len(run_result.errors)} errors.",
        f"Signals: {len(run_result.signals)} raw -> {len(merged_signals)} merged, {len(dropped_signals)} dropped.",
        f"Targets: {len(targets)} positions, sizing={sizing_method.value}.",
    ]

    intent = PortfolioIntent(
        intent_id=intent_id,
        as_of_ts=eval_ts,
        portfolio_state=portfolio,
        universe=universe_result,
        signals_used=tuple(merged_signals),
        signals_dropped=tuple(dropped_signals),
        targets=tuple(targets),
        sizing_method=sizing_method,
        strategy_run_id=None,
        trade_allowed=True,
        elapsed_ms=round(elapsed_ms, 2),
        explain=" ".join(explain_parts),
    )

    # ── Optional: persist ────────────────────────────────────────────
    if persist:
        ensure_intent_schema(conn)
        write_intent(conn, intent)

    orch_log.info(
        "orchestrate_complete",
        targets=len(targets),
        elapsed_ms=round(elapsed_ms, 2),
    )

    return intent
