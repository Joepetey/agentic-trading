"""Optuna-based parameter optimization for OPPW+Carlos strategy."""
from __future__ import annotations

from datetime import date

import optuna

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from core.calendar import warm_cache
from core.strategies.oppw_carlos import OPPWCarlosConfig, OPPWCarlosStrategy
from core.types import Bar, IntradayBar
from data.entry_prices import (
    ENTRY_TIMING_MODELS,
    EXIT_TIMING_MODELS,
    compute_all_entry_prices,
    compute_all_exit_prices,
)


def make_objective(
    tqqq_bars: list[Bar],
    bil_bar_map: dict[date, Bar],
    entry_price_maps: dict[str, dict[date, float] | None] | None = None,
    exit_price_maps: dict[str, dict[date, float] | None] | None = None,
    initial_cash: float = 100_000.0,
):
    """Build an Optuna objective that returns (CAGR, Sharpe)."""

    has_entry_timing = entry_price_maps is not None and len(entry_price_maps) > 1
    has_exit_timing = exit_price_maps is not None and len(exit_price_maps) > 1

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        # Strategy parameters
        tp_a = trial.suggest_float("profit_target_A", 0.02, 0.20)
        tp_c_ratio = trial.suggest_float("profit_target_C_ratio", 0.1, 0.8)
        stop = trial.suggest_float("stop_trigger_close", -0.05, -0.003)
        weakness = trial.suggest_categorical("weakness_mode", [True, False])

        # Engine parameters
        entry_offset = trial.suggest_int("entry_day_offset", 0, 2)

        # Entry timing
        entry_override = None
        if has_entry_timing:
            timing = trial.suggest_categorical("entry_timing", ENTRY_TIMING_MODELS)
            entry_override = entry_price_maps[timing]
            trial.set_user_attr("entry_timing", timing)

        # Exit timing
        exit_override = None
        if has_exit_timing:
            exit_timing = trial.suggest_categorical("exit_timing", EXIT_TIMING_MODELS)
            exit_override = exit_price_maps[exit_timing]
            trial.set_user_attr("exit_timing", exit_timing)

        cfg = OPPWCarlosConfig(
            profit_target_A=tp_a,
            profit_target_C=tp_a * tp_c_ratio,
            stop_trigger_close=stop,
            weakness_mode=weakness,
        )
        strategy = OPPWCarlosStrategy(cfg)

        result = run_backtest(
            tqqq_bars,
            strategy,
            initial_cash=initial_cash,
            full_exposure=True,
            sweep_bars=bil_bar_map,
            entry_day_offset=entry_offset,
            entry_price_override=entry_override,
            exit_price_override=exit_override,
        )
        metrics = compute_metrics(result)

        # Store extra metrics for analysis
        trial.set_user_attr("max_drawdown", metrics.max_drawdown)
        trial.set_user_attr("total_trades", metrics.total_trades)
        trial.set_user_attr("win_rate", metrics.win_rate)
        trial.set_user_attr("final_value", metrics.final_value)
        trial.set_user_attr("exposure_pct", metrics.exposure_pct)
        trial.set_user_attr("profit_target_C", tp_a * tp_c_ratio)

        return metrics.cagr, metrics.sharpe_ratio

    return objective


def run_optimization(
    tqqq_bars: list[Bar],
    bil_bar_map: dict[date, Bar],
    n_trials: int = 200,
    initial_cash: float = 100_000.0,
    study_name: str = "carlos_optimize",
    seed: int = 42,
    n_jobs: int = -1,
    intraday_bars: list[IntradayBar] | None = None,
) -> optuna.Study:
    """Run multi-objective optimization and return the study.

    Args:
        n_jobs: Number of parallel workers. -1 = use all CPU cores.
        intraday_bars: Optional 5-min bars for entry timing optimization.
            When provided, adds entry_timing as an optimizable parameter.
    """
    # Pre-warm calendar cache with a single bulk call
    bar_dates = [b.ts.date() for b in tqqq_bars]
    warm_cache(min(bar_dates), max(bar_dates))

    # Pre-compute entry/exit prices for all timing models (if intraday data available)
    # Pass daily bars for split adjustment (yfinance is split-adjusted, Alpaca is not)
    entry_price_maps = None
    exit_price_maps = None
    if intraday_bars:
        entry_price_maps = compute_all_entry_prices(intraday_bars, tqqq_bars)
        exit_price_maps = compute_all_exit_prices(intraday_bars, tqqq_bars)

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize", "maximize"],
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    objective = make_objective(
        tqqq_bars, bil_bar_map, entry_price_maps, exit_price_maps, initial_cash,
    )
    study.optimize(
        objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True,
    )

    return study
