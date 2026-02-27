"""Run Optuna parameter optimization for OPPW+Carlos strategy."""
from __future__ import annotations

import csv
import logging
from datetime import date, datetime

from data.store import load_bars, load_intraday_bars
from optimize.search import run_optimization


# Suppress Optuna's verbose trial-by-trial logging
optuna_logger = logging.getLogger("optuna")
optuna_logger.setLevel(logging.WARNING)


DAY_NAMES = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}


def main() -> None:
    start = date(2010, 1, 1)
    end = date.today()

    # Load data once — shared across all trials
    print("Loading TQQQ bars ...")
    tqqq_bars = load_bars("TQQQ", start, end)
    print(f"  {len(tqqq_bars)} bars")

    print("Loading BIL bars ...")
    bil_bars_list = load_bars("BIL", start, end)
    bil_bar_map = {b.ts.date(): b for b in bil_bars_list}
    print(f"  {len(bil_bar_map)} bars")

    # Load intraday bars for entry timing optimization
    print("Loading intraday bars ...")
    intraday_bars = load_intraday_bars("TQQQ", start, end)
    if intraday_bars:
        first_date = intraday_bars[0].ts.date()
        last_date = intraday_bars[-1].ts.date()
        print(f"  {len(intraday_bars)} bars ({first_date} -> {last_date})")
    else:
        print("  No intraday data — entry_timing will not be optimized")

    # Run optimization
    n_trials = 200
    print(f"\nRunning {n_trials}-trial optimization (CAGR + Sharpe) ...")
    study = run_optimization(
        tqqq_bars, bil_bar_map, n_trials=n_trials,
        intraday_bars=intraday_bars if intraday_bars else None,
    )

    # Pareto front
    pareto = study.best_trials
    pareto_sorted = sorted(pareto, key=lambda t: t.values[0], reverse=True)

    has_entry_timing = any("entry_timing" in t.params for t in pareto_sorted)
    has_exit_timing = any("exit_timing" in t.params for t in pareto_sorted)

    print()
    print("=" * 110)
    print(f"  Optimization Complete: {n_trials} trials")
    print("=" * 110)

    # Pareto front table
    header = (f"  {'#':>3}  {'CAGR':>7}  {'Sharpe':>7}  {'MaxDD':>7}  "
              f"{'TP_A':>6}  {'TP_C':>6}  {'Stop':>7}  {'Day':>4}  {'Weak':>5}")
    divider = (f"  {'---':>3}  {'------':>7}  {'------':>7}  {'------':>7}  "
               f"{'-----':>6}  {'-----':>6}  {'------':>7}  {'---':>4}  {'----':>5}")
    if has_entry_timing:
        header += f"  {'Entry':>8}"
        divider += f"  {'-------':>8}"
    if has_exit_timing:
        header += f"  {'Exit':>7}"
        divider += f"  {'------':>7}"

    print(f"\n  Pareto Front ({len(pareto_sorted)} solutions):")
    print(header)
    print(divider)
    for i, trial in enumerate(pareto_sorted, 1):
        cagr_pct = trial.values[0] * 100
        sharpe = trial.values[1]
        max_dd = trial.user_attrs["max_drawdown"] * 100
        tp_a = trial.params["profit_target_A"] * 100
        tp_c = trial.user_attrs["profit_target_C"] * 100
        stop = trial.params["stop_trigger_close"] * 100
        day = DAY_NAMES.get(trial.params["entry_day_offset"], "?")
        weak = trial.params["weakness_mode"]
        line = (f"  {i:>3}  {cagr_pct:>6.1f}%  {sharpe:>7.2f}  {max_dd:>6.1f}%  "
                f"{tp_a:>5.1f}%  {tp_c:>5.1f}%  {stop:>6.2f}%  {day:>4}  {weak!s:>5}")
        if has_entry_timing:
            timing = trial.params.get("entry_timing", "open")
            line += f"  {timing:>8}"
        if has_exit_timing:
            exit_t = trial.params.get("exit_timing", "close")
            line += f"  {exit_t:>7}"
        print(line)

    # Best by each objective
    best_cagr = max(pareto_sorted, key=lambda t: t.values[0])
    best_sharpe = max(pareto_sorted, key=lambda t: t.values[1])

    print()
    print(f"  Best by CAGR:   {best_cagr.values[0]*100:.1f}% "
          f"(Sharpe {best_cagr.values[1]:.2f}, "
          f"MaxDD {best_cagr.user_attrs['max_drawdown']*100:.1f}%)")
    print(f"  Best by Sharpe: {best_sharpe.values[1]:.2f} "
          f"(CAGR {best_sharpe.values[0]*100:.1f}%, "
          f"MaxDD {best_sharpe.user_attrs['max_drawdown']*100:.1f}%)")

    # Current defaults for comparison
    print()
    default_trial = None
    for trial in study.trials:
        p = trial.params
        if (abs(p["profit_target_A"] - 0.081) < 0.001
                and abs(p["stop_trigger_close"] - (-0.013)) < 0.001
                and p["entry_day_offset"] == 0
                and p["weakness_mode"] is True):
            default_trial = trial
            break
    if default_trial:
        print(f"  Current defaults: CAGR {default_trial.values[0]*100:.1f}%, "
              f"Sharpe {default_trial.values[1]:.2f}")

    print("=" * 110)

    # Save all trials to CSV
    csv_path = "optimize_results.csv"
    csv_fields = [
        "trial", "cagr", "sharpe", "max_drawdown", "win_rate",
        "total_trades", "final_value", "exposure_pct",
        "profit_target_A", "profit_target_C", "profit_target_C_ratio",
        "stop_trigger_close", "entry_day_offset", "weakness_mode",
    ]
    if has_entry_timing:
        csv_fields.append("entry_timing")
    if has_exit_timing:
        csv_fields.append("exit_timing")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
        for trial in study.trials:
            if trial.values is None:
                continue
            row = [
                trial.number,
                f"{trial.values[0]:.6f}",
                f"{trial.values[1]:.4f}",
                f"{trial.user_attrs['max_drawdown']:.6f}",
                f"{trial.user_attrs['win_rate']:.4f}",
                trial.user_attrs["total_trades"],
                f"{trial.user_attrs['final_value']:.2f}",
                f"{trial.user_attrs['exposure_pct']:.4f}",
                f"{trial.params['profit_target_A']:.6f}",
                f"{trial.user_attrs['profit_target_C']:.6f}",
                f"{trial.params['profit_target_C_ratio']:.6f}",
                f"{trial.params['stop_trigger_close']:.6f}",
                trial.params["entry_day_offset"],
                trial.params["weakness_mode"],
            ]
            if has_entry_timing:
                row.append(trial.params.get("entry_timing", "open"))
            if has_exit_timing:
                row.append(trial.params.get("exit_timing", "close"))
            writer.writerow(row)
    print(f"\n  All {n_trials} trials saved to {csv_path}")

    # Save Pareto-front profiles to TOML config
    _save_config(pareto_sorted, has_entry_timing, has_exit_timing)


def _trial_to_profile(
    trial, label: str,
    include_entry_timing: bool = False,
    include_exit_timing: bool = False,
) -> str:
    """Format a single Optuna trial as a TOML section."""
    p = trial.params
    a = trial.user_attrs
    tp_c = p["profit_target_A"] * p["profit_target_C_ratio"]
    lines = [
        f"[{label}]",
        f"profit_target_A = {p['profit_target_A']:.6f}",
        f"profit_target_C = {tp_c:.6f}",
        f"stop_trigger_close = {p['stop_trigger_close']:.6f}",
        f"entry_day_offset = {p['entry_day_offset']}",
        f"weakness_mode = {'true' if p['weakness_mode'] else 'false'}",
    ]
    if include_entry_timing:
        timing = p.get("entry_timing", "open")
        lines.append(f'entry_timing = "{timing}"')
    if include_exit_timing:
        exit_t = p.get("exit_timing", "close")
        lines.append(f'exit_timing = "{exit_t}"')
    lines.extend([
        f"# Backtest metrics",
        f"cagr = {trial.values[0]:.6f}",
        f"sharpe = {trial.values[1]:.4f}",
        f"max_drawdown = {a['max_drawdown']:.6f}",
        f"win_rate = {a['win_rate']:.4f}",
        f"total_trades = {a['total_trades']}",
    ])
    return "\n".join(lines)


def _save_config(
    pareto_sorted: list,
    has_entry_timing: bool = False,
    has_exit_timing: bool = False,
) -> None:
    """Save Pareto-front results to strategy_config.toml."""
    config_path = "strategy_config.toml"
    best_cagr = max(pareto_sorted, key=lambda t: t.values[0])
    best_sharpe = max(pareto_sorted, key=lambda t: t.values[1])

    sections = [
        f"# Auto-generated by run_optimize.py on {datetime.now():%Y-%m-%d %H:%M}",
        f"# {len(pareto_sorted)} Pareto-optimal solutions from optimization",
        "",
        _trial_to_profile(best_cagr, "best_cagr", has_entry_timing, has_exit_timing),
        "",
        _trial_to_profile(best_sharpe, "best_sharpe", has_entry_timing, has_exit_timing),
    ]

    # Add remaining Pareto solutions as numbered profiles
    seen = {best_cagr.number, best_sharpe.number}
    for i, trial in enumerate(pareto_sorted):
        if trial.number not in seen:
            sections.append("")
            sections.append(_trial_to_profile(trial, f"pareto_{i + 1}", has_entry_timing, has_exit_timing))

    with open(config_path, "w") as f:
        f.write("\n".join(sections) + "\n")
    print(f"  Best results saved to {config_path}")


if __name__ == "__main__":
    main()
