"""Run OPPW+Carlos backtest on TQQQ historical data.

Usage:
    uv run python -m scripts.run_backtest              # default parameters
    uv run python -m scripts.run_backtest best_cagr     # load profile from strategy_config.toml
    uv run python -m scripts.run_backtest best_sharpe   # load profile from strategy_config.toml
"""

import csv
import sys
import tomllib
from datetime import date
from pathlib import Path

from backtest.engine import run_backtest
from backtest.metrics import compute_metrics
from core.strategies.oppw_carlos import OPPWCarlosConfig, OPPWCarlosStrategy
from core.types import Bar
from data.entry_prices import compute_entry_prices, compute_exit_prices
from data.store import load_bars, load_intraday_bars

CONFIG_PATH = Path("strategy_config.toml")


def _load_profile(profile: str) -> tuple[OPPWCarlosConfig, int, str, str]:
    """Load a named profile from strategy_config.toml.

    Returns (strategy_config, entry_day_offset, entry_timing, exit_timing).
    """
    with open(CONFIG_PATH, "rb") as f:
        config = tomllib.load(f)

    if profile not in config:
        available = [k for k in config if isinstance(config[k], dict)]
        print(f"  Profile '{profile}' not found. Available: {', '.join(available)}")
        sys.exit(1)

    p = config[profile]
    cfg = OPPWCarlosConfig(
        profit_target_A=p["profit_target_A"],
        profit_target_C=p["profit_target_C"],
        stop_trigger_close=p["stop_trigger_close"],
        weakness_mode=p["weakness_mode"],
    )
    entry_day_offset = p.get("entry_day_offset", 0)
    entry_timing = p.get("entry_timing", "open")
    exit_timing = p.get("exit_timing", "close")
    return cfg, entry_day_offset, entry_timing, exit_timing


def main() -> None:
    symbol = "TQQQ"
    start = date(2010, 1, 1)
    end = date.today()

    # Check for profile argument
    profile = sys.argv[1] if len(sys.argv) > 1 else None
    entry_day_offset = 0
    entry_timing = "open"
    exit_timing = "close"

    if profile and CONFIG_PATH.exists():
        print(f"Loading profile '{profile}' from {CONFIG_PATH} ...")
        cfg, entry_day_offset, entry_timing, exit_timing = _load_profile(profile)
        strategy = OPPWCarlosStrategy(cfg)
        print(f"  TP_A={cfg.profit_target_A:.3f}  TP_C={cfg.profit_target_C:.3f}  "
              f"Stop={cfg.stop_trigger_close:.4f}  "
              f"Day={entry_day_offset}  Weakness={cfg.weakness_mode}  "
              f"Entry={entry_timing}  Exit={exit_timing}")
    elif profile:
        print(f"  {CONFIG_PATH} not found. Run 'python -m scripts.run_optimize' first.")
        sys.exit(1)
    else:
        strategy = OPPWCarlosStrategy(OPPWCarlosConfig(qty=100))

    print(f"\nLoading {symbol} bars from DB ...")
    bars = load_bars(symbol, start, end)
    print(f"  {len(bars)} bars loaded")

    # Load BIL bars for treasury sweep
    print("Loading BIL bars from DB ...")
    bil_bars_list = load_bars("BIL", start, end)
    bil_bar_map: dict[date, Bar] = {b.ts.date(): b for b in bil_bars_list}
    print(f"  {len(bil_bar_map)} BIL bars loaded")

    # Compute entry/exit price overrides if using non-default timing
    entry_price_override = None
    exit_price_override = None
    needs_intraday = entry_timing != "open" or exit_timing != "close"
    intraday = None
    if needs_intraday:
        print(f"Loading intraday bars for timing overrides ...")
        intraday = load_intraday_bars(symbol, start, end)
        if intraday:
            print(f"  {len(intraday)} intraday bars loaded")
        else:
            print("  No intraday data — falling back to defaults")

    if entry_timing != "open" and intraday:
        entry_price_override = compute_entry_prices(intraday, entry_timing, bars)
        print(f"  Entry ({entry_timing}): {len(entry_price_override)} days with override prices")

    if exit_timing != "close" and intraday:
        exit_price_override = compute_exit_prices(intraday, exit_timing, bars)
        print(f"  Exit ({exit_timing}): {len(exit_price_override)} days with override prices")

    label = f"Profile: {profile}" if profile else "OPPW + Carlos Mod"
    print(f"Running backtest ({label}, full exposure, treasury sweep ON) ...")
    result = run_backtest(
        bars, strategy, initial_cash=100_000.0, full_exposure=True,
        sweep_bars=bil_bar_map,
        entry_day_offset=entry_day_offset,
        entry_price_override=entry_price_override,
        exit_price_override=exit_price_override,
    )

    metrics = compute_metrics(result)

    print()
    print("=" * 50)
    print(f"  {label} Backtest Results")
    print("=" * 50)
    print(f"  Period:           {bars[0].ts.date()} -> {bars[-1].ts.date()}")
    print(f"  Initial cash:     ${result.initial_cash:,.2f}")
    print(f"  Final value:      ${metrics.final_value:,.2f}")
    print(f"  Total return:     {metrics.total_return_pct:,.2f}%")
    print(f"  CAGR:             {metrics.cagr * 100:.2f}%")
    print(f"  Max drawdown:     {metrics.max_drawdown * 100:.2f}%")
    print(f"  Sharpe ratio:     {metrics.sharpe_ratio:.2f}")
    print(f"  Exposure:         {metrics.exposure_pct * 100:.1f}%")
    print()
    print(f"  Total trades:     {metrics.total_trades}")
    print(f"  Winners:          {metrics.winners} ({metrics.win_rate * 100:.1f}%)")
    print(f"  Losers:           {metrics.losers}")
    print(f"  Avg return:       {metrics.avg_return_pct * 100:.3f}%")
    print(f"  Best trade:       {metrics.best_trade_pct * 100:.3f}%")
    print(f"  Worst trade:      {metrics.worst_trade_pct * 100:.3f}%")
    print()
    print("  Exit reasons:")
    for reason, count in sorted(metrics.exit_reason_counts.items()):
        print(f"    {reason:8s}: {count}")
    if metrics.sweep_trades > 0:
        print()
        print(f"  Treasury sweep:   {metrics.sweep_trades} BIL round-trips")
        print(f"  Sweep P&L:        ${metrics.sweep_pnl:,.2f}")
    print("=" * 50)

    # Save equity curve to CSV
    csv_path = "backtest_equity.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "portfolio_value"])
        for d, v in result.equity_curve:
            writer.writerow([d.isoformat(), f"{v:.2f}"])
    print(f"\nEquity curve saved to {csv_path}")

    # Print last 5 strategy trades as a sample
    strategy_trades = [t for t in result.trades if t.symbol != "BIL"]
    print("\nLast 5 trades:")
    print(f"  {'Entry Date':<12} {'Entry $':>9} {'Exit Date':<12} {'Exit $':>9} {'Reason':<6} {'Return':>8}")
    for t in strategy_trades[-5:]:
        print(
            f"  {t.entry_ts!s:<12} {t.entry_price:>9.2f} "
            f"{t.exit_ts!s:<12} {t.exit_price:>9.2f} "
            f"{t.exit_reason:<6} {t.return_pct * 100:>7.3f}%"
        )


if __name__ == "__main__":
    main()
