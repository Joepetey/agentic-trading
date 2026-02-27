from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from backtest.engine import BacktestResult, Trade


@dataclass
class BacktestMetrics:
    total_trades: int
    winners: int
    losers: int
    win_rate: float
    avg_return_pct: float
    best_trade_pct: float
    worst_trade_pct: float
    cagr: float
    max_drawdown: float
    exposure_pct: float
    final_value: float
    total_return_pct: float
    exit_reason_counts: dict[str, int]
    sharpe_ratio: float = 0.0
    sweep_pnl: float = 0.0
    sweep_trades: int = 0


def compute_metrics(result: BacktestResult) -> BacktestMetrics:
    # Separate strategy trades from treasury sweep trades
    trades = [t for t in result.trades if t.symbol != "BIL"]
    bil_trades = [t for t in result.trades if t.symbol == "BIL"]
    curve = result.equity_curve

    total = len(trades)
    winners = sum(1 for t in trades if t.return_pct > 0)
    losers = sum(1 for t in trades if t.return_pct <= 0)
    win_rate = winners / total if total else 0.0

    returns = [t.return_pct for t in trades]
    avg_ret = sum(returns) / total if total else 0.0
    best = max(returns) if returns else 0.0
    worst = min(returns) if returns else 0.0

    # CAGR
    if curve and len(curve) > 1:
        first_date, _ = curve[0]
        last_date, last_val = curve[-1]
        years = (last_date - first_date).days / 365.25
        total_return = last_val / result.initial_cash
        cagr = (total_return ** (1 / years) - 1) if years > 0 else 0.0
    else:
        cagr = 0.0

    # Max drawdown
    max_dd = _max_drawdown(curve)

    # Exposure: fraction of trading days where a position was open
    exposure = _exposure_pct(trades, curve)

    total_return_pct = (result.final_value / result.initial_cash - 1) * 100

    # Exit reason breakdown (strategy trades only)
    reason_counts: dict[str, int] = {}
    for t in trades:
        reason_counts[t.exit_reason] = reason_counts.get(t.exit_reason, 0) + 1

    sharpe = _sharpe_ratio(curve)

    return BacktestMetrics(
        total_trades=total,
        winners=winners,
        losers=losers,
        win_rate=win_rate,
        avg_return_pct=avg_ret,
        best_trade_pct=best,
        worst_trade_pct=worst,
        cagr=cagr,
        max_drawdown=max_dd,
        exposure_pct=exposure,
        final_value=result.final_value,
        total_return_pct=total_return_pct,
        exit_reason_counts=reason_counts,
        sharpe_ratio=sharpe,
        sweep_pnl=sum(t.pnl for t in bil_trades),
        sweep_trades=len(bil_trades),
    )


def _max_drawdown(curve: list[tuple[date, float]]) -> float:
    if not curve:
        return 0.0
    peak = curve[0][1]
    max_dd = 0.0
    for _, val in curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _sharpe_ratio(
    curve: list[tuple[date, float]], risk_free_annual: float = 0.04
) -> float:
    """Annualized Sharpe ratio from daily equity curve.

    Uses daily log returns, annualized with sqrt(252).
    Default risk-free rate is 4% (approximate T-bill yield).
    """
    if len(curve) < 2:
        return 0.0

    # Daily log returns
    daily_returns: list[float] = []
    for i in range(1, len(curve)):
        prev_val = curve[i - 1][1]
        curr_val = curve[i][1]
        if prev_val > 0:
            daily_returns.append(math.log(curr_val / prev_val))

    if not daily_returns:
        return 0.0

    n = len(daily_returns)
    mean_daily = sum(daily_returns) / n
    variance = sum((r - mean_daily) ** 2 for r in daily_returns) / n
    std_daily = math.sqrt(variance)

    if std_daily == 0:
        return 0.0

    # Annualize
    annualized_return = mean_daily * 252
    annualized_std = std_daily * math.sqrt(252)
    risk_free_daily_log = math.log(1 + risk_free_annual) / 252
    excess_return = (mean_daily - risk_free_daily_log) * 252

    return excess_return / annualized_std


def _exposure_pct(trades: list[Trade], curve: list[tuple[date, float]]) -> float:
    """Fraction of trading days where a position was held."""
    if not curve or not trades:
        return 0.0
    total_days = len(curve)

    held_days: set[date] = set()
    all_dates = sorted(d for d, _ in curve)
    for t in trades:
        for d in all_dates:
            if t.entry_ts <= d <= t.exit_ts:
                held_days.add(d)

    return len(held_days) / total_days if total_days else 0.0
