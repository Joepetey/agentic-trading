"""Volatility estimation from bar data for position sizing."""

from __future__ import annotations

import math
import sqlite3
import statistics
from datetime import datetime

import structlog

from src.data.query import get_window

logger = structlog.get_logger(__name__)


def estimate_volatilities(
    conn: sqlite3.Connection,
    symbols: list[str],
    eval_ts: datetime,
    *,
    timeframe: str = "1Day",
    lookback_bars: int = 20,
    default_vol: float = 0.30,
    min_bars: int = 5,
) -> dict[str, float]:
    """Estimate annualized realized volatility per symbol from recent bars.

    Uses close-to-close log-return standard deviation, annualized with sqrt(252)
    for daily bars.

    Args:
        conn:           SQLite connection with bars table.
        symbols:        Symbols to estimate vol for.
        eval_ts:        Point-in-time ceiling for bar lookups.
        timeframe:      Bar timeframe (default "1Day").
        lookback_bars:  Number of bars to use for vol estimation.
        default_vol:    Fallback annualized vol when insufficient data.
        min_bars:       Minimum bars needed (log-returns = bars - 1).

    Returns:
        Dict mapping symbol → annualized vol (always > 0).
    """
    if not symbols:
        return {}

    # Annualization factor by timeframe
    annualization = _annualization_factor(timeframe)

    result: dict[str, float] = {}

    for symbol in symbols:
        bars = get_window(conn, symbol, timeframe, eval_ts, lookback_bars)

        if len(bars) < min_bars:
            logger.debug(
                "vol_insufficient_bars",
                symbol=symbol,
                bars=len(bars),
                min_bars=min_bars,
                using_default=default_vol,
            )
            result[symbol] = default_vol
            continue

        # Compute log-returns from close prices
        log_returns: list[float] = []
        for i in range(1, len(bars)):
            prev_close = bars[i - 1].close
            curr_close = bars[i].close
            if prev_close > 0 and curr_close > 0:
                log_returns.append(math.log(curr_close / prev_close))

        if len(log_returns) < 2:
            result[symbol] = default_vol
            continue

        # Sample standard deviation of log-returns
        daily_vol = statistics.stdev(log_returns)

        # Annualize
        annual_vol = daily_vol * annualization

        # Clamp above minimum to prevent division-by-zero in sizing
        result[symbol] = max(annual_vol, 0.01)

    logger.info(
        "volatilities_estimated",
        symbol_count=len(result),
        default_count=sum(1 for v in result.values() if v == default_vol),
    )

    return result


def _annualization_factor(timeframe: str) -> float:
    """Return sqrt(N) where N is the number of bars per year."""
    factors = {
        "1Day": math.sqrt(252),
        "5Min": math.sqrt(252 * 78),       # 78 five-min bars per trading day
        "1Min": math.sqrt(252 * 390),       # 390 one-min bars per trading day
        "1Hour": math.sqrt(252 * 6.5),      # 6.5 trading hours per day
    }
    return factors.get(timeframe, math.sqrt(252))
