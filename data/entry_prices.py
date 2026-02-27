"""Compute entry/exit fill prices from intraday bars for different timing models."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, time

from core.types import Bar, IntradayBar

ENTRY_TIMING_MODELS = ["open", "9:35", "10:00", "vwap_30m", "vwap_60m"]
EXIT_TIMING_MODELS = ["close", "15:30", "15:55"]

# 5-min bar start times for each model
_T_0930 = time(9, 30)
_T_0935 = time(9, 35)
_T_0955 = time(9, 55)
_T_1000 = time(10, 0)
_T_1025 = time(10, 25)
_T_1030 = time(10, 30)
_T_1525 = time(15, 25)
_T_1550 = time(15, 50)


def _compute_split_ratios(
    daily_bars: list[Bar],
    intraday_bars: list[IntradayBar],
) -> dict[date, float]:
    """Compute per-date adjustment ratio: daily_open / intraday_first_open.

    yfinance daily bars are split-adjusted; Alpaca intraday bars are NOT.
    This ratio lets us convert raw intraday prices to split-adjusted prices.
    """
    # Daily open by date
    daily_open: dict[date, float] = {}
    for b in daily_bars:
        daily_open[b.ts.date()] = b.open

    # First intraday bar (9:30) open by date
    intraday_open: dict[date, float] = {}
    for b in intraday_bars:
        d = b.ts.date()
        t = b.ts.time()
        if t == _T_0930 and d not in intraday_open:
            intraday_open[d] = b.open

    ratios: dict[date, float] = {}
    for d in intraday_open:
        if d in daily_open and intraday_open[d] > 0:
            ratios[d] = daily_open[d] / intraday_open[d]

    return ratios


def compute_entry_prices(
    intraday_bars: list[IntradayBar],
    model: str,
    daily_bars: list[Bar] | None = None,
) -> dict[date, float]:
    """Compute a date->price map for a given entry timing model.

    Args:
        intraday_bars: 5-min bars sorted by timestamp.
        model: One of ENTRY_TIMING_MODELS (excluding "open").
        daily_bars: Daily bars for split adjustment. Required for correct
            prices when intraday data is not split-adjusted.

    Returns:
        Dict mapping trading dates to the entry fill price for that model.
        Dates without sufficient intraday data are omitted.
    """
    # Compute split adjustment ratios
    ratios: dict[date, float] = {}
    if daily_bars:
        ratios = _compute_split_ratios(daily_bars, intraday_bars)

    # Group bars by date
    by_date: dict[date, list[IntradayBar]] = defaultdict(list)
    for bar in intraday_bars:
        by_date[bar.ts.date()].append(bar)

    prices: dict[date, float] = {}

    for d, bars in by_date.items():
        ratio = ratios.get(d, 1.0)
        bar_by_time = {b.ts.time(): b for b in bars}

        if model == "9:35":
            # Close of the 9:30–9:35 bar
            b = bar_by_time.get(_T_0930)
            if b:
                prices[d] = b.close * ratio

        elif model == "10:00":
            # Close of the 9:55–10:00 bar
            b = bar_by_time.get(_T_0955)
            if b:
                prices[d] = b.close * ratio

        elif model == "vwap_30m":
            # VWAP of 9:30–9:55 bars (6 bars covering 9:30–10:00)
            vwap_bars = [
                b for b in bars
                if _T_0930 <= b.ts.time() < _T_1000
            ]
            if vwap_bars:
                prices[d] = _vwap(vwap_bars) * ratio

        elif model == "vwap_60m":
            # VWAP of 9:30–10:25 bars (12 bars covering 9:30–10:30)
            vwap_bars = [
                b for b in bars
                if _T_0930 <= b.ts.time() < _T_1030
            ]
            if vwap_bars:
                prices[d] = _vwap(vwap_bars) * ratio

        else:
            raise ValueError(f"Unknown entry timing model: {model!r}")

    return prices


def _vwap(bars: list[IntradayBar]) -> float:
    """Compute VWAP from a list of bars: sum(typical * vol) / sum(vol)."""
    total_tv = 0.0
    total_vol = 0.0
    for b in bars:
        typical = (b.high + b.low + b.close) / 3
        total_tv += typical * b.volume
        total_vol += b.volume
    if total_vol == 0:
        # Fallback: simple average of closes
        return sum(b.close for b in bars) / len(bars)
    return total_tv / total_vol


def compute_all_entry_prices(
    intraday_bars: list[IntradayBar],
    daily_bars: list[Bar] | None = None,
) -> dict[str, dict[date, float] | None]:
    """Pre-compute entry prices for all timing models.

    Args:
        intraday_bars: 5-min intraday bars.
        daily_bars: Daily bars for split adjustment.

    Returns:
        Dict mapping model name to date->price map. "open" maps to None
        (engine uses bar.open by default).
    """
    result: dict[str, dict[date, float] | None] = {"open": None}
    for model in ENTRY_TIMING_MODELS:
        if model != "open":
            result[model] = compute_entry_prices(intraday_bars, model, daily_bars)
    return result


def compute_exit_prices(
    intraday_bars: list[IntradayBar],
    model: str,
    daily_bars: list[Bar] | None = None,
) -> dict[date, float]:
    """Compute a date->price map for a given exit timing model.

    Args:
        intraday_bars: 5-min bars sorted by timestamp.
        model: One of EXIT_TIMING_MODELS (excluding "close").
        daily_bars: Daily bars for split adjustment.

    Returns:
        Dict mapping trading dates to the exit fill price for that model.
    """
    ratios: dict[date, float] = {}
    if daily_bars:
        ratios = _compute_split_ratios(daily_bars, intraday_bars)

    by_date: dict[date, list[IntradayBar]] = defaultdict(list)
    for bar in intraday_bars:
        by_date[bar.ts.date()].append(bar)

    prices: dict[date, float] = {}

    for d, bars in by_date.items():
        ratio = ratios.get(d, 1.0)
        bar_by_time = {b.ts.time(): b for b in bars}

        if model == "15:30":
            # Close of the 15:25–15:30 bar
            b = bar_by_time.get(_T_1525)
            if b:
                prices[d] = b.close * ratio

        elif model == "15:55":
            # Close of the 15:50–15:55 bar
            b = bar_by_time.get(_T_1550)
            if b:
                prices[d] = b.close * ratio

        else:
            raise ValueError(f"Unknown exit timing model: {model!r}")

    return prices


def compute_all_exit_prices(
    intraday_bars: list[IntradayBar],
    daily_bars: list[Bar] | None = None,
) -> dict[str, dict[date, float] | None]:
    """Pre-compute exit prices for all timing models.

    Returns:
        Dict mapping model name to date->price map. "close" maps to None
        (engine uses bar.close by default).
    """
    result: dict[str, dict[date, float] | None] = {"close": None}
    for model in EXIT_TIMING_MODELS:
        if model != "close":
            result[model] = compute_exit_prices(intraday_bars, model, daily_bars)
    return result
