from __future__ import annotations

from datetime import date

import yfinance
from zoneinfo import ZoneInfo

from core.types import Bar

_ET = ZoneInfo("US/Eastern")


def fetch_daily_bars(
    symbol: str, start: date, end: date, adjusted: bool = False
) -> list[Bar]:
    """Download daily OHLCV from yfinance and return as Bar objects.

    When adjusted=True, prices are dividend- and split-adjusted (total return).
    Use this for income instruments like BIL where return comes from dividends.
    """
    df = yfinance.download(
        symbol, start=str(start), end=str(end), interval="1d",
        auto_adjust=adjusted,
    )
    if df.empty:
        return []

    # yfinance may return MultiIndex columns for single ticker — flatten
    if hasattr(df.columns, "levels") and len(df.columns.levels) > 1:
        df.columns = df.columns.get_level_values(0)

    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                ts=ts.to_pydatetime().replace(tzinfo=_ET),
                symbol=symbol,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
        )
    return bars
