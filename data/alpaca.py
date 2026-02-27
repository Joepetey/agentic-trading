"""Fetch intraday bars from Alpaca Market Data API."""
from __future__ import annotations

import os
from datetime import date, datetime, time

from dotenv import load_dotenv
from zoneinfo import ZoneInfo

from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame

from core.types import IntradayBar

load_dotenv()

_ET = ZoneInfo("US/Eastern")
_MARKET_OPEN = time(9, 30)
_MARKET_CLOSE = time(16, 0)


def _get_client() -> StockHistoricalDataClient:
    api_key = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    return StockHistoricalDataClient(api_key, secret_key)


def fetch_intraday_bars(
    symbol: str,
    start: date,
    end: date,
) -> list[IntradayBar]:
    """Fetch 5-min intraday bars from Alpaca (regular market hours only).

    Returns:
        List of IntradayBar objects sorted by timestamp, filtered to 9:30–16:00 ET.
    """
    client = _get_client()
    tf = TimeFrame(5, TimeFrame.Minute.unit)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=tf,
        start=datetime(start.year, start.month, start.day, tzinfo=_ET),
        end=datetime(end.year, end.month, end.day, 23, 59, 59, tzinfo=_ET),
        feed=DataFeed.IEX,
    )

    response = client.get_stock_bars(request)
    bar_list = response.data.get(symbol, [])

    bars: list[IntradayBar] = []
    for b in bar_list:
        ts = b.timestamp.astimezone(_ET)

        # Filter to regular market hours only
        bar_time = ts.time()
        if bar_time < _MARKET_OPEN or bar_time >= _MARKET_CLOSE:
            continue

        bars.append(
            IntradayBar(
                ts=ts,
                symbol=symbol,
                open=float(b.open),
                high=float(b.high),
                low=float(b.low),
                close=float(b.close),
                volume=int(b.volume),
            )
        )

    bars.sort(key=lambda x: x.ts)
    return bars
