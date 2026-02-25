"""Sanity check: fetch a small historical bar sample from Alpaca."""

from datetime import datetime

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.core import load_settings


def main() -> None:
    cfg = load_settings()
    client = StockHistoricalDataClient(cfg.alpaca.api_key, cfg.alpaca.api_secret)

    request = StockBarsRequest(
        symbol_or_symbols=["AAPL"],
        timeframe=TimeFrame.Day,
        start=datetime(2025, 1, 2),
        end=datetime(2025, 1, 10),
    )
    bars = client.get_stock_bars(request)

    print(f"Fetched {len(bars.df)} bars for AAPL:\n")
    print(bars.df.to_string())


if __name__ == "__main__":
    main()
