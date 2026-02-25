"""Sanity check: fetch a small historical bar sample from Alpaca."""

from datetime import datetime

from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from src.alpaca import AlpacaDataClient
from src.core import load_settings, setup_logging


def main() -> None:
    setup_logging()
    cfg = load_settings()
    client = AlpacaDataClient(cfg.alpaca)

    request = StockBarsRequest(
        symbol_or_symbols=["AAPL"],
        timeframe=TimeFrame.Day,
        start=datetime(2025, 1, 2),
        end=datetime(2025, 1, 10),
    )
    bars = client.get_stock_bars(request)

    print(f"\nFetched {len(bars.df)} bars for AAPL:\n")
    print(bars.df.to_string())


if __name__ == "__main__":
    main()
