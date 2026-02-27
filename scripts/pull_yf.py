"""Pull daily bars from yfinance and store to SQLite."""

from datetime import date

from data.yf import fetch_daily_bars
from data.store import upsert_bars

SYMBOLS = ["TQQQ", "BIL", "QQQ"]
# BIL return comes from dividends — use adjusted prices for total return
ADJUSTED_SYMBOLS = {"BIL"}


def main() -> None:
    start = date(2010, 1, 1)
    end = date.today()

    for symbol in SYMBOLS:
        adjusted = symbol in ADJUSTED_SYMBOLS
        label = " (adjusted)" if adjusted else ""
        print(f"Fetching {symbol}{label} daily bars {start} -> {end} ...")
        bars = fetch_daily_bars(symbol, start, end, adjusted=adjusted)
        print(f"  Downloaded {len(bars)} bars")

        written = upsert_bars(bars)
        print(f"  Wrote {written} new bars to DB")


if __name__ == "__main__":
    main()
