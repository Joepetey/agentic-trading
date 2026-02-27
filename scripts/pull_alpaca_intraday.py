"""Pull 5-minute intraday bars from Alpaca and store to SQLite."""
from __future__ import annotations

from datetime import date

from data.alpaca import fetch_intraday_bars
from data.store import upsert_intraday_bars

SYMBOL = "TQQQ"


def main() -> None:
    # Alpaca free tier: ~5 years of minute data
    start = date(2016, 1, 1)
    end = date.today()

    print(f"Fetching {SYMBOL} 5-min bars from Alpaca ({start} -> {end}) ...")
    bars = fetch_intraday_bars(SYMBOL, start, end)
    print(f"  Downloaded {len(bars)} bars")

    if bars:
        first_ts = bars[0].ts
        last_ts = bars[-1].ts
        print(f"  Range: {first_ts} -> {last_ts}")

    written = upsert_intraday_bars(bars, timeframe="5Min")
    print(f"  Wrote {written} new bars to DB")


if __name__ == "__main__":
    main()
