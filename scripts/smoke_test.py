"""Smoke test: config → bars → place order → cancel order → exit cleanly."""

from __future__ import annotations

import sys
from datetime import datetime

import structlog
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.alpaca import AlpacaDataClient, AlpacaTradingClient
from src.core import Environment, load_settings, setup_logging

log = structlog.get_logger()


def main() -> int:
    setup_logging()
    log.info("smoke_test_start")

    # 1. Load config
    cfg = load_settings()
    log.info("config_loaded", env=cfg.alpaca.env.value, symbols=cfg.symbols.symbols)

    if cfg.alpaca.env != Environment.PAPER:
        log.error("refusing_live_smoke_test", env=cfg.alpaca.env.value)
        return 1

    # 2. Pull bars
    data_client = AlpacaDataClient(cfg.alpaca)
    bars = data_client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=["AAPL"],
            timeframe=TimeFrame.Day,
            start=datetime(2025, 1, 2),
            end=datetime(2025, 1, 10),
        )
    )
    assert len(bars.df) > 0, "Expected at least one bar"
    log.info("bars_ok", count=len(bars.df))

    # 3. Place a paper order
    trading_client = AlpacaTradingClient(cfg.alpaca)
    order = trading_client.submit_order(
        MarketOrderRequest(
            symbol="AAPL",
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    )
    order_id = str(order.id)
    log.info("order_placed", order_id=order_id, status=str(order.status))

    # 4. Cancel the order
    try:
        trading_client.cancel_order(order_id)
        log.info("order_cancel_ok", order_id=order_id)
    except Exception:
        # Order may have already filled instantly on paper — that's fine
        log.warning("order_cancel_skipped", order_id=order_id, reason="already_filled_or_done")

    log.info("smoke_test_pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
