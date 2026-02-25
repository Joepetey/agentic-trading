"""Sanity check: submit a tiny paper market order via Alpaca."""

import structlog
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.alpaca import AlpacaTradingClient
from src.core import Environment, load_settings, setup_logging

log = structlog.get_logger()


def main() -> None:
    setup_logging()
    cfg = load_settings()

    if cfg.alpaca.env != Environment.PAPER:
        log.error("refusing_live_order", env=cfg.alpaca.env.value)
        raise SystemExit(1)

    client = AlpacaTradingClient(cfg.alpaca)

    order = client.submit_order(
        MarketOrderRequest(
            symbol="AAPL",
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    )

    print(f"\nOrder submitted:")
    print(f"  id:     {order.id}")
    print(f"  symbol: {order.symbol}")
    print(f"  side:   {order.side}")
    print(f"  qty:    {order.qty}")
    print(f"  status: {order.status}")


if __name__ == "__main__":
    main()
