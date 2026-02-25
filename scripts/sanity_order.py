"""Sanity check: submit a tiny paper market order via Alpaca."""

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.core import Environment, load_settings


def main() -> None:
    cfg = load_settings()

    if cfg.alpaca.env != Environment.PAPER:
        print("ABORT: this script only runs against paper. Set TRADING_ENV=paper.")
        raise SystemExit(1)

    client = TradingClient(cfg.alpaca.api_key, cfg.alpaca.api_secret, paper=True)

    order = client.submit_order(
        MarketOrderRequest(
            symbol="AAPL",
            qty=1,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
    )

    print(f"Order submitted:")
    print(f"  id:     {order.id}")
    print(f"  symbol: {order.symbol}")
    print(f"  side:   {order.side}")
    print(f"  qty:    {order.qty}")
    print(f"  status: {order.status}")


if __name__ == "__main__":
    main()
