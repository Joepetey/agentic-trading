from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from core.calendar import last_trading_day_of_week, nth_trading_day_of_week
from core.types import Bar, OrderIntent, Strategy, StrategyState


@dataclass
class Trade:
    symbol: str
    entry_ts: date
    entry_price: float
    exit_ts: date
    exit_price: float
    exit_reason: str  # TP_A / TP_C / STOP / EOW
    qty: float
    pnl: float
    return_pct: float


@dataclass
class OpenOrder:
    intent: OrderIntent
    limit_price: float | None = None
    stop_price: float | None = None


@dataclass
class BacktestResult:
    trades: list[Trade]
    equity_curve: list[tuple[date, float]]  # (date, portfolio_value)
    initial_cash: float
    final_value: float


@dataclass
class _SimState:
    cash: float
    pos_qty: float = 0.0
    pos_avg_entry: float = 0.0
    bil_qty: float = 0.0
    bil_avg_entry: float = 0.0
    bil_entry_date: date | None = None
    open_orders: list[OpenOrder] = field(default_factory=list)
    strategy_state: StrategyState | None = None
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[date, float]] = field(default_factory=list)


def run_backtest(
    bars: list[Bar],
    strategy: Strategy,
    initial_cash: float = 100_000.0,
    slippage_bps: float = 0.0,
    commission_per_trade: float = 0.0,
    full_exposure: bool = False,
    sweep_bars: dict[date, Bar] | None = None,
    entry_day_offset: int = 0,
    entry_price_override: dict[date, float] | None = None,
    exit_price_override: dict[date, float] | None = None,
) -> BacktestResult:
    """Run a deterministic daily-loop backtest."""
    if not bars:
        return BacktestResult([], [], initial_cash, initial_cash)

    symbol = bars[0].symbol
    sim = _SimState(cash=initial_cash)

    # Pre-compute first/last trading days per ISO week to avoid repeated lookups
    week_cache: dict[str, tuple[date, date]] = {}

    def _week_bounds(d: date) -> tuple[str, date, date]:
        wid = f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"
        if wid not in week_cache:
            entry = nth_trading_day_of_week(d, entry_day_offset)
            last = last_trading_day_of_week(d)
            if entry > last:
                entry = last
            week_cache[wid] = (entry, last)
        first, last = week_cache[wid]
        return wid, first, last

    def _apply_slippage(price: float, side: str) -> float:
        mult = 1 + slippage_bps / 10_000 if side == "buy" else 1 - slippage_bps / 10_000
        return price * mult

    def _fill_buy(price: float, qty: float) -> None:
        fill_price = _apply_slippage(price, "buy")
        cost = fill_price * qty + commission_per_trade
        sim.cash -= cost
        sim.pos_avg_entry = fill_price
        sim.pos_qty = qty

    def _fill_sell(price: float, qty: float, reason: str, exit_date: date) -> None:
        fill_price = _apply_slippage(price, "sell")
        proceeds = fill_price * qty - commission_per_trade
        sim.cash += proceeds

        pnl = (fill_price - sim.pos_avg_entry) * qty - 2 * commission_per_trade
        ret = (fill_price / sim.pos_avg_entry - 1) if sim.pos_avg_entry else 0.0

        sim.trades.append(
            Trade(
                symbol=symbol,
                entry_ts=sim.strategy_state.entry_date if sim.strategy_state else exit_date,
                entry_price=sim.pos_avg_entry,
                exit_ts=exit_date,
                exit_price=fill_price,
                exit_reason=reason,
                qty=qty,
                pnl=pnl,
                return_pct=ret,
            )
        )
        sim.pos_qty = 0.0
        sim.pos_avg_entry = 0.0

    def _cancel_orders() -> None:
        sim.open_orders.clear()

    def _sweep_into_bil(bil_bar: Bar, today: date) -> None:
        """Buy BIL with all available cash at bil_bar.close."""
        fill_price = _apply_slippage(bil_bar.close, "buy")
        qty = int(sim.cash // fill_price)
        if qty <= 0:
            return
        cost = fill_price * qty + commission_per_trade
        sim.cash -= cost
        sim.bil_qty = qty
        sim.bil_avg_entry = fill_price
        sim.bil_entry_date = today

    def _sweep_out_of_bil(bil_bar: Bar, today: date) -> None:
        """Sell all BIL at bil_bar.open, returning cash."""
        if sim.bil_qty <= 0:
            return
        fill_price = _apply_slippage(bil_bar.open, "sell")
        proceeds = fill_price * sim.bil_qty - commission_per_trade
        sim.cash += proceeds

        pnl = (fill_price - sim.bil_avg_entry) * sim.bil_qty - 2 * commission_per_trade
        ret = (fill_price / sim.bil_avg_entry - 1) if sim.bil_avg_entry else 0.0

        sim.trades.append(
            Trade(
                symbol="BIL",
                entry_ts=sim.bil_entry_date or today,
                entry_price=sim.bil_avg_entry,
                exit_ts=today,
                exit_price=fill_price,
                exit_reason="SWEEP_OUT",
                qty=sim.bil_qty,
                pnl=pnl,
                return_pct=ret,
            )
        )
        sim.bil_qty = 0.0
        sim.bil_avg_entry = 0.0
        sim.bil_entry_date = None

    def _process_intents(intents: list[OrderIntent], bar: Bar, phase: str) -> None:
        """Process intents from strategy calls."""
        for intent in intents:
            if intent.action == "CANCEL":
                # Remove matching open orders by tag
                sim.open_orders = [
                    o for o in sim.open_orders if o.intent.tag != intent.tag
                ]
            elif intent.action == "SUBMIT":
                if intent.side == "buy" and intent.type == "market":
                    # Market buy — fill at override price if available, else bar.open
                    if phase == "week_start":
                        today = bar.ts.date()
                        entry_px = bar.open
                        if entry_price_override and today in entry_price_override:
                            entry_px = entry_price_override[today]
                        qty = intent.qty
                        if full_exposure:
                            fill_price = _apply_slippage(entry_px, "buy")
                            qty = int(sim.cash // fill_price)
                            if qty <= 0:
                                continue
                        _fill_buy(entry_px, qty)
                elif intent.side == "sell" and intent.type == "market":
                    qty = sim.pos_qty if full_exposure else intent.qty
                    if intent.time_in_force == "opg":
                        # Queue for next day open
                        order_intent = intent.model_copy(update={"qty": qty})
                        sim.open_orders.append(OpenOrder(intent=order_intent))
                    elif intent.time_in_force == "cls":
                        # Fill at this bar's close (or exit override on week_end)
                        exit_px = bar.close
                        if phase == "week_end" and exit_price_override:
                            today = bar.ts.date()
                            if today in exit_price_override:
                                exit_px = exit_price_override[today]
                        _fill_sell(exit_px, qty, intent.tag, bar.ts.date())
                elif intent.side == "sell" and intent.type == "limit":
                    qty = sim.pos_qty if full_exposure else intent.qty
                    order_intent = intent.model_copy(update={"qty": qty})
                    sim.open_orders.append(
                        OpenOrder(intent=order_intent, limit_price=intent.limit_price)
                    )
                elif intent.side == "sell" and intent.type == "stop":
                    qty = sim.pos_qty if full_exposure else intent.qty
                    order_intent = intent.model_copy(update={"qty": qty})
                    sim.open_orders.append(
                        OpenOrder(intent=order_intent, stop_price=intent.stop_price)
                    )

    def _check_open_orders(bar: Bar) -> None:
        """Check and fill open orders against today's bar.

        Handles OCO: when any sell fills, all other open orders are cancelled.
        Stop order simulation:
          - If bar opens at/below stop_price → fill at open (gap down)
          - If bar.low <= stop_price → fill at stop_price
        """
        if not sim.open_orders:
            return

        filled = False

        # 1) Market-on-open sells (highest priority)
        for order in list(sim.open_orders):
            if (
                order.intent.side == "sell"
                and order.intent.type == "market"
                and order.intent.time_in_force == "opg"
            ):
                _fill_sell(bar.open, order.intent.qty, order.intent.tag, bar.ts.date())
                sim.open_orders.clear()  # OCO
                return

        # 2) Check for gap-down through stop at open (must fill before anything)
        for order in list(sim.open_orders):
            if (
                order.intent.side == "sell"
                and order.intent.type == "stop"
                and order.stop_price is not None
                and bar.open <= order.stop_price
            ):
                _fill_sell(bar.open, order.intent.qty, order.intent.tag, bar.ts.date())
                sim.open_orders.clear()
                return

        # 3) Limit sells (TP) — checked before intraday stops (earlier order has priority)
        for order in list(sim.open_orders):
            if (
                order.intent.side == "sell"
                and order.intent.type == "limit"
                and order.limit_price is not None
            ):
                if bar.high >= order.limit_price:
                    # Limit sell fills at limit or better (open if gapped above)
                    fill_px = max(order.limit_price, bar.open)
                    _fill_sell(
                        fill_px,
                        order.intent.qty,
                        order.intent.tag,
                        bar.ts.date(),
                    )
                    filled = True
                    break

        if filled:
            sim.open_orders.clear()  # OCO
            return

        # 4) Intraday stop fills
        for order in list(sim.open_orders):
            if (
                order.intent.side == "sell"
                and order.intent.type == "stop"
                and order.stop_price is not None
            ):
                if bar.low <= order.stop_price:
                    _fill_sell(order.stop_price, order.intent.qty, order.intent.tag, bar.ts.date())
                    filled = True
                    break

        if filled:
            sim.open_orders.clear()  # OCO

    # ---- Main loop ----
    for bar in bars:
        today = bar.ts.date()
        week_id, first_day, last_day = _week_bounds(today)
        bil_bar = sweep_bars.get(today) if sweep_bars else None

        # Check pending open orders (market-on-open sells, limit fills)
        if sim.pos_qty > 0:
            _check_open_orders(bar)

        # If position was closed by an open order, sync strategy state
        if sim.pos_qty == 0 and sim.strategy_state and sim.strategy_state.position_open:
            sim.strategy_state = sim.strategy_state.model_copy(
                update={"position_open": False, "active_exit_tag": None, "stop_pending": False}
            )

        # Week start
        is_first = today == first_day
        is_last = today == last_day

        # Sweep OUT of BIL before TQQQ entry (cash must be available for full_exposure sizing)
        if is_first and sim.bil_qty > 0 and bil_bar:
            _sweep_out_of_bil(bil_bar, today)

        if is_first:
            bil_val = sim.bil_qty * bil_bar.close if bil_bar and sim.bil_qty > 0 else 0.0
            pv = sim.cash + sim.pos_qty * bar.open + bil_val

            state = StrategyState(
                week_id=week_id,
                symbol=symbol,
                mode="NORMAL",
                position_open=sim.pos_qty > 0,
                portfolio_value=pv,
            )
            if sim.pos_qty > 0 and sim.strategy_state:
                state = state.model_copy(
                    update={
                        "entry_date": sim.strategy_state.entry_date,
                        "entry_price": sim.strategy_state.entry_price,
                        "mode": sim.strategy_state.mode,
                        "active_exit_tag": sim.strategy_state.active_exit_tag,
                        "stop_pending": sim.strategy_state.stop_pending,
                    }
                )
            sim.strategy_state = state

            intents, sim.strategy_state = strategy.on_week_start(bar, sim.strategy_state)
            _process_intents(intents, bar, "week_start")

            # After entry, check if limit TP fills today
            if sim.pos_qty > 0:
                _check_open_orders(bar)

        # Daily close logic (only if holding, skip on last day — rule 5 EOW is absolute)
        if sim.pos_qty > 0 and sim.strategy_state and not is_last:
            intents, sim.strategy_state = strategy.on_daily_close(bar, sim.strategy_state)
            _process_intents(intents, bar, "daily_close")

        # Week end — always exit at close on last trading day (rule 5)
        if is_last and sim.pos_qty > 0 and sim.strategy_state:
            intents, sim.strategy_state = strategy.on_week_end(bar, sim.strategy_state)
            _process_intents(intents, bar, "week_end")

        # Sweep idle cash into BIL if flat on TQQQ
        if sweep_bars and bil_bar and sim.pos_qty == 0 and sim.bil_qty == 0:
            _sweep_into_bil(bil_bar, today)

        # Record equity
        bil_value = sim.bil_qty * bil_bar.close if bil_bar and sim.bil_qty > 0 else 0.0
        portfolio_value = sim.cash + sim.pos_qty * bar.close + bil_value
        sim.equity_curve.append((today, portfolio_value))

    return BacktestResult(
        trades=sim.trades,
        equity_curve=sim.equity_curve,
        initial_cash=initial_cash,
        final_value=sim.equity_curve[-1][1] if sim.equity_curve else initial_cash,
    )
