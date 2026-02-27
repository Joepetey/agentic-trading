from __future__ import annotations

from dataclasses import dataclass

from core.types import Bar, OrderIntent, StrategyState


@dataclass
class OPPWCarlosConfig:
    profit_target_A: float = 0.081
    profit_target_C: float = 0.025
    stop_trigger_close: float = -0.013
    stop_exit: float = -0.015
    qty: float = 100.0
    weakness_mode: bool = True


class OPPWCarlosStrategy:
    def __init__(self, config: OPPWCarlosConfig | None = None) -> None:
        self.cfg = config or OPPWCarlosConfig()

    # ------------------------------------------------------------------
    # Week start: enter at open, place TP_A
    # ------------------------------------------------------------------
    def on_week_start(
        self, bar: Bar, state: StrategyState
    ) -> tuple[list[OrderIntent], StrategyState]:
        intents: list[OrderIntent] = []

        if state.position_open:
            return intents, state

        # Entry: buy at open
        intents.append(
            OrderIntent(
                action="SUBMIT",
                symbol=bar.symbol,
                side="buy",
                type="market",
                time_in_force="opg",
                qty=self.cfg.qty,
                tag="ENTRY",
            )
        )

        # Simulate fill at open
        entry_price = bar.open
        state = state.model_copy(
            update={
                "entry_date": bar.ts.date(),
                "entry_price": entry_price,
                "mode": "NORMAL",
                "position_open": True,
                "active_exit_tag": "TP_A",
            }
        )

        # Place TP_A limit sell
        tp_a_price = round(entry_price * (1 + self.cfg.profit_target_A), 2)
        intents.append(
            OrderIntent(
                action="SUBMIT",
                symbol=bar.symbol,
                side="sell",
                type="limit",
                time_in_force="gtc",
                qty=self.cfg.qty,
                limit_price=tp_a_price,
                tag="TP_A",
            )
        )

        return intents, state

    # ------------------------------------------------------------------
    # Daily close: check stop trigger, weakness mode
    # ------------------------------------------------------------------
    def on_daily_close(
        self, bar: Bar, state: StrategyState
    ) -> tuple[list[OrderIntent], StrategyState]:
        intents: list[OrderIntent] = []

        if not state.position_open or state.entry_price is None:
            return intents, state

        entry = state.entry_price

        # --- Stop trigger (Carlos): close breaches stop_trigger_close ---
        # Only on days after entry — entry day uses weakness mode instead
        if (
            bar.close <= entry * (1 + self.cfg.stop_trigger_close)
            and state.entry_date != bar.ts.date()
        ):
            # Cancel any active TP
            if state.active_exit_tag:
                intents.append(
                    OrderIntent(
                        action="CANCEL",
                        symbol=bar.symbol,
                        side="sell",
                        type="limit",
                        time_in_force="gtc",
                        qty=self.cfg.qty,
                        tag=state.active_exit_tag,
                    )
                )

            # Exit next open via market sell
            intents.append(
                OrderIntent(
                    action="SUBMIT",
                    symbol=bar.symbol,
                    side="sell",
                    type="market",
                    time_in_force="opg",
                    qty=self.cfg.qty,
                    tag="STOP",
                )
            )

            state = state.model_copy(
                update={
                    "position_open": False,
                    "active_exit_tag": None,
                    "notes": state.notes + " STOP triggered at close",
                }
            )
            return intents, state

        # --- Weakness mode (Carlos): entry day close below entry price ---
        if (
            self.cfg.weakness_mode
            and state.entry_date == bar.ts.date()
            and bar.close < entry
            and state.mode == "NORMAL"
        ):
            # Cancel TP_A
            intents.append(
                OrderIntent(
                    action="CANCEL",
                    symbol=bar.symbol,
                    side="sell",
                    type="limit",
                    time_in_force="gtc",
                    qty=self.cfg.qty,
                    tag="TP_A",
                )
            )

            # Place TP_C at reduced target
            tp_c_price = round(entry * (1 + self.cfg.profit_target_C), 2)
            intents.append(
                OrderIntent(
                    action="SUBMIT",
                    symbol=bar.symbol,
                    side="sell",
                    type="limit",
                    time_in_force="gtc",
                    qty=self.cfg.qty,
                    limit_price=tp_c_price,
                    tag="TP_C",
                )
            )

            state = state.model_copy(
                update={
                    "mode": "WEAKNESS",
                    "active_exit_tag": "TP_C",
                    "notes": state.notes + " WEAKNESS mode entered",
                }
            )
            return intents, state

        return intents, state

    # ------------------------------------------------------------------
    # Week end: liquidate at close if still holding
    # ------------------------------------------------------------------
    def on_week_end(
        self, bar: Bar, state: StrategyState
    ) -> tuple[list[OrderIntent], StrategyState]:
        """Rule 5: absolute exit at close on last trading day."""
        intents: list[OrderIntent] = []

        # Cancel any active TP
        if state.active_exit_tag:
            intents.append(
                OrderIntent(
                    action="CANCEL",
                    symbol=bar.symbol,
                    side="sell",
                    type="limit",
                    time_in_force="gtc",
                    qty=self.cfg.qty,
                    tag=state.active_exit_tag,
                )
            )

        # Sell at close
        intents.append(
            OrderIntent(
                action="SUBMIT",
                symbol=bar.symbol,
                side="sell",
                type="market",
                time_in_force="cls",
                qty=self.cfg.qty,
                tag="EOW",
            )
        )

        state = state.model_copy(
            update={
                "position_open": False,
                "active_exit_tag": None,
                "notes": state.notes + " EOW exit at close",
            }
        )

        return intents, state
