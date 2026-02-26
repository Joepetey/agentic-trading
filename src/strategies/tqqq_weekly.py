"""TQQQ Weekly Cycle strategy with Carlos modifications.

Trade TQQQ in weekly cycles:
- Enter at the open of the first trading day each week
- Hunt an 8.1% Type-A profit target
- On weakness (day-1 close < entry), switch to a +2.5% Carlos target
- Stop trigger: close <= entry*(1 - 1.3%) -> exit next session ~-1.5%
- Fallback: exit at the close of the last trading day of the week
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import structlog

from src.strategies.base import Strategy
from src.strategies.context import StrategyContext
from src.strategies.signal import EntryType, Side, Signal

logger = structlog.get_logger(__name__)


# ── Helpers ──────────────────────────────────────────────────────────


def _iso_week_key(dt: datetime) -> tuple[int, int]:
    """Return (iso_year, iso_week) for grouping bars into trading weeks."""
    cal = dt.isocalendar()
    return (cal[0], cal[1])


# ── Strategy ─────────────────────────────────────────────────────────


class TQQQWeekly(Strategy):
    """TQQQ weekly-cycle strategy (Carlos modifications).

    Evaluates the current week's bars and emits a single intent signal.
    The strategy does NOT simulate fills or track position state —
    that belongs to the execution / backtest layer.
    """

    strategy_id = "tqqq_weekly"
    version = "1.0.0"

    def __init__(
        self,
        *,
        symbol: str = "TQQQ",
        profit_target_a: float = 0.081,
        profit_target_carlos: float = 0.025,
        stop_trigger_close: float = -0.013,
        stop_exit: float = -0.015,
        stop_exit_method: str = "moo",
    ) -> None:
        self._symbol = symbol
        self._profit_target_a = profit_target_a
        self._profit_target_carlos = profit_target_carlos
        self._stop_trigger_close = stop_trigger_close
        self._stop_exit = stop_exit
        self._stop_exit_method = stop_exit_method

    # ── ABC implementation ───────────────────────────────────────────

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 10

    def params(self) -> dict[str, Any]:
        return {
            "symbol": self._symbol,
            "profit_target_a": self._profit_target_a,
            "profit_target_carlos": self._profit_target_carlos,
            "stop_trigger_close": self._stop_trigger_close,
            "stop_exit": self._stop_exit,
            "stop_exit_method": self._stop_exit_method,
        }

    def run(self, ctx: StrategyContext) -> list[Signal]:
        """Evaluate the current week and emit 0 or 1 signals."""
        symbol = self._symbol
        if symbol not in ctx.universe:
            return []

        bars = ctx.data.get_window(symbol, "1Day", self.required_lookback_bars())
        if not bars:
            return []

        # Bars belonging to the current ISO week.
        week_key = _iso_week_key(ctx.now_ts)
        week_bars = [b for b in bars if _iso_week_key(b.ts) == week_key]

        if not week_bars:
            return [self._entry_signal(symbol)]

        entry_price = week_bars[0].open

        # Determine mode from entry-day close.
        mode = "NORMAL"
        target_price = entry_price * (1 + self._profit_target_a)
        if week_bars[0].close < entry_price:
            mode = "WEAKNESS"
            target_price = entry_price * (1 + self._profit_target_carlos)

        # Check if latest close triggered the stop.
        stop_trigger_level = entry_price * (1 + self._stop_trigger_close)
        if week_bars[-1].close <= stop_trigger_level:
            return [self._stop_exit_signal(symbol, entry_price, mode)]

        # Last trading day -> end-of-week exit.
        if self._is_last_trading_day(ctx.now_ts, week_bars):
            return [self._eow_exit_signal(symbol, entry_price, mode)]

        # Mid-week hold with current targets.
        return [self._hold_signal(symbol, entry_price, mode, target_price, week_bars)]

    # ── Last-day detection ───────────────────────────────────────────

    @staticmethod
    def _is_last_trading_day(now_ts: datetime, week_bars: list) -> bool:
        """Friday (weekday 4) or 5+ bars already in the week."""
        return now_ts.weekday() == 4 or len(week_bars) >= 5

    # ── Signal builders ──────────────────────────────────────────────

    def _entry_signal(self, symbol: str) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=Side.LONG,
            strength=0.6,
            confidence=0.7,
            horizon_bars=5,
            entry=EntryType.MARKET,
            tags=("tqqq_weekly", "entry"),
            explain="Weekly cycle entry: buy TQQQ at market open.",
        )

    def _hold_signal(
        self,
        symbol: str,
        entry_price: float,
        mode: str,
        target_price: float,
        week_bars: list,
    ) -> Signal:
        remaining = max(1, 5 - len(week_bars))
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=Side.LONG,
            strength=0.5,
            confidence=0.7,
            horizon_bars=remaining,
            entry=EntryType.LIMIT,
            take_profit_price=round(target_price, 4),
            stop_price=round(entry_price * (1 + self._stop_exit), 4),
            time_stop_bars=remaining,
            tags=("tqqq_weekly", "hold", f"mode:{mode.lower()}"),
            explain=(
                f"Holding — mode={mode}, "
                f"target=${target_price:.2f}, "
                f"stop=${entry_price * (1 + self._stop_exit):.2f}."
            ),
        )

    def _stop_exit_signal(
        self, symbol: str, entry_price: float, mode: str
    ) -> Signal:
        stop_price = round(entry_price * (1 + self._stop_exit), 4)
        if self._stop_exit_method == "stop_order":
            entry_type = EntryType.STOP
        else:
            entry_type = EntryType.MARKET

        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=Side.FLAT,
            strength=-1.0,
            confidence=0.9,
            horizon_bars=1,
            entry=entry_type,
            stop_price=stop_price,
            tags=("tqqq_weekly", "exit", "stop"),
            explain=(
                f"Stop triggered: close breached "
                f"{self._stop_trigger_close:.1%} threshold. "
                f"Exiting at {'stop order' if self._stop_exit_method == 'stop_order' else 'market'} "
                f"(~{self._stop_exit:.1%})."
            ),
        )

    def _eow_exit_signal(
        self, symbol: str, entry_price: float, mode: str
    ) -> Signal:
        return Signal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            side=Side.FLAT,
            strength=-0.5,
            confidence=0.8,
            horizon_bars=1,
            entry=EntryType.MARKET,
            tags=("tqqq_weekly", "exit", "end_of_week"),
            explain=(
                f"End-of-week exit: mode={mode}, "
                f"entry=${entry_price:.2f}."
            ),
        )
