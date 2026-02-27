from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional, Protocol

from pydantic import BaseModel


class Bar(BaseModel):
    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float | int


class IntradayBar(BaseModel):
    ts: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float | int


class OrderIntent(BaseModel):
    """What a strategy wants the broker to do."""

    action: Literal["SUBMIT", "CANCEL", "REPLACE"]
    symbol: str
    side: Literal["buy", "sell"]
    type: Literal["market", "limit", "stop"]
    time_in_force: Literal["day", "gtc", "opg", "cls"]
    qty: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    tag: str  # e.g. "ENTRY", "TP_A", "TP_C", "STOP", "EOW"


class PositionState(BaseModel):
    symbol: str
    qty: float
    avg_entry_price: float


class StrategyState(BaseModel):
    """Weekly state for a strategy."""

    week_id: str  # e.g. "2026-W09"
    symbol: str
    entry_date: Optional[date] = None
    entry_price: Optional[float] = None
    mode: Literal["NORMAL", "WEAKNESS"]
    position_open: bool
    active_exit_tag: Optional[str] = None  # TP_A or TP_C or STOP
    stop_pending: bool = False
    notes: str = ""
    portfolio_value: Optional[float] = None


class Strategy(Protocol):
    """Interface any weekly strategy must implement."""

    def on_week_start(
        self, bar: Bar, state: StrategyState
    ) -> tuple[list[OrderIntent], StrategyState]: ...

    def on_daily_close(
        self, bar: Bar, state: StrategyState
    ) -> tuple[list[OrderIntent], StrategyState]: ...

    def on_week_end(
        self, bar: Bar, state: StrategyState
    ) -> tuple[list[OrderIntent], StrategyState]: ...
