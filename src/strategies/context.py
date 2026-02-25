"""StrategyContext — the single object every strategy receives."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from typing import Any

import structlog
from pydantic import BaseModel, ConfigDict, Field

from src.data.query import Bar, get_latest, get_range, get_window

logger = structlog.get_logger(__name__)

# Approximate trading bars per day, used to convert lookback_bars → calendar window.
_BARS_PER_TRADING_DAY: dict[str, float] = {
    "1Min": 390,
    "5Min": 78,
    "15Min": 26,
    "1Hour": 7,
    "1Day": 1,
    "1Week": 0.2,
}

# Max symbols per SQL IN-clause chunk.
_CHUNK_SIZE = 50


def _estimate_calendar_days(timeframe: str, lookback_bars: int) -> int:
    """Convert a bar count to a generous calendar-day window."""
    bars_per_day = _BARS_PER_TRADING_DAY.get(timeframe, 1.0)
    trading_days = max(1.0, lookback_bars / bars_per_day)
    calendar_days = trading_days * 7 / 5  # account for weekends
    return int(calendar_days * 1.5) + 5   # generous buffer


# ── DataAccess ────────────────────────────────────────────────────────


class DataAccess:
    """Read-only DAO that enforces a temporal ceiling.

    Wraps ``src.data.query`` functions so strategies can never peek
    past ``as_of``.

    **Batch prefetch** — call :meth:`prefetch` before a strategy run to
    pull the full universe in one ``get_range`` query (chunked at 50
    symbols).  Subsequent :meth:`get_window` / :meth:`get_latest` calls
    serve from the in-memory cache when possible, falling back to SQLite
    on cache miss.
    """

    def __init__(self, conn: sqlite3.Connection, as_of: datetime) -> None:
        self._conn = conn
        self._as_of = as_of
        # Cache: (symbol, timeframe) → ascending-sorted list[Bar]
        self._cache: dict[tuple[str, str], list[Bar]] = {}

    @property
    def as_of(self) -> datetime:
        return self._as_of

    # ── prefetch ──────────────────────────────────────────────────────

    def prefetch(
        self,
        symbols: list[str],
        timeframe: str,
        lookback_bars: int,
    ) -> None:
        """Batch-load bar windows for the universe into the in-memory cache.

        Issues one ``get_range`` query per chunk of 50 symbols, then
        splits the result per-symbol.  After this call, :meth:`get_window`
        and :meth:`get_latest` will serve from cache for any
        ``(symbol, timeframe)`` that was prefetched.
        """
        cal_days = _estimate_calendar_days(timeframe, lookback_bars)
        start_ts = self._as_of - timedelta(days=cal_days)

        all_bars: list[Bar] = []
        for i in range(0, len(symbols), _CHUNK_SIZE):
            chunk = symbols[i : i + _CHUNK_SIZE]
            all_bars.extend(get_range(self._conn, chunk, timeframe, start_ts, self._as_of))

        # Group by symbol (bars arrive sorted by (symbol, ts) from get_range)
        grouped: dict[str, list[Bar]] = {s: [] for s in symbols}
        for bar in all_bars:
            if bar.symbol in grouped:
                grouped[bar.symbol].append(bar)

        for sym, bars in grouped.items():
            self._cache[(sym, timeframe)] = bars

        logger.debug(
            "prefetch_complete",
            symbols=len(symbols),
            timeframe=timeframe,
            lookback_bars=lookback_bars,
            calendar_days=cal_days,
            total_bars=len(all_bars),
        )

    # ── per-symbol access (cache-aware) ───────────────────────────────

    def get_window(
        self,
        symbol: str,
        timeframe: str,
        lookback_bars: int,
    ) -> list[Bar]:
        """Last *lookback_bars* bars ending at or before ``as_of``.

        Serves from cache when available, falls back to SQLite.
        """
        cached = self._cache.get((symbol, timeframe))
        if cached is not None:
            return cached[-lookback_bars:]
        return get_window(self._conn, symbol, timeframe, self._as_of, lookback_bars)

    def get_range(
        self,
        symbols: list[str],
        timeframe: str,
        start_ts: datetime | str,
    ) -> list[Bar]:
        """Bars for multiple symbols from *start_ts* up to ``as_of``."""
        return get_range(self._conn, symbols, timeframe, start_ts, self._as_of)

    def get_latest(
        self,
        symbol: str,
        timeframe: str,
        *,
        max_staleness: timedelta | None = None,
    ) -> Bar:
        """Most recent bar for (symbol, timeframe).

        Serves from cache when available.
        Raises ``LookupError`` if none exist.
        Raises ``StaleDataError`` if older than *max_staleness*.
        """
        cached = self._cache.get((symbol, timeframe))
        if cached:
            bar = cached[-1]
            if max_staleness is not None:
                from datetime import timezone

                cutoff = datetime.now(timezone.utc) - max_staleness
                bar_ts = bar.ts if bar.ts.tzinfo else bar.ts.replace(tzinfo=timezone.utc)
                if bar_ts < cutoff:
                    from src.core.errors import StaleDataError

                    raise StaleDataError(
                        f"{symbol}/{timeframe} latest bar {bar.ts.isoformat()} "
                        f"is older than {max_staleness}"
                    )
            return bar
        return get_latest(self._conn, symbol, timeframe, max_staleness=max_staleness)

    # ── bulk access ───────────────────────────────────────────────────

    def get_universe_windows(
        self,
        timeframe: str,
        lookback_bars: int,
    ) -> dict[str, list[Bar]]:
        """Return cached windows for every symbol that was prefetched.

        Useful for cross-sectional strategies that need the whole universe
        at once.  Returns ``{symbol: list[Bar]}`` with each list trimmed
        to the last *lookback_bars* entries.

        Only returns data that was previously loaded via :meth:`prefetch`.
        """
        result: dict[str, list[Bar]] = {}
        for (sym, tf), bars in self._cache.items():
            if tf == timeframe:
                result[sym] = bars[-lookback_bars:]
        return result


# ── Constraints ───────────────────────────────────────────────────────


class Constraints(BaseModel):
    """Pre-evaluation filters that strategies may consult."""

    model_config = ConfigDict(frozen=True)

    max_names: int | None = Field(default=None, description="Max symbols to signal on")
    min_avg_volume: int | None = Field(default=None, description="Min average daily volume")
    min_price: float | None = Field(default=None, description="Min price filter")
    extras: dict[str, Any] = Field(default_factory=dict, description="Escape hatch for ad-hoc filters")


# ── StrategyContext ───────────────────────────────────────────────────


class StrategyContext(BaseModel):
    """Everything a strategy needs.  Built by the runner, consumed by ``Strategy.run()``."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    now_ts: datetime
    universe: tuple[str, ...]
    timeframe: str
    data: DataAccess
    config: dict[str, Any] = Field(default_factory=dict, description="Per-strategy params from TOML")
    constraints: Constraints = Field(default_factory=Constraints)
    feature_version: str | None = None
