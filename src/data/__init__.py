"""Data — cache layer and feature computation."""

from src.data.db import DB_PATH, get_connection
from src.data.ingest import backfill_bars, incremental_update, sync_universe
from src.data.integrity import check_monotonic, detect_gaps, repair_gaps
from src.data.query import Bar, get_latest, get_range, get_window
from src.data.signals import (
    complete_run,
    create_run,
    get_latest_signals,
    get_signals,
    write_signals,
    write_signals_from_result,
)

__all__ = [
    "Bar",
    "DB_PATH",
    "backfill_bars",
    "check_monotonic",
    "complete_run",
    "create_run",
    "detect_gaps",
    "get_connection",
    "get_latest",
    "get_latest_signals",
    "get_range",
    "get_signals",
    "get_window",
    "incremental_update",
    "repair_gaps",
    "sync_universe",
    "write_signals",
    "write_signals_from_result",
]
