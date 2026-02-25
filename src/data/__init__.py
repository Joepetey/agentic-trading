"""Data — cache layer and feature computation."""

from src.data.db import DB_PATH, get_connection
from src.data.ingest import backfill_bars, incremental_update, sync_universe
from src.data.integrity import check_monotonic, detect_gaps, repair_gaps
from src.data.query import Bar, get_latest, get_range, get_window

__all__ = [
    "DB_PATH",
    "Bar",
    "backfill_bars",
    "check_monotonic",
    "detect_gaps",
    "get_connection",
    "get_latest",
    "get_range",
    "get_window",
    "incremental_update",
    "repair_gaps",
    "sync_universe",
]
