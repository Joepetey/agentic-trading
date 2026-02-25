"""Backfill historical bars for all configured symbols and timeframes."""

import sys

import structlog

from src.alpaca import AlpacaDataClient
from src.core import load_settings, setup_logging
from src.data import get_connection, sync_universe
from src.data.ingest import backfill_bars


def main() -> int:
    setup_logging()
    log = structlog.get_logger()

    cfg = load_settings()
    conn = get_connection()

    # Sync universe first so FK constraints are satisfied
    sync_universe(conn, cfg.symbols.symbols)

    client = AlpacaDataClient(cfg.alpaca)

    active_rows = conn.execute(
        "SELECT symbol FROM symbols WHERE active = 1 ORDER BY symbol"
    ).fetchall()
    symbols = [r["symbol"] for r in active_rows]

    if not symbols:
        log.warning("no_active_symbols")
        conn.close()
        return 0

    log.info("starting_backfill", symbols=symbols, timeframes=cfg.ingest.timeframes)

    run_id = backfill_bars(
        conn=conn,
        client=client,
        symbols=symbols,
        timeframes=cfg.ingest.timeframes,
        lookback=cfg.ingest.lookback,
        feed=cfg.ingest.feed,
    )

    log.info("backfill_script_done", run_id=run_id)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
