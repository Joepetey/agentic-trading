"""Incremental bar update — fetch only new bars since last watermark."""

import sys

import structlog

from src.alpaca import AlpacaDataClient
from src.core import load_settings, setup_logging
from src.data import get_connection, sync_universe
from src.data.ingest import incremental_update


def main() -> int:
    setup_logging()
    log = structlog.get_logger()

    cfg = load_settings()
    conn = get_connection()

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

    run_id = incremental_update(
        conn=conn,
        client=client,
        symbols=symbols,
        timeframes=cfg.ingest.timeframes,
        feed=cfg.ingest.feed,
    )

    log.info("incremental_script_done", run_id=run_id)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
