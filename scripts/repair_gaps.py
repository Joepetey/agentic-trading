"""Re-fetch and fill all open data gaps."""

import sys

import structlog

from src.alpaca import AlpacaDataClient
from src.core import load_settings, setup_logging
from src.data import get_connection, repair_gaps


def main() -> int:
    setup_logging()
    log = structlog.get_logger()

    cfg = load_settings()
    conn = get_connection()
    client = AlpacaDataClient(cfg.alpaca)

    run_id = repair_gaps(conn, client, feed=cfg.ingest.feed)

    open_remaining = conn.execute(
        "SELECT COUNT(*) as cnt FROM data_gaps WHERE status = 'open'"
    ).fetchone()["cnt"]

    log.info("repair_script_done", run_id=run_id, open_remaining=open_remaining)
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
