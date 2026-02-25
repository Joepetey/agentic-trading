"""Run integrity checks and gap detection on the bars database."""

from src.core import load_settings, setup_logging
from src.data import get_connection, detect_gaps


def main() -> None:
    setup_logging()
    cfg = load_settings()
    conn = get_connection()

    active_rows = conn.execute(
        "SELECT symbol FROM symbols WHERE active = 1 ORDER BY symbol"
    ).fetchall()
    symbols = [r["symbol"] for r in active_rows]

    if not symbols:
        print("No active symbols.")
        conn.close()
        return

    new_gaps = detect_gaps(conn, symbols, cfg.ingest.timeframes)

    # Summary
    all_open = conn.execute(
        "SELECT COUNT(*) as cnt FROM data_gaps WHERE status = 'open'"
    ).fetchone()["cnt"]

    print(f"\nNew gaps detected: {len(new_gaps)}")
    print(f"Total open gaps:   {all_open}")

    if new_gaps:
        print("\nNew gaps:")
        for g in new_gaps:
            print(f"  {g['symbol']:6s} {g['timeframe']:5s}  "
                  f"{g['gap_start']}  →  {g['gap_end']}  "
                  f"(~{g['expected_bars']} bars)")

    conn.close()


if __name__ == "__main__":
    main()
