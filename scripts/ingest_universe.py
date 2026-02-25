"""Sync the symbols table with the config universe."""

from src.core import load_settings, setup_logging
from src.data import get_connection, sync_universe


def main() -> None:
    setup_logging()
    cfg = load_settings()
    conn = get_connection()

    added, reactivated, deactivated = sync_universe(conn, cfg.symbols.symbols)

    print(f"Added:       {added or '(none)'}")
    print(f"Reactivated: {reactivated or '(none)'}")
    print(f"Deactivated: {deactivated or '(none)'}")

    conn.close()


if __name__ == "__main__":
    main()
