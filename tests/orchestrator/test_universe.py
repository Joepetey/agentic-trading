"""Tests for universe filtering."""

from __future__ import annotations

from datetime import datetime, timezone

from src.orchestrator.models import ExclusionReason
from src.orchestrator.universe import filter_universe
from src.strategies.context import Constraints
from tests.conftest import insert_bars

NOW = datetime(2024, 2, 1, 16, 0, tzinfo=timezone.utc)


class TestFilterUniverse:
    def test_all_pass_no_constraints(self, conn):
        """No constraints applied — all symbols included."""
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "MSFT", "1Day", [350.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(conn, ["AAPL", "MSFT"], Constraints(), NOW)
        assert result.included == ("AAPL", "MSFT")
        assert result.excluded == ()

    def test_min_price_excludes(self, conn):
        """Symbol below price threshold is excluded."""
        insert_bars(conn, "PENNY", "1Day", [2.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(
            conn, ["PENNY", "AAPL"],
            Constraints(min_price=5.0), NOW,
        )
        assert result.included == ("AAPL",)
        assert len(result.excluded) == 1
        assert result.excluded[0].symbol == "PENNY"
        assert result.excluded[0].reason == ExclusionReason.BELOW_MIN_PRICE

    def test_min_avg_volume_excludes(self, conn):
        """Symbol below avg volume threshold is excluded."""
        # insert_bars uses volume=1000 per bar
        insert_bars(conn, "LOW_VOL", "1Day", [100.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "HIGH_VOL", "1Day", [100.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        # Override volume for HIGH_VOL
        conn.execute("UPDATE bars SET volume = 50000 WHERE symbol = 'HIGH_VOL'")
        conn.commit()

        result = filter_universe(
            conn, ["LOW_VOL", "HIGH_VOL"],
            Constraints(min_avg_volume=10000), NOW,
        )
        assert result.included == ("HIGH_VOL",)
        assert len(result.excluded) == 1
        assert result.excluded[0].symbol == "LOW_VOL"
        assert result.excluded[0].reason == ExclusionReason.BELOW_MIN_VOLUME

    def test_insufficient_data_excluded(self, conn):
        """Symbol with no bars is excluded."""
        conn.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", ("EMPTY",))
        conn.commit()
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(conn, ["EMPTY", "AAPL"], Constraints(), NOW)
        assert result.included == ("AAPL",)
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == ExclusionReason.INSUFFICIENT_DATA

    def test_max_names_keeps_top_by_volume(self, conn):
        """When more pass than max_names, top N by volume kept."""
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "MSFT", "1Day", [350.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "GOOGL", "1Day", [140.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        # Give MSFT highest volume, AAPL second
        conn.execute("UPDATE bars SET volume = 50000 WHERE symbol = 'MSFT'")
        conn.execute("UPDATE bars SET volume = 30000 WHERE symbol = 'AAPL'")
        conn.execute("UPDATE bars SET volume = 10000 WHERE symbol = 'GOOGL'")
        conn.commit()

        result = filter_universe(
            conn, ["AAPL", "MSFT", "GOOGL"],
            Constraints(max_names=2), NOW,
        )
        assert set(result.included) == {"AAPL", "MSFT"}
        excluded_symbols = {e.symbol for e in result.excluded}
        assert "GOOGL" in excluded_symbols

    def test_combined_constraints(self, conn):
        """Price + volume + max_names applied together."""
        insert_bars(conn, "PENNY", "1Day", [2.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "MSFT", "1Day", [350.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "GOOGL", "1Day", [140.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        conn.execute("UPDATE bars SET volume = 50000 WHERE symbol = 'MSFT'")
        conn.execute("UPDATE bars SET volume = 30000 WHERE symbol = 'AAPL'")
        conn.execute("UPDATE bars SET volume = 10000 WHERE symbol = 'GOOGL'")
        conn.commit()

        result = filter_universe(
            conn, ["PENNY", "AAPL", "MSFT", "GOOGL"],
            Constraints(min_price=5.0, max_names=2), NOW,
        )
        # PENNY excluded by price, then top 2 of (AAPL, MSFT, GOOGL) by volume
        assert set(result.included) == {"AAPL", "MSFT"}

    def test_empty_universe(self, conn):
        """Empty input yields empty output."""
        result = filter_universe(conn, [], Constraints(), NOW)
        assert result.included == ()
        assert result.excluded == ()

    def test_deterministic(self, conn):
        """Same inputs produce same output."""
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "MSFT", "1Day", [350.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        r1 = filter_universe(conn, ["AAPL", "MSFT"], Constraints(), NOW)
        r2 = filter_universe(conn, ["AAPL", "MSFT"], Constraints(), NOW)
        assert r1.included == r2.included
        assert len(r1.excluded) == len(r2.excluded)

    def test_exclusion_detail_nonempty(self, conn):
        """Each excluded symbol has a non-empty detail string."""
        insert_bars(conn, "PENNY", "1Day", [2.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(
            conn, ["PENNY"], Constraints(min_price=5.0), NOW,
        )
        assert len(result.excluded) == 1
        assert result.excluded[0].detail != ""
        assert "2.00" in result.excluded[0].detail

    def test_sorted_output(self, conn):
        """Included symbols are sorted alphabetically."""
        insert_bars(conn, "MSFT", "1Day", [350.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(conn, ["MSFT", "AAPL"], Constraints(), NOW)
        assert result.included == ("AAPL", "MSFT")


class TestExcludeSymbols:
    def test_exclude_symbols_removes_before_db_queries(self, conn):
        """Symbols in exclude_symbols are removed with MANUALLY_EXCLUDED reason."""
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "MSFT", "1Day", [350.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "TSLA", "1Day", [250.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(
            conn, ["AAPL", "MSFT", "TSLA"],
            Constraints(exclude_symbols=frozenset({"TSLA", "MSFT"})), NOW,
        )
        assert result.included == ("AAPL",)
        excluded_map = {e.symbol: e for e in result.excluded}
        assert "TSLA" in excluded_map
        assert "MSFT" in excluded_map
        assert excluded_map["TSLA"].reason == ExclusionReason.MANUALLY_EXCLUDED
        assert excluded_map["MSFT"].reason == ExclusionReason.MANUALLY_EXCLUDED

    def test_exclude_symbols_combined_with_price(self, conn):
        """exclude_symbols + min_price work together."""
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "PENNY", "1Day", [2.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))
        insert_bars(conn, "MSFT", "1Day", [350.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(
            conn, ["AAPL", "PENNY", "MSFT"],
            Constraints(exclude_symbols=frozenset({"MSFT"}), min_price=5.0), NOW,
        )
        assert result.included == ("AAPL",)
        reasons = {e.symbol: e.reason for e in result.excluded}
        assert reasons["MSFT"] == ExclusionReason.MANUALLY_EXCLUDED
        assert reasons["PENNY"] == ExclusionReason.BELOW_MIN_PRICE

    def test_exclude_empty_set_no_effect(self, conn):
        """Empty exclude_symbols has no effect."""
        insert_bars(conn, "AAPL", "1Day", [150.0] * 5, start=datetime(2024, 1, 25, tzinfo=timezone.utc))

        result = filter_universe(
            conn, ["AAPL"], Constraints(exclude_symbols=frozenset()), NOW,
        )
        assert result.included == ("AAPL",)
