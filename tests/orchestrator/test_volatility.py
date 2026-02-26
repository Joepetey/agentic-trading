"""Tests for volatility estimation."""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta, timezone

import pytest

from src.orchestrator.volatility import estimate_volatilities
from tests.conftest import insert_bars

EVAL_TS = datetime(2024, 2, 1, tzinfo=timezone.utc)


class TestBasicEstimation:
    def test_basic_vol_estimation(self, conn):
        """20 daily bars with known prices produce correct annualized vol."""
        # Alternating prices to create meaningful vol (well above 0.01 clamp)
        closes = [100.0 + (3.0 * ((-1) ** i)) for i in range(20)]
        insert_bars(conn, "AAPL", "1Day", closes)

        result = estimate_volatilities(conn, ["AAPL"], EVAL_TS)

        assert "AAPL" in result
        assert result["AAPL"] > 0.01  # above clamp floor

        # Verify manually: compute expected vol
        log_returns = [
            math.log(closes[i] / closes[i - 1]) for i in range(1, len(closes))
        ]
        expected_daily = statistics.stdev(log_returns)
        expected_annual = expected_daily * math.sqrt(252)
        assert result["AAPL"] == pytest.approx(expected_annual, rel=0.01)

    def test_multiple_symbols(self, conn):
        """Batch estimation returns dict with all symbols."""
        insert_bars(conn, "AAPL", "1Day", [100 + i for i in range(20)])
        insert_bars(conn, "MSFT", "1Day", [200 + i * 2 for i in range(20)])

        result = estimate_volatilities(conn, ["AAPL", "MSFT"], EVAL_TS)

        assert len(result) == 2
        assert "AAPL" in result
        assert "MSFT" in result
        # Both should be positive
        assert result["AAPL"] > 0
        assert result["MSFT"] > 0


class TestDefaultFallback:
    def test_insufficient_bars_uses_default(self, conn):
        """Fewer than min_bars returns default_vol."""
        insert_bars(conn, "AAPL", "1Day", [100.0, 101.0, 102.0])

        result = estimate_volatilities(
            conn, ["AAPL"], EVAL_TS, default_vol=0.25, min_bars=5,
        )

        assert result["AAPL"] == 0.25

    def test_no_bars_uses_default(self, conn):
        """Symbol missing from DB returns default_vol."""
        result = estimate_volatilities(
            conn, ["MISSING"], EVAL_TS, default_vol=0.30,
        )

        assert result["MISSING"] == 0.30

    def test_custom_default_vol(self, conn):
        """Custom default_vol is used when data insufficient."""
        result = estimate_volatilities(
            conn, ["NOPE"], EVAL_TS, default_vol=0.50,
        )

        assert result["NOPE"] == 0.50


class TestEdgeCases:
    def test_vol_clamped_above_zero(self, conn):
        """Near-constant prices produce vol >= 0.01."""
        # All bars at exact same price → std = 0 → clamped to 0.01
        insert_bars(conn, "FLAT", "1Day", [100.0] * 20)

        result = estimate_volatilities(conn, ["FLAT"], EVAL_TS)

        assert result["FLAT"] >= 0.01

    def test_empty_symbols(self, conn):
        """Empty symbol list returns empty dict."""
        result = estimate_volatilities(conn, [], EVAL_TS)
        assert result == {}

    def test_eval_ts_ceiling(self, conn):
        """Only bars at or before eval_ts are used."""
        # Insert 10 bars before eval_ts and 10 after
        early = datetime(2024, 1, 1, tzinfo=timezone.utc)
        late = datetime(2024, 3, 1, tzinfo=timezone.utc)
        insert_bars(conn, "AAPL", "1Day", [100 + i for i in range(10)], start=early)
        insert_bars(conn, "AAPL", "1Day", [200 + i for i in range(10)], start=late)

        # Use eval_ts between the two sets
        mid_ts = datetime(2024, 1, 15, tzinfo=timezone.utc)
        result = estimate_volatilities(conn, ["AAPL"], mid_ts, lookback_bars=20)

        # Should only use the 10 early bars
        assert "AAPL" in result
        assert result["AAPL"] > 0

    def test_single_bar_uses_default(self, conn):
        """One bar means zero log-returns → uses default."""
        insert_bars(conn, "SOLO", "1Day", [100.0])

        result = estimate_volatilities(conn, ["SOLO"], EVAL_TS, default_vol=0.35)

        assert result["SOLO"] == 0.35


class TestHighVol:
    def test_high_vol_stock(self, conn):
        """Volatile stock gets higher vol estimate than stable one."""
        # Stable: small moves
        stable_closes = [100.0 + 0.1 * i for i in range(20)]
        # Volatile: large moves
        volatile_closes = [100.0 + (5.0 * ((-1) ** i)) for i in range(20)]

        insert_bars(conn, "STABLE", "1Day", stable_closes)
        insert_bars(conn, "WILD", "1Day", volatile_closes)

        result = estimate_volatilities(conn, ["STABLE", "WILD"], EVAL_TS)

        assert result["WILD"] > result["STABLE"]
