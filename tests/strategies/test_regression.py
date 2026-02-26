"""Regression tests — golden output comparison.

Captures a full week of TQQQWeekly signals against known bar data.
If a code change alters the output, this test fails.  The fix is
either to revert the change or bump the strategy version and
regenerate the golden file.

Golden file: tests/golden/tqqq_weekly_v1.0.0.json
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from src.strategies.context import DataAccess, StrategyContext
from src.strategies.tqqq_weekly import TQQQWeekly
from tests.conftest import insert_bars

# ── Paths ────────────────────────────────────────────────────────────

_GOLDEN_DIR = Path(__file__).resolve().parent.parent / "golden"
_GOLDEN_FILE = _GOLDEN_DIR / "tqqq_weekly_v1.0.0.json"

# ── Helpers ──────────────────────────────────────────────────────────

MON = datetime(2024, 1, 8, tzinfo=timezone.utc)
SYMBOL = "TQQQ"
TF = "1Day"

DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday"]


def _load_golden() -> dict[str, Any]:
    return json.loads(_GOLDEN_FILE.read_text())


def _insert_golden_bars(conn: sqlite3.Connection, golden: dict[str, Any]) -> None:
    """Reproduce the exact bar data from the golden file."""
    conn.execute("INSERT OR IGNORE INTO symbols (symbol) VALUES (?)", (SYMBOL,))

    # Prior-week lookback bars.
    prior_mon = MON - timedelta(weeks=1)
    for i, c in enumerate([48.0, 49.0, 50.0, 51.0, 52.0]):
        ts = prior_mon + timedelta(days=i)
        conn.execute(
            "INSERT INTO bars "
            "(symbol, timeframe, ts, open, high, low, close, volume, trade_count, vwap) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (SYMBOL, TF, ts.isoformat(), c * 0.999, c * 1.005, c * 0.995, c, 1000, 100, c),
        )

    # The test week from golden data.
    for i, (o, h, l, c) in enumerate(golden["week_data"]):
        ts = MON + timedelta(days=i)
        conn.execute(
            "INSERT INTO bars "
            "(symbol, timeframe, ts, open, high, low, close, volume, trade_count, vwap) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (SYMBOL, TF, ts.isoformat(), o, h, l, c, 50000, 5000, c),
        )
    conn.commit()


def _signal_to_comparable(sig_dump: dict[str, Any]) -> dict[str, Any]:
    """Normalise a signal dict for comparison (convert lists → tuples, etc.)."""
    d = dict(sig_dump)
    # Convert list fields to tuples for stable comparison.
    if "invalidate" in d:
        d["invalidate"] = tuple(tuple(ic.items()) if isinstance(ic, dict) else ic for ic in d["invalidate"])
    if "tags" in d:
        d["tags"] = tuple(d["tags"])
    return d


# ── Tests ────────────────────────────────────────────────────────────


class TestGoldenOutput:
    """Compare live strategy output against the stored golden file."""

    @pytest.fixture
    def golden(self) -> dict[str, Any]:
        return _load_golden()

    def test_golden_file_exists(self) -> None:
        assert _GOLDEN_FILE.exists(), f"Golden file missing: {_GOLDEN_FILE}"

    def test_version_matches(self, golden: dict[str, Any]) -> None:
        """Strategy version in golden file matches the code."""
        strat = TQQQWeekly()
        assert strat.version == golden["version"], (
            f"Strategy version changed from {golden['version']} to {strat.version}. "
            f"Regenerate the golden file if this is intentional."
        )

    def test_params_hash_matches(self, golden: dict[str, Any]) -> None:
        """params_hash in golden file matches the code."""
        strat = TQQQWeekly()
        assert strat.params_hash == golden["params_hash"], (
            f"params_hash changed from {golden['params_hash']} to {strat.params_hash}. "
            f"Default params may have changed — regenerate golden file."
        )

    def test_params_match(self, golden: dict[str, Any]) -> None:
        """Default params match the golden file."""
        strat = TQQQWeekly()
        assert strat.params() == golden["params"]

    @pytest.mark.parametrize("day_index,day_name", list(enumerate(DAY_NAMES)))
    def test_day_signals_match_golden(
        self,
        conn: sqlite3.Connection,
        golden: dict[str, Any],
        day_index: int,
        day_name: str,
    ) -> None:
        """Signals for each day of the week match the golden output exactly."""
        _insert_golden_bars(conn, golden)

        eval_ts = MON + timedelta(days=day_index)
        dao = DataAccess(conn, eval_ts)
        dao.prefetch([SYMBOL], TF, 10)
        ctx = StrategyContext(
            now_ts=eval_ts,
            universe=(SYMBOL,),
            timeframe=TF,
            data=dao,
        )

        strat = TQQQWeekly()
        signals = strat.run(ctx)

        expected = golden["days"][day_name]
        assert len(signals) == len(expected), (
            f"{day_name}: expected {len(expected)} signals, got {len(signals)}"
        )

        for i, (actual_sig, expected_dict) in enumerate(zip(signals, expected)):
            actual_dict = actual_sig.model_dump(mode="python")
            # Normalise enums to string values for comparison.
            for k, v in actual_dict.items():
                if hasattr(v, "value"):
                    actual_dict[k] = v.value
                if isinstance(v, tuple):
                    actual_dict[k] = list(v)

            for field, expected_val in expected_dict.items():
                actual_val = actual_dict[field]
                assert actual_val == expected_val, (
                    f"{day_name} signal[{i}].{field}: "
                    f"expected {expected_val!r}, got {actual_val!r}"
                )

    def test_full_week_signal_count(
        self, conn: sqlite3.Connection, golden: dict[str, Any],
    ) -> None:
        """Total signal count across the week matches golden."""
        _insert_golden_bars(conn, golden)

        total_expected = sum(len(sigs) for sigs in golden["days"].values())
        total_actual = 0

        for day_index in range(5):
            eval_ts = MON + timedelta(days=day_index)
            dao = DataAccess(conn, eval_ts)
            dao.prefetch([SYMBOL], TF, 10)
            ctx = StrategyContext(
                now_ts=eval_ts,
                universe=(SYMBOL,),
                timeframe=TF,
                data=dao,
            )
            strat = TQQQWeekly()
            total_actual += len(strat.run(ctx))

        assert total_actual == total_expected
