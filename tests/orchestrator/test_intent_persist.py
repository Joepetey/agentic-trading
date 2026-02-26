"""Tests for PortfolioIntent persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from src.orchestrator.intent_persist import (
    ensure_intent_schema,
    get_intent,
    get_latest_intent,
    write_intent,
)
from src.orchestrator.models import (
    PortfolioIntent,
    PortfolioState,
    SizingMethod,
    TargetPosition,
    UniverseResult,
)

NOW = datetime(2024, 1, 10, 16, 0, tzinfo=timezone.utc)
LATER = datetime(2024, 1, 11, 16, 0, tzinfo=timezone.utc)


def _portfolio() -> PortfolioState:
    return PortfolioState(
        as_of_ts=NOW, equity=100_000.0, cash=50_000.0, buying_power=50_000.0,
    )


def _intent(
    intent_id: str = "abc123",
    as_of_ts: datetime = NOW,
    **kwargs,
) -> PortfolioIntent:
    defaults = dict(
        intent_id=intent_id,
        as_of_ts=as_of_ts,
        portfolio_state=_portfolio(),
        universe=UniverseResult(included=("AAPL", "MSFT")),
        sizing_method=SizingMethod.SIGNAL_WEIGHTED,
        elapsed_ms=42.5,
        explain="Test cycle.",
    )
    defaults.update(kwargs)
    return PortfolioIntent(**defaults)


class TestEnsureSchema:
    def test_idempotent(self, conn):
        """Calling ensure_intent_schema twice is safe."""
        ensure_intent_schema(conn)
        ensure_intent_schema(conn)
        # Should not raise
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='portfolio_intents'"
        ).fetchone()
        assert row is not None


class TestWriteAndRead:
    def test_round_trip(self, conn):
        """Write intent, read back via get_intent."""
        ensure_intent_schema(conn)
        intent = _intent()
        write_intent(conn, intent)

        row = get_intent(conn, "abc123")
        assert row is not None
        assert row["intent_id"] == "abc123"
        assert row["portfolio_equity"] == 100_000.0
        assert row["portfolio_cash"] == 50_000.0
        assert row["sizing_method"] == "signal_weighted"
        assert row["elapsed_ms"] == 42.5
        assert row["explain"] == "Test cycle."

    def test_universe_included_stored_as_json(self, conn):
        ensure_intent_schema(conn)
        intent = _intent()
        write_intent(conn, intent)

        row = get_intent(conn, "abc123")
        included = json.loads(row["universe_included"])
        assert included == ["AAPL", "MSFT"]

    def test_targets_stored_as_json(self, conn):
        ensure_intent_schema(conn)
        target = TargetPosition(
            symbol="AAPL", target_notional=5000.0, target_pct=0.05,
            confidence=0.9, horizon_bars=5,
        )
        intent = _intent(targets=(target,))
        write_intent(conn, intent)

        row = get_intent(conn, "abc123")
        targets = json.loads(row["targets"])
        assert len(targets) == 1
        assert targets[0]["symbol"] == "AAPL"
        assert targets[0]["target_notional"] == 5000.0

    def test_intent_not_found(self, conn):
        ensure_intent_schema(conn)
        assert get_intent(conn, "nonexistent") is None


class TestGetLatest:
    def test_returns_most_recent(self, conn):
        ensure_intent_schema(conn)
        write_intent(conn, _intent(intent_id="earlier", as_of_ts=NOW))
        write_intent(conn, _intent(intent_id="later", as_of_ts=LATER))

        row = get_latest_intent(conn)
        assert row is not None
        assert row["intent_id"] == "later"

    def test_empty_table(self, conn):
        ensure_intent_schema(conn)
        assert get_latest_intent(conn) is None

    def test_replace_on_conflict(self, conn):
        """INSERT OR REPLACE updates existing intent with same ID."""
        ensure_intent_schema(conn)
        write_intent(conn, _intent(intent_id="abc", explain="v1"))
        write_intent(conn, _intent(intent_id="abc", explain="v2"))

        row = get_intent(conn, "abc")
        assert row["explain"] == "v2"
