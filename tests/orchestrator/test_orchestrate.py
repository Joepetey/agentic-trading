"""Integration tests for the orchestrate() pipeline."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from src.core.config import OrchestratorConfig, RiskLimits
from src.orchestrator.intent_persist import ensure_intent_schema, get_intent
from src.orchestrator.models import (
    ExclusionReason,
    PortfolioState,
    SizingMethod,
)
from src.orchestrator.orchestrate import orchestrate
from src.strategies.base import Strategy
from src.strategies.context import Constraints, StrategyContext
from src.strategies.signal import Side, Signal
from tests.conftest import insert_bars

NOW = datetime(2024, 1, 10, 16, 0, tzinfo=timezone.utc)


# ── Stub strategies ──────────────────────────────────────────────────


class AlwaysLong(Strategy):
    """Emits LONG for every symbol in the universe."""

    @property
    def strategy_id(self) -> str:
        return "always_long"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def params(self) -> dict[str, Any]:
        return {}

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=sym,
                side=Side.LONG,
                strength=0.7,
                confidence=0.8,
                horizon_bars=5,
                explain="always long",
            )
            for sym in ctx.universe
        ]


class AlwaysFlat(Strategy):
    """Emits FLAT exit signals for every symbol."""

    @property
    def strategy_id(self) -> str:
        return "always_flat"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def params(self) -> dict[str, Any]:
        return {}

    def run(self, ctx: StrategyContext) -> list[Signal]:
        return [
            Signal(
                strategy_id=self.strategy_id,
                symbol=sym,
                side=Side.FLAT,
                strength=-0.5,
                confidence=0.8,
                horizon_bars=1,
                explain="always flat",
            )
            for sym in ctx.universe
        ]


class Exploding(Strategy):
    """Always raises an exception."""

    @property
    def strategy_id(self) -> str:
        return "exploding"

    @property
    def version(self) -> str:
        return "1.0.0"

    def required_timeframes(self) -> list[str]:
        return ["1Day"]

    def required_lookback_bars(self) -> int:
        return 1

    def params(self) -> dict[str, Any]:
        return {}

    def run(self, ctx: StrategyContext) -> list[Signal]:
        raise RuntimeError("boom")


def _portfolio(equity: float = 100_000.0) -> PortfolioState:
    return PortfolioState(
        as_of_ts=NOW, equity=equity, cash=50_000.0, buying_power=50_000.0,
    )


def _risk() -> RiskLimits:
    return RiskLimits(
        max_position_pct=0.05,
        max_portfolio_exposure_pct=0.90,
    )


def _setup_bars(conn, symbols: list[str]):
    """Insert bars for symbols so they pass universe filtering."""
    for sym in symbols:
        insert_bars(conn, sym, "1Day", [150.0] * 5,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))


# ── Tests ────────────────────────────────────────────────────────────


class TestFullCycle:
    def test_happy_path(self, conn):
        """Strategies produce signals, orchestrator produces intent with targets."""
        _setup_bars(conn, ["AAPL", "MSFT"])

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "MSFT"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
        )

        assert intent.intent_id  # non-empty
        assert intent.as_of_ts == NOW
        assert len(intent.universe.included) == 2
        assert len(intent.targets) == 2
        assert all(t.target_notional > 0 for t in intent.targets)
        assert intent.elapsed_ms >= 0

    def test_explain_populated(self, conn):
        """Explain field contains cycle summary."""
        _setup_bars(conn, ["AAPL"])

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
        )

        assert "Cycle at" in intent.explain
        assert "Universe:" in intent.explain
        assert "Targets:" in intent.explain


class TestEmptyInputs:
    def test_no_strategies(self, conn):
        """Empty strategy list produces intent with empty targets."""
        _setup_bars(conn, ["AAPL"])

        intent = orchestrate(
            strategies=[],
            universe=["AAPL"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
        )

        assert len(intent.targets) == 0
        assert len(intent.signals_used) == 0

    def test_empty_universe(self, conn):
        """No symbols produces empty targets."""
        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=[],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
        )

        assert len(intent.targets) == 0


class TestErrorIsolation:
    def test_strategy_errors_isolated(self, conn):
        """One failing strategy does not prevent intent generation."""
        _setup_bars(conn, ["AAPL"])

        intent = orchestrate(
            strategies=[AlwaysLong(), Exploding()],
            universe=["AAPL"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
        )

        # AlwaysLong should still produce a target despite Exploding failing
        assert len(intent.targets) == 1
        assert intent.targets[0].symbol == "AAPL"


class TestUniverseFiltering:
    def test_constraints_filter_before_execution(self, conn):
        """Constraints filter symbols before strategy execution."""
        _setup_bars(conn, ["AAPL", "MSFT"])
        # Set MSFT to a low price
        conn.execute("UPDATE bars SET close = 2.0 WHERE symbol = 'MSFT'")
        conn.commit()

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "MSFT"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            constraints=Constraints(min_price=5.0),
        )

        assert "AAPL" in intent.universe.included
        assert "MSFT" not in intent.universe.included
        # Only AAPL should have a target
        target_symbols = {t.symbol for t in intent.targets}
        assert target_symbols == {"AAPL"}


class TestPortfolioStateFlowThrough:
    def test_portfolio_state_in_intent(self, conn):
        """Portfolio state is accessible in intent output."""
        _setup_bars(conn, ["AAPL"])

        port = _portfolio(equity=200_000.0)
        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL"],
            conn=conn,
            now_ts=NOW,
            portfolio=port,
            risk_limits=_risk(),
        )

        assert intent.portfolio_state.equity == 200_000.0


class TestExitSignals:
    def test_all_flat_produces_exit_targets(self, conn):
        """When all strategies say FLAT, targets have notional=0."""
        _setup_bars(conn, ["AAPL"])

        intent = orchestrate(
            strategies=[AlwaysFlat()],
            universe=["AAPL"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
        )

        assert len(intent.targets) == 1
        assert intent.targets[0].target_notional == 0.0


class TestSizingMethod:
    def test_configurable_sizing(self, conn):
        """Different SizingMethod produces different allocations."""
        _setup_bars(conn, ["AAPL", "MSFT"])

        intent_eq = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "MSFT"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            sizing_method=SizingMethod.EQUAL_WEIGHT,
        )

        intent_sw = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "MSFT"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            sizing_method=SizingMethod.SIGNAL_WEIGHTED,
        )

        # Both should produce 2 targets
        assert len(intent_eq.targets) == 2
        assert len(intent_sw.targets) == 2
        assert intent_eq.sizing_method == SizingMethod.EQUAL_WEIGHT
        assert intent_sw.sizing_method == SizingMethod.SIGNAL_WEIGHTED


class TestPersistence:
    def test_persist_writes_intent(self, conn):
        """persist=True writes to portfolio_intents table."""
        _setup_bars(conn, ["AAPL"])

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL"],
            conn=conn,
            now_ts=NOW,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            persist=True,
        )

        row = get_intent(conn, intent.intent_id)
        assert row is not None
        assert row["intent_id"] == intent.intent_id


# ── Step 0: Evaluation timestamp resolution tests ────────────────────


class TestAutoEvalTs:
    def test_auto_eval_ts_resolved_from_bars(self, conn):
        """now_ts=None resolves eval_ts from bar timestamps."""
        # AAPL bars up to Jan 7, MSFT bars up to Jan 8
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))
        insert_bars(conn, "MSFT", "1Day", [200.0] * 6,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "MSFT"],
            conn=conn,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            now_ts=None,  # auto-resolve
        )

        # eval_ts = min(latest) across fresh symbols
        # AAPL latest = Jan 3 + 4d = Jan 7, MSFT latest = Jan 3 + 5d = Jan 8
        # eval_ts = min(Jan 7, Jan 8) = Jan 7
        expected = datetime(2024, 1, 7, tzinfo=timezone.utc)
        assert intent.as_of_ts == expected
        assert intent.trade_allowed is True

    def test_override_eval_ts(self, conn):
        """Passing now_ts skips resolution, uses the provided value."""
        _setup_bars(conn, ["AAPL"])

        override = datetime(2099, 6, 15, 12, 0, tzinfo=timezone.utc)
        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL"],
            conn=conn,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            now_ts=override,
        )

        assert intent.as_of_ts == override
        assert intent.trade_allowed is True


class TestNoTrade:
    def test_no_trade_when_too_many_stale(self, conn):
        """Stale fraction exceeds threshold → trade_allowed=False, empty targets."""
        # Only AAPL has data; MSFT, GOOGL have none → 2/3 missing > 50%
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))

        orch_cfg = OrchestratorConfig(
            max_stale_pct=0.50,
            primary_timeframe="1Day",
        )

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "MSFT", "GOOGL"],
            conn=conn,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            orchestrator_config=orch_cfg,
        )

        assert intent.trade_allowed is False
        assert len(intent.targets) == 0
        assert "NO_TRADE" in intent.explain

    def test_no_trade_persist(self, conn):
        """NO_TRADE intent can be persisted."""
        insert_bars(conn, "AAPL", "1Day", [100.0] * 3,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))

        orch_cfg = OrchestratorConfig(max_stale_pct=0.30)

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "GHOST1", "GHOST2"],
            conn=conn,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            orchestrator_config=orch_cfg,
            persist=True,
        )

        assert intent.trade_allowed is False
        row = get_intent(conn, intent.intent_id)
        assert row is not None
        assert row["trade_allowed"] == 0


class TestStaleExclusions:
    def test_stale_symbols_excluded_from_universe(self, conn):
        """Stale symbols get DATA_TOO_STALE exclusion, don't reach strategies."""
        # AAPL: fresh bars up to Jan 10
        insert_bars(conn, "AAPL", "1Day", [100.0] * 8,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))
        # MSFT: only 1 bar at Jan 3 — 7 days behind AAPL's latest (Jan 10)
        insert_bars(conn, "MSFT", "1Day", [200.0] * 1,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))

        orch_cfg = OrchestratorConfig(
            max_stale_pct=0.90,  # high threshold so we don't trigger NO_TRADE
            primary_timeframe="1Day",
            max_staleness={"1Day": 2880},  # 2 days in minutes
        )

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "MSFT"],
            conn=conn,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            orchestrator_config=orch_cfg,
        )

        assert intent.trade_allowed is True
        assert "AAPL" in intent.universe.included
        assert "MSFT" not in intent.universe.included
        # MSFT should be in exclusions with DATA_TOO_STALE
        stale_exclusions = [
            e for e in intent.universe.excluded
            if e.reason == ExclusionReason.DATA_TOO_STALE
        ]
        assert any(e.symbol == "MSFT" for e in stale_exclusions)

    def test_missing_symbols_excluded(self, conn):
        """Symbols with no data get DATA_TOO_STALE exclusion (below threshold)."""
        insert_bars(conn, "AAPL", "1Day", [100.0] * 5,
                    start=datetime(2024, 1, 3, tzinfo=timezone.utc))

        orch_cfg = OrchestratorConfig(
            max_stale_pct=0.90,  # high, don't trigger NO_TRADE
            primary_timeframe="1Day",
        )

        intent = orchestrate(
            strategies=[AlwaysLong()],
            universe=["AAPL", "GHOST"],
            conn=conn,
            portfolio=_portfolio(),
            risk_limits=_risk(),
            orchestrator_config=orch_cfg,
        )

        assert intent.trade_allowed is True
        assert "AAPL" in intent.universe.included
        stale_exclusions = [
            e for e in intent.universe.excluded
            if e.reason == ExclusionReason.DATA_TOO_STALE
        ]
        assert any(e.symbol == "GHOST" for e in stale_exclusions)
