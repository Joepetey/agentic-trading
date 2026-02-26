"""Tests for orchestrator data models."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.orchestrator.models import (
    DroppedSignal,
    DropReason,
    ExclusionReason,
    MergedSignal,
    OpenOrderSnapshot,
    PortfolioIntent,
    PortfolioState,
    PositionSnapshot,
    SignalContribution,
    SizingMethod,
    SymbolExclusion,
    TargetPosition,
    UniverseResult,
)


NOW = datetime(2024, 1, 10, 16, 0, tzinfo=timezone.utc)


# ── PositionSnapshot ─────────────────────────────────────────────────


class TestPositionSnapshot:
    def test_construction(self):
        pos = PositionSnapshot(
            symbol="AAPL", qty=100.0, market_value=15000.0,
            avg_entry_price=145.0, unrealized_pnl=500.0,
        )
        assert pos.symbol == "AAPL"
        assert pos.qty == 100.0
        assert pos.market_value == 15000.0

    def test_frozen(self):
        pos = PositionSnapshot(
            symbol="AAPL", qty=100.0, market_value=15000.0,
            avg_entry_price=145.0, unrealized_pnl=500.0,
        )
        with pytest.raises(ValidationError):
            pos.qty = 200.0  # type: ignore[misc]

    def test_short_position_negative_qty(self):
        pos = PositionSnapshot(
            symbol="SPY", qty=-50.0, market_value=-22500.0,
            avg_entry_price=450.0, unrealized_pnl=-100.0,
        )
        assert pos.qty == -50.0
        assert pos.market_value == -22500.0


# ── OpenOrderSnapshot ────────────────────────────────────────────────


class TestOpenOrderSnapshot:
    def test_construction(self):
        order = OpenOrderSnapshot(
            order_id="abc123", symbol="MSFT", side="buy",
            qty=10.0, order_type="limit", limit_price=350.0,
        )
        assert order.order_id == "abc123"
        assert order.limit_price == 350.0
        assert order.stop_price is None

    def test_frozen(self):
        order = OpenOrderSnapshot(
            order_id="abc123", symbol="MSFT", side="buy",
            qty=10.0, order_type="market",
        )
        with pytest.raises(ValidationError):
            order.side = "sell"  # type: ignore[misc]


# ── PortfolioState ───────────────────────────────────────────────────


class TestPortfolioState:
    def _make_state(self, **kwargs):
        defaults = dict(
            as_of_ts=NOW, equity=100_000.0, cash=50_000.0, buying_power=50_000.0,
        )
        defaults.update(kwargs)
        return PortfolioState(**defaults)

    def test_empty_positions(self):
        state = self._make_state()
        assert state.positions == ()
        assert state.open_orders == ()
        assert state.position_map == {}
        assert state.total_exposure == 0.0
        assert state.exposure_pct == 0.0

    def test_position_map(self):
        p1 = PositionSnapshot(
            symbol="AAPL", qty=100.0, market_value=15000.0,
            avg_entry_price=145.0, unrealized_pnl=500.0,
        )
        p2 = PositionSnapshot(
            symbol="MSFT", qty=50.0, market_value=17500.0,
            avg_entry_price=340.0, unrealized_pnl=250.0,
        )
        state = self._make_state(positions=(p1, p2))
        assert set(state.position_map.keys()) == {"AAPL", "MSFT"}
        assert state.position_map["AAPL"] is p1

    def test_total_exposure(self):
        p1 = PositionSnapshot(
            symbol="AAPL", qty=100.0, market_value=15000.0,
            avg_entry_price=145.0, unrealized_pnl=500.0,
        )
        p2 = PositionSnapshot(
            symbol="SPY", qty=-50.0, market_value=-22500.0,
            avg_entry_price=450.0, unrealized_pnl=-100.0,
        )
        state = self._make_state(positions=(p1, p2))
        assert state.total_exposure == 15000.0 + 22500.0

    def test_exposure_pct(self):
        p1 = PositionSnapshot(
            symbol="AAPL", qty=100.0, market_value=45000.0,
            avg_entry_price=400.0, unrealized_pnl=5000.0,
        )
        state = self._make_state(equity=100_000.0, positions=(p1,))
        assert state.exposure_pct == pytest.approx(0.45)

    def test_zero_equity(self):
        state = self._make_state(equity=0.0)
        assert state.exposure_pct == 0.0

    def test_frozen(self):
        state = self._make_state()
        with pytest.raises(ValidationError):
            state.equity = 200_000.0  # type: ignore[misc]


# ── UniverseResult ───────────────────────────────────────────────────


class TestUniverseResult:
    def test_construction(self):
        excl = SymbolExclusion(
            symbol="PENNY", reason=ExclusionReason.BELOW_MIN_PRICE,
            detail="close=0.50 < min=5.00",
        )
        result = UniverseResult(included=("AAPL", "MSFT"), excluded=(excl,))
        assert len(result.included) == 2
        assert len(result.excluded) == 1
        assert result.excluded[0].reason == ExclusionReason.BELOW_MIN_PRICE

    def test_empty(self):
        result = UniverseResult(included=())
        assert result.included == ()
        assert result.excluded == ()


# ── MergedSignal ─────────────────────────────────────────────────────


class TestMergedSignal:
    def test_construction(self):
        contrib = SignalContribution(
            strategy_id="strat_a", side="long", strength=0.8,
            confidence=0.9, weight=0.72, horizon_bars=5,
        )
        merged = MergedSignal(
            symbol="AAPL", side="long", agg_strength=0.8,
            agg_confidence=0.9, horizon_bars=5,
            contributions=(contrib,),
        )
        assert merged.symbol == "AAPL"
        assert merged.agg_strength == pytest.approx(0.8)
        assert len(merged.contributions) == 1

    def test_strength_clamping(self):
        merged = MergedSignal(
            symbol="AAPL", side="long", agg_strength=1.5,
            agg_confidence=0.9, horizon_bars=5,
        )
        assert merged.agg_strength == 1.0

        merged2 = MergedSignal(
            symbol="AAPL", side="short", agg_strength=-1.5,
            agg_confidence=0.9, horizon_bars=5,
        )
        assert merged2.agg_strength == -1.0

    def test_confidence_clamping(self):
        merged = MergedSignal(
            symbol="AAPL", side="long", agg_strength=0.5,
            agg_confidence=1.5, horizon_bars=5,
        )
        assert merged.agg_confidence == 1.0

        merged2 = MergedSignal(
            symbol="AAPL", side="long", agg_strength=0.5,
            agg_confidence=-0.3, horizon_bars=5,
        )
        assert merged2.agg_confidence == 0.0

    def test_frozen(self):
        merged = MergedSignal(
            symbol="AAPL", side="long", agg_strength=0.5,
            agg_confidence=0.9, horizon_bars=5,
        )
        with pytest.raises(ValidationError):
            merged.agg_strength = 0.1  # type: ignore[misc]


# ── TargetPosition ───────────────────────────────────────────────────


class TestTargetPosition:
    def test_long_target(self):
        target = TargetPosition(
            symbol="AAPL", target_notional=5000.0, target_pct=0.05,
            confidence=0.9, horizon_bars=5,
            explain="side=long, notional=$5,000",
        )
        assert target.target_notional == 5000.0
        assert target.target_pct == 0.05

    def test_short_target_negative(self):
        target = TargetPosition(
            symbol="SPY", target_notional=-3000.0, target_pct=-0.03,
            confidence=0.7, horizon_bars=3,
        )
        assert target.target_notional == -3000.0

    def test_exit_target_zero(self):
        target = TargetPosition(
            symbol="MSFT", target_notional=0.0, target_pct=0.0,
            confidence=0.8, horizon_bars=1,
        )
        assert target.target_notional == 0.0

    def test_provenance(self):
        contrib = SignalContribution(
            strategy_id="strat_a", side="long", strength=0.8,
            confidence=0.9, weight=0.72, horizon_bars=5,
        )
        target = TargetPosition(
            symbol="AAPL", target_notional=5000.0, target_pct=0.05,
            confidence=0.9, horizon_bars=5, provenance=(contrib,),
        )
        assert len(target.provenance) == 1
        assert target.provenance[0].strategy_id == "strat_a"


# ── DroppedSignal ────────────────────────────────────────────────────


class TestDroppedSignal:
    def test_construction(self):
        dropped = DroppedSignal(
            strategy_id="strat_b", symbol="TSLA", side="short",
            strength=-0.5, confidence=0.6,
            reason=DropReason.CONFLICTING_SIDES,
            detail="Opposite to consensus side=long",
        )
        assert dropped.reason == DropReason.CONFLICTING_SIDES
        assert "Opposite" in dropped.detail


# ── PortfolioIntent ──────────────────────────────────────────────────


class TestPortfolioIntent:
    def test_full_construction(self):
        state = PortfolioState(
            as_of_ts=NOW, equity=100_000.0, cash=50_000.0, buying_power=50_000.0,
        )
        universe = UniverseResult(included=("AAPL", "MSFT"))
        target = TargetPosition(
            symbol="AAPL", target_notional=5000.0, target_pct=0.05,
            confidence=0.9, horizon_bars=5,
        )
        intent = PortfolioIntent(
            intent_id="abc123",
            as_of_ts=NOW,
            portfolio_state=state,
            universe=universe,
            targets=(target,),
            sizing_method=SizingMethod.SIGNAL_WEIGHTED,
            elapsed_ms=42.5,
            explain="Test cycle.",
        )
        assert intent.intent_id == "abc123"
        assert len(intent.targets) == 1
        assert intent.sizing_method == SizingMethod.SIGNAL_WEIGHTED
        assert intent.elapsed_ms == 42.5

    def test_defaults(self):
        state = PortfolioState(
            as_of_ts=NOW, equity=100_000.0, cash=50_000.0, buying_power=50_000.0,
        )
        universe = UniverseResult(included=())
        intent = PortfolioIntent(
            intent_id="def456",
            as_of_ts=NOW,
            portfolio_state=state,
            universe=universe,
        )
        assert intent.signals_used == ()
        assert intent.signals_dropped == ()
        assert intent.targets == ()
        assert intent.strategy_run_id is None
        assert intent.elapsed_ms == 0.0
        assert intent.explain == ""

    def test_frozen(self):
        state = PortfolioState(
            as_of_ts=NOW, equity=100_000.0, cash=50_000.0, buying_power=50_000.0,
        )
        universe = UniverseResult(included=())
        intent = PortfolioIntent(
            intent_id="abc", as_of_ts=NOW,
            portfolio_state=state, universe=universe,
        )
        with pytest.raises(ValidationError):
            intent.explain = "modified"  # type: ignore[misc]


# ── Enum values ──────────────────────────────────────────────────────


class TestEnums:
    def test_exclusion_reason_values(self):
        assert ExclusionReason.BELOW_MIN_PRICE.value == "below_min_price"
        assert ExclusionReason.INSUFFICIENT_DATA.value == "insufficient_data"

    def test_drop_reason_values(self):
        assert DropReason.CONFLICTING_SIDES.value == "conflicting_sides"
        assert DropReason.ZERO_STRENGTH.value == "zero_strength"

    def test_sizing_method_values(self):
        assert SizingMethod.EQUAL_WEIGHT.value == "equal_weight"
        assert SizingMethod.SIGNAL_WEIGHTED.value == "signal_weighted"
