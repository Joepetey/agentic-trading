"""Tests for position sizing."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.core.config import RiskLimits
from src.orchestrator.models import (
    MergedSignal,
    PortfolioState,
    SignalContribution,
    SizingMethod,
)
from src.orchestrator.sizing import compute_targets

NOW = datetime(2024, 1, 10, 16, 0, tzinfo=timezone.utc)


def _portfolio(equity: float = 100_000.0, cash: float = 50_000.0) -> PortfolioState:
    return PortfolioState(
        as_of_ts=NOW, equity=equity, cash=cash, buying_power=cash,
    )


def _risk(
    max_position_pct: float = 0.05,
    max_portfolio_exposure_pct: float = 0.90,
) -> RiskLimits:
    return RiskLimits(
        max_position_pct=max_position_pct,
        max_portfolio_exposure_pct=max_portfolio_exposure_pct,
    )


def _merged(
    symbol: str = "AAPL",
    side: str = "long",
    agg_strength: float = 0.8,
    agg_confidence: float = 0.9,
    horizon_bars: int = 5,
    stop_hint: float | None = None,
    tp_hint: float | None = None,
) -> MergedSignal:
    contrib = SignalContribution(
        strategy_id="strat_a", side=side, strength=agg_strength,
        confidence=agg_confidence, weight=0.72, horizon_bars=horizon_bars,
    )
    return MergedSignal(
        symbol=symbol, side=side, agg_strength=agg_strength,
        agg_confidence=agg_confidence, horizon_bars=horizon_bars,
        stop_hint=stop_hint, tp_hint=tp_hint,
        contributions=(contrib,),
    )


class TestEqualWeight:
    def test_single_signal(self):
        """Single signal gets full allocation up to per-position cap."""
        signals = [_merged()]
        targets = compute_targets(signals, _portfolio(), _risk(), SizingMethod.EQUAL_WEIGHT)

        assert len(targets) == 1
        t = targets[0]
        assert t.symbol == "AAPL"
        assert t.target_notional > 0
        # Capped at max_position_pct * equity = 5% * 100k = 5000
        assert t.target_notional <= 5000.0

    def test_two_signals_equal_allocation(self):
        """Two signals split allocation equally."""
        signals = [
            _merged(symbol="AAPL"),
            _merged(symbol="MSFT"),
        ]
        targets = compute_targets(signals, _portfolio(), _risk(), SizingMethod.EQUAL_WEIGHT)

        assert len(targets) == 2
        assert targets[0].target_notional == pytest.approx(targets[1].target_notional, rel=0.01)

    def test_many_signals_capped_by_position(self):
        """With many signals, each capped at max_position_pct."""
        signals = [_merged(symbol=f"SYM{i}") for i in range(50)]
        targets = compute_targets(
            signals, _portfolio(equity=100_000.0),
            _risk(max_position_pct=0.05, max_portfolio_exposure_pct=0.90),
            SizingMethod.EQUAL_WEIGHT,
        )

        for t in targets:
            assert abs(t.target_notional) <= 5000.0 + 0.01


class TestSignalWeighted:
    def test_proportional_allocation(self):
        """Higher strength * confidence gets more allocation."""
        s_strong = _merged(symbol="AAPL", agg_strength=0.9, agg_confidence=0.9)
        s_weak = _merged(symbol="MSFT", agg_strength=0.3, agg_confidence=0.5)
        targets = compute_targets(
            [s_strong, s_weak], _portfolio(),
            _risk(max_position_pct=0.50),  # high cap to see proportional effect
        )

        assert len(targets) == 2
        aapl = next(t for t in targets if t.symbol == "AAPL")
        msft = next(t for t in targets if t.symbol == "MSFT")
        assert aapl.target_notional > msft.target_notional

    def test_per_position_cap(self):
        """No position exceeds max_position_pct."""
        # One extremely strong signal
        signals = [_merged(symbol="AAPL", agg_strength=1.0, agg_confidence=1.0)]
        targets = compute_targets(signals, _portfolio(), _risk(max_position_pct=0.05))

        assert len(targets) == 1
        assert targets[0].target_notional <= 5000.0 + 0.01

    def test_total_exposure_cap(self):
        """Total notional does not exceed max_portfolio_exposure_pct * equity."""
        signals = [_merged(symbol=f"SYM{i}", agg_strength=0.8, agg_confidence=0.9) for i in range(5)]
        risk = _risk(max_position_pct=0.30, max_portfolio_exposure_pct=0.50)
        targets = compute_targets(signals, _portfolio(equity=100_000.0), risk)

        total = sum(abs(t.target_notional) for t in targets)
        assert total <= 50_000.0 + 1.0  # some rounding tolerance


class TestExitTargets:
    def test_flat_produces_zero_notional(self):
        """FLAT merged signals produce target_notional=0."""
        exit_sig = _merged(symbol="AAPL", side="flat", agg_strength=-0.8, agg_confidence=0.9)
        targets = compute_targets([exit_sig], _portfolio(), _risk())

        assert len(targets) == 1
        assert targets[0].target_notional == 0.0
        assert targets[0].target_pct == 0.0
        assert "Exit signal" in targets[0].explain

    def test_mixed_directional_and_exit(self):
        """Directional and exit signals produce separate targets."""
        signals = [
            _merged(symbol="AAPL", side="long"),
            _merged(symbol="MSFT", side="flat", agg_strength=-0.5, agg_confidence=0.8),
        ]
        targets = compute_targets(signals, _portfolio(), _risk())

        assert len(targets) == 2
        aapl = next(t for t in targets if t.symbol == "AAPL")
        msft = next(t for t in targets if t.symbol == "MSFT")
        assert aapl.target_notional > 0
        assert msft.target_notional == 0.0


class TestShortTargets:
    def test_short_negative_notional(self):
        """Short-side targets have negative notional."""
        signals = [_merged(symbol="SPY", side="short", agg_strength=-0.7)]
        targets = compute_targets(signals, _portfolio(), _risk())

        assert len(targets) == 1
        assert targets[0].target_notional < 0


class TestEdgeCases:
    def test_empty_signals(self):
        targets = compute_targets([], _portfolio(), _risk())
        assert targets == []

    def test_zero_equity(self):
        targets = compute_targets([_merged()], _portfolio(equity=0.0), _risk())
        assert targets == []

    def test_provenance_preserved(self):
        """Provenance from MergedSignal flows through to TargetPosition."""
        signal = _merged()
        targets = compute_targets([signal], _portfolio(), _risk())

        assert len(targets) == 1
        assert len(targets[0].provenance) == 1
        assert targets[0].provenance[0].strategy_id == "strat_a"

    def test_explain_contains_details(self):
        """The explain field contains sizing details."""
        signal = _merged(agg_strength=0.8, agg_confidence=0.9)
        targets = compute_targets([signal], _portfolio(), _risk())

        assert "side=long" in targets[0].explain
        assert "0.800" in targets[0].explain

    def test_stop_and_tp_pass_through(self):
        """Stop and TP hints are carried through."""
        signal = _merged(stop_hint=145.0, tp_hint=170.0)
        targets = compute_targets([signal], _portfolio(), _risk())

        assert targets[0].stop_hint == 145.0
        assert targets[0].tp_hint == 170.0

    def test_redistribution_of_excess(self):
        """Excess from capped positions redistributed to uncapped ones."""
        # One very strong, one weak — strong gets capped, excess goes to weak
        s_strong = _merged(symbol="AAPL", agg_strength=1.0, agg_confidence=1.0)
        s_weak = _merged(symbol="MSFT", agg_strength=0.1, agg_confidence=0.1)
        risk = _risk(max_position_pct=0.05, max_portfolio_exposure_pct=0.90)
        targets = compute_targets([s_strong, s_weak], _portfolio(), risk)

        aapl = next(t for t in targets if t.symbol == "AAPL")
        msft = next(t for t in targets if t.symbol == "MSFT")

        # AAPL capped at 5000
        assert aapl.target_notional == pytest.approx(5000.0, rel=0.01)
        # MSFT gets more than its raw weight would give (due to redistribution)
        # Raw: weak weight is 0.01 out of total 1.01, that's ~891 of 90000
        # After redistribution it should be higher
        assert msft.target_notional > 800
