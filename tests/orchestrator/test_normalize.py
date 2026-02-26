"""Tests for signal normalization — alpha_net computation."""

from __future__ import annotations

import math

import pytest

from src.orchestrator.normalize import normalize_signals
from src.strategies.signal import Side, Signal


def _sig(
    strategy_id: str = "strat_a",
    symbol: str = "AAPL",
    side: Side = Side.LONG,
    strength: float = 0.8,
    confidence: float = 0.9,
    horizon_bars: int = 5,
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        strength=strength,
        confidence=confidence,
        horizon_bars=horizon_bars,
    )


class TestRawAlpha:
    def test_raw_alpha_computation(self):
        """alpha_net = strength * confidence when no weights or costs."""
        sig = _sig(strength=0.8, confidence=0.5)
        [result] = normalize_signals([sig], "1Day", cost_bps={})
        assert result.alpha_net == pytest.approx(0.8 * 0.5)

    def test_short_signal_negative_alpha(self):
        """Short signals have negative strength, producing negative alpha."""
        sig = _sig(side=Side.SHORT, strength=-0.6, confidence=0.8)
        [result] = normalize_signals([sig], "1Day", cost_bps={})
        assert result.alpha_net == pytest.approx(-0.6 * 0.8)
        assert result.alpha_net < 0

    def test_flat_signal_normalization(self):
        """FLAT exit signals (strength < 0) also get alpha_net."""
        sig = _sig(side=Side.FLAT, strength=-0.5, confidence=0.9)
        [result] = normalize_signals([sig], "1Day", cost_bps={})
        assert result.alpha_net == pytest.approx(-0.5 * 0.9)

    def test_zero_strength_produces_zero_alpha(self):
        """Zero strength → alpha_net = 0.0."""
        sig = _sig(strength=0.0, confidence=0.9)
        [result] = normalize_signals([sig], "1Day", cost_bps={})
        assert result.alpha_net == 0.0


class TestCalibration:
    def test_strategy_weight_multiplier(self):
        """strategy_weight scales the alpha."""
        sig = _sig(strength=0.5, confidence=1.0)
        [result] = normalize_signals(
            [sig], "1Day",
            strategy_weights={"strat_a": 2.0},
            cost_bps={},
        )
        assert result.alpha_net == pytest.approx(0.5 * 1.0 * 2.0)

    def test_edge_scale_multiplier(self):
        """edge_scale calibrates the alpha."""
        sig = _sig(strength=0.5, confidence=1.0)
        [result] = normalize_signals(
            [sig], "1Day",
            edge_scales={"strat_a": 1.5},
            cost_bps={},
        )
        assert result.alpha_net == pytest.approx(0.5 * 1.0 * 1.5)

    def test_weight_and_scale_combined(self):
        """Both weight and scale multiply together."""
        sig = _sig(strength=0.4, confidence=1.0)
        [result] = normalize_signals(
            [sig], "1Day",
            strategy_weights={"strat_a": 2.0},
            edge_scales={"strat_a": 1.5},
            cost_bps={},
        )
        assert result.alpha_net == pytest.approx(0.4 * 2.0 * 1.5)

    def test_default_weights_and_scales(self):
        """Defaults are 1.0 — no effect on alpha."""
        sig = _sig(strength=0.6, confidence=0.8)
        [result] = normalize_signals([sig], "1Day", cost_bps={})
        # No weights/scales specified → defaults to 1.0
        assert result.alpha_net == pytest.approx(0.6 * 0.8)

    def test_unknown_strategy_uses_defaults(self):
        """Strategies not in weights/scales dicts get 1.0."""
        sig = _sig(strategy_id="unknown_strat", strength=0.5, confidence=1.0)
        [result] = normalize_signals(
            [sig], "1Day",
            strategy_weights={"other": 2.0},
            edge_scales={"other": 1.5},
            cost_bps={},
        )
        # unknown_strat not in maps → weight=1.0, edge_scale=1.0
        assert result.alpha_net == pytest.approx(0.5)


class TestCostSubtraction:
    def test_cost_reduces_long_magnitude(self):
        """Cost reduces positive alpha toward zero."""
        sig = _sig(strength=0.5, confidence=1.0)
        cost = 10.0  # 10 bps
        [result] = normalize_signals(
            [sig], "1Day",
            cost_bps={"1Day": cost},
        )
        expected = 0.5 - (cost / 10_000)
        assert result.alpha_net == pytest.approx(expected)
        assert result.alpha_net < 0.5

    def test_cost_reduces_short_magnitude(self):
        """Cost reduces negative alpha magnitude (toward zero)."""
        sig = _sig(side=Side.SHORT, strength=-0.5, confidence=1.0)
        cost = 10.0  # 10 bps
        [result] = normalize_signals(
            [sig], "1Day",
            cost_bps={"1Day": cost},
        )
        # -0.5 + 0.001 = -0.499 (magnitude reduced)
        expected = -0.5 + (cost / 10_000)
        assert result.alpha_net == pytest.approx(expected)
        assert abs(result.alpha_net) < 0.5  # type: ignore[arg-type]

    def test_cost_eats_alpha(self):
        """When |calibrated| <= cost, alpha_net = 0.0."""
        sig = _sig(strength=0.001, confidence=0.001)  # tiny alpha
        [result] = normalize_signals(
            [sig], "1Day",
            cost_bps={"1Day": 100.0},  # 100 bps = 0.01
        )
        assert result.alpha_net == 0.0

    def test_cost_exactly_equals_alpha(self):
        """When |calibrated| == cost exactly, alpha_net = 0.0."""
        # calibrated = 0.5 * 1.0 = 0.5
        # cost = 5000 bps / 10000 = 0.5
        sig = _sig(strength=0.5, confidence=1.0)
        [result] = normalize_signals(
            [sig], "1Day",
            cost_bps={"1Day": 5000.0},
        )
        assert result.alpha_net == 0.0

    def test_no_cost_when_missing_timeframe(self):
        """Missing timeframe in cost_bps → 0 cost."""
        sig = _sig(strength=0.5, confidence=1.0)
        [result] = normalize_signals(
            [sig], "1Week",
            cost_bps={"1Day": 5.0},
        )
        # No cost for 1Week → alpha_net = raw_alpha
        assert result.alpha_net == pytest.approx(0.5)


class TestPureFunction:
    def test_input_signals_unchanged(self):
        """Original signal objects are not mutated."""
        sig = _sig(strength=0.5, confidence=1.0)
        original_alpha = sig.alpha_net
        normalize_signals([sig], "1Day", cost_bps={"1Day": 5.0})
        assert sig.alpha_net == original_alpha  # None

    def test_returns_new_list(self):
        """Returned list is a new object."""
        sigs = [_sig()]
        result = normalize_signals(sigs, "1Day", cost_bps={})
        assert result is not sigs

    def test_multiple_signals(self):
        """Each signal gets its own alpha_net."""
        s1 = _sig(strategy_id="strat_a", strength=0.8, confidence=0.9)
        s2 = _sig(strategy_id="strat_b", strength=0.3, confidence=0.5)
        results = normalize_signals([s1, s2], "1Day", cost_bps={})
        assert len(results) == 2
        assert results[0].alpha_net == pytest.approx(0.8 * 0.9)
        assert results[1].alpha_net == pytest.approx(0.3 * 0.5)

    def test_empty_signals(self):
        """Empty input produces empty output."""
        result = normalize_signals([], "1Day")
        assert result == []

    def test_preserves_signal_fields(self):
        """Normalization preserves all original Signal fields."""
        sig = _sig(
            strategy_id="strat_a",
            symbol="AAPL",
            side=Side.LONG,
            strength=0.8,
            confidence=0.9,
            horizon_bars=5,
        )
        [result] = normalize_signals([sig], "1Day", cost_bps={})
        assert result.strategy_id == sig.strategy_id
        assert result.symbol == sig.symbol
        assert result.side == sig.side
        assert result.strength == sig.strength
        assert result.confidence == sig.confidence
        assert result.horizon_bars == sig.horizon_bars
