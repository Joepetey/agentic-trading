"""Tests for signal deconfliction."""

from __future__ import annotations

import pytest

from src.orchestrator.deconflict import deconflict_signals
from src.orchestrator.models import DropReason
from src.strategies.signal import EntryType, Side, Signal


def _sig(
    strategy_id: str = "strat_a",
    symbol: str = "AAPL",
    side: Side = Side.LONG,
    strength: float = 0.8,
    confidence: float = 0.9,
    horizon_bars: int = 5,
    stop_price: float | None = None,
    take_profit_price: float | None = None,
) -> Signal:
    return Signal(
        strategy_id=strategy_id,
        symbol=symbol,
        side=side,
        strength=strength,
        confidence=confidence,
        horizon_bars=horizon_bars,
        stop_price=stop_price,
        take_profit_price=take_profit_price,
    )


UNIVERSE = ("AAPL", "MSFT", "GOOGL")


class TestSingleStrategy:
    def test_single_signal_passes_through(self):
        """Single signal for single symbol merges as-is."""
        sig = _sig()
        merged, dropped = deconflict_signals([sig], UNIVERSE)

        assert len(merged) == 1
        assert len(dropped) == 0
        m = merged[0]
        assert m.symbol == "AAPL"
        assert m.side == "long"
        assert m.agg_strength == pytest.approx(0.8)
        assert m.agg_confidence == pytest.approx(0.9)
        assert m.horizon_bars == 5

    def test_single_short_signal(self):
        sig = _sig(side=Side.SHORT, strength=0.6)
        merged, dropped = deconflict_signals([sig], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].side == "short"


class TestMultipleStrategiesSameSymbol:
    def test_two_long_same_symbol(self):
        """Two LONG signals for same symbol are merged."""
        s1 = _sig(strategy_id="strat_a", strength=0.8, confidence=0.9)
        s2 = _sig(strategy_id="strat_b", strength=0.6, confidence=0.7)
        merged, dropped = deconflict_signals([s1, s2], UNIVERSE)

        assert len(merged) == 1
        assert len(dropped) == 0
        m = merged[0]
        assert m.side == "long"
        assert len(m.contributions) == 2
        # Weighted average: strength should be between 0.6 and 0.8
        assert 0.6 < m.agg_strength < 0.8

    def test_opposite_sides_net_vote_determines_winner(self):
        """LONG vs SHORT: net vote determines winner, loser dropped."""
        s_long = _sig(strategy_id="strat_a", side=Side.LONG, strength=0.9, confidence=0.9)
        s_short = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.3, confidence=0.5)
        merged, dropped = deconflict_signals([s_long, s_short], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].side == "long"
        assert len(dropped) == 1
        assert dropped[0].reason == DropReason.CONFLICTING_SIDES
        assert dropped[0].strategy_id == "strat_b"

    def test_perfect_cancellation(self):
        """Equal weight opposite signals both dropped."""
        s1 = _sig(strategy_id="strat_a", side=Side.LONG, strength=0.5, confidence=0.8)
        s2 = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.5, confidence=0.8)
        merged, dropped = deconflict_signals([s1, s2], UNIVERSE)

        assert len(merged) == 0
        assert len(dropped) == 2
        assert all(d.reason == DropReason.CONFLICTING_SIDES for d in dropped)


class TestFlatSignals:
    def test_flat_exit_signal(self):
        """FLAT with negative strength is an exit intent."""
        sig = _sig(side=Side.FLAT, strength=-0.8, confidence=0.9)
        merged, dropped = deconflict_signals([sig], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].side == "flat"
        assert merged[0].agg_strength < 0

    def test_exit_wins_tie_against_directional(self):
        """When exit and directional have equal weight, exit wins."""
        s_exit = _sig(strategy_id="strat_exit", side=Side.FLAT, strength=-0.5, confidence=0.8)
        s_long = _sig(strategy_id="strat_long", side=Side.LONG, strength=0.5, confidence=0.8)
        merged, dropped = deconflict_signals([s_exit, s_long], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].side == "flat"
        assert any(d.strategy_id == "strat_long" for d in dropped)

    def test_directional_wins_when_stronger(self):
        """Directional outweighs exit when significantly stronger."""
        s_exit = _sig(strategy_id="strat_exit", side=Side.FLAT, strength=-0.3, confidence=0.5)
        s_long = _sig(strategy_id="strat_long", side=Side.LONG, strength=0.9, confidence=0.9)
        merged, dropped = deconflict_signals([s_exit, s_long], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].side == "long"
        assert any(d.strategy_id == "strat_exit" for d in dropped)


class TestDropping:
    def test_zero_strength_dropped(self):
        """Signals with strength=0.0 are always dropped."""
        sig = _sig(strength=0.0)
        merged, dropped = deconflict_signals([sig], UNIVERSE)

        assert len(merged) == 0
        assert len(dropped) == 1
        assert dropped[0].reason == DropReason.ZERO_STRENGTH

    def test_symbol_not_in_universe(self):
        """Signals for excluded symbols are dropped."""
        sig = _sig(symbol="TSLA")
        merged, dropped = deconflict_signals([sig], UNIVERSE)

        assert len(merged) == 0
        assert len(dropped) == 1
        assert dropped[0].reason == DropReason.SYMBOL_EXCLUDED
        assert "TSLA" in dropped[0].detail


class TestStrategyWeights:
    def test_custom_weights_shift_consensus(self):
        """Higher-weighted strategy tips the consensus."""
        s_long = _sig(strategy_id="strat_a", side=Side.LONG, strength=0.5, confidence=0.8)
        s_short = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.5, confidence=0.8)

        # Without custom weights, these perfectly cancel
        merged, _ = deconflict_signals([s_long, s_short], UNIVERSE)
        assert len(merged) == 0

        # With strat_a weighted 2x, LONG wins
        merged, dropped = deconflict_signals(
            [s_long, s_short], UNIVERSE,
            strategy_weights={"strat_a": 2.0},
        )
        assert len(merged) == 1
        assert merged[0].side == "long"


class TestStopAndTakeProfit:
    def test_tightest_stop_for_long(self):
        """For long: tightest stop = highest stop_price."""
        s1 = _sig(strategy_id="strat_a", stop_price=140.0)
        s2 = _sig(strategy_id="strat_b", stop_price=145.0)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert merged[0].stop_hint == 145.0  # tightest for long

    def test_tightest_stop_for_short(self):
        """For short: tightest stop = lowest stop_price."""
        s1 = _sig(strategy_id="strat_a", side=Side.SHORT, strength=0.8, stop_price=160.0)
        s2 = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.7, stop_price=155.0)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert merged[0].stop_hint == 155.0

    def test_nearest_tp_for_long(self):
        """For long: nearest TP = lowest take_profit_price."""
        s1 = _sig(strategy_id="strat_a", take_profit_price=170.0)
        s2 = _sig(strategy_id="strat_b", take_profit_price=165.0)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert merged[0].tp_hint == 165.0

    def test_nearest_tp_for_short(self):
        """For short: nearest TP = highest take_profit_price."""
        s1 = _sig(strategy_id="strat_a", side=Side.SHORT, strength=0.8, take_profit_price=130.0)
        s2 = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.7, take_profit_price=135.0)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert merged[0].tp_hint == 135.0

    def test_no_stop_or_tp(self):
        """No stop/tp hints when no signals provide them."""
        sig = _sig()
        merged, _ = deconflict_signals([sig], UNIVERSE)
        assert merged[0].stop_hint is None
        assert merged[0].tp_hint is None


class TestMultipleSymbols:
    def test_independent_per_symbol(self):
        """Each symbol is processed independently."""
        s1 = _sig(strategy_id="strat_a", symbol="AAPL", strength=0.8)
        s2 = _sig(strategy_id="strat_a", symbol="MSFT", strength=0.6)
        merged, dropped = deconflict_signals([s1, s2], UNIVERSE)

        assert len(merged) == 2
        assert len(dropped) == 0
        symbols = {m.symbol for m in merged}
        assert symbols == {"AAPL", "MSFT"}

    def test_sorted_by_strength(self):
        """Output sorted by abs(agg_strength) descending."""
        s1 = _sig(strategy_id="strat_a", symbol="AAPL", strength=0.3, confidence=0.9)
        s2 = _sig(strategy_id="strat_a", symbol="MSFT", strength=0.9, confidence=0.9)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert merged[0].symbol == "MSFT"
        assert merged[1].symbol == "AAPL"


class TestDeterminism:
    def test_same_inputs_same_outputs(self):
        """Same inputs produce same outputs."""
        signals = [
            _sig(strategy_id="strat_a", symbol="AAPL", strength=0.8),
            _sig(strategy_id="strat_b", symbol="AAPL", strength=0.6),
            _sig(strategy_id="strat_a", symbol="MSFT", strength=0.5),
        ]

        r1 = deconflict_signals(signals, UNIVERSE)
        r2 = deconflict_signals(signals, UNIVERSE)

        assert len(r1[0]) == len(r2[0])
        assert len(r1[1]) == len(r2[1])
        for m1, m2 in zip(r1[0], r2[0]):
            assert m1.symbol == m2.symbol
            assert m1.agg_strength == pytest.approx(m2.agg_strength)
            assert m1.agg_confidence == pytest.approx(m2.agg_confidence)


class TestEdgeCases:
    def test_empty_signals(self):
        merged, dropped = deconflict_signals([], UNIVERSE)
        assert merged == []
        assert dropped == []

    def test_empty_universe(self):
        """All signals dropped when universe is empty."""
        sig = _sig()
        merged, dropped = deconflict_signals([sig], ())
        assert len(merged) == 0
        assert len(dropped) == 1

    def test_provenance_contribution_count(self):
        """Contributions tuple length matches number of same-side signals."""
        s1 = _sig(strategy_id="strat_a")
        s2 = _sig(strategy_id="strat_b")
        s3 = _sig(strategy_id="strat_c")
        merged, _ = deconflict_signals([s1, s2, s3], UNIVERSE)

        assert len(merged) == 1
        assert len(merged[0].contributions) == 3
