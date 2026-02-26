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
    tags: tuple[str, ...] = (),
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
        tags=tags,
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


class TestAlphaNet:
    """Tests for alpha_net-aware deconfliction."""

    def test_alpha_net_used_for_voting(self):
        """Signals with alpha_net use it for effective weight."""
        s_long = _sig(strategy_id="strat_a", strength=0.5, confidence=0.8)
        s_short = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.5, confidence=0.8)

        # Without alpha_net, these cancel (equal |strength| * confidence)
        merged, _ = deconflict_signals([s_long, s_short], UNIVERSE)
        assert len(merged) == 0

        # With alpha_net, strat_a has 2x the weight → LONG wins
        s_long_normed = s_long.model_copy(update={"alpha_net": 0.8})
        s_short_normed = s_short.model_copy(update={"alpha_net": -0.2})
        merged, dropped = deconflict_signals([s_long_normed, s_short_normed], UNIVERSE)
        assert len(merged) == 1
        assert merged[0].side == "long"

    def test_alpha_net_zero_dropped(self):
        """Signals with alpha_net=0.0 are dropped as BELOW_COST_THRESHOLD."""
        sig = _sig(strength=0.5, confidence=0.8)
        sig_zero = sig.model_copy(update={"alpha_net": 0.0})
        merged, dropped = deconflict_signals([sig_zero], UNIVERSE)

        assert len(merged) == 0
        assert len(dropped) == 1
        assert dropped[0].reason == DropReason.BELOW_COST_THRESHOLD

    def test_agg_alpha_in_merged(self):
        """MergedSignal carries agg_alpha when signals have alpha_net."""
        sig = _sig(strength=0.8, confidence=0.9)
        sig_normed = sig.model_copy(update={"alpha_net": 0.6})
        merged, _ = deconflict_signals([sig_normed], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].agg_alpha == pytest.approx(0.6)

    def test_agg_alpha_none_without_normalization(self):
        """MergedSignal.agg_alpha is None when signals lack alpha_net."""
        sig = _sig()
        merged, _ = deconflict_signals([sig], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].agg_alpha is None

    def test_backward_compat_no_alpha_net(self):
        """Signals without alpha_net (None) use classic weight formula."""
        s1 = _sig(strategy_id="strat_a", strength=0.8, confidence=0.9)
        s2 = _sig(strategy_id="strat_b", strength=0.6, confidence=0.7)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].side == "long"
        # Classic formula: agg_strength is weighted average
        assert 0.6 < merged[0].agg_strength < 0.8

    def test_contribution_carries_alpha_net(self):
        """SignalContribution records alpha_net for audit."""
        sig = _sig(strength=0.8, confidence=0.9)
        sig_normed = sig.model_copy(update={"alpha_net": 0.5})
        merged, _ = deconflict_signals([sig_normed], UNIVERSE)

        assert merged[0].contributions[0].alpha_net == pytest.approx(0.5)


class TestVeto:
    """Tests for veto tag handling."""

    def test_veto_tag_drops_all_signals(self):
        """Signal with do_not_trade tag drops all signals for that symbol."""
        s_veto = _sig(
            strategy_id="risk_tool", side=Side.FLAT, strength=-0.5,
            confidence=1.0, tags=("do_not_trade",),
        )
        s_long = _sig(strategy_id="strat_a", strength=0.8)
        merged, dropped = deconflict_signals([s_veto, s_long], UNIVERSE)

        assert len(merged) == 0
        assert len(dropped) == 2
        assert all(d.reason == DropReason.VETOED for d in dropped)

    def test_veto_only_affects_vetoed_symbol(self):
        """Veto on AAPL does not affect MSFT."""
        s_veto = _sig(
            strategy_id="risk_tool", symbol="AAPL", side=Side.FLAT,
            strength=-0.5, confidence=1.0, tags=("do_not_trade",),
        )
        s_aapl = _sig(strategy_id="strat_a", symbol="AAPL", strength=0.8)
        s_msft = _sig(strategy_id="strat_a", symbol="MSFT", strength=0.6)
        merged, dropped = deconflict_signals([s_veto, s_aapl, s_msft], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].symbol == "MSFT"
        vetoed = [d for d in dropped if d.reason == DropReason.VETOED]
        assert len(vetoed) == 2
        assert all(d.symbol == "AAPL" for d in vetoed)

    def test_veto_with_multiple_strategies(self):
        """Veto drops the veto signal itself and all other strategies' signals."""
        s_veto = _sig(
            strategy_id="risk_tool", side=Side.FLAT, strength=-0.3,
            confidence=1.0, tags=("do_not_trade",),
        )
        s_a = _sig(strategy_id="strat_a", strength=0.8)
        s_b = _sig(strategy_id="strat_b", strength=0.6)
        merged, dropped = deconflict_signals([s_veto, s_a, s_b], UNIVERSE)

        assert len(merged) == 0
        assert len(dropped) == 3
        strategy_ids = {d.strategy_id for d in dropped}
        assert strategy_ids == {"risk_tool", "strat_a", "strat_b"}

    def test_custom_veto_tag(self):
        """Custom veto_tags parameter works."""
        s_halt = _sig(
            strategy_id="risk_tool", strength=0.5, tags=("halt_trading",),
        )
        s_long = _sig(strategy_id="strat_a", strength=0.8)
        # Default veto_tags doesn't include "halt_trading" → no veto
        merged, dropped = deconflict_signals([s_halt, s_long], UNIVERSE)
        assert len(merged) == 1

        # With custom veto_tags → veto triggers
        merged, dropped = deconflict_signals(
            [s_halt, s_long], UNIVERSE,
            veto_tags=("halt_trading",),
        )
        assert len(merged) == 0
        assert all(d.reason == DropReason.VETOED for d in dropped)

    def test_no_veto_when_tag_absent(self):
        """Normal signals without veto tags are unaffected."""
        s1 = _sig(strategy_id="strat_a", tags=("momentum",))
        merged, dropped = deconflict_signals([s1], UNIVERSE)
        assert len(merged) == 1
        assert len(dropped) == 0


class TestRegimePrecedence:
    """Tests for regime-based weight multipliers."""

    def test_regime_tips_consensus(self):
        """In trend regime, trend-category strategy gets 2x, tipping consensus."""
        s_long = _sig(strategy_id="trend_strat", side=Side.LONG, strength=0.5, confidence=0.8)
        s_short = _sig(strategy_id="mr_strat", side=Side.SHORT, strength=0.5, confidence=0.8)

        # Without regime, these cancel
        merged, _ = deconflict_signals([s_long, s_short], UNIVERSE)
        assert len(merged) == 0

        # With trend regime, trend gets 2x → LONG wins
        merged, dropped = deconflict_signals(
            [s_long, s_short], UNIVERSE,
            regime="trend",
            regime_weights={"trend": {"trend": 2.0, "mean_rev": 0.5}},
            strategy_categories={"trend_strat": "trend", "mr_strat": "mean_rev"},
        )
        assert len(merged) == 1
        assert merged[0].side == "long"

    def test_regime_by_category(self):
        """Category-based regime weight lookup works."""
        s1 = _sig(strategy_id="strat_a", strength=0.5, confidence=0.8)
        s2 = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.5, confidence=0.8)

        merged, _ = deconflict_signals(
            [s1, s2], UNIVERSE,
            regime="chop",
            regime_weights={"chop": {"mean_rev": 2.0}},
            strategy_categories={"strat_b": "mean_rev"},
        )
        # strat_b (mean_rev) gets 2x in chop → SHORT wins
        assert len(merged) == 1
        assert merged[0].side == "short"

    def test_regime_by_strategy_id(self):
        """strategy_id takes precedence over category in regime_weights."""
        s1 = _sig(strategy_id="strat_a", strength=0.5, confidence=0.8)
        s2 = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.5, confidence=0.8)

        merged, _ = deconflict_signals(
            [s1, s2], UNIVERSE,
            regime="trend",
            # strat_a has both category and direct ID match — ID takes precedence
            regime_weights={"trend": {"strat_a": 3.0, "trend": 0.5}},
            strategy_categories={"strat_a": "trend"},
        )
        # strat_a gets 3.0 (by ID, not 0.5 by category) → LONG wins
        assert len(merged) == 1
        assert merged[0].side == "long"

    def test_no_regime_no_effect(self):
        """regime=None leaves behavior unchanged."""
        s1 = _sig(strategy_id="strat_a", strength=0.5, confidence=0.8)
        s2 = _sig(strategy_id="strat_b", side=Side.SHORT, strength=0.5, confidence=0.8)

        # With regime_weights defined but regime=None → no effect
        merged, _ = deconflict_signals(
            [s1, s2], UNIVERSE,
            regime=None,
            regime_weights={"trend": {"strat_a": 10.0}},
            strategy_categories={"strat_a": "trend"},
        )
        assert len(merged) == 0  # still cancels


class TestAlphaThreshold:
    """Tests for post-merge alpha threshold filtering."""

    def test_below_threshold_dropped(self):
        """Weak merged signals are filtered out."""
        sig = _sig(strength=0.5, confidence=0.8)
        sig_normed = sig.model_copy(update={"alpha_net": 0.01})
        merged, dropped = deconflict_signals(
            [sig_normed], UNIVERSE,
            min_symbol_alpha=0.05,
        )

        assert len(merged) == 0
        assert len(dropped) == 1
        assert dropped[0].reason == DropReason.BELOW_ALPHA_THRESHOLD
        assert "0.0100" in dropped[0].detail

    def test_above_threshold_kept(self):
        """Strong signals pass through."""
        sig = _sig(strength=0.8, confidence=0.9)
        sig_normed = sig.model_copy(update={"alpha_net": 0.5})
        merged, dropped = deconflict_signals(
            [sig_normed], UNIVERSE,
            min_symbol_alpha=0.05,
        )

        assert len(merged) == 1
        assert merged[0].agg_alpha == pytest.approx(0.5)

    def test_zero_threshold_keeps_all(self):
        """min_symbol_alpha=0.0 (default) means no filtering."""
        sig = _sig(strength=0.1, confidence=0.1)
        sig_normed = sig.model_copy(update={"alpha_net": 0.001})
        merged, dropped = deconflict_signals([sig_normed], UNIVERSE)

        assert len(merged) == 1

    def test_threshold_only_filters_alpha_signals(self):
        """Non-normalized signals (agg_alpha=None) are not filtered by threshold."""
        sig = _sig(strength=0.1, confidence=0.1)
        merged, dropped = deconflict_signals(
            [sig], UNIVERSE,
            min_symbol_alpha=0.5,
        )

        # agg_alpha is None → threshold check skipped
        assert len(merged) == 1


class TestAggAlphaSum:
    """Tests for agg_alpha = sum of alpha_net (not weighted average)."""

    def test_agg_alpha_is_sum(self):
        """agg_alpha is the sum of alpha_net values for multi-signal merge."""
        s1 = _sig(strategy_id="strat_a", strength=0.8, confidence=0.9)
        s2 = _sig(strategy_id="strat_b", strength=0.6, confidence=0.7)
        s1_normed = s1.model_copy(update={"alpha_net": 0.3})
        s2_normed = s2.model_copy(update={"alpha_net": 0.2})

        merged, _ = deconflict_signals([s1_normed, s2_normed], UNIVERSE)

        assert len(merged) == 1
        # Sum, not average: 0.3 + 0.2 = 0.5, not (0.3 + 0.2) / 2 = 0.25
        assert merged[0].agg_alpha == pytest.approx(0.5)

    def test_agg_alpha_sum_three_signals(self):
        """Three same-side signals: agg_alpha = sum of all alpha_net."""
        s1 = _sig(strategy_id="strat_a", strength=0.8, confidence=0.9)
        s2 = _sig(strategy_id="strat_b", strength=0.6, confidence=0.7)
        s3 = _sig(strategy_id="strat_c", strength=0.5, confidence=0.8)
        s1_n = s1.model_copy(update={"alpha_net": 0.4})
        s2_n = s2.model_copy(update={"alpha_net": 0.3})
        s3_n = s3.model_copy(update={"alpha_net": 0.2})

        merged, _ = deconflict_signals([s1_n, s2_n, s3_n], UNIVERSE)

        assert len(merged) == 1
        assert merged[0].agg_alpha == pytest.approx(0.9)


class TestWeightedAvgHorizon:
    """Tests for horizon_bars = weighted average (not min)."""

    def test_weighted_avg_horizon(self):
        """Horizon is weighted average of contributors."""
        # strat_a: strength=0.8, conf=0.9 → weight ≈ 0.72
        # strat_b: strength=0.4, conf=0.5 → weight ≈ 0.20
        s1 = _sig(strategy_id="strat_a", strength=0.8, confidence=0.9, horizon_bars=20)
        s2 = _sig(strategy_id="strat_b", strength=0.4, confidence=0.5, horizon_bars=5)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert len(merged) == 1
        # Weighted avg: (20*0.72 + 5*0.20) / (0.72+0.20) ≈ 16.7 → 17
        # (Not min=5)
        assert merged[0].horizon_bars > 5
        assert merged[0].horizon_bars <= 20

    def test_same_horizon_unchanged(self):
        """When all signals have the same horizon, result is that value."""
        s1 = _sig(strategy_id="strat_a", horizon_bars=10)
        s2 = _sig(strategy_id="strat_b", horizon_bars=10)
        merged, _ = deconflict_signals([s1, s2], UNIVERSE)

        assert merged[0].horizon_bars == 10


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
