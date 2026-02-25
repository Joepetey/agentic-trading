"""Tests for Signal, Side, EntryType, InvalidateCondition, and related enums."""

from __future__ import annotations

import pytest

from src.strategies.signal import (
    CompareOp,
    EntryType,
    InvalidateCondition,
    PriceField,
    Side,
    Signal,
)


# ── Enum basics ───────────────────────────────────────────────────────


class TestSide:
    def test_values(self) -> None:
        assert Side.LONG.value == "long"
        assert Side.SHORT.value == "short"
        assert Side.FLAT.value == "flat"

    def test_str_serialisation(self) -> None:
        assert str(Side.LONG) == "Side.LONG"
        assert Side.LONG == "long"  # str mixin


class TestEntryType:
    def test_values(self) -> None:
        assert EntryType.MARKET.value == "market"
        assert EntryType.LIMIT.value == "limit"
        assert EntryType.STOP.value == "stop"
        assert EntryType.STOP_LIMIT.value == "stop_limit"


# ── InvalidateCondition ──────────────────────────────────────────────


class TestInvalidateCondition:
    def test_construction(self) -> None:
        cond = InvalidateCondition(
            field=PriceField.CLOSE,
            op=CompareOp.GT,
            value=150.0,
        )
        assert cond.field == PriceField.CLOSE
        assert cond.op == CompareOp.GT
        assert cond.value == 150.0

    def test_frozen(self) -> None:
        cond = InvalidateCondition(
            field=PriceField.HIGH, op=CompareOp.LT, value=200.0,
        )
        with pytest.raises(Exception):  # ValidationError on frozen model
            cond.value = 999.0  # type: ignore[misc]


# ── Signal construction + clamping ───────────────────────────────────


def _make_signal(**overrides) -> Signal:
    defaults = dict(
        strategy_id="test_strat",
        symbol="AAPL",
        side=Side.LONG,
        strength=0.5,
        confidence=0.8,
        horizon_bars=10,
    )
    defaults.update(overrides)
    return Signal(**defaults)


class TestSignalClamping:
    def test_strength_clamped_high(self) -> None:
        sig = _make_signal(strength=2.5)
        assert sig.strength == 1.0

    def test_strength_clamped_low(self) -> None:
        sig = _make_signal(strength=-3.0)
        assert sig.strength == -1.0

    def test_strength_in_range_unchanged(self) -> None:
        sig = _make_signal(strength=-0.75)
        assert sig.strength == -0.75

    def test_confidence_clamped_high(self) -> None:
        sig = _make_signal(confidence=1.5)
        assert sig.confidence == 1.0

    def test_confidence_clamped_low(self) -> None:
        sig = _make_signal(confidence=-0.3)
        assert sig.confidence == 0.0

    def test_confidence_in_range_unchanged(self) -> None:
        sig = _make_signal(confidence=0.42)
        assert sig.confidence == 0.42


class TestSignalFrozen:
    def test_cannot_set_field(self) -> None:
        sig = _make_signal()
        with pytest.raises(Exception):
            sig.strength = 0.99  # type: ignore[misc]


class TestSignalDefaults:
    def test_entry_defaults_to_market(self) -> None:
        sig = _make_signal()
        assert sig.entry == EntryType.MARKET

    def test_optional_fields_default_to_none(self) -> None:
        sig = _make_signal()
        assert sig.entry_price_hint is None
        assert sig.stop_price is None
        assert sig.take_profit_price is None
        assert sig.time_stop_bars is None

    def test_invalidate_defaults_empty(self) -> None:
        sig = _make_signal()
        assert sig.invalidate == ()

    def test_tags_defaults_empty(self) -> None:
        sig = _make_signal()
        assert sig.tags == ()

    def test_explain_defaults_empty(self) -> None:
        sig = _make_signal()
        assert sig.explain == ""


class TestSignalValidation:
    def test_horizon_bars_must_be_positive(self) -> None:
        with pytest.raises(Exception):
            _make_signal(horizon_bars=0)

    def test_horizon_bars_negative_rejected(self) -> None:
        with pytest.raises(Exception):
            _make_signal(horizon_bars=-5)

    def test_time_stop_bars_must_be_positive_if_set(self) -> None:
        with pytest.raises(Exception):
            _make_signal(time_stop_bars=0)


# ── Signal ordering ──────────────────────────────────────────────────


class TestSignalOrdering:
    def test_higher_abs_strength_sorts_first(self) -> None:
        weak = _make_signal(strength=0.3, confidence=0.8, symbol="AAPL")
        strong = _make_signal(strength=-0.9, confidence=0.8, symbol="AAPL")
        assert strong < weak  # strong has higher |strength|, lower sort key

    def test_same_strength_higher_confidence_sorts_first(self) -> None:
        low_conf = _make_signal(strength=0.5, confidence=0.3, symbol="AAPL")
        high_conf = _make_signal(strength=0.5, confidence=0.9, symbol="AAPL")
        assert high_conf < low_conf

    def test_same_strength_confidence_sorts_by_symbol(self) -> None:
        apple = _make_signal(strength=0.5, confidence=0.8, symbol="AAPL")
        meta = _make_signal(strength=0.5, confidence=0.8, symbol="META")
        assert apple < meta  # AAPL < META alphabetically

    def test_sorted_list(self) -> None:
        signals = [
            _make_signal(strength=0.2, confidence=0.5, symbol="C"),
            _make_signal(strength=-0.9, confidence=0.9, symbol="A"),
            _make_signal(strength=0.5, confidence=0.7, symbol="B"),
        ]
        ordered = sorted(signals)
        assert ordered[0].symbol == "A"  # |strength|=0.9
        assert ordered[1].symbol == "B"  # |strength|=0.5
        assert ordered[2].symbol == "C"  # |strength|=0.2


# ── Signal serialisation ─────────────────────────────────────────────


class TestSignalModelDump:
    def test_model_dump_contains_all_fields(self) -> None:
        sig = _make_signal(
            stop_price=145.0,
            tags=("trend", "earnings_risk"),
            invalidate=(
                InvalidateCondition(
                    field=PriceField.CLOSE,
                    op=CompareOp.GT,
                    value=160.0,
                ),
            ),
            explain="test reason",
        )
        d = sig.model_dump()
        assert d["strategy_id"] == "test_strat"
        assert d["symbol"] == "AAPL"
        assert d["side"] == "long"
        assert d["stop_price"] == 145.0
        assert d["tags"] == ("trend", "earnings_risk")
        assert len(d["invalidate"]) == 1
        assert d["invalidate"][0]["field"] == "close"
        assert d["explain"] == "test reason"
