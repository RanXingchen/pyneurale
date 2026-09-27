#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.decoding.sequence import Vocabulary
from neurale.exceptions import ValidationError

_TOKENS = ["alpha", "beta", "gamma", "<end>"]


def _vocabulary(**overrides: object) -> Vocabulary:
    settings: dict[str, object] = {
        "tokens": list(_TOKENS),
        "bos": "<start>",
        "eos": "<end>",
    }
    settings.update(overrides)
    return Vocabulary(**settings)


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def test_declared_order_is_column_order() -> None:
    vocabulary = _vocabulary()

    assert vocabulary.tokens == tuple(_TOKENS)
    assert vocabulary.size == 4
    assert [vocabulary.index(token) for token in _TOKENS] == [0, 1, 2, 3]
    assert [vocabulary.token(idx) for idx in range(4)] == _TOKENS


def test_reordered_vocabulary_is_different() -> None:
    """Order is contract, not presentation: it settles ties during a search."""

    forward = _vocabulary()
    reversed_order = _vocabulary(tokens=["beta", "alpha", "gamma", "<end>"])

    assert forward != reversed_order
    assert forward.index("alpha") == 0
    assert reversed_order.index("alpha") == 1


def test_eos_is_ordinary_token_with_column() -> None:
    vocabulary = _vocabulary()

    assert vocabulary.eos == "<end>"
    assert vocabulary.eos_index == 3
    assert vocabulary.eos in vocabulary


def test_bos_is_context_never_emission() -> None:
    vocabulary = _vocabulary()

    assert vocabulary.bos == "<start>"
    assert vocabulary.bos not in vocabulary
    assert vocabulary.bos_index == vocabulary.size
    assert vocabulary.n_contexts == vocabulary.size + 1
    assert vocabulary.context_index(vocabulary.bos) == 4
    assert vocabulary.context_index("beta") == 1


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tokens": []}, "at least one token"),
        ({"tokens": "abc"}, "sequence of strings"),
        ({"tokens": ["a", "b", "a", "<end>"]}, "must not repeat"),
        ({"tokens": ["a", "", "<end>"]}, "must not be empty"),
        ({"tokens": ["a", 3, "<end>"]}, "must be a string"),
        ({"bos": "alpha"}, "bos must not be one of tokens"),
        ({"bos": ""}, "bos must not be empty"),
        ({"bos": 7}, "bos must be a string"),
        ({"eos": "absent"}, "eos must be one of tokens"),
        ({"unknown": "absent"}, "unknown must be one of tokens or None"),
    ],
)
def test_invalid_vocabularies_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _vocabulary(**overrides)


def test_tokens_may_be_any_length() -> None:
    """Nothing here assumes a token is one character wide."""

    vocabulary = Vocabulary(
        ["select", "backspace", "END OF MESSAGE"],
        bos="BEGIN",
        eos="END OF MESSAGE",
    )

    assert vocabulary.index("backspace") == 1
    assert vocabulary.token(2) == "END OF MESSAGE"


# --------------------------------------------------------------------------------------
# The unknown-token policy
# --------------------------------------------------------------------------------------


def test_unknown_token_is_error_by_default() -> None:
    vocabulary = _vocabulary()

    assert vocabulary.unknown is None
    with pytest.raises(ValidationError, match="not in the vocabulary"):
        vocabulary.index("delta")


def test_configured_unknown_token_absorbs_oov() -> None:
    vocabulary = _vocabulary(tokens=[*_TOKENS, "<unk>"], unknown="<unk>")

    assert vocabulary.index("delta") == vocabulary.index("<unk>") == 4
    assert vocabulary.index("alpha") == 0


def test_unknown_cannot_be_eos() -> None:
    """Folding an unseen token onto eos would end the sequence, not stand in for it.

    Accepting this would let a corpus be counted with transitions out of eos,
    which is exactly what eos is defined to make impossible.
    """

    with pytest.raises(ValidationError, match="unknown must differ from eos"):
        _vocabulary(unknown="<end>")


def test_symbol_roles_are_consistent() -> None:
    """The four rules the class documents, frozen together."""

    vocabulary = _vocabulary(tokens=[*_TOKENS, "<unk>"], unknown="<unk>")

    assert vocabulary.bos not in vocabulary.tokens
    assert vocabulary.eos in vocabulary.tokens
    assert vocabulary.unknown in vocabulary.tokens
    assert vocabulary.unknown != vocabulary.eos
    assert vocabulary.unknown != vocabulary.bos


@pytest.mark.parametrize("token", [123, 1.5, None, b"alpha", ["alpha"]])
def test_non_string_token_is_error_with_unknown_configured(token: object) -> None:
    """The unknown policy covers unseen tokens, not input of the wrong type."""

    vocabulary = _vocabulary(tokens=[*_TOKENS, "<unk>"], unknown="<unk>")

    with pytest.raises(ValidationError, match="token must be a string"):
        vocabulary.index(token)


@pytest.mark.parametrize("unknown", [None, "<unk>"], ids=["no-unknown", "with-unknown"])
def test_bos_has_no_emission_column_and_never_folds_onto_unknown(unknown: str | None) -> None:
    """bos is a context. Handing back the <unk> column would encode a caller's mistake.

    The class contract says bos is never emitted, so a configured unknown
    policy must not quietly give it a column anyway -- ``encode([bos])`` would
    then produce a plausible-looking sequence that means something else.
    """

    vocabulary = _vocabulary(tokens=[*_TOKENS, "<unk>"], unknown=unknown)

    with pytest.raises(ValidationError, match="has no emission column"):
        vocabulary.index(vocabulary.bos)
    with pytest.raises(ValidationError, match="has no emission column"):
        vocabulary.encode(["alpha", vocabulary.bos])


def test_bos_remains_usable_as_context() -> None:
    vocabulary = _vocabulary(tokens=[*_TOKENS, "<unk>"], unknown="<unk>")

    assert vocabulary.context_index(vocabulary.bos) == vocabulary.bos_index


def test_numpy_string_token_is_ordinary() -> None:
    vocabulary = _vocabulary()

    assert vocabulary.index(np.str_("alpha")) == 0


def test_encoding_and_decoding_round_trip() -> None:
    vocabulary = _vocabulary()
    indices = vocabulary.encode(["gamma", "alpha", "<end>"])

    assert np.array_equal(indices, [2, 0, 3])
    assert vocabulary.decode(indices) == ("gamma", "alpha", "<end>")


@pytest.mark.parametrize("idx", [-1, 4, 100])
def test_out_of_range_column_is_rejected(idx: int) -> None:
    with pytest.raises(ValidationError, match=r"index must lie in \[0, 4\)"):
        _vocabulary().token(idx)


def test_non_integer_column_is_rejected() -> None:
    with pytest.raises(ValidationError, match="index must be an integer"):
        _vocabulary().token("alpha")


# --------------------------------------------------------------------------------------
# Value semantics
# --------------------------------------------------------------------------------------


def test_equal_vocabularies_compare_and_hash_equal() -> None:
    assert _vocabulary() == _vocabulary()
    assert hash(_vocabulary()) == hash(_vocabulary())
    assert _vocabulary() != _vocabulary(unknown="gamma")
    assert _vocabulary() != "not a vocabulary"


def test_repr_names_boundary_symbols() -> None:
    text = repr(_vocabulary())

    assert "size=4" in text
    assert "'<start>'" in text
    assert "'<end>'" in text
