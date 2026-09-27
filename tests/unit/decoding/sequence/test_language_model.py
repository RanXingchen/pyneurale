#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.decoding.sequence import NgramLanguageModel, Vocabulary
from neurale.exceptions import ValidationError

# Three tokens and an explicit terminator, so every count below can be written
# out by hand and compared against the implementation rather than against
# another run of it.
_VOCABULARY = Vocabulary(["a", "b", "stop"], bos="<s>", eos="stop")

# Counted as "<s> a b stop", "<s> b a b stop", and "<s> a stop".
_CORPUS = [["a", "b"], ["b", "a", "b"], ["a"]]

# Emission counts: a appears in all three sequences; b once in the first and
# twice in the second; stop once per sequence because the model appends it.
_UNIGRAM_COUNTS = [3.0, 3.0, 3.0]

# Rows are contexts in vocabulary order, then the bos context last.
_TRANSITION_COUNTS = [
    [0.0, 2.0, 1.0],  # a -> b twice, a -> stop once
    [1.0, 0.0, 2.0],  # b -> a once, b -> stop twice
    [0.0, 0.0, 0.0],  # stop is never a context: counting ends there
    [2.0, 1.0, 0.0],  # two sequences start with a, one with b
]


def _model(**overrides: object) -> NgramLanguageModel:
    settings: dict[str, object] = {"vocabulary": _VOCABULARY}
    settings.update(overrides)
    return NgramLanguageModel(**settings)


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"vocabulary": "abc"}, "vocabulary must be a Vocabulary"),
        ({"order": 0}, "order must be positive"),
        ({"order": 3}, "order must be 1 or 2"),
        ({"order": 1.5}, "order"),
        ({"smoothing": -0.5}, "smoothing"),
        ({"smoothing": "heavy"}, "smoothing"),
    ],
)
def test_invalid_arguments_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _model(**overrides)


def test_unfitted_model_reports_nothing() -> None:
    model = _model()

    assert not model.is_fitted
    for name in (
        "unigram_counts_",
        "transition_counts_",
        "unigram_log_probabilities_",
        "transition_log_probabilities_",
        "n_sequences_",
        "n_tokens_",
    ):
        with pytest.raises(ValidationError, match="not fitted"):
            getattr(model, name)


# --------------------------------------------------------------------------------------
# Counts from an independently defined corpus
# --------------------------------------------------------------------------------------


def test_unigram_counts_match_hand_counted_corpus() -> None:
    model = _model().fit(_CORPUS)

    assert np.array_equal(model.unigram_counts_, _UNIGRAM_COUNTS)
    assert model.n_sequences_ == 3
    assert model.n_tokens_ == 9


def test_transition_counts_match_hand_counted_corpus() -> None:
    model = _model().fit(_CORPUS)

    assert np.array_equal(model.transition_counts_, _TRANSITION_COUNTS)


def test_bos_row_counts_sequence_starts() -> None:
    model = _model().fit(_CORPUS)
    starts = model.transition_counts_[_VOCABULARY.bos_index]

    assert np.array_equal(starts, [2.0, 1.0, 0.0])


def test_eos_is_counted_once_and_never_as_context() -> None:
    model = _model().fit(_CORPUS)

    assert model.unigram_counts_[_VOCABULARY.eos_index] == 3.0
    assert np.array_equal(model.transition_counts_[_VOCABULARY.eos_index], [0.0, 0.0, 0.0])


def test_counts_are_read_only() -> None:
    model = _model().fit(_CORPUS)

    with pytest.raises(ValueError, match="read-only"):
        model.transition_counts_[0, 0] = 99.0


# --------------------------------------------------------------------------------------
# Probabilities
# --------------------------------------------------------------------------------------


def test_unsmoothed_probabilities_are_normalized_counts() -> None:
    model = _model().fit(_CORPUS)

    assert model.probability("b", given="a") == pytest.approx(2 / 3)
    assert model.probability("stop", given="a") == pytest.approx(1 / 3)
    assert model.probability("a", given="a") == 0.0
    assert model.probability("a", given=_VOCABULARY.bos) == pytest.approx(2 / 3)
    assert model.probability("a") == pytest.approx(1 / 3)


def test_unseen_event_scores_zero_probability() -> None:
    model = _model().fit(_CORPUS)

    assert model.probability("a", given="a") == 0.0
    assert model.log_probability("a", given="a") == -np.inf


def test_unseen_context_has_no_distribution() -> None:
    """A uniform fallback would be an estimate the corpus never supported."""

    model = _model().fit(_CORPUS)
    row = model.log_transition("stop")

    assert np.all(np.isneginf(row))
    assert not np.any(np.isnan(row))


def test_additive_smoothing_moves_mass_onto_unseen_events() -> None:
    model = _model(smoothing=1.0).fit(_CORPUS)

    # Row "a" counted [0, 2, 1]; add-1 over three tokens gives [1, 3, 2] / 6.
    assert model.probability("a", given="a") == pytest.approx(1 / 6)
    assert model.probability("b", given="a") == pytest.approx(1 / 2)
    assert model.probability("stop", given="a") == pytest.approx(1 / 3)


def test_smoothing_gives_unseen_context_uniform_distribution() -> None:
    model = _model(smoothing=1.0).fit(_CORPUS)
    row = np.exp(model.log_transition("stop"))

    assert row == pytest.approx([1 / 3, 1 / 3, 1 / 3])


@pytest.mark.parametrize("smoothing", [0.0, 0.5, 1.0, 10.0])
def test_reachable_context_rows_are_distributions(smoothing: float) -> None:
    model = _model(smoothing=smoothing).fit(_CORPUS)
    rows = np.exp(model.transition_log_probabilities_)
    totals = rows.sum(axis=1)
    counted = model.transition_counts_.sum(axis=1) > 0

    assert totals[counted | (smoothing > 0)] == pytest.approx(1.0)


def test_unigram_distribution_is_normalized() -> None:
    model = _model(smoothing=0.0).fit([["a", "a"], ["a"]])

    # Emissions are a, a, stop, a, stop: five tokens, three of them "a".
    assert np.exp(model.unigram_log_probabilities_) == pytest.approx([3 / 5, 0.0, 2 / 5])
    assert model.n_tokens_ == 5


def test_smoothed_unigram_admits_unseen_token() -> None:
    model = _model(smoothing=1.0).fit([["a", "a"], ["a"]])

    assert np.exp(model.unigram_log_probabilities_) == pytest.approx([4 / 8, 1 / 8, 3 / 8])


# --------------------------------------------------------------------------------------
# Order
# --------------------------------------------------------------------------------------


def test_unigram_model_conditions_on_nothing() -> None:
    model = _model(order=1).fit(_CORPUS)
    rows = model.transition_log_probabilities_

    # Every context row, the bos row included, is the same distribution.
    for context in range(_VOCABULARY.n_contexts):
        assert np.array_equal(rows[context], rows[0])
    assert np.exp(rows[0]) == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_unigram_model_still_publishes_bigram_counts() -> None:
    """The counts describe the corpus; the order decides what is estimated from them."""

    model = _model(order=1).fit(_CORPUS)

    assert np.array_equal(model.transition_counts_, _TRANSITION_COUNTS)


def test_bigram_model_differs_from_unigram() -> None:
    unigram = _model(order=1).fit(_CORPUS)
    bigram = _model(order=2).fit(_CORPUS)

    assert unigram.probability("b", given="a") == pytest.approx(1 / 3)
    assert bigram.probability("b", given="a") == pytest.approx(2 / 3)


# --------------------------------------------------------------------------------------
# Joint sequence scores
# --------------------------------------------------------------------------------------


def test_sequence_score_sums_transition_log_probabilities() -> None:
    model = _model(smoothing=1.0).fit(_CORPUS)
    expected = (
        model.log_probability("a", given=_VOCABULARY.bos)
        + model.log_probability("b", given="a")
        + model.log_probability("stop", given="b")
    )

    assert model.sequence_log_probability(["a", "b"]) == pytest.approx(expected)


def test_score_without_boundaries_omits_terminator() -> None:
    model = _model(smoothing=1.0).fit(_CORPUS)
    expected = model.log_probability("a", given=_VOCABULARY.bos) + model.log_probability(
        "b", given="a"
    )

    assert model.sequence_log_probability(["a", "b"], boundaries=False) == pytest.approx(expected)


def test_impossible_sequence_scores_minus_infinity() -> None:
    model = _model().fit(_CORPUS)

    assert model.sequence_log_probability(["a", "a"], boundaries=False) == -np.inf


def test_scoring_with_boundaries_rejects_supplied_terminator() -> None:
    """``boundaries=True`` appends eos, so an eos in the input would count twice.

    Silently accepting it is the trap: the result looks like a score and is
    not the one the caller meant.
    """

    model = _model(smoothing=1.0).fit(_CORPUS)

    with pytest.raises(ValidationError, match=r"tokens\[1\] is the boundary symbol 'stop'"):
        model.sequence_log_probability(["a", "stop"])


def test_scoring_without_boundaries_accepts_completed_sequence() -> None:
    """A completed hypothesis carries its own eos; that is the boundaries=False case."""

    model = _model(smoothing=1.0).fit(_CORPUS)
    expected = model.sequence_log_probability(["a"])

    assert model.sequence_log_probability(["a", "stop"], boundaries=False) == pytest.approx(
        expected
    )


def test_bos_is_never_scorable() -> None:
    model = _model(smoothing=1.0).fit(_CORPUS)

    for boundaries in (True, False):
        with pytest.raises(ValidationError, match="has no emission column"):
            model.sequence_log_probability(["a", _VOCABULARY.bos], boundaries=boundaries)


# --------------------------------------------------------------------------------------
# The corpus a fit accepts
# --------------------------------------------------------------------------------------


def test_sequence_with_boundary_symbol_is_rejected() -> None:
    """The model brackets sequences itself, so a supplied one would count twice."""

    for boundary in (_VOCABULARY.bos, _VOCABULARY.eos):
        with pytest.raises(ValidationError, match="boundary symbol"):
            _model().fit([["a", boundary]])


@pytest.mark.parametrize(
    ("corpus", "message"),
    [
        ([], "at least one sequence"),
        ([["a"], []], "sequence 1 must contain at least one token"),
        (["ab"], "not a bare string"),
        ([["a", "z"]], "sequence 0: token 'z' is not in the vocabulary"),
    ],
)
def test_invalid_corpus_is_rejected(corpus: list, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _model().fit(corpus)


def test_out_of_vocabulary_token_folds_to_unknown() -> None:
    vocabulary = Vocabulary(["a", "b", "stop", "<unk>"], bos="<s>", eos="stop", unknown="<unk>")
    model = NgramLanguageModel(vocabulary).fit([["a", "zebra"], ["quokka", "b"]])

    unknown = vocabulary.index("<unk>")
    assert model.unigram_counts_[unknown] == 2.0
    assert model.transition_counts_[vocabulary.index("a"), unknown] == 1.0
    assert model.transition_counts_[vocabulary.bos_index, unknown] == 1.0
    # The unknown token stands in for the token; it never ends the sequence.
    assert np.all(model.transition_counts_[vocabulary.eos_index] == 0.0)


@pytest.mark.parametrize("token", [123, None, 1.5])
def test_non_string_token_is_rejected(token: object) -> None:
    """Unknown means "a token I have not seen", not "a value of the wrong type"."""

    vocabulary = Vocabulary(["a", "b", "stop", "<unk>"], bos="<s>", eos="stop", unknown="<unk>")

    with pytest.raises(ValidationError, match="sequence 0: token must be a string"):
        NgramLanguageModel(vocabulary).fit([[token, "a"]])


@pytest.mark.parametrize(
    ("vocabulary", "corpus"),
    [
        (_VOCABULARY, [["a", "b"], ["b"]]),
        (
            Vocabulary(["a", "b", "stop", "<unk>"], bos="<s>", eos="stop", unknown="<unk>"),
            [["a", "b"], ["zebra", "a"], ["quokka"]],
        ),
    ],
    ids=["no-unknown", "with-unknown-and-oov"],
)
def test_counting_always_stops_at_eos(vocabulary: Vocabulary, corpus: list) -> None:
    """The eos row has no corpus count, whatever the unknown-token policy is.

    This is the invariant that makes eos mean "the sequence is over": a
    completed hypothesis is never extended because there is nothing to extend
    it with. An out-of-vocabulary token must not be able to reach eos early.
    """

    model = NgramLanguageModel(vocabulary).fit(corpus)

    assert np.all(model.transition_counts_[vocabulary.eos_index] == 0.0)
    assert np.all(np.isneginf(model.transition_log_probabilities_[vocabulary.eos_index]))


def test_rejected_corpus_leaves_model_untouched() -> None:
    model = _model().fit(_CORPUS)
    counts = np.array(model.transition_counts_)

    with pytest.raises(ValidationError, match="not in the vocabulary"):
        model.fit([["a", "z"]])

    assert model.is_fitted
    assert np.array_equal(model.transition_counts_, counts)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("vocabulary", "abc", "vocabulary must be a Vocabulary"),
        ("order", 3, "order must be 1 or 2"),
        ("order", 0, "order must be positive"),
        ("order", 1.5, "order"),
        ("smoothing", float("nan"), "smoothing must be a finite number"),
        ("smoothing", -1.0, "smoothing"),
        ("smoothing", "heavy", "smoothing"),
    ],
)
def test_rebound_configuration_is_revalidated_before_refit(
    attribute: str, value: object, message: str
) -> None:
    """Construction-time validation alone would let a refit publish a lie.

    ``order = 3`` would fit a bigram while reporting order three, and a ``nan``
    weight would publish all-``nan`` distributions. The check runs before the
    corpus is touched, so the previous fit survives -- the same contract the
    other decoders in this package hold.
    """

    model = _model().fit(_CORPUS)
    counts = np.array(model.transition_counts_)
    setattr(model, attribute, value)

    with pytest.raises(ValidationError, match=message):
        model.fit(_CORPUS)

    assert model.is_fitted
    assert np.array_equal(model.transition_counts_, counts)


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("vocabulary", "abc"),
        ("order", 3),
        ("smoothing", float("nan")),
    ],
)
def test_rejected_refit_leaves_old_model_usable(attribute: str, value: object) -> None:
    """Surviving a rejected refit means inference still works, not just that counts are intact.

    Reading the writable attributes at inference time would make a fitted
    model depend on configuration it was never fitted under -- a rebound
    ``vocabulary`` would raise ``AttributeError`` out of ``log_probability``
    even though ``is_fitted`` was still true.
    """

    model = _model().fit(_CORPUS)
    expected = {
        "log_probability": model.log_probability("a", given="<s>"),
        "probability": model.probability("b", given="a"),
        "log_transition": np.array(model.log_transition("a")),
        "sequence": model.sequence_log_probability(["a", "b"]),
    }
    setattr(model, attribute, value)

    with pytest.raises(ValidationError):
        model.fit(_CORPUS)

    assert model.is_fitted
    assert model.log_probability("a", given="<s>") == expected["log_probability"]
    assert model.probability("b", given="a") == expected["probability"]
    assert np.array_equal(model.log_transition("a"), expected["log_transition"])
    assert model.sequence_log_probability(["a", "b"]) == expected["sequence"]
    assert repr(model)  # must not raise either


def test_fitted_snapshot_ignores_rebound_configuration() -> None:
    model = _model(order=2, smoothing=0.0).fit(_CORPUS)
    replacement = Vocabulary(["a", "b", "c", "stop"], bos="<s>", eos="stop")

    model.vocabulary = replacement
    model.order = 1
    model.smoothing = 0.5

    assert model.fitted_vocabulary_ is _VOCABULARY
    assert model.fitted_order_ == 2
    assert model.fitted_smoothing_ == 0.0
    assert model.transition_log_probabilities_.shape == (_VOCABULARY.n_contexts, _VOCABULARY.size)


def test_snapshot_moves_only_on_successful_fit() -> None:
    model = _model().fit(_CORPUS)
    replacement = Vocabulary(["a", "b", "c", "stop"], bos="<s>", eos="stop")
    model.vocabulary = replacement

    assert model.fitted_vocabulary_ is _VOCABULARY

    model.fit([["a", "c"]])

    assert model.fitted_vocabulary_ is replacement
    assert model.log_probability("c", given="a") > -np.inf


def test_unfitted_model_has_no_snapshot() -> None:
    model = _model()

    for name in ("fitted_vocabulary_", "fitted_order_", "fitted_smoothing_"):
        with pytest.raises(ValidationError, match="not fitted"):
            getattr(model, name)


def test_valid_rebound_configuration_takes_effect() -> None:
    model = _model(order=2).fit(_CORPUS)
    model.order = 1
    model.fit(_CORPUS)

    rows = model.transition_log_probabilities_
    assert np.array_equal(np.repeat(rows[:1], rows.shape[0], axis=0), rows)


def test_fitted_model_never_publishes_nan() -> None:
    model = _model(smoothing=0.0).fit(_CORPUS)

    assert not np.any(np.isnan(model.transition_log_probabilities_))
    assert not np.any(np.isnan(model.unigram_log_probabilities_))


def test_refit_replaces_counts() -> None:
    model = _model().fit(_CORPUS)
    model.fit([["a", "a"]])

    assert model.n_sequences_ == 1
    assert np.array_equal(model.unigram_counts_, [2.0, 0.0, 1.0])


def test_iterator_corpus_is_consumed_once() -> None:
    model = _model().fit(iter([["a", "b"], ["b", "a", "b"], ["a"]]))

    assert np.array_equal(model.transition_counts_, _TRANSITION_COUNTS)


def test_repr_reports_configuration_and_state() -> None:
    assert "unfitted" in repr(_model())
    assert "fitted" in repr(_model().fit(_CORPUS))
    assert "order=1" in repr(_model(order=1))
