#!/usr/bin/env python3

from __future__ import annotations

import itertools

import numpy as np
import pytest

from neurale.decoding.sequence import (
    BeamSearchDecoder,
    Hypothesis,
    NgramLanguageModel,
    Vocabulary,
)
from neurale.exceptions import ValidationError

_VOCABULARY = Vocabulary(["a", "b", "stop"], bos="<s>", eos="stop")
_CORPUS = [["a", "b"], ["b", "a", "b"], ["a"], ["b", "b", "a"]]

# Three steps over three tokens: 39 distinct paths, few enough to enumerate
# exhaustively and compare a bounded search against.
_STEPS = np.array(
    [
        [0.60, 0.30, 0.10],
        [0.25, 0.65, 0.10],
        [0.15, 0.15, 0.70],
    ]
)


def _model(**overrides: object) -> NgramLanguageModel:
    settings: dict[str, object] = {"vocabulary": _VOCABULARY, "smoothing": 1.0}
    settings.update(overrides)
    return NgramLanguageModel(**settings).fit(_CORPUS)


def _decoder(model: NgramLanguageModel | None = None, **overrides: object) -> BeamSearchDecoder:
    settings: dict[str, object] = {"beam_width": 4, "max_completed": 4}
    settings.update(overrides)
    return BeamSearchDecoder(model if model is not None else _model(), **settings)


# --------------------------------------------------------------------------------------
# An independent enumeration of the whole search tree
# --------------------------------------------------------------------------------------


def _brute_force(
    model: NgramLanguageModel,
    log_steps: np.ndarray,
    *,
    language_weight: float,
    length_normalization: float,
) -> list[tuple[tuple[str, ...], float, bool]]:
    """Score every path the search could take, without a beam or any pruning.

    Written from the definition rather than from the implementation: expand
    the full tree, sum the two log-score sources along each path, and rank the
    survivors. Anything a bounded search returns has to appear here.
    """

    vocabulary = model.vocabulary
    transitions = model.transition_log_probabilities_
    finished: list[tuple[tuple[int, ...], float, float]] = []
    frontier: list[tuple[tuple[int, ...], float, float]] = [((), 0.0, 0.0)]

    for row in log_steps:
        extended: list[tuple[tuple[int, ...], float, float]] = []
        for indices, model_score, language_score in frontier:
            context = vocabulary.bos_index if not indices else indices[-1]
            for token in range(vocabulary.size):
                step_model = float(row[token])
                step_language = float(transitions[context, token])
                step_combined = step_model + language_weight * step_language
                if np.isneginf(step_combined):
                    continue
                record = (
                    (*indices, token),
                    model_score + step_model,
                    language_score + step_language,
                )
                if token == vocabulary.eos_index:
                    finished.append(record)
                else:
                    extended.append(record)
        frontier = extended

    scored = []
    for indices, model_score, language_score in [*frontier, *finished]:
        combined = model_score + language_weight * language_score
        normalized = combined / float(len(indices) ** length_normalization)
        scored.append((normalized, indices, combined, indices in {r[0] for r in finished}))
    scored.sort(key=lambda record: (-record[0], record[1]))
    return [
        (vocabulary.decode(indices), combined, complete)
        for _, indices, combined, complete in scored
    ]


# --------------------------------------------------------------------------------------
# Constructor
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"beam_width": 0}, "beam_width must be positive"),
        ({"beam_width": 1.5}, "beam_width"),
        ({"max_completed": 0}, "max_completed must be positive"),
        ({"language_weight": -1.0}, "language_weight"),
        ({"length_normalization": -0.5}, "length_normalization"),
        ({"inputs": "logits"}, "inputs must be one of"),
        ({"normalization": "always"}, "normalization must be one of"),
        ({"probability_floor": 0.0}, "probability_floor"),
        ({"probability_floor": 2.0}, "probability_floor"),
    ],
)
def test_invalid_arguments_are_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _decoder(**overrides)


def test_language_model_is_required() -> None:
    with pytest.raises(ValidationError, match="must be an NgramLanguageModel"):
        BeamSearchDecoder("bigram")


def test_unfitted_language_model_is_rejected() -> None:
    with pytest.raises(ValidationError, match="must be fitted"):
        BeamSearchDecoder(NgramLanguageModel(_VOCABULARY))


def test_fresh_decoder_starts_empty_sequence() -> None:
    decoder = _decoder()

    assert decoder.n_steps == 0
    assert decoder.completed == ()
    assert len(decoder.active) == 1
    assert decoder.active[0].tokens == ()
    assert decoder.active[0].score == 0.0
    assert decoder.vocabulary == _VOCABULARY


# --------------------------------------------------------------------------------------
# Exhaustive comparison on a small vocabulary
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("language_weight", [0.0, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("length_normalization", [0.0, 1.0])
def test_unbounded_beam_matches_full_enumeration(
    language_weight: float,
    length_normalization: float,
) -> None:
    model = _model()
    decoder = _decoder(
        model,
        beam_width=64,
        max_completed=64,
        language_weight=language_weight,
        length_normalization=length_normalization,
    )

    decoded = decoder.decode(_STEPS)
    expected = _brute_force(
        model,
        np.log(_STEPS),
        language_weight=language_weight,
        length_normalization=length_normalization,
    )

    assert [(item.tokens, item.is_complete) for item in decoded] == [
        (tokens, complete) for tokens, _, complete in expected
    ]
    for item, (_, score, _) in zip(decoded, expected, strict=True):
        assert item.score == pytest.approx(score)


@pytest.mark.parametrize("beam_width", [1, 2, 3, 5])
def test_bounded_beam_matches_enumeration_on_best(beam_width: int) -> None:
    """A narrow beam may miss paths, but never invents one or misscores one."""

    model = _model()
    decoder = _decoder(model, beam_width=beam_width, max_completed=8)
    decoded = decoder.decode(_STEPS)
    expected = dict(
        (tokens, score)
        for tokens, score, _ in _brute_force(
            model, np.log(_STEPS), language_weight=1.0, length_normalization=0.0
        )
    )

    for item in decoded:
        assert item.tokens in expected
        assert item.score == pytest.approx(expected[item.tokens])


def test_wider_beam_never_scores_worse() -> None:
    model = _model()
    scores = [_decoder(model, beam_width=width).decode(_STEPS)[0].score for width in (1, 2, 4, 16)]

    assert scores == sorted(scores)


def test_score_sources_sum_to_combined_score() -> None:
    decoder = _decoder(language_weight=0.75)

    for item in decoder.decode(_STEPS):
        assert item.score == pytest.approx(item.model_score + 0.75 * item.language_score)


def test_model_score_sums_input_log_probabilities() -> None:
    decoder = _decoder()
    log_steps = np.log(_STEPS)

    for item in decoder.decode(_STEPS):
        expected = sum(log_steps[step, idx] for step, idx in enumerate(item.indices))
        assert item.model_score == pytest.approx(expected)


def test_language_score_is_model_sequence_score() -> None:
    model = _model()
    decoder = _decoder(model)

    for item in decoder.decode(_STEPS):
        expected = model.sequence_log_probability(item.tokens, boundaries=False)
        assert item.language_score == pytest.approx(expected)


# --------------------------------------------------------------------------------------
# Determinism and ties
# --------------------------------------------------------------------------------------


def test_ties_break_by_vocabulary_order() -> None:
    """Every candidate scores identically, so only the declared order decides."""

    uniform = np.full((2, 3), 1 / 3)
    decoder = _decoder(beam_width=2, language_weight=0.0)
    decoder.update(uniform)

    assert [item.tokens for item in decoder.active] == [("a", "a"), ("a", "b")]
    assert [item.tokens for item in decoder.completed] == [
        ("stop",),
        ("a", "stop"),
        ("b", "stop"),
    ]


def test_reordering_vocabulary_reorders_tie() -> None:
    reordered = Vocabulary(["b", "a", "stop"], bos="<s>", eos="stop")
    model = NgramLanguageModel(reordered, smoothing=1.0).fit(_CORPUS)
    decoder = _decoder(model, beam_width=2, language_weight=0.0)
    decoder.update(np.full((2, 3), 1 / 3))

    assert [item.tokens for item in decoder.active] == [("b", "b"), ("b", "a")]


def test_decoding_is_repeatable() -> None:
    model = _model()

    first = _decoder(model).decode(_STEPS)
    second = _decoder(model).decode(_STEPS)

    assert first == second


def test_ranking_key_is_strict_total_order() -> None:
    decoder = _decoder(beam_width=16, max_completed=16)
    keys = [item.ranking_key for item in decoder.decode(_STEPS)]

    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


# --------------------------------------------------------------------------------------
# Completion
# --------------------------------------------------------------------------------------


def test_eos_completes_hypothesis() -> None:
    decoder = _decoder()
    decoder.update(np.array([[0.0, 0.0, 1.0]]))

    assert len(decoder.completed) == 1
    completed = decoder.completed[0]
    assert completed.is_complete
    assert completed.tokens == ("stop",)
    assert completed.n_tokens == 1


def test_completed_hypothesis_is_never_extended() -> None:
    decoder = _decoder()
    decoder.update(np.array([[0.0, 0.0, 1.0]]))
    finished = decoder.completed

    decoder.update(np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 0.0]]))

    assert decoder.completed == finished
    assert all(not item.tokens or item.tokens[-1] != "stop" for item in decoder.active)


def test_terminator_is_part_of_reported_sequence() -> None:
    """Hiding it would leave the score of a completed hypothesis unaccountable."""

    decoder = _decoder()
    completed = decoder.decode(_STEPS)

    ended = [item for item in completed if item.is_complete]
    assert ended
    assert all(item.tokens[-1] == "stop" for item in ended)
    assert all("stop" not in item.tokens[:-1] for item in ended)


def test_completed_hypotheses_are_bounded() -> None:
    decoder = _decoder(beam_width=8, max_completed=2)
    decoder.update(np.full((4, 3), 1 / 3))

    assert len(decoder.completed) == 2
    assert len(decoder.active) <= 8


def test_active_beam_is_bounded() -> None:
    decoder = _decoder(beam_width=3, max_completed=8)

    for _ in range(6):
        decoder.update(np.full((1, 3), 1 / 3))
        assert len(decoder.active) <= 3

    assert len(decoder.hypotheses) <= 3 + 8


# --------------------------------------------------------------------------------------
# Zero probabilities
# --------------------------------------------------------------------------------------


def test_zero_probability_token_is_never_emitted() -> None:
    decoder = _decoder()
    decoder.update(np.array([[0.5, 0.5, 0.0]]))

    assert decoder.completed == ()
    assert {item.tokens for item in decoder.active} == {("a",), ("b",)}


def test_probability_floor_admits_zero_at_stated_cost() -> None:
    decoder = _decoder(probability_floor=1e-6)
    decoder.update(np.array([[0.5, 0.5, 0.0]]))

    assert [item.tokens for item in decoder.completed] == [("stop",)]
    assert decoder.completed[0].model_score == pytest.approx(np.log(1e-6))


def test_impossible_language_step_removes_hypothesis() -> None:
    # Unsmoothed, so a bigram the corpus never saw is exactly impossible.
    decoder = _decoder(_model(smoothing=0.0))
    decoder.update(np.full((1, 3), 1 / 3))

    # The corpus starts only with "a" or "b", never with the terminator.
    assert decoder.completed == ()
    assert {item.tokens for item in decoder.active} == {("a",), ("b",)}


def test_search_without_survivor_is_reported() -> None:
    model = NgramLanguageModel(_VOCABULARY, smoothing=0.0).fit([["a"]])
    decoder = _decoder(model)

    # "<s> a stop" is the only sequence the corpus supports, and the model
    # rules out its only continuation.
    decoder.update(np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]))

    assert decoder.active == ()
    assert decoder.completed == ()
    assert decoder.hypotheses == ()
    with pytest.raises(ValidationError, match="no hypothesis survives"):
        _ = decoder.best


def test_zero_language_weight_still_reports_score() -> None:
    decoder = _decoder(_model(smoothing=0.0), language_weight=0.0)
    decoder.update(np.full((1, 3), 1 / 3))

    terminator = next(item for item in decoder.completed if item.tokens == ("stop",))
    assert terminator.language_score == -np.inf
    assert np.isfinite(terminator.score)


# --------------------------------------------------------------------------------------
# Input handling
# --------------------------------------------------------------------------------------


def test_probabilities_and_log_probabilities_agree() -> None:
    model = _model()
    from_probabilities = _decoder(model).decode(_STEPS)
    from_logs = _decoder(model, inputs="log_probabilities").decode(np.log(_STEPS))

    assert [item.tokens for item in from_probabilities] == [item.tokens for item in from_logs]
    for left, right in zip(from_probabilities, from_logs, strict=True):
        assert left.score == pytest.approx(right.score)


def test_single_step_may_be_one_dimensional() -> None:
    model = _model()
    stepwise = _decoder(model)
    for row in _STEPS:
        stepwise.update(row)

    assert stepwise.hypotheses == _decoder(model).decode(_STEPS)


@pytest.mark.parametrize(
    ("scores", "message"),
    [
        (np.full((2, 4), 0.25), "one column per vocabulary token, 3"),
        (np.full((2, 2), 0.5), "one column per vocabulary token, 3"),
        (np.full((2, 2, 3), 1 / 3), "1D step or a 2D sequence"),
        (np.array([[1, 0, 0]]), "floating-point"),
        (np.array([[np.nan, 0.5, 0.5]]), "must not contain nan"),
        (np.array([[-0.5, 1.0, 0.5]]), r"must lie in \[0, 1\]"),
        (np.array([[1.5, 0.0, 0.0]]), r"must lie in \[0, 1\]"),
        (np.array([[0.5, 0.2, 0.2]]), "must sum to one"),
    ],
)
def test_invalid_score_arrays_are_rejected(scores: np.ndarray, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _decoder().update(scores)


def test_non_array_input_is_rejected() -> None:
    with pytest.raises(ValidationError, match=r"must be a numpy\.ndarray"):
        _decoder().update([[0.3, 0.3, 0.4]])


def test_renormalization_divides_by_row_mass() -> None:
    model = _model()
    raw = np.array([[2.0, 1.0, 1.0]]) / 8.0
    renormalized = _decoder(model, normalization="renormalize").decode(raw)
    exact = _decoder(model).decode(raw * 2.0)

    assert [item.tokens for item in renormalized] == [item.tokens for item in exact]
    for left, right in zip(renormalized, exact, strict=True):
        assert left.score == pytest.approx(right.score)


def test_renormalization_rejects_massless_row() -> None:
    with pytest.raises(ValidationError, match="zero mass"):
        _decoder(normalization="renormalize").update(np.zeros((1, 3)))


def test_unnormalized_scores_are_taken_as_given() -> None:
    decoder = _decoder(normalization="none")
    decoder.update(np.array([[0.5, 0.2, 0.2]]))

    assert decoder.n_steps == 1


def test_log_probability_rows_check_normalization() -> None:
    with pytest.raises(ValidationError, match="must sum to one in probability space"):
        _decoder(inputs="log_probabilities").update(np.log(np.array([[0.5, 0.2, 0.2]])))


def test_positive_infinity_is_never_log_probability() -> None:
    with pytest.raises(ValidationError, match="positive infinity"):
        _decoder(inputs="log_probabilities", normalization="none").update(
            np.array([[np.inf, -1.0, -1.0]])
        )


def test_zero_mass_log_row_cannot_be_renormalized() -> None:
    with pytest.raises(ValidationError, match="zero mass"):
        _decoder(inputs="log_probabilities", normalization="renormalize").update(
            np.full((1, 3), -np.inf)
        )


def test_input_array_is_not_modified() -> None:
    scores = _STEPS.copy()
    _decoder().decode(scores)

    assert np.array_equal(scores, _STEPS)


# --------------------------------------------------------------------------------------
# Chunked decoding
# --------------------------------------------------------------------------------------


_LONG_STEPS = np.array(
    [
        [0.50, 0.30, 0.20],
        [0.20, 0.70, 0.10],
        [0.60, 0.25, 0.15],
        [0.10, 0.30, 0.60],
        [0.45, 0.45, 0.10],
        [0.20, 0.20, 0.60],
    ]
)


def _splits(n_steps: int) -> list[list[int]]:
    """Every way of cutting a run of steps into contiguous chunks."""

    boundaries = range(1, n_steps)
    found = []
    for size in range(len(boundaries) + 1):
        for cuts in itertools.combinations(boundaries, size):
            edges = [0, *cuts, n_steps]
            found.append([edges[i + 1] - edges[i] for i in range(len(edges) - 1)])
    return found


@pytest.mark.parametrize("chunks", _splits(_LONG_STEPS.shape[0]))
def test_chunked_decoding_equals_batch_sequence(chunks: list[int]) -> None:
    model = _model()
    batch = _decoder(model, beam_width=4, max_completed=4).decode(_LONG_STEPS)

    stepwise = _decoder(model, beam_width=4, max_completed=4)
    offset = 0
    for size in chunks:
        stepwise.update(_LONG_STEPS[offset : offset + size])
        offset += size

    assert stepwise.n_steps == _LONG_STEPS.shape[0]
    assert stepwise.hypotheses == batch


def test_empty_batch_is_identity() -> None:
    """A stream delivers whatever arrived, sometimes nothing.

    An empty chunk has to be a no-op for chunked decoding to agree with the
    equivalent whole sequence; this freezes it as documented behaviour rather
    than an accident of the loop.
    """

    model = _model()
    decoder = _decoder(model).update(_LONG_STEPS[:3])
    before = (decoder.n_steps, decoder.hypotheses)

    decoder.update(np.empty((0, model.vocabulary.size)))

    assert (decoder.n_steps, decoder.hypotheses) == before
    assert decoder.update(_LONG_STEPS[3:]).hypotheses == _decoder(model).decode(_LONG_STEPS)


def test_empty_batch_of_wrong_width_is_rejected() -> None:
    """Emptiness does not excuse the width check; the vocabulary is fixed."""

    with pytest.raises(ValidationError, match="one column per vocabulary token"):
        _decoder(_model()).update(np.empty((0, 5)))


def test_update_returns_decoder_for_chaining() -> None:
    model = _model()
    chained = _decoder(model).update(_LONG_STEPS[:2]).update(_LONG_STEPS[2:]).hypotheses

    assert chained == _decoder(model).decode(_LONG_STEPS)


# --------------------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------------------


def test_reset_starts_new_sequence() -> None:
    decoder = _decoder()
    decoder.update(_STEPS)

    assert decoder.reset() is decoder
    assert decoder.n_steps == 0
    assert decoder.completed == ()
    assert [item.tokens for item in decoder.active] == [()]


def test_reset_does_not_rebuild_language_model() -> None:
    model = _model()
    decoder = _decoder(model)
    counts = np.array(model.transition_counts_)

    decoder.update(_STEPS)
    decoder.reset()

    assert decoder.language_model is model
    assert model.n_sequences_ == len(_CORPUS)
    assert np.array_equal(model.transition_counts_, counts)


def test_repeated_decoding_gives_same_result() -> None:
    decoder = _decoder()

    assert decoder.decode(_STEPS) == decoder.decode(_STEPS)


def test_new_sequence_carries_nothing_over() -> None:
    model = _model()
    decoder = _decoder(model)
    decoder.decode(_LONG_STEPS)

    assert decoder.decode(_STEPS) == _decoder(model).decode(_STEPS)


# --------------------------------------------------------------------------------------
# One sequence, one prior
# --------------------------------------------------------------------------------------


_OTHER_CORPUS = [["b", "b"], ["b"], ["b", "a"]]


def test_refit_during_search_keeps_prior() -> None:
    """An adaptive model must not rewrite the prior underneath a sequence in flight.

    Reading the model per step would score the first tokens under one
    distribution and the rest under another, and the result would record
    neither.
    """

    model = _model()
    decoder = _decoder(model)
    decoder.update(_LONG_STEPS[:3])
    snapshot = np.array(decoder.transitions)

    model.fit(_OTHER_CORPUS)
    decoder.update(_LONG_STEPS[3:])

    assert np.array_equal(decoder.transitions, snapshot)
    assert not np.array_equal(snapshot, model.transition_log_probabilities_)


def test_refit_during_search_keeps_result() -> None:
    reference = _decoder(_model()).decode(_LONG_STEPS)

    model = _model()
    decoder = _decoder(model)
    decoder.update(_LONG_STEPS[:3])
    model.fit(_OTHER_CORPUS)
    decoder.update(_LONG_STEPS[3:])

    assert decoder.hypotheses == reference


def test_language_score_comes_from_one_snapshot() -> None:
    """Every hypothesis is reproducible from the single prior in force."""

    model = _model()
    decoder = _decoder(model)
    decoder.update(_LONG_STEPS[:2])
    model.fit(_OTHER_CORPUS)
    decoder.update(_LONG_STEPS[2:])

    transitions = decoder.transitions
    for item in decoder.hypotheses:
        context = decoder.vocabulary.bos_index
        total = 0.0
        for i in item.indices:
            total += float(transitions[context, i])
            context = i
        assert total == pytest.approx(item.language_score)


def test_vocabulary_changing_refit_leaks_no_numpy_error() -> None:
    """A wider prior must not reach a search in progress at all.

    Were the matrix read per step, the next update would broadcast a wider row
    against the beam and raise a raw ``ValueError`` from NumPy out of a public
    method.
    """

    model = _model()
    decoder = _decoder(model)
    decoder.update(_LONG_STEPS[:2])

    wider = Vocabulary(["a", "b", "c", "stop"], bos="<s>", eos="stop")
    model.vocabulary = wider
    model.fit([["a", "c"], ["c", "b"]])

    with pytest.raises(ValidationError, match="one column per vocabulary token, 3; got 4"):
        decoder.update(np.full((1, 4), 0.25))

    decoder.update(_LONG_STEPS[2:])
    assert decoder.vocabulary is _VOCABULARY
    assert decoder.n_steps == _LONG_STEPS.shape[0]


def test_reset_adopts_current_prior() -> None:
    """Between sequences is exactly when an adaptive model is allowed to take effect."""

    model = _model()
    decoder = _decoder(model)
    decoder.decode(_LONG_STEPS)

    model.fit(_OTHER_CORPUS)
    decoder.reset()

    assert np.array_equal(decoder.transitions, model.transition_log_probabilities_)
    assert decoder.decode(_LONG_STEPS) == _decoder(model).decode(_LONG_STEPS)


def test_reset_adopts_new_vocabulary() -> None:
    model = _model()
    decoder = _decoder(model)

    wider = Vocabulary(["a", "b", "c", "stop"], bos="<s>", eos="stop")
    model.vocabulary = wider
    model.fit([["a", "c"], ["c", "b"]])
    decoder.reset()

    assert decoder.vocabulary is wider
    assert decoder.update(np.full((1, 4), 0.25)).n_steps == 1


def test_snapshot_is_read_only() -> None:
    with pytest.raises(ValueError, match="read-only"):
        _decoder().transitions[0, 0] = 0.0


def test_rebound_language_model_is_revalidated_at_reset() -> None:
    """``language_model`` is writable, so reset re-checks it instead of trusting construction."""

    decoder = _decoder()
    decoder.language_model = "abc"

    with pytest.raises(ValidationError, match="language_model must be an NgramLanguageModel"):
        decoder.reset()


def test_unfitted_language_model_is_rejected_at_reset() -> None:
    decoder = _decoder()
    decoder.language_model = NgramLanguageModel(_VOCABULARY)

    with pytest.raises(ValidationError, match="must be fitted"):
        decoder.reset()


# --------------------------------------------------------------------------------------
# Length normalization
# --------------------------------------------------------------------------------------


def test_unnormalized_score_is_raw_combined_score() -> None:
    for item in _decoder().decode(_STEPS):
        assert item.normalized_score == item.score


def test_normalization_divides_by_token_count() -> None:
    for item in _decoder(length_normalization=1.0).decode(_STEPS):
        assert item.normalized_score == pytest.approx(item.score / item.n_tokens)


def test_normalization_stops_short_sequence_winning() -> None:
    model = _model()
    raw = _decoder(model, length_normalization=0.0).decode(_LONG_STEPS)[0]
    normalized = _decoder(model, length_normalization=1.0).decode(_LONG_STEPS)[0]

    assert normalized.n_tokens > raw.n_tokens


# --------------------------------------------------------------------------------------
# The typed hypothesis
# --------------------------------------------------------------------------------------


def test_hypothesis_is_frozen() -> None:
    hypothesis = _decoder().decode(_STEPS)[0]

    with pytest.raises(AttributeError):
        hypothesis.score = 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"tokens": ["a"]}, "tokens must be a tuple of strings"),
        ({"tokens": (1,)}, "tokens must be a tuple of strings"),
        ({"indices": [0]}, "indices must be a tuple of integers"),
        ({"indices": (True,)}, "indices must be a tuple of integers"),
        ({"indices": (0, 1)}, "same length"),
        ({"score": float("nan")}, "score must be a float and not nan"),
        ({"model_score": 1}, "model_score must be a float"),
        ({"is_complete": 1}, "is_complete must be a bool"),
    ],
)
def test_inconsistent_hypothesis_is_rejected(overrides: dict, message: str) -> None:
    settings: dict[str, object] = {
        "tokens": ("a",),
        "indices": (0,),
        "model_score": -1.0,
        "language_score": -2.0,
        "score": -3.0,
        "normalized_score": -3.0,
        "is_complete": False,
    }
    settings.update(overrides)
    with pytest.raises(ValidationError, match=message):
        Hypothesis(**settings)


def test_repr_names_tokens_and_state() -> None:
    decoder = _decoder()
    decoder.update(np.array([[0.25, 0.25, 0.5]]))

    assert "complete" in repr(decoder.completed[0])
    assert "'stop'" in repr(decoder.completed[0])
    assert "active" in repr(decoder.active[0])
    assert "beam_width=4" in repr(decoder)
