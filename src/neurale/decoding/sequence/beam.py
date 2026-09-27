#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Deterministic beam search over per-step token scores."""

from __future__ import annotations

from typing import Literal, Self

import numpy as np

from neurale._validation import validate_choice, validate_integer, validate_number
from neurale.exceptions import ValidationError

from .hypotheses import Hypothesis
from .language_model import NgramLanguageModel
from .vocabulary import Vocabulary

#: A probability row is compared against one rather than required to equal it:
#: a softmax in float64 leaves a few ULPs of slack, and rejecting that would
#: reject correct callers.
_SUM_TOLERANCE = 1e-9

_Inputs = Literal["probabilities", "log_probabilities"]
_Normalization = Literal["require", "renormalize", "none"]


class BeamSearchDecoder:
    """Search for the best token sequences under a model and a language model.

    The decoder consumes one row of per-step token scores at a time and keeps
    a bounded set of partial hypotheses. It owns no model of its own: the
    per-step scores come from the caller and the sequence prior comes from the
    :class:`NgramLanguageModel` handed to the constructor.

    Everything that is usually implicit in a beam search is a stated argument
    here. Whether the input is probabilities or log probabilities, whether
    rows must already be normalized, what a zero probability means, how the
    two score sources are weighed, and whether scores are length-normalized
    are all decided by the caller, because each of them silently changes which
    sequence wins.

    Parameters
    ----------
    language_model : NgramLanguageModel
        Fitted model supplying the sequence prior. Its fitted vocabulary fixes
        the required width of every score row. The prior is snapshotted per
        sequence rather than read per step -- see :meth:`reset`.
    beam_width : int, optional
        Maximum number of active hypotheses carried between steps.
    max_completed : int, optional
        Maximum number of completed hypotheses retained. Both bounds are hard:
        neither set can grow with the number of steps decoded.
    language_weight : float, optional
        Weight on the language-model term: ``score = model_score +
        language_weight * language_score``. ``0.0`` searches on the model
        alone, and still reports what the language model would have said.
    length_normalization : float, optional
        Exponent in the length divisor ``len(tokens) ** length_normalization``
        applied before ranking. ``0.0`` is no normalization; ``1.0`` ranks by
        mean log score per token, which stops a beam from preferring short
        sequences simply because they add fewer negative terms.
    inputs : {"probabilities", "log_probabilities"}, optional
        How to read a score row.
    normalization : {"require", "renormalize", "none"}, optional
        What to do about rows that do not sum to one. ``"require"`` rejects
        them, ``"renormalize"`` divides by the row's own mass, and ``"none"``
        takes the numbers as given -- useful for scores that were never a
        distribution, at the cost of scores that are not comparable across
        steps.
    probability_floor : float or None, optional
        Lower bound applied to a probability before its logarithm. ``None``
        keeps a zero exactly impossible: the token cannot be emitted and any
        hypothesis through it is discarded rather than carried at ``-inf``. A
        floor in ``(0, 1]`` instead admits it at a stated cost.

    Raises
    ------
    neurale.exceptions.ValidationError
        If any argument is invalid.

    Notes
    -----
    Ties are broken by vocabulary order, through
    :attr:`Hypothesis.ranking_key`. Because the token sequences in a beam are
    distinct, that key is a strict total order and the result never depends on
    the order candidates were generated in.

    Decoding is incremental and exact: :meth:`update` may be called with one
    step or many, and any split of the same steps gives the same hypotheses as
    one batch call.

    One sequence is decoded under one prior. :meth:`reset` snapshots the
    language model's fitted vocabulary and transition matrix, and the search
    uses only that snapshot until the next reset. Refitting the model mid
    sequence therefore cannot mix two priors into a single score, and cannot
    change the required row width underneath a search in progress; the new
    prior takes effect at the next reset. Nothing in the language model is
    modified -- it is counted once and reused.

    A completed hypothesis is never extended. There is no continuation past
    ``eos``, no re-entry into the active beam, and no rule that ends a
    sequence other than the model emitting ``eos``.

    Examples
    --------
    >>> import numpy as np
    >>> from neurale.decoding.sequence import (
    ...     BeamSearchDecoder,
    ...     NgramLanguageModel,
    ...     Vocabulary,
    ... )
    >>> vocabulary = Vocabulary(["yes", "no", "stop"], bos="<s>", eos="stop")
    >>> model = NgramLanguageModel(vocabulary, smoothing=1.0)
    >>> model = model.fit([["yes", "no"], ["yes", "yes"], ["no", "yes"]])
    >>> decoder = BeamSearchDecoder(model, beam_width=4)
    >>> steps = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])
    >>> best = decoder.decode(steps)[0]
    >>> best.tokens, best.is_complete
    (('yes', 'no', 'stop'), True)
    """

    def __init__(
        self,
        language_model: NgramLanguageModel,
        *,
        beam_width: int = 8,
        max_completed: int = 8,
        language_weight: float = 1.0,
        length_normalization: float = 0.0,
        inputs: _Inputs = "probabilities",
        normalization: _Normalization = "require",
        probability_floor: float | None = None,
    ) -> None:
        self.language_model = _validate_language_model(language_model)
        self.beam_width = validate_integer(beam_width, "beam_width", minimum=1)
        self.max_completed = validate_integer(max_completed, "max_completed", minimum=1)
        self.language_weight = float(
            validate_number(language_weight, "language_weight", kind="real", minimum=0, coerce=True)
        )
        self.length_normalization = float(
            validate_number(
                length_normalization,
                "length_normalization",
                kind="real",
                minimum=0,
                coerce=True,
            )
        )
        self.inputs = validate_choice(inputs, ("probabilities", "log_probabilities"), "inputs")
        self.normalization = validate_choice(
            normalization, ("require", "renormalize", "none"), "normalization"
        )
        self.probability_floor = _validate_floor(probability_floor)
        self.reset()

    # ------------------------------------------------------------------ state

    @property
    def vocabulary(self) -> Vocabulary:
        """Token inventory the current sequence is being searched over.

        This is the vocabulary of the prior snapshotted at the last
        :meth:`reset`, so it is stable for as long as the sequence lasts even
        if the language model is refitted meanwhile.
        """

        return self._vocabulary

    @property
    def transitions(self) -> np.ndarray:
        """The ``(n_contexts, size)`` prior in force for the current sequence.

        Read-only, and identical from the first step of a sequence to the
        last. Comparing it against
        :attr:`NgramLanguageModel.transition_log_probabilities_` is how a
        caller sees whether the model has moved on since this sequence began.
        """

        return self._transitions

    def reset(self) -> Self:
        """Start a new sequence and fix the prior it will be decoded under.

        Clears the active beam, the completed set, and the step count, then
        snapshots the language model's fitted vocabulary and transition matrix.
        The search reads only that snapshot afterwards, so a sequence is always
        decoded under exactly one prior.

        This is what makes an adaptive model safe to use online: refit between
        sequences and the next ``reset`` picks the new prior up, but a refit
        *during* a sequence cannot silently score its first tokens under one
        model and its last under another. It also means a reported
        :attr:`Hypothesis.language_score` always comes from a single snapshot
        and is reproducible from it.

        Returns
        -------
        Self
            The decoder, so a reset can be chained.

        Raises
        ------
        neurale.exceptions.ValidationError
            If :attr:`language_model` is no longer a fitted
            :class:`NgramLanguageModel`. It is a writable attribute, so it is
            re-checked here rather than trusted from construction.

        Notes
        -----
        Nothing in the language model is touched: it is counted once and
        reused, and the snapshot is a reference to arrays it already publishes
        as read-only.
        """

        model = _validate_language_model(self.language_model)
        self._vocabulary = model.fitted_vocabulary_
        self._transitions = model.transition_log_probabilities_
        self._active: tuple[Hypothesis, ...] = (_empty_hypothesis(),)
        self._completed: tuple[Hypothesis, ...] = ()
        self._n_steps = 0
        return self

    @property
    def n_steps(self) -> int:
        """Number of score rows consumed since the last :meth:`reset`."""

        return self._n_steps

    @property
    def active(self) -> tuple[Hypothesis, ...]:
        """Incomplete hypotheses, best first, at most :attr:`beam_width` of them."""

        return self._active

    @property
    def completed(self) -> tuple[Hypothesis, ...]:
        """Hypotheses that emitted ``eos``, best first, at most :attr:`max_completed`."""

        return self._completed

    @property
    def hypotheses(self) -> tuple[Hypothesis, ...]:
        """Active and completed hypotheses in one ranking, best first.

        Both sets are bounded, so this one is too. Note that a shorter
        sequence sums fewer negative log terms: at ``length_normalization=0``
        this ranking prefers short hypotheses, which is a property of the
        score and not something the decoder corrects behind the caller.
        """

        return tuple(sorted(self._active + self._completed, key=_ranking_key))

    @property
    def best(self) -> Hypothesis:
        """Highest ranked hypothesis.

        Raises
        ------
        neurale.exceptions.ValidationError
            If there is none, which happens when every candidate has been
            ruled out by a zero probability.
        """

        ranked = self.hypotheses
        if not ranked:
            raise ValidationError(
                "no hypothesis survives: every candidate was ruled out by a zero "
                "probability. Configure probability_floor= or smoothing= to admit them."
            )
        return ranked[0]

    # --------------------------------------------------------------- decoding

    def decode(self, scores: np.ndarray) -> tuple[Hypothesis, ...]:
        """Decode a whole sequence of steps and return the ranked hypotheses.

        Equivalent to :meth:`reset` followed by :meth:`update` and reading
        :attr:`hypotheses`.
        """

        self.reset()
        self.update(scores)
        return self.hypotheses

    def update(self, scores: np.ndarray) -> Self:
        """Advance the search by one or more steps.

        Parameters
        ----------
        scores : numpy.ndarray
            ``(size,)`` for a single step or ``(n_steps, size)`` for several,
            where ``size`` is the vocabulary's. The width is checked against
            the vocabulary, never adapted to it.

        Returns
        -------
        Self
            The decoder, so steps can be chained.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the array's shape, dtype, or values are invalid for the
            configured input kind and normalization.

        Notes
        -----
        ``n_steps`` may be zero: a ``(0, size)`` batch advances nothing and
        leaves every hypothesis and :attr:`n_steps` unchanged. This is
        deliberate rather than an oversight -- a stream delivers whatever
        arrived in an interval, sometimes nothing, and an empty chunk must be
        the identity if chunked decoding is to agree with the equivalent whole
        sequence. The width is still checked, so an empty batch of the wrong
        width is still an error.
        """

        steps = self._prepare(scores)
        for row in steps:
            self._advance(row)
            self._n_steps += 1
        return self

    def _advance(self, model_log: np.ndarray) -> None:
        weight = self.language_weight
        transitions = self._transitions
        vocabulary = self._vocabulary
        bos_idx = vocabulary.bos_index
        eos_idx = vocabulary.eos_index
        tokens = vocabulary.tokens
        exponent = self.length_normalization

        candidates: list[Hypothesis] = []
        for parent in self._active:
            context = bos_idx if not parent.indices else parent.indices[-1]
            language_log = transitions[context]
            combined = model_log if weight == 0.0 else model_log + weight * language_log

            # A step that is impossible under the combined score removes the
            # hypothesis rather than carrying it at -inf, where it would sort
            # ahead of nothing and still occupy a beam slot.
            for i in np.flatnonzero(~np.isneginf(combined)).tolist():
                score = parent.score + float(combined[i])
                length = len(parent.indices) + 1
                candidates.append(
                    Hypothesis(
                        tokens=(*parent.tokens, tokens[i]),
                        indices=(*parent.indices, i),
                        model_score=parent.model_score + float(model_log[i]),
                        language_score=parent.language_score + float(language_log[i]),
                        score=score,
                        normalized_score=score / float(length**exponent),
                        is_complete=i == eos_idx,
                    )
                )

        active = [candidate for candidate in candidates if not candidate.is_complete]
        completed = [candidate for candidate in candidates if candidate.is_complete]

        self._active = tuple(sorted(active, key=_ranking_key)[: self.beam_width])
        self._completed = tuple(
            sorted([*self._completed, *completed], key=_ranking_key)[: self.max_completed]
        )

    # ------------------------------------------------------------ input rules

    def _prepare(self, scores: object) -> np.ndarray:
        size = self._vocabulary.size
        if not isinstance(scores, np.ndarray):
            raise ValidationError("scores must be a numpy.ndarray.")
        if scores.ndim not in (1, 2):
            raise ValidationError(
                f"scores must be a 1D step or a 2D sequence of steps; got {scores.ndim} dimensions."
            )
        steps = scores.reshape(1, -1) if scores.ndim == 1 else scores
        if steps.shape[1] != size:
            raise ValidationError(
                f"scores must have one column per vocabulary token, {size}; got "
                f"{steps.shape[1]}. The vocabulary is fixed and is never padded or "
                "truncated to fit an input."
            )
        if not np.issubdtype(steps.dtype, np.floating):
            raise ValidationError("scores must contain floating-point values.")
        if np.any(np.isnan(steps)):
            raise ValidationError("scores must not contain nan.")

        values = np.array(steps, dtype=np.float64, copy=True)
        if self.inputs == "probabilities":
            return self._prepare_probabilities(values)
        return self._prepare_log_probabilities(values)

    def _prepare_probabilities(self, values: np.ndarray) -> np.ndarray:
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValidationError("probabilities must lie in [0, 1].")
        totals = values.sum(axis=1)
        if self.normalization == "require":
            if not np.allclose(totals, 1.0, rtol=0.0, atol=_SUM_TOLERANCE):
                raise ValidationError(
                    "probability rows must sum to one. Pass normalization='renormalize' "
                    "to divide by the row mass, or 'none' to use the values as given."
                )
        elif self.normalization == "renormalize":
            if np.any(totals <= 0.0):
                raise ValidationError("a probability row with zero mass cannot be normalized.")
            values = values / totals[:, None]

        if self.probability_floor is not None:
            values = np.maximum(values, self.probability_floor)
        with np.errstate(divide="ignore"):
            return np.log(values)

    def _prepare_log_probabilities(self, values: np.ndarray) -> np.ndarray:
        if np.any(np.isposinf(values)):
            raise ValidationError("log probabilities must not contain positive infinity.")
        if self.normalization == "require":
            totals = _logsumexp(values)
            if not np.allclose(totals, 0.0, rtol=0.0, atol=_SUM_TOLERANCE):
                raise ValidationError(
                    "log-probability rows must sum to one in probability space. Pass "
                    "normalization='renormalize' to subtract the row's log mass, or "
                    "'none' to use the values as given."
                )
        elif self.normalization == "renormalize":
            totals = _logsumexp(values)
            if np.any(np.isneginf(totals)):
                raise ValidationError("a log-probability row with zero mass cannot be normalized.")
            values = values - totals[:, None]

        if self.probability_floor is not None:
            values = np.maximum(values, float(np.log(self.probability_floor)))
        return values

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(beam_width={self.beam_width}, "
            f"max_completed={self.max_completed}, language_weight={self.language_weight}, "
            f"n_steps={self._n_steps})"
        )


def _empty_hypothesis() -> Hypothesis:
    return Hypothesis(
        tokens=(),
        indices=(),
        model_score=0.0,
        language_score=0.0,
        score=0.0,
        normalized_score=0.0,
        is_complete=False,
    )


def _ranking_key(hypothesis: Hypothesis) -> tuple[float, tuple[int, ...]]:
    return hypothesis.ranking_key


def _logsumexp(values: np.ndarray) -> np.ndarray:
    """Row-wise log of the summed exponentials, stable for very negative rows."""

    largest = values.max(axis=1, keepdims=True)
    finite = np.where(np.isneginf(largest), 0.0, largest)
    with np.errstate(invalid="ignore"):
        shifted = np.exp(values - finite)
    with np.errstate(divide="ignore"):
        return (finite + np.log(shifted.sum(axis=1, keepdims=True)))[:, 0]


def _validate_language_model(value: object) -> NgramLanguageModel:
    if not isinstance(value, NgramLanguageModel):
        raise ValidationError("language_model must be an NgramLanguageModel.")
    if not value.is_fitted:
        raise ValidationError(
            "language_model must be fitted before it can supply a sequence prior."
        )
    return value


def _validate_floor(value: object) -> float | None:
    if value is None:
        return None
    return float(
        validate_number(
            value,
            "probability_floor",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            maximum=1,
            coerce=True,
        )
    )


__all__ = ["BeamSearchDecoder"]
