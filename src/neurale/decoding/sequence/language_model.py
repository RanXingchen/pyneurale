#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Unigram and bigram language models over a fixed vocabulary."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np

from neurale._validation import validate_integer, validate_number
from neurale.exceptions import ValidationError

from .._validation import readonly
from .vocabulary import Vocabulary


class NgramLanguageModel:
    """A counted unigram or bigram model over a :class:`Vocabulary`.

    The model is estimated from token sequences by counting, and nothing about
    the estimate is implicit. Every training sequence is bracketed by the
    vocabulary's own :attr:`~Vocabulary.bos` and :attr:`~Vocabulary.eos`
    symbols, so the first token is conditioned on a real starting context and
    ending a sequence is a real event with a probability rather than a rule
    imposed by the decoder. Counts, probabilities, and log probabilities are
    all published, so a caller can check the estimate rather than trust it.

    Parameters
    ----------
    vocabulary : Vocabulary
        Fixed token inventory. It fixes the width of every count and
        probability array, and its order fixes their column order.
    order : {1, 2}, optional
        ``1`` for a unigram model, ``2`` for a bigram model. A unigram model
        still publishes a full transition matrix, with the same unigram
        distribution in every context row, so a decoder consumes both the same
        way.
    smoothing : float, optional
        Additive (add-k) weight applied to every count before normalizing:
        ``(count + k) / (total + k * size)``. The default ``0.0`` is *no*
        smoothing, which means an unseen event keeps probability exactly zero
        and log probability ``-inf``. That is a deliberate default: a zero is
        a fact about the corpus, and hiding it behind a floor no caller asked
        for is how a decoder ends up scoring sequences the corpus never
        supported.

    Raises
    ------
    neurale.exceptions.ValidationError
        If any argument is invalid.

    Notes
    -----
    Add-k is the only smoothing here. There is no backoff, no interpolation
    between the bigram and unigram estimates, and no discounting: those are
    modelling choices with their own parameters, and a caller who wants one
    composes it rather than discovering it inside a fit.

    An unsmoothed context that the corpus never saw has no distribution at all
    -- its row is entirely zero and its log row entirely ``-inf`` -- rather
    than a uniform one. A uniform fallback would be an invented estimate.

    A successful fit takes a snapshot of the configuration it ran under, and
    every fitted lookup resolves through that snapshot rather than through the
    writable attributes. So :attr:`fitted_vocabulary_`, not
    :attr:`vocabulary`, is what a fitted model answers for. The constructor
    arguments stay writable and are re-validated at the start of the next
    :meth:`fit`, which makes the two states independent: a rejected refit
    leaves the previous fit not merely stored but *usable*, and a pending
    configuration change never reinterprets counts taken under a different
    one.

    Examples
    --------
    >>> from neurale.decoding.sequence import NgramLanguageModel, Vocabulary
    >>> vocabulary = Vocabulary(["yes", "no", "stop"], bos="<s>", eos="stop")
    >>> model = NgramLanguageModel(vocabulary).fit([["yes", "no"], ["yes", "yes"]])
    >>> model.n_sequences_, model.n_tokens_
    (2, 6)
    >>> round(model.probability("no", given="yes"), 4)
    0.3333
    """

    def __init__(
        self,
        vocabulary: Vocabulary,
        *,
        order: int = 2,
        smoothing: float = 0.0,
    ) -> None:
        self.vocabulary = _validate_vocabulary(vocabulary)
        self.order = _validate_order(order)
        self.smoothing = _validate_smoothing(smoothing)
        self._clear_fitted()

    def _validate_configuration(self) -> tuple[Vocabulary, int, float]:
        """Re-validate the public configuration and return it normalized.

        ``vocabulary``, ``order``, and ``smoothing`` are plain attributes a
        caller can rebind between fits, so validating them only in
        :meth:`__init__` would let a refit publish an estimate that does not
        match what the model reports -- a bigram fit on a model claiming
        ``order=3``, or all-``nan`` distributions from a ``nan`` weight. This
        runs before a fit touches anything, so an invalid configuration raises
        before a previous fit is discarded.
        """

        return (
            _validate_vocabulary(self.vocabulary),
            _validate_order(self.order),
            _validate_smoothing(self.smoothing),
        )

    # ----------------------------------------------------------------- fitting

    def fit(self, sequences: Iterable[Sequence[str]]) -> Self:
        """Count ``sequences`` and publish the resulting distributions.

        Parameters
        ----------
        sequences : iterable of sequence of str
            Training sequences of vocabulary tokens. Each is counted as
            ``bos``, its own tokens, then ``eos``; the boundary symbols are
            supplied by the model, so a sequence that already contains one is
            rejected rather than counted twice.

        Returns
        -------
        Self
            The fitted model.

        Raises
        ------
        neurale.exceptions.ValidationError
            If a sequence is empty, contains a boundary symbol, or contains a
            token outside the vocabulary with no unknown token configured.

        Notes
        -----
        The whole corpus is validated and counted before anything is
        published, so a rejected corpus leaves a fitted model untouched. The
        model's own configuration is re-validated first, for the same reason:
        an argument error is raised before anything is discarded.
        """

        vocabulary, order, smoothing = self._validate_configuration()
        size = vocabulary.size
        unigrams = np.zeros(size, dtype=np.float64)
        transitions = np.zeros((vocabulary.n_contexts, size), dtype=np.float64)

        n_sequences = 0
        for pos, sequence in enumerate(sequences):
            encoded = _encode_sequence(vocabulary, sequence, pos)
            context = vocabulary.bos_index
            for i in encoded:
                unigrams[i] += 1.0
                transitions[context, i] += 1.0
                context = i
            n_sequences += 1

        if n_sequences == 0:
            raise ValidationError("sequences must contain at least one sequence.")

        return self._publish(unigrams, transitions, vocabulary, order, smoothing, n_sequences)

    def _publish(
        self,
        unigrams: np.ndarray,
        transitions: np.ndarray,
        vocabulary: Vocabulary,
        order: int,
        smoothing: float,
        n_sequences: int,
    ) -> Self:
        """Derive the distributions from counts and publish one fit.

        The single place a fit becomes visible, so counting a corpus and
        restoring stored counts cannot drift apart: both normalize the same way
        and both snapshot the configuration the arrays were produced under.
        Inference reads that snapshot, never the public attributes, so a fitted
        model stays usable and self-consistent no matter what a caller rebinds
        afterwards -- including a configuration that a later fit rejects.
        """

        self._unigram_counts = readonly(unigrams)
        self._transition_counts = readonly(transitions)
        self._unigram_log = readonly(_log_normalize(unigrams[None, :], smoothing)[0])
        self._transition_log = readonly(
            _transition_log_probabilities(transitions, vocabulary, order, smoothing)
        )
        self._fitted_vocabulary = vocabulary
        self._fitted_order = order
        self._fitted_smoothing = smoothing
        self._n_sequences = n_sequences
        self._n_tokens = int(unigrams.sum())
        return self

    def _restore_fitted(
        self,
        *,
        vocabulary: Vocabulary,
        order: int,
        smoothing: float,
        unigram_counts: np.ndarray,
        transition_counts: np.ndarray,
        n_sequences: int,
    ) -> Self:
        """Rebuild a fitted model from the counts a corpus produced.

        Private, and the only supported way to reconstitute a fit: decoder
        persistence stores the counts, which are the primitive facts a fit
        establishes, and the distributions are recomputed here through the same
        normalization :meth:`fit` uses. Storing the probabilities too would let
        an artifact contradict itself; deriving them from the counts under the
        recorded configuration cannot.
        """

        vocabulary = _validate_vocabulary(vocabulary)
        order = _validate_order(order)
        smoothing = _validate_smoothing(smoothing)
        unigrams = _restored_counts(unigram_counts, (vocabulary.size,), "unigram_counts")
        transitions = _restored_counts(
            transition_counts,
            (vocabulary.n_contexts, vocabulary.size),
            "transition_counts",
        )
        if np.any(transitions[vocabulary.eos_index] != 0.0):
            raise ValidationError(
                "transition_counts must be zero in the eos row: counting stops at eos, so a "
                "corpus can contain no transition out of it."
            )

        return self._publish(
            unigrams,
            transitions,
            vocabulary,
            order,
            smoothing,
            validate_integer(n_sequences, "n_sequences", minimum=1),
        )

    def _clear_fitted(self) -> None:
        self._unigram_counts = None
        self._transition_counts = None
        self._unigram_log = None
        self._transition_log = None
        self._fitted_vocabulary: Vocabulary | None = None
        self._fitted_order: int | None = None
        self._fitted_smoothing: float | None = None
        self._n_sequences = None
        self._n_tokens = None

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been counted."""

        return self._transition_log is not None

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise ValidationError(f"{type(self).__name__} instance is not fitted.")

    # ---------------------------------------------------------- fitted results

    @property
    def unigram_counts_(self) -> np.ndarray:
        """``(size,)`` token counts, one per emission column.

        Every emitted position is counted, which includes the ``eos`` the
        model appends to each sequence and excludes the ``bos`` it prepends:
        ``bos`` is a context, never an emission.
        """

        self._check_is_fitted()
        return self._unigram_counts

    @property
    def transition_counts_(self) -> np.ndarray:
        """``(n_contexts, size)`` bigram counts, conditioned on the row's context.

        The last row is the ``bos`` context, so it counts how often each token
        started a sequence. The ``eos`` row is always zero: counting stops
        there, which is what ends a sequence.
        """

        self._check_is_fitted()
        return self._transition_counts

    @property
    def unigram_log_probabilities_(self) -> np.ndarray:
        """``(size,)`` unigram log probabilities, ``-inf`` where a count is zero."""

        self._check_is_fitted()
        return self._unigram_log

    @property
    def transition_log_probabilities_(self) -> np.ndarray:
        """``(n_contexts, size)`` conditional log probabilities.

        Row ``c`` is the distribution over the next token given context ``c``,
        and sums to one in probability space unless the context was never
        seen and no smoothing was asked for, in which case it is all ``-inf``.
        """

        self._check_is_fitted()
        return self._transition_log

    @property
    def fitted_vocabulary_(self) -> Vocabulary:
        """The vocabulary the published arrays were counted over.

        This is what every fitted lookup resolves through, and it is what a
        consumer should hold onto. It is a snapshot taken when :meth:`fit`
        succeeded, so it does not follow a later rebinding of
        :attr:`vocabulary`: a fitted model always answers for the corpus it
        actually counted.
        """

        self._check_is_fitted()
        return self._fitted_vocabulary

    @property
    def fitted_order_(self) -> int:
        """The order the published arrays were estimated at."""

        self._check_is_fitted()
        return self._fitted_order

    @property
    def fitted_smoothing_(self) -> float:
        """The additive weight the published arrays were normalized under."""

        self._check_is_fitted()
        return self._fitted_smoothing

    @property
    def n_sequences_(self) -> int:
        """Number of sequences counted."""

        self._check_is_fitted()
        return self._n_sequences

    @property
    def n_tokens_(self) -> int:
        """Number of emitted tokens counted, including one ``eos`` per sequence."""

        self._check_is_fitted()
        return self._n_tokens

    # -------------------------------------------------------------- inference

    def log_probability(self, token: str, *, given: str | None = None) -> float:
        """Return ``log P(token | given)``, or the unigram log probability.

        Parameters
        ----------
        token : str
            Token to score. Out-of-vocabulary tokens follow the vocabulary's
            unknown-token policy.
        given : str or None, optional
            Context to condition on, which may be the vocabulary's ``bos``.
            ``None`` asks for the unigram log probability, whatever the
            model's order.

        Returns
        -------
        float
            A log probability in ``[-inf, 0]``.
        """

        self._check_is_fitted()
        vocabulary = self._fitted_vocabulary
        idx = vocabulary.index(token)
        if given is None:
            return float(self._unigram_log[idx])
        return float(self._transition_log[vocabulary.context_index(given), idx])

    def probability(self, token: str, *, given: str | None = None) -> float:
        """Return ``P(token | given)`` as a plain probability in ``[0, 1]``."""

        return float(np.exp(self.log_probability(token, given=given)))

    def log_transition(self, context: str) -> np.ndarray:
        """Return the ``(size,)`` log-probability row for ``context``.

        This is the row a decoder adds to a step of model scores, so reading
        it is how a caller checks what the language model contributed.
        """

        self._check_is_fitted()
        return self._transition_log[self._fitted_vocabulary.context_index(context)]

    def sequence_log_probability(self, tokens: Sequence[str], *, boundaries: bool = True) -> float:
        """Return the joint log probability of a token sequence.

        Parameters
        ----------
        tokens : sequence of str
            Tokens to score. Under ``boundaries=True`` they must not contain a
            boundary symbol, because the terminator is appended for you.
        boundaries : bool, optional
            Whether to score the sequence the way :meth:`fit` counted it --
            conditioned on ``bos`` and terminated by an appended ``eos``.
            ``False`` scores exactly the tokens given, starting from the
            ``bos`` context and appending nothing. ``False`` is therefore what
            a :class:`Hypothesis` needs: its ``tokens`` already carry the
            ``eos`` it emitted, so appending another would score a sequence it
            never proposed.

        Returns
        -------
        float
            The summed log probabilities, ``-inf`` if any step is impossible.

        Raises
        ------
        neurale.exceptions.ValidationError
            If ``boundaries`` is true and ``tokens`` contains ``eos``, which
            would be counted twice. ``bos`` is never scorable and is rejected
            either way.

        Examples
        --------
        Recomputing what a decoder reported, for a complete hypothesis:

        >>> import numpy as np
        >>> from neurale.decoding.sequence import (
        ...     BeamSearchDecoder,
        ...     NgramLanguageModel,
        ...     Vocabulary,
        ... )
        >>> vocabulary = Vocabulary(["yes", "no", "stop"], bos="<s>", eos="stop")
        >>> model = NgramLanguageModel(vocabulary, smoothing=1.0)
        >>> model = model.fit([["yes", "no"], ["yes", "yes"], ["no", "yes"]])
        >>> best = BeamSearchDecoder(model).decode(np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]))[0]
        >>> best.tokens, best.is_complete
        (('yes', 'stop'), True)
        >>> score = model.sequence_log_probability(best.tokens, boundaries=False)
        >>> bool(np.isclose(score, best.language_score))
        True
        """

        self._check_is_fitted()
        vocabulary = self._fitted_vocabulary
        if boundaries:
            for pos, token in enumerate(tokens):
                if token == vocabulary.eos:
                    raise ValidationError(
                        f"tokens[{pos}] is the boundary symbol {token!r}, and "
                        "boundaries=True appends one itself, so it would be counted twice. "
                        "Pass boundaries=False to score a sequence that already ends in eos, "
                        "which is what a completed Hypothesis carries."
                    )
        indices = [vocabulary.index(token) for token in tokens]
        if boundaries:
            indices.append(vocabulary.eos_index)
        if not indices:
            return 0.0

        total = 0.0
        context = vocabulary.bos_index
        for i in indices:
            total += float(self._transition_log[context, i])
            context = i
        return total

    def __repr__(self) -> str:
        # Report the configuration that is in effect: the fitted snapshot once
        # there is one, the pending arguments before that. Reading the public
        # attributes unconditionally would let a rebound vocabulary make repr
        # itself raise, and would describe an estimate the model does not hold.
        fitted = self.is_fitted
        vocabulary = self._fitted_vocabulary if fitted else self.vocabulary
        order = self._fitted_order if fitted else self.order
        smoothing = self._fitted_smoothing if fitted else self.smoothing
        size = vocabulary.size if isinstance(vocabulary, Vocabulary) else "?"
        return (
            f"{type(self).__name__}(order={order}, smoothing={smoothing}, "
            f"size={size}, {'fitted' if fitted else 'unfitted'})"
        )


def _validate_vocabulary(value: object) -> Vocabulary:
    if not isinstance(value, Vocabulary):
        raise ValidationError("vocabulary must be a Vocabulary.")
    return value


def _validate_order(value: object) -> int:
    order = validate_integer(value, "order", minimum=1)
    if order > 2:
        raise ValidationError(f"order must be 1 or 2; got {order}.")
    return order


def _validate_smoothing(value: object) -> float:
    return float(validate_number(value, "smoothing", kind="real", minimum=0, coerce=True))


def _transition_log_probabilities(
    transitions: np.ndarray,
    vocabulary: Vocabulary,
    order: int,
    smoothing: float,
) -> np.ndarray:
    if order == 1:
        # A unigram model conditions on nothing, so every context row is the
        # same distribution. Publishing the full matrix anyway means a decoder
        # never asks which order it is holding.
        unigram = _log_normalize(transitions.sum(axis=0, keepdims=True), smoothing)
        return np.repeat(unigram, vocabulary.n_contexts, axis=0)
    return _log_normalize(transitions, smoothing)


def _encode_sequence(vocabulary: Vocabulary, sequence: object, pos: int) -> list[int]:
    if isinstance(sequence, str) or not isinstance(sequence, Sequence | np.ndarray):
        raise ValidationError(f"sequence {pos} must be a sequence of tokens, not a bare string.")
    tokens = list(sequence)
    if not tokens:
        raise ValidationError(f"sequence {pos} must contain at least one token.")

    encoded: list[int] = []
    for token in tokens:
        if token == vocabulary.bos or token == vocabulary.eos:
            raise ValidationError(
                f"sequence {pos} contains the boundary symbol {token!r}. The model "
                "brackets every sequence with bos and eos itself, so a sequence that "
                "carries one would be counted twice."
            )
        try:
            encoded.append(vocabulary.index(token))
        except ValidationError as exc:
            raise ValidationError(f"sequence {pos}: {exc}") from exc
    encoded.append(vocabulary.eos_index)
    return encoded


def _restored_counts(values: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray.")
    counts = np.asarray(values, dtype=np.float64)
    if counts.shape != shape:
        raise ValidationError(f"{name} must have shape {shape}; got {counts.shape}.")
    if not np.all(np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValidationError(f"{name} must contain finite, non-negative counts.")
    return np.ascontiguousarray(counts)


def _log_normalize(counts: np.ndarray, smoothing: float) -> np.ndarray:
    """Normalize count rows to log probabilities under add-k smoothing.

    A row whose smoothed mass is zero -- an unseen context with no smoothing
    -- has no distribution, and every entry becomes ``-inf`` rather than a
    uniform guess the corpus never supported.
    """

    smoothed = counts + smoothing
    totals = smoothed.sum(axis=1, keepdims=True)
    empty = totals <= 0.0
    # Divide against a safe denominator, then mask: a zero row must produce
    # -inf, not a nan that would silently poison every score it reaches.
    probabilities = smoothed / np.where(empty, 1.0, totals)
    with np.errstate(divide="ignore"):
        log_probabilities = np.log(probabilities)
    return np.where(empty, -np.inf, log_probabilities)


__all__ = ["NgramLanguageModel"]
