#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The typed result of a sequence search."""

from __future__ import annotations

import math
from dataclasses import dataclass

from neurale.exceptions import ValidationError


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One candidate token sequence and the scores that ranked it.

    A hypothesis reports its two contributions separately as well as combined,
    so a caller can see how much of a ranking came from the model and how much
    from the language model rather than having to re-derive it from a weight.

    Parameters
    ----------
    tokens : tuple of str
        The token sequence, oldest first, without boundary symbols. A complete
        hypothesis ends with the vocabulary's ``eos``, which *is* included:
        completing is an emission, and hiding it would make the score
        unaccountable.
    indices : tuple of int
        Emission columns of ``tokens``, in the same order. This is also the
        tie-breaking key: two hypotheses with equal scores are ordered by
        vocabulary position, which is deterministic and independent of how the
        strings happen to collate.
    model_score : float
        Summed log probability the per-step model assigned to ``tokens``.
    language_score : float
        Summed log probability the language model assigned to ``tokens``.
    score : float
        Combined log score, ``model_score + language_weight * language_score``.
    normalized_score : float
        ``score`` divided by ``len(tokens) ** length_normalization``. This is
        what a beam ranks by; with the default exponent of zero the divisor is
        one and it equals ``score`` exactly.
    is_complete : bool
        Whether the hypothesis has emitted ``eos``. A complete hypothesis is
        never extended again.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the fields are inconsistent.
    """

    tokens: tuple[str, ...]
    indices: tuple[int, ...]
    model_score: float
    language_score: float
    score: float
    normalized_score: float
    is_complete: bool

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, tuple) or not all(
            isinstance(token, str) for token in self.tokens
        ):
            raise ValidationError("tokens must be a tuple of strings.")
        if not isinstance(self.indices, tuple) or not all(
            isinstance(idx, int) and not isinstance(idx, bool) for idx in self.indices
        ):
            raise ValidationError("indices must be a tuple of integers.")
        if len(self.tokens) != len(self.indices):
            raise ValidationError(
                f"tokens and indices must have the same length; got "
                f"{len(self.tokens)} and {len(self.indices)}."
            )
        for name in ("model_score", "language_score", "score", "normalized_score"):
            value = getattr(self, name)
            if not isinstance(value, float) or math.isnan(value):
                raise ValidationError(f"{name} must be a float and not nan.")
        if not isinstance(self.is_complete, bool):
            raise ValidationError("is_complete must be a bool.")

    @property
    def n_tokens(self) -> int:
        """Number of tokens, counting a terminal ``eos``."""

        return len(self.tokens)

    @property
    def ranking_key(self) -> tuple[float, tuple[int, ...]]:
        """Total order a beam sorts by: best normalized score, then token order.

        Every hypothesis in one beam has a distinct token sequence, so this is
        a strict total order and a beam's contents never depend on the order
        candidates happened to be generated in.
        """

        return (-self.normalized_score, self.indices)

    def __repr__(self) -> str:
        state = "complete" if self.is_complete else "active"
        return f"{type(self).__name__}(tokens={self.tokens!r}, score={self.score:.6g}, {state})"


__all__ = ["Hypothesis"]
