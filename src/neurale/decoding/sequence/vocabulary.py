#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The fixed token inventory a sequence decoder works over."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from neurale.exceptions import ValidationError


class Vocabulary:
    """A fixed, ordered token inventory with explicit boundary symbols.

    The vocabulary is the contract between the model that emits per-step
    scores and the decoder that searches over them: ``tokens`` is exactly the
    column order of every score row, and a row of any other width is a caller
    error rather than something a decoder pads or truncates.

    Boundary symbols are named by the caller and are ordinary members of the
    inventory, not characters a decoder assumes. ``eos`` is one of ``tokens``,
    because completing a sequence is something the model emits and therefore
    something it must be able to score. ``bos`` is deliberately *not*: it is
    the context a sequence starts in and is never emitted, so giving it a
    column would create a score no model could produce.

    Parameters
    ----------
    tokens : sequence of str
        Emission tokens in the order the score columns use. Tokens are
        non-empty strings of any length, compared by value; there is no
        single-character assumption anywhere in this package. The order is
        preserved exactly as given, and it is what settles a tie between two
        equally scored hypotheses.
    bos : str
        Beginning-of-sequence symbol. It is the language-model context a fresh
        sequence starts in and must not appear in ``tokens``.
    eos : str
        End-of-sequence symbol, which must be one of ``tokens``. A hypothesis
        completes exactly when it emits this token.
    unknown : str or None, optional
        Token that stands for anything outside the vocabulary. When given it
        must be one of ``tokens`` and must not be ``eos``: folding unseen
        tokens onto the end-of-sequence symbol would turn "a word I have not
        seen" into "the sequence ends here", which silently rewrites a corpus
        and lets a language model count transitions out of ``eos``. When
        ``None`` an out-of-vocabulary token is an error, which is the default
        because silently folding unseen tokens together changes what a corpus
        says.

    Notes
    -----
    The four symbol roles are fixed and mutually consistent:

    ============ =========================================================
    Rule         Reason
    ============ =========================================================
    ``bos``      not in ``tokens``  -- a context, never emitted, so it has
                 no score column.
    ``eos``      in ``tokens``      -- completing is an emission, so the
                 model must be able to score it.
    ``unknown``  in ``tokens``      -- it is emitted like any other token.
    ``unknown``  is not ``eos``     -- see above.
    ``unknown``  is not ``bos``     -- implied, since ``bos`` is not a token.
    ============ =========================================================

    Raises
    ------
    neurale.exceptions.ValidationError
        If the tokens or the boundary symbols are invalid or inconsistent.

    Examples
    --------
    >>> from neurale.decoding.sequence import Vocabulary
    >>> vocabulary = Vocabulary(["yes", "no", "stop"], bos="<s>", eos="stop")
    >>> vocabulary.size, vocabulary.index("no"), vocabulary.eos_index
    (3, 1, 2)
    """

    __slots__ = ("_bos", "_eos", "_eos_idx", "_positions", "_tokens", "_unknown")

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        bos: str,
        eos: str,
        unknown: str | None = None,
    ) -> None:
        ordered = _validate_tokens(tokens)
        positions = {token: idx for idx, token in enumerate(ordered)}

        bos = _validate_symbol(bos, "bos")
        if bos in positions:
            raise ValidationError(
                f"bos must not be one of tokens; {bos!r} would need an emission column for a "
                "symbol no model emits. It is the starting context, not a token."
            )
        eos = _validate_symbol(eos, "eos")
        if eos not in positions:
            raise ValidationError(
                f"eos must be one of tokens; got {eos!r}. Completing a sequence is something "
                "the model emits, so it needs a column the model can score."
            )
        if unknown is not None:
            unknown = _validate_symbol(unknown, "unknown")
            if unknown not in positions:
                raise ValidationError(f"unknown must be one of tokens or None; got {unknown!r}.")
            if unknown == eos:
                raise ValidationError(
                    f"unknown must differ from eos; both are {eos!r}. Mapping an unseen token "
                    "onto eos would end the sequence instead of standing in for the token, so "
                    "a corpus would be counted with sequences it does not contain."
                )

        self._tokens = ordered
        self._positions = positions
        self._bos = bos
        self._eos = eos
        self._eos_idx = positions[eos]
        self._unknown = unknown

    # ------------------------------------------------------------- inventory

    @property
    def tokens(self) -> tuple[str, ...]:
        """Emission tokens, in the column order of every score row."""

        return self._tokens

    @property
    def size(self) -> int:
        """Number of emission tokens, which is the required width of a score row."""

        return len(self._tokens)

    @property
    def bos(self) -> str:
        """Beginning-of-sequence symbol; a context only, never emitted."""

        return self._bos

    @property
    def eos(self) -> str:
        """End-of-sequence symbol; emitting it completes a hypothesis."""

        return self._eos

    @property
    def eos_index(self) -> int:
        """Column of :attr:`eos`."""

        return self._eos_idx

    @property
    def unknown(self) -> str | None:
        """Token unseen input is mapped onto, or ``None`` when that is an error."""

        return self._unknown

    # -------------------------------------------------------------- contexts

    @property
    def n_contexts(self) -> int:
        """Number of language-model contexts: every token, plus :attr:`bos`."""

        return len(self._tokens) + 1

    @property
    def bos_index(self) -> int:
        """Context row of :attr:`bos`.

        It sits past the last emission column, so a transition matrix is
        ``(n_contexts, size)``: every token can be conditioned on, and the
        starting context conditions nothing on itself.
        """

        return len(self._tokens)

    # ---------------------------------------------------------------- lookup

    def index(self, token: str) -> int:
        """Return the emission column of ``token``.

        Applies the unknown-token policy: an out-of-vocabulary *string* becomes
        :attr:`unknown` when one is configured, and is an error otherwise. A
        non-string is always an error: the unknown policy stands in for tokens
        this vocabulary has not seen, not for input of the wrong type, and
        folding an ``int`` onto ``<unk>`` would hide a malformed corpus. So is
        :attr:`bos`, which has no emission column to return at all.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the token is not a string, is :attr:`bos`, or is out of
            vocabulary with no ``unknown`` configured.
        """

        if isinstance(token, np.str_):
            token = str(token)
        if not isinstance(token, str):
            raise ValidationError(
                f"token must be a string; got {type(token).__name__}. The unknown-token policy "
                "covers unseen tokens, not values of the wrong type."
            )
        pos = self._positions.get(token)
        if pos is not None:
            return pos
        # bos is a context, never an emission, so it has no column to return.
        # Letting it reach the unknown policy would hand back the <unk> column
        # and turn a caller's mistake into a plausible-looking encoding.
        if token == self._bos:
            raise ValidationError(
                f"bos {token!r} has no emission column: it is the context a sequence starts "
                "in and is never emitted. Use context_index() to condition on it."
            )
        if self._unknown is not None:
            return self._positions[self._unknown]
        raise ValidationError(
            f"token {token!r} is not in the vocabulary, and no unknown token is configured. "
            "Add it to the vocabulary, or configure unknown= to fold unseen tokens onto."
        )

    def token(self, idx: int) -> str:
        """Return the token in column ``index``."""

        if not isinstance(idx, int | np.integer) or isinstance(idx, bool):
            raise ValidationError("index must be an integer.")
        pos = int(idx)
        if not 0 <= pos < len(self._tokens):
            raise ValidationError(f"index must lie in [0, {len(self._tokens)}); got {pos}.")
        return self._tokens[pos]

    def context_index(self, token: str) -> int:
        """Return the transition-matrix row to condition on after ``token``.

        :attr:`bos` maps onto :attr:`bos_index`; every other token maps onto
        its own emission column through :meth:`index`.
        """

        if token == self._bos:
            return self.bos_index
        return self.index(token)

    def encode(self, tokens: Iterable[str]) -> np.ndarray:
        """Return the emission columns of ``tokens`` as an integer array."""

        return np.array([self.index(token) for token in tokens], dtype=np.intp)

    def decode(self, indices: Iterable[int]) -> tuple[str, ...]:
        """Return the tokens in the given emission columns."""

        return tuple(self.token(idx) for idx in indices)

    # ---------------------------------------------------------------- dunder

    def __len__(self) -> int:
        return len(self._tokens)

    def __contains__(self, token: object) -> bool:
        return token in self._positions

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vocabulary):
            return NotImplemented
        return (
            self._tokens == other._tokens
            and self._bos == other._bos
            and self._eos == other._eos
            and self._unknown == other._unknown
        )

    def __hash__(self) -> int:
        return hash((self._tokens, self._bos, self._eos, self._unknown))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(size={self.size}, bos={self._bos!r}, "
            f"eos={self._eos!r}, unknown={self._unknown!r})"
        )


def _validate_tokens(tokens: object) -> tuple[str, ...]:
    if isinstance(tokens, str) or not isinstance(tokens, Sequence | np.ndarray):
        raise ValidationError("tokens must be a sequence of strings.")
    ordered = tuple(str(token) if isinstance(token, np.str_) else token for token in tokens)
    if not ordered:
        raise ValidationError("tokens must contain at least one token.")
    for token in ordered:
        _validate_symbol(token, "tokens entries")
    seen: dict[str, int] = {}
    for pos, token in enumerate(ordered):
        if token in seen:
            raise ValidationError(
                f"tokens must not repeat; {token!r} appears at positions {seen[token]} and {pos}."
            )
        seen[token] = pos
    return ordered


def _validate_symbol(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be a string.")
    if not value:
        raise ValidationError(f"{name} must not be empty.")
    return value


__all__ = ["Vocabulary"]
