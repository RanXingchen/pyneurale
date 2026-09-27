#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Deterministic sequence decoding over a fixed token vocabulary.

Where the rest of :mod:`neurale.decoding` turns feature frames into a target,
this package turns *per-step token scores* into token sequences. It sits after
a classifier rather than beside one: whatever produced the per-step
distribution -- an :class:`~neurale.decoding.LDADecoder`, a network, a
simulation -- is the caller's business, and nothing here fits a model on
neural data. That is why these classes are not
:class:`~neurale.decoding.BaseDecoder` subclasses: there is no feature schema,
no timeline, and no device to resolve, so inheriting that lifecycle would
promise checks this search cannot perform.

Three pieces compose:

- :class:`Vocabulary` fixes the token inventory, its order, and the boundary
  symbols. Its size is the required width of every score row, and its order is
  what breaks a tie between equally scored hypotheses.
- :class:`NgramLanguageModel` counts a corpus into a unigram or bigram
  distribution and publishes the counts alongside the probabilities.
- :class:`BeamSearchDecoder` searches over per-step scores under that prior and
  returns typed :class:`Hypothesis` values.

The search is deterministic. Two runs over the same input give the same
hypotheses in the same order, ties included, and decoding a sequence in chunks
gives exactly what decoding it in one call gives. Every choice that would
change which sequence wins -- whether the input is probabilities or log
probabilities, whether rows must already be normalized, what a zero
probability means, how the two score sources are weighed, how length is
normalized -- is a stated argument rather than a default buried in the search.

Boundary symbols are named by the caller and are ordinary tokens of any
length. There are no implicit single-character ``B`` or ``E`` markers, no
string concatenation standing in for a sequence, and no assumption that a
token is one character wide.

Importing this package pulls in NumPy and nothing else from the wider project
beyond the shared validation helpers: no native extension, no CUDA, no Torch,
no I/O, no experiment state, and no global context.
"""

from .beam import BeamSearchDecoder
from .hypotheses import Hypothesis
from .language_model import NgramLanguageModel
from .vocabulary import Vocabulary

__all__ = [
    "BeamSearchDecoder",
    "Hypothesis",
    "NgramLanguageModel",
    "Vocabulary",
]
