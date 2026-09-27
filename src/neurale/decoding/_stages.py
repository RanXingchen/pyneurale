#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The explicit preprocessing a decoder may own.

A decoder owns at most two stages, and both are objects the caller passes in
rather than behavior a decoder decides to apply: a feature scaler and a
temporal context. Everything else -- dimensionality reduction, feature
selection, trial slicing -- is composed *outside* a decoder, on the feature
matrix, where the caller can see it.

The order is fixed and is part of the contract: scaling first, then context
stacking. A scaler measures one statistic per real feature that way. Stacking
first would present the same feature as ``n_frames`` separate columns measured
over shifted windows, so a caller asking for "z-scored rates" would silently
get something else.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np

from neurale.exceptions import ValidationError
from neurale.models.preprocessing import MinMaxScaler, StandardScaler, TemporalContext

#: Scalers a decoder can own. Both are per-feature affine maps whose fitted
#: statistics describe the fit data and never change afterwards.
FeatureScaler = StandardScaler | MinMaxScaler


def require_fittable(scaler: FeatureScaler | None) -> None:
    """Reject a frozen scaler as configuration.

    A frozen scaler is a published fit, not a recipe for one, and a decoder
    fits its own copy of whatever it is given. Refusing it up front rather than
    at ``fit_transform`` keeps the failure an argument error, raised before a
    fitted decoder is discarded.
    """

    if scaler is not None and scaler.is_frozen:
        raise ValidationError(
            "scaler is frozen and cannot be fitted. Pass an unfitted scaler to configure "
            "one; the decoder fits its own copy and freezes that."
        )


def validate_scaler(scaler: object) -> FeatureScaler | None:
    """Validate a configured scaler."""

    if scaler is not None and not isinstance(scaler, StandardScaler | MinMaxScaler):
        raise ValidationError("scaler must be a StandardScaler, a MinMaxScaler, or None.")
    require_fittable(scaler)
    return scaler


def validate_context(context: object) -> TemporalContext | None:
    """Validate a configured temporal context."""

    if context is not None and not isinstance(context, TemporalContext):
        raise ValidationError("context must be a TemporalContext or None.")
    return context


def validate_stages(
    scaler: object,
    context: object,
) -> tuple[FeatureScaler | None, TemporalContext | None]:
    """Validate the owned stage configuration and return it.

    Used by a constructor to accept a configuration and again by every fit
    before it discards anything. Both stages stay plain public attributes, so a
    caller may rebind one between fits; the fit that consumes a configuration
    is therefore the fit that has to check it, and it has to check it early
    enough that a rejected refit leaves a fitted decoder untouched rather than
    failing from inside the model with the previous fit already gone.
    """

    return validate_scaler(scaler), validate_context(context)


class FeatureStages:
    """The owned scaling and context stages of one fitted decoder.

    Constructed by a fit from the decoder's configuration: the scaler is a
    fitted, frozen *copy*, so the configured instance stays a configuration,
    and the context is the caller's immutable transform used as given.

    A context drops the first ``left`` and last ``right`` frames of every
    block, which is why :meth:`trim` exists: a target and an output timestamp
    vector must lose exactly the same rows the features did, and they lose them
    through the same transform rather than through a second implementation of
    the same arithmetic.
    """

    __slots__ = ("_context", "_scaler")

    def __init__(self, scaler: FeatureScaler | None, context: TemporalContext | None) -> None:
        self._scaler = scaler
        self._context = context

    @classmethod
    def fitted(
        cls,
        scaler: FeatureScaler | None,
        context: TemporalContext | None,
        blocks: Sequence[np.ndarray],
    ) -> tuple[FeatureStages, list[np.ndarray]]:
        """Fit the stages on ``blocks`` and return them with the transformed blocks.

        The scaler measures every frame of every block, because that is the
        data the caller offered as the fit; the context is then applied to each
        block on its own, so a stacked frame never straddles two independent
        stretches of data.
        """

        owned = None if scaler is None else copy.deepcopy(scaler)
        if owned is not None:
            owned.fit(np.concatenate(list(blocks), axis=0, dtype=np.float64))
            # Published frozen: what a caller reads from a fitted decoder must
            # not be able to change what that decoder computes.
            owned.freeze()
        stages = cls(owned, context)
        return stages, [stages.transform(block) for block in blocks]

    @property
    def scaler(self) -> FeatureScaler | None:
        """The fitted, frozen scaler, or ``None`` when the features are unscaled."""

        return self._scaler

    @property
    def context(self) -> TemporalContext | None:
        """The temporal context, or ``None`` when frames are consumed one at a time."""

        return self._context

    @property
    def n_frames(self) -> int:
        """Number of source frames stacked into one model row."""

        return 1 if self._context is None else self._context.n_frames

    def n_model_features(self, n_features: int) -> int:
        """Number of columns the model consumes for ``n_features`` input features."""

        return int(n_features) * self.n_frames

    def transform(self, X: np.ndarray, *, name: str = "X") -> np.ndarray:
        """Scale and context-stack one block of feature frames."""

        stacked = np.ascontiguousarray(X, dtype=np.float64)
        if self._scaler is not None:
            stacked = self._scaler.transform(stacked)
        if self._context is None:
            return stacked
        self.require_frames(stacked.shape[0], name)
        return self._context.transform(stacked)

    def trim(self, values: np.ndarray, *, name: str = "y") -> np.ndarray:
        """Drop the rows the context drops, from a target or a timestamp vector."""

        if self._context is None:
            return values
        self.require_frames(values.shape[0], name)
        return self._context.trim(values)

    def require_frames(self, n_frames: int, name: str) -> None:
        """Raise when a block is too short for the context to produce a row."""

        # Checked here rather than left to the transform so that the message
        # names the decoder's input and the context that rejected it, instead
        # of an "X" the caller never passed.
        context = self._context
        if context is not None and n_frames < context.n_frames:
            raise ValidationError(
                f"{name} must have at least {context.n_frames} frames for a temporal context "
                f"of left={context.left}, right={context.right}; got {n_frames}."
            )
