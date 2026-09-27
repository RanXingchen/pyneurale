#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Kalman decoding over a fitted linear-Gaussian state-space model."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Self

import numpy as np

from neurale._validation import validate_choice, validate_number
from neurale.data import FeatureMatrix, SignalArray
from neurale.exceptions import ValidationError
from neurale.models.state_space import (
    KalmanFilter,
    LinearGaussianParameters,
    LinearGaussianStateSpace,
)

from ._alignment import require_same_clock
from ._stages import FeatureScaler, require_fittable, validate_scaler
from ._validation import readonly
from .base import ContinuousDecoder, FitSegment
from .schema import ContinuousTargetSchema, FeatureSchema

# The missing-observation policy itself belongs to KalmanFilter.filter(); only
# the spelling is checked here, so that a typo fails at construction rather
# than at the first prediction.
_MISSING_POLICIES = ("error", "predict")

# A step more than half a bin longer than the one the frames declare is a gap,
# not timing jitter. Where a recording is split is the caller's decision, so a
# gap is refused rather than silently turned into a segment boundary -- and it
# is certainly never estimated across.
_MAX_STEP_RATIO = 1.5

# Target metadata that has to agree across the segments of one fit, because a
# prediction reproduces exactly these. Segment attrs may differ (a trial index,
# say); the first segment's are the ones a prediction carries.
_SHARED_TARGET_FIELDS = ("output_names", "unit", "name", "fs")


class KalmanDecoder(ContinuousDecoder):
    """Decode a continuous target with a linear-Gaussian state-space model.

    The decoder composes three owned pieces and adds no arithmetic of its own:
    an optional feature :attr:`scaler`, a
    :class:`~neurale.models.state_space.LinearGaussianStateSpace` estimator,
    and the :class:`~neurale.models.state_space.KalmanFilter` that runs its
    parameters. The latent state *is* the target -- one state dimension per
    target channel -- and the observation is one frame of prepared features.

    A fit keeps the immutable parameter snapshot and the filter bound to it,
    and lets the estimator go. There is deliberately no handle onto a live,
    refittable model inside a fitted decoder: :attr:`parameters_` is the model
    :meth:`predict` runs, and nothing can make the two disagree.

    A fit may span several independent segments, supplied to
    :meth:`fit_segments`. No transition pair is ever formed across a segment
    boundary, and a segment whose own frames are not contiguous is refused
    rather than fitted across: which rows belong to one stretch of dynamics is
    the caller's declaration, and the decoder only verifies it.

    Prediction is stateful. Each call continues the filter from where the last
    one left it, so decoding a recording in chunks gives exactly what decoding
    it whole gives. :meth:`reset` returns the filter to the fitted initial
    state, which is the mean state at the *start* of the fitted segments, or to
    a caller-supplied state and covariance. Nothing fitted -- the scaler
    statistics, the model parameters, the feature schema, the device -- ever
    changes during a prediction.

    Parameters
    ----------
    scaler : neurale.models.preprocessing.StandardScaler or \
neurale.models.preprocessing.MinMaxScaler or None, optional
        Feature scaler to own. The decoder fits a *copy*, so the instance
        passed here stays a configuration and is never mutated; the fitted copy
        is :attr:`scaler_`, published frozen. A scaler that is already frozen
        is a published fit rather than a configuration, and is rejected.
        ``None`` feeds the features to the model unscaled.
    feature_names : sequence of str or None, optional
        Names of the features to consume, in the order to consume them. They
        must all be present in the fit features. ``None`` consumes every
        feature in its original order. This is the only feature selection there
        is: the decoder never searches for a subset.
    fit_offsets : bool, optional
        Whether the estimated transition and observation equations carry an
        offset term.
    jitter : float, optional
        Added to the diagonal of every estimated covariance. A fit on a single
        segment yields an exactly zero initial covariance without it.
    innovation_jitter : float, optional
        Added to the diagonal of the innovation covariance before each gain
        solve.
    missing : {"error", "predict"}, optional
        What a non-finite observation row means. The behavior is the state-space
        model's: ``"predict"`` runs the predict step alone for rows that are
        *entirely* ``nan``, and everything else -- a partially missing row, an
        infinity -- stays an error.

    Raises
    ------
    neurale.exceptions.ValidationError
        If any argument is invalid.

    Notes
    -----
    The state dimension is exactly the number of target channels; no constant
    bias state is appended. Feature selection is by name, from the caller --
    there is no greedy or ranked search. Feature statistics do not adapt
    during prediction, so decoding a recording in chunks gives the same result
    as decoding it whole. The initial state is the mean state at the start of
    the fitted segments. There is no override that replaces fitted transition
    or covariance diagonals with constants: a decoder reports the model it
    actually fitted.

    Examples
    --------
    >>> import numpy as np
    >>> from neurale.data import FeatureMatrix, SignalArray
    >>> from neurale.decoding import KalmanDecoder
    >>> from neurale.models import StandardScaler
    >>> rng = np.random.default_rng(0)
    >>> rate = 50.0
    >>> position = np.cumsum(rng.normal(scale=0.05, size=(200, 2)), axis=0)
    >>> X = FeatureMatrix(
    ...     data=position @ rng.normal(size=(2, 6)) + rng.normal(scale=0.1, size=(200, 6)),
    ...     fs=rate,
    ...     feature_names=[f"rate_{i}" for i in range(6)],
    ...     unit="Hz",
    ...     shift=1.0 / rate,
    ... )
    >>> y = SignalArray.from_array(
    ...     position,
    ...     fs=rate,
    ...     time=X.time.copy(),
    ...     channel_names=["x", "y"],
    ...     channel_types="behavior",
    ...     units="m",
    ...     name="cursor",
    ... )
    >>> decoder = KalmanDecoder(scaler=StandardScaler()).fit(X, y)
    >>> predicted = decoder.predict(X)
    >>> predicted.channels.names, predicted.unit
    (['x', 'y'], 'm')
    """

    def __init__(
        self,
        *,
        scaler: FeatureScaler | None = None,
        feature_names: Sequence[str] | None = None,
        fit_offsets: bool = True,
        jitter: float = 0.0,
        innovation_jitter: float = 0.0,
        missing: str = "error",
    ) -> None:
        self.scaler = validate_scaler(scaler)
        if not isinstance(fit_offsets, bool):
            raise ValidationError("fit_offsets must be a bool.")
        self.feature_names = _validate_selection(feature_names)
        self.fit_offsets = fit_offsets
        self.jitter = _nonnegative(jitter, "jitter")
        self.innovation_jitter = _nonnegative(innovation_jitter, "innovation_jitter")
        self.missing = validate_choice(missing, _MISSING_POLICIES, "missing")

    # ----------------------------------------------------------------- fitting

    def fit_segments(self, segments: Sequence[tuple[FeatureMatrix, SignalArray]]) -> Self:
        """Fit from one or more independent segments.

        Parameters
        ----------
        segments : sequence of (FeatureMatrix, SignalArray)
            Non-empty sequence of aligned pairs, each an independent stretch of
            data: a separate recording, a separate trial, or one side of a
            discontinuity. Every segment must describe the same features and
            the same target channels, units, name, and rate, and must belong to
            the same timeline.

        Returns
        -------
        Self
            The fitted decoder.

        Raises
        ------
        neurale.exceptions.ValidationError
            If a segment is invalid, is not internally contiguous, is not
            aligned, or disagrees with the first segment.
        neurale.exceptions.DeviceUnavailableError
            If ``cuda`` is requested; this decoder is CPU-only.

        Notes
        -----
        ``fit(X, y)`` is exactly ``fit_segments([(X, y)])``. The segments are
        never concatenated into one sequence for
        estimation purposes: the transition is estimated from within-segment
        pairs only, and the initial state from the segment starts. The first
        segment's target ``attrs`` are the ones a prediction carries.
        """

        return self._fit_prepared(self._prepare_segments(segments))

    def _prepare_segments(self, segments: object) -> list[FitSegment]:
        if isinstance(segments, str | FeatureMatrix | SignalArray) or not isinstance(
            segments, Sequence
        ):
            raise ValidationError("segments must be a sequence of (X, y) pairs.")
        if not segments:
            raise ValidationError("segments must contain at least one segment.")

        prepared: list[FitSegment] = []
        for i, segment in enumerate(segments):
            if not isinstance(segment, tuple) or len(segment) != 2:
                raise ValidationError(f"segment {i} must be a (X, y) pair.")
            X, y = segment
            try:
                prepared.append(self._prepare_fit_segment(X, y))
            except ValidationError as exc:
                raise ValidationError(f"segment {i}: {exc}") from exc

        first = prepared[0]
        for i, segment in enumerate(prepared[1:], start=1):
            first.schema.require_compatible(segment.schema, name=f"segment {i} X")
            _require_same_outputs(first.target_schema, segment.target_schema, i)
            require_same_clock(
                first.clock,
                segment.clock,
                name=f"segment {i}",
                reference="segment 0",
            )
        return prepared

    def _prepare_fit_segment(self, X: FeatureMatrix, y: SignalArray) -> FitSegment:
        # The base checks alignment; a state-space fit additionally needs each
        # segment to be one uninterrupted stretch, and the configured selection
        # to name features this data actually has. Both are argument errors, so
        # both are raised here, before _fit_prepared discards anything: a
        # rejected refit leaves an already fitted decoder untouched.
        prepared = super()._prepare_fit_segment(X, y)
        _require_contiguous(prepared.y.time, prepared.schema)
        _resolve_selection(self.feature_names, prepared.schema)
        # The configuration is checked again here, not only in __init__: the
        # attributes are plain and a caller may have frozen the scaler since.
        require_fittable(self.scaler)
        return prepared

    def _fit_decoder(self, segments: Sequence[FitSegment]) -> None:
        # Already validated in _prepare_fit_segment; resolved again here from
        # the same schema so that what the model consumes is derived where it
        # is used rather than carried across the discard.
        selection = _resolve_selection(self.feature_names, segments[0].schema)
        # Selection first, then scaling: the scaler then measures exactly the
        # columns the model consumes. Both scalers are per-feature, so the
        # statistics of the kept columns are the same either way.
        X = np.concatenate(
            [_select(segment.X, selection) for segment in segments],
            axis=0,
            dtype=np.float64,
        )
        y = np.concatenate(
            [np.asarray(segment.y.data, dtype=np.float64) for segment in segments],
            axis=0,
        )
        lengths = np.array([segment.X.shape[0] for segment in segments], dtype=np.intp)

        scaler = None if self.scaler is None else copy.deepcopy(self.scaler)
        if scaler is not None:
            X = scaler.fit_transform(X)
            # Published frozen, for the same reason the estimator is not
            # published at all: nothing a caller does to what it reads may
            # change what predict() computes.
            scaler.freeze()

        estimator = LinearGaussianStateSpace(fit_offsets=self.fit_offsets, jitter=self.jitter)
        estimator.fit(
            np.ascontiguousarray(y),
            np.ascontiguousarray(X, dtype=np.float64),
            segment_lengths=lengths,
        )

        # The estimator is not kept. What a fitted decoder reports is the
        # snapshot its filter runs plus the counts this fit produced, so there
        # is no second, refittable copy of the model to disagree with it.
        self._selection = selection
        self._selected_feature_names = _selected_names(selection, segments[0].schema)
        self._scaler = scaler
        self._filter = KalmanFilter(estimator.parameters_, jitter=self.innovation_jitter)
        self._segment_lengths = readonly(lengths)
        self._n_samples = estimator.n_samples_
        self._n_transitions = estimator.n_transitions_
        self._at_segment_start = True

    def _clear_fitted(self) -> None:
        super()._clear_fitted()
        self._selection = None
        self._selected_feature_names = None
        self._scaler = None
        self._filter = None
        self._segment_lengths = None
        self._n_samples = None
        self._n_transitions = None
        self._at_segment_start = True

    # -------------------------------------------------------------- prediction

    def _decode(self, X: np.ndarray) -> np.ndarray:
        result = self._filter.filter(
            self._observations(X),
            missing=self.missing,
            predict_first=not self._at_segment_start,
        )
        # Only the runtime position in the sequence moves. The first frame
        # after a fit or a reset is corrected in place, because the initial
        # state estimates the state at the start of a segment rather than the
        # step before it; every later frame costs a predict and an update.
        self._at_segment_start = False
        return np.array(result.states, dtype=np.float64)

    def _observations(self, X: np.ndarray) -> np.ndarray:
        selected = np.ascontiguousarray(_select(X, self._selection), dtype=np.float64)
        if self._scaler is None:
            return selected
        if self.missing == "predict":
            # A row the policy treats as absent has nothing to scale, and the
            # scaler rejects non-finite input, so it is passed straight through
            # and the state-space model decides what an absent row means. Any
            # other non-finite row still reaches the scaler and is rejected.
            absent = np.isnan(selected).all(axis=1)
            if absent.any():
                scaled = np.empty_like(selected)
                scaled[absent] = np.nan
                present = ~absent
                if present.any():
                    scaled[present] = self._scaler.transform(selected[present])
                return scaled
        return np.ascontiguousarray(self._scaler.transform(selected), dtype=np.float64)

    def reset(
        self,
        state: np.ndarray | None = None,
        covariance: np.ndarray | None = None,
    ) -> None:
        """Return the filter to the fitted initial state, or to explicit values.

        Parameters
        ----------
        state : numpy.ndarray or None, optional
            Explicit ``(n_outputs,)`` state to continue from. ``None`` restores
            :attr:`initial_state_`, the mean state at the start of the fitted
            segments.
        covariance : numpy.ndarray or None, optional
            Explicit ``(n_outputs, n_outputs)`` state covariance. ``None``
            restores :attr:`initial_covariance_`.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the decoder is unfitted and an explicit state or covariance is
            given, or if either has the wrong shape or is not a valid
            covariance.

        Notes
        -----
        Nothing fitted is touched -- not the model parameters, the scaler
        statistics, the schema, or the device. Whatever the filter is reset to
        is the estimate for the *first frame of the next prediction* rather
        than for the step before it, which is the same convention the fitted
        initial state follows.

        Resetting an unfitted decoder with no arguments does nothing, as it
        does for every decoder. Resetting one with an explicit state is an
        error rather than a silent no-op: there is no fitted state space to
        place that state in.
        """

        if not self.is_fitted:
            if state is None and covariance is None:
                return
            raise ValidationError(
                f"{type(self).__name__} instance is not fitted; there is no state space to "
                "place an explicit state or covariance in."
            )
        self._restore(state, covariance)

    def _reset_runtime_state(self) -> None:
        self._restore(None, None)

    def _restore(self, state: np.ndarray | None, covariance: np.ndarray | None) -> None:
        self._filter.reset(state, covariance)
        self._at_segment_start = True

    # ---------------------------------------------------------- fitted results

    @property
    def parameters_(self) -> LinearGaussianParameters:
        """Immutable parameter snapshot the filter runs.

        This is the model the decoder actually predicts with, not a reading of
        an estimator that could since have been refitted: the estimator is
        discarded once the fit is recorded, so there is only one model here.
        Refitting the decoder builds a new snapshot and a new filter.
        """

        self._check_is_fitted()
        return self._filter.parameters

    @property
    def scaler_(self) -> FeatureScaler | None:
        """Fitted feature scaler, or ``None`` when the features are unscaled.

        This is the decoder's own copy of the configured :attr:`scaler`; the
        instance handed to the constructor is never fitted or mutated. The copy
        is frozen, so it can be read from and applied but cannot be refitted
        behind the decoder that runs it.
        """

        self._check_is_fitted()
        return self._scaler

    @property
    def selected_feature_names_(self) -> tuple[str, ...]:
        """Feature names the model consumes, in the order it consumes them.

        Recorded at fit time from the columns the fit actually selected, so
        changing the configured :attr:`feature_names` afterwards does not
        change what a fitted decoder reports it consumes.
        """

        self._check_is_fitted()
        return self._selected_feature_names

    @property
    def n_observations_(self) -> int:
        """Number of features the model consumes after selection."""

        return self.parameters_.n_observations_

    @property
    def n_samples_(self) -> int:
        """Total number of frames seen during the fit, over all segments."""

        self._check_is_fitted()
        return self._n_samples

    @property
    def n_transitions_(self) -> int:
        """Within-segment frame pairs the transition was estimated from.

        Fewer than :attr:`n_samples_` by exactly :attr:`n_segments_`: no pair
        is ever formed across a segment boundary.
        """

        self._check_is_fitted()
        return self._n_transitions

    @property
    def n_segments_(self) -> int:
        """Number of independent segments seen during the fit."""

        return int(self.segment_lengths_.shape[0])

    @property
    def segment_lengths_(self) -> np.ndarray:
        """Frame count of each fitted segment, in the order they were given."""

        self._check_is_fitted()
        return self._segment_lengths

    @property
    def state_(self) -> np.ndarray:
        """Current runtime state estimate; changed by predict and reset alone."""

        self._check_is_fitted()
        return self._filter.state

    @property
    def covariance_(self) -> np.ndarray:
        """Current runtime state covariance."""

        self._check_is_fitted()
        return self._filter.covariance

    @property
    def initial_state_(self) -> np.ndarray:
        """State :meth:`reset` restores: the mean state at the segment starts."""

        self._check_is_fitted()
        return self._filter.initial_state

    @property
    def initial_covariance_(self) -> np.ndarray:
        """State covariance :meth:`reset` restores."""

        self._check_is_fitted()
        return self._filter.initial_covariance


def _nonnegative(value: object, name: str) -> float:
    return float(validate_number(value, name, kind="real", minimum=0, coerce=True))


def _validate_selection(feature_names: object) -> tuple[str, ...] | None:
    if feature_names is None:
        return None
    if isinstance(feature_names, str) or not isinstance(feature_names, Sequence | np.ndarray):
        raise ValidationError("feature_names must be a sequence of strings or None.")
    names = tuple(str(name) if isinstance(name, np.str_) else name for name in feature_names)
    if not names:
        raise ValidationError("feature_names must select at least one feature.")
    for name in names:
        if not isinstance(name, str) or not name:
            raise ValidationError("feature_names must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValidationError("feature_names must be unique.")
    return names


def _resolve_selection(
    feature_names: tuple[str, ...] | None,
    schema: FeatureSchema,
) -> np.ndarray | None:
    if feature_names is None:
        return None
    positions = {name: idx for idx, name in enumerate(schema.feature_names)}
    missing = [name for name in feature_names if name not in positions]
    if missing:
        raise ValidationError(
            f"feature_names selects features the fit data does not provide: {missing!r}."
        )
    return np.array([positions[name] for name in feature_names], dtype=np.intp)


def _select(X: np.ndarray, selection: np.ndarray | None) -> np.ndarray:
    return X if selection is None else X[:, selection]


def _selected_names(selection: np.ndarray | None, schema: FeatureSchema) -> tuple[str, ...]:
    names = schema.feature_names
    return names if selection is None else tuple(names[idx] for idx in selection)


def _require_same_outputs(
    expected: ContinuousTargetSchema,
    found: ContinuousTargetSchema,
    idx: int,
) -> None:
    for attribute in _SHARED_TARGET_FIELDS:
        if getattr(expected, attribute) != getattr(found, attribute):
            raise ValidationError(
                f"segment {idx} y does not match segment 0: {attribute} is "
                f"{getattr(found, attribute)!r}, expected {getattr(expected, attribute)!r}."
            )
    if expected.channels != found.channels:
        raise ValidationError(f"segment {idx} y channels do not match segment 0.")


def _require_contiguous(time: np.ndarray, schema: FeatureSchema) -> None:
    """Raise when a segment's own frames are interrupted by a gap."""

    values = np.asarray(time, dtype=np.float64)
    if values.size < 2:
        return
    steps = np.diff(values)
    reference = _declared_step(schema)
    if reference is None:
        # Nothing declares the bin, so the frames are judged against their own
        # typical step. This is a self-consistency check, not an inference of
        # where a boundary belongs: a gap is still refused rather than split.
        reference = float(np.median(steps))
    if not reference > 0.0:
        return
    gaps = np.flatnonzero(steps > _MAX_STEP_RATIO * reference)
    if gaps.size:
        row = int(gaps[0])
        raise ValidationError(
            f"the frames are not contiguous: frame {row + 1} follows a gap of "
            f"{steps[row]:.6g} s where the frames advance by {reference:.6g} s. Split the "
            "data into one segment per uninterrupted stretch; a decoder never estimates a "
            "transition across a discontinuity, and never decides where one is."
        )


def _declared_step(schema: FeatureSchema) -> float | None:
    if schema.shift is not None:
        return float(schema.shift)
    if schema.fs is not None:
        return 1.0 / float(schema.fs)
    return None


__all__ = ["KalmanDecoder"]
