#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared lifecycle for classical decoders.

The bases here own everything a decoder must do identically -- input
validation, schema capture, device resolution, alignment checking, fitted and
reset semantics, and result construction -- and leave the concrete decoder only
the arithmetic. See :mod:`neurale.decoding` for what a decoder deliberately
does not own.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Self

import numpy as np

from neurale.data import Clock, FeatureMatrix, SignalArray
from neurale.exceptions import ValidationError
from neurale.runtime._types import ResolvedDevice

from ._alignment import feature_clock, require_aligned, require_same_clock
from ._dispatch import resolve_decoder_device
from .results import ClassificationPrediction
from .schema import ContinuousTargetSchema, FeatureSchema
from .targets import ClassificationTarget


@dataclass(frozen=True, slots=True)
class FitSegment:
    """One validated feature/target pair that a continuous fit consumes.

    A continuous fit spans one or more *independent* segments -- separate
    recordings, separate trials, or the two sides of a discontinuity. Each is
    validated on its own and handed to
    :meth:`ContinuousDecoder._fit_decoder` as one of these.

    What a decoder does with the boundaries is its own contract: a model with
    no notion of time may simply concatenate the segments, while one that
    estimates dynamics must not form a transition pair across two of them.

    Attributes
    ----------
    X : numpy.ndarray
        Borrowed ``(n_frames, n_features)`` feature buffer, already checked for
        shape and emptiness.
    y : neurale.data.arrays.SignalArray
        The aligned continuous target of this segment.
    schema : neurale.decoding.FeatureSchema
        Canonical schema of this segment's features.
    target_schema : neurale.decoding.ContinuousTargetSchema
        Captured metadata of this segment's target.
    clock : neurale.data.time.Clock or None
        Timeline this segment declares, or ``None`` when it declares none.
    """

    X: np.ndarray
    y: SignalArray
    schema: FeatureSchema
    target_schema: ContinuousTargetSchema
    clock: Clock | None


class BaseDecoder(abc.ABC):
    """Fitted-state, device, and reset semantics shared by every decoder.

    A decoder is unfitted until a :meth:`fit` call completes. Fit fixes two
    read-only facts -- :attr:`device_` and :attr:`feature_schema_` -- and every
    later prediction is checked against the schema *before* the model runs, so
    an incompatible input can never reach the arithmetic.

    Refitting replaces those facts. Argument errors are raised before anything
    is discarded, so a rejected call leaves a fitted decoder untouched; a
    failure inside the fit itself leaves the decoder unfitted rather than
    holding parameters that half-describe two datasets.

    :meth:`reset` is the other half of the contract: it returns runtime state
    to its post-fit condition and never touches a fitted parameter. A decoder
    that carries no runtime state resets to itself.

    A fitted decoder also belongs to one timeline, recorded as :attr:`clock_`.
    A prediction input that declares a different clock is rejected before the
    model runs, for the same reason a misaligned target is: converting between
    timelines is the caller's job. Every result is stamped with the clock of
    the timestamps it actually carries, so a result's ``time`` and ``clock``
    always describe the same domain.

    Notes
    -----
    Concrete decoders expose their own typed constructors. There is no string
    factory and no untyped parameter bag: what a decoder accepts is what its
    signature says.
    """

    #: Whether the decoder has a CUDA implementation. A requested ``cuda``
    #: fails for a decoder that has none, rather than falling back to the CPU.
    supports_cuda: ClassVar[bool] = False

    @abc.abstractmethod
    def fit(self, X: FeatureMatrix, y: Any) -> Self:
        """Fit the decoder on prepared features and an already aligned target."""

    @abc.abstractmethod
    def predict(self, X: FeatureMatrix) -> Any:
        """Decode prepared feature frames."""

    @property
    def is_fitted(self) -> bool:
        """Whether a :meth:`fit` call has completed."""

        return getattr(self, "_feature_schema", None) is not None

    @property
    def device_(self) -> ResolvedDevice:
        """Device the fitted decoder runs on, fixed by :meth:`fit`."""

        self._check_is_fitted()
        return self._device

    @property
    def feature_schema_(self) -> FeatureSchema:
        """Canonical schema of the features the decoder was fitted on."""

        self._check_is_fitted()
        return self._feature_schema

    @property
    def n_features_in_(self) -> int:
        """Number of features seen during :meth:`fit`."""

        return self.feature_schema_.n_features

    @property
    def feature_names_in_(self) -> tuple[str, ...]:
        """Feature names seen during :meth:`fit`, in column order."""

        return self.feature_schema_.feature_names

    @property
    def clock_(self) -> Clock | None:
        """Timeline the decoder was fitted to, or ``None`` if none was declared.

        Taken from the fit target when it declares a clock, otherwise from the
        fit features. The two are required to agree, so there is only ever one.
        """

        self._check_is_fitted()
        return self._fitted_clock

    def reset(self) -> None:
        """Return runtime state to its post-fit condition.

        Fitted parameters, the fitted schema, and the fitted device are never
        affected. Resetting an unfitted decoder does nothing: there is no
        post-fit condition to return to, and the hook below is entitled to read
        fitted state, so the base does not call it before there is any.
        """

        if not self.is_fitted:
            return
        self._reset_runtime_state()

    # Deliberately not abstract: a decoder that carries no runtime state resets
    # to itself, and making every stateless decoder write an empty override
    # would say nothing. Called only on a fitted decoder, so an override may
    # read fitted parameters without guarding.
    def _reset_runtime_state(self) -> None:  # noqa: B027
        """Hook for decoders that carry runtime state."""

    def _check_is_fitted(self) -> None:
        if not self.is_fitted:
            raise ValidationError(f"{type(self).__name__} instance is not fitted.")

    def _clear_fitted(self) -> None:
        # Subclasses extend this to drop whatever else fit() recorded.
        self._feature_schema = None
        self._device = None
        self._fitted_clock = None

    def _validate_features(self, X: object) -> np.ndarray:
        if not isinstance(X, FeatureMatrix):
            raise ValidationError("X must be a FeatureMatrix.")
        X = X.data
        if X.shape[0] == 0:
            raise ValidationError("X must contain at least one frame.")
        if X.shape[1] == 0:
            raise ValidationError("X must contain at least one feature.")
        return X

    def _prepare_prediction(self, X: object) -> tuple[np.ndarray, Clock | None]:
        """Validate a prediction input and return its X data and result clock.

        Runs before any model execution, so a decoder never sees frames it was
        not fitted to consume, nor frames from a timeline it was not fitted to.
        The returned clock is the one the result must carry: the input's own,
        when it declares one, and otherwise the fitted timeline.
        """

        self._check_is_fitted()
        input_X = X
        X = self._validate_features(X)
        self._feature_schema.require_compatible(
            FeatureSchema.from_feature_matrix(input_X),
            name="X",
        )
        clock = feature_clock(input_X)
        require_same_clock(
            self._fitted_clock,
            clock,
            name="X",
            reference="the fitted decoder",
        )
        return X, self._fitted_clock if clock is None else clock

    def _output_time(self, time: np.ndarray) -> np.ndarray:
        """Timestamps of the rows :meth:`_decode` returns, given the input frames'.

        The identity for a decoder that predicts one row per frame. A decoder
        that consumes several frames per prediction -- a temporal context --
        overrides this to drop exactly the frames its features lost, so that a
        result can never carry a timestamp for a row it did not predict.
        """

        return np.asarray(time, dtype=np.float64)

    @staticmethod
    def _fitted_timeline(X: FeatureMatrix, y_clock: Clock | None) -> Clock | None:
        # One timeline per fitted decoder. Alignment has already required the
        # two declarations to agree, so preferring either is the same answer;
        # the target is preferred because it is the side a result describes.
        return y_clock if y_clock is not None else feature_clock(X)

    def _resolve_device(self) -> ResolvedDevice:
        return resolve_decoder_device(
            f"{type(self).__name__}.fit",
            supports_cuda=self.supports_cuda,
        )


class ContinuousDecoder(BaseDecoder):
    """Base for decoders that reconstruct a continuous signal.

    The target is a :class:`~neurale.data.arrays.SignalArray` whose rows and
    timestamps already line up with the feature frames. Its metadata -- the
    channels, units, name, rate, and clock -- is captured at fit time and is
    what makes a prediction a metadata-complete signal rather than a bare
    array.

    A fit is expressed as a sequence of :class:`FitSegment` values, one per
    independent stretch of aligned data. :meth:`fit` is the one-segment case;
    a decoder whose model spans several recordings validates each of them with
    :meth:`_prepare_fit_segment` and records the whole fit through
    :meth:`_fit_prepared`, so every path performs the same checks in the same
    order.
    """

    def fit(self, X: FeatureMatrix, y: SignalArray) -> Self:
        """Fit the decoder on prepared features and an aligned target.

        Parameters
        ----------
        X : neurale.data.arrays.FeatureMatrix
            Prepared feature frames. The decoder extracts nothing further.
        y : neurale.data.arrays.SignalArray
            Continuous target with one row per feature frame and matching
            timestamps.

        Returns
        -------
        Self
            The fitted decoder.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the inputs are invalid or the target is not already aligned.
        neurale.exceptions.DeviceUnavailableError
            If ``cuda`` is requested and the decoder has no CUDA implementation.
        """

        return self._fit_prepared([self._prepare_fit_segment(X, y)])

    def fit_segments(self, segments: Sequence[tuple[FeatureMatrix, SignalArray]]) -> Self:
        """Fit from independent aligned stretches without crossing boundaries."""

        if isinstance(segments, str | FeatureMatrix | SignalArray) or not isinstance(
            segments, Sequence
        ):
            raise ValidationError("segments must be a sequence of (X, y) pairs.")
        if not segments:
            raise ValidationError("segments must contain at least one segment.")

        prepared: list[FitSegment] = []
        for idx, segment in enumerate(segments):
            if not isinstance(segment, tuple) or len(segment) != 2:
                raise ValidationError(f"segment {idx} must be a (X, y) pair.")
            try:
                prepared.append(self._prepare_fit_segment(*segment))
            except ValidationError as exc:
                raise ValidationError(f"segment {idx}: {exc}") from exc

        first = prepared[0]
        target_fields = ("output_names", "unit", "name", "fs")
        for idx, segment in enumerate(prepared[1:], start=1):
            first.schema.require_compatible(segment.schema, name=f"segment {idx} X")
            for field in target_fields:
                expected = getattr(first.target_schema, field)
                found = getattr(segment.target_schema, field)
                if found != expected:
                    raise ValidationError(
                        f"segment {idx} y does not match segment 0: {field} is "
                        f"{found!r}, expected {expected!r}."
                    )
            if first.target_schema.channels != segment.target_schema.channels:
                raise ValidationError(f"segment {idx} y channels do not match segment 0.")
            require_same_clock(
                first.clock,
                segment.clock,
                name=f"segment {idx}",
                reference="segment 0",
            )
        return self._fit_prepared(prepared)

    def _prepare_fit_segment(self, X: FeatureMatrix, y: SignalArray) -> FitSegment:
        """Validate one feature/target pair and capture what a fit records from it.

        Everything here runs before any fitted state is discarded, so a
        rejected pair leaves a fitted decoder untouched. A decoder with further
        per-segment preconditions extends this rather than checking them inside
        :meth:`_fit_decoder`, which keeps that guarantee.
        """

        input_X = X
        X = self._validate_features(X)
        if not isinstance(y, SignalArray):
            raise ValidationError("y must be a SignalArray.")
        require_aligned(
            input_X,
            n_rows=y.n_samples,
            time=y.time,
            clock=y.clock,
            name="y",
        )
        return FitSegment(
            X=X,
            y=y,
            schema=FeatureSchema.from_feature_matrix(input_X),
            target_schema=ContinuousTargetSchema.from_signal(y),
            clock=self._fitted_timeline(input_X, y.clock),
        )

    def _fit_prepared(self, segments: Sequence[FitSegment]) -> Self:
        """Run and record one fit over already-validated segments.

        The segments must already have been checked against each other; the
        first one supplies the schema, the target metadata, and the timeline
        the fitted decoder reports.
        """

        device = self._resolve_device()

        self._clear_fitted()
        self._fit_decoder(segments)
        self._feature_schema = segments[0].schema
        self._target_schema = segments[0].target_schema
        self._fitted_clock = segments[0].clock
        self._device = device
        return self

    def predict(self, X: FeatureMatrix) -> SignalArray:
        """Decode a continuous signal from prepared feature frames.

        The result carries the fitted target's channels, units, name, and rate,
        and the input frames' timestamps and clock: the fit says what the
        outputs are, the input says when they were predicted and on which
        timeline. A decoder that predicts from several frames at once carries
        the timestamps of the frames it actually predicted for, through
        :meth:`_output_time`.
        """

        input_X = X
        X, clock = self._prepare_prediction(X)
        predicted = self._decode(X)
        if not isinstance(predicted, np.ndarray):
            raise ValidationError(f"{type(self).__name__}._decode must return a numpy.ndarray.")
        return self._target_schema.build(predicted, self._output_time(input_X.time), clock=clock)

    @property
    def target_schema_(self) -> ContinuousTargetSchema:
        """Metadata of the continuous target the decoder was fitted on."""

        self._check_is_fitted()
        return self._target_schema

    @property
    def n_outputs_(self) -> int:
        """Number of predicted outputs."""

        return self.target_schema_.n_outputs

    def _clear_fitted(self) -> None:
        super()._clear_fitted()
        self._target_schema = None

    @abc.abstractmethod
    def _fit_decoder(self, segments: Sequence[FitSegment]) -> None:
        """Fit the decoder on validated, aligned segments.

        There is always at least one segment, and they have already been
        checked against each other, so the schema, the target metadata, and the
        timeline are the same across all of them. Implementations record their
        own parameters and return nothing; the base records the schema, the
        target metadata, and the device once this returns.
        """

    @abc.abstractmethod
    def _decode(self, X: np.ndarray) -> np.ndarray:
        """Return predicted outputs with shape ``(n_frames, n_outputs)``.

        The input has already been checked against the fitted schema.
        """


class ClassificationDecoder(BaseDecoder):
    """Base for decoders that classify feature frames.

    The target is a :class:`.ClassificationTarget` whose rows and timestamps
    already line up with the feature frames. Its class order is captured at fit
    time and becomes the column order of every score and probability the
    decoder produces.
    """

    def fit(self, X: FeatureMatrix, y: ClassificationTarget) -> Self:
        """Fit the decoder on prepared features and an aligned target.

        Parameters
        ----------
        X : neurale.data.arrays.FeatureMatrix
            Prepared feature frames. The decoder extracts nothing further.
        y : neurale.decoding.ClassificationTarget
            Labels with one entry per feature frame, matching timestamps, and
            at least two classes.

        Returns
        -------
        Self
            The fitted decoder.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the inputs are invalid or the target is not already aligned.
        neurale.exceptions.DeviceUnavailableError
            If ``cuda`` is requested and the decoder has no CUDA implementation.
        """

        input_X = X
        X = self._validate_features(X)
        if not isinstance(y, ClassificationTarget):
            raise ValidationError("y must be a ClassificationTarget.")
        if y.n_classes < 2:
            raise ValidationError("y must declare at least two classes.")
        require_aligned(
            input_X,
            n_rows=y.n_samples,
            time=y.time,
            clock=y.clock,
            name="y",
        )
        self._validate_fit_arguments(input_X, y)
        schema = FeatureSchema.from_feature_matrix(input_X)
        clock = self._fitted_timeline(input_X, y.clock)
        device = self._resolve_device()

        self._clear_fitted()
        self._fit_decoder(X, y)
        self._feature_schema = schema
        self._classes = y.classes
        self._fitted_clock = clock
        self._device = device
        return self

    def _validate_fit_arguments(
        self,
        X: FeatureMatrix,
        y: ClassificationTarget,
    ) -> None:
        """Hook for a decoder's own preconditions on a fit.

        Runs after the shared checks and before anything fitted is discarded,
        so a decoder that rejects a fit here leaves an already fitted decoder
        untouched. Deliberately not abstract: most decoders have no
        preconditions beyond the shared ones, and an empty override would say
        nothing. The continuous side has :meth:`ContinuousDecoder.
        _prepare_fit_segment` for the same purpose.
        """

    def predict(self, X: FeatureMatrix) -> ClassificationPrediction:
        """Decode classes from prepared feature frames.

        The result carries the fitted class order, and the input frames'
        timestamps and clock.
        """

        input_X = X
        X, clock = self._prepare_prediction(X)
        decoded = self._decode(X)
        if not isinstance(decoded, tuple) or len(decoded) != 3:
            raise ValidationError(
                f"{type(self).__name__}._decode must return (labels, scores, probabilities)."
            )
        labels, scores, probabilities = decoded
        return self._classification_result(
            labels,
            input_X,
            clock,
            scores=scores,
            probabilities=probabilities,
        )

    def _classification_result(
        self,
        labels: np.ndarray,
        X: FeatureMatrix,
        clock: Clock | None,
        *,
        scores: np.ndarray | None = None,
        probabilities: np.ndarray | None = None,
    ) -> ClassificationPrediction:
        """Wrap decoded labels in the fitted class order and the input frame times.

        Every typed classification result is built here, so a decoder that adds
        a second prediction entry point -- posterior probabilities, say --
        reports the same class order and the same timestamps as :meth:`predict`
        by construction rather than by repeating itself.
        """

        return ClassificationPrediction(
            labels=labels,
            classes=self._classes,
            time=self._output_time(X.time),
            scores=scores,
            probabilities=probabilities,
            clock=clock,
        )

    @property
    def classes_(self) -> np.ndarray:
        """Class order fixed by :meth:`fit`; the column order of every result."""

        self._check_is_fitted()
        return self._classes

    @property
    def n_classes_(self) -> int:
        """Number of classes seen during :meth:`fit`."""

        return int(self.classes_.shape[0])

    def _clear_fitted(self) -> None:
        super()._clear_fitted()
        self._classes = None

    @abc.abstractmethod
    def _fit_decoder(self, X: np.ndarray, y: ClassificationTarget) -> None:
        """Fit the decoder on validated, aligned inputs.

        ``X`` is the borrowed feature buffer, already checked for shape
        and emptiness, and ``y`` declares the class order the decoder must
        produce its columns in.
        """

    @abc.abstractmethod
    def _decode(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Return ``(labels, scores, probabilities)`` for validated frames.

        ``scores`` and ``probabilities`` are ``(n_frames, n_classes)`` in the
        fitted class order, or ``None`` when the decoder does not produce them.
        The base wraps the three into a :class:`.ClassificationPrediction`
        together with the fitted class order and the input frame times.
        """
