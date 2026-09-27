#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""One reader and one writer per supported decoder type.

Every decoder this format supports has an explicit codec registered under a
stable type identifier. The identifier names a *kind* of decoder and carries
its own version, so it is neither an import path nor a class name: nothing in
an artifact selects what a load will construct, and renaming a class does not
invalidate the artifacts already written.

A codec stores three things and keeps them apart. The *configuration* is what
the constructor was given, so a loaded decoder can be refitted and reproduce
the original recipe -- an owned scaler is therefore restored twice, once as an
unfitted configuration and once as the fitted, frozen copy the fit produced.
The *fitted state* is what predicting depends on. The *runtime state* is
whatever a prediction has moved since, and is stored only when the caller asks
for it: a saved decoder defaults to its post-reset condition, because that is
what a fitted decoder means outside the session that produced it.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from neurale.exceptions import ValidationError
from neurale.models.classification import LDA
from neurale.models.linear_model import LinearRegression, Ridge
from neurale.models.preprocessing import MinMaxScaler, StandardScaler
from neurale.models.state_space import KalmanFilter, LinearGaussianParameters
from neurale.runtime._types import ResolvedDevice

from .._stages import FeatureScaler
from .._validation import readonly
from ..base import BaseDecoder
from ..kalman import KalmanDecoder
from ..lda import LDADecoder
from ..linear import LinearDecoder, RidgeDecoder
from ..sequence import BeamSearchDecoder, Hypothesis, NgramLanguageModel, Vocabulary
from . import _canonical as canonical
from ._artifact import ArrayReader, ArrayWriter
from ._errors import DecoderArtifactFormatError
from ._metadata import (
    decode_clock,
    decode_feature_schema,
    decode_target_schema,
    encode_clock,
    encode_feature_schema,
    encode_target_schema,
)
from ._stages import (
    decode_context,
    decode_feature_range,
    decode_scaler,
    decode_stages,
    encode_scaler,
    encode_stages,
)


class DecoderCodec(abc.ABC):
    """The reader and writer of one decoder type."""

    #: Stable identifier of the decoder kind. Never an import path.
    type_id: str
    #: Version of *this* codec's payload layout, independent of the format version.
    type_version: int
    #: The class a load constructs, and the one a save recognizes.
    decoder_type: type
    #: Devices this codec can actually restore a fit onto. Every current codec
    #: rebuilds a CPU-only model, so ``device_`` is restricted to what the
    #: rebuilt state really is: setting the outer attribute to ``"cuda"`` over a
    #: CPU model would report a device the parameters were never on. A CUDA
    #: decoder becomes loadable when a codec restores CUDA state, not before.
    devices: tuple[str, ...] = ("cpu",)

    @abc.abstractmethod
    def write(self, decoder: Any, writer: ArrayWriter, *, runtime_state: bool) -> dict[str, Any]:
        """Return the payload section for one fitted decoder."""

    @abc.abstractmethod
    def read(self, payload: Mapping[str, Any], reader: ArrayReader) -> Any:
        """Rebuild one decoder from its payload section."""

    @property
    def has_runtime_state(self) -> bool:
        """Whether this decoder carries state a prediction moves."""

        return False


# --------------------------------------------------------- shared decoding


def _encode_common(decoder: BaseDecoder) -> dict[str, Any]:
    return {
        # The device the fit actually ran on. It is recorded, not re-resolved,
        # so a decoder loaded under a different ambient runtime still reports
        # where its parameters came from.
        "device": decoder.device_,
        "feature_schema": encode_feature_schema(decoder.feature_schema_),
        "clock": encode_clock(decoder.clock_),
    }


def _restore_common(
    decoder: BaseDecoder, payload: Mapping[str, Any], device: ResolvedDevice
) -> None:
    decoder._feature_schema = decode_feature_schema(
        canonical.require(payload, "feature_schema", path="payload"),
        path="payload.feature_schema",
    )
    decoder._fitted_clock = decode_clock(
        canonical.require(payload, "clock", path="payload"),
        path="payload.clock",
    )
    # Taken from the artifact rather than resolved again: the device is a fact
    # about the fit that produced these parameters, and the runtime the load
    # happens in has no say in it.
    decoder._device = device


def _configuration_scaler(payload: object, *, path: str) -> FeatureScaler | None:
    """Rebuild the *unfitted* scaler the decoder was configured with.

    A fitted scaler's statistics belong to the fit; its kind and options are the
    configuration. Restoring both means a loaded decoder can be refitted and
    will build the same recipe, instead of quietly dropping to unscaled
    features.
    """

    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise DecoderArtifactFormatError(f"{path} must be a JSON object or null.")
    kind = canonical.require_typed(payload, "kind", str, path=path)
    options = canonical.require_typed(payload, "options", dict, path=path)
    where = f"{path}.options"
    try:
        if kind == "standard":
            return StandardScaler(
                with_mean=canonical.require_typed(options, "with_mean", bool, path=where),
                with_std=canonical.require_typed(options, "with_std", bool, path=where),
            )
        if kind == "minmax":
            # The same validation the fitted scaler's range goes through: this
            # is one field of one artifact, and it cannot mean two things
            # depending on which half of the payload names it.
            return MinMaxScaler(feature_range=decode_feature_range(options, path=where))
    except ValidationError as exc:
        raise DecoderArtifactFormatError(
            f"{path} is not a valid scaler configuration: {exc}"
        ) from exc
    raise DecoderArtifactFormatError(f"{path}.kind is {kind!r}; this format knows a scaler kind.")


def _index_array(
    values: np.ndarray,
    name: str,
    *,
    minimum: int = 0,
    unique: bool = False,
) -> np.ndarray:
    """Validate a stored integer vector and return it as an index array.

    Converting first and checking afterwards would be the wrong order: a stored
    float array cast to an index type truncates silently, so a value that was
    never an index would become one.
    """

    if values.dtype.kind not in "iu":
        raise DecoderArtifactFormatError(
            f"{name} must hold integers; the artifact stores {values.dtype.str!r}."
        )
    if values.ndim != 1:
        raise DecoderArtifactFormatError(
            f"{name} must be 1D; the artifact stores shape {values.shape}."
        )
    if values.size and int(values.min()) < minimum:
        raise DecoderArtifactFormatError(
            f"{name} must hold values of at least {minimum}; the artifact stores "
            f"{int(values.min())}."
        )
    indices = np.asarray(values, dtype=np.intp)
    if unique and np.unique(indices).shape != indices.shape:
        raise DecoderArtifactFormatError(
            f"{name} repeats an index. Each stored index names one distinct column."
        )
    return indices


def _require_stage_contract(decoder: Any) -> None:
    """Check the fitted scaler against the width the feature schema declares.

    The scaler runs *before* the temporal context, on raw feature frames, so
    its own fitted width answers to the schema and not to the model. A model
    width check cannot see this: a scaler fitted on a different number of
    features leaves every stored shape downstream consistent, and the artifact
    restores a decoder that looks fitted and fails on the first prediction.

    Kalman is checked separately, against its observation count: there the
    scaler sees the *selected* features, not the whole schema.
    """

    scaler = decoder._stages.scaler
    if scaler is not None and scaler.n_features_in_ != decoder.n_features_in_:
        raise DecoderArtifactFormatError(
            f"the fitted scaler was fitted on {scaler.n_features_in_} features, but the stored "
            f"feature schema declares {decoder.n_features_in_}. The scaler sees raw feature "
            "frames, so an artifact where the two disagree describes no decoder that could "
            "predict."
        )


def _require_model_width(decoder: Any, n_features: int) -> None:
    """Check the model's input width against what the stages actually produce."""

    expected = decoder._stages.n_model_features(decoder.n_features_in_)
    if expected != n_features:
        raise DecoderArtifactFormatError(
            f"the stored model consumes {n_features} columns, but the stored feature schema and "
            f"stages produce {expected}. Restoring it would fail only once something predicted, "
            "on data that is not what the artifact describes."
        )


def _bounded_int(payload: Mapping[str, Any], key: str, *, path: str, minimum: int = 0) -> int:
    value = canonical.require_typed(payload, key, int, path=path)
    if value < minimum:
        raise DecoderArtifactFormatError(f"{path}.{key} must be at least {minimum}; got {value}.")
    return value


def _number(payload: Mapping[str, Any], key: str, *, path: str) -> float:
    value = canonical.optional_number(payload, key, path=path)
    if value is None:
        raise DecoderArtifactFormatError(f"{path}.{key} must be a number.")
    return value


# ------------------------------------------------------------------ kalman


class _KalmanCodec(DecoderCodec):
    type_id = "kalman"
    type_version = 1
    decoder_type = KalmanDecoder

    #: The eight parameter arrays of the fitted linear-Gaussian model.
    _PARAMETERS = (
        "transition",
        "transition_offset",
        "observation",
        "observation_offset",
        "process_covariance",
        "observation_covariance",
        "initial_state",
        "initial_covariance",
    )

    @property
    def has_runtime_state(self) -> bool:
        return True

    def write(
        self, decoder: KalmanDecoder, writer: ArrayWriter, *, runtime_state: bool
    ) -> dict[str, Any]:
        parameters = decoder.parameters_
        for name in self._PARAMETERS:
            writer.add(f"model.{name}", getattr(parameters, f"{name}_"))
        writer.add("fitted.segment_lengths", decoder.segment_lengths_)
        selection = decoder._selection
        if selection is not None:
            writer.add("fitted.selection", np.asarray(selection, dtype=np.int64))

        payload: dict[str, Any] = {
            **_encode_common(decoder),
            "target_schema": encode_target_schema(decoder.target_schema_),
            "configuration": {
                "feature_names": None
                if decoder.feature_names is None
                else list(decoder.feature_names),
                "fit_offsets": bool(decoder.fit_offsets),
                "jitter": float(decoder.jitter),
                "innovation_jitter": float(decoder.innovation_jitter),
                "missing": decoder.missing,
                "scaler": encode_scaler(decoder.scaler_, writer, "fitted.scaler"),
            },
            "fitted": {
                "selected_feature_names": list(decoder.selected_feature_names_),
                "has_selection": selection is not None,
                "n_samples": int(decoder.n_samples_),
                "n_transitions": int(decoder.n_transitions_),
            },
            "runtime": None,
        }
        if runtime_state:
            writer.add("runtime.state", decoder.state_)
            writer.add("runtime.covariance", decoder.covariance_)
            payload["runtime"] = {"at_segment_start": bool(decoder._at_segment_start)}
        return payload

    def read(self, payload: Mapping[str, Any], reader: ArrayReader) -> KalmanDecoder:
        configuration = canonical.require_typed(payload, "configuration", dict, path="payload")
        fitted = canonical.require_typed(payload, "fitted", dict, path="payload")
        where = "payload.configuration"

        names = canonical.require(configuration, "feature_names", path=where)
        # Read once and used twice: the same entry describes the unfitted
        # configuration a refit would rebuild and the fitted statistics a
        # prediction runs through.
        scaler_entry = canonical.require(configuration, "scaler", path=where)
        decoder = _construct(
            KalmanDecoder,
            "payload.configuration",
            scaler=_configuration_scaler(scaler_entry, path=f"{where}.scaler"),
            feature_names=None
            if names is None
            else canonical.text_sequence(configuration, "feature_names", path=where),
            fit_offsets=canonical.require_typed(configuration, "fit_offsets", bool, path=where),
            jitter=_number(configuration, "jitter", path=where),
            innovation_jitter=_number(configuration, "innovation_jitter", path=where),
            missing=canonical.require_typed(configuration, "missing", str, path=where),
        )

        parameters = _restore(
            LinearGaussianParameters._restore_fitted,
            "payload.model",
            **{name: reader.get(f"model.{name}") for name in self._PARAMETERS},
        )

        decoder._clear_fitted()
        decoder._scaler = decode_scaler(
            scaler_entry,
            reader,
            "fitted.scaler",
            path=f"{where}.scaler",
        )
        decoder._filter = _restore(
            KalmanFilter,
            "payload.model",
            parameters,
            jitter=decoder.innovation_jitter,
        )
        decoder._selection = (
            _index_array(reader.get("fitted.selection"), "fitted.selection", unique=True)
            if canonical.require_typed(fitted, "has_selection", bool, path="payload.fitted")
            else None
        )
        decoder._selected_feature_names = tuple(
            canonical.text_sequence(fitted, "selected_feature_names", path="payload.fitted")
        )
        decoder._segment_lengths = readonly(
            _index_array(reader.get("fitted.segment_lengths"), "fitted.segment_lengths", minimum=1)
        )
        decoder._n_samples = _bounded_int(fitted, "n_samples", path="payload.fitted", minimum=1)
        decoder._n_transitions = _bounded_int(fitted, "n_transitions", path="payload.fitted")
        decoder._at_segment_start = True

        _restore_common(decoder, payload, _device_of(payload, self))
        decoder._target_schema = decode_target_schema(
            canonical.require(payload, "target_schema", path="payload"),
            path="payload.target_schema",
        )
        self._require_contract(decoder)

        runtime = canonical.require(payload, "runtime", path="payload")
        if runtime is not None:
            if not isinstance(runtime, Mapping):
                raise DecoderArtifactFormatError("payload.runtime must be a JSON object or null.")
            _restore(
                decoder.reset,
                "payload.runtime",
                reader.get("runtime.state"),
                reader.get("runtime.covariance"),
            )
            decoder._at_segment_start = canonical.require_typed(
                runtime, "at_segment_start", bool, path="payload.runtime"
            )
        return decoder

    @staticmethod
    def _require_contract(decoder: KalmanDecoder) -> None:
        """Check that everything the observation vector passes through agrees.

        The feature *identity* is the model contract here, and it is the one
        thing a dimension check cannot see: a selection of the right length
        pointing at the wrong columns feeds the model different neural features
        under the names it reports, and every matrix still multiplies. So the
        selection is checked against the schema it indexes, not merely against
        the width the model expects.
        """

        names = decoder.feature_names_in_
        selection = decoder._selection
        selected = decoder._selected_feature_names
        if selection is not None:
            if selection.size and int(selection.max()) >= len(names):
                raise DecoderArtifactFormatError(
                    f"fitted.selection indexes column {int(selection.max())}, but the stored "
                    f"feature schema declares {len(names)} features."
                )
            expected = tuple(names[int(idx)] for idx in selection.tolist())
        else:
            expected = tuple(names)
        if selected != expected:
            raise DecoderArtifactFormatError(
                f"payload.fitted.selected_feature_names is {list(selected)}, but the stored "
                f"selection names {list(expected)} in the stored feature schema. A decoder that "
                "reported one feature and consumed another would be silently wrong rather than "
                "unloadable."
            )

        parameters = decoder.parameters_
        if len(expected) != parameters.n_observations_:
            raise DecoderArtifactFormatError(
                f"the artifact selects {len(expected)} features, but the stored model observes "
                f"{parameters.n_observations_}."
            )
        scaler = decoder.scaler_
        if scaler is not None and scaler.n_features_in_ != parameters.n_observations_:
            raise DecoderArtifactFormatError(
                f"the fitted scaler was fitted on {scaler.n_features_in_} features, but the "
                f"stored model observes {parameters.n_observations_}."
            )
        n_channels = len(decoder.target_schema_.channels)
        if n_channels != parameters.n_states_:
            raise DecoderArtifactFormatError(
                f"payload.target_schema declares {n_channels} channels, but the stored model has "
                f"a {parameters.n_states_}-dimensional state. The state is the target."
            )

        lengths = decoder.segment_lengths_
        if int(lengths.sum()) != decoder.n_samples_:
            raise DecoderArtifactFormatError(
                f"payload.fitted.n_samples is {decoder.n_samples_}, but the stored segment "
                f"lengths sum to {int(lengths.sum())}."
            )
        transitions = int(lengths.sum()) - int(lengths.shape[0])
        if transitions != decoder.n_transitions_:
            raise DecoderArtifactFormatError(
                f"payload.fitted.n_transitions is {decoder.n_transitions_}, but {lengths.shape[0]} "
                f"segments of those lengths form {transitions} within-segment pairs."
            )


# ------------------------------------------------------------- regressions


class _RegressionCodec(DecoderCodec):
    """Shared payload of the two multi-output regression decoders."""

    def write(self, decoder: Any, writer: ArrayWriter, *, runtime_state: bool) -> dict[str, Any]:
        writer.add("model.coef", decoder.coef_)
        writer.add("model.intercept", decoder.intercept_)
        writer.add("model.singular_values", decoder.singular_values_)
        return {
            **_encode_common(decoder),
            "target_schema": encode_target_schema(decoder.target_schema_),
            "configuration": {
                **self._configuration(decoder),
                "fit_intercept": bool(decoder.fit_intercept),
                **encode_stages(decoder._stages, writer, "fitted"),
            },
            "fitted": {
                "n_samples": int(decoder.n_samples_),
                "rank": int(decoder.rank_),
            },
        }

    def read(self, payload: Mapping[str, Any], reader: ArrayReader) -> Any:
        configuration = canonical.require_typed(payload, "configuration", dict, path="payload")
        fitted = canonical.require_typed(payload, "fitted", dict, path="payload")
        where = "payload.configuration"

        decoder = _construct(
            self.decoder_type,
            where,
            scaler=_configuration_scaler(
                canonical.require(configuration, "scaler", path=where),
                path=f"{where}.scaler",
            ),
            context=decode_context(
                canonical.require(configuration, "context", path=where),
                path=f"{where}.context",
            ),
            fit_intercept=canonical.require_typed(configuration, "fit_intercept", bool, path=where),
            **self._arguments(configuration, path=where),
        )

        decoder._clear_fitted()
        decoder._stages = decode_stages(configuration, reader, "fitted", path=where)
        n_samples = _bounded_int(fitted, "n_samples", path="payload.fitted", minimum=1)
        decoder._model = _restore(
            self._build_model(decoder)._restore_fitted,
            "payload.model",
            coef=reader.get("model.coef"),
            intercept=reader.get("model.intercept"),
            singular_values=reader.get("model.singular_values"),
            rank=_bounded_int(fitted, "rank", path="payload.fitted"),
            n_samples=n_samples,
        )
        decoder._n_samples = n_samples

        _restore_common(decoder, payload, _device_of(payload, self))
        decoder._target_schema = decode_target_schema(
            canonical.require(payload, "target_schema", path="payload"),
            path="payload.target_schema",
        )
        self._require_contract(decoder)
        return decoder

    @staticmethod
    def _require_contract(decoder: Any) -> None:
        model = decoder._model
        _require_stage_contract(decoder)
        _require_model_width(decoder, model.n_features_in_)
        n_channels = len(decoder.target_schema_.channels)
        if n_channels != model.n_outputs_:
            raise DecoderArtifactFormatError(
                f"payload.target_schema declares {n_channels} channels, but the stored model "
                f"predicts {model.n_outputs_} outputs."
            )
        # The design the solver factorized had n_samples rows and n_features
        # columns, so both the count of singular values and the rank read off
        # them are fixed by those two numbers. A stored rank above that is not
        # a rank of anything.
        expected = min(model.n_samples_, model.n_features_in_)
        if model.singular_values_.shape != (expected,):
            raise DecoderArtifactFormatError(
                f"the artifact stores {model.singular_values_.shape[0]} singular values, but a "
                f"{model.n_samples_}x{model.n_features_in_} design has {expected}."
            )
        if model.rank_ > expected:
            raise DecoderArtifactFormatError(
                f"payload.fitted.rank is {model.rank_}, above the {expected} the stored design "
                "can have."
            )

    def _configuration(self, decoder: Any) -> dict[str, Any]:
        return {}

    def _arguments(self, configuration: Mapping[str, Any], *, path: str) -> dict[str, Any]:
        return {}

    def _build_model(self, decoder: Any) -> Any:
        raise NotImplementedError


class _LinearCodec(_RegressionCodec):
    type_id = "linear"
    type_version = 1
    decoder_type = LinearDecoder

    def _build_model(self, decoder: LinearDecoder) -> LinearRegression:
        return LinearRegression(fit_intercept=decoder.fit_intercept)


class _RidgeCodec(_RegressionCodec):
    type_id = "ridge"
    type_version = 1
    decoder_type = RidgeDecoder

    def _configuration(self, decoder: RidgeDecoder) -> dict[str, Any]:
        return {"alpha": float(decoder.alpha)}

    def _arguments(self, configuration: Mapping[str, Any], *, path: str) -> dict[str, Any]:
        return {"alpha": _number(configuration, "alpha", path=path)}

    def _build_model(self, decoder: RidgeDecoder) -> Ridge:
        return Ridge(decoder.alpha, fit_intercept=decoder.fit_intercept)


# --------------------------------------------------------------------- lda


class _LdaCodec(DecoderCodec):
    type_id = "lda"
    type_version = 1
    decoder_type = LDADecoder

    def write(
        self, decoder: LDADecoder, writer: ArrayWriter, *, runtime_state: bool
    ) -> dict[str, Any]:
        model = decoder._model
        writer.add("model.classes", model.classes_)
        writer.add("model.coef", model.coef_)
        writer.add("model.intercept", model.intercept_)
        writer.add("model.priors", model.priors_)
        writer.add("model.means", model.means_)
        writer.add("model.covariance", model.covariance_)
        writer.add("fitted.classes", decoder.classes_)
        writer.add("fitted.order", np.asarray(decoder._order, dtype=np.int64))
        return {
            **_encode_common(decoder),
            "configuration": {
                "shrinkage": _shrinkage_entry(decoder.shrinkage),
                **encode_stages(decoder._stages, writer, "fitted"),
            },
            "fitted": {"n_samples": int(decoder.n_samples_)},
        }

    def read(self, payload: Mapping[str, Any], reader: ArrayReader) -> LDADecoder:
        configuration = canonical.require_typed(payload, "configuration", dict, path="payload")
        fitted = canonical.require_typed(payload, "fitted", dict, path="payload")
        where = "payload.configuration"

        decoder = _construct(
            LDADecoder,
            where,
            scaler=_configuration_scaler(
                canonical.require(configuration, "scaler", path=where),
                path=f"{where}.scaler",
            ),
            context=decode_context(
                canonical.require(configuration, "context", path=where),
                path=f"{where}.context",
            ),
            shrinkage=_shrinkage_value(
                canonical.require(configuration, "shrinkage", path=where), path=where
            ),
        )

        decoder._clear_fitted()
        decoder._stages = decode_stages(configuration, reader, "fitted", path=where)
        decoder._model = _restore(
            LDA(shrinkage=decoder.shrinkage)._restore_fitted,
            "payload.model",
            classes=reader.get("model.classes"),
            coef=reader.get("model.coef"),
            intercept=reader.get("model.intercept"),
            priors=reader.get("model.priors"),
            means=reader.get("model.means"),
            covariance=reader.get("model.covariance"),
        )
        decoder._order = _index_array(reader.get("fitted.order"), "fitted.order", unique=True)
        decoder._n_samples = _bounded_int(fitted, "n_samples", path="payload.fitted", minimum=1)
        decoder._classes = readonly(reader.get("fitted.classes"))
        _require_permutation(decoder._order, int(decoder._classes.shape[0]))

        _restore_common(decoder, payload, _device_of(payload, self))
        self._require_contract(decoder)
        return decoder

    @staticmethod
    def _require_contract(decoder: LDADecoder) -> None:
        model = decoder._model
        _require_stage_contract(decoder)
        _require_model_width(decoder, model.n_features_in_)
        declared = decoder._classes
        if declared.ndim != 1:
            raise DecoderArtifactFormatError(
                f"fitted.classes must be 1D; the artifact stores shape {declared.shape}."
            )
        if declared.shape[0] != model.classes_.shape[0]:
            raise DecoderArtifactFormatError(
                f"the artifact declares {declared.shape[0]} classes, but the stored discriminant "
                f"has {model.classes_.shape[0]}."
            )
        # The permutation is what maps the model's own sorted class order onto
        # the declared one, and every reported column is indexed through it. If
        # it does not reproduce the declared classes, the probabilities would be
        # labelled with classes that are not theirs.
        if model.classes_[decoder._order].tolist() != declared.tolist():
            raise DecoderArtifactFormatError(
                "the stored class permutation does not map the model's classes onto the declared "
                f"ones: it yields {model.classes_[decoder._order].tolist()} where the artifact "
                f"declares {declared.tolist()}. Every reported column is selected through it."
            )


# ---------------------------------------------------------- beam search


class _BeamSearchCodec(DecoderCodec):
    type_id = "beam-search"
    type_version = 1
    decoder_type = BeamSearchDecoder

    @property
    def has_runtime_state(self) -> bool:
        return True

    def write(
        self, decoder: BeamSearchDecoder, writer: ArrayWriter, *, runtime_state: bool
    ) -> dict[str, Any]:
        model = decoder.language_model
        vocabulary = model.fitted_vocabulary_
        writer.add("model.unigram_counts", model.unigram_counts_)
        writer.add("model.transition_counts", model.transition_counts_)

        payload: dict[str, Any] = {
            "vocabulary": _encode_vocabulary(vocabulary),
            "input_schema": {
                "width": int(vocabulary.size),
                "inputs": decoder.inputs,
                "normalization": decoder.normalization,
                "probability_floor": decoder.probability_floor,
            },
            "language_model": {
                "order": int(model.fitted_order_),
                "smoothing": float(model.fitted_smoothing_),
                "n_sequences": int(model.n_sequences_),
            },
            "configuration": {
                "beam_width": int(decoder.beam_width),
                "max_completed": int(decoder.max_completed),
                "language_weight": float(decoder.language_weight),
                "length_normalization": float(decoder.length_normalization),
            },
            "runtime": None,
        }
        if runtime_state:
            # The prior in force is *not* the language model's current one. A
            # sequence is decoded under the snapshot taken at its reset, so an
            # adaptive model refitted mid-sequence leaves the two different --
            # and resuming under the newer one would score the tail of a
            # sequence against a distribution its head was never scored on.
            # Both facts are therefore stored: the model, for the next reset,
            # and the snapshot, for the sequence that is still running.
            prior = decoder.vocabulary
            writer.add("runtime.prior.transitions", decoder.transitions)
            payload["runtime"] = {
                "n_steps": int(decoder.n_steps),
                "prior_vocabulary": _encode_vocabulary(prior),
            }
            _write_beam(writer, "runtime.active", decoder.active)
            _write_beam(writer, "runtime.completed", decoder.completed)
        return payload

    def read(self, payload: Mapping[str, Any], reader: ArrayReader) -> BeamSearchDecoder:
        vocabulary_payload = canonical.require_typed(payload, "vocabulary", dict, path="payload")
        language = canonical.require_typed(payload, "language_model", dict, path="payload")
        configuration = canonical.require_typed(payload, "configuration", dict, path="payload")
        schema = canonical.require_typed(payload, "input_schema", dict, path="payload")

        vocabulary = _decode_vocabulary(vocabulary_payload, path="payload.vocabulary")
        width = _bounded_int(schema, "width", path="payload.input_schema", minimum=1)
        if width != vocabulary.size:
            raise DecoderArtifactFormatError(
                f"payload.input_schema.width is {width}, but the stored vocabulary declares "
                f"{vocabulary.size} tokens. The score width is the vocabulary's, so the two "
                "cannot disagree."
            )

        order = _bounded_int(language, "order", path="payload.language_model", minimum=1)
        smoothing = _number(language, "smoothing", path="payload.language_model")
        model = _construct(
            NgramLanguageModel,
            "payload.language_model",
            vocabulary,
            order=order,
            smoothing=smoothing,
        )
        _restore(
            model._restore_fitted,
            "payload.language_model",
            vocabulary=vocabulary,
            order=order,
            smoothing=smoothing,
            unigram_counts=reader.get("model.unigram_counts"),
            transition_counts=reader.get("model.transition_counts"),
            n_sequences=_bounded_int(
                language, "n_sequences", path="payload.language_model", minimum=1
            ),
        )

        where = "payload.configuration"
        decoder = _construct(
            BeamSearchDecoder,
            where,
            model,
            beam_width=_bounded_int(configuration, "beam_width", path=where, minimum=1),
            max_completed=_bounded_int(configuration, "max_completed", path=where, minimum=1),
            language_weight=_number(configuration, "language_weight", path=where),
            length_normalization=_number(configuration, "length_normalization", path=where),
            inputs=canonical.require_typed(schema, "inputs", str, path="payload.input_schema"),
            normalization=canonical.require_typed(
                schema, "normalization", str, path="payload.input_schema"
            ),
            probability_floor=canonical.optional_number(
                schema, "probability_floor", path="payload.input_schema"
            ),
        )

        runtime = canonical.require(payload, "runtime", path="payload")
        if runtime is not None:
            if not isinstance(runtime, Mapping):
                raise DecoderArtifactFormatError("payload.runtime must be a JSON object or null.")
            # Construction reset the decoder onto the restored model's prior,
            # which is the right prior for the *next* sequence. The sequence
            # being resumed keeps the snapshot it started under.
            where = "payload.runtime.prior_vocabulary"
            prior = _decode_vocabulary(
                canonical.require_typed(runtime, "prior_vocabulary", dict, path="payload.runtime"),
                path=where,
            )
            decoder._vocabulary = prior
            decoder._transitions = _read_prior(reader, "runtime.prior.transitions", prior)
            n_steps = _bounded_int(runtime, "n_steps", path="payload.runtime")
            decoder._active = _read_beam(
                reader,
                "runtime.active",
                prior,
                complete=False,
                limit=decoder.beam_width,
                n_steps=n_steps,
            )
            decoder._completed = _read_beam(
                reader,
                "runtime.completed",
                prior,
                complete=True,
                limit=decoder.max_completed,
                n_steps=n_steps,
            )
            decoder._n_steps = n_steps
        return decoder


def _encode_vocabulary(vocabulary: Vocabulary) -> dict[str, Any]:
    return {
        "tokens": list(vocabulary.tokens),
        "bos": vocabulary.bos,
        "eos": vocabulary.eos,
        "unknown": vocabulary.unknown,
    }


def _decode_vocabulary(payload: Mapping[str, Any], *, path: str) -> Vocabulary:
    return _restore(
        Vocabulary,
        path,
        canonical.text_sequence(payload, "tokens", path=path),
        bos=canonical.require_typed(payload, "bos", str, path=path),
        eos=canonical.require_typed(payload, "eos", str, path=path),
        unknown=canonical.optional_text(payload, "unknown", path=path),
    )


def _read_prior(reader: ArrayReader, name: str, vocabulary: Vocabulary) -> np.ndarray:
    """Read the transition prior one active sequence is being decoded under.

    Stored as an array rather than in the manifest because a log probability of
    an unseen event is ``-inf``, which JSON cannot carry and which this format
    stores exactly rather than flooring.
    """

    values = reader.get(name)
    expected = (vocabulary.n_contexts, vocabulary.size)
    if values.dtype != np.float64:
        raise DecoderArtifactFormatError(
            f"{name} must hold float64 log probabilities; the artifact stores {values.dtype.str!r}."
        )
    if values.shape != expected:
        raise DecoderArtifactFormatError(
            f"{name} has shape {values.shape}, but the stored prior vocabulary needs {expected}."
        )
    if np.isnan(values).any() or (values > 0.0).any():
        raise DecoderArtifactFormatError(
            f"{name} must hold log probabilities: every entry is at most 0, and none is nan. "
            "A prior that is not one would make every reported language score meaningless."
        )
    return readonly(np.ascontiguousarray(values, dtype=np.float64))


def _write_beam(writer: ArrayWriter, prefix: str, beam: Sequence[Hypothesis]) -> None:
    # A beam is a ragged list of token sequences with scores that may be -inf,
    # so it is stored as arrays rather than as JSON: lengths plus a flat index
    # run, and one score row per hypothesis. JSON carries no infinity, and this
    # keeps every score exactly what the search computed.
    lengths = np.array([item.n_tokens for item in beam], dtype=np.int64)
    indices = np.array(
        [idx for item in beam for idx in item.indices],
        dtype=np.int64,
    )
    scores = np.array(
        [
            [item.model_score, item.language_score, item.score, item.normalized_score]
            for item in beam
        ],
        dtype=np.float64,
    )
    writer.add(f"{prefix}.lengths", lengths)
    writer.add(f"{prefix}.indices", indices)
    writer.add(f"{prefix}.scores", scores)


def _read_beam(
    reader: ArrayReader,
    prefix: str,
    vocabulary: Vocabulary,
    *,
    complete: bool,
    limit: int,
    n_steps: int,
) -> tuple[Hypothesis, ...]:
    """Read one stored beam and check it against the search's own invariants.

    A beam is state the decoder will keep extending, so a stored one has to
    satisfy what the search guarantees rather than merely parse: bounded size,
    one token per step for an active hypothesis, and ``eos`` exactly where a
    completion puts it. A restored active hypothesis that already ended would
    otherwise be extended past its own end, which the search itself can never
    do.
    """

    lengths = _index_array(reader.get(f"{prefix}.lengths"), f"{prefix}.lengths")
    indices = _index_array(reader.get(f"{prefix}.indices"), f"{prefix}.indices")
    scores = reader.get(f"{prefix}.scores")
    if scores.dtype != np.float64 or scores.shape != (lengths.shape[0], 4):
        raise DecoderArtifactFormatError(
            f"{prefix}.scores must be a float64 array with one row of four scores per "
            f"hypothesis; the artifact stores {scores.dtype.str!r} of shape {scores.shape}."
        )
    if np.isnan(scores).any():
        raise DecoderArtifactFormatError(f"{prefix}.scores holds a nan; a ruled-out score is -inf.")
    if int(lengths.shape[0]) > limit:
        raise DecoderArtifactFormatError(
            f"{prefix} holds {lengths.shape[0]} hypotheses, above the {limit} this decoder's "
            "configuration bounds it to."
        )
    if int(lengths.sum()) != int(indices.shape[0]):
        raise DecoderArtifactFormatError(
            f"{prefix}.indices holds {indices.shape[0]} tokens, but the lengths sum to "
            f"{int(lengths.sum())}."
        )
    if indices.size and int(indices.max()) >= vocabulary.size:
        raise DecoderArtifactFormatError(
            f"{prefix}.indices names column {int(indices.max())}, outside the "
            f"{vocabulary.size}-token prior vocabulary."
        )

    hypotheses: list[Hypothesis] = []
    offset = 0
    for pos, length in enumerate(lengths.tolist()):
        where = f"{prefix}[{pos}]"
        columns = tuple(int(value) for value in indices[offset : offset + length].tolist())
        offset += length
        _require_termination(columns, vocabulary, where, complete=complete, n_steps=n_steps)
        hypotheses.append(
            _restore(
                Hypothesis,
                where,
                tokens=_restore(vocabulary.decode, where, columns),
                indices=columns,
                model_score=float(scores[pos, 0]),
                language_score=float(scores[pos, 1]),
                score=float(scores[pos, 2]),
                normalized_score=float(scores[pos, 3]),
                is_complete=complete,
            )
        )
    return tuple(hypotheses)


def _require_termination(
    columns: tuple[int, ...],
    vocabulary: Vocabulary,
    where: str,
    *,
    complete: bool,
    n_steps: int,
) -> None:
    eos = vocabulary.eos_index
    if complete:
        if not columns or columns[-1] != eos:
            raise DecoderArtifactFormatError(
                f"{where} is stored as completed but does not end in {vocabulary.eos!r}. Nothing "
                "but an emitted eos completes a sequence."
            )
        if eos in columns[:-1]:
            raise DecoderArtifactFormatError(
                f"{where} emits {vocabulary.eos!r} before its last token; a completed hypothesis "
                "ends at the first one."
            )
        if len(columns) > n_steps:
            raise DecoderArtifactFormatError(
                f"{where} holds {len(columns)} tokens after {n_steps} steps."
            )
        return
    if eos in columns:
        raise DecoderArtifactFormatError(
            f"{where} is stored as active but already emits {vocabulary.eos!r}. Resuming would "
            "extend a sequence past its own end, which the search never does."
        )
    if len(columns) != n_steps:
        raise DecoderArtifactFormatError(
            f"{where} holds {len(columns)} tokens after {n_steps} steps; every active hypothesis "
            "is extended by exactly one token per step."
        )


# ----------------------------------------------------------------- helpers


def _shrinkage_entry(value: object) -> Any:
    if value is None or isinstance(value, str):
        return value
    return canonical.portable(float(value), path="configuration.shrinkage")


def _shrinkage_value(value: object, *, path: str) -> str | float | None:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DecoderArtifactFormatError(f"{path}.shrinkage must be a number, a string, or null.")
    return float(value)


def _device_of(payload: Mapping[str, Any], codec: DecoderCodec) -> ResolvedDevice:
    device = canonical.require_typed(payload, "device", str, path="payload")
    if device not in codec.devices:
        raise DecoderArtifactFormatError(
            f"payload.device is {device!r}, and a {codec.type_id} artifact restores a fit on "
            f"{list(codec.devices)}. The device is a fact about the parameters this artifact "
            "holds; reporting one the restored model does not run on would make device_ describe "
            "nothing."
        )
    return device  # type: ignore[return-value]


def _construct(factory: Any, path: str, *arguments: Any, **keywords: Any) -> Any:
    try:
        return factory(*arguments, **keywords)
    except ValidationError as exc:
        raise DecoderArtifactFormatError(f"{path} is not a valid configuration: {exc}") from exc


def _restore(factory: Any, path: str, *arguments: Any, **keywords: Any) -> Any:
    try:
        return factory(*arguments, **keywords)
    except ValidationError as exc:
        raise DecoderArtifactFormatError(f"{path} does not describe a valid fit: {exc}") from exc


def _require_permutation(order: np.ndarray, n_classes: int) -> None:
    if order.shape != (n_classes,) or sorted(order.tolist()) != list(range(n_classes)):
        raise DecoderArtifactFormatError(
            "the stored class permutation is not a permutation of the declared classes; the "
            "probability columns it selects would not be the ones the class order promises."
        )


CODECS: tuple[DecoderCodec, ...] = (
    _KalmanCodec(),
    _LinearCodec(),
    _RidgeCodec(),
    _LdaCodec(),
    _BeamSearchCodec(),
)

BY_ID: dict[str, DecoderCodec] = {codec.type_id: codec for codec in CODECS}
BY_TYPE: dict[type, DecoderCodec] = {codec.decoder_type: codec for codec in CODECS}


__all__ = ["BY_ID", "BY_TYPE", "CODECS", "DecoderCodec"]
