#!/usr/bin/env python3

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from neurale.data import Clock, FeatureMatrix, SignalArray
from neurale.decoding import (
    BaseDecoder,
    ClassificationDecoder,
    ClassificationPrediction,
    ClassificationTarget,
    ContinuousDecoder,
    ContinuousTargetSchema,
    FeatureSchema,
    FitSegment,
)
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.runtime import runtime_context

_RATE = 20.0
_N_FRAMES = 24


# --------------------------------------------------------------------------------------
# Stub decoders: only the arithmetic, so the shared lifecycle is what is under test
# --------------------------------------------------------------------------------------


class _LeastSquaresDecoder(ContinuousDecoder):
    """Continuous decoder with fitted parameters and a runtime counter."""

    def __init__(self, *, gain: float = 1.0) -> None:
        self.gain = gain
        self.frames_seen = 0
        self.decode_calls = 0

    def _fit_decoder(self, segments: Sequence[FitSegment]) -> None:
        X = np.concatenate([segment.X for segment in segments], axis=0)
        y = np.concatenate([segment.y.data for segment in segments], axis=0)
        self._weights = np.linalg.lstsq(X, y, rcond=None)[0]

    def _decode(self, X: np.ndarray) -> np.ndarray:
        self.decode_calls += 1
        self.frames_seen += X.shape[0]
        return self.gain * (X @ self._weights)

    def _reset_runtime_state(self) -> None:
        self.frames_seen = 0

    @property
    def weights_(self) -> np.ndarray:
        self._check_is_fitted()
        return self._weights


class _NearestMeanDecoder(ClassificationDecoder):
    """Classification decoder over per-class feature means."""

    def __init__(self) -> None:
        self.decode_calls = 0

    def _fit_decoder(self, X: np.ndarray, y: ClassificationTarget) -> None:
        self._means = np.stack(
            [X[y.class_indices == idx].mean(axis=0) for idx in range(y.n_classes)]
        )

    def _decode(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        self.decode_calls += 1
        scores = -np.linalg.norm(X[:, None, :] - self._means[None, :, :], axis=2)
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        probabilities = weights / weights.sum(axis=1, keepdims=True)
        return self.classes_[np.argmax(scores, axis=1)], scores, probabilities


class _FailingDecoder(_LeastSquaresDecoder):
    def _fit_decoder(self, segments: Sequence[FitSegment]) -> None:
        raise RuntimeError("the fit itself failed")


class _WrongShapeDecoder(_LeastSquaresDecoder):
    def _decode(self, X: np.ndarray) -> np.ndarray:
        return np.zeros((X.shape[0], 7))


class _WrongReturnDecoder(_NearestMeanDecoder):
    def _decode(self, X: np.ndarray) -> tuple:
        return (np.zeros(X.shape[0]),)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------


def _features(
    *, n_frames: int = _N_FRAMES, clock: Clock | None = None, **overrides
) -> FeatureMatrix:
    rng = np.random.default_rng(20260803)
    attrs: dict[str, object] = {"timestamp_reference": "window_center"}
    if clock is not None:
        attrs["clock"] = clock
    settings: dict[str, object] = {
        "data": rng.normal(size=(n_frames, 3)),
        "fs": _RATE,
        "feature_names": ["lmp", "beta", "gamma"],
        "unit": ["uV", "uV^2", "uV^2"],
        "source_signal": "m1_lfp",
        "window_size": 0.256,
        "shift": 1.0 / _RATE,
        "attrs": attrs,
    }
    settings.update(overrides)
    return FeatureMatrix(**settings)


def _continuous_target(features: FeatureMatrix, *, clock: Clock | None = None) -> SignalArray:
    vel = features.data @ np.array([[1.0, 0.0], [0.0, 2.0], [0.5, -0.5]])
    return SignalArray.from_array(
        vel,
        fs=_RATE,
        time=features.time.copy(),
        channel_names=["vx", "vy"],
        channel_types="behavior",
        units="m/s",
        name="hand_velocity",
        clock=clock,
        attrs={"task": "center_out"},
    )


def _classification_target(
    features: FeatureMatrix,
    *,
    clock: Clock | None = None,
    classes: list[int] | None = None,
) -> ClassificationTarget:
    labels = np.arange(features.n_frames) % 3
    return ClassificationTarget(
        labels=labels,
        time=features.time.copy(),
        classes=classes,
        clock=clock,
    )


# --------------------------------------------------------------------------------------
# Fitted state
# --------------------------------------------------------------------------------------


def test_fit_fixes_device_and_feature_schema() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    assert decoder.is_fitted
    assert decoder.device_ == "cpu"
    assert decoder.feature_schema_ == FeatureSchema.from_feature_matrix(features)
    assert decoder.n_features_in_ == 3
    assert decoder.feature_names_in_ == ("lmp", "beta", "gamma")


def test_fit_returns_decoder() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder()

    assert decoder.fit(features, _continuous_target(features)) is decoder


def test_fitted_device_and_schema_are_read_only() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    with pytest.raises(AttributeError):
        decoder.device_ = "cuda"
    with pytest.raises(AttributeError):
        decoder.feature_schema_ = FeatureSchema.from_feature_matrix(features)


@pytest.mark.parametrize(
    "attribute",
    ["device_", "feature_schema_", "n_features_in_", "feature_names_in_", "target_schema_"],
)
def test_unfitted_decoder_reports_no_fitted_state(attribute: str) -> None:
    decoder = _LeastSquaresDecoder()

    assert not decoder.is_fitted
    with pytest.raises(ValidationError, match="is not fitted"):
        getattr(decoder, attribute)


def test_predicting_before_fit_is_rejected() -> None:
    with pytest.raises(ValidationError, match="is not fitted"):
        _LeastSquaresDecoder().predict(_features())


def test_rejected_fit_argument_leaves_decoder_untouched() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))
    schema = decoder.feature_schema_

    with pytest.raises(ValidationError, match="must be a SignalArray"):
        decoder.fit(features, "hand velocity")

    assert decoder.is_fitted
    assert decoder.feature_schema_ is schema


def test_failed_fit_leaves_decoder_unfitted() -> None:
    """Half-fitted parameters describing two datasets would be worse than none."""

    features = _features()
    decoder = _FailingDecoder()

    with pytest.raises(RuntimeError, match="the fit itself failed"):
        decoder.fit(features, _continuous_target(features))

    assert not decoder.is_fitted


def test_refit_replaces_fitted_schema() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    renamed = _features(feature_names=["a", "b", "c"])
    decoder.fit(renamed, _continuous_target(renamed))

    assert decoder.feature_names_in_ == ("a", "b", "c")


def test_explicit_cuda_fails_for_cpu_only_decoder() -> None:
    features = _features()
    target = _continuous_target(features)

    with runtime_context(device="cuda"):
        with pytest.raises(
            DeviceUnavailableError,
            match=r"decoding\._LeastSquaresDecoder\.fit.*no CUDA implementation",
        ):
            _LeastSquaresDecoder().fit(features, target)


# --------------------------------------------------------------------------------------
# Alignment
# --------------------------------------------------------------------------------------


def test_target_needs_one_row_per_feature_frame() -> None:
    features = _features()
    short = _continuous_target(_features(n_frames=_N_FRAMES - 1))

    with pytest.raises(ValidationError, match="one row per feature frame"):
        _LeastSquaresDecoder().fit(features, short)


def test_shifted_target_is_rejected() -> None:
    features = _features()
    target = _continuous_target(features)
    shifted = SignalArray.from_array(
        target.data,
        fs=_RATE,
        time=target.time + 1e-6,
        channel_names=["vx", "vy"],
        channel_types="behavior",
        units="m/s",
        name="hand_velocity",
    )

    with pytest.raises(ValidationError, match="no resampling or time shifting"):
        _LeastSquaresDecoder().fit(features, shifted)


def test_timestamps_within_few_ulps_still_align() -> None:
    """Equivalent arithmetic on the same grid does not have to produce identical bits."""

    features = _features()
    target = _continuous_target(features)
    jittered = SignalArray.from_array(
        target.data,
        fs=_RATE,
        time=np.nextafter(target.time, np.inf),
        channel_names=["vx", "vy"],
        channel_types="behavior",
        units="m/s",
        name="hand_velocity",
    )

    assert _LeastSquaresDecoder().fit(features, jittered).is_fitted


def test_classification_target_must_align() -> None:
    features = _features()
    target = ClassificationTarget(
        labels=np.arange(features.n_frames) % 3,
        time=features.time + 0.5,
    )

    with pytest.raises(ValidationError, match="timestamps must match"):
        _NearestMeanDecoder().fit(features, target)


def test_disagreeing_declared_clocks_are_rejected() -> None:
    features = _features(clock=Clock(name="neural", type="acquisition"))
    target = _continuous_target(features, clock=Clock(name="task", type="host"))

    with pytest.raises(ValidationError, match="no clock conversion"):
        _LeastSquaresDecoder().fit(features, target)


def test_agreeing_declared_clocks_are_accepted() -> None:
    clock = Clock(name="neural", type="acquisition", offset=0.25)
    features = _features(clock=clock)
    target = _continuous_target(
        features, clock=Clock(name="neural", type="acquisition", offset=0.25)
    )

    assert _LeastSquaresDecoder().fit(features, target).is_fitted


def test_same_clock_name_with_new_offset_is_new_timeline() -> None:
    features = _features(clock=Clock(name="neural", type="acquisition", offset=0.0))
    target = _continuous_target(
        features, clock=Clock(name="neural", type="acquisition", offset=1.0)
    )

    with pytest.raises(ValidationError, match="no clock conversion"):
        _LeastSquaresDecoder().fit(features, target)


def test_undeclared_clock_is_not_checked() -> None:
    """Absent metadata is not evidence of a conflict."""

    features = _features()
    target = _continuous_target(features, clock=Clock(name="task", type="host"))

    assert _LeastSquaresDecoder().fit(features, target).is_fitted


def test_wrongly_typed_clock_is_metadata_error() -> None:
    features = _features(attrs={"timestamp_reference": "window_center", "clock": "neural"})

    with pytest.raises(ValidationError, match=r"attrs\['clock'\] must be a Clock"):
        _LeastSquaresDecoder().fit(features, _continuous_target(features))


# --------------------------------------------------------------------------------------
# The schema gate
# --------------------------------------------------------------------------------------


def test_incompatible_schema_fails_before_model_runs() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))
    calls = decoder.decode_calls

    with pytest.raises(ValidationError, match="do not match the fitted feature schema"):
        decoder.predict(_features(feature_names=["lmp", "beta", "delta"]))

    assert decoder.decode_calls == calls


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"feature_names": ["lmp", "gamma", "beta"]}, id="reordered"),
        pytest.param({"unit": "uV"}, id="units"),
        pytest.param({"fs": 25.0, "shift": 0.04}, id="observation-rate"),
        pytest.param({"window_size": 0.5}, id="window"),
        pytest.param({"attrs": {"timestamp_reference": "window_end"}}, id="timestamp-reference"),
        pytest.param({"source_signal": "pmd_lfp"}, id="source"),
    ],
)
def test_prediction_input_matches_compatibility_fields(overrides: dict) -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    with pytest.raises(ValidationError, match="do not match the fitted feature schema"):
        decoder.predict(_features(**overrides))


def test_prediction_input_may_differ_in_length_and_start() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    prediction = decoder.predict(_features(n_frames=5, t0=931.0))

    assert prediction.n_samples == 5


def test_prediction_input_must_be_feature_matrix() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    with pytest.raises(ValidationError, match="must be a FeatureMatrix"):
        decoder.predict(features.data)


def test_empty_input_is_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one frame"):
        _LeastSquaresDecoder().fit(
            _features(data=np.zeros((0, 3))),
            _continuous_target(_features()),
        )


# --------------------------------------------------------------------------------------
# Continuous prediction
# --------------------------------------------------------------------------------------


def test_continuous_prediction_is_metadata_complete() -> None:
    features = _features()
    target = _continuous_target(features)
    decoder = _LeastSquaresDecoder().fit(features, target)

    prediction = decoder.predict(features)

    assert isinstance(prediction, SignalArray)
    assert prediction.name == target.name
    assert prediction.channel_names == ["vx", "vy"]
    assert prediction.unit == "m/s"
    assert prediction.fs == target.fs
    assert dict(prediction.attrs) == {"task": "center_out"}
    assert np.allclose(prediction.data, target.data)


def test_continuous_prediction_is_timed_by_input() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    later = _features(n_frames=6, t0=931.0)
    prediction = decoder.predict(later)

    assert np.array_equal(prediction.time, later.time)
    assert prediction.t0 == pytest.approx(931.0)


def test_continuous_prediction_carries_fitted_target_clock() -> None:
    clock = Clock(name="task", type="host")
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features, clock=clock))

    assert decoder.predict(features).clock == clock


def test_fitted_target_schema_describes_outputs() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    assert isinstance(decoder.target_schema_, ContinuousTargetSchema)
    assert decoder.target_schema_.output_names == ["vx", "vy"]
    assert decoder.n_outputs_ == 2


def test_wrong_output_count_is_caught() -> None:
    features = _features()
    decoder = _WrongShapeDecoder().fit(features, _continuous_target(features))

    with pytest.raises(ValidationError, match="must have 2 outputs; got 7"):
        decoder.predict(features)


# --------------------------------------------------------------------------------------
# Classification prediction
# --------------------------------------------------------------------------------------


def test_classification_prediction_carries_classes_and_scores() -> None:
    features = _features()
    target = _classification_target(features)
    decoder = _NearestMeanDecoder().fit(features, target)

    prediction = decoder.predict(features)

    assert isinstance(prediction, ClassificationPrediction)
    assert np.array_equal(prediction.classes, target.classes)
    assert np.array_equal(prediction.time, features.time)
    assert prediction.scores.shape == (features.n_frames, 3)
    assert prediction.probabilities.shape == (features.n_frames, 3)
    assert set(prediction.labels.tolist()) <= set(target.classes.tolist())


def test_fitted_class_order_is_result_column_order() -> None:
    features = _features()
    reversed_order = _classification_target(features, classes=[2, 1, 0])
    decoder = _NearestMeanDecoder().fit(features, reversed_order)

    assert np.array_equal(decoder.classes_, np.array([2, 1, 0]))
    assert decoder.n_classes_ == 3
    assert np.array_equal(decoder.predict(features).classes, np.array([2, 1, 0]))


def test_classification_prediction_carries_fitted_clock() -> None:
    clock = Clock(name="task", type="host")
    features = _features()
    decoder = _NearestMeanDecoder().fit(features, _classification_target(features, clock=clock))

    assert decoder.predict(features).clock == clock


def test_classification_target_needs_two_classes() -> None:
    features = _features()
    single = ClassificationTarget(
        labels=np.zeros(features.n_frames, dtype=np.int64),
        time=features.time.copy(),
    )

    with pytest.raises(ValidationError, match="at least two classes"):
        _NearestMeanDecoder().fit(features, single)


def test_classification_target_type_is_enforced() -> None:
    features = _features()

    with pytest.raises(ValidationError, match="must be a ClassificationTarget"):
        _NearestMeanDecoder().fit(features, np.arange(features.n_frames) % 3)


def test_wrong_result_shape_is_caught() -> None:
    features = _features()
    decoder = _WrongReturnDecoder().fit(features, _classification_target(features))

    with pytest.raises(ValidationError, match=r"labels, scores, probabilities"):
        decoder.predict(features)


# --------------------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------------------


def test_reset_clears_runtime_state_keeps_parameters() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))
    before = decoder.predict(features)
    assert decoder.frames_seen == features.n_frames

    decoder.reset()

    assert decoder.frames_seen == 0
    assert decoder.is_fitted
    assert decoder.device_ == "cpu"
    assert decoder.feature_schema_ == FeatureSchema.from_feature_matrix(features)
    assert np.array_equal(decoder.predict(features).data, before.data)


def test_reset_on_unfitted_decoder_is_allowed() -> None:
    decoder = _LeastSquaresDecoder()

    decoder.reset()

    assert not decoder.is_fitted


def test_stateless_decoder_resets_to_itself() -> None:
    features = _features()
    decoder = _NearestMeanDecoder().fit(features, _classification_target(features))
    before = decoder.predict(features)

    decoder.reset()

    assert np.array_equal(decoder.predict(features).labels, before.labels)


# --------------------------------------------------------------------------------------
# Typed constructors
# --------------------------------------------------------------------------------------


def test_decoder_keeps_own_typed_constructor() -> None:
    """The contract adds no string factory and no untyped parameter bag."""

    features = _features()
    decoder = _LeastSquaresDecoder(gain=2.0).fit(features, _continuous_target(features))

    assert decoder.gain == 2.0
    assert isinstance(decoder, BaseDecoder)
    assert np.allclose(decoder.predict(features).data, 2.0 * _continuous_target(features).data)


def test_bases_are_abstract() -> None:
    for base in (BaseDecoder, ContinuousDecoder, ClassificationDecoder):
        with pytest.raises(TypeError):
            base()


# --------------------------------------------------------------------------------------
# The prediction-time clock gate
# --------------------------------------------------------------------------------------


def test_fitted_timeline_is_recorded() -> None:
    clock = Clock(name="task", type="host")
    features = _features()

    assert _LeastSquaresDecoder().fit(features, _continuous_target(features)).clock_ is None
    assert (
        _LeastSquaresDecoder().fit(features, _continuous_target(features, clock=clock)).clock_
        == clock
    )
    with_features = _features(clock=clock)
    assert _LeastSquaresDecoder().fit(with_features, _continuous_target(with_features)).clock_ == (
        clock
    )


def test_foreign_timeline_fails_before_model_runs() -> None:
    """A result stamped with one clock and timed by another describes neither."""

    features = _features(clock=Clock(name="neuralA", type="acquisition"))
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))
    calls = decoder.decode_calls

    with pytest.raises(ValidationError, match="no clock conversion"):
        decoder.predict(_features(clock=Clock(name="neuralB", type="acquisition")))

    assert decoder.decode_calls == calls


def test_classification_decoder_gates_prediction_timeline() -> None:
    features = _features()
    target = _classification_target(features, clock=Clock(name="task", type="host"))
    decoder = _NearestMeanDecoder().fit(features, target)
    calls = decoder.decode_calls

    with pytest.raises(ValidationError, match="no clock conversion"):
        decoder.predict(_features(clock=Clock(name="neuralB", type="acquisition")))

    assert decoder.decode_calls == calls


def test_target_clock_gates_prediction_without_feature_clock() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(
        features,
        _continuous_target(features, clock=Clock(name="task", type="host")),
    )

    with pytest.raises(ValidationError, match="no clock conversion"):
        decoder.predict(_features(clock=Clock(name="neuralB", type="acquisition")))


def test_continuous_result_time_and_clock_share_domain() -> None:
    clock = Clock(name="task", type="host")
    features = _features(clock=clock)
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features, clock=clock))

    prediction = decoder.predict(_features(clock=clock, n_frames=6, t0=931.0))

    assert prediction.clock == clock
    assert np.array_equal(prediction.time, _features(n_frames=6, t0=931.0).time)


def test_classification_result_time_and_clock_share_domain() -> None:
    clock = Clock(name="task", type="host")
    features = _features(clock=clock)
    decoder = _NearestMeanDecoder().fit(features, _classification_target(features, clock=clock))

    later = _features(clock=clock, n_frames=6, t0=931.0)
    prediction = decoder.predict(later)

    assert prediction.clock == clock
    assert np.array_equal(prediction.time, later.time)


def test_input_declaring_new_timeline_stamps_own() -> None:
    """The clock describes the time axis, and the time axis is the input's."""

    clock = Clock(name="neural", type="acquisition")
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    assert decoder.predict(_features(clock=clock)).clock == clock


def test_input_without_timeline_keeps_fitted_one() -> None:
    clock = Clock(name="task", type="host")
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features, clock=clock))

    assert decoder.predict(features).clock == clock


# --------------------------------------------------------------------------------------
# Epoch-scale timestamps
# --------------------------------------------------------------------------------------

_EPOCH = 1_700_000_000.0
_ULP = float(np.spacing(_EPOCH))


def _epoch_features(shift: float = 0.0) -> FeatureMatrix:
    # An epoch-scale grid cannot declare a sampling rate: float64 cannot place
    # 1.7e9 + k/20 on an exact grid, so the frames are irregularly timed.
    time = _EPOCH + np.arange(_N_FRAMES, dtype=np.float64) * 0.05 + shift
    return _features(fs=None, time=time)


def _epoch_target(features: FeatureMatrix) -> SignalArray:
    return SignalArray.from_array(
        features.data @ np.array([[1.0, 0.0], [0.0, 2.0], [0.5, -0.5]]),
        fs=_RATE,
        time=_epoch_features().time,
        channel_names=["vx", "vy"],
        channel_types="behavior",
        units="m/s",
        name="hand_velocity",
    )


def test_epoch_scale_timestamps_within_one_ulp_align() -> None:
    nudged = _features(
        fs=None,
        time=np.nextafter(_epoch_features().time, np.inf),
    )

    assert _LeastSquaresDecoder().fit(nudged, _epoch_target(nudged)).is_fitted


@pytest.mark.parametrize(
    "shift",
    [
        pytest.param(1e-3, id="1ms"),
        pytest.param(1e-4, id="100us"),
        pytest.param(1e-5, id="10us"),
    ],
)
def test_sub_bin_shift_is_rejected_at_epoch_scale(shift: float) -> None:
    """A relative tolerance would have turned 1.7e9 seconds into millisecond slack."""

    features = _epoch_features(shift=shift)

    with pytest.raises(ValidationError, match="timestamps must match"):
        _LeastSquaresDecoder().fit(features, _epoch_target(features))


def test_epoch_scale_slack_follows_representation() -> None:
    """Eight ULPs of 1.7e9 s is a couple of microseconds; float64 has nothing finer."""

    assert _ULP == pytest.approx(2.4e-7, rel=0.1)
    assert 8 * _ULP < 2e-6


# --------------------------------------------------------------------------------------
# Reset before fit
# --------------------------------------------------------------------------------------


class _FittedStateResetDecoder(_LeastSquaresDecoder):
    """A decoder whose reset hook reads fitted state, as a Kalman decoder would."""

    def _reset_runtime_state(self) -> None:
        self.frames_seen = 0
        self.state = np.zeros(self.n_outputs_)


def test_reset_on_unfitted_stateful_decoder_is_no_op() -> None:
    """The hook may read fitted state, so the base does not call it before there is any."""

    decoder = _FittedStateResetDecoder()

    decoder.reset()

    assert not decoder.is_fitted
    assert not hasattr(decoder, "state")


def test_reset_on_fitted_stateful_decoder_runs_hook() -> None:
    features = _features()
    decoder = _FittedStateResetDecoder().fit(features, _continuous_target(features))
    decoder.predict(features)

    decoder.reset()

    assert decoder.frames_seen == 0
    assert np.array_equal(decoder.state, np.zeros(2))
    assert decoder.is_fitted


# --------------------------------------------------------------------------------------
# Fitted metadata immutability
# --------------------------------------------------------------------------------------


def test_fitted_target_metadata_cannot_be_edited() -> None:
    features = _features()
    decoder = _LeastSquaresDecoder().fit(features, _continuous_target(features))

    with pytest.raises(TypeError):
        decoder.target_schema_.attrs["task"] = "other"
    assert dict(decoder.predict(features).attrs) == {"task": "center_out"}
