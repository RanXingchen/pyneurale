#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import Clock, FeatureMatrix, SignalArray
from neurale.decoding import ContinuousDecoder, KalmanDecoder
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import MinMaxScaler, StandardScaler
from neurale.models.state_space import KalmanFilter, LinearGaussianStateSpace
from neurale.runtime import runtime_context

_RATE = 50.0
_N_FEATURES = 6
_OUTPUT_NAMES = ["px", "py", "vx", "vy"]
_OUTPUT_UNITS = ["m", "m", "m/s", "m/s"]
_N_OUTPUTS = len(_OUTPUT_NAMES)
_FEATURE_NAMES = [f"rate_{idx}" for idx in range(_N_FEATURES)]

# A constant-velocity point mass with damping, observed through a fixed random
# mixing matrix. Both are known here, so a fit can be checked against the
# parameters the data actually came from rather than against itself.
_TRANSITION = np.array(
    [
        [1.0, 0.0, 1.0 / _RATE, 0.0],
        [0.0, 1.0, 0.0, 1.0 / _RATE],
        [0.0, 0.0, 0.92, 0.0],
        [0.0, 0.0, 0.0, 0.92],
    ]
)
_OBSERVATION = np.random.default_rng(11).normal(size=(_N_FEATURES, _N_OUTPUTS))
_PROCESS_SCALE = np.array([1e-4, 1e-4, 0.05, 0.05])
_OBSERVATION_SCALE = 0.05


# --------------------------------------------------------------------------------------
# Synthetic data with known state-space parameters
# --------------------------------------------------------------------------------------


def _simulate(n_frames: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.zeros((n_frames, _N_OUTPUTS))
    states[0] = rng.normal(scale=0.1, size=_N_OUTPUTS)
    for step in range(1, n_frames):
        states[step] = _TRANSITION @ states[step - 1] + rng.normal(scale=_PROCESS_SCALE)
    observations = states @ _OBSERVATION.T + rng.normal(
        scale=_OBSERVATION_SCALE, size=(n_frames, _N_FEATURES)
    )
    return states, observations


def _features(
    observations: np.ndarray,
    *,
    t0: float = 0.0,
    clock: Clock | None = None,
    **overrides: object,
) -> FeatureMatrix:
    attrs: dict[str, object] = {"timestamp_reference": "window_center"}
    if clock is not None:
        attrs["clock"] = clock
    settings: dict[str, object] = {
        "data": observations,
        "fs": _RATE,
        "t0": t0,
        "feature_names": list(_FEATURE_NAMES),
        "unit": "spikes/s",
        "source_signal": "m1_units",
        "window_size": 0.05,
        "shift": 1.0 / _RATE,
        "attrs": attrs,
    }
    settings.update(overrides)
    return FeatureMatrix(**settings)


def _target(
    states: np.ndarray,
    features: FeatureMatrix,
    *,
    clock: Clock | None = None,
    **overrides: object,
) -> SignalArray:
    settings: dict[str, object] = {
        "fs": _RATE,
        "time": features.time.copy(),
        "channel_names": list(_OUTPUT_NAMES),
        "channel_types": "behavior",
        "units": list(_OUTPUT_UNITS),
        "name": "cursor",
        "clock": clock,
        "attrs": {"task": "center_out"},
    }
    settings.update(overrides)
    return SignalArray.from_array(states, **settings)


def _segment(
    n_frames: int = 400,
    *,
    seed: int = 1,
    t0: float = 0.0,
    clock: Clock | None = None,
) -> tuple[FeatureMatrix, SignalArray]:
    states, observations = _simulate(n_frames, seed=seed)
    features = _features(observations, t0=t0, clock=clock)
    return features, _target(states, features, clock=clock)


def _frames(features: FeatureMatrix, start: int, stop: int) -> FeatureMatrix:
    """Re-wrap a row range as an independently constructed feature matrix."""

    return _features(features.data[start:stop], t0=float(features.time[start]))


def _gappy_pair(n_frames: int = 200, *, gap: float = 1.0, **overrides: object):
    """A single segment whose own frames are interrupted by a gap.

    The features must declare no rate, because a ``FeatureMatrix`` cross-checks
    its timestamps against one; the target may still declare it, which is what
    lets an irregular pair exist at all.
    """

    states, observations = _simulate(n_frames, seed=4)
    time = np.arange(n_frames, dtype=np.float64) / _RATE
    time[n_frames // 2 :] += gap
    features = _features(observations, fs=None, time=time, **overrides)
    return features, _target(states, features)


# --------------------------------------------------------------------------------------
# Constructor
# --------------------------------------------------------------------------------------


def test_decoder_is_continuous_decoder() -> None:
    assert issubclass(KalmanDecoder, ContinuousDecoder)
    assert KalmanDecoder.supports_cuda is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"scaler": "standard"}, "scaler must be"),
        ({"scaler": object()}, "scaler must be"),
        ({"feature_names": "rate_0"}, "feature_names must be a sequence"),
        ({"feature_names": []}, "at least one feature"),
        ({"feature_names": ["rate_0", ""]}, "non-empty strings"),
        ({"feature_names": ["rate_0", 3]}, "non-empty strings"),
        ({"feature_names": ["rate_0", "rate_0"]}, "must be unique"),
        ({"fit_offsets": 1}, "fit_offsets must be a bool"),
        ({"jitter": -1.0}, "jitter"),
        ({"innovation_jitter": -1.0}, "innovation_jitter"),
        ({"missing": "interpolate"}, "missing must be one of"),
    ],
)
def test_invalid_arguments_are_rejected(arguments: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        KalmanDecoder(**arguments)


def test_defaults_consume_every_feature_unscaled() -> None:
    features, target = _segment(200)
    decoder = KalmanDecoder().fit(features, target)

    assert decoder.scaler_ is None
    assert decoder.selected_feature_names_ == tuple(_FEATURE_NAMES)
    assert decoder.n_observations_ == _N_FEATURES


# --------------------------------------------------------------------------------------
# Recovering a known state-space model
# --------------------------------------------------------------------------------------


def test_simulated_parameters_are_recovered() -> None:
    features, target = _segment(4000, seed=2)
    decoder = KalmanDecoder().fit(features, target)

    assert decoder.parameters_.n_states_ == _N_OUTPUTS
    assert decoder.parameters_.n_observations_ == _N_FEATURES
    assert np.allclose(decoder.parameters_.transition_, _TRANSITION, atol=0.02)
    assert np.allclose(decoder.parameters_.observation_, _OBSERVATION, atol=0.02)
    assert np.allclose(decoder.parameters_.transition_offset_, 0.0, atol=0.02)


def test_held_out_recording_decodes_near_true_state() -> None:
    train_features, train_target = _segment(3000, seed=3)
    test_features, test_target = _segment(1000, seed=17)

    decoder = KalmanDecoder(scaler=StandardScaler()).fit(train_features, train_target)
    predicted = decoder.predict(test_features)

    error = np.sqrt(np.mean(np.square(predicted.data - test_target.data), axis=0))
    spread = np.std(test_target.data, axis=0)
    correlation = [
        np.corrcoef(predicted.data[:, output], test_target.data[:, output])[0, 1]
        for output in range(_N_OUTPUTS)
    ]

    assert np.all(error < 0.5 * spread)
    assert np.all(np.array(correlation) > 0.9)


def test_prediction_is_metadata_complete_signal() -> None:
    clock = Clock(name="neural", type="device", rate=30_000.0)
    features, target = _segment(300, clock=clock)
    decoder = KalmanDecoder().fit(features, target)

    predicted = decoder.predict(features)

    assert isinstance(predicted, SignalArray)
    assert predicted.channels.names == _OUTPUT_NAMES
    assert predicted.channels.units == _OUTPUT_UNITS
    assert predicted.unit == tuple(_OUTPUT_UNITS)
    assert predicted.name == "cursor"
    assert predicted.fs == _RATE
    assert dict(predicted.attrs) == {"task": "center_out"}
    assert predicted.clock == clock
    assert np.array_equal(predicted.time, features.time)
    assert predicted.data.shape == (300, _N_OUTPUTS)


# --------------------------------------------------------------------------------------
# The reported model is the running model
# --------------------------------------------------------------------------------------


def test_reported_parameters_are_running_snapshot() -> None:
    features, target = _segment(300, seed=2)
    decoder = KalmanDecoder().fit(features, target)

    # Not "equal to": the same object, so no reading can drift from the model
    # predict() executes.
    assert decoder.parameters_ is decoder._filter.parameters
    assert decoder.parameters_.n_observations_ == _N_FEATURES
    assert decoder.parameters_.n_states_ == _N_OUTPUTS


def test_fitted_decoder_exposes_no_refittable_estimator() -> None:
    features, target = _segment(300, seed=2)
    decoder = KalmanDecoder().fit(features, target)

    # A handle onto the live estimator would let a caller refit it behind the
    # filter, leaving parameters_ describing one model and predict() running
    # another. The fit keeps the snapshot and lets the estimator go.
    assert not hasattr(decoder, "state_space_")


def test_refit_replaces_running_snapshot() -> None:
    first_features, first_target = _segment(300, seed=2)
    second_features, second_target = _segment(300, seed=19)
    decoder = KalmanDecoder().fit(first_features, first_target)
    first = decoder.parameters_

    decoder.fit(second_features, second_target)

    assert decoder.parameters_ is not first
    assert decoder.parameters_ is decoder._filter.parameters
    assert not np.array_equal(decoder.parameters_.transition_, first.transition_)


def test_fit_statistics_describe_producing_fit() -> None:
    decoder = KalmanDecoder().fit_segments([_segment(300, seed=8), _segment(120, seed=9, t0=100.0)])

    assert decoder.n_samples_ == 420
    assert decoder.n_segments_ == 2
    assert decoder.n_transitions_ == 418
    assert decoder.n_observations_ == decoder.parameters_.n_observations_
    assert decoder.n_outputs_ == decoder.parameters_.n_states_


@pytest.mark.parametrize(
    "attribute",
    ["parameters_", "n_samples_", "n_transitions_", "selected_feature_names_"],
)
def test_fitted_results_require_fit(attribute) -> None:
    decoder = KalmanDecoder()

    with pytest.raises(ValidationError, match="not fitted"):
        getattr(decoder, attribute)


# --------------------------------------------------------------------------------------
# Whole-batch versus chunked prediction
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk", [1, 7, 64, 400])
def test_chunked_prediction_matches_whole_batch_prediction(chunk: int) -> None:
    features, target = _segment(400, seed=5)
    decoder = KalmanDecoder(scaler=StandardScaler()).fit(features, target)

    whole = decoder.predict(features).data
    decoder.reset()
    chunks = [
        decoder.predict(_frames(features, start, min(start + chunk, 400))).data
        for start in range(0, 400, chunk)
    ]

    assert np.array_equal(np.concatenate(chunks, axis=0), whole)


def test_chunk_boundary_is_not_segment_boundary() -> None:
    """Continuing the stream is a predict/update, not a restart from the fit."""

    features, target = _segment(200, seed=6)
    decoder = KalmanDecoder().fit(features, target)

    decoder.predict(_frames(features, 0, 100))
    continued = decoder.predict(_frames(features, 100, 200)).data

    decoder.reset()
    restarted = decoder.predict(_frames(features, 100, 200)).data

    assert not np.allclose(continued, restarted)


# --------------------------------------------------------------------------------------
# Segments
# --------------------------------------------------------------------------------------


def test_fit_is_single_segment_fit_segments() -> None:
    features, target = _segment(300, seed=7)

    single = KalmanDecoder(scaler=StandardScaler()).fit(features, target)
    listed = KalmanDecoder(scaler=StandardScaler()).fit_segments([(features, target)])

    assert single.n_segments_ == listed.n_segments_ == 1
    assert np.array_equal(single.segment_lengths_, listed.segment_lengths_)
    assert np.array_equal(single.parameters_.transition_, listed.parameters_.transition_)
    assert np.array_equal(single.parameters_.observation_, listed.parameters_.observation_)
    assert np.array_equal(single.scaler_.mean_, listed.scaler_.mean_)
    assert np.array_equal(single.predict(features).data, listed.predict(features).data)


def test_segments_are_recorded_in_given_order() -> None:
    first = _segment(300, seed=8)
    second = _segment(120, seed=9, t0=100.0)

    decoder = KalmanDecoder().fit_segments([first, second])

    assert decoder.n_segments_ == 2
    assert np.array_equal(decoder.segment_lengths_, np.array([300, 120]))
    assert not decoder.segment_lengths_.flags.writeable
    assert decoder.n_samples_ == 420


def test_no_transition_crosses_segment_boundary() -> None:
    first = _segment(300, seed=8)
    second = _segment(300, seed=9, t0=100.0)
    decoder = KalmanDecoder().fit_segments([first, second])

    states = np.concatenate([first[1].data, second[1].data], axis=0)
    observations = np.concatenate([first[0].data, second[0].data], axis=0)
    respecting = LinearGaussianStateSpace().fit(
        states, observations, segment_lengths=np.array([300, 300])
    )
    ignoring = LinearGaussianStateSpace().fit(states, observations)

    # 598 within-segment pairs, not the 599 a concatenated sequence would give.
    assert decoder.n_transitions_ == 598
    assert decoder.n_transitions_ == decoder.n_samples_ - decoder.n_segments_
    assert ignoring.n_transitions_ == 599
    assert np.array_equal(decoder.parameters_.transition_, respecting.transition_)
    assert not np.allclose(decoder.parameters_.transition_, ignoring.transition_)


def test_initial_state_comes_from_segment_starts() -> None:
    first = _segment(300, seed=8)
    second = _segment(300, seed=9, t0=100.0)
    decoder = KalmanDecoder().fit_segments([first, second])

    expected = np.mean([first[1].data[0], second[1].data[0]], axis=0)
    assert np.allclose(decoder.initial_state_, expected)
    # The initial state comes from segment starts, not the final training sample.
    assert not np.allclose(decoder.initial_state_, second[1].data[-1])


def test_gap_between_segments_is_not_internal_discontinuity() -> None:
    first = _segment(200, seed=8)
    second = _segment(200, seed=9, t0=3600.0)

    decoder = KalmanDecoder().fit_segments([first, second])

    assert decoder.n_segments_ == 2


@pytest.mark.parametrize(
    ("segments", "message"),
    [
        ("not a sequence", "sequence of"),
        (42, "sequence of"),
        ([], "at least one segment"),
        ([("features", "target")], "segment 0: X must be a FeatureMatrix"),
    ],
)
def test_invalid_segment_arguments_are_rejected(segments: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        KalmanDecoder().fit_segments(segments)


def test_non_pair_segment_is_rejected() -> None:
    features, target = _segment(50)

    with pytest.raises(ValidationError, match="segment 1 must be a"):
        KalmanDecoder().fit_segments([(features, target), (features, target, features)])


def test_segment_with_different_features_is_rejected() -> None:
    first = _segment(200, seed=8)
    states, observations = _simulate(200, seed=9)
    renamed = _features(observations, feature_names=[f"unit_{i}" for i in range(_N_FEATURES)])

    with pytest.raises(ValidationError, match="segment 1 X do not match"):
        KalmanDecoder().fit_segments([first, (renamed, _target(states, renamed))])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "other"}, "segment 1 y does not match segment 0: name"),
        ({"units": "cm"}, "segment 1 y does not match segment 0"),
        ({"channel_names": ["a", "b", "c", "d"]}, "output_names"),
    ],
)
def test_segment_with_different_target_is_rejected(overrides: dict, message: str) -> None:
    first = _segment(200, seed=8)
    states, observations = _simulate(200, seed=9)
    features = _features(observations)

    with pytest.raises(ValidationError, match=message):
        KalmanDecoder().fit_segments([first, (features, _target(states, features, **overrides))])


def test_segment_on_another_timeline_is_rejected() -> None:
    first = _segment(200, seed=8, clock=Clock(name="neuralA", type="device", rate=30_000.0))
    second = _segment(200, seed=9, clock=Clock(name="neuralB", type="device", rate=30_000.0))

    with pytest.raises(ValidationError, match="clock mismatch: segment 1"):
        KalmanDecoder().fit_segments([first, second])


def test_first_segment_supplies_schema_and_timeline() -> None:
    clock = Clock(name="neural", type="device", rate=30_000.0)
    first = _segment(200, seed=8, clock=clock)
    second = _segment(200, seed=9, t0=100.0, clock=clock)

    decoder = KalmanDecoder().fit_segments([first, second])

    assert decoder.clock_ == clock
    assert decoder.feature_names_in_ == tuple(_FEATURE_NAMES)
    assert decoder.target_schema_.output_names == _OUTPUT_NAMES


# --------------------------------------------------------------------------------------
# Discontinuity inside one segment
# --------------------------------------------------------------------------------------


def test_segment_interrupted_by_gap_is_rejected() -> None:
    features, target = _gappy_pair()

    with pytest.raises(ValidationError, match=r"frame 100 follows a gap of 1\.02 s"):
        KalmanDecoder().fit(features, target)


def test_gap_is_detected_without_declared_step() -> None:
    features, target = _gappy_pair(shift=None, window_size=None)

    assert features.shift is None and features.fs is None
    with pytest.raises(ValidationError, match="not contiguous"):
        KalmanDecoder().fit(features, target)


def test_gap_names_its_segment() -> None:
    first = _segment(200, seed=8)

    with pytest.raises(ValidationError, match="segment 1: the frames are not contiguous"):
        KalmanDecoder().fit_segments([first, _gappy_pair()])


def test_rejected_gap_leaves_fitted_decoder_untouched() -> None:
    features, target = _segment(300, seed=8)
    decoder = KalmanDecoder().fit(features, target)
    before = np.array(decoder.parameters_.transition_)

    with pytest.raises(ValidationError, match="not contiguous"):
        decoder.fit(*_gappy_pair())

    assert decoder.is_fitted
    assert np.array_equal(decoder.parameters_.transition_, before)
    assert decoder.n_segments_ == 1


def test_timing_jitter_is_not_gap() -> None:
    n_frames = 200
    states, observations = _simulate(n_frames, seed=4)
    rng = np.random.default_rng(21)
    time = np.arange(n_frames, dtype=np.float64) / _RATE
    time += np.sort(rng.uniform(0.0, 0.2 / _RATE, size=n_frames))
    features = _features(observations, fs=None, time=time)

    assert KalmanDecoder().fit(features, _target(states, features)).is_fitted


# --------------------------------------------------------------------------------------
# Reset
# --------------------------------------------------------------------------------------


def test_reset_replays_same_prediction() -> None:
    features, target = _segment(200, seed=10)
    decoder = KalmanDecoder(jitter=1e-6).fit(features, target)

    first = decoder.predict(features).data
    decoder.reset()
    second = decoder.predict(features).data

    assert np.array_equal(first, second)


def test_reset_restores_fitted_initial_state() -> None:
    features, target = _segment(200, seed=10)
    decoder = KalmanDecoder(jitter=1e-6).fit(features, target)
    decoder.predict(features)

    assert not np.allclose(decoder.state_, decoder.initial_state_)
    decoder.reset()
    assert np.array_equal(decoder.state_, decoder.initial_state_)
    assert np.array_equal(decoder.covariance_, decoder.initial_covariance_)


def test_reset_accepts_explicit_state_and_covariance() -> None:
    features, target = _segment(200, seed=10)
    decoder = KalmanDecoder(jitter=1e-6).fit(features, target)
    default = decoder.predict(features).data

    state = np.array([0.5, -0.25, 1.0, -1.0])
    covariance = np.diag([0.1, 0.1, 0.5, 0.5])
    decoder.reset(state, covariance)

    assert np.array_equal(decoder.state_, state)
    assert np.array_equal(decoder.covariance_, covariance)
    assert not np.allclose(decoder.predict(features).data, default)


def test_explicit_state_estimates_next_frame() -> None:
    """The same convention the fitted initial state follows: no free predict step."""

    features, target = _segment(200, seed=10)
    decoder = KalmanDecoder().fit(features, target)

    state = np.array([0.5, -0.25, 1.0, -1.0])
    decoder.reset(state, np.zeros((_N_OUTPUTS, _N_OUTPUTS)))
    predicted = decoder.predict(features).data

    # A zero covariance gives a zero gain, so a corrected-in-place first frame
    # is exactly the supplied state; a predicted-first one could not be.
    assert np.array_equal(predicted[0], state)


def test_prediction_moves_only_runtime_state() -> None:
    features, target = _segment(200, seed=10)
    decoder = KalmanDecoder(scaler=StandardScaler(), jitter=1e-6).fit(features, target)
    parameters = decoder.parameters_
    fitted = {
        "transition": np.array(decoder.parameters_.transition_),
        "observation": np.array(decoder.parameters_.observation_),
        "process": np.array(decoder.parameters_.process_covariance_),
        "initial_state": np.array(decoder.initial_state_),
        "initial_covariance": np.array(decoder.initial_covariance_),
        "mean": np.array(decoder.scaler_.mean_),
    }
    schema = decoder.feature_schema_
    state = np.array(decoder.state_)

    decoder.predict(features)

    assert decoder.parameters_ is parameters
    assert np.array_equal(decoder.parameters_.transition_, fitted["transition"])
    assert np.array_equal(decoder.parameters_.observation_, fitted["observation"])
    assert np.array_equal(decoder.parameters_.process_covariance_, fitted["process"])
    assert np.array_equal(decoder.initial_state_, fitted["initial_state"])
    assert np.array_equal(decoder.initial_covariance_, fitted["initial_covariance"])
    assert np.array_equal(decoder.scaler_.mean_, fitted["mean"])
    assert decoder.feature_schema_ == schema
    assert decoder.device_ == "cpu"
    # Only the runtime position moved.
    assert not np.array_equal(decoder.state_, state)


def test_reset_does_not_touch_fitted_state() -> None:
    features, target = _segment(200, seed=10)
    decoder = KalmanDecoder(scaler=StandardScaler()).fit(features, target)
    parameters = decoder.parameters_
    transition = np.array(decoder.parameters_.transition_)
    mean = np.array(decoder.scaler_.mean_)

    decoder.predict(features)
    decoder.reset(np.ones(_N_OUTPUTS))

    assert decoder.parameters_ is parameters
    assert np.array_equal(decoder.parameters_.transition_, transition)
    assert np.array_equal(decoder.scaler_.mean_, mean)
    assert decoder.device_ == "cpu"
    assert decoder.feature_schema_.fingerprint


def test_reset_on_unfitted_decoder_is_no_op() -> None:
    decoder = KalmanDecoder()

    decoder.reset()

    assert not decoder.is_fitted


def test_reset_with_explicit_state_needs_fitted_decoder() -> None:
    with pytest.raises(ValidationError, match="no state space to place"):
        KalmanDecoder().reset(np.zeros(_N_OUTPUTS))
    with pytest.raises(ValidationError, match="no state space to place"):
        KalmanDecoder().reset(covariance=np.eye(_N_OUTPUTS))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"state": np.zeros(3)}, "state must have shape"),
        ({"covariance": np.eye(3)}, "covariance must have shape"),
        ({"covariance": np.full((4, 4), -1.0)}, "positive semidefinite"),
    ],
)
def test_reset_validates_explicit_values(arguments: dict, message: str) -> None:
    features, target = _segment(100, seed=10)
    decoder = KalmanDecoder().fit(features, target)

    with pytest.raises(ValidationError, match=message):
        decoder.reset(**arguments)


# --------------------------------------------------------------------------------------
# Schema compatibility at prediction time
# --------------------------------------------------------------------------------------


def test_predict_from_other_features_fails_early() -> None:
    features, target = _segment(200, seed=11)
    decoder = KalmanDecoder().fit(features, target)
    state = np.array(decoder.state_)

    _, observations = _simulate(50, seed=12)
    renamed = _features(observations, feature_names=[f"unit_{i}" for i in range(_N_FEATURES)])

    with pytest.raises(ValidationError, match="feature_names"):
        decoder.predict(renamed)
    assert np.array_equal(decoder.state_, state)


def test_predict_from_different_width_fails() -> None:
    features, target = _segment(200, seed=11)
    decoder = KalmanDecoder().fit(features, target)

    _, observations = _simulate(50, seed=12)
    narrower = _features(observations[:, :3], feature_names=_FEATURE_NAMES[:3])

    with pytest.raises(ValidationError, match="do not match the fitted feature schema"):
        decoder.predict(narrower)


# --------------------------------------------------------------------------------------
# Parity with a hand-built composition
# --------------------------------------------------------------------------------------


def test_decoder_is_scaler_plus_state_space_model() -> None:
    features, target = _segment(400, seed=13)
    decoder = KalmanDecoder(
        scaler=StandardScaler(),
        jitter=1e-8,
        innovation_jitter=1e-10,
    ).fit(features, target)
    predicted = decoder.predict(features).data

    scaler = StandardScaler().fit(features.data)
    observations = scaler.transform(features.data)
    estimator = LinearGaussianStateSpace(jitter=1e-8).fit(
        target.data, observations, segment_lengths=np.array([features.n_frames])
    )
    running = KalmanFilter(estimator.parameters_, jitter=1e-10)
    expected = running.filter(observations, predict_first=False).states

    assert np.array_equal(predicted, expected)


def test_unscaled_decoder_is_state_space_model() -> None:
    features, target = _segment(300, seed=14)
    decoder = KalmanDecoder().fit(features, target)
    predicted = decoder.predict(features).data

    estimator = LinearGaussianStateSpace().fit(
        target.data, features.data, segment_lengths=np.array([features.n_frames])
    )
    expected = KalmanFilter(estimator.parameters_).filter(features.data, predict_first=False).states

    assert np.array_equal(predicted, expected)


# --------------------------------------------------------------------------------------
# Owned scaling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scaler", [StandardScaler(), MinMaxScaler(feature_range=(-1.0, 1.0))])
def test_decoder_fits_own_scaler_copy(scaler: object) -> None:
    features, target = _segment(200, seed=15)
    decoder = KalmanDecoder(scaler=scaler).fit(features, target)

    assert decoder.scaler_ is not scaler
    assert type(decoder.scaler_) is type(scaler)
    assert decoder.scaler_.n_features_in_ == _N_FEATURES
    with pytest.raises(ValidationError, match="is not fitted"):
        _ = scaler.scale_


def test_fitted_statistics_do_not_change_during_prediction() -> None:
    features, target = _segment(300, seed=15)
    decoder = KalmanDecoder(scaler=StandardScaler()).fit(features, target)
    mean = np.array(decoder.scaler_.mean_)
    scale = np.array(decoder.scaler_.scale_)

    _, drifted = _simulate(300, seed=16)
    decoder.predict(_features(drifted * 50.0 + 100.0))
    decoder.predict(_features(drifted * 50.0 + 100.0))

    assert np.array_equal(decoder.scaler_.mean_, mean)
    assert np.array_equal(decoder.scaler_.scale_, scale)


def test_scaler_measures_whole_fit_across_segments() -> None:
    first = _segment(300, seed=8)
    second = _segment(200, seed=9, t0=100.0)
    decoder = KalmanDecoder(scaler=StandardScaler()).fit_segments([first, second])

    stacked = np.concatenate([first[0].data, second[0].data], axis=0)
    assert np.allclose(decoder.scaler_.mean_, stacked.mean(axis=0))
    assert decoder.scaler_.n_samples_ == 500


def test_published_scaler_cannot_be_refitted() -> None:
    features, target = _segment(300, seed=15)
    decoder = KalmanDecoder(scaler=StandardScaler()).fit(features, target)
    predicted = decoder.predict(features).data
    decoder.reset()
    _, drifted = _simulate(300, seed=16)

    assert decoder.scaler_.is_frozen
    with pytest.raises(ValidationError, match="is frozen"):
        decoder.scaler_.fit(drifted * 50.0 + 100.0)
    with pytest.raises(AttributeError, match="is frozen"):
        decoder.scaler_._scale = np.ones(_N_FEATURES)

    # Reading the scaler is allowed; changing what the decoder computes with it
    # is not.
    assert np.allclose(
        decoder.scaler_.transform(features.data[:4]),
        (features.data[:4] - decoder.scaler_.mean_) / decoder.scaler_.scale_,
    )
    assert np.array_equal(decoder.predict(features).data, predicted)


def test_frozen_scaler_is_not_configuration() -> None:
    features, target = _segment(200, seed=15)
    published = StandardScaler().fit(features.data).freeze()

    with pytest.raises(ValidationError, match="scaler is frozen"):
        KalmanDecoder(scaler=published)

    # And a decoder whose configured scaler is frozen after construction fails
    # before it discards anything.
    configured = StandardScaler()
    decoder = KalmanDecoder(scaler=configured).fit(features, target)
    parameters = decoder.parameters_
    configured.fit(features.data).freeze()

    with pytest.raises(ValidationError, match="scaler is frozen"):
        decoder.fit(features, target)

    assert decoder.is_fitted
    assert decoder.parameters_ is parameters


# --------------------------------------------------------------------------------------
# Caller-supplied feature selection
# --------------------------------------------------------------------------------------


def test_feature_selection_follows_caller_order() -> None:
    features, target = _segment(300, seed=18)
    chosen = ["rate_4", "rate_0", "rate_3"]
    decoder = KalmanDecoder(feature_names=chosen).fit(features, target)

    assert decoder.selected_feature_names_ == tuple(chosen)
    assert decoder.n_observations_ == 3
    assert decoder.n_features_in_ == _N_FEATURES

    columns = features.data[:, [4, 0, 3]]
    estimator = LinearGaussianStateSpace().fit(
        target.data, columns, segment_lengths=np.array([features.n_frames])
    )
    expected = KalmanFilter(estimator.parameters_).filter(columns, predict_first=False).states

    assert np.array_equal(decoder.predict(features).data, expected)


def test_selection_runs_before_scaling() -> None:
    features, target = _segment(300, seed=18)
    decoder = KalmanDecoder(feature_names=["rate_4", "rate_0"], scaler=StandardScaler()).fit(
        features, target
    )

    assert decoder.scaler_.n_features_in_ == 2
    assert np.allclose(decoder.scaler_.mean_, features.data[:, [4, 0]].mean(axis=0))


def test_selecting_absent_feature_is_rejected() -> None:
    features, target = _segment(100, seed=18)
    decoder = KalmanDecoder(feature_names=["rate_0", "beta"])

    with pytest.raises(ValidationError, match=r"does not provide: \['beta'\]"):
        decoder.fit(features, target)


def test_refit_missing_selected_feature_leaves_decoder_intact() -> None:
    features, target = _segment(300, seed=18)
    decoder = KalmanDecoder(feature_names=["rate_4", "rate_0"]).fit(features, target)
    parameters = decoder.parameters_
    predicted = decoder.predict(features).data
    decoder.reset()

    renamed = _features(features.data, feature_names=[f"other_{idx}" for idx in range(_N_FEATURES)])
    with pytest.raises(ValidationError, match=r"does not provide: \['rate_4', 'rate_0'\]"):
        decoder.fit(renamed, _target(target.data, renamed))

    # An argument error is raised before anything is discarded, so the decoder
    # that was usable a moment ago still is -- with the same model, and the
    # same predictions.
    assert decoder.is_fitted
    assert decoder.parameters_ is parameters
    assert decoder.selected_feature_names_ == ("rate_4", "rate_0")
    assert np.array_equal(decoder.predict(features).data, predicted)


def test_refit_with_gap_leaves_decoder_intact() -> None:
    features, target = _segment(300, seed=18)
    decoder = KalmanDecoder().fit(features, target)
    parameters = decoder.parameters_

    with pytest.raises(ValidationError, match="not contiguous"):
        decoder.fit(*_gappy_pair(200))

    assert decoder.is_fitted
    assert decoder.parameters_ is parameters


def test_selected_names_are_fit_time_snapshot() -> None:
    features, target = _segment(300, seed=18)
    decoder = KalmanDecoder(feature_names=["rate_4", "rate_0"]).fit(features, target)

    # The constructor argument stays a configuration for the *next* fit. What
    # the fitted decoder consumes was decided by the fit that ran.
    decoder.feature_names = ("rate_1",)

    assert decoder.selected_feature_names_ == ("rate_4", "rate_0")
    assert decoder.n_observations_ == 2


def test_selection_is_only_feature_search() -> None:
    """No greedy forward search: an unconfigured decoder keeps every feature."""

    features, target = _segment(300, seed=18)
    decoder = KalmanDecoder().fit(features, target)

    assert decoder.n_observations_ == _N_FEATURES
    assert decoder.selected_feature_names_ == decoder.feature_names_in_


# --------------------------------------------------------------------------------------
# Missing observations
# --------------------------------------------------------------------------------------


def _with_missing_row(features: FeatureMatrix, row: int, *, whole: bool = True) -> FeatureMatrix:
    values = np.array(features.data, copy=True)
    if whole:
        values[row] = np.nan
    else:
        values[row, 0] = np.nan
    return _features(values)


@pytest.mark.parametrize("scaler", [None, StandardScaler()])
def test_fully_missing_row_runs_predict_step_alone(scaler: object) -> None:
    features, target = _segment(200, seed=19)
    decoder = KalmanDecoder(scaler=scaler, missing="predict", jitter=1e-6).fit(features, target)

    predicted = decoder.predict(_with_missing_row(features, 100))

    assert np.all(np.isfinite(predicted.data))
    assert predicted.data.shape == (200, _N_OUTPUTS)


def test_missing_row_errors_under_default_policy() -> None:
    features, target = _segment(200, seed=19)
    decoder = KalmanDecoder().fit(features, target)

    with pytest.raises(ValidationError, match="finite"):
        decoder.predict(_with_missing_row(features, 100))


@pytest.mark.parametrize("scaler", [None, StandardScaler()])
def test_partially_missing_row_is_rejected(scaler: object) -> None:
    features, target = _segment(200, seed=19)
    decoder = KalmanDecoder(scaler=scaler, missing="predict").fit(features, target)

    with pytest.raises(ValidationError):
        decoder.predict(_with_missing_row(features, 100, whole=False))


def test_scaled_missing_row_matches_unscaled_absence() -> None:
    """Routing an absent row past the scaler must not change what absence means."""

    features, target = _segment(200, seed=19)
    decoder = KalmanDecoder(scaler=StandardScaler(), missing="predict", jitter=1e-6)
    decoder.fit(features, target)

    with_gap = decoder.predict(_with_missing_row(features, 100)).data
    decoder.reset()
    complete = decoder.predict(features).data

    # The absent row is propagated rather than corrected, so it differs there
    # and stays finite, while the frames before it are untouched.
    assert np.array_equal(with_gap[:100], complete[:100])
    assert not np.allclose(with_gap[100], complete[100])


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------


def test_fitted_device_is_cpu_and_fixed() -> None:
    features, target = _segment(200, seed=20)
    decoder = KalmanDecoder().fit(features, target)

    assert decoder.device_ == "cpu"
    assert decoder.parameters_.device_ == "cpu"
    decoder.predict(features)
    decoder.reset()
    assert decoder.device_ == "cpu"
    with pytest.raises(AttributeError):
        decoder.device_ = "cuda"


def test_requesting_cuda_fails_without_fallback() -> None:
    features, target = _segment(100, seed=20)

    with runtime_context(device="cuda"), pytest.raises(DeviceUnavailableError):
        KalmanDecoder().fit(features, target)


# --------------------------------------------------------------------------------------
# Decoder behavior contracts
# --------------------------------------------------------------------------------------


def test_no_bias_state_is_appended_to_target() -> None:
    features, target = _segment(200, seed=22)
    decoder = KalmanDecoder().fit(features, target)

    assert decoder.parameters_.n_states_ == _N_OUTPUTS
    assert decoder.n_outputs_ == _N_OUTPUTS
    assert decoder.predict(features).data.shape[1] == _N_OUTPUTS


def test_target_is_decoded_in_own_units() -> None:
    """No hidden min-max of the target into [-1, 1] and back."""

    features, target = _segment(600, seed=22)
    scaled = _target(target.data * 1000.0, features)
    decoder = KalmanDecoder().fit(features, scaled)

    predicted = decoder.predict(features).data
    assert np.abs(predicted).max() > 100.0
    assert np.allclose(predicted.mean(axis=0), scaled.data.mean(axis=0), atol=1.0)


def test_no_steady_state_override_is_exposed() -> None:
    decoder = KalmanDecoder()

    for unsupported_name in (
        "steady_state",
        "steady_state_A",
        "steady_state_Q",
        "x0",
        "best_indexs",
    ):
        assert not hasattr(decoder, unsupported_name)
    with pytest.raises(TypeError):
        KalmanDecoder(steady_state=True)
