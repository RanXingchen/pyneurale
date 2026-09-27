#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import Clock, FeatureMatrix, SignalArray
from neurale.decoding import ContinuousDecoder, LinearDecoder, RidgeDecoder
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import (
    PCA,
    LinearRegression,
    MinMaxScaler,
    Ridge,
    StandardScaler,
    TemporalContext,
)
from neurale.runtime import runtime_context

_RATE = 50.0
_N_FEATURES = 6
_OUTPUT_NAMES = ["px", "py", "vx"]
_OUTPUT_UNITS = ["m", "m", "m/s"]
_N_OUTPUTS = len(_OUTPUT_NAMES)
_FEATURE_NAMES = [f"rate_{idx}" for idx in range(_N_FEATURES)]

# A known linear map from three latent outputs onto the observed features, so a
# fit can be checked against the arithmetic the data came from.
_MIXING = np.random.default_rng(7).normal(size=(_N_OUTPUTS, _N_FEATURES))
_NOISE = 0.05

_DECODERS = [LinearDecoder, RidgeDecoder]
_DECODER_IDS = ["linear", "ridge"]


# --------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------


def _simulate(n_frames: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.cumsum(rng.normal(scale=0.05, size=(n_frames, _N_OUTPUTS)), axis=0)
    observations = states @ _MIXING + rng.normal(scale=_NOISE, size=(n_frames, _N_FEATURES))
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


def _pair(
    n_frames: int = 400,
    *,
    seed: int = 1,
    t0: float = 0.0,
    clock: Clock | None = None,
) -> tuple[FeatureMatrix, SignalArray]:
    states, observations = _simulate(n_frames, seed=seed)
    features = _features(observations, t0=t0, clock=clock)
    return features, _target(states, features, clock=clock)


# --------------------------------------------------------------------------------------
# Constructor
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
def test_decoders_are_continuous_decoders(factory) -> None:
    assert issubclass(factory, ContinuousDecoder)
    assert factory.supports_cuda is False


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"scaler": "standard"}, "scaler must be"),
        ({"scaler": object()}, "scaler must be"),
        ({"context": 3}, "context must be a TemporalContext"),
        ({"context": (1, 1)}, "context must be a TemporalContext"),
        ({"fit_intercept": 1}, "fit_intercept must be a bool"),
    ],
)
def test_invalid_arguments_are_rejected(factory, arguments: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        factory(**arguments)


@pytest.mark.parametrize("alpha", [-1.0, "strong", None])
def test_ridge_penalty_is_validated(alpha) -> None:
    with pytest.raises(ValidationError, match="alpha"):
        RidgeDecoder(alpha)


def test_defaults_consume_every_frame_unscaled() -> None:
    features, target = _pair(200)
    decoder = LinearDecoder().fit(features, target)

    assert decoder.scaler_ is None
    assert decoder.context_ is None
    assert decoder.n_model_features_ == _N_FEATURES
    assert decoder.n_samples_ == 200


# --------------------------------------------------------------------------------------
# Configuration is revalidated by every fit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("scaler", "standard", "scaler must be"),
        ("context", 3, "context must be a TemporalContext"),
        ("fit_intercept", 1, "fit_intercept must be a bool"),
    ],
    ids=["scaler", "context", "fit_intercept"],
)
def test_invalid_rebound_configuration_is_rejected_early(
    factory,
    attribute: str,
    value: object,
    message: str,
) -> None:
    """A refit checks what it will actually use, not what ``__init__`` accepted."""

    features, target = _pair(200, seed=11)
    decoder = factory().fit(features, target)
    coefs = np.array(decoder.coef_)
    predicted = decoder.predict(features).data.copy()

    setattr(decoder, attribute, value)
    with pytest.raises(ValidationError, match=message):
        decoder.fit(features, target)

    assert decoder.is_fitted
    assert np.array_equal(decoder.coef_, coefs)
    assert np.array_equal(decoder.predict(features).data, predicted)


def test_invalid_rebound_ridge_penalty_is_rejected_early() -> None:
    features, target = _pair(200, seed=11)
    decoder = RidgeDecoder(0.5).fit(features, target)
    coefs = np.array(decoder.coef_)

    decoder.alpha = -1.0
    with pytest.raises(ValidationError, match="alpha"):
        decoder.fit(features, target)

    assert decoder.is_fitted
    assert np.array_equal(decoder.coef_, coefs)


def test_scaler_frozen_after_construction_is_rejected() -> None:
    features, target = _pair(200, seed=11)
    scaler = StandardScaler()
    decoder = LinearDecoder(scaler=scaler).fit(features, target)
    coefs = np.array(decoder.coef_)

    scaler.fit(features.data).freeze()
    with pytest.raises(ValidationError, match="scaler is frozen"):
        decoder.fit(features, target)

    assert decoder.is_fitted
    assert np.array_equal(decoder.coef_, coefs)


def test_valid_rebound_configuration_is_used_next_fit() -> None:
    """Revalidating a configuration checks it; it does not freeze it."""

    features, target = _pair(200, seed=11)
    decoder = RidgeDecoder(0.5).fit(features, target)

    decoder.alpha = 5.0
    decoder.fit(features, target)

    assert np.array_equal(decoder.coef_, RidgeDecoder(5.0).fit(features, target).coef_)


# --------------------------------------------------------------------------------------
# Parity with a hand-built composition
# --------------------------------------------------------------------------------------


def test_linear_decoder_is_regression() -> None:
    features, target = _pair(300, seed=2)
    decoder = LinearDecoder().fit(features, target)

    expected = LinearRegression().fit(features.data, target.data).predict(features.data)

    assert np.array_equal(decoder.predict(features).data, expected)


def test_ridge_decoder_is_regression() -> None:
    features, target = _pair(300, seed=2)
    decoder = RidgeDecoder(0.5).fit(features, target)

    expected = Ridge(0.5).fit(features.data, target.data).predict(features.data)

    assert np.array_equal(decoder.predict(features).data, expected)


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
@pytest.mark.parametrize(
    "scaler",
    [None, StandardScaler(), MinMaxScaler(feature_range=(-1.0, 1.0))],
    ids=["unscaled", "standard", "min-max"],
)
@pytest.mark.parametrize(
    "context",
    [None, TemporalContext(left=4), TemporalContext(left=2, right=3)],
    ids=["no-context", "causal", "acausal"],
)
def test_decoder_is_stages_and_model(factory, scaler, context) -> None:
    """The composition is the contract: scale, then stack, then regress."""

    features, target = _pair(300, seed=3)
    decoder = factory(scaler=scaler, context=context).fit(features, target)

    values = features.data
    if scaler is not None:
        values = type(scaler)(**_configuration(scaler)).fit(features.data).transform(features.data)
    design = values if context is None else context.transform(values)
    targets = target.data if context is None else context.trim(target.data)
    model = LinearRegression() if factory is LinearDecoder else Ridge(1.0)
    expected = model.fit(design, targets).predict(design)

    assert np.array_equal(decoder.predict(features).data, expected)
    assert np.array_equal(decoder.coef_, model.coef_)
    assert np.array_equal(decoder.intercept_, model.intercept_)


def _configuration(scaler: object) -> dict:
    if isinstance(scaler, MinMaxScaler):
        return {"feature_range": scaler.feature_range}
    return {"with_mean": scaler.with_mean, "with_std": scaler.with_std}


# --------------------------------------------------------------------------------------
# Multi-output regression
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
def test_every_output_channel_is_decoded(factory) -> None:
    features, target = _pair(2000, seed=4)
    decoder = factory(0.001) if factory is RidgeDecoder else factory()
    decoder.fit(features, target)

    predicted = decoder.predict(features)

    assert predicted.data.shape == (2000, _N_OUTPUTS)
    assert decoder.coef_.shape == (_N_OUTPUTS, _N_FEATURES)
    assert decoder.intercept_.shape == (_N_OUTPUTS,)
    error = np.sqrt(np.mean(np.square(predicted.data - target.data), axis=0))
    assert np.all(error < 0.1 * np.std(target.data, axis=0))


def test_single_output_target_decodes_two_dimensional_signal() -> None:
    features, _ = _pair(300, seed=5)
    states, _ = _simulate(300, seed=5)
    target = _target(
        states[:, :1],
        features,
        channel_names=["px"],
        units=["m"],
    )

    decoder = LinearDecoder().fit(features, target)
    predicted = decoder.predict(features)

    assert predicted.data.shape == (300, 1)
    assert decoder.n_outputs_ == 1
    assert decoder.coef_.shape == (1, _N_FEATURES)


def test_output_columns_are_independent() -> None:
    """Each channel is its own regression against the shared design."""

    features, target = _pair(400, seed=6)
    joint = LinearDecoder().fit(features, target).predict(features).data

    for output in range(_N_OUTPUTS):
        alone = _target(
            target.data[:, output : output + 1],
            features,
            channel_names=[_OUTPUT_NAMES[output]],
            units=[_OUTPUT_UNITS[output]],
        )
        separate = LinearDecoder().fit(features, alone).predict(features).data
        assert np.allclose(joint[:, output : output + 1], separate)


def test_held_out_recording_decodes_near_true_state() -> None:
    train_features, train_target = _pair(2000, seed=7)
    test_features, test_target = _pair(600, seed=17)

    decoder = RidgeDecoder(0.01, scaler=StandardScaler()).fit(train_features, train_target)
    predicted = decoder.predict(test_features)

    # Measured ratios are [0.10, 0.24, 0.08]; the bound is what a linear decode
    # of a linear mixture must clear, not the best it happens to do here.
    error = np.sqrt(np.mean(np.square(predicted.data - test_target.data), axis=0))
    assert np.all(error < 0.3 * np.std(test_target.data, axis=0))


# --------------------------------------------------------------------------------------
# Temporal context boundaries
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [(1, 0), (0, 1), (3, 2), (7, 7)],
    ids=["one-past", "one-future", "mixed", "wide"],
)
def test_context_trims_targets_and_times_identically(left: int, right: int) -> None:
    features, target = _pair(200, seed=8)
    context = TemporalContext(left=left, right=right)
    decoder = LinearDecoder(context=context).fit(features, target)

    predicted = decoder.predict(features)
    kept = 200 - left - right

    assert predicted.n_samples == kept
    assert decoder.n_samples_ == kept
    # The times the result carries are exactly the frames the features kept,
    # which is exactly what the target lost during the fit.
    assert np.array_equal(predicted.time, features.time[left : left + kept])
    assert np.array_equal(predicted.time, context.trim(features.time))
    assert np.array_equal(context.trim(target.time), predicted.time)


def test_context_stacks_frames_oldest_to_newest() -> None:
    features, target = _pair(100, seed=8)
    context = TemporalContext(left=2)
    decoder = LinearDecoder(context=context).fit(features, target)

    assert decoder.n_model_features_ == _N_FEATURES * 3
    assert decoder.coef_.shape == (_N_OUTPUTS, _N_FEATURES * 3)
    assert decoder.context_ == context
    assert decoder.n_features_in_ == _N_FEATURES


def test_frames_shorter_than_context_are_rejected() -> None:
    features, target = _pair(4, seed=8)
    decoder = LinearDecoder(context=TemporalContext(left=3, right=2))

    with pytest.raises(ValidationError, match="at least 6 frames"):
        decoder.fit(features, target)


def test_prediction_shorter_than_context_is_rejected() -> None:
    features, target = _pair(100, seed=8)
    decoder = LinearDecoder(context=TemporalContext(left=3, right=2)).fit(features, target)

    with pytest.raises(ValidationError, match="at least 6 frames"):
        decoder.predict(_features(features.data[:4]))


def test_rejected_fit_leaves_decoder_untouched() -> None:
    features, target = _pair(200, seed=8)
    decoder = LinearDecoder(context=TemporalContext(left=3, right=2)).fit(features, target)
    coefs = np.array(decoder.coef_)

    short = _pair(4, seed=9)
    with pytest.raises(ValidationError, match="at least 6 frames"):
        decoder.fit(*short)

    assert decoder.is_fitted
    assert np.array_equal(decoder.coef_, coefs)


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("context", [None, TemporalContext(left=2, right=1)])
def test_prediction_is_metadata_complete_signal(context) -> None:
    clock = Clock(name="neural", type="device", rate=30_000.0)
    features, target = _pair(300, clock=clock)
    decoder = LinearDecoder(context=context).fit(features, target)

    predicted = decoder.predict(features)

    assert isinstance(predicted, SignalArray)
    assert predicted.channels.names == _OUTPUT_NAMES
    assert predicted.channels.units == _OUTPUT_UNITS
    assert predicted.unit == tuple(_OUTPUT_UNITS)
    assert predicted.name == "cursor"
    assert predicted.fs == _RATE
    assert dict(predicted.attrs) == {"task": "center_out"}
    assert predicted.clock == clock
    assert decoder.clock_ == clock
    assert decoder.feature_names_in_ == tuple(_FEATURE_NAMES)


def test_prediction_on_another_timeline_is_rejected() -> None:
    features, target = _pair(200, clock=Clock(name="neural", type="device", rate=30_000.0))
    decoder = LinearDecoder().fit(features, target)

    other = _features(features.data, clock=Clock(name="behavior", type="device", rate=1_000.0))
    with pytest.raises(ValidationError, match="clock mismatch"):
        decoder.predict(other)


# --------------------------------------------------------------------------------------
# Schema compatibility at prediction time
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"feature_names": [f"other_{idx}" for idx in range(_N_FEATURES)]}, "feature_names"),
        ({"unit": "uV"}, "units"),
        ({"window_size": 0.1}, "window_size"),
        ({"source_signal": "pmd_units"}, "source_signal"),
    ],
)
def test_incompatible_features_are_rejected_early(
    override: dict,
    message: str,
) -> None:
    features, target = _pair(200, seed=10)
    decoder = LinearDecoder().fit(features, target)

    with pytest.raises(ValidationError, match=message):
        decoder.predict(_features(features.data, **override))


def test_different_feature_count_is_rejected() -> None:
    features, target = _pair(200, seed=10)
    decoder = LinearDecoder().fit(features, target)

    fewer = FeatureMatrix(
        data=features.data[:, :3],
        fs=_RATE,
        feature_names=_FEATURE_NAMES[:3],
        unit="spikes/s",
        source_signal="m1_units",
        window_size=0.05,
        shift=1.0 / _RATE,
        attrs={"timestamp_reference": "window_center"},
    )
    with pytest.raises(ValidationError, match="feature_names"):
        decoder.predict(fewer)


def test_fitted_results_require_fit() -> None:
    decoder = RidgeDecoder()

    for attribute in (
        "coef_",
        "intercept_",
        "rank_",
        "singular_values_",
        "scaler_",
        "context_",
        "n_model_features_",
        "n_samples_",
    ):
        with pytest.raises(ValidationError, match="not fitted"):
            getattr(decoder, attribute)


# --------------------------------------------------------------------------------------
# Statelessness and reset
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
def test_prediction_is_stateless(factory) -> None:
    features, target = _pair(300, seed=11)
    decoder = factory(scaler=StandardScaler()).fit(features, target)

    first = decoder.predict(features).data
    decoder.predict(_features(features.data * 3.0 + 10.0))
    again = decoder.predict(features).data

    assert np.array_equal(first, again)


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
def test_chunked_prediction_equals_whole_batch_prediction(factory) -> None:
    features, target = _pair(300, seed=11)
    decoder = factory().fit(features, target)
    whole = decoder.predict(features).data

    chunks = [
        decoder.predict(
            _features(features.data[start : start + 60], t0=float(features.time[start]))
        )
        for start in range(0, 300, 60)
    ]

    assert np.array_equal(np.concatenate([chunk.data for chunk in chunks], axis=0), whole)


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
def test_reset_is_no_op(factory) -> None:
    features, target = _pair(200, seed=11)
    decoder = factory().fit(features, target)
    before = decoder.predict(features).data
    coefs = np.array(decoder.coef_)
    schema = decoder.feature_schema_

    decoder.reset()
    decoder.reset()

    assert decoder.is_fitted
    assert np.array_equal(decoder.coef_, coefs)
    assert decoder.feature_schema_ == schema
    assert np.array_equal(decoder.predict(features).data, before)


def test_reset_on_unfitted_decoder_does_nothing() -> None:
    decoder = LinearDecoder()

    decoder.reset()

    assert not decoder.is_fitted


# --------------------------------------------------------------------------------------
# Owned scaling
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("scaler", [StandardScaler(), MinMaxScaler()])
def test_decoder_fits_own_frozen_scaler_copy(scaler: object) -> None:
    features, target = _pair(200, seed=12)
    decoder = LinearDecoder(scaler=scaler).fit(features, target)

    assert decoder.scaler_ is not scaler
    assert type(decoder.scaler_) is type(scaler)
    assert decoder.scaler_.is_frozen
    assert decoder.scaler_.n_features_in_ == _N_FEATURES
    with pytest.raises(ValidationError, match="is not fitted"):
        _ = scaler.scale_


def test_published_scaler_cannot_be_refitted() -> None:
    features, target = _pair(300, seed=12)
    decoder = RidgeDecoder(scaler=StandardScaler()).fit(features, target)
    predicted = decoder.predict(features).data

    with pytest.raises(ValidationError, match="is frozen"):
        decoder.scaler_.fit(features.data * 40.0 + 5.0)

    assert np.array_equal(decoder.predict(features).data, predicted)


def test_frozen_scaler_is_not_configuration() -> None:
    features, _ = _pair(200, seed=12)
    published = StandardScaler().fit(features.data).freeze()

    with pytest.raises(ValidationError, match="scaler is frozen"):
        LinearDecoder(scaler=published)


def test_fitted_statistics_do_not_change_during_prediction() -> None:
    features, target = _pair(300, seed=12)
    decoder = LinearDecoder(scaler=StandardScaler()).fit(features, target)
    mean = np.array(decoder.scaler_.mean_)

    decoder.predict(_features(features.data * 50.0 + 100.0))
    decoder.predict(_features(features.data * 50.0 + 100.0))

    assert np.array_equal(decoder.scaler_.mean_, mean)


def test_scaler_runs_before_context() -> None:
    """One statistic per real feature, not one per (feature, lag) pair."""

    features, target = _pair(300, seed=12)
    decoder = LinearDecoder(scaler=StandardScaler(), context=TemporalContext(left=4)).fit(
        features,
        target,
    )

    assert decoder.scaler_.n_features_in_ == _N_FEATURES
    assert decoder.scaler_.n_samples_ == 300
    assert np.allclose(decoder.scaler_.mean_, features.data.mean(axis=0))
    assert decoder.n_model_features_ == _N_FEATURES * 5


# --------------------------------------------------------------------------------------
# Composition outside the decoder
# --------------------------------------------------------------------------------------


def test_fitted_pca_is_composed_outside() -> None:
    """No decoder inserts a PCA; a caller applies one and fits on the result."""

    features, target = _pair(400, seed=13)
    pca = PCA(n_components=3).fit(features.data)
    reduced = _features(
        pca.transform(features.data),
        feature_names=["pc_0", "pc_1", "pc_2"],
        unit="a.u.",
    )

    decoder = LinearDecoder().fit(reduced, target)

    assert decoder.n_features_in_ == 3
    assert decoder.feature_names_in_ == ("pc_0", "pc_1", "pc_2")
    assert decoder.predict(reduced).data.shape == (400, _N_OUTPUTS)
    # The fitted schema describes the reduced input, so the raw features it was
    # derived from are no longer accepted.
    with pytest.raises(ValidationError, match="feature_names"):
        decoder.predict(features)


def test_explicit_feature_subset_is_composed_outside() -> None:
    features, target = _pair(300, seed=13)
    kept = [4, 0, 3]
    subset = _features(
        features.data[:, kept],
        feature_names=[_FEATURE_NAMES[idx] for idx in kept],
    )

    decoder = LinearDecoder().fit(subset, target)

    expected = LinearRegression().fit(features.data[:, kept], target.data)
    assert decoder.n_model_features_ == 3
    assert np.array_equal(decoder.coef_, expected.coef_)


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
def test_fitted_device_is_cpu_and_fixed(factory) -> None:
    features, target = _pair(200, seed=14)
    decoder = factory().fit(features, target)

    assert decoder.device_ == "cpu"
    decoder.predict(features)
    decoder.reset()
    assert decoder.device_ == "cpu"
    with pytest.raises(AttributeError):
        decoder.device_ = "cuda"


@pytest.mark.parametrize("factory", _DECODERS, ids=_DECODER_IDS)
def test_requesting_cuda_fails_before_native_execution(factory) -> None:
    features, target = _pair(200, seed=14)
    decoder = factory().fit(features, target)
    coefs = np.array(decoder.coef_)

    with runtime_context(device="cuda"), pytest.raises(DeviceUnavailableError):
        decoder.fit(features, target)

    # The device is resolved before anything is discarded or computed, so the
    # decoder that was fitted a moment ago is still exactly itself.
    assert decoder.is_fitted
    assert np.array_equal(decoder.coef_, coefs)


@pytest.mark.parametrize("requested", ["auto", "cpu"])
def test_ambient_context_does_not_change_fitted_device(requested: str) -> None:
    features, target = _pair(200, seed=14)
    decoder = LinearDecoder().fit(features, target)
    predicted = decoder.predict(features).data

    with runtime_context(device=requested):
        assert decoder.device_ == "cpu"
        assert np.array_equal(decoder.predict(features).data, predicted)
