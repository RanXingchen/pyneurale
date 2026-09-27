#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.data import Clock, FeatureMatrix
from neurale.decoding import (
    ClassificationDecoder,
    ClassificationPrediction,
    ClassificationTarget,
    LDADecoder,
)
from neurale.exceptions import DeviceUnavailableError, ValidationError
from neurale.models import LDA, MinMaxScaler, StandardScaler, TemporalContext
from neurale.runtime import runtime_context

_RATE = 20.0
_N_FEATURES = 4
_FEATURE_NAMES = ["beta", "gamma", "hfa", "lmp"]

# One fixed direction per class in feature space, so a fit can be checked
# against data that really is linearly separable up to the noise.
_DIRECTIONS = np.random.default_rng(5).normal(size=(6, _N_FEATURES))
_NOISE = 0.6


# --------------------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------------------


def _simulate(
    labels: np.ndarray,
    classes: list,
    *,
    seed: int,
    noise: float = _NOISE,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positions = {value: idx for idx, value in enumerate(classes)}
    centers = np.array([_DIRECTIONS[positions[value]] for value in labels.tolist()])
    return centers + rng.normal(scale=noise, size=(labels.shape[0], _N_FEATURES))


def _features(
    values: np.ndarray,
    *,
    t0: float = 0.0,
    clock: Clock | None = None,
    **overrides: object,
) -> FeatureMatrix:
    attrs: dict[str, object] = {"timestamp_reference": "window_center"}
    if clock is not None:
        attrs["clock"] = clock
    settings: dict[str, object] = {
        "data": values,
        "fs": _RATE,
        "t0": t0,
        "feature_names": list(_FEATURE_NAMES),
        "unit": "uV^2",
        "source_signal": "m1_lfp",
        "window_size": 0.2,
        "shift": 1.0 / _RATE,
        "attrs": attrs,
    }
    settings.update(overrides)
    return FeatureMatrix(**settings)


def _pair(
    classes: list,
    n_per_class: int = 60,
    *,
    seed: int = 1,
    clock: Clock | None = None,
    order: list | None = None,
    noise: float = _NOISE,
    **target_overrides: object,
) -> tuple[FeatureMatrix, ClassificationTarget]:
    labels = np.array(list(classes) * n_per_class)
    values = _simulate(labels, list(classes), seed=seed, noise=noise)
    features = _features(values, clock=clock)
    target = ClassificationTarget(
        labels=labels,
        time=features.time.copy(),
        classes=order,
        clock=clock,
        **target_overrides,
    )
    return features, target


def _scalar_features(values: np.ndarray) -> FeatureMatrix:
    """One feature, so a decision boundary can be placed on an exact value."""

    return FeatureMatrix(
        data=values,
        fs=_RATE,
        feature_names=["beta"],
        unit="uV^2",
        shift=1.0 / _RATE,
    )


_BINARY = ["left", "right"]
_MULTICLASS = ["up", "down", "left", "right"]


# --------------------------------------------------------------------------------------
# Constructor
# --------------------------------------------------------------------------------------


def test_decoder_is_classification_decoder() -> None:
    assert issubclass(LDADecoder, ClassificationDecoder)
    assert LDADecoder.supports_cuda is False


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"scaler": "standard"}, "scaler must be"),
        ({"context": 3}, "context must be a TemporalContext"),
        ({"shrinkage": "ledoit"}, "shrinkage must be"),
        ({"shrinkage": 1.5}, "shrinkage"),
        ({"shrinkage": -0.1}, "shrinkage"),
    ],
)
def test_invalid_arguments_are_rejected(arguments: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        LDADecoder(**arguments)


@pytest.mark.parametrize("shrinkage", [None, "auto", 0.0, 0.35, 1.0])
def test_accepted_shrinkage_values_fit(shrinkage) -> None:
    features, target = _pair(_MULTICLASS, seed=2)

    decoder = LDADecoder(shrinkage=shrinkage).fit(features, target)

    assert decoder.shrinkage == shrinkage
    assert decoder.predict(features).n_samples == features.n_frames


# --------------------------------------------------------------------------------------
# Configuration is revalidated by every fit
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("scaler", "standard", "scaler must be"),
        ("context", 3, "context must be a TemporalContext"),
        ("shrinkage", "ledoit", "shrinkage must be"),
    ],
    ids=["scaler", "context", "shrinkage"],
)
def test_invalid_rebound_configuration_is_rejected_early(
    attribute: str,
    value: object,
    message: str,
) -> None:
    """A refit checks what it will actually use, not what ``__init__`` accepted."""

    features, target = _pair(_MULTICLASS, seed=11)
    decoder = LDADecoder().fit(features, target)
    priors = np.array(decoder.priors_)
    decoded = decoder.predict(features).labels.copy()

    setattr(decoder, attribute, value)
    with pytest.raises(ValidationError, match=message):
        decoder.fit(features, target)

    assert decoder.is_fitted
    assert np.array_equal(decoder.priors_, priors)
    assert np.array_equal(decoder.predict(features).labels, decoded)


def test_scaler_frozen_after_construction_is_rejected() -> None:
    features, target = _pair(_MULTICLASS, seed=11)
    scaler = StandardScaler()
    decoder = LDADecoder(scaler=scaler).fit(features, target)
    priors = np.array(decoder.priors_)

    # The configured instance belongs to the caller, so it can become a
    # published fit between two fits of the decoder that reads it.
    scaler.fit(features.data).freeze()
    with pytest.raises(ValidationError, match="scaler is frozen"):
        decoder.fit(features, target)

    assert decoder.is_fitted
    assert np.array_equal(decoder.priors_, priors)


def test_valid_rebound_configuration_is_used_next_fit() -> None:
    """Revalidating a configuration checks it; it does not freeze it."""

    features, target = _pair(_MULTICLASS, seed=11)
    decoder = LDADecoder().fit(features, target)

    decoder.shrinkage = 0.5
    decoder.fit(features, target)

    expected = LDADecoder(shrinkage=0.5).fit(features, target)
    assert np.array_equal(decoder.covariance_, expected.covariance_)


# --------------------------------------------------------------------------------------
# The fit needs more frames than classes
# --------------------------------------------------------------------------------------


def test_fit_with_too_few_frames_is_rejected() -> None:
    """One mean per class and a shared covariance are not identified from that."""

    features, target = _pair(_MULTICLASS, 1, seed=12)

    with pytest.raises(ValidationError, match="more frames than classes"):
        LDADecoder().fit(features, target)


def test_context_trimming_below_class_count_is_rejected() -> None:
    features, target = _pair(_MULTICLASS, 2, seed=12)
    context = TemporalContext(left=2, right=2)

    # Eight frames, four left after trimming, and four classes to estimate.
    with pytest.raises(ValidationError, match="more frames than classes"):
        LDADecoder(context=context).fit(features, target)


def test_refit_with_too_few_frames_leaves_decoder_intact() -> None:
    """A short calibration block must not destroy a decoder that still works."""

    features, target = _pair(_MULTICLASS, seed=12)
    decoder = LDADecoder().fit(features, target)
    priors = np.array(decoder.priors_)
    decoded = decoder.predict(features).labels.copy()

    with pytest.raises(ValidationError, match="more frames than classes"):
        decoder.fit(*_pair(_MULTICLASS, 1, seed=13))

    assert decoder.is_fitted
    assert decoder.n_samples_ == features.n_frames
    assert np.array_equal(decoder.priors_, priors)
    assert np.array_equal(decoder.predict(features).labels, decoded)


# --------------------------------------------------------------------------------------
# Binary and multiclass decoding
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "classes",
    [_BINARY, ["a", "b", "c"], _MULTICLASS, list(range(6))],
    ids=["binary", "three", "four", "six"],
)
def test_separable_classes_are_decoded(classes: list) -> None:
    features, target = _pair(classes, seed=3, noise=0.25)
    decoder = LDADecoder().fit(features, target)

    decoded = decoder.predict(features)

    assert isinstance(decoded, ClassificationPrediction)
    assert decoded.n_classes == len(classes)
    assert float(np.mean(decoded.labels == target.labels)) > 0.9


@pytest.mark.parametrize(
    "labels",
    [
        np.array(["left", "right"] * 40),
        np.array([0, 1] * 40),
        np.array([-7, 12] * 40),
        np.array([False, True] * 40),
        np.array([2.5, 9.75] * 40),
    ],
    ids=["text", "zero-based", "arbitrary-ints", "bool", "float"],
)
def test_labels_are_opaque_and_returned_unchanged(labels: np.ndarray) -> None:
    values = _simulate(labels, sorted(set(labels.tolist())), seed=4, noise=0.25)
    features = _features(values)
    target = ClassificationTarget(labels=labels, time=features.time.copy())

    decoder = LDADecoder().fit(features, target)
    decoded = decoder.predict(features)

    assert set(decoded.labels.tolist()) <= set(labels.tolist())
    assert decoded.labels.dtype == target.labels.dtype
    assert np.array_equal(np.asarray(decoded.classes), np.asarray(target.classes))


# --------------------------------------------------------------------------------------
# Parity with a hand-built composition
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("classes", [_BINARY, _MULTICLASS], ids=["binary", "multiclass"])
@pytest.mark.parametrize(
    "scaler",
    [None, StandardScaler(), MinMaxScaler()],
    ids=["unscaled", "standard", "min-max"],
)
@pytest.mark.parametrize(
    "context",
    [None, TemporalContext(left=3), TemporalContext(left=1, right=2)],
    ids=["no-context", "causal", "acausal"],
)
def test_decoder_is_stages_and_model(classes, scaler, context) -> None:
    """The composition is the contract: scale, then stack, then discriminate."""

    features, target = _pair(classes, seed=5)
    decoder = LDADecoder(scaler=scaler, context=context).fit(features, target)

    values = features.data
    if scaler is not None:
        values = type(scaler)().fit(features.data).transform(features.data)
    design = values if context is None else context.transform(values)
    labels = target.labels if context is None else context.trim(np.asarray(target.labels))
    model = LDA().fit(design, labels)

    assert np.array_equal(decoder.predict(features).labels, model.predict(design))
    assert np.allclose(decoder.predict_proba(features).probabilities, model.predict_proba(design))


@pytest.mark.parametrize("shrinkage", ["auto", 0.5], ids=["auto", "fixed"])
def test_shrinkage_reaches_model(shrinkage) -> None:
    features, target = _pair(_MULTICLASS, seed=5)
    decoder = LDADecoder(shrinkage=shrinkage).fit(features, target)

    shrunk = LDA(shrinkage=shrinkage).fit(features.data, target.labels)
    plain = LDA().fit(features.data, target.labels)

    assert np.allclose(decoder.covariance_, shrunk.covariance_)
    assert not np.allclose(shrunk.covariance_, plain.covariance_)


# --------------------------------------------------------------------------------------
# Probabilities and class order
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("classes", [_BINARY, _MULTICLASS], ids=["binary", "multiclass"])
def test_probabilities_are_typed_in_declared_class_order(classes: list) -> None:
    features, target = _pair(classes, seed=6)
    decoder = LDADecoder().fit(features, target)

    posterior = decoder.predict_proba(features)

    assert isinstance(posterior, ClassificationPrediction)
    assert posterior.probabilities.shape == (features.n_frames, len(classes))
    assert np.allclose(posterior.probabilities.sum(axis=1), 1.0)
    assert np.all(posterior.probabilities >= 0.0)
    assert np.array_equal(np.asarray(posterior.classes), np.asarray(target.classes))
    assert np.array_equal(posterior.time, features.time)
    assert np.array_equal(posterior.labels, decoder.predict(features).labels)


def test_declared_class_order_is_column_order() -> None:
    """The model sorts its own columns; the caller's order is what is reported."""

    declared = ["right", "up", "left", "down"]
    features, target = _pair(_MULTICLASS, seed=6, order=declared)
    decoder = LDADecoder().fit(features, target)

    posterior = decoder.predict_proba(features)
    fitted = LDA().fit(features.data, target.labels)
    sorted_columns = fitted.predict_proba(features.data)

    assert list(np.asarray(posterior.classes)) == declared
    for column, name in enumerate(declared):
        source = list(fitted.classes_).index(name)
        assert np.allclose(posterior.probabilities[:, column], sorted_columns[:, source])


def test_decoded_label_is_largest_probability_column() -> None:
    features, target = _pair(_MULTICLASS, seed=6, order=["right", "up", "left", "down"])
    decoder = LDADecoder().fit(features, target)

    posterior = decoder.predict_proba(features)
    argmax = np.asarray(posterior.classes)[posterior.probabilities.argmax(axis=1)]

    assert np.array_equal(argmax, posterior.labels)


def test_tie_breaks_by_model_class_order() -> None:
    """The label is the decision rule's answer, and stays so on an exact tie.

    A label and an argmax over the declared columns break a tie in different
    orders, so the two can differ on one. Reporting the rule's own answer is
    what keeps a label and the rule that produced it from disagreeing; the
    result is deterministic either way.
    """

    # Two classes symmetric about zero, so the boundary sits exactly on 0.0.
    features = _scalar_features(np.array([[-1.5], [-0.5], [0.5], [1.5]]))
    target = ClassificationTarget(
        labels=np.array([0, 0, 1, 1]),
        time=features.time.copy(),
        classes=[1, 0],  # the reverse of the order the model sorts into
    )
    decoder = LDADecoder().fit(features, target)

    probe = _scalar_features(np.array([[0.0]]))
    posterior = decoder.predict_proba(probe)
    argmax = np.asarray(posterior.classes)[posterior.probabilities.argmax(axis=1)]

    assert np.array_equal(posterior.probabilities, [[0.5, 0.5]])
    assert list(np.asarray(posterior.classes)) == [1, 0]
    assert np.array_equal(posterior.labels, [0])
    assert np.array_equal(argmax, [1])
    assert np.array_equal(decoder.predict(probe).labels, posterior.labels)


def test_scores_are_left_empty() -> None:
    """A per-class decision value is not identified for two classes."""

    features, target = _pair(_BINARY, seed=6)
    decoder = LDADecoder().fit(features, target)

    assert decoder.predict(features).scores is None
    assert decoder.predict_proba(features).scores is None


def test_probabilities_are_raw() -> None:
    """Each frame is decoded on its own evidence."""

    features, target = _pair(_BINARY, seed=7, noise=1.6)
    decoder = LDADecoder().fit(features, target)

    posterior = decoder.predict_proba(features)
    shuffled = np.random.default_rng(0).permutation(features.n_frames)
    reordered = decoder.predict_proba(_features(features.data[shuffled]))

    # Reordering the frames reorders the answers and changes nothing else: no
    # state carries across a row.
    assert np.allclose(reordered.probabilities, posterior.probabilities[shuffled])
    assert np.array_equal(reordered.labels, posterior.labels[shuffled])
    # And the raw posteriors reach the caller, including the uncertain ones.
    assert float(posterior.probabilities.max(axis=1).min()) < 0.75


# --------------------------------------------------------------------------------------
# Temporal context boundaries
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("left", "right"),
    [(1, 0), (0, 2), (3, 2)],
    ids=["one-past", "two-future", "mixed"],
)
def test_context_trims_labels_and_times_identically(left: int, right: int) -> None:
    features, target = _pair(_MULTICLASS, seed=8)
    context = TemporalContext(left=left, right=right)
    decoder = LDADecoder(context=context).fit(features, target)

    decoded = decoder.predict(features)
    kept = features.n_frames - left - right

    assert decoded.n_samples == kept
    assert decoder.n_samples_ == kept
    assert np.array_equal(decoded.time, context.trim(features.time))
    assert np.array_equal(decoder.predict_proba(features).time, decoded.time)
    assert decoder.predict_proba(features).probabilities.shape == (kept, len(_MULTICLASS))
    assert decoder.n_model_features_ == _N_FEATURES * context.n_frames


def test_frames_shorter_than_context_are_rejected() -> None:
    features, target = _pair(_BINARY, 2, seed=8)
    decoder = LDADecoder(context=TemporalContext(left=4, right=1))

    with pytest.raises(ValidationError, match="at least 6 frames"):
        decoder.fit(features, target)


def test_class_missing_after_trimming_is_rejected() -> None:
    labels = np.array(["rare"] + ["common"] * 40 + ["other"] * 40)
    values = _simulate(labels, ["common", "other", "rare"], seed=9, noise=0.3)
    features = _features(values)
    target = ClassificationTarget(labels=labels, time=features.time.copy())

    # The only "rare" frame is the first one, which a left context drops.
    with pytest.raises(ValidationError, match=r"missing \['rare'\]"):
        LDADecoder(context=TemporalContext(left=2)).fit(features, target)


def test_declared_class_never_observed_is_rejected() -> None:
    features, target = _pair(_BINARY, seed=9, order=["left", "right", "centre"])

    with pytest.raises(ValidationError, match=r"missing \['centre'\]"):
        LDADecoder().fit(features, target)


def test_rejected_fit_leaves_decoder_untouched() -> None:
    features, target = _pair(_MULTICLASS, seed=9)
    decoder = LDADecoder().fit(features, target)
    priors = np.array(decoder.priors_)

    _, bad = _pair(_MULTICLASS, seed=9, order=[*_MULTICLASS, "absent"])
    with pytest.raises(ValidationError, match="missing"):
        decoder.fit(features, bad)

    assert decoder.is_fitted
    assert np.array_equal(decoder.priors_, priors)
    assert decoder.n_classes_ == len(_MULTICLASS)


# --------------------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("context", [None, TemporalContext(left=2, right=1)])
def test_prediction_carries_source_frames(context) -> None:
    clock = Clock(name="neural", type="device", rate=30_000.0)
    features, target = _pair(_MULTICLASS, seed=10, clock=clock)
    decoder = LDADecoder(context=context).fit(features, target)

    decoded = decoder.predict(features)

    assert decoded.clock == clock
    assert decoder.clock_ == clock
    assert np.array_equal(np.asarray(decoded.classes), np.asarray(target.classes))
    assert decoder.feature_names_in_ == tuple(_FEATURE_NAMES)
    assert decoder.n_features_in_ == _N_FEATURES
    assert decoder.predict_proba(features).clock == clock


def test_prediction_on_another_timeline_is_rejected() -> None:
    features, target = _pair(_BINARY, seed=10, clock=Clock(name="neural", type="device", rate=3e4))
    decoder = LDADecoder().fit(features, target)

    other = _features(features.data, clock=Clock(name="behavior", type="device", rate=1_000.0))
    for call in (decoder.predict, decoder.predict_proba):
        with pytest.raises(ValidationError, match="clock mismatch"):
            call(other)


# --------------------------------------------------------------------------------------
# Schema compatibility at prediction time
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"feature_names": ["a", "b", "c", "d"]}, "feature_names"),
        ({"unit": "Hz"}, "units"),
        ({"window_size": 0.4}, "window_size"),
        ({"source_signal": "pmd_lfp"}, "source_signal"),
    ],
)
def test_incompatible_features_are_rejected_early(override, message) -> None:
    features, target = _pair(_BINARY, seed=11)
    decoder = LDADecoder().fit(features, target)

    for call in (decoder.predict, decoder.predict_proba):
        with pytest.raises(ValidationError, match=message):
            call(_features(features.data, **override))


def test_fitted_results_require_fit() -> None:
    decoder = LDADecoder()

    for attribute in (
        "priors_",
        "means_",
        "covariance_",
        "scaler_",
        "context_",
        "n_model_features_",
        "n_samples_",
        "classes_",
    ):
        with pytest.raises(ValidationError, match="not fitted"):
            getattr(decoder, attribute)


def test_predicting_before_fit_is_rejected() -> None:
    features, _ = _pair(_BINARY, seed=11)
    decoder = LDADecoder()

    for call in (decoder.predict, decoder.predict_proba):
        with pytest.raises(ValidationError, match="not fitted"):
            call(features)


# --------------------------------------------------------------------------------------
# Statelessness and reset
# --------------------------------------------------------------------------------------


def test_prediction_is_stateless() -> None:
    features, target = _pair(_MULTICLASS, seed=12)
    decoder = LDADecoder(scaler=StandardScaler()).fit(features, target)

    first = decoder.predict_proba(features).probabilities
    decoder.predict(_features(features.data * 5.0 + 20.0))
    again = decoder.predict_proba(features).probabilities

    assert np.array_equal(first, again)


def test_reset_is_no_op() -> None:
    features, target = _pair(_MULTICLASS, seed=12)
    decoder = LDADecoder().fit(features, target)
    before = decoder.predict(features).labels
    priors = np.array(decoder.priors_)

    decoder.reset()

    assert decoder.is_fitted
    assert np.array_equal(decoder.priors_, priors)
    assert np.array_equal(decoder.predict(features).labels, before)


def test_reset_on_unfitted_decoder_does_nothing() -> None:
    decoder = LDADecoder()

    decoder.reset()

    assert not decoder.is_fitted


# --------------------------------------------------------------------------------------
# Owned scaling and fitted statistics
# --------------------------------------------------------------------------------------


def test_decoder_fits_own_frozen_scaler_copy() -> None:
    scaler = StandardScaler()
    features, target = _pair(_BINARY, seed=13)
    decoder = LDADecoder(scaler=scaler).fit(features, target)

    assert decoder.scaler_ is not scaler
    assert decoder.scaler_.is_frozen
    with pytest.raises(ValidationError, match="is frozen"):
        decoder.scaler_.fit(features.data * 30.0)
    with pytest.raises(ValidationError, match="is not fitted"):
        _ = scaler.mean_


def test_frozen_scaler_is_not_configuration() -> None:
    features, _ = _pair(_BINARY, seed=13)
    published = StandardScaler().fit(features.data).freeze()

    with pytest.raises(ValidationError, match="scaler is frozen"):
        LDADecoder(scaler=published)


def test_class_statistics_follow_declared_order() -> None:
    declared = ["right", "up", "left", "down"]
    features, target = _pair(_MULTICLASS, seed=13, order=declared)
    decoder = LDADecoder().fit(features, target)

    fitted = LDA().fit(features.data, target.labels)

    assert decoder.priors_.shape == (4,)
    assert decoder.means_.shape == (4, _N_FEATURES)
    assert np.allclose(decoder.covariance_, fitted.covariance_)
    for row, name in enumerate(declared):
        source = list(fitted.classes_).index(name)
        assert np.allclose(decoder.means_[row], fitted.means_[source])
        assert np.isclose(decoder.priors_[row], fitted.priors_[source])
    assert not decoder.priors_.flags.writeable
    assert not decoder.means_.flags.writeable


# --------------------------------------------------------------------------------------
# Composition outside the decoder
# --------------------------------------------------------------------------------------


def test_no_feature_ranking_or_search_happens() -> None:
    """Every feature reaches the discriminant; nothing is dropped for being weak."""

    features, target = _pair(_MULTICLASS, seed=14)
    noisy = _features(
        np.column_stack(
            [features.data, np.random.default_rng(0).normal(size=(features.n_frames, 2))]
        ),
        feature_names=[*_FEATURE_NAMES, "noise_0", "noise_1"],
    )

    decoder = LDADecoder().fit(noisy, target)

    assert decoder.n_features_in_ == _N_FEATURES + 2
    assert decoder.n_model_features_ == _N_FEATURES + 2
    assert decoder.means_.shape == (len(_MULTICLASS), _N_FEATURES + 2)


def test_fitted_reduction_is_composed_outside() -> None:
    from neurale.models import PCA

    features, target = _pair(_MULTICLASS, seed=14)
    pca = PCA(n_components=2).fit(features.data)
    reduced = _features(
        pca.transform(features.data),
        feature_names=["pc_0", "pc_1"],
        unit="a.u.",
    )

    decoder = LDADecoder().fit(reduced, target)

    assert decoder.n_features_in_ == 2
    assert decoder.predict(reduced).n_samples == reduced.n_frames
    with pytest.raises(ValidationError, match="feature_names"):
        decoder.predict(features)


# --------------------------------------------------------------------------------------
# Device
# --------------------------------------------------------------------------------------


def test_fitted_device_is_cpu_and_fixed() -> None:
    features, target = _pair(_BINARY, seed=15)
    decoder = LDADecoder().fit(features, target)

    assert decoder.device_ == "cpu"
    decoder.predict(features)
    decoder.predict_proba(features)
    decoder.reset()
    assert decoder.device_ == "cpu"
    with pytest.raises(AttributeError):
        decoder.device_ = "cuda"


def test_requesting_cuda_fails_before_native_execution() -> None:
    features, target = _pair(_MULTICLASS, seed=15)
    decoder = LDADecoder().fit(features, target)
    priors = np.array(decoder.priors_)

    with runtime_context(device="cuda"), pytest.raises(DeviceUnavailableError):
        decoder.fit(features, target)

    assert decoder.is_fitted
    assert np.array_equal(decoder.priors_, priors)


@pytest.mark.parametrize("requested", ["auto", "cpu"])
def test_ambient_context_does_not_change_fitted_device(requested: str) -> None:
    features, target = _pair(_BINARY, seed=15)
    decoder = LDADecoder().fit(features, target)
    posterior = decoder.predict_proba(features).probabilities

    with runtime_context(device=requested):
        assert decoder.device_ == "cpu"
        assert np.array_equal(decoder.predict_proba(features).probabilities, posterior)
