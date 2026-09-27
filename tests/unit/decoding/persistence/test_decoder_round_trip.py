#!/usr/bin/env python3

"""A saved decoder comes back predicting exactly what it predicted before.

Exactly, not approximately. The native models are rebuilt from their own
parameters rather than reimplemented, so any difference here is a real defect
and not floating-point weather -- which is why every comparison below is an
equality and never a tolerance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neurale.data import FeatureMatrix, SignalArray
from neurale.decoding import KalmanDecoder, LDADecoder, LinearDecoder
from neurale.decoding.persistence import load_decoder, read_manifest, save_decoder
from neurale.models import StandardScaler
from neurale.models.preprocessing import TemporalContext
from neurale.runtime import runtime_context

# --------------------------------------------------------------------------------------
# Continuous decoders
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["kalman", "linear", "ridge"])
def test_continuous_decoder_predicts_identically_after_round_trip(
    name: str, continuous, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    decoder = request.getfixturevalue(name)
    expected = decoder.predict(continuous.features)
    decoder.reset()

    save_decoder(decoder, tmp_path / name)
    restored = load_decoder(tmp_path / name)
    found = restored.predict(continuous.features)

    assert type(restored) is type(decoder)
    assert np.array_equal(found.data, expected.data)
    assert np.array_equal(found.time, expected.time)
    assert found.channels.names == expected.channels.names
    assert found.unit == expected.unit
    assert found.name == expected.name
    assert found.fs == expected.fs


@pytest.mark.parametrize("name", ["kalman", "linear", "ridge"])
def test_continuous_decoder_restores_fitted_facts(
    name: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    decoder = request.getfixturevalue(name)
    save_decoder(decoder, tmp_path / name)
    restored = load_decoder(tmp_path / name)

    assert restored.is_fitted
    assert restored.device_ == decoder.device_
    assert restored.feature_schema_.fingerprint == decoder.feature_schema_.fingerprint
    assert restored.feature_names_in_ == decoder.feature_names_in_
    assert restored.clock_ == decoder.clock_
    assert restored.n_outputs_ == decoder.n_outputs_
    assert restored.n_samples_ == decoder.n_samples_


def test_kalman_model_and_scaler_survive_round_trip(kalman, tmp_path: Path) -> None:
    save_decoder(kalman, tmp_path / "kalman")
    restored = load_decoder(tmp_path / "kalman")

    for name in ("transition_", "observation_", "process_covariance_", "initial_state_"):
        assert np.array_equal(
            getattr(restored.parameters_, name), getattr(kalman.parameters_, name)
        )
    assert np.array_equal(restored.scaler_.mean_, kalman.scaler_.mean_)
    assert np.array_equal(restored.segment_lengths_, kalman.segment_lengths_)
    assert restored.n_transitions_ == kalman.n_transitions_
    assert restored.selected_feature_names_ == kalman.selected_feature_names_


def test_regression_model_and_stages_survive_round_trip(linear, tmp_path: Path) -> None:
    save_decoder(linear, tmp_path / "linear")
    restored = load_decoder(tmp_path / "linear")

    assert np.array_equal(restored.coef_, linear.coef_)
    assert np.array_equal(restored.intercept_, linear.intercept_)
    assert np.array_equal(restored.singular_values_, linear.singular_values_)
    assert restored.rank_ == linear.rank_
    assert restored.context_ == linear.context_
    assert restored.n_model_features_ == linear.n_model_features_
    assert np.array_equal(restored.scaler_.scale_, linear.scaler_.scale_)


def test_ridge_penalty_survives_as_configuration(ridge, tmp_path: Path) -> None:
    save_decoder(ridge, tmp_path / "ridge")
    restored = load_decoder(tmp_path / "ridge")

    assert restored.alpha == ridge.alpha
    assert restored.fit_intercept == ridge.fit_intercept


def test_selected_feature_subset_survives(continuous, tmp_path: Path) -> None:
    """Selection is the one thing a decoder may narrow, so it has to round-trip."""

    decoder = KalmanDecoder(feature_names=["rate_4", "rate_1", "rate_0"]).fit(
        continuous.features, continuous.target
    )
    expected = decoder.predict(continuous.features)
    decoder.reset()

    save_decoder(decoder, tmp_path / "subset")
    restored = load_decoder(tmp_path / "subset")

    assert restored.selected_feature_names_ == ("rate_4", "rate_1", "rate_0")
    assert restored.n_observations_ == 3
    assert np.array_equal(restored.predict(continuous.features).data, expected.data)


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------


def test_classifier_predicts_identically_after_round_trip(
    lda, classification, tmp_path: Path
) -> None:
    expected = lda.predict(classification.features)
    posterior = lda.predict_proba(classification.features)

    save_decoder(lda, tmp_path / "lda")
    restored = load_decoder(tmp_path / "lda")

    found = restored.predict(classification.features)
    assert np.array_equal(found.labels, expected.labels)
    assert np.array_equal(found.classes, expected.classes)
    assert np.array_equal(found.time, expected.time)
    assert np.array_equal(
        restored.predict_proba(classification.features).probabilities,
        posterior.probabilities,
    )


def test_classifier_restores_class_order_and_parameters(lda, tmp_path: Path) -> None:
    save_decoder(lda, tmp_path / "lda")
    restored = load_decoder(tmp_path / "lda")

    assert np.array_equal(restored.classes_, lda.classes_)
    assert restored.n_classes_ == lda.n_classes_
    assert np.array_equal(restored.priors_, lda.priors_)
    assert np.array_equal(restored.means_, lda.means_)
    assert np.array_equal(restored.covariance_, lda.covariance_)


def test_declared_class_order_survives_model_disagreement(classification, tmp_path: Path) -> None:
    """The model sorts its own columns; the decoder reports the declared order.

    Saving the permutation rather than re-deriving it is what keeps column
    ``i`` the posterior of ``classes_[i]`` after a load.
    """

    from neurale.decoding import ClassificationTarget

    reversed_order = ClassificationTarget(
        labels=classification.target.labels,
        time=classification.target.time,
        classes=np.array(["rest", "right", "left"]),
    )
    decoder = LDADecoder().fit(classification.features, reversed_order)
    expected = decoder.predict_proba(classification.features)

    save_decoder(decoder, tmp_path / "reordered")
    restored = load_decoder(tmp_path / "reordered")

    assert [str(name) for name in restored.classes_] == ["rest", "right", "left"]
    assert np.array_equal(
        restored.predict_proba(classification.features).probabilities,
        expected.probabilities,
    )


# --------------------------------------------------------------------------------------
# Sequence decoding
# --------------------------------------------------------------------------------------


def test_sequence_decoder_decodes_identically_after_round_trip(beam, steps, tmp_path: Path) -> None:
    expected = beam.decode(steps)

    save_decoder(beam, tmp_path / "beam")
    restored = load_decoder(tmp_path / "beam")

    assert restored.decode(steps) == expected


def test_sequence_decoder_restores_vocabulary_and_prior(
    beam, language_model, tmp_path: Path
) -> None:
    save_decoder(beam, tmp_path / "beam")
    restored = load_decoder(tmp_path / "beam")

    assert restored.vocabulary == beam.vocabulary
    assert restored.vocabulary.unknown == "<unk>"
    assert np.array_equal(restored.transitions, beam.transitions)
    assert np.array_equal(
        restored.language_model.transition_counts_, language_model.transition_counts_
    )
    assert np.array_equal(restored.language_model.unigram_counts_, language_model.unigram_counts_)
    assert restored.language_model.fitted_order_ == language_model.fitted_order_
    assert restored.language_model.fitted_smoothing_ == language_model.fitted_smoothing_
    assert restored.language_model.n_sequences_ == language_model.n_sequences_
    assert restored.language_model.n_tokens_ == language_model.n_tokens_


def test_sequence_decoder_restores_search_configuration(beam, tmp_path: Path) -> None:
    save_decoder(beam, tmp_path / "beam")
    restored = load_decoder(tmp_path / "beam")

    assert restored.beam_width == beam.beam_width
    assert restored.max_completed == beam.max_completed
    assert restored.language_weight == beam.language_weight
    assert restored.length_normalization == beam.length_normalization
    assert restored.inputs == beam.inputs
    assert restored.normalization == beam.normalization
    assert restored.probability_floor == beam.probability_floor


def test_stored_score_width_must_match_vocabulary(beam, steps, tmp_path: Path) -> None:
    """The input-score schema is the vocabulary's width, and a load re-checks it."""

    save_decoder(beam, tmp_path / "beam")
    restored = load_decoder(tmp_path / "beam")

    from neurale.exceptions import ValidationError

    with pytest.raises(ValidationError, match="one column per vocabulary token"):
        restored.update(np.full((1, 3), 1 / 3))
    assert restored.decode(steps) == beam.decode(steps)


# --------------------------------------------------------------------------------------
# Reset and runtime state
# --------------------------------------------------------------------------------------


def test_stateful_decoder_is_saved_post_reset(kalman, continuous, tmp_path: Path) -> None:
    """A prediction moves the filter; what a save records is the model, not the session."""

    kalman.predict(continuous.features)
    assert not np.array_equal(kalman.state_, kalman.initial_state_)

    save_decoder(kalman, tmp_path / "kalman")
    restored = load_decoder(tmp_path / "kalman")

    assert np.array_equal(restored.state_, kalman.initial_state_)
    assert np.array_equal(restored.covariance_, kalman.initial_covariance_)


def test_reset_agrees_across_round_trip(kalman, continuous, tmp_path: Path) -> None:
    save_decoder(kalman, tmp_path / "kalman")
    restored = load_decoder(tmp_path / "kalman")

    kalman.predict(continuous.features)
    restored.predict(continuous.features)
    kalman.reset()
    restored.reset()

    assert np.array_equal(restored.state_, kalman.state_)
    assert np.array_equal(restored.covariance_, kalman.covariance_)
    assert np.array_equal(restored.initial_state_, kalman.initial_state_)
    assert np.array_equal(restored.initial_covariance_, kalman.initial_covariance_)
    assert np.array_equal(
        restored.predict(continuous.features).data,
        kalman.predict(continuous.features).data,
    )


def test_runtime_state_is_stored_only_on_request(kalman, continuous, tmp_path: Path) -> None:
    half = continuous.features.n_frames // 2
    first = FeatureMatrix(
        data=continuous.features.data[:half],
        fs=continuous.features.fs,
        time=continuous.features.time[:half].copy(),
        feature_names=list(continuous.features.feature_names),
        unit=continuous.features.unit,
        shift=continuous.features.shift,
        source_signal=continuous.features.source_signal,
    )
    kalman.predict(first)
    expected_state = np.array(kalman.state_)

    save_decoder(kalman, tmp_path / "resumed", runtime_state=True)
    resumed = load_decoder(tmp_path / "resumed")

    assert np.array_equal(resumed.state_, expected_state)
    assert np.array_equal(resumed.covariance_, kalman.covariance_)
    assert read_manifest(tmp_path / "resumed")["runtime_state"] is True


def test_resumed_decoder_continues_same_sequence(kalman, continuous, tmp_path: Path) -> None:
    """Decoding a recording in two halves must not depend on a save in between."""

    frames = continuous.features
    half = frames.n_frames // 2

    def block(start: int, stop: int) -> FeatureMatrix:
        return FeatureMatrix(
            data=frames.data[start:stop],
            fs=frames.fs,
            time=frames.time[start:stop].copy(),
            feature_names=list(frames.feature_names),
            unit=frames.unit,
            shift=frames.shift,
            source_signal=frames.source_signal,
        )

    kalman.predict(block(0, half))
    expected = kalman.predict(block(half, frames.n_frames))

    kalman.reset()
    kalman.predict(block(0, half))
    save_decoder(kalman, tmp_path / "resumed", runtime_state=True)
    resumed = load_decoder(tmp_path / "resumed")

    assert np.array_equal(resumed.predict(block(half, frames.n_frames)).data, expected.data)


def test_in_progress_sequence_search_resumes(beam, steps, tmp_path: Path) -> None:
    expected = beam.decode(steps)
    beam.reset()
    beam.update(steps[:2])

    save_decoder(beam, tmp_path / "beam", runtime_state=True)
    resumed = load_decoder(
        tmp_path / "beam",
    )

    assert resumed.n_steps == 2
    assert resumed.hypotheses == beam.hypotheses
    assert resumed.update(steps[2:]).hypotheses == expected


def test_resumed_sequence_keeps_starting_prior(
    adaptive_beam, steps, adaptive_language_model, tmp_path: Path
) -> None:
    """A refit between the save and the resume must not reach the running sequence.

    One sequence is decoded under one prior. The language model may have moved
    on by the time an artifact is written -- adaptive spelling refits between
    predictions, not between sessions -- so the artifact stores both the model,
    for the next sequence, and the snapshot the current one began under.
    """

    adaptive_beam.update(steps[:2])
    started_under = adaptive_beam.transitions.copy()
    adaptive_language_model.fit([["a", "a", "b"], ["a"], ["b", "a"]])
    assert not np.array_equal(started_under, adaptive_language_model.transition_log_probabilities_)

    save_decoder(adaptive_beam, tmp_path / "beam", runtime_state=True)
    resumed = load_decoder(tmp_path / "beam")

    assert np.array_equal(resumed.transitions, started_under)
    assert resumed.vocabulary == adaptive_beam.vocabulary
    assert resumed.hypotheses == adaptive_beam.hypotheses
    assert resumed.update(steps[2:]).hypotheses == adaptive_beam.update(steps[2:]).hypotheses


def test_resumed_decoder_adopts_stored_model_on_reset(
    adaptive_beam, steps, adaptive_language_model, tmp_path: Path
) -> None:
    adaptive_beam.update(steps[:2])
    adaptive_language_model.fit([["a", "a", "b"], ["a"], ["b", "a"]])
    refitted = adaptive_language_model.transition_log_probabilities_.copy()

    save_decoder(adaptive_beam, tmp_path / "beam", runtime_state=True)
    resumed = load_decoder(tmp_path / "beam")
    assert not np.array_equal(resumed.transitions, refitted)

    # The next sequence is the point at which a new prior may take effect, and
    # the artifact stores the refitted model precisely so that it can.
    resumed.reset()
    assert np.array_equal(resumed.transitions, refitted)
    assert np.array_equal(
        resumed.language_model.transition_counts_, adaptive_language_model.transition_counts_
    )


def test_saved_sequence_search_defaults_to_new_sequence(beam, steps, tmp_path: Path) -> None:
    beam.update(steps[:2])

    save_decoder(beam, tmp_path / "beam")
    restored = load_decoder(tmp_path / "beam")

    assert restored.n_steps == 0
    assert restored.completed == ()
    assert [item.tokens for item in restored.active] == [()]


@pytest.mark.parametrize("name", ["linear", "ridge", "lda"])
def test_stateless_decoder_refuses_runtime_state(
    name: str, tmp_path: Path, request: pytest.FixtureRequest
) -> None:
    from neurale.exceptions import ValidationError

    decoder = request.getfixturevalue(name)
    with pytest.raises(ValidationError, match="carries no runtime state"):
        save_decoder(decoder, tmp_path / name, runtime_state=True)


# --------------------------------------------------------------------------------------
# The device the fit ran on
# --------------------------------------------------------------------------------------


def test_ambient_runtime_does_not_change_loaded_device(ridge, continuous, tmp_path: Path) -> None:
    """A load reports where the fit ran, not where the loading process would run.

    Re-resolving the device here would make a fitted decoder mean something
    different in each process that opened it -- and for a CPU-only decoder
    under a CUDA request, it would fail outright on an artifact that is
    perfectly valid.
    """

    save_decoder(ridge, tmp_path / "ridge")
    expected = ridge.predict(continuous.features).data

    with runtime_context(device="cuda"):
        restored = load_decoder(tmp_path / "ridge")
        assert restored.device_ == "cpu"
        assert np.array_equal(restored.predict(continuous.features).data, expected)


# --------------------------------------------------------------------------------------
# Configuration survives, so a loaded decoder can be refitted
# --------------------------------------------------------------------------------------


def test_loaded_decoder_refits_to_same_recipe(continuous, tmp_path: Path) -> None:
    """The scaler's kind and options are configuration; its statistics are the fit.

    Restoring only the fitted copy would leave a loaded decoder configured with
    no scaler, and a refit would quietly produce an unscaled model.
    """

    decoder = LinearDecoder(
        scaler=StandardScaler(with_mean=True, with_std=False),
        context=TemporalContext(left=1, right=0),
    ).fit(continuous.features, continuous.target)

    save_decoder(decoder, tmp_path / "linear")
    restored = load_decoder(tmp_path / "linear")

    assert isinstance(restored.scaler, StandardScaler)
    assert restored.scaler.with_mean is True
    assert restored.scaler.with_std is False
    assert not restored.scaler.is_frozen
    assert restored.context == TemporalContext(left=1, right=0)

    refitted = restored.fit(continuous.features, continuous.target)
    assert np.array_equal(refitted.coef_, decoder.coef_)
    assert np.array_equal(
        refitted.predict(continuous.features).data,
        decoder.predict(continuous.features).data,
    )


def test_loaded_decoder_publishes_frozen_scaler(ridge, tmp_path: Path) -> None:
    save_decoder(ridge, tmp_path / "ridge")
    restored = load_decoder(tmp_path / "ridge")

    assert restored.scaler_.is_frozen
    assert restored.scaler_ is not restored.scaler


def test_kalman_decoder_keeps_filtering_configuration(continuous, tmp_path: Path) -> None:
    decoder = KalmanDecoder(
        fit_offsets=False,
        jitter=1e-6,
        innovation_jitter=1e-7,
        missing="predict",
    ).fit(continuous.features, continuous.target)

    save_decoder(decoder, tmp_path / "kalman")
    restored = load_decoder(tmp_path / "kalman")

    assert restored.fit_offsets is False
    assert restored.jitter == 1e-6
    assert restored.innovation_jitter == 1e-7
    assert restored.missing == "predict"


def test_absent_stage_stays_absent(continuous, tmp_path: Path) -> None:
    decoder = LinearDecoder().fit(continuous.features, continuous.target)

    save_decoder(decoder, tmp_path / "bare")
    restored = load_decoder(tmp_path / "bare")

    assert restored.scaler is None
    assert restored.scaler_ is None
    assert restored.context is None
    assert restored.context_ is None
    assert np.array_equal(
        restored.predict(continuous.features).data,
        decoder.predict(continuous.features).data,
    )


def test_target_metadata_survives_including_attrs(continuous, tmp_path: Path) -> None:
    target = SignalArray.from_array(
        continuous.target.data,
        fs=continuous.target.fs,
        time=continuous.target.time.copy(),
        channel_names=["x", "y"],
        channel_types="behavior",
        units="m",
        name="cursor",
        attrs={"task": "center-out", "session": 3, "calibrated": True, "gains": [1.0, 2.0]},
    )
    decoder = LinearDecoder().fit(continuous.features, target)

    save_decoder(decoder, tmp_path / "attrs")
    restored = load_decoder(tmp_path / "attrs")

    predicted = restored.predict(continuous.features)
    assert dict(predicted.attrs) == {
        "task": "center-out",
        "session": 3,
        "calibrated": True,
        "gains": (1.0, 2.0),
    }
