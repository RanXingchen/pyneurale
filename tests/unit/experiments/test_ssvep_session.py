# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""Closed-loop acceptance uses ordinary fixed tones, not target-driven signals."""

import json

import numpy as np
import pytest

from neurale import pipeline
from neurale.data import FeatureMatrix
from neurale.decoding import LDADecoder
from neurale.devices.simulation import SimulatedNeuralDevice
from neurale.experiments import ssvep
from neurale.models.preprocessing import MinMaxScaler, StandardScaler
from neurale.recording import RecorderConfig, SessionRecorder
from neurale.signal.simulation import SignalGenerator
from neurale.streaming import StreamStatus


def task(n_targets=2):
    return ssvep.SSVEPTask(
        targets=[ssvep.SSVEPTarget(i + 1, f) for i, f in enumerate((8, 15, 10, 12)[:n_targets])],
        cue_duration=0.1,
        stimulation_duration=0.8,
        decision_timeout=0.5,
        feedback_duration=0.1,
        inter_trial=0.1,
        seed=42,
    )


def test_protocol_validation():
    protocol = ssvep.SSVEPProtocol(task())
    assert protocol.calibration_trials == 6 and protocol.trials == 14
    with pytest.raises(ValueError, match="at least 2"):
        ssvep.SSVEPProtocol(task(), calibration_trials_per_target=1)
    with pytest.raises(TypeError, match="integer"):
        ssvep.SSVEPProtocol(task(), evaluation_trials=True)


def test_stopped_trial_does_not_record_missing_aggregate_as_zero(tmp_path):
    from neurale.io.nrf import NrfReader

    device = SimulatedNeuralDevice(
        SignalGenerator.tones(2, 256.0, [8.3, 15.2]), samples_per_frame=4, paced=True
    )
    plan = pipeline.PipelinePlan(
        stages=(
            pipeline.FeatureStage(
                window_seconds=0.5,
                update_interval_seconds=0.125,
                features=(
                    pipeline.MultitaperBandpowerFeature(
                        bands=(pipeline.Band("eight", 6, 10),), time_bandwidth=1.5
                    ),
                ),
            ),
        )
    )
    path = tmp_path / "stopped_ssvep.nrf"
    recorder = SessionRecorder.create(RecorderConfig(path=path), device)
    session = ssvep.SSVEPSession(
        ssvep.SSVEPProtocol(task(), 2, 1),
        plan,
        device,
        LDADecoder(),
        recording=recorder,
    )
    try:
        session.stop()
        outcome = session.run()
        assert outcome.stopped
        with NrfReader.open(path) as reader:
            records = [
                record
                for schema_id in reader.record_schema_ids()
                for record in reader.iter_records(schema_id)
            ]
        trials = [record for record in records if "trial_id" in record]
        assert trials
        assert all(json.loads(record["label"])["feature_windows"] == 0 for record in trials)
        assert not any(record.get("name") == "ssvep.trial_feature" for record in records)
    finally:
        device.close()


@pytest.mark.parametrize(
    ("n_targets", "scaler"),
    [(2, None), (2, StandardScaler()), (2, MinMaxScaler()), (4, StandardScaler())],
)
def test_fixed_tone_closed_loop_and_recording(tmp_path, n_targets, scaler):
    from neurale.io.nrf import NrfReader

    device = SimulatedNeuralDevice(
        SignalGenerator.tones(2, 256.0, [8.3, 15.2], amps=[4e-6, 1e-6]),
        samples_per_frame=4,
        paced=True,
    )
    plan = pipeline.PipelinePlan(
        stages=(
            pipeline.FeatureStage(
                window_seconds=0.5,
                update_interval_seconds=0.125,
                features=(
                    pipeline.MultitaperBandpowerFeature(
                        bands=(pipeline.Band("eight", 6, 10), pipeline.Band("fifteen", 13, 17)),
                        time_bandwidth=1.5,
                    ),
                ),
            ),
        )
    )
    decoder = LDADecoder(scaler=scaler, shrinkage="auto")
    calibration = 3 * n_targets
    path = tmp_path / "ssvep.nrf"
    recorder = SessionRecorder.create(RecorderConfig(path=path), device)
    session = ssvep.SSVEPSession(
        ssvep.SSVEPProtocol(task(n_targets), 3, 2), plan, device, decoder, recording=recorder
    )
    try:
        outcome = session.run()
        assert len(outcome.calibration_trials) == calibration
        assert len(outcome.evaluation_trials) == 2
        assert outcome.decoder_version == 2 and len(outcome.decoder_fingerprint) == 64
        assert outcome.experiment_trace_complete and outcome.trace_drops == 0
        assert outcome.runtime_status == StreamStatus.OK
        from neurale.streaming import ObservationTiming

        aggregated = session._native.feature_schema
        assert aggregated.signals[0].observation_timing == ObservationTiming.IRREGULAR
        assert aggregated.signals[0].fs.numerator == 0
        assert aggregated.feature_sets[0].shift_ns == 0
        assert aggregated.feature_sets[0].window_length_ns == 800_000_000
        # Compare native decisions to the fitted public decoder, without target labels.
        rows = np.asarray(session._native.features)[calibration:]
        features = FeatureMatrix(
            data=rows,
            fs=None,
            time=(np.asarray(session._native.feature_times_ns)[calibration:] - session._epoch)
            / 1e9,
            feature_names=session._schema.feature_sets[0].feature_names,
        )
        expected = decoder.predict(features).labels
        assert [
            trial.selection.selected_id for trial in outcome.evaluation_trials
        ] == expected.tolist()
        with NrfReader.open(path) as reader:
            assert reader.complete
            assert reader.read_stream("neural").shape[0] > 0
            records = [
                record
                for schema_id in reader.record_schema_ids()
                for record in reader.iter_records(schema_id)
            ]
            publications = [
                json.loads(record["text"])
                for record in records
                if record.get("name") == "ssvep.decoder_publication"
            ]
            assert len(publications) == 1
            assert publications[0]["fingerprint"] == outcome.decoder_fingerprint
            assert publications[0]["training_trials"] == calibration
            chunks = sorted(
                (
                    json.loads(record["text"])
                    for record in records
                    if record.get("name") == "ssvep.decoder_plan"
                ),
                key=lambda chunk: chunk["offset"],
            )
            document = json.loads("".join(chunk["json"] for chunk in chunks))
            restored = pipeline.PipelinePlan.from_document(document)
            assert restored.fingerprint == outcome.decoder_fingerprint
            assert isinstance(restored.stages[0], pipeline.LdaDecoderStage)
            assert restored.stages[0].fitted_feature_contract.observation_rate_numerator == 0
            assert restored.stages[0].fitted_feature_contract.shift_ns == 0
            assert pipeline.compile_pipeline(restored).stage_count == 1
            assert any(record.get("name") == "ssvep.feature" for record in records)
            intervals = [
                value
                for record in records
                if "trial_id" in record
                for value in [json.loads(record["label"])]
                if "feature_interval_key" in value
            ]
            assert len(intervals) == calibration + 2
            for index, value in enumerate(intervals):
                assert value["feature_interval_key"] == index + 1
                assert value["feature_end_ns"] - value["feature_start_ns"] == 800_000_000
                assert value["feature_center_ns"] == value["feature_start_ns"] + 400_000_000
                assert value["feature_center_ns"] == (
                    session._native.feature_times_ns[index] - session._epoch
                )
            recorded = np.zeros_like(np.asarray(session._native.features))
            for record in records:
                if record.get("name") == "ssvep.trial_feature":
                    value = json.loads(record["text"])
                    recorded[value["trial"], value["column"]] = value["value"]
            np.testing.assert_array_equal(recorded, np.asarray(session._native.features))
        with pytest.raises(RuntimeError, match="only run once"):
            session.run()
    finally:
        device.close()
