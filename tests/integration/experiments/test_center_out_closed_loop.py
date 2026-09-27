# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Installed-wheel acceptance tests for adaptive Center-Out sessions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import neurale.pipeline as pipeline
import neurale.streaming as streaming
from neurale.decoding import KalmanDecoder, LinearDecoder
from neurale.devices.simulation import SimulatedNeuralDevice
from neurale.experiments import center_out as co
from neurale.experiments.center_out_training import AssistanceBlock, OnlineDecoderKind
from neurale.signal.simulation import SignalGenerator

pytestmark = pytest.mark.slow


def _task() -> object:
    layout = co.build_radial_layout(co.RadialLayoutRequest(radius=0.25))
    return co.CenterOutTask(
        geometry_unit=co.GeometryUnit.NORMALIZED,
        layout=layout,
        acceptance=0.08,
        cursor_extent=0.025,
        movement_timeout_seconds=3.0,
        selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS,
        seed=17,
    )


def _presentation_config() -> object:
    from neurale.experiments.presentation import CenterOutPresentationConfig
    from neurale.experiments.presentation._center_out import _with_visibility

    return _with_visibility(
        CenterOutPresentationConfig(
            title="PyNeurale formal Center-Out closed loop",
            monitor=0,
            vsync=False,
        ),
        visible=False,
    )


def _named_json_records(reader: object, schema_id: str, name: str) -> list[dict[str, object]]:
    columns = reader.read_records(schema_id)
    return [
        json.loads(text)
        for record_name, text in zip(columns["name"], columns["text"], strict=True)
        if record_name == name
    ]


def _verify_recording(output: Path, outcome: object, *, presentation: bool) -> None:
    from neurale.io.nrf import NrfReader

    reader = NrfReader(output)
    try:
        frame_count = reader.record_extent("native-frames-v1")
        if frame_count <= 0:
            raise RuntimeError("the unified NRF does not contain acquisition frames")
        if reader.record_extent("native-signal-blocks-v1") != frame_count:
            raise RuntimeError("the unified NRF frame and signal-block counts do not match")
        if reader.record_extent("trials-v1") != 2:
            raise RuntimeError("the unified NRF does not contain both Center-Out trials")
        if reader.record_extent("drops-v1") or reader.record_extent("faults-v1"):
            raise RuntimeError("the unified NRF contains a drop or fault")

        blocks = _named_json_records(reader, "assistance-v1", "center_out.assistance_block")
        if [(value["assistance"], value["trials"]) for value in blocks] != [
            (1, 1),
            (0.9, 1),
        ]:
            raise RuntimeError("the recorded assistance schedule is not 100% then 90%")
        guidance_configs = _named_json_records(
            reader, "assistance-v1", "center_out.guidance_config"
        )
        if len(guidance_configs) != 1 or guidance_configs[0]["resolver_version"] != 1:
            raise RuntimeError("the NRF does not identify automatically resolved guidance")
        state_columns = reader.read_records("experiment-state-v1")
        if not any(
            name == "center_out.transition" and int(time_ns) == 0
            for time_ns, name in zip(state_columns["time_ns"], state_columns["name"], strict=True)
        ):
            raise RuntimeError("the recorded experiment does not start at time zero")
        assisted = _named_json_records(reader, "assistance-v1", "center_out.assisted_velocity")
        assistance_columns = reader.read_records("assistance-v1")
        assisted_times = [
            int(time_ns)
            for time_ns, name in zip(
                assistance_columns["time_ns"], assistance_columns["name"], strict=True
            )
            if name == "center_out.assisted_velocity"
        ]
        if not assisted_times:
            raise RuntimeError("the unified NRF has no assisted velocity observations")
        computer_guided = [value for value in assisted if value["assistance"] == 1]
        if not computer_guided:
            raise RuntimeError("the unified NRF has no 100% assistance observations")
        if any(value["assisted"] != value["guidance"] for value in computer_guided):
            raise RuntimeError("100% assistance did not apply the exact computer guidance velocity")
        if not any(
            any(component != 0 for component in value["guidance"]) for value in computer_guided
        ):
            raise RuntimeError("100% assistance never moved the cursor under computer control")

        provenance = _named_json_records(reader, "labels-v1", "center_out.observation_provenance")
        observed_versions = sorted({int(value["decoder_version"]) for value in provenance})
        if not observed_versions or max(observed_versions) < 2:
            raise RuntimeError("the NRF has no observation produced by an online-fitted decoder")

        publications = _named_json_records(
            reader, "task-variables-v1", "center_out.decoder_publication"
        )
        expected = {
            value.version: (
                value.plan_fingerprint,
                value.training_blocks,
                value.training_trials,
                value.completed_trials_at_publication,
            )
            for value in outcome.publications
        }
        recorded = {int(value["version"]): value for value in publications}
        if 1 not in recorded or len(str(recorded[1]["plan_fingerprint"])) != 64:
            raise RuntimeError("the initial decoder pipeline fingerprint is missing from the NRF")
        for version, evidence in expected.items():
            value = recorded.get(version)
            actual = (
                None
                if value is None
                else (
                    value["plan_fingerprint"],
                    value["training_blocks"],
                    value["training_trials"],
                    value["completed_trials"],
                )
            )
            if actual != evidence:
                raise RuntimeError(
                    f"decoder publication {version} does not match its recorded NRF provenance"
                )

        presentation_records = _named_json_records(
            reader, "experiment-state-v1", "center_out.presentation"
        )
        presented = [
            value
            for value in presentation_records
            if value["event"] == 2 and value["has_software_times"]
        ]
        if presentation and (
            not presented or any(not value["has_source_ordinal"] for value in presented)
        ):
            raise RuntimeError(
                "the NRF presentation evidence lacks source identity or software times"
            )
        presentation_configs = _named_json_records(
            reader, "task-variables-v1", "center_out.presentation_config"
        )
        presentation_acceptance = _named_json_records(
            reader, "task-variables-v1", "center_out.presentation_acceptance"
        )
        presentation_windows = _named_json_records(
            reader, "task-variables-v1", "center_out.presentation_window"
        )
        if presentation:
            if len(presentation_configs) != 1:
                raise RuntimeError("the NRF does not contain one presentation configuration")
            resolved = presentation_configs[0]
            if (
                resolved["target_radius"] != 0.08
                or resolved["cursor_radius"] != 0.025
                or resolved["logical_width"] != resolved["logical_height"]
            ):
                raise RuntimeError("the NRF presentation geometry was not derived from the task")
            if (
                len(presentation_acceptance) != 1
                or not presentation_acceptance[0]["cursor_radius_matches_task_extent"]
            ):
                raise RuntimeError("the NRF does not link drawn and scored cursor geometry")
            if len(presentation_windows) != 1 or presentation_windows[0] != {
                "window_width": 800,
                "window_height": 600,
                "monitor_index": 0,
                "swap_interval": 0,
                "fullscreen": False,
                "resizable": False,
                "visible": False,
            }:
                raise RuntimeError("the NRF does not contain the resolved presentation window")
    finally:
        reader.close()


def _run_session(decoder_kind: OnlineDecoderKind, output: Path, *, presentation: bool) -> None:
    device = SimulatedNeuralDevice(
        SignalGenerator.sine(2, 100.0, (10.0, 20.0), amp=(1.0, 0.5), phase=(0.0, 0.5)),
        samples_per_frame=1,
        physical_unit="dimensionless",
        paced=True,
    )
    feature_plan = pipeline.PipelinePlan(
        stages=(
            pipeline.FeatureStage(
                window_seconds=0.32,
                update_interval_seconds=0.04,
                features=(
                    pipeline.MultitaperBandpowerFeature(
                        bands=(pipeline.Band("alpha", 8.0, 12.0),),
                        time_bandwidth=2.5,
                        n_tapers=3,
                    ),
                    pipeline.HilbertEnvelopeFeature(
                        bands=(pipeline.Band("alpha", 8.0, 12.0),),
                        filter_order=2,
                    ),
                    pipeline.LmpFeature(cutoff_hz=4.0, filter_order=2),
                ),
            ),
        )
    )
    from neurale.recording import RecorderConfig, SessionRecorder

    recorder = SessionRecorder.create(RecorderConfig(path=output), device)
    session = co.CenterOutSession(
        co.CenterOutProtocol(
            task=_task(),
            assistance_blocks=(AssistanceBlock(1.0, 1), AssistanceBlock(0.9, 1)),
        ),
        feature_plan,
        device,
        (
            LinearDecoder()
            if decoder_kind == OnlineDecoderKind.LINEAR
            else KalmanDecoder(jitter=1e-6, innovation_jitter=1e-6)
        ),
        presentation_config=_presentation_config() if presentation else None,
        recording=recorder,
    )
    outcome = session.run()
    if outcome.runtime_status != streaming.StreamStatus.OK:
        raise RuntimeError(f"runtime failed: {outcome.runtime_status}")
    if not outcome.experiment_trace_complete or outcome.trace_losses:
        raise RuntimeError("Center-Out experiment trace is incomplete")
    if outcome.completed_trials != 2 or session.snapshot.state != co.CenterOutState.COMPLETE:
        raise RuntimeError("the two assistance trials did not complete")
    if not outcome.publications or outcome.active_decoder_version < 2:
        raise RuntimeError("the fitted decoder was not activated online")
    if presentation and (outcome.presented_frames <= 0 or outcome.presentation_state_drops):
        raise RuntimeError("Center-Out presentation evidence is incomplete")

    from neurale.recording import ReplayConfig, build_replay_image

    if outcome.recording is None or not outcome.recording.complete:
        raise RuntimeError("the unified NRF recording is incomplete")
    _verify_recording(output, outcome, presentation=presentation)
    image = build_replay_image(
        output, ReplayConfig(mode="exact_frames"), path=output.with_suffix(".nrimg")
    )
    try:
        if image.metadata.n_items <= 0:
            raise RuntimeError("the unified NRF produced an empty replay image")
    finally:
        image.close()


@pytest.mark.parametrize(
    "decoder_kind", (OnlineDecoderKind.LINEAR, OnlineDecoderKind.KALMAN), ids=("linear", "kalman")
)
def test_headless_center_out(decoder_kind: OnlineDecoderKind, tmp_path: Path) -> None:
    _run_session(
        decoder_kind, tmp_path / f"center-out-{decoder_kind.value}.nrf", presentation=False
    )


@pytest.mark.visualization
def test_center_out_presentation(tmp_path: Path) -> None:
    _run_session(OnlineDecoderKind.LINEAR, tmp_path / "center-out-presented.nrf", presentation=True)
