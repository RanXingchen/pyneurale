# Center-Out closed loop

The high-level `CenterOutSession` owns the native pipeline, task, online decoder
training, control, and optional presentation and recording.

## Run the demo

This demo opens a Center-Out presentation window. It needs a supported desktop
wheel or a source build with `NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON`, plus a
desktop OpenGL display. The {doc}`getting started <../getting_started/index>`
build uses `OFF` and cannot open the window.

```python
from neurale.decoding import LinearDecoder
from neurale.devices.simulation import SimulatedNeuralDevice
from neurale.experiments import center_out as co
from neurale.experiments.presentation import CenterOutPresentationConfig
from neurale.pipeline import FeatureStage, PipelinePlan
from neurale.signal.simulation import SignalGenerator

task = co.CenterOutTask(
    geometry_unit=co.GeometryUnit.NORMALIZED,
    layout=co.build_radial_layout(co.RadialLayoutRequest(radius=0.8)),
    acceptance=0.08, cursor_extent=0.05, movement_timeout_seconds=3,
    selection=co.TargetSelectionPolicy.BALANCED_SHUFFLED_CYCLES,
)
protocol = co.CenterOutProtocol(task=task, assistance_blocks=(
    co.AssistanceBlock(1.0, 8), co.AssistanceBlock(0.5, 5),
))
device = SimulatedNeuralDevice(
    SignalGenerator.sine(8, 1000, [7, 11, 13, 17, 19, 23, 29, 31]),
    samples_per_frame=4, physical_unit="dimensionless",
)
session = co.CenterOutSession(
    protocol, PipelinePlan(stages=(FeatureStage(window_seconds=0.1),)),
    device, LinearDecoder(), presentation_config=CenterOutPresentationConfig(),
)
result = session.run()
print(result.completed_trials, result.active_decoder_version)
```

The window shows 13 simulated trials. The first eight provide data for online
decoder fitting; the next five use 50% assistance while the fitted decoder is
activated at a trial boundary. The final line prints `13 2` when all trials
complete and decoder version 2 is active. This demonstrates the software loop,
not real-device performance or physical display timing.

## Assistance and training

Block trial counts define session length. At assistance `1.0`, guidance controls
the cursor and decoded velocity has zero weight. Assistance changes only after
`trial_stop`. The session derives and validates guidance against the task's
movement deadlines; use `CenterOutGuidanceConfig` only for a reviewed fixed
profile.

Pass the protocol, a declarative feature `PipelinePlan`, a device, and an
unfitted `LinearDecoder` or `KalmanDecoder` to `CenterOutSession`, as in the
demo. The session compiles the plan itself. Its first assisted block supplies
training data; a fitted native decoder activates at a subsequent trial boundary.
`OnlineDecoderTrainingConfig` overrides the default fit schedule or label lag
when the protocol requires it. A positive `label_lag_seconds` pairs an earlier
feature observation with later intended guidance velocity.

Check `result.runtime_status`, `result.experiment_trace_complete`,
`result.training_capture_drops`, and, when recording, `result.recording.complete`
before treating a run as complete. For a longer configurable run with optional
NRF recording, see `examples/center_out_closed_loop.py` and the
[recording guide](recording_replay.md).

Optional presentation derives its geometry from the task. Software submit and
swap-return timestamps are recorded, but do not measure physical pixel onset.
