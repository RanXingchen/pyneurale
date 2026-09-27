# SSVEP closed loop

`neurale.experiments.ssvep` provides a native selection task and a high-level
`SSVEPSession` that connects acquisition, a native feature pipeline, labelled
calibration, LDA deployment, task feedback and optional persistent NRF recording.

## Run the demo

This demo opens an SSVEP window and runs calibration followed by evaluation.
It needs a supported desktop wheel or a source build with
`NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON`, plus a desktop OpenGL display.
The {doc}`getting started <../getting_started/index>` headless build cannot
open the window.

```python
from neurale import pipeline
from neurale.decoding import LDADecoder
from neurale.devices.simulation import SimulatedNeuralDevice
from neurale.experiments import ssvep
from neurale.experiments.presentation import SSVEPDisplayConfig
from neurale.models.preprocessing import StandardScaler
from neurale.signal.simulation import SignalGenerator

frequencies = (8.0, 10.0, 12.0, 15.0)
task = ssvep.SSVEPTask(
    targets=[ssvep.SSVEPTarget(i + 1, f) for i, f in enumerate(frequencies)],
    cue_duration=0.5, stimulation_duration=1.6, decision_timeout=0.5,
    feedback_duration=0.5, inter_trial=0.2, seed=42,
)
bands = tuple(
    pipeline.Band(f"target-{i + 1}", f - 0.75, f + 0.75)
    for i, f in enumerate(frequencies)
)
feature_plan = pipeline.PipelinePlan(stages=(
    pipeline.LineNoiseFilterStage(harmonics=1),
    pipeline.FilterStage(filter_type="bandpass", cutoff_hz=(5.0, 40.0)),
    pipeline.FeatureStage(
        window_seconds=1.0, update_interval_seconds=0.25,
        features=(pipeline.MultitaperBandpowerFeature(bands=bands, time_bandwidth=1.5),),
    ),
))
device = SimulatedNeuralDevice(
    SignalGenerator.noise(4, 256.0, seed=7),
    samples_per_frame=4, channel_names=("O1", "Oz", "O2", "POz"), paced=True,
)

try:
    session = ssvep.SSVEPSession(
        ssvep.SSVEPProtocol(task, calibration_trials_per_target=2, evaluation_trials=4),
        feature_plan, device,
        LDADecoder(scaler=StandardScaler(), shrinkage=0.5),
        presentation_config=SSVEPDisplayConfig(),
    )
    result = session.run()
    print(len(result.evaluation_trials), result.correct_count, result.runtime_status.name)
finally:
    device.close()
```

The window shows eight labelled calibration trials and four evaluation trials.
The seeded simulated signal does not encode the cued target, so `correct_count`
is not a decoding-accuracy benchmark. During evaluation, a correct
selection bursts turquoise; a wrong selection fills the selected target red
while the intended target remains amber. No selection before the deadline is
a timeout, not a wrong prediction. The README GIF uses scripted choices to show
both feedback states, not EEG decoding.

For a longer configurable run with optional NRF recording or monitor selection,
see `examples/ssvep_closed_loop.py` (`--show`, `--output`, and `--monitor`).
The task defines frequencies and phase durations; the protocol defines the
balanced calibration count (at least two trials per target) and evaluation
count. This small calibration set uses explicit covariance shrinkage. The decoder
must initially be an unfitted `ClassificationDecoder` with native deployment
support (currently `LDADecoder`). A session runs once;
`stop()` requests termination from another thread.

Only feature windows wholly inside stimulation contribute to that trial's mean
feature vector. Empty or stopped-before-ready intervals produce no prediction or
`ssvep.trial_feature` record. Predictions are accepted only for the active trial
before its deadline. Target IDs label calibration trials but are not classifier
inputs. Training occurs after calibration, outside the realtime acquisition and
task threads; the fitted model then serves evaluation.

The demo uses 256 Hz so a 1-second window and 0.25-second update correspond
to exactly 256 and 64 samples. The longer example also reports a model
fingerprint, trace drops and software presentation statistics. No target-aware
simulator or special device API is required.

When supplied, the recorder persists acquisition, features, decoder publication,
task transitions, selections, trials, and software presentation evidence. See
the [recording guide](recording_replay.md) for inspecting the finished NRF.
`result.calibration_trials` and `result.evaluation_trials` keep the phases
separate. Queue overflow, incomplete calibration and training errors fail the
session rather than appearing as an incorrect choice.

## Display configuration

`SSVEPDisplayConfig` retains `title`, `window_size`, `fullscreen` and `monitor`.
The optional `positions` and `target_size` control stimulus geometry:

```python
display = SSVEPDisplayConfig(
    fullscreen=True,
    positions=((-0.6, 0.6), (0.6, 0.6), (-0.6, -0.6), (0.6, -0.6)),
    target_size=0.3,
)
```

Positions are target centers in task order, x rightwards and y upwards. Both
axes span -1 to 1 within a square workspace; `target_size` is the circle diameter
in those same units. The workspace is letterboxed, so changing window aspect
does not stretch targets. These are not degrees of visual angle or millimetres.
Omitting either option uses its automatic grid/size default. Centers must match
the number of targets, and target bounding boxes must fit without
overlap. Native validation runs before opening a window.

## Advanced state-machine API

The explicit-time machine remains available for custom integration. Ordinary
sessions do not require the following driving code. The native catalog
`stimulus_id` defaults to 1 and only needs overriding when managing a catalog.

```python
from neurale.experiments import ContractStatus, SelectionEvent, ssvep

task = ssvep.SSVEPTask(
    targets=[ssvep.SSVEPTarget(i + 1, f) for i, f in enumerate((8, 10, 12, 15))],
    seed=42,
    cue_duration=1,
    stimulation_duration=2.5,
    decision_timeout=1,
    feedback_duration=0.5,
    inter_trial=0.5,
)
machine = ssvep.SSVEPMachine()
status, result = machine.start(paradigm=4, task=task, time_ns=0, trials=40)
assert status == ContractStatus.OK

# This synthetic choice demonstrates task semantics, not EEG decoding.
trial = result.snapshot.trial
selection = SelectionEvent(
    time_ns=3_000_000_000,
    sequence=0,
    trial=trial,
    paradigm=4,
    selected_id=trial.target_id,
    intended_id=trial.target_id,
    correct=True,
)
status, result = machine.step(3_000_000_000, selection)
assert status == ContractStatus.OK
assert result.snapshot.state == ssvep.SSVEPState.STIMULATION
status, result = machine.step(3_500_000_000)
assert status == ContractStatus.OK
assert result.snapshot.state == ssvep.SSVEPState.FEEDBACK
status, result = machine.step(4_000_000_000)
assert status == ContractStatus.OK and result.trial_decided
```

`SSVEPTask` takes keyword-only parameters and validates them at construction,
raising `ValueError` with the rejected field. All five durations must be finite
and positive; `start(..., trials=...)` supplies the positive session trial count.
The same immutable task can be reused for sessions of different lengths.
There are 2–64 targets with
unique nonzero IDs and distinct finite positive frequencies. Frequencies are
semantic catalog values: this API does not validate them against a display's
refresh rate. No rendering parameters or classifier scores enter the machine.

`prepare_trial(task, ordinal)` uses the machine's balanced-cycle sampler:
each complete cycle contains every target once. A partial final cycle need not
be balanced.

## Input and timing

The states are `IDLE`, `CUE`, `STIMULATION`, `AWAIT_DECISION`, `FEEDBACK`,
`INTER_TRIAL`, and `COMPLETE`. Stimulus duration is fixed. A selection during
stimulation is saved and feedback starts no earlier than stimulus end. With no
selection at that boundary, the machine waits at most `decision_timeout`;
selection then starts feedback immediately, or the deadline produces TIMEOUT.
Correct, incorrect, and timeout trials all advance the target ordinal once.

The acceptance interval is `[stimulus_start, decision_deadline)`. Exact ties at
the deadline time out. A selection carries the **current observation time**,
equal to `step(time_ns)`, not an earlier decoder timestamp. It must be discrete,
match all fields of the current trial identity and paradigm, name a configured
target, and have an internally consistent intended target and correctness flag.
The decoder should not receive the intended target: its adapter adds the trial
context only after prediction.

At most one selection is accepted per trial. Accepted input sequence numbers
must increase across the session; they are independent of output record
sequence numbers. Duplicate input, a second selection, wrong identity, a choice
during CUE, or regressed time is an error leaving state and output unchanged.
A structurally valid late choice for the current trial reports `EXPIRED` and
still advances timers. It is never forwarded to the next trial. No synthetic
selection is made for timeout.

Timers use half-open intervals. A delayed call records elapsed phase boundaries
at their scheduled times, not at the polling time. Each step completes at most
one trial. When `result.settled` is false, consume its output and call
`step(the_same_time)` without resubmitting a selection until settled. A repeated
settled time emits nothing. Polling density does not change timed boundaries,
but **when a selection is observed does affect feedback timing**.

If a choice is supplied exactly at stimulus end, no wait phase is entered. If
a separate timer-only call already entered the wait at that same time, its
onset/request remains observable; the subsequent choice closes it immediately.
There is still no zero-length wait interval in the completed trial.

## Results, stop and ownership

`SSVEPStepResult` exposes the snapshot, transitions, events, presentation
requests, selection disposition, and at most one completed `SSVEPTrial`.
`validate(trial)` checks completed records.

The trial interval extends from cue onset through feedback end. At that instant
`trial_decided` becomes true, `completed` increments, and `trial_stop` is emitted.
Inter-trial rest is separate; the final trial's rest also runs before session
completion. `trial_outcome` is emitted when feedback starts. A stored selection
alone does not end a trial or make its snapshot outcome successful.

`stop(time_ns)` aborts the current unfinished trial and emits terminal black
presentation plus session stop. During rest it creates no second trial record.
Stop takes precedence over unobserved timer boundaries: it closes the currently
observed phase at the supplied time, without synthesizing missed trials. A
trial stopped during feedback is ABORTED even if a provisional successful
outcome was emitted earlier. Terminal and idle calls return NOT_RUNNING;
`reset()` returns the machine, configuration and counters to their initial state.

`SSVEPDisplay` draws the current machine snapshot, not a backlog of historical
requests. The state machine requests presentation; it does not claim a display
actually showed it.

The high-level session connects this machine to native acquisition and decoding.
Custom integrations remain responsible for their own clock and recording contracts.


## Native experiment window

Escape or closing the window stops the experiment. Evaluation selections come
from the fitted decoder; keyboard and automatic synthetic selections are not
provided.

`SSVEPDisplay` is the lower-level presenter for custom drivers. Its constructor
accepts only a task and optional display configuration. `poll()` processes window
events and `should_close` reports Escape/close. `render(snapshot, time_ns)` returns
the shared typed software timing result and raises on presentation failure.
Custom drivers use `renderer_monotonic_now_ns()` consistently for task and render
times. No clock-origin mapping is required. All window operations run on the
thread that created the display. Importing the module creates no window.

Targets use the configured positions or an automatic grid in task order. An amber ring cues
the intended target; all targets flicker during stimulation. Correct feedback
shows a short turquoise burst around the selected target; an incorrect choice
fills the selected target red while the intended target stays amber. A timeout
leaves only the intended target amber.
Waiting, rest and completion show only the dark background.
Target luminance is `0.5 + 0.5*cos(2*pi*frequency*elapsed)`, with zero phase at
the scheduled stimulation onset. Gamma correction is not applied. Sampling
uses the supplied experiment time, so a missed frame does not slow the phase
by incrementing a software frame counter. The lower-level renderer rejects expired active snapshots. The high-level session
renders black for an expired asynchronous snapshot and records the miss explicitly
in `expired_presentation_frames` and presentation evidence.

Vsync is required. Fullscreen preserves the selected monitor's current mode;
this API does not switch it to 240 Hz. Configure the desired mode in the OS.
Opening rejects frequencies at or above half the reported monitor refresh rate.
This check is not a guarantee of display accuracy. Window compositing, dropped
frames, scanout and panel response still affect the physical stimulus. The
software swap timestamps and the monitor's reported refresh rate do not measure
photon onset. The session's frame loop is Python-driven; this is a runnable
presentation interface, not evidence of validated 240 Hz or EEG synchronization.
