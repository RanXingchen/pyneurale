# Realtime pipelines

`neurale.pipeline` lets a researcher select, order, configure, save, and load
the built-in native processing stages used by `StreamRunner`. The Python layer
is a control plane: it constructs the chain before runtime startup. Frame
processing remains in C++ and does not call Python.

## Build a plan

A `PipelinePlan` is an ordered, non-empty tuple of immutable stage
specifications. Order is execution order.

```python
from neurale.pipeline import (
    BadChannelRemovalStage,
    FilterStage,
    LineNoiseFilterStage,
    PipelinePlan,
    SpatialReferenceStage,
    compile_pipeline,
)

plan = PipelinePlan(
    (
        BadChannelRemovalStage(("Fp1", "Fp2")),
        SpatialReferenceStage(reference_channels=(0, 1), statistic="mean"),
        LineNoiseFilterStage(),
        FilterStage(
            filter_type="bandpass",
            cutoff_hz=(1.0, 200.0),
            filter_order=4,
        ),
    )
)
pipeline = compile_pipeline(plan)
```

Compilation constructs one owning native processor chain. It does not prepare
or start a runtime. The following attachment sketch assumes that `schema`,
`native_source`, `native_actuator`, and `native_safety_controller` have been
created by the acquisition and control integration:

```python
import neurale.streaming as streaming

realtime_config = streaming.RealtimeConfig()
runner = pipeline.create_runner(
    schema,
    realtime_config,
    native_source,
    native_actuator,
    profile="realtime",
    safety_controller=native_safety_controller,
)
runner.prepare()
runner.arm()
runner.run()
```

`RealtimeConfig()` is directly usable. It contains a 100 ms latency budget and
a one-second source timeout; native frame storage, processor leases, queue
capacities, discontinuity storage, output staleness, and watchdog cadence are
derived during `prepare()`. To change operational intent, use for example
`RealtimeConfig(latency_budget_seconds=0.05, source_timeout_seconds=2.0)`.
CPU placement and scheduling remain optional advanced
`RealtimePlatformConfig` settings. The execution profile selects strict,
best-effort, or disabled platform application; users do not configure a second
runtime mode.

The `realtime` execution profile still enforces the existing `StreamRunner`
contract: source, processor, actuator, and safety controller must be native;
there is no Python or CPU fallback. One compiled pipeline can attach to only
one runner because its stages may hold online state. Compile the plan again for
another runner.

## Available stages

Version 1 supports these fixed built-ins:

- bad-channel removal by channel name, index, or measured impedance:
  `BadChannelRemovalStage`
- spatial reference: `SpatialReferenceStage`
- 50/60 Hz mains-interference rejection: `LineNoiseFilterStage`
- low-pass, high-pass, band-pass, and band-stop filtering: `FilterStage`
- rational resampling with a library-designed anti-alias filter: `ResampleStage`
- synchronized same-input LMP, Hilbert envelope, and multitaper bandpower
  extraction: `FeatureStage`
- threshold spike detection: `SpikeDetectorStage`
- fitted linear, Kalman and LDA decoding: `LinearDecoderStage`,
  `KalmanDecoderStage`, `LdaDecoderStage`

`supported_stage_kinds()` returns the stable identifiers accepted by plan
documents. This is a fixed whitelist, not a plugin registry. A plan cannot name
an import path or execute code from its JSON.

`FilterStage` accepts scientific design parameters, not coefficients. At native
`prepare()` time it validates the cutoffs against the actual input sampling
rate, designs the SOS cascade once, and retains that representation inside the
native realtime processor. The default is a fourth-order Butterworth design;
`filter_kind="bessel"` and `filter_kind="elliptic"` are available when the
experiment requires them. Elliptic designs additionally require
`passband_ripple_db` and `stopband_attenuation_db`. `ResampleStage` likewise
derives its bounded anti-alias filter from `up` and `down`; neither stage has a
public coefficient argument.

`BadChannelRemovalStage(("Fp1", "Fp2"))` resolves names against the prepared
input schema, preserves the original order of every remaining channel, and
publishes a new schema containing only those channels. Indices can be used when
the source does not provide channel names, for example
`BadChannelRemovalStage((0, 7))`. A stage cannot mix names and indices or remove
every channel. To retain only channels whose measured impedance is between 5
kOhm and 100 kOhm, use
`BadChannelRemovalStage(min_impedance_ohm=5_000, max_impedance_ohm=100_000)`.
Either limit may be left as `None`; that side of the range is then unbounded.
Impedance filtering reads `SignalSchema.channel_impedances_ohm`, and prepare
fails explicitly when the source did not provide those measurements. Explicit
bad channels and impedance rules can be combined; a channel is removed when
either rule rejects it.

`LineNoiseFilterStage()` creates second-order, 2 Hz-bandwidth Butterworth
notches at 50, 100, and 150 Hz. `harmonics=3` means the first three frequency
multiples, including the 50 Hz fundamental. For a 60 Hz installation, use
`LineNoiseFilterStage(frequency_hz=60.0)`. The cutoff interval is derived from
`frequency_hz` and `bandwidth_hz`; the native stage validates the fundamental
against the actual sampling rate, automatically omits requested higher
harmonics that do not fit below Nyquist, and designs the complete SOS cascade
once during `prepare()`. Use `harmonics=1` when only the fundamental should be
removed.

## Configure feature extraction

`FeatureStage` runs an ordered, non-empty subset of PMTM bandpower, Hilbert
envelope, and LMP definitions on the same sampled input. Each feature type may
appear once. All features share the stage's window and update interval, so
every output row describes the same observation time. The declaration order is
also the column order presented to a decoder.

`FeatureStage()` alone selects a 200 ms window, a 50 ms update interval, and a
4 Hz LMP feature. Override only the scientific choices the experiment
needs:

```python
from neurale.pipeline import (
    Band,
    FeatureStage,
    HilbertEnvelopeFeature,
    LmpFeature,
    MultitaperBandpowerFeature,
)

features = FeatureStage(
    window_seconds=0.2,
    update_interval_seconds=0.05,
    features=(
        MultitaperBandpowerFeature(
            bands=(Band("alpha", 8.0, 12.0),),
        ),
        HilbertEnvelopeFeature(
            bands=(Band("beta", 13.0, 30.0),),
            filter_order=4,
        ),
        LmpFeature(
            cutoff_hz=4.0,
            filter_order=4,
        ),
    ),
)
```

Combined names are prefixed automatically, for example
`hilbert:beta:C3` and `lmp:C3`. PMTM columns use the `pmtm:` prefix. The
stack inherits channel names from the prepared input schema; sources without
names receive stable `channel_0`, `channel_1`, ... labels. Output schema, signal,
feature-set, and unit identifiers are allocated during native `prepare()`.
Hilbert and LMP preserve the input amplitude unit, while PMTM derives its power
unit; no feature takes a feature-unit or unit-ID argument. The stage emits one
regular feature signal and never starts worker threads or calls Python from the
realtime path. The durations must resolve to exact integer sample counts at the
prepared input rate. Different windows or update intervals require separate
pipelines; the stack does not resample or align asynchronous feature streams.

The feature API stores filter-design intent rather than raw coefficients. At
native `prepare()` time, PyNeurale reads the actual input sampling rate and
designs the SOS cascade once. The resulting coefficients are owned by the
native feature processor; filter design, allocation, and Python calls never
occur in the realtime `process()` path. `filter_kind` defaults to
`"butterworth"`; `"bessel"` and `"elliptic"` are also available, with
`passband_ripple_db` and `stopband_attenuation_db` required only for elliptic
designs. All feature constructors are keyword-only. PMTM defaults to
`time_bandwidth=2.5`, automatically chooses the corresponding maximum valid
taper count when `n_tapers` is omitted, and still accepts an explicit taper
count for advanced use. PMTM and Hilbert FFT lengths are deterministically
derived from the prepared window. Algorithm-version metadata is library-owned
and is not a constructor argument.

## Save reproducible configuration

The document format contains an explicit kind and version. `save_plan` writes
canonical UTF-8 JSON; `load_plan` accepts only the version-1 top-level shape,
the fixed stage whitelist, and the exact dataclass fields for each stage.
Numbers, strings, and integer indices are validated without implicit scalar
conversion.

```python
from neurale.pipeline import load_plan, save_plan

save_plan(plan, "motor-control.pipeline.json")
restored = load_plan("motor-control.pipeline.json")
assert restored.fingerprint == plan.fingerprint
```

`PipelinePlan.fingerprint` is the SHA-256 digest of the canonical document.
Fitted decoder coefficients and state are stored in the plan itself, so the
digest covers them. The fingerprint identifies processing configuration; it
does not claim that two machines, devices, runtime policies, or builds are the
same.

## Scope and experiment boundary

The portable plan deliberately describes the processing chain only. Runtime
objects such as a live device handle, actuator, safety controller, recorder,
or presentation window are selected explicitly by the caller and are not
serialized into executable JSON.

Experiment semantics, including Center-Out and SSVEP, remain in
`neurale.experiments`.
Presentation remains optional and separate; a pipeline timestamp is not a
physical display-onset measurement.

## Center-Out and recording

Use the [Center-Out guide](center_out.md) for the high-level closed-loop session,
online decoder training, and outcome checks. The
[recording and replay guide](recording_replay.md) covers NRF capture,
inspection, and replay.

## Deploy fitted decoders

`decoder_stage(fitted_decoder, feature_schema)` exports a fitted `LinearDecoder`,
`KalmanDecoder` or `LDADecoder` into a native stage, including its feature selection
and fitted scaler. Here `fitted_decoder` is a fitted model and `feature_schema`
is the schema of its input features. The export is independent of experiment
paradigms:

```python
from neurale.pipeline import PipelinePlan, compile_pipeline, decoder_stage

plan = PipelinePlan((decoder_stage(fitted_decoder, feature_schema),))
native_decoder = compile_pipeline(plan)
```

Linear and Kalman stages output continuous values. `LdaDecoderStage` outputs one
numeric class label per feature observation, using the fitted model's class order
and binary/multiclass tie rules. Native classification currently requires integer
labels in `[-(2**53 - 1), 2**53 - 1]`; strings and temporal context are rejected.
There is no Python inference fallback. Output identity overrides are optional.
Linear and LDA deployment also accept explicitly timestamped, single-observation
feature frames (`ObservationTiming.IRREGULAR`, rate and shift zero). Their outputs
are events, not fabricated regularly sampled signals. Kalman deployment continues
to require regular observations. The internal interval-mean processor supplies
this frame contract for trial-based decoding and retains the source window identity
in its algorithm metadata.
Plans retain the model and scaler parameters and can be serialized/fingerprinted
through the ordinary `PipelinePlan` API. Experimental sessions use this same export
path internally; users need not construct algorithm-specific experiment adapters.
