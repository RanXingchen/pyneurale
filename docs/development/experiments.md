# Experiments domain design

## Scope

This document defines the ownership and behavior contracts of
`neurale.experiments`.

The domain provides Center-Out 2D, WebGrid, Speech cue, and SSVEP selection
paradigms over a small shared value contract, without a generic experiment
framework. Center-Out, WebGrid and Speech have native/headless integrations,
semantic replay and provenance support. SSVEP provides its pure state machine,
native presentation and acquisition-to-LDA closed-loop integration; see
{doc}`../user_guide/ssvep` for its complete timing and selection contract.

SSVEP reuses shared time, identity, event, selection and trial values. Its
balanced-cycle target ordering shares the existing Center-Out shuffle. It does
not link acquisition, decoding or rendering; its catalog frequency is semantic,
not a verified physical stimulus frequency. The private SSVEP integration owns
the task clock, interval requests, result acceptance and recording bridge.
The pipeline owns the paradigm-independent interval mean processor. The
execution layer owns bounded frame handoff, interval scheduling, its native
timed worker and optional compiled-decoder invocation. Interval results carry
the decoder's output frame; each paradigm interprets that frame and supplies
its own training labels. No feature vector is extracted and repackaged by the
task controller. Calibration fitting runs off the native task and acquisition
threads; both Center-out and SSVEP export fitted models through `pipeline.decoder_stage()`
and consume compiled `NativeFrameProcessor` outputs. Model/scaler implementation
and classification rules belong to the pipeline, not paradigm adapters. SSVEP
accepts one numeric target ID; Center-out accepts two velocity values. The
integration target composes the internal pipeline but has no direct model-domain
dependency. Full semantic SSVEP
replay and physical display timing validation remain separate work.

Domain, session, and optional presentation bindings share `neurale._native`.
Their internal C++ targets retain separate dependency boundaries. Importing
the Python experiment namespaces creates no native session or window.

## Why no generic paradigm runtime

The three initial paradigms are deliberately dissimilar, and the differences
are the evidence base for the ownership split:

```text
Center-Out 2D
    continuous closed-loop motor task
    decoder output / cursor state / target hit / hold / timeout
    task-specific target guidance; generic velocity assistance is composed
    outside the paradigm

WebGrid
    continuous cursor control plus discrete selection benchmark
    pointer state / target cell / selection / correct-and-misclick accounting
    target-selection throughput metrics

Speech cue paradigm
    randomized-per-trial timed stimulus task with deterministic replay
    BLACK -> optional CROSS -> CONTENT
    emits stimulus/phase/trial events; does not itself decode speech
```

Their different state, geometry, and metrics justify a small shared **value**
contract, not a shared **inheritance** hierarchy. SSVEP adds another concrete
task without changing that boundary.

## Ownership rules

These rules are normative. A change that violates one is a change to this
document first.

### 1. Determinism

Experiment semantics are deterministic transitions driven only by explicit
inputs, explicit time, immutable configuration, and owned deterministic state.

For a fixed configuration, sampler version, seed, trial identity, and explicit
input sequence, experiment semantics are deterministic: the same transition
sequence, the same event sequence, and the same realized randomized timing, on
every run.

Explicit time means integer nanoseconds supplied by the caller. A paradigm has
no notion of "now" that it did not receive as an argument.

#### The determining tuple

Every randomized experiment value is a function of:

```text
(configuration, seed, trial index, sampler version)
```

Given that tuple, the value is regenerated bit-exactly, including across
platforms and toolchains. This is a promise about a specified sampler, not a
hope about a standard-library one, and it holds only because the sampler is
pinned. Implementing it therefore *requires* that the schedule contract define:

- a stable pseudorandom algorithm, specified by its transformation rather than
  named as a library facility — `std::mt19937` seeded from `std::random_device`,
  and any distribution from `<random>`, are excluded, because
  distribution results are not portable across standard-library implementations;
- a stable bounded-integer mapping from the generator's output to a specified
  duration or index range, stating exactly which endpoints are included and how
  bias is removed;
- a sampler version that changes whenever either of the above changes.

Determinism claims that are not conditioned on a sampler version are not made
anywhere in this document.

#### Replay authority

Realized randomized values are always emitted as trace records, so a session
always carries both a way to regenerate its randomized values and a record of
what they were. Which of those two is authoritative depends on one condition,
and only on that condition:

```text
recorded sampler version is supported
    -> regeneration from the determining tuple is authoritative
    -> the recorded realized values are verification provenance:
       a reader checks the regeneration against them

recorded sampler version is unavailable
    -> the persisted realized schedule is the fallback replay authority
    -> resampling under any other sampler version is forbidden
```

There is no third case, and neither branch is a matter of preference at replay
time. A session must never be silently regenerated under a sampler version
other than the one it recorded: that would produce a different experiment while
claiming to reproduce the recorded one.

### 2. Forbidden reads

A state machine does not read:

- a global clock;
- a global RNG;
- a physical device object;
- GUI, window, or input state;
- decoder private state;
- recorder internals;
- streaming queues.

Each of these would make a trial sequence impossible to reproduce: a trial
driven by a file-scope `mt19937` seeded from `random_device`, by
`timeSetEvent`/`timerfd` callbacks, or by reaching into a live display object
for cursor position and hit state has no replay authority at all.

### 3. No generic paradigm runtime

The experiments domain has no `ParadigmType`, `ParadigmCreate`/
`ParadigmDestroy`, a generic `Experiment`/`Task` abstract base class, or a
string registry or factory.

### 4. Shared value contract, not shared inheritance

The paradigms share a small value contract:
explicit experiment time, trial identity, state transitions, experiment events,
presentation and command requests, outcomes and status, bounded trace records,
deterministic seed and schedule state including realized randomized trial
timing, and recording/replay/provenance hooks.

Each paradigm keeps its own configuration, state enumeration, transition logic,
geometry and stimulus semantics, and metrics. These are not lifted into a base
class, a template method, or a common `step()` interface.

### 5. Semantics and rendering are separate

The experiments domain owns semantic target and cue identity, the semantic
content those identifiers stand for, semantic geometry, and hit and
selection rules. Experiment presentation owns pixels, fonts, colors, monitor
selection, GLFW/OpenGL/browser surfaces, window and input handling, and render
threads.

The hit test is pure task semantics, never a property of the display: the
Center-Out containment predicate lives in the experiment contract and reads
semantic geometry, never render objects. Coupling task success to a renderer
would make a session that recorded identifiers whose meaning lived only in the
presenter unreproducible.

### 6. Semantics and hardware are separate

Hardware enters only through the streaming and device contracts:
`NativeFrameSource` for acquisition, `NativeActuator` and
`NativeFrameConsumer` for output. A paradigm never opens, runs, or closes a
device.

### 7. Assistance is not owned by a paradigm

Velocity assistance and shared control are reusable across velocity-based BCI
movement paradigms. They are owned by the shared
`neurale.experiments.assistance` submodule, not by Center-Out and not by any
other paradigm.

The experiments domain does not create a top-level `neurale.control`
domain. The current shared capability is a small amount of velocity
shared-control mathematics, which is not enough to form a standalone domain.
Reconsider only after multiple
non-experiment control capabilities — for example several actuator command
spaces, trajectory constraints, or robot/FES shared-control primitives with an
independent control lifecycle — demonstrate a stable shared owner.

### 8. Guidance in, transform blind

A paradigm may provide a task-specific reference or guidance velocity. The
generic assistance transform must not know target geometry, trial state, task
phase, GUI state, or experiment identity. It receives vectors and parameters
and returns a vector.

### 9. Target-directed guidance is not assistance

`CenterOutGuidance` is Center-Out target-directed guidance. Its output may be
consumed by generic velocity assistance, but Center-Out does not own the
assistance algorithm. Guidance and assistance are kept separate: the guidance
vector and the assistance blend are computed by distinct modules, never fused
into one place on a display object.

### 10. `OrthoImpedance` is assistance

`OrthoImpedance` lives under `neurale.experiments.assistance` as a reusable
velocity shared-control method. It is not decoder logic, not paradigm
state-machine semantics, and not runtime safety.

### 11. WebGrid is a task, not a browser

WebGrid is a task-semantic cursor-target selection benchmark. Browser and
desktop rendering, and pointer input adapters, are presentation concerns.

Its coordinates are renderer-independent logical task coordinates. WebGrid v1
uses explicit discrete `SelectionEvent` input: pointer movement and cell entry
never imply a click. Dwell, drag, double-click, and browser event semantics are
not part of this version.

### 12. Speech is a timed cue paradigm

The experiments domain owns deterministic random sampling of per-trial BLACK
and CONTENT durations, and — when explicitly enabled by configuration — a
strictly-positive CROSS duration, plus the intended phase and stimulus
timeline.

Whether CROSS is enabled is configuration, not a random outcome. When it is
disabled the phase is absent from the timeline — not present with a zero
duration — and a timeline must not represent "no CROSS phase" and "a CROSS
phase that lasted no time" identically. The exact sampling ranges are in the
Speech first-stage requirements below.

The experiments domain also owns *what* is to be presented:
`neurale.experiments.speech` owns the immutable semantic stimulus catalog
mapping each `stimulus_id` to its content, optional label or class, and
semantic metadata. Experiment presentation owns *how* it is drawn. Deciding
the stimulus is a task decision and cannot be delegated to a renderer; a
session that recorded identifiers whose meaning lived only in the presenter
would not be reproducible.

Audio and microphone acquisition, if later needed, belongs to Devices and
Recording rather than to the task state machine. The experiments domain emits
stimulus, phase, and trial events; it does not decode speech.

### 13. No invented acknowledgement protocol

Current `NativeActuator` acknowledgement is the synchronous `StreamStatus`
returned by `submit`/`write`, plus runtime timing and status. The experiments
domain does not invent a universal asynchronous actuator acknowledgement
protocol.

### 14. Persistence goes through the recorder

Experiment state, targets, stimuli, commands, assistance values, selections,
task variables, and trials are persisted through `SessionRecorder` and NRF
control records. The experiments domain does not write NRF directly, and does
not write anything at all on the critical path.

## Boundary against the existing runtime

`NativeStreamRunner` already owns, and the experiments domain must not
duplicate, bypass, or reimplement:

- lifecycle `prepare` → `arm` → `start`/`run` → `stop`/`abort` → `reset`;
- startup safety inhibition at `prepare`, released only by `arm`;
- processor and output deadline checks;
- terminal frame conversion into an `ActuatorCommand`;
- `ActuatorCommand` expiry, enforced inside `NativeActuator::submit` before the
  device-specific `write` is reached;
- actuator submission;
- watchdog and fault propagation;
- critical-observer failure handling, including recorder failure;
- shutdown on stop and abort.

The relevant vocabulary is fixed and the experiments domain reuses it rather
than defining parallel terms: `StreamStatus`, `FaultCode`, `FaultStage`,
`FaultRecord`, and
`SafetyReason` — whose enumerators already cover `startup`, `explicit_stop`,
`end_of_stream`, `source_stall`, `ingress_dwell`, `processor_deadline`,
`output_stale`, `shutdown_timeout`, `actuator_failure`,
`critical_observer_failure`, and `runtime_fault`.

`NativeFrameConsumer` derives from `NativeActuator` and finalizes `write` as
`consume(command.payload)`. A terminal consumer therefore inherits expiry
enforcement for free, and also owns `handle_discontinuity`. A paradigm that
consumes decoder output must have an explicit, documented response to a
discontinuity; silently continuing the trial as if samples had arrived is not
one.

Decoder output enters as data, never as decoder internals. The fitted Kalman
path lives in the internal pipeline adapter and the public
`neurale.decoding.KalmanDecoder`; its feature contract is described by
`FeatureSetDescriptor` — feature names, unit ids, source stream, algorithm name
and version, `window_length_ns`, `shift_ns`, and a timestamp reference. A
paradigm reads decoded values and the descriptor that identifies them. It does
not read filter state, covariance, or scaling parameters.

## Time

Experiment time is an integer nanosecond count. Experiments define their own
monotonic experiment time type rather than importing `HostTimeNs`, because
`HostTimeNs` is declared in `neurale/streaming/clock.h` and the assistance
submodule must not depend on streaming. Conversion between the host clock
domain and experiment time happens only in the native experiment/streaming
integration layer.

There are no floating-point seconds anywhere in the experiment contract.
Every duration is an integer nanosecond count; conversion into bin counts
against a sampling period does not enter the task.

## Persistence and the control plane

The recording control plane already carries exactly the record kinds the
experiments domain needs. `ProducerIdentityKind` reserves `events = 3`,
`trials = 4`, `experiment_states = 5`, `commands = 6`, `targets = 7`,
`labels = 8`, `assistance = 9`, `faults = 10`, and `task_variables = 11`, and
`SessionRecorder` exposes the matching `record_event`, `record_trial`,
`record_state`, `record_command`, `record_target`, `record_label`,
`record_assistance`, and `record_task_variable` entry points, each taking an
integer `time_ns`.

The experiments domain therefore introduces **no** new recording format and
**no** new record kind. It maps its typed trace records onto these existing
streams.

Each submission returns a boolean acceptance. A refused control record is
counted as a rejection with a recoverable first-rejection position; it is not a
silent drop, and the experiments domain must not treat a `False` return as
success. The recording layer's completeness and accounting contract — offered,
accepted, spool-committed, NRF-committed, rejected, and lost counters, with a
completeness verdict and an accounting-verified flag — continues to be the
authority on whether a session is complete. The experiments domain adds no
second notion of completeness.

## The shared value contract

The contract is implemented natively under `cpp/include/neurale/experiments/`
and `cpp/src/experiments/`, built as `neurale_experiments`, and reached from
Python through pybind11 as `neurale.experiments`. There is no Python
reimplementation of any of it. The private native execution layer is built
separately from the sibling `cpp/src/execution/` directory.

`neurale_experiments` **links nothing**. Not streaming, not signal, not
recording, not devices, and not Python. That is stated as "links no internal
target" in `neurale_assert_experiments_dependencies`, rather than as a list of
forbidden targets, because a list silently admits the next target somebody
adds. The same check forbids every foreign `neurale/<domain>/` include under
the contract's own headers and sources, and forbids every lower-level target
from linking back to it.

The headers and what each owns:

| Header | Owns |
| --- | --- |
| `contract.h` | Scalar vocabulary — time, durations, ordinals, identifiers — and `ContractStatus`, the exact reason a value was rejected |
| `time.h` | `TimeInterval` with half-open `[start, end)` semantics, and `MonotonicTimeGate` |
| `identity.h` | `TrialIdentity`, `TrialOutcome`, and the deterministic `OrdinalCounter` behind `TrialCounter` and `SequenceCounter` |
| `schedule.h` | The pinned sampler, `DrawKey`, bounded-integer mappings, `FingerprintAccumulator`, `ScheduleIdentity`, and `ReplayAuthority` |
| `events.h` | `ExperimentEvent`, `StateTransition`, `SelectionEvent`, `TrialRecord` |
| `presentation.h` | `CueKind`, `PresentationRequest`, `PresentationState`, `PresentationOutcome` |
| `command.h` | `CommandSpace`, `CommandRequest`, `CommandOutcome` |
| `replay.h` | `ExperimentSnapshot` and `DecisionSnapshot` |
| `assistance.h` | `VelocityVector`, `DesiredVelocitySet`, the two assistance methods, and their trace records |
| `center_out.h` | Center-Out configuration, target layout, schedule, `CenterOutMachine`, trial outcomes, and derived statistics |
| `center_out_guidance.h` | `CenterOutGuidance` and the target-directed reference-velocity profile — nested for the same reason, and separate from both the machine and assistance |
| `webgrid.h` | WebGrid configuration, geometry, target schedule, `WebGridMachine`, selections, and metrics |
| `speech.h` | Speech cue configuration, stimulus catalog, trial schedule and timeline, `SpeechMachine`, phase markers, and completed trials |
| `ssvep.h` | SSVEP target catalog, trial schedule, `SSVEPMachine`, phases, selections, and completed trials |

### What the shape of these values is doing

Every runtime record is a fixed-size, trivially copyable aggregate holding
integers, enumerators, and — in `CommandRequest` alone — a fixed array of
doubles. No record holds a string, a pointer, a container, or a callback.
`neurale_experiments_contract_test` asserts that as `static_assert`s and then
proves the consequence at runtime: the entire contract battery runs with a
global `operator new` counter and completes with **zero** allocations.

Strings and larger payloads are reached through prepared immutable identifiers
resolved outside any hot path. A `PresentationRequest` carries a
`stimulus_id`, never text.

Validation returns a `ContractStatus` rather than throwing. A realtime caller
therefore never pays for an exception to learn that a value is wrong, and a
Python caller still learns exactly what was wrong rather than receiving a bool.

Every persisted enumeration has a `*_declared` predicate — `cue_kind_declared`,
`trial_outcome_declared`, `command_frame_declared`, and so on — and every
validator checks it before anything else, reporting
`ContractStatus::enum_undeclared`. These types are stable, append-only, and
replay-visible, so a record can arrive holding a number that names no enumerator:
corrupt, or written by a build that declared more values than this one. A
validator that only asks "is this the *unspecified* one" would wave that through
as a perfectly good value, which is how `SelectionKind(255)` would have been read
as a discrete selection and `CommandApplication(255)` as a successful one.
`enum_undeclared` is deliberately distinct from `identity_missing`: an unset
field is a validly incomplete record, an undeclared one is an unreadable record,
and reporting them alike lets the second pass as the first. It is equally
distinct from `presentation_invalid` and `outcome_invalid`, and the boundary is
worth stating once:

```text
enum_undeclared       a field holds a number this build has no name for
                      -> corruption, or a newer writer

presentation_invalid  every field is individually legal, but together they
outcome_invalid       describe a state that cannot occur
```

The first is read by whoever is chasing a bad file or a version mismatch; the
second by whoever is chasing a bug in a paradigm. Merging them would send both to
the same place.

### Immutability

The value records are immutable across the Python boundary: each is built by a
complete keyword constructor and exposes readonly fields. The C++ aggregates stay
plain, so native code still builds them by value and they remain trivially
copyable.

This is not an API-style preference. Without it the sequence

```text
construct -> validate == ok -> fingerprint -> persist -> mutate
```

is available, and every guarantee taken before the last step silently stops being
true — a `ScheduleIdentity` could be fingerprinted and then have its seed
replaced, leaving a recorded fingerprint that identifies nothing. Construction is
therefore the only point at which a value is decided, and a `validate()` result
stays true for the lifetime of the value it was taken over. `CommandSpace` in
particular has no `set_axis`: a space that a `CommandSpaceId` stands for cannot
be edited after something has referred to it.

The stateful helpers — `MonotonicTimeGate`, `TrialCounter`, `SequenceCounter`,
`FingerprintAccumulator` — remain mutable. They are not records; advancing is
what they exist for.

### Time

`ExperimentTimeNs` is `std::uint64_t`. There are no floating-point seconds in
the contract.

Every interval is half-open. `TimeInterval::contains` is
`t >= start_ns && t < end_ns`, so the exact end instant belongs to the
following phase and to nothing else, and `elapsed_at(end_ns)` is true at the
same instant `contains(end_ns)` is false. An interval with `end_ns == start_ns`
is empty and contains nothing, which is how a `TrialRecord` says its trial is
still pending.

`MonotonicTimeGate` guards the "supplied time is monotonic non-decreasing"
precondition. It never samples a clock; it remembers the last accepted instant
and refuses one that moves backwards, leaving itself unchanged when it does.
Equal instants are accepted, because two semantic decisions may share one.

The three time roles a request distinguishes are separate fields and never
collapse into one:

```text
requested_ns   when the paradigm decided             (produced by the experiments domain)
onset_ns       when the cue is intended to show      (produced by the experiments domain)
presented_ns   presenter-reported software instant   (only presentation can report this)
```

The experiments domain never produces `presented_ns`. A `PresentationOutcome`
whose status is not `presented` is required to carry `presented_ns == 0`, so a
headless session cannot accidentally claim display timing it never measured.

### The sampler

`sampler_mix64` is three multiply-xorshift rounds over 64-bit unsigned
arithmetic, with its constants written out. `sample_slot_bits` addresses it by
`(seed, stream, trial index, draw)`. The sampler is stateless: it is addressed,
not advanced, so a replay obtains a value without having produced the draws
before it.

Bounded mapping is masked rejection, capped at `kMaxRejectionDraws` (64)
attempts. There is **no fallback mapping**. Every value the sampler returns is
exactly uniform on its range; exhausting the attempts returns
`ContractStatus::sampling_exhausted` and writes nothing.

That combination — bounded execution, exact uniformity, explicit failure on the
extreme path — is the whole contract, and the third part is what buys the second.
Reducing a further draw modulo the span would be uniform only when the span
divides 2^64, so a fallback would leave the sampler biased on precisely the path
too rare for any test to catch it on. "Uniform except very rarely" is not a
property a replay contract can be written against. Each attempt is accepted with
probability above one half, so exhaustion has probability below 2^-64 for any
range; it is reported anyway, because the alternative is a sampler whose stated
distribution is not the one it implements.

Whether some `DrawKey` reaches that branch is unknown, and deliberately not
claimed either way: a per-attempt acceptance probability above one half makes
exhaustion overwhelmingly unlikely, not impossible, and the key space is far too
large to search. That is why the mapping is factored into
`detail::sample_inclusive_from` over an arbitrary bit source — the test drives it
with a source that rejects on purpose. A branch no test can enter is a branch no
reviewer can check.

`SamplerVersion 1` names exactly this algorithm and nothing else:

```text
sampler_mix64
  + a value addressed by DrawKey rather than advanced
  + bounded masked rejection onto the range
  + at most kMaxRejectionDraws (64) attempts
  + sampling_exhausted on exhaustion, no fallback mapping
```

Any change to any line of that is a different sampler and takes a different
version. No paradigm has frozen a persisted schedule yet, so version 1 is
defined by the algorithm above rather than by anything an earlier draft of this
contract computed.

Rejection attempts consume slots inside `kDrawSlotStride`, which is exactly
`kMaxRejectionDraws`: one slot per attempt the loop may make, so one draw's
attempts never collide with the next draw's and a caller's draw ordinal advances
by exactly one per logical draw whatever the loop did. That is what makes a
schedule resumable from a draw ordinal alone.

`sample_inclusive` includes both endpoints; `sample_exclusive` excludes both and
refuses a range containing no integer; `sample_index` covers `[0, count)`.

`ReplayAuthority` makes the two-case replay rule executable, and the enumeration
has exactly two enumerators because the rule has exactly two cases. An
unsupported sampler version is *not* rejected by `validate(ScheduleIdentity)` —
that session replays from its persisted realized schedule. Only a version of
zero is invalid, because it names no sampler at all.

### Commands

`kMaxCommandDimension` is 8. A `CommandSpace` states dimension, coordinate
frame, and per-axis name and unit as enumerators, and is prepared before the
session; a realtime `CommandRequest` refers to it by identifier and carries only
numbers. Nothing dispatches on a name and no record carries text.

Every value below `dimension` must be finite. Every slot at or above it must be
exactly zero, so two commands that mean the same thing also have the same bytes
and can be compared and fingerprinted without knowing the dimension first.

Because a fixed-size record cannot take a variable-length sequence, the Python
constructors pad the sequence they are given out to the capacity — but only at
or beyond `dimension`, where the contract requires the slot to be unset anyway.
A slot the caller *declared to be in use* is never invented:

```python
CommandRequest(space=1, dimension=2, values=[0.25])          # ValueError
CommandRequest(space=1, dimension=2, values=[0.25, 0.0])     # ok
```

The first call never stated a second velocity component. Zero-filling it would
produce the same bytes as the second call, `validate()` would return `ok`, and
no reader downstream — assistance transform, replay, analysis — could ever
recover the difference between "not stated" and "stated as zero". Construction is
the last point where the caller's own sequence is still visible, so it is where
the difference is caught. `CommandSpace` treats a missing axis the same way, and
an over-long sequence is refused rather than truncated for the mirror reason.

A `dimension` above `kMaxCommandDimension` is therefore refused at construction
too: no sequence could name every axis it declares. Zero remains constructible
and is rejected by `validate()` as `dimension_invalid`, because zero *is*
representable — it just means nothing.

`CommandOutcome` records the synchronous application status and the runtime's
own status enumerator as an opaque integer code — the contract cannot include
`neurale/streaming/`, so the integration layer is the only place that knows
which enumeration the code came from. It is not a hardware acknowledgement; see
rule 13.

### Joining a report to the request it is about

`PresentationOutcome` and `CommandOutcome` each carry two ordinals: `sequence`,
the emission ordinal of the report itself, and `request_sequence`, the `sequence`
of the request it reports on. They are separate because one field cannot mean
both "which record is this" and "which record is this about", and a traceability
pass that has to guess which one was intended is not traceability.

The times in a report — `requested_ns`, `generated_ns` — are corroboration, not
identity. The time contract accepts equal instants, so two requests within one
trial may legitimately share one, and no instant identifies a request.

### Ordinals

`OrdinalCounter::issue` returns a `ContractStatus` and writes through a
reference; an exhausted counter issues nothing. It neither wraps — which would
re-issue ordinals already persisted — nor saturates onto its last value, which
would hand the same identity out repeatedly. `SequenceOrdinal` is provenance
identity, so a repeated ordinal is worse than a refusal.

The maximum representable ordinal is reserved as the exhaustion boundary and is
never issued. Spending one value out of 2^64 makes `issued()` exact and
unwrappable, and makes exhaustion a state the counter can be *in* rather than one
it can only be about to enter.

### What it deliberately does not add

No `TaskConfig`, no `ExperimentConfig`, no base class, no registry, no factory.
`tests/unit/experiments/test_experiments_import.py` asserts by name that none of
them has appeared, and that `Event`, `EventSeries`, `Trial`, and `TrialTable`
are not restated here.

## Shared velocity assistance

`neurale.experiments.assistance` owns generic velocity shared control for any
paradigm whose movement command is a velocity vector. It is implemented natively
in `assistance.h` and `assistance.cpp` inside `neurale_experiments` — which still
links nothing — and reached through pybind11 as a nested namespace, so
`neurale.experiments.assistance` is one capability built on this contract's own
vocabulary rather than a second contract beside it.

### The input contract

A velocity belongs to a `CommandSpace`. That is the whole of the frozen input
contract, and it is deliberately not restated: a space already fixes the
dimension, the per-axis names *and their order*, the units, and the coordinate
frame. Every transform validates its inputs against one prepared space, so
"dimensions, units and frame match exactly" is a check against a single
authority rather than a comparison between two records that each believe
something. A velocity naming a different space is refused as
`identity_missing`, which is exactly how a unit or frame mismatch surfaces.

| Input | Type |
| --- | --- |
| external or decoded command `u` | `VelocityVector` |
| one guidance velocity `g` | `VelocityVector` |
| desired manifold `D` | `DesiredVelocitySet`, at most `kMaxDesiredVelocities` (8) rows |
| linear parameters | `LinearAssistance` |
| orthogonal-impedance parameters | `OrthoImpedanceParameters` |

`VelocityVector` is deliberately *not* a `CommandRequest`. A request carries
trial identity, an emission ordinal, and times; handing one to a transform that
is required to be blind would leave nothing but a comment stopping it from
reading them. Rule 8 is enforced by the type, not by review.

Every meaningful component must be finite, and every component at or beyond the
dimension exactly zero, for the same reason as in `CommandRequest`. The Python
constructors refuse a sequence shorter than the dimension it declares rather
than filling it in.

`OrthoImpedanceParameters` states its own dimension, because both of its fields
are per-axis: a parameter set describing fewer axes than the space it is used
with would silently leave the rest at "not in the domain, impedance zero", which
is a decision nobody made. A domain selecting no axis at all is refused as
`domain_mask_invalid` — a caller that meant "no assistance" has
`AssistanceMethod::none`.

### Linear assistance

```text
assisted = (1 - assistance) * external + assistance * guidance
```

Per component. `assistance` lies in the declared interval `[kMinAssistance,
kMaxAssistance]`, initially `[0, 1]`, and a value outside it is reported as
`assistance_out_of_range` rather than clipped: a transform that quietly moved a
gain of 1.5 back to 1.0 would return a velocity nobody asked for, and the caller
computing 1.5 would keep computing it. A NaN fails every comparison and is
therefore out of range, which is the answer that matters.

The endpoints are the identity on the stored values rather than an arithmetic
coincidence — at zero the external command comes back byte for byte, at one the
guidance does. There is no clipping, no normalization, no adaptation, and no
state, so a guidance velocity larger than any plausible bound comes through at
its own size; the transform has no idea what a plausible velocity is, and one
that rescaled it would be deciding that.

### OrthoImpedance

Validated against a reference implementation. It applies

```text
v = S u + (I - R) (I - S) u
```

over the axes in the domain, where `S` projects onto the span of the desired
directions the command actually points along and `R` is the diagonal orthogonal
impedance. Only the diagonal is stored because only the diagonal is used.

Four design points are worth stating, each a deliberate decision:

- **Coordinate order is preserved.** Each domain axis is scattered back to the
  axis it came from, and an axis outside the domain is passed through unchanged
  and *in place*, so the command is not reordered when the domain is not a
  prefix.
- **The positive-span search is iterative and bounded.** Every pass consumes one
  desired direction, so it runs at most `count` times — a bounded loop, not
  unbounded recursion.
- **The factorization is written out.** `S` needs the pseudo-inverse of a Gram
  matrix. Assistance links nothing, and a recorded session has to regenerate on
  a machine whose LAPACK is not this one's, so the rank decision is made by a
  cyclic Jacobi rotation with a stated sweep cap, exactly as the sampler is
  written out rather than delegated to `<random>`.
- **The conditioning cut is in the eigenvalue domain.** A small eigenvalue is the
  numerical risk, since its inverse blows up, so the directions dropped are the
  small ones: an eigenvalue is kept only when it clears an absolute floor
  (`rank_tolerance`) and is at least `conditioning_tolerance` of the largest,
  which bounds the condition number of the inverted block by its reciprocal.
  Dropping the small eigenvalues keeps the well-conditioned directions the
  manifold should pass through. The floor requires a strictly positive
  eigenvalue before inverting, so a zero eigenvalue is never divided by. If the
  Jacobi rotation exhausts its sweep cap without converging the transform
  reports `ContractStatus::numerical_failure` rather than hand back a velocity
  built on an unknown numerical state.

Degenerate inputs are decided rather than left to the arithmetic:

```text
no desired vector points anywhere      -> S = 0, so v = (I - R) u
the command opposes every direction    -> S = 0, so v = (I - R) u
the command is zero                    -> v = 0
the span is rank deficient             -> S projects onto the span it has
```

The first two are the same statement: with nothing to project onto, every
component is orthogonal to the manifold and impedance is all that remains.

### What assistance never does

It never arms or releases a `SafetyController`, changes a deadline policy,
retries an expired command, or reaches a device. It produces a velocity;
`neurale.streaming` remains the only place that decides whether a command is
applied. A paradigm produces task-specific guidance — `CenterOutGuidance` is
Center-Out's, see rule 9 — and orchestration composes guidance, the shared
transform, and the existing runtime command path.

No `Controller` base class, no controller registry, and no string factory.
`tests/unit/experiments/test_experiments_import.py` asserts by name that none of
them has appeared, that no top-level `neurale.control` module exists, and that no
exported name mentions safety, deadlines, expiry, retries, or actuators.

### Realtime standing

`blend_velocity` is bounded and allocates nothing.
`neurale_experiments_assistance_test` asserts the second half the same way the
contract test does: a global `operator new` counter across the whole battery,
which must finish at zero.

`apply_ortho_impedance` is also bounded — the span search by the manifold size,
the rotation by its sweep cap — and also allocates nothing on the same evidence.
That is *not* a strict-realtime claim: no latency measurement has been executed
for it, so none is made, and it is not on the realtime data plane. Any future
native adapter that puts either transform on a streaming thread belongs to the
integration layer, not to the pure assistance submodule.

### Provenance

`LinearAssistanceRecord` and `OrthoImpedanceRecord` carry the method, its
version, the parameters that were in force, the external command, and the
resulting velocity — enough to recompute the reported velocity rather than take
it on trust, with the orthogonal-impedance record reconstructing its desired
manifold from experiment provenance and checking it against the recorded
`manifold_fingerprint`. They are two records and not one because their parameter sets are
different, and a single record would have needed a field meaning two things.
Each states its own method anyway: a record read back out of a control stream
has to say what produced it. The version is validated to the one this build
replays: a record naming any other version is `ContractStatus::version_unsupported`,
since the record is structurally readable but this build cannot recompute the
velocity it reports. This differs from the schedule's sampler version, which an
unsupported value still has a replay path through the recorded schedule.

The linear record inlines its guidance vector. A manifold is up to eight vectors
wide and is identified by `manifold_fingerprint` instead, so a replay checks the
manifold it reconstructs rather than storing one per command. The digest is
defined by the contract, not by each caller, and folds negative zero onto zero
so that two manifolds nobody could tell apart do not fingerprint apart.

These records are control provenance, not task-state truth: nothing reads one to
decide what the experiment does next. They carry a `TrialIdentity` for linkage
only, and the transform that produced them never saw it. They map onto the
existing `assistance` control kind through `SessionRecorder.record_assistance`;
the private experiment integration layer emits them, and the experiments domain
introduces no new record kind for them.

## Center-Out 2D

`neurale.experiments.center_out` holds what a Center-Out session *is* — its
immutable configuration, the geometry of its targets, the containment predicate
its hit test rests on, and the rule by which each trial's outward target is
chosen — `CenterOutMachine`, the deterministic task state machine that runs one,
and `CenterOutGuidance`, the target-directed reference-velocity generator. None
of the three reads a clock, starts a timer, spawns a thread, touches a device,
writes a recording, or draws anything. `center_out.h` contains the task
configuration, pure geometry and scheduling functions, and the state machine;
`center_out_guidance.h` contains the reference-velocity generator.

It is a nested submodule for the opposite reason `assistance` is. Assistance is
nested because it is shared by every velocity paradigm but is too small to be a
domain; Center-Out is nested because it is **not** shared, and one paradigm's
configuration reachable as though it were part of the contract would invite a
second paradigm to build on it. A test asserts that `CenterOutTask`,
`CenterOut2DLayout`, `TargetSelectionPolicy`, `CenterOutMachine`,
`CenterOutState`, `CenterOutGuidance`, and `CenterOutGuidanceConfig` are not
attributes of `neurale.experiments`.

### What the configuration carries

| Field | Meaning |
| --- | --- |
| `geometry_unit` | The one unit every geometry field below is measured in |
| `layout` | The centre target and the surrounding targets, in caller order |
| `acceptance` | Half-extents of the acceptance rectangle, applied to every target |
| `cursor` | The cursor's half-extent — the only cursor geometry the hit test reads |
| `movement_timeout` | Per-phase movement timeout, one value for each leg of a trial |
| `hold_ns` | How long the cursor must stay inside the acceptance region |
| `reward_dwell` | Per-phase success dwell |
| `punish_dwell` | Per-phase failure dwell |
| `selection` | Which target-selection policy is in force |
| `seed`, `sampler_version` | The schedule the outward targets are drawn from |
| `trial_limit` | Optional session limit; zero means no limit |

This table describes the native deterministic record. The installed Python
API deliberately does not mirror that record's constructor. Its
`CenterOutTask` constructor is keyword-only, validates immediately, accepts
durations in seconds, and accepts scalar shortcuts for equal x/y acceptance
and equal per-leg durations. `sampler_version` remains a read-only provenance
field, while session orchestration supplies the native `trial_limit` from
the sum of `CenterOutProtocol.assistance_blocks[*].trials`.

It carries no device or screen type, no colour, monitor or window, no decoder
parameter, no recorder path, and no streaming configuration. Those are not
properties of the task, and a configuration that carried them could not be
compared across two sessions that ran the same task on different hardware.

`GeometryUnit` is its own enumeration rather than a reuse of `CommandUnit`,
which names velocity units too: a geometry field that could legally say "metres
per second" is not an explicit unit. It includes `MILLIMETRES` because that is
the natural unit for cursor and acceptance geometry.

`CenterOutPhase` has no unset enumerator. Every leg of a trial is either
to-centre or to-out, so a "no phase" value would name nothing. The ordinals are
stable, so a persisted phase index reads back as the same leg.
`TargetSelectionPolicy` does have one, and it is rejected: the two policies
produce different experiments, and picking one silently would decide that for
the caller.

### Target layout

The centre is a field of `CenterOut2DLayout` rather than an entry of its array.
"The centre target cannot be selected as an outward target" therefore holds by
construction, rather than resting on a `+ 1` offset in the selection expression.
`TargetId` 0 is the centre's name — the contract already spends 0 on "unset".

Positions are explicit coordinates. The Python
`RadialLayoutRequest(radius=...)` default creates the classic eight-direction
layout: centre identifier 1, outward identifiers 2 through 9, and spokes 0
through 7. Supplying both `ids` and `spokes` selects the advanced path; supplying
only one is rejected. The native request remains an explicit fixed-size value.

`build_radial_layout` is a convenience for the radial case and states its
angular rule:

```text
step  = 2π / max(count, MIN_RADIAL_SPOKES)   with MIN_RADIAL_SPOKES = 8
angle = step * spokes[i]
```

The `8` floor means a ring of fewer than eight targets does **not** spread evenly
around the circle: four targets on spokes 0..3 sit at 0°, 45°, 90° and 135°
and leave the other half of the circle empty. A helper that quietly placed them
at 90° would be a different experiment wearing the same configuration.

A spoke is an independent number, not a target identifier. The identifier is a
label and nothing derives geometry from it — a test builds a layout with
descending identifiers on ascending spokes and checks that the ring still walks
counter-clockwise.

Nothing modifies a caller's value. `build_radial_layout` takes its request by
const reference, builds into a local, and copies out only on success. Every
Python record is constructed complete and exposes read-only fields, so there is
no second way in. The native function retains its `ContractStatus` return; the
Python binding returns the completed layout directly and raises a field-specific
`ValueError` when the request is rejected.

A layout is refused when it names no centre, names a surrounding target that is
unset or that reuses another's identifier or the centre's, has no surrounding
targets, or has more than the capacity. Those are `ContractStatus::identity_missing`
and `ContractStatus::target_set_invalid` — a new status, because a duplicated
identifier is neither a wrong-width value (`dimension_invalid`) nor two fields
disagreeing (`outcome_invalid`): each entry is perfectly consistent, and what is
wrong is that the set cannot say which one a record naming that identifier
meant. Two identifiers at the same coordinates are *not* refused; nothing in the
contract says a layout may not stack them.

### Containment

`contains_cursor` is the containment predicate in semantic coordinates:

```text
cursor.x − extent >= target.x − half_extent_x
cursor.x + extent <= target.x + half_extent_x
cursor.y − extent >= target.y − half_extent_y
cursor.y + extent <= target.y + half_extent_y
```

Full containment of the cursor's axis-aligned extent, not a centre-to-centre
distance and not an overlap: a cursor whose edge merely touches the region is
outside. The comparison is closed, so a position exactly on the boundary is
inside. `contains_point` is the extent-zero case and is implemented as one.

It is written as four edge comparisons rather than two absolute values on
purpose. `|cursor − target| <= half_extent − extent` is the same inequality in
exact arithmetic and a different one in floating point, because the two group
their subtractions differently; an absolute-value form would disagree with the
edge-comparison form on positions within an ulp of the boundary. A test of 42
000 comparisons over random and deliberately boundary-adjacent positions, across
four cursor extents, pins the predicate to the edge-comparison form.

Deliberate properties of the predicate:

- **`double`, not `float`.** A containment answer that depends on float rounding
  is not an answer a trial outcome should rest on.
- **No clamping.** The hit test reads semantic positions only; a cursor driven
  beyond the workspace is not clamped to any display rectangle, so it cannot be
  reported inside a target sitting at the edge.
- **No default acceptance region.** The acceptance half-extent comes from
  configuration, never from a renderer default.
- **A region smaller than the cursor is refused at configuration.** The
  predicate stays total and answers `false` everywhere, because that is the
  honest answer to the question; but `validate(CenterOutTask)` rejects the
  configuration, since no cursor position could ever succeed.
- **No target radius.** The predicate does not read a target radius; a rendered
  sphere size is not experiment configuration.

The arithmetic is symmetric in the axes, so y-down and y-up screen conventions
read the same way and the axis names carry no orientation.

### Target schedule

Two named target-selection policies are available, because the choice between
stepping and resampling is a configuration value rather than something readable
only from the ternary that consumes it:

| Policy | Behaviour |
| --- | --- |
| `REPEAT_UNTIL_SUCCESS` | Steps through the surrounding targets in layout order, advancing exactly one step per success, so a failed trial re-presents the same target. |
| `SAMPLE_EACH_TRIAL` | Draws a fresh surrounding target for every outward leg, so a failure changes the target as readily as a success does. |

`SAMPLE_EACH_TRIAL` is what is usually meant by "change after failure", stated
as what it is: it changes after a success too.

`select_outward_target(config, trial, successes)` is a pure function and holds
no state. Under `REPEAT_UNTIL_SUCCESS` the answer depends on `successes` and not
on `trial`; under `SAMPLE_EACH_TRIAL` it depends on `trial` and not on
`successes`. The answer is an index into the surrounding array, so the centre is
never a possible answer.

The draw goes through the contract's own bounded-integer mapping on
`DrawKey{seed, TARGET_SELECTION_STREAM, trial, 0}`. The draw uses no global
`random_device`/`mt19937` and no `std::uniform_int_distribution`: a global
entropy-seeded generator is the single largest obstacle to replay, and the
distribution's results are not portable across standard-library
implementations. Resetting a schedule is therefore the caller returning its
ordinals to where they started — the sampler holds nothing to reset — after
which the same trials yield the same targets. Draw-stream tags are
paradigm-local, so a session running more than one paradigm gives each its own
seed rather than partitioning a shared stream space.

`select_outward_target` reports `ContractStatus::version_unsupported` when a
draw would be needed and this build cannot evaluate the configured sampler, but
`validate` accepts such a configuration, exactly as `validate(ScheduleIdentity)`
accepts an unsupported sampler version: that session still replays from its
recorded schedule. The cycling policy makes no draw and is unaffected.

`configuration_fingerprint` absorbs every field, so a session that changed one
is a different schedule and will not be mistaken for the one it came from; a
test asserts one changed field per row of the configuration. `layout_fingerprint`
is order-sensitive, because the selection index refers to a position, and folds
negative zero onto zero so that geometry in the same place digests the same.

### The task state machine

`CenterOutMachine` is a pure function of what it is given. Its only inputs are
an explicit `time_ns`, an observed cursor position, the configuration captured
at `start`, and its own state. It is a value: trivially copyable, with no
pointer and no owned storage, so copying a machine copies a session and a copy
stepped with the same observations stays identical to its original. That is the
whole production core, and the Python binding is a thin wrapper that
reimplements none of it.

#### States

| State | Meaning |
| --- | --- |
| `IDLE` | No session. Only `start` leaves it; `step` is refused with `NOT_RUNNING` |
| `MOVE_TO_CENTER` | Centre leg, before the cursor has been observed inside |
| `HOLD_CENTER` | Centre leg, cursor inside, hold accumulating |
| `CENTER_SUCCESS_DWELL` | After the centre was acquired |
| `CENTER_FAILURE_DWELL` | After the centre leg timed out |
| `MOVE_TO_OUT` | Outward leg, before the cursor has been observed inside |
| `HOLD_OUT` | Outward leg, cursor inside, hold accumulating |
| `OUT_SUCCESS_DWELL` | After the outward target was acquired |
| `OUT_FAILURE_DWELL` | After the outward leg timed out |
| `COMPLETE` | The configured trial limit was reached. Terminal |

The state is one enumeration rather than a pair that had to be kept consistent
by hand. `MOVE_TO_CENTER` and `HOLD_CENTER` are the same leg before and after
the cursor arrived, and how long it has been there is a time, not a state;
"inside the acceptance region" and "holding" are distinct, not conflated.

A trial is one out-and-back attempt: the centre leg, then the outward leg. Its
outcome is `SUCCESS` when the outward target was held, and `TIMEOUT` when either
leg's movement window ended first, with `decided_phase` saying which. Every
trial begins by returning to the centre; there is no random initial cursor
placement, because the first leg is always to-centre.

#### Time and precedence

Every window is a half-open `TimeInterval`, and that one frozen rule decides
every boundary case rather than each being a separate preference:

- In a move state the movement timeout is checked first. Containment observed
  **at** the window's end instant does not start a hold, because at that instant
  the window is already over.
- In a hold state a hold that completed **strictly before** the window's end
  wins outright — even if the cursor has since left, and even if the observation
  arrives long afterwards, because it happened first.
- Otherwise the movement timeout wins, **including the exact tie** where the
  hold would have completed at the very instant the window ends.
- Otherwise containment observed to be lost discards the hold and returns to the
  move state with the movement window unchanged: leaving the target does not buy
  time. A later containment starts a *new* hold from the instant it is observed.

A hold begins at the instant containment is first observed and never earlier,
so a zero hold duration acquires a leg on the first observation inside the
region. Between two observations the machine assumes containment persisted: a
cursor that left and returned unobserved is a hold it will credit. That is a
statement about what the observations determine, not a rounding rule.

Deadlines are instants, not counters. A leg times out because `time_ns` reached
the instant its window ends, never because `step` was called some number of
times — a test drives one machine with a single step at the deadline and another
with a thousand steps before it, and both decide the same trial at the same
instant.

#### One step

`step` advances repeatedly while some window has already ended at `time_ns`, and
stops when none has — or when one trial's outcome has been decided, whichever
comes first. Deciding at most one trial per call is what bounds the work: a
caller that stopped polling for several trials' worth of time would otherwise
have a single call infer an unbounded run of timeouts. Such a caller drains them
one call at a time, and `settled` says whether anything is still pending.

When `settled` is true — the case for any caller polling faster than one trial —
`step` is idempotent. Repeating it at the same `time_ns` with the same cursor
produces no transition, no event, and above all no second copy of the trial that
was just decided.

Two cadences over the same piecewise-constant trajectory therefore decide the
same trials, and a test asserts it at 10 ms and 50 ms over the same script. Two
machines given the same configuration and the same observations emit identical
transitions, events, and trials, and so does the same machine after `reset` —
which returns it to the state a fresh one has, configuration included. A reset
session is a *new* session rather than a rewound one, so its session-local
ordinals restart from their origin and must not be read as a continuation of the
previous session's ordinals.

#### What it produces

Transitions are the contract's own `StateTransition`, carrying the Center-Out
state enumerators widened into `StateId` and a `CenterOutCause`. Events are
`ExperimentEvent`s for the session and trial boundaries only — `SESSION_START`,
`TRIAL_START`, `TRIAL_STOP`, `SESSION_STOP` — because a state-transition event
beside the transitions themselves would be the same fact twice. A decided trial
is a `CenterOutTrial`: the contract's `TrialRecord`, whose persistence is owned
by the integration bridge and not by this machine, plus the outward
target, the deciding leg, and the two acquisition durations the shared record
has nowhere to put.

Which of those durations is meaningful follows from the outcome and the deciding
leg, so neither carries a validity flag: a success has both, an outward timeout
has the centre one only, and a centre timeout has neither. They must also agree
with the trial interval and with the deciding leg, and these are the strongest
such checks that need no configuration: a meaningful duration never exceeds the
trial's length; a leg that did not acquire reports zero; a success satisfies
`center_acquire_ns + outward_acquire_ns` <= the trial duration (the outward leg
cannot begin before the centre is acquired, and the sum is checked by subtraction
so it cannot overflow); and a timeout is not a zero-length trial, with an outward
timeout ending strictly after the centre was acquired — every movement window is
strictly positive, so the machine could only ever have produced a trial that
satisfies these. `validate` rejects a hand-built trial whose interval and
durations describe something that could not have happened rather than letting it
enter the statistics.

The outward target is chosen through the target schedule and nowhere else. It
is chosen when the trial begins rather than when its outward leg does, so the
whole
trial knows what it is for; the success count cannot change in between, so the
answer is the same either way. A test recomputes every trial's target with
`select_outward_target` from the trial ordinal and the running success count.

Statistics are **not** accumulated. `summarize` computes them from decided
trials, so a success rate and a mean time to target cannot disagree with the
trials that produced them — separately-updated mutable count and running-average
fields could. The mean is truncated integer nanoseconds and reproduces exactly;
a running average accumulated in `double` would drift with the trial count.
The one count the machine does keep is the success count, because
`select_outward_target` takes it — and a test asserts it equals what
`summarize` derives.

#### What it does not do

It never moves the cursor. Nothing here writes a position at all, and the next
leg begins from wherever the caller's cursor actually is — a leg that failed
does not teleport the cursor to the origin. A test drives two consecutive
centre-leg timeouts to pin it.

It reads no clock, starts no timer, spawns no thread, blocks nothing, touches
no device, writes no recording, and draws nothing. A test asserts that no
exported name here mentions a timer, a sleep, a thread, a clock, a device, a
screen, a renderer, a decoder, a recorder, or a stream.

### Target-directed guidance

`CenterOutGuidance` is the target-directed reference-velocity generator: the
acceleration, deceleration, maximum-speed and target-precision policy and its
phase decision. It carries no `std::string` phase labels, no heap allocations
per call, no `double`-seconds step, no precision derived from rendered sphere
radii, and no acceleration derived from the target-layout distance.

It is three things it might have been mistaken for and is not. It is not
assistance: there is no blend coefficient here and no orthogonal impedance, per
rules 8 and 9, and orchestration is what feeds a reference velocity and a
decoded velocity into `neurale.experiments.assistance`. It is not the state
machine: it decides no trial, holds no clock, owns no ordinal, and cannot end a
session — a test drives a whole successful trial through `CenterOutMachine`
without evaluating a single reference velocity, because Center-Out has to remain
usable without either. And it is not a simulation, a decoder, an actuator, a
device, a recorder, or a renderer.

#### The profile

The installed Python API exposes three required, keyword-only design values:
`max_speed`, `acceleration`, and `arrival_radius`. `deceleration` is optional
and defaults to `acceleration`. The public config does not repeat
`geometry_unit`; session composition resolves it from `CenterOutTask`.
Invalid and non-finite values are rejected when the Python config is built,
and the arrival radius must be strictly positive.

`CenterOutProtocol.guidance` defaults to `None`. Once the decoded
schema supplies the exact control rate, session preparation deterministically
derives a symmetric profile from the task geometry, both movement timeouts,
the hold duration, and the initial cursor position. The resolver uses half of
the usable acceptance radius as its arrival radius, limits one maximum-speed
step to that usable radius, and targets 60% of the available movement time.
It then executes the native discrete guidance policy for the initial centre
leg and every centre/outward target pair. The same reachability check applies
to an explicit profile whenever the assistance schedule contains a 100%
block. Hold completion must be strictly before the movement deadline, matching
the state machine's timeout precedence.

The resolved numeric profile is exposed by `session.resolved_guidance`. Native
control provenance records resolver version 1 for an automatically derived
profile and 0 for an explicit profile, so replay does not depend on rerunning a
possibly newer resolver.

The public `CenterOutProtocol` contains only the task, its trial-aligned
assistance schedule, optional guidance override, optional initial position, and an
optional seconds-based stale-command policy. The schedule defines the native
trial limit. The sole prepared output signal supplies the decoded identity;
Center-Out supplies its fixed paradigm and command-space identities; and the
prepared schema supplies bounded trace/drain capacities. The host-relative
experiment clock uses the host-monotonic domain. Adaptive-loop training capture
derives its capacity from the declared experiment, task duration, and actual
control period rather than accepting a queue size from the experiment author.
`CenterOutSession` drains online capture while the experiment runs instead of
retaining an unknown source duration for post-run training.

The public online-training config is keyword-only. It retains `target`, but the
formal loop currently accepts only velocity supervision. Decoder family is
inferred from the supplied decoder object. `update_interval=None` resolves to
the number of trials in the initial 100%-assisted block, while
`training_window=None` resolves to the total declared session trials. Explicit
windows count trials, not update batches. `label_lag_seconds` is converted only
when the prepared feature rate proves it is an exact whole-observation lag; a
positive lag pairs an earlier feature row with later intended velocity inside
the same trial.

`CenterOutProtocol` is the experiment protocol consumed by `CenterOutSession`.
There is one public runnable Center-Out session; omitting presentation makes the
same session headless rather than selecting another session type.
`CenterOutSession` accepts an optional keyword-only `training` override and
otherwise constructs the default online-training configuration internally. Its
control-plane polling interval, experiment-time origin, and lifecycle deadline are private implementation
state. `run()` takes no configuration: the final task dwell triggers a normal
runtime stop, while the prepared task durations, trial count, feature cadence,
and source-stall policy determine the internal lifecycle guard.

Decoder deployment identity is session-owned rather than public configuration.
At preparation, the adaptive loop deterministically chooses the smallest
positive schema, signal, and channel-set identifiers not used by the input
schema, prepared feature schema, or declared feature stages. The neutral
decoder and every later Linear or Kalman candidate reuse those resolved
identifiers, so experiment authors cannot accidentally create a stream-identity
collision or change output identity between publications.

The internal native record still carries four numbers and the geometry unit
needed by the controller and persistence. Its `precision` field is the public
`arrival_radius`; zero remains representable only for low-level native contract
and replay work, not as an ordinary experiment setting. The profile is the
`CenterOutGuidance` policy, validated against a reference implementation's recorded velocities.
Given the retained velocity, the observed position, the active target and an
explicit elapsed step:

```text
v      = previous velocity, or (0, 0) when the target identifier changed
offset = target - position
d      = sqrt(offset.x^2 + offset.y^2)

d <= precision  ->  at_target: velocity = (0, 0), and return

u    = offset / d                       (unit direction)
s    = dot(v, u)                        (speed towards the target)

s <= 0                                  -> speeding_up
otherwise  d / s <= s / deceleration    -> slowing_down
            d / s >  s / deceleration    -> speeding_up

w = max(s, 0) * u                       (the entering velocity, projected)

slowing_down: w -= deceleration * u * dt
              velocity = max(dot(w, u), 0) * u

speeding_up:  w += acceleration * u * dt
              velocity = |w| > max_speed ? max_speed * w / |w| : w
```

The braking test is not the minimum-time braking point. It compares the time
to reach the target at the current speed against the time to brake to rest, so
it begins braking at `d <= s^2 / deceleration` — twice the distance a
constant-deceleration stop actually needs — and it measures to the target
itself, not to the edge of the precision region. The comparison is closed on the
braking side, matching the closed acceptance region of `contains_cursor`: a
profile exactly at its braking point brakes. The consequence is visible in the
reference output: on the approach the profile alternates between braking and
re-accelerating on successive steps, because one braking step is enough to put
the test back on the other side. That limit cycle is the intended behaviour, and
it is why the precision region exists.

There is no separate cruising phase: the maximum-speed clamp is applied only
while speeding up, and only when the speed strictly exceeds it, so a profile
pinned at the clamp is still speeding up as far as the phase is concerned, and a
braking run above the maximum is left alone because it is already on its way
down. Braking never runs the speed below zero, but a profile standing still
outside the precision region is still slowing down rather than arrived. Inside
the precision region the velocity is exactly `(0, 0)` and the retained velocity
is dropped rather than decayed — a reference mover that has arrived has stopped,
and a cursor that later drifts out is accelerated from rest.

The projection is what handles overshoot. A retained velocity that points away
from the target has a negative component along it, which is clamped to zero, so
the approach begins again from rest rather than reversing at speed. Direction is
otherwise never retained: it is recomputed from the observed offset on every
update, so the velocity cannot point where the cursor no longer is. The
distance is `sqrt(dx*dx + dy*dy)` rather than `std::hypot`: `hypot` avoids the
intermediate overflow that this reports as `value_not_finite`, but is not
required to be correctly rounded, and two platforms disagreeing on a reference
velocity costs more than the overflow range.

The retained state is the previous reference velocity in the Center-Out
workspace — a two-dimensional vector, not a scalar speed — and the target
identifier it belongs to. The phase beside it is written and never read, and
every phase is derived afresh from the configuration, the retained velocity and
the observed geometry, so a corrupted phase cannot change a velocity. Carrying
the vector rather than its scalar projection is what lets a change of heading
matter: when the cursor's geometry shifted the previous heading influences the
next update, which a retained scalar could not express.

#### Units, the step, and cadence

Positions and the arrival radius use the task's `GeometryUnit`, the same unit
and the same two-dimensional workspace as `CenterOut2DLayout`, components in
`(x, y)` order; speed is that unit per second and acceleration that unit per
second squared. There is no second unit vocabulary and no conversion.
`validate_against(guidance, config)` resolves the internal unit from the task
and makes the check that neither record can make alone: the arrival radius must not exceed the smaller
acceptance half-extent less the cursor extent, so that arriving *implies* being
contained. Without that bound, guidance can settle at a standstill outside the
acceptance region while reporting that it arrived, and the leg times out anyway.

The step is an explicit elapsed duration in integer nanoseconds, never an
instant and never a call count, so guidance has no clock discipline of its own —
it is never told what time it is. The same sequence of `(dt_ns, position)`
observations therefore produces the same result whatever the wall-clock spacing
between the calls, and a hundred unrelated evaluations interleaved between two
updates change neither.

A *different partition* of the same elapsed time is a different run, and that is
stated rather than hidden. The profile integrates its step and remakes the
accelerate-or-brake decision from the speed each step entered with, so four
steps sitting exactly at the braking point (`d = s^2 / deceleration`) each brake
one step, and because braking reduces `s`, the time to slow down shrinks faster
than the time to reach the target, flipping the comparison back to speeding up;
the profile chatters — brake, accelerate, brake, accelerate — while one step of
four times the duration brakes the whole way, because the decision is made
once. A test pins both numbers.

#### Explicit reset

An experiment reset is `reset()`, which drops the retained velocity and keeps
the configuration. A target change is the other reset, and it is not silent: the
retained velocity is discarded before the update and
`CenterOutGuidanceSample` reports `retargeted`. Velocity belongs to an approach,
and carrying it into a new one would have the reference mover arrive at a target
it had never set off for. A caller that prefers to say so itself calls `reset()`
first and gets the same state; a first update after a reset reports
`retargeted == False`, because there was no velocity to discard. The same
identifier at a different place is not a target change — it is the same target,
moved.

### Realtime standing

Every function here is bounded and allocates nothing, and the native tests
assert the second part with a global `operator new` counter — including a full
scripted session driven through the machine and a full scripted approach driven
through the guidance generator. The native and headless integration target
exercises those values through the private `neurale_execution`
target described below. The benchmark adds measured latency distributions for
linear assistance, the Center-Out machine, guidance, and the
decoded-frame-to-headless-cursor path in `neurale_experiments_benchmark`. Those
rows report the exact build and host and do not establish a portable deadline
or a universal realtime claim.

### A configuration this does not reject

Two targets whose acceptance regions overlap are accepted. One cursor position
could then acquire either of them, so "which target did the subject acquire" has
no answer, and the state machine's answer depends on which leg happens to be
open. It is not rejected because nothing in the experiments domain needs it to
be — the step chain is bounded by the one-trial-per-call rule and not by the
geometry — and adding a
rejection would be a geometry decision made for a state-machine reason. It is
recorded here so the gap is a decision rather than an oversight.

## Paradigm requirements

### Center-Out 2D

Deterministic task semantics.

Configuration, target geometry, containment, the target schedule, and the
deterministic state machine are implemented; the "Center-Out 2D" section above
describes what they actually do. The requirements below cover the whole
paradigm, and target-directed guidance remains a constraint on code that does
not exist yet.

Owned: trial phase (to-centre and to-out), target identity and semantic
position, hold accumulation, per-phase timeout, success and failure outcomes,
reward and punishment intervals as timeline semantics, inter-trial transition,
success and failure counts, and mean time to target.

Configuration is immutable and carries no display or device fields. The
configuration carries no `device_type`; hardware enters only through the
streaming and device contracts. The centre-to-surround distance is the semantic
distance of the target layout only. Guidance parameters are stated in their
own configuration and are not derived from layout distance.

Target layout is an explicit helper producing explicit coordinates. The layout
is stated as coordinates, and any radial convenience helper is a named function
with the angular rule in its contract — the `8` floor of
`2π / max(n − 1, 8)` is visible in the helper, not hidden in a parameter
struct.

The hit test is pure geometry. It is full containment of the cursor's
axis-aligned extent within an acceptance box centred on the target — not a
centre-to-centre distance test — and the acceptance region is taken from
configuration.

Containment therefore needs exactly three things: the target position, the
acceptance half-extent, and the cursor's own extent. It does not need a target
radius; a rendered sphere size is not experiment configuration.

Guidance: `CenterOutGuidance` is Center-Out target-directed guidance, producing
a reference velocity and a movement phase. The phase is an enumeration,
storage is fixed-size, and the step is integer nanoseconds — no `std::string`
phase labels, no heap-allocated temporary arrays per call, no `double` seconds.

Guidance precision is explicit Center-Out guidance configuration. A rendering
change must not be able to change task guidance, so precision is not derived
from rendered sphere radii.

### WebGrid

`neurale.experiments.webgrid` currently implements immutable configuration,
pure grid geometry, deterministic target selection, explicit discrete
`SelectionEvent` construction, and the pure deterministic `WebGridMachine`.
It has no external runtime dependency.

The grid occupies `[min_x, max_x) × [min_y, max_y)` in logical task space.
Cells are half-open, row zero begins at `min_y`, column zero at `min_x`, and
cell IDs are one-based row-major values (`row * columns + column + 1`). The
outer maximum edges are outside. Physical hit testing and candidate membership
are separate, so a pointer can occupy a real but non-selectable cell.

The candidate pool is explicit. A schedule is either a finite explicit sequence
that never wraps or a seeded schedule derived from seed and target ordinal
through the shared sampler. Seeded configuration states whether immediate
repetition is allowed; when it is forbidden, the previous target is an explicit
input after ordinal zero and is ignored at ordinal zero. An optional initial
target occupies ordinal zero. Reset is the caller returning to ordinal zero,
which reproduces the sequence because the schedule holds no mutable state.
Unknown nonzero sampler versions remain structurally valid for recorded
realized-schedule replay; only a call that must execute an unknown seeded draw
returns `version_unsupported`. Explicit schedules and an initial target require
no sampler evaluation.

Correct selection policy v1 is `advance_target`. Incorrect policy v1 records
the misselection, creates no successful trial, and keeps the current target and
its onset active. `WebGridMachine` owns those transitions and the schedule
ordinal. Its states are `idle`, `active_target`, and `complete`. A step consumes
explicit integer-nanosecond time, a logical pointer position, and zero or one
discrete `SelectionEvent`; pointer-only steps cannot select. The event sequence
is its session-local identity and must increase strictly, so repeated or
regressed IDs are rejected without changing state. Optional duration has an
exact half-open terminal endpoint, optional target count counts correct target
completions, and explicit schedule exhaustion is terminal. Reset returns the
machine and schedule to their initial deterministic state.

Raw metric version 1 preserves correct and incorrect selection counts, elapsed
active nanoseconds, and each accepted selection's target onset and elapsed
duration. It defines:

```text
correct targets/min = correct * 60e9 / elapsed_active_ns
net correct targets/min = (correct - incorrect) * 60e9 / elapsed_active_ns
```

Both rates are marked undefined and stored as zero when elapsed time is zero.
Acquisition count, sum, minimum, maximum, and integer-truncated mean use correct
selections only. `summarize` recomputes the published metrics from the raw
selection records. It requires an explicit metric version and validates the
cross-record trial lifecycle: an incorrect selection retains the same trial and
onset, while a correct selection closes the trial and the next record must use
the next ordinal with its onset at that closing time. No BPS or
achieved-bitrate formula is defined. A nonzero unknown metric version remains
structurally readable in configuration, while either a machine asked to execute
it or `summarize` asked to recompute it returns `version_unsupported`.

Out of scope for the experiments domain: browsers, DOM, desktop windows,
pointer-device drivers, and any rendering. Those are presentation adapters
that consume the presentation contract.

### Speech cue paradigm

A new paradigm, using deterministic per-trial random duration sampling and
an explicit timeline. Implemented natively in `speech.h` / `speech.cpp` and
bound as `neurale.experiments.speech`. There is no prior Speech paradigm to
reconstruct; every decision below was made here, and each one is stated for
that reason.

The timeline is:

```text
BLACK -> CROSS (optional, configuration-controlled) -> CONTENT
```

First-stage requirements:

- per-trial durations are sampled deterministically from the schedule seed, the
  trial index, and the sampler version, into these ranges, where `t1`, `t2` and
  `t3` are the configured upper bounds and every value is an integer nanosecond
  count:

  ```text
  black_duration    ∈ (0, t1)

  cross_enabled = true:
      cross_duration ∈ (0, t2)

  cross_enabled = false:
      cross_duration = 0
      no CROSS phase in the timeline

  content_duration  ∈ (0, t3)
  ```

- both ends of each range are **exclusive**. A sampled duration is never zero
  and never equals its configured bound, so an upper bound is a value the
  timeline approaches and never realizes;
- `cross_enabled` is configuration, never a random outcome. When it is false no
  CROSS duration is sampled at all;
- the zero in the disabled case is unambiguous precisely because the enabled
  case is strictly positive: a recorded `cross_duration` of `0` means "there was
  no CROSS phase" and can mean nothing else. It is not a CROSS phase that
  happened to be short, and a timeline must not represent the two identically;
- the realized per-trial durations are emitted as trace records as provenance,
  under the replay-authority rule above.

#### The configuration

`SpeechCueConfig` is immutable, fixed-capacity, and trivially copyable, so it
can be compared and fingerprinted without owning storage.

| Field | Meaning |
| --- | --- |
| `black_bound_ns`, `cross_bound_ns`, `content_bound_ns` | The exclusive upper bounds `t1`, `t2`, `t3` |
| `cross_enabled` | Whether trials have a CROSS phase. Configuration, never a draw |
| `inter_trial_ns` | Exact gap after CONTENT, separate from BLACK. Zero means none |
| `seed`, `sampler_version` | The determining tuple the schedule is regenerated from |
| `trial_count` | Trials in the session |
| `schedule` | `SEEDED` or `EXPLICIT_SEQUENCE` |
| `stimulus_order` | How each trial's stimulus is drawn from the ordered set |
| `stimuli` | The ordered stimulus set, up to `MAX_SPEECH_STIMULI` |
| `explicit_schedule` | The frozen realized schedule, up to `MAX_SPEECH_EXPLICIT_TRIALS` |

The three bounds are validated whichever schedule kind is in force, so an
explicit schedule is checked against them too. That is what makes a supplied
schedule the same experiment as the seeded one it replaces, rather than an
unrelated set of numbers wearing its configuration.

An empty sampling domain is rejected, never widened: `(0, t1)` holds an integer
only when `t1 >= 2`, and a bound below that is `RANGE_EMPTY`. A rejected bound
is left exactly as it was supplied.

A `cross_bound_ns` that is nonzero while `cross_enabled` is false is rejected as
`OUTCOME_INVALID`. Every other field of a disabled cross says the phase does not
exist; a bound left behind says a domain is being sampled, and one of the two
would be believed.

#### Draw coordinates

The sampler is addressed rather than advanced, so a trial's values are a pure
function of `(seed, stream, trial ordinal, draw ordinal, sampler version)`.
Preparing trial 900 does not require having prepared the 899 before it: a
session of any length is scheduled with no prepared table and no per-trial
state, and a replay reconstructs any trial on its own.

| Coordinate | Value |
| --- | --- |
| `TRIAL_TIMING_STREAM` | 3 — the per-trial phase durations |
| `STIMULUS_ORDER_STREAM` | 4 — the stimulus order |
| `BLACK_DRAW`, `CROSS_DRAW`, `CONTENT_DRAW` | 0, 1, 2 within the timing stream |

The phase draw ordinals are **fixed**, not allocated in the order draws happen.
Toggling `cross_enabled` therefore changes only whether the cross value is
realized, and leaves every BLACK and CONTENT duration in the session exactly
where it was. Had the ordinals been a running counter, disabling the cross would
have silently re-rolled every content duration under the same seed — a change
nobody asked for, in a value nobody was looking at.

Stream tags are paradigm-local. Center-Out uses 1 and WebGrid uses 2 under their
own seeds; a session running more than one paradigm gives each its own
`ScheduleSeed` rather than partitioning one stream space between them.

#### Stimulus order and repetition

| `StimulusOrderPolicy` | Behaviour |
| --- | --- |
| `SEQUENTIAL` | The set in order, wrapping. Every stimulus appears equally often and the order never varies. Draws nothing, so it needs no sampler version |
| `RANDOM_WITH_REPLACEMENT` | Independently sampled per trial. A stimulus may follow itself and short sessions are not balanced |
| `SHUFFLED_BLOCKS` | A fresh permutation of the whole set every `stimulus_count` trials, so counts are balanced at every block boundary |

`SHUFFLED_BLOCKS` uses Durstenfeld's shuffle, written out rather than delegated:
`std::shuffle` takes a `UniformRandomBitGenerator` and its result is not
specified across standard libraries, so a session shuffled on one platform would
not replay on another. The permutation is addressed by the **block** index in
the draw key's trial field, so every trial of a block rebuilds the same
permutation and reads its own position out of it — which is what keeps a single
trial reconstructible without the ones before it. Two trials across a block
boundary may present the same stimulus; the permutations are independent, and
forbidding that would make them non-uniform.

#### Two authorities, and only ever one at a time

A seeded configuration regenerates its schedule; an explicit one carries it and
samples nothing. `replay_authority(config)` says which, and an explicit schedule
is always `RECORDED_SCHEDULE` — there is nothing to regenerate, so the sampler
version does not enter into it. A seeded configuration under an unsupported
sampler version is structurally valid but `prepare_trial` returns
`VERSION_UNSUPPORTED` rather than regenerating with the current sampler.

Fields that would decide a realized value must be absent on the side that does
not decide it. An explicit configuration carries no stimulus set, no ordering
policy, and no cross bound when the cross is off; a seeded one carries no
explicit entries. The seed is the one exception, and deliberately: it records
where a frozen schedule came from, and identifying provenance is not deciding a
value.

A supplied explicit schedule is validated, not trusted. Each entry's ordinal
must name its own position, its `cross_enabled` must agree with the
configuration, its sampler version must be the configuration's, and every
duration must lie strictly inside the bound it was drawn against.

`validate_against(schedule, config)` asks whether a schedule *belongs* to a
configuration, not whether it is plausible under one, and the difference is the
point. Being well-formed and inside the bounds is not evidence of provenance, so
on top of those checks:

- the sampler version must be the configuration's, because
  `ScheduleIdentity.sampler_version` is taken from the configuration and a
  session cannot name two samplers;
- under `SEEDED`, the stimulus must be one the configuration can present, so a
  schedule cannot put content into a timeline that the deterministic order would
  never have selected;
- under `EXPLICIT_SEQUENCE`, the schedule must equal
  `config.explicit_schedule[ordinal]` field for field, because that entry *is*
  the replay authority for the trial. A substitute that merely fits the bounds is
  a second answer to a question the configuration already answered.

What it deliberately does not do is re-run the sampler to confirm a seeded
schedule's realized values. Admitting a schedule and verifying that a recorded
session reproduces are different questions, and the second one belongs to replay
verification rather than to the gate every timeline passes through.

The unused capacity of `explicit_schedule` is held to one exact value, sampler
version included, and that field defaults to unset rather than to the current
sampler for this reason: `configuration_fingerprint` absorbs the whole array, so
a padding default that tracked `CURRENT_SAMPLER_VERSION` would move the
fingerprint of an unchanged configuration the first time that constant is
bumped, and a value nothing reads would be able to separate two configurations
that schedule identically.

#### The timeline

`build_timeline` produces the intended phase timeline of one realized trial —
intended, not observed. The phases are half-open and contiguous: each begins
exactly where the last ended, so no instant belongs to two of them and none
belongs to neither.

A disabled CROSS contributes no entry, and neither does a zero
`inter_trial_ns`. A timeline therefore holds two, three, or four phases, and
`count` is the answer to "which phases does this trial have". A cross entry of
zero length would be indistinguishable from one that was configured away, and
the whole enabled/disabled distinction rests on those two never looking alike.

The gap is not part of the trial: `timeline.trial` ends at the end of CONTENT,
and `next_trial_start_ns` is where the next trial's BLACK begins. The gap
carries `CueKind::NONE`, and the experiments domain makes no claim about what a
display shows during it.

`make_schedule_draws` emits the realized durations as shared `ScheduleDraw`
records — two when the cross is disabled, three when it is enabled — so a reader
that knows nothing about this paradigm can still check a session's draws against
a regeneration. The `TrialIdentity` these records are filed under has to be the
schedule's own: its ordinal and stimulus are checked rather than copied, because
attributing a value drawn for one trial to another does not produce slightly
wrong provenance, it produces provenance that is wrong in the one field it exists
to carry. The identity's key and block are free — a draw record does not speak
for them.

#### The state machine

`SpeechMachine` runs one trial of that timeline. Its states are exactly
`IDLE -> BLACK -> [FIXATION_CROSS] -> CONTENT`, ending in `COMPLETE`, and its
only inputs are an explicit `time_ns`, the configuration captured at `start()`,
and one prepared `SpeechTrialSchedule` per trial.

**It does not sample.** Not "it samples reproducibly" — it does not sample at
all. Every realized duration and the stimulus reach the machine inside a
schedule that was decided before the trial began, which is the rule the previous
section froze. That is why `start()` and `step()` take schedules rather than
ordinals: a machine holding a seed could regenerate one, and then the question of
which artefact decided a session would have two answers.

**It runs the timeline it was given.** Every boundary is read out of the
`SpeechTimeline` that `build_timeline` produced for the trial, computed once when
the trial begins. The machine does not re-derive a boundary from a duration, so
the timeline a caller can inspect and the boundaries the machine takes cannot
drift apart — they are the same values. `build_timeline` runs
`validate_against`, so the ownership checks above gate every trial: a schedule
that does not belong to the configuration never reaches a single transition.

**A phase advances because time reached the instant its window ends**, never
because `step()` was called some number of times. A caller polling at 10 Hz and
one polling at 1 kHz cross the same boundaries at the same instants; a caller
that stopped polling for a whole trial crosses all of its boundaries in one step,
each exactly once, each stamped with the instant it actually occurred rather than
the instant the caller asked; and repeating a timestamp emits nothing at all.

**A disabled cross is absent, not empty.** There is no `FIXATION_CROSS` state, no
cross transition, no cross onset or offset marker, no cross `PresentationRequest`
and no cross entry in the recorded timeline. CONTENT begins exactly at BLACK's
end.

**There is no inter-trial state.** A step that ends a trial also begins the next
one at the same instant — CONTENT's end *is* the next BLACK's start, and
half-open intervals leave no gap between them for the machine to be in — so the
next trial's BLACK *is* the blank period between the two. That is also why the
next trial's schedule is an argument to the step that crosses the boundary:
there is no position between two trials, only a missing input, and a step that
would cross without one reports `IDENTITY_MISSING` and leaves the machine exactly
as it was.

For the same reason a nonzero `inter_trial_ns` is **refused** at `start()` with
`OUTCOME_INVALID` rather than ignored. This machine has no state to spend a gap
in; accepting the field and dropping it would run a session that
`build_timeline` describes differently — two answers to when the next trial
begins, one of which nobody would notice. Honouring a gap means adding the
state, not adding a silent behaviour.

At most one trial is decided per step. A caller that stopped polling for several
trials drains them one call at a time, supplying one schedule each, rather than
having a single call infer an unbounded run of trials.

What a step produces: `StateTransition`s; `ExperimentEvent`s —
`session_start`/`session_stop`, `trial_start`/`trial_stop`, and a
`paradigm_marker` at every phase onset and offset; a `PresentationRequest` per
phase onset; a believed `PresentationState` in the snapshot; and a completed
`SpeechTrial` at CONTENT end carrying the record, the schedule, and the timeline
as run. The markers exist because `ExperimentEvent` is the stream a recording
stores and `PresentationRequest` is not one of them: a session whose event series
held only session and trial events would have no record of when its phases began
and ended. Every record of one step draws from the same session-local sequence
counter, so the arrays interleave by `sequence` into one ordered stream.

The identity the machine stamps — the trial ordinal and the schedule's stimulus —
is exactly the identity `make_schedule_draws` requires, so a caller can file a
trial's realized durations as provenance without rebuilding one.

**Semantic time is not presentation time.** Every `PresentationRequest` carries
the instant the paradigm *decided* to present, and never the instant anything
appeared; `valid_until_ns` is the phase's own end, because presenting a phase
after it is over is pointless. The machine produces no `PresentationOutcome` and
cannot: only a presenter can report its documented software presentation event.
A headless run is evidence about semantic timing and carries none about a
monitor, a vsync, or a photodiode. Software evidence is not physical onset.

There is one `TrialOutcome`, and deliberately so: the Speech paradigm observes
nothing — no microphone, no decoder, no response — so a trial that was
presented in full is the only thing that can have happened, and inventing a
failure code for a failure this paradigm cannot detect would be a vocabulary
for something that never occurs. No Speech metric is defined and none is
accumulated, for the same reason.

#### The stimulus catalog

`neurale.experiments.speech` owns an immutable semantic stimulus catalog. This
is a single named owner, and it is stated here because "owned outside the state
machine" is not an owner.

The catalog maps each `stimulus_id` to its semantic content: the text or
content to be presented, an optional label or class, and semantic metadata. It
is prepared before a session starts and does not change during one.

`SpeechStimulus` is deliberately narrow. `text` is canonical UTF-8 bytes,
validated as one complete well-formed UTF-8 sequence, up to
`MAX_SPEECH_TEXT_BYTES`; `label` is an optional
class, with zero meaning unset so that "no class" and "class zero" stay
distinguishable; `metadata` is one opaque integer the contract stores,
fingerprints, and never interprets. That it is a fixed-width integer is the
point — it cannot grow into an arbitrary object payload. Images, audio, and
phoneme-specific presentation need a new `SpeechContentKind` enumerator and its
own fields, which is a change to this contract rather than something a caller
smuggles through.

The catalog fingerprint covers content, not just identifiers: a catalog that
reassigned an identifier to different text would otherwise be mistaken for the
one a session actually presented. Bytes beyond `text_length` are required to be
zero, so two entries with the same prompt have the same bytes and therefore the
same fingerprint.

The split is:

```text
the experiments domain decides what is to be presented   -> stimulus_id and its catalog entry
presentation decides how it is drawn                     -> fonts, layout, colors, timing of frames
```

The state machine and any native hot path carry only the numeric `stimulus_id`,
so they perform no allocation and no text handling. A `PresentationRequest`
likewise carries `stimulus_id`, and the presenter resolves it against the
prepared catalog at render time.

The catalog is not a presentation concern and must not be reachable only from
the presenter. A recorded session that carries stimulus identifiers but no way
to recover what those identifiers meant is not a reproducible session. The
catalog, or a fingerprint identifying it exactly, therefore enters recording
and session provenance alongside the schedule seed and sampler version.

## Native experiment execution

The execution layer uses the internal C++ namespace `neurale::execution` and
the internal-only CMake target `neurale_execution`. It links `neurale::experiments`,
`neurale::streaming`, `neurale_recording`, and `neurale_pipeline`. It has no
public include directory, install, or export surface. It connects four concrete
paradigms to the runtime; it is not a generic experiment runtime.

Its sources live in `cpp/src/execution/`, beside the pure experiments domain.
`neurale_assert_experiments_dependencies` checks the entire
`cpp/src/experiments/` tree without an execution carve-out. The separate
`neurale_assert_execution_dependencies` rule permits execution to
include `neurale/experiments/` and `neurale/streaming/`, but no other domain;
it also forbids Python, device and ambient-randomness dependencies.

`CenterOutController` is a `NativeFrameConsumer`. `prepare()` fixes one sampled,
float64, sample-major, two-channel decoded stream with an exact integral-
nanosecond observation period, constructs its frame validator, and allocates a
bounded SPSC trace handoff. `start()` fixes the host-to-experiment time origin.
Each block's `observation_time_start_ns` and prepared period determine the
observation times; rows are processed in order. The current decoded velocity is
optionally linearly blended with `CenterOutGuidance`, then the headless cursor is
advanced by exactly `velocity * dt` before the Center-Out machine observes the
new position. No position clipping or hidden unit conversion is performed. A
discontinuity resets the task and guidance state, preserves only the physical
cursor position, and starts a new task segment at the next observation with
`dt == 0`; the trace carries both the restart result and the following step.
This prevents a hold, reach, or guidance velocity from crossing the segment.

The velocity space must be the two-dimensional workspace `(x, y)`. This first
integration accepts dimensionless or normalized geometry and requires the
decoded `SignalSchema` to state `PhysicalUnit::dimensionless`, with the more
specific semantic choice carried by `CommandUnit::dimensionless` or
`CommandUnit::normalized`. Metre and millimetre geometry are rejected because
the current streaming physical-unit vocabulary cannot state either velocity
unit; neither is silently interpreted or converted.

`WebGridHeadlessController` accepts an explicit time, pointer, and zero or one
`SelectionEvent`. In the combined operation the pointer value is the observation
used first and selection is evaluated at that position at the same timestamp.
It delegates all target, correctness, misselection, and termination decisions
to `WebGridMachine`. It contains no click inference, assistance, browser, or
window code.

`SpeechHeadlessScheduler` is driven only by explicit `advance(time_ns)` calls.
`prepare()` copies and validates the complete finite trial schedule and resolves
each trial's content payload against the immutable catalog. The running path
does not sample, search the catalog, sleep, or read a wall clock. Each bounded
trace contains the `SpeechStepResult`, its `PresentationState`, every emitted
`PresentationRequest`, and the already-resolved payload when the request names
a stimulus. Explicit time is also the deterministic manual-clock interface used
by tests; paced presentation remains outside the pure scheduler.

All three controllers use a fixed-capacity trace queue allocated during
`prepare()`. Full queues return `StreamStatus::queue_overflow` before a
WebGrid/Speech state transition and before a Center-Out frame begins. Cancel is
reported as `StreamStatus::stopped`; reset restores the post-prepare state; close
is idempotent and terminal. The native test checks the global `operator new`
count across each controller's start and running operations after prepare, and
the Center-Out path has an end-to-end `NativeStreamRunner` test with an explicit
discontinuity. These tests establish bounded native execution for the tested
path, not a latency or universal realtime claim.

## Session orchestration and the recording bridge

The session layer is **native**. It lives in
`cpp/src/execution/` beside the experiments domain, in the
same private `neurale_execution` target, and that target now
links `neurale_recording` as well as `neurale::experiments` and
`neurale::streaming`. It has no public header: the target is internal-only
and the dependency guard forbids it a public surface.

It has to be native because of where the records come from. The integration
controllers publish bounded, trivially-copyable trace structs into a bounded
queue from whichever thread is driving the paradigm -- for Center-Out that is
the runtime's realtime path -- and something has to be on the other end of that
queue. Putting that something in Python would mean either formatting on the
producer's thread or carrying every trace record across a binding that this
target deliberately does not have.

It builds no acquisition source, processor, decoder, actuator, recorder, spool,
or clock. The caller constructs each of those and hands over a reference. There
is no experiment base class, registry, factory, graph, or scheduler, and the
three trace writers share no parent beyond the `ExperimentTraceSource` interface
that says how a paradigm is drained -- a shared parent with behaviour in it is
how one paradigm starts deciding another's records.

### The order, and why it is the order

`ExperimentSession::start()`:

0. the **session's own configuration** is checked, before anything is touched. A
   `drain_budget` of zero is refused: every drain loop is bounded by it, so a
   zero budget is a session that never moves a record while reporting that it
   drained -- including the synchronous drain in step 7. Refusing it here is
   refusing it before a recorder is prepared, a runtime is armed, or a paradigm
   is started;
1. the paradigm's configuration and its prepared deterministic schedule are
   validated, through the writer's `validate()`. This runs before anything is
   allocated, attached, or armed, so a bad configuration never reaches a spool;
2. the recorder is prepared. `prepare()` writes no durable byte, so a failure
   here leaves nothing behind to clean up;
3. the runtime is prepared, allocating the data plane while safety is still
   inhibited. The runtime is what guarantees that, and its `arm()` refuses if it
   is not;
4. the recorder's readiness gate is passed;
5. `NativeStreamRunner::arm()` runs, which checks every critical edge's
   readiness and only then releases the inhibition through the runtime's own
   `SafetyController`. This layer never touches a controller and offers no
   second way to reach one;
6. the **paradigm's own execution** starts, through the writer's
   `start_execution()`. A record stamped before the run started would describe a
   run that had not;
7. the recorder starts accepting, the opening records are written, and whatever
   the paradigm's own start already produced is drained **synchronously, on the
   caller's thread**;
8. the bridge starts, and only then does the runtime start.

Steps 6 to 8 are in that order because the runtime is what delivers frames and
the paradigm is what consumes them. Starting the runtime first would let the
first frame reach a controller that has not started; starting the paradigm
before step 5 would put execution in front of the safety release. A `start()`
that returns `ok` therefore means the experiment *is running* -- the caller has
nothing left to start, and cannot get the order wrong by starting it itself.

The runtime is last, and the synchronous drain in step 7 is why. A paradigm that
has started has already produced its opening trace, and the smallest capacity
the integration allows for a trace queue is one slot. A runtime started while
that slot is
still occupied can overflow the queue on its very first frame -- and a larger
capacity does not remove that race, it only lowers the odds of losing it.
Emptying the queue on the starting thread makes the opening trace's departure a
fact; starting the bridge before the runtime is the same argument for every
frame after the first.

For the same reason `attach_runtime()` refuses a session configured with
`bridge_poll_nanos == 0`. Without a bridge the only drain is a `pump()` the
caller makes, and a runtime produces on threads no caller's pump is ordered
against; a session that accepted both would be one whose losses depend on how
often the caller happens to call. A caller-stepped paradigm has no such
producer, which is why the same configuration is the right one there.

A failure at any step unwinds what came up and leaves the session `failed`. That
includes a paradigm this session had already started, a bridge it had already
launched, and -- the part that is easy to miss -- a runtime it had **armed**.
`arm()` releases safety without moving the runtime out of `prepared`, so an
armed-but-never-started runtime is indistinguishable, by its own state, from one
that was never armed; the unwind therefore aborts a `prepared` runtime as well
as a running one, which is what re-inhibits the actuator path.

A recorder the unwind finds already accepting is *aborted*, not closed. The
recording layer's own `close()` ends a recording session as a normal stop, and
a recording that reads as a run which finished is the wrong artefact to leave
behind a start the caller
was told had failed -- the same reason the terminal intent at shutdown is
settled after the final drain rather than before it.

The invariant covers a failure that arrives as an **exception**, too. Bringing a
session up allocates and creates a thread, and both fail that way rather than by
returning a status; without the unwind, the one bring-up failure nobody wrote a
status for would be the one that leaves safety released, a paradigm running, and
a recorder open. `start()` therefore unwinds and rethrows: the caller gets the
exception that says what actually failed rather than a status invented for it,
and the session is left `failed` with nothing running. This is the one place
`start()` is not a status-only surface, and the header says so.

### Destroying a session ends it

A session that owns a bring-up order owns the unwinding of it, and that includes
its destructor. Destroying a *running* session runs `close()`: an abort with no
fabricated end time, because there is no real one to use and a destructor cannot
ask the caller for one. Ending only the bridge -- which is all the destructor
used to do -- left the paradigm running, the recorder open, and an attached
runtime producing into a queue whose only consumer had just been joined.

A session that was never started, or that has already been stopped or closed, is
left alone: its recorder is the caller's, and closing one that this session
never prepared would be this layer reaching past its own lifecycle.

`stop()` and `abort()`:

1. the intent is latched first, before anything that can fail, so a failure part
   way down cannot be mistaken for a session still running and a second call
   cannot re-enter;
2. and 3. the runtime's `stop()`/`abort()` performs the safety inhibition on the
   actuator path, and `join()` waits for the workers. Neither is reimplemented
   here. Then the paradigm's execution is stopped through `stop_execution()`,
   which is what makes a caller-stepped paradigm quiet at all. Together they are
   the precondition step 4 depends on;
4. every accepted trace record is drained **while the recorder is still
   accepting**, and the summary is written into that same still-open window;
5. the recorder's terminal intent is settled **after** that final drain, the
   recorder stops or aborts accordingly, and `close()` reports what the
   recording layer says rather than a verdict of its own;
6. repeating any of `stop`, `abort`, and `close` changes nothing and reports
   what the first call did.

Step 5 comes after step 4 rather than before it because the final drain is where
a loss is most likely to be found: a queue that had overflowed, a record the
recorder refuses, or a summary that does not fit. An intent frozen before the
drain would close such a recording as a clean stop while the session's own
outcome said the trace had a hole in it, and the recording would then be the
more optimistic of the two artefacts. The intent the recorder was actually ended
under is reported as `SessionOutcome::terminal_abort`.

### The recorder is not attached to the runtime, and that is the point

Step 4 is only orderable because the recorder this session drives is **not**
added to the runtime as a critical observer. A critical observer's terminal path
runs on the runtime's observer-dispatch thread: `finish_critical_observers()`
calls `NativeRecorderCore::drain()`, which shuts the recorder down. A recorder
attached there therefore stops accepting at a moment this session cannot order
itself against, and every trace record still queued at that moment could only be
reported as loss -- never delivered. Draining before `join()` instead does not
fix it; it only moves the race.

A caller that also wants the data plane recorded gives the runtime a recorder of
its own. The two are separate recording sessions, and neither has to guess
about the other's shutdown.

**A recorded experiment is therefore not necessarily one self-contained NRF
artefact.** A run that records both produces two: an experiment-control
recording driven by this session, and a data-plane recording driven by the
runtime. Nothing here claims otherwise, and nothing above should be read as
saying the experiment trace and the neural streams land in one file.

**Deterministic semantic replay answered the narrow version of that question
and left the wide one open.** Deterministic *semantic* replay needs neither the
neural streams nor a link to them: it re-executes the paradigm's decisions from
the experiment control records alone, and the provenance it checks is the
experiment's. So two artefacts are enough for replay, which did not have to
require one.

What is still unanswered is the question a reader joining the two asks: nothing
recorded links the data-plane session to the experiment-control session, so a
caller that wants them associated still has to carry the association itself.
That is a recording-identity decision, not a replay one, and it is still open.

### Provenance has exactly one owner

A trace writer is constructed from the controller or scheduler that will run,
and nothing else -- `WebGridTraceWriter(controller, paradigm)`,
`SpeechTraceWriter(scheduler, paradigm)`,
`CenterOutTraceWriter(controller, host_epoch_ns)`. It reads the configuration,
the catalog, and the prepared schedule out of that object.

The configuration is deliberately *not* a constructor parameter. A writer given
its own copy can hold a perfectly valid configuration that the controller never
saw: the controller runs configuration A, the recording describes configuration
B, both validate, and nothing in either object is in a position to notice. For a
recording bridge that is not a small inconsistency -- it is a session whose
recorded provenance describes a run that did not happen.

`validate()` therefore also fails when the execution owner is not prepared at
all. There is no frozen configuration to validate in that state, and validating
the default-constructed one would pass or fail on a configuration nothing is
going to execute.

The one thing a writer is still told is what only the caller knows: the paradigm
identity the run is recorded under, and -- for Center-Out -- the host instant
experiment time zero is anchored at.

### Where the formatting happens

`ExperimentSession::pump()` drains a bounded budget of trace records, encodes
each one, and offers it to the recorder. With `bridge_poll_nanos` set, the
session owns a bridge thread that calls it; with it zero there is no thread and
the caller pumps from whichever control thread already steps the paradigm.
Either way the encoding happens on a thread that is allowed to be slow, never on
the one that produced the trace.

The encoder allocates nothing after construction. One `ControlRecordWriter`
serves a whole session, building into fixed storage; an overflow is latched and
turned into a counted refusal, because a truncated JSON document is not a
shorter record but an unreadable one.

### What the records look like

No new record kind and no new format. The nine control kinds are the recording
layer's producer-identity registry numbers, and the three body shapes are
exactly the three the recording layer's finalizer accepts -- a body carrying a
field those frozen NRF
schemas have no column for is rejected there, so inventing one here would
produce a session that records fine and finalizes never.

**Numbers, not names.** The experiment contract stores a paradigm's own state,
phase, cause, and outcome enumerators as opaque integers beside the
`ParadigmId` they belong to, and declares that it never interprets them. The
contract exposes no `to_string` for any of them, on purpose. Spelling them in
the bridge would create a second vocabulary to keep in step with each
paradigm's own, so the records carry the contract's numbers and the record
*name* -- which this layer does own -- says which field is which.

Two rules decide where a number goes. A record's `value` is a float64, so it
carries quantities and small identifiers, and an integer at or above 2^53 is
refused rather than rounded. A record's `text` carries structure as a compact
JSON document, which is where every 64-bit identity goes: JSON integers are
exact at any width, and a fingerprint rounded into a float64 reads as an
identity and is not one.

| Paradigm | What it records |
| --- | --- |
| all | `task_variables` `experiment.metadata`: type, version, trace version, configuration fingerprint, schedule *identity* fingerprint, **realized** schedule fingerprint, seed, sampler version, input and output schema fingerprints, metric and policy versions. Written first, so the trace is self-describing when it is read on its own. |
| Center-Out | the machine's start result for **every segment** -- the session's own, and each restart after a discontinuity -- so a replay never sees a segment that has no beginning; `task_variables` `center_out.configuration`; `assistance` `center_out.assistance_method` and `center_out.guidance_config`; `targets` `center_out.layout` and one `center_out.target` per placement, plus `center_out.active_target` when the leg's target changes; `experiment_states` per transition and `events` per event, both with the contract's own numbers; `assistance` `center_out.assisted_velocity` carrying method, version, parameter, decoded, guidance and assisted velocities, the displacement they produced and whether they were applied; `assistance` `center_out.guidance_sample` for the reference the assistance record cannot explain; `commands` `center_out.command` plus `center_out.command_outcome` using the shared request/outcome sequence relation; `labels` `center_out.cursor` for the player state a replay needs; `labels` `center_out.observation_provenance` joining the experiment-local observation/decoder aliases to the decoder frame sequence and absolute sample index, and naming the **pre-step** state/guidance/assistance identities; `trials` per decided trial; `task_variables` `center_out.summary`. Cursor, guidance, assistance, and command rows repeat the same stable observation ordinal. |
| WebGrid | `task_variables` `webgrid.configuration` with the grid, its schedule, its policies and its fingerprint; `labels` `webgrid.pointer_source` stating whether the pointer travels here or in a named stream; one `webgrid.pointer` or `webgrid.pointer_reference` row per accepted update with an experiment-local pointer-update ordinal; `targets` `webgrid.target_onset` with a stable onset ordinal when the target changes; `events` `webgrid.selection_correct`/`webgrid.selection_incorrect` with explicit pointer-update and target-onset parents, the selected and intended cell, correctness and the elapsed time the score is derived from; `trials` per completed target with its selection parent; `task_variables` `webgrid.metrics` with the metric inputs and the summary, under the metric version that produced it. An external pointer reference is complete only when its claimed frame/sample identity resolves to separately recovered source evidence. |
| Speech | `task_variables` `speech.configuration` with the configuration, catalog and schedule fingerprints, the seed and the sampler version; per trial `labels` `speech.schedule` with the realized BLACK/CROSS/CONTENT durations and the intended trial interval, one `experiment_states` `speech.intended_phase` per phase interval, and `targets` `speech.stimulus`; `experiment_states` per transition and `events` per event; `commands` `speech.presentation_request` per request, with the resolved payload when the catalog resolved one; `labels` `speech.presentation_state` for the believed state; `trials` per completed trial; `task_variables` `speech.summary`. |

A disabled cross is absent from the phase records rather than present with a
zero duration, exactly as it is absent from the timeline itself.

**Two schedule fingerprints, because one is not enough.**
`schedule_fingerprint` digests the schedule's *identity* -- seed, sampler
version, configuration and catalog fingerprints. `realized_schedule_fingerprint`
digests the prepared entries themselves. They are separate because
`validate_against(schedule, config)` admits a seeded schedule on membership,
bounds, and sampler version alone and deliberately does **not** re-run the
sampler: one configuration therefore admits many legal realizations, and all of
them share an identity. Recording only the identity would leave every one of
them indistinguishable in the record. Speech, which freezes its schedule before
the run, digests it field by field in trial order; Center-Out and WebGrid draw
as the run goes, have no prepared schedule to digest, and record zero -- which
says "there was none" in the same field every paradigm uses.

**Center-Out records the configuration's sampler version, not the build's.** A
configuration may legally name a sampler this build cannot regenerate, and
Center-Out can still run under one because nothing in its path re-executes a
sampler. Stamping the build's constant instead would produce metadata whose
sampler contradicts the configuration fingerprint printed beside it.

WebGrid's pointer samples are recorded here by default, because a headless
WebGrid has no cursor stream to point at and a replay that cannot see the
pointer cannot reproduce a selection. A caller whose pointer really does come
from a recorded signal names that stream through `reference_cursor_stream()`,
and the samples then travel once as a stream-level reference instead of twice
as values. That API does not receive an external frame/sample identity, so the
offline provenance verdict remains `PARTIAL` (`LINK_OMITTED`) even when the
named stream was recorded, and reports `RAW_STREAM_NOT_RECORDED` when it was
not. A future adapter may persist an explicit external frame sequence and
sample index without changing WebGrid task semantics. Even then, the builder
requires a matching one-sample `SourceRangeEvidence` resolved from that stream;
the control row cannot prove its own parent merely by naming one. Timestamps are
never a join key.

### Presenter-reported software presentation time is a separate record

`SpeechTraceWriter::report_presentation()` takes a `PresentationOutcome` on a
bounded lock-free queue and writes it as `labels` `speech.presentation_outcome`,
joined to the request it answers by `request_sequence` and stamped with the
reported instant. It never edits the request. The request's `onset_ns` is the
semantic time the paradigm decided; `presented_ns` is the presenter's documented
software event. Overwriting the first with the second would destroy the only
evidence of their difference. That difference supports software timing analysis;
physical-onset analysis requires separate acquired evidence. Presentation owns
producing the report; this owns where it goes.

### Trace loss

A control record the recorder refuses never reached the recording. A record a
bounded producer dropped never reached the recorder. The session counts both.
The first is automatic -- `commit()` treats a refusal, and a body the fixed
storage could not hold, as loss.

The second is read from the producer. Every integration controller counts the
trace records its bounded queue could not keep, monotonically and for its whole
lifetime, and exposes the count through `ExperimentTraceSource::dropped_trace_count()`;
the session samples it on every drain and on shutdown, and turns each increase
into a loss. It has to work that way because a controller running inside a
`NativeStreamRunner` returns its `queue_overflow` to the *runtime*, not to
anyone who could report it -- so a session that learned about drops only from
`note_step()` would report a complete trace for a run whose observations were
never produced. `note_step(status, time_ns)` still exists for a caller-stepped
paradigm, but it reports nothing of its own: it samples the same counter, which
is what keeps one drop from being counted twice by two observers of it.
Inferring loss from a runtime fault instead would be wrong in both directions --
the runtime faults for many reasons, and a full trace queue does not always
fault it.

That counter is a *lifetime* counter, so the session cannot read it as a count.
`start()` samples it once as a baseline and reports only what accrued after
that, because a controller that is reset and started again still carries every
drop of every run it has had; a session that measured from zero would open by
charging a clean run for the previous run's losses.
`SessionOutcome::producer_trace_drops` is therefore this session's drops, not
the producer's total.

Either one writes an
`experiment-trace-loss` fault row and permanently clears
`SessionOutcome::experiment_trace_complete`. The row is stamped with a real
experiment instant -- the step's, or the refused record's -- because a fault
row stamped at zero in a session that started later reads as one that happened
before the session began.

A loss report that is itself refused is counted once and absorbed, so a
permanently refusing recorder cannot turn one loss into an unbounded run of
reports about reports.

`TraceLossPolicy` decides what else happens. `fault`, the default, latches the
loss and aborts the runtime immediately -- through the runtime, not through the
session's own shutdown, because the loss can be observed on the bridge thread
while a caller holds the lifecycle lock, and a bridge that waited for that lock
would stop draining exactly when the shutdown needs it most. A later `stop()`
after a latched loss is an abort whatever the caller asked for. It does not end
the session on the caller's behalf, and a caller-stepped paradigm has no runtime
to abort at all -- what `fault` guarantees is the latch, not a shutdown.
`mark_incomplete` keeps running, and the trace can never be reported complete
afterwards.

Because the intent is settled after the final drain, a loss first discovered
*during* shutdown -- including a summary record the recorder refuses -- still
ends the recording as an abort. A session cannot report a graceful stop for a
trace it knows has a hole in it.

This is **not** a second notion of session completeness. The recording layer's
verdict is carried through `SessionOutcome::recorder` unchanged and remains the
authority on the session; what this layer adds is the one fact the recording
layer cannot supply, because a record dropped before it was offered is
invisible to the recorder that never saw it.

### Finalization, stated rather than claimed

The native recorder core does not finalize: `finalize()` lives in the offline
Python finalizer. `ExperimentSession::close()` therefore reports the recording
layer's status as it stands -- `finalization_required` true, `nrf_committed`
zero -- rather than claiming a sealed NRF session that nothing wrote. An
interrupted session leaves a spool whose committed prefix is promotable exactly
as it stands, which is what the recording layer's recovery mechanisms expect
to find, and the test suite asserts that against a scan rather than against a
counter.

### Known gaps

`close()` from a running session is an abort and writes no summary. A summary
record needs the experiment time the run actually ended at, that time belongs to
the caller, and a layer that invented one would be recording a fact nobody
supplied.

No latency or throughput measurement has been executed on the bridge, so
nothing here is a realtime or performance claim. What is asserted is that the
encoder allocates nothing after construction and that no formatting happens on
the thread that produced a trace record. The bridge's steady-state allocation is
not under an automated gate either: the integration allocation assertions
surround the controllers, not the drain.

The data-plane and experiment-control recordings are not linked by any recorded
identity. See the note above: deterministic semantic replay established that
semantic replay does not need such a link, which settles whether replay may
proceed and not whether a
reader joining the two should have one. That is still open.

The bridge is the only drain a session with an attached runtime has, and its
pace is a single sleep interval. Nothing here measures whether a given
`bridge_poll_nanos` and `drain_budget` keep up with a given frame rate and trace
capacity; the ordering above removes the start-up race, not the steady-state
one. A caller that undersizes a trace queue for its frame rate will still lose
records -- and will be told so, which is the guarantee that is actually made.

## Safety, faults, and abnormal experiment semantics

A run meets conditions its task rules say nothing about: a decoded value that is
not a number, a break in the data it is measuring against, a presenter that
could not present, a recorder that stopped taking records. The three task state
machines model none of them, and deliberately so — they are pure functions of
explicit time and task input, and a machine that also modelled acquisition
failure would be a machine whose replay depended on how the hardware behaved.

This layer decides what a run does about them. It sits in the contract (the
vocabulary) and in the private integration seam (the decisions), and it
changes nothing about the pure machines.

### The line it does not cross

**Only the streaming runtime and its `SafetyController` inhibit or release an
actuator path.** Nothing in this layer touches a `SafetyController`, and
nothing in it has a second opinion about whether a device is live. A paradigm
that decides
its run cannot continue *ends the run*, and the existing fault path does what it
already does: the session aborts the runtime, the runtime inhibits safety, and
the actuator edge is taken down by the one decision procedure there is.

That is why an emergency stop reads the way it does below. The task-side halt
stops commands from being *produced*; the runtime abort stops them from being
*applied*. Both are needed, and only one of them is this layer's.

### The vocabulary

`neurale/experiments/abnormal.h` carries four values, and the reason there are
four rather than two is that a run has to record three different things about
one event.

`AbnormalCondition` says **what was observed** — sixteen values, append-only
like every other persisted enumeration in the contract. Several distinctions in
it are the whole point:

- `source_discontinuity` against `input_gap`. A paradigm may consume no stream
  at all and still lose its input: a pointer that stopped arriving is a gap in
  the task's input without being a discontinuity in anything the runtime is
  carrying.
- `input_gap` against `observer_frame_drop`. A dropped monitoring or rendering
  frame did not change what the task was given. Recording it through the same
  entry point as a pointer gap would make the two indistinguishable in the
  record, and every dropped frame would then invalidate a perfectly good
  acquisition time. If a drop *did* change the task's input, then it is an
  `input_gap`, and the paradigm reports it as one.
- `decoded_command_invalid` against `input_schema_mismatch`. One is the decoder
  producing something a task cannot integrate; the other is the stream not being
  the stream the run prepared against. They are not the same person's problem,
  and only the second is fatal by construction.
- `presentation_failed` against `presentation_evidence_missing`. A presenter
  reporting that it skipped is a statement about a presentation; no report at all
  is a statement about the presenter.
- `presentation_report_unmatched` against both. A report the run cannot match to
  a request it actually emitted is a statement about the presenter's
  bookkeeping, and is not evidence about any trial — neither by supplying
  presentation evidence nor by destroying it.

`AbnormalPolicy` says **how severely this run treats that class of condition** —
`record`, `abort_trial`, `abort_session`. `record` is never the absence of a
policy: it is the statement that this condition does not invalidate a
measurement.

`AbnormalResponse` says **what actually happened** — `recorded`,
`input_refused`, `trial_invalidated`, `trial_aborted`, `session_aborted`.

The policy and the response are recorded side by side because they are two
facts, not one. A policy of `abort_trial` does not always end a trial: a
paradigm whose task state machine owns trial termination and offers no way to
abandon a trial can only mark the trial in flight inadmissible. Writing
`trial_aborted` for a trial that actually ran to its normal end would be this
layer claiming a termination that never happened, and a record that conflated
the two could not be read back. Center-Out produces `trial_aborted`; WebGrid and
Speech produce `trial_invalidated`. That asymmetry is real, and it is in the
record rather than in a comment.

`AbnormalEvent::sequence` is the emission ordinal **within the reporter that
produced it**, not within the session. One run may hold more than one reporter —
Speech holds two, because the scheduler and the presenter bridge are two
producers on two threads — so two events of one session can legitimately carry
the same ordinal. Ordering across a session comes from the control records
themselves, which is where it already was.

`AbnormalPolicySet` groups the severities by what a paradigm can actually
distinguish — five fields, not one per condition, because a configuration with a
knob nothing turns reads as more considered than it is. `policy_for()` maps a
condition onto the set, and four conditions do not consult it at all:
`observer_frame_drop` is always `record`, `input_after_terminal` is always
`record` (the run it would have ended has already ended),
`presentation_report_unmatched` is always `record` (there is no trial it is
entitled to speak for), and `emergency_stop` is always `abort_session`, because
that is the whole of what it means. A configuration cannot make a dropped
monitoring frame end a run, and cannot make an emergency stop not one.

`validate(AbnormalPolicySet)` is called where each Center-Out, WebGrid, or
Speech headless controller freezes its configuration, for the same reason every
other persisted enumeration is checked there: a severity nothing declared
cannot be compared, recorded, or replayed.

One field in the set is narrower than it first reads. `acquisition_fault` maps
`recorder_fault`, `actuator_fault`, `runtime_fault`, and `deadline_missed`, but
it does **not** govern the runtime's own faults or the recorder's own refusals.
When the streaming runtime faults, its fault path decides what happens — and it
is the only thing that inhibits an actuator. When the recorder refuses what a
session produced, `TraceLossPolicy` decides. Both were settled before this
vocabulary existed, and this layer does not duplicate them: two policies for one
decision would be one policy too many. What `acquisition_fault` governs is a
paradigm that *itself* observes one of those conditions and reports it as an
`AbnormalEvent`. No current paradigm does so today, so on the shipped paths the
field
is inert; it is declared because the conditions it maps are part of the
vocabulary and a reporter needs an answer for them.

The defaults are the conservative ones: anything that breaks the continuity a
trial's measurement rests on ends that trial, anything that breaks the run's
ability to record ends the run, and nothing that left the task's input untouched
invalidates anything.

Beside the policy sits a *floor*. A paradigm may state a severity it will not go
below whatever the configuration says, because the accumulation a lower severity
implies demonstrably cannot continue. `escalate()` is that operation, and
Center-Out's discontinuity handling is where it earns its keep.

### No silent success

`abnormal_response_admits_trial()` is the whole of the rule, stated once so that
no paradigm restates it differently: a trial an abnormal condition invalidated or
aborted is never reported as `TrialOutcome::success`.

Where a paradigm's machine would have said otherwise, the **writer** is what
refuses. WebGrid's machine decides a correct selection and is right to; what it
cannot know is that the interval the selection was made over is not a
measurement any more. The trial record written for it carries
`TrialOutcome::aborted`, and the machine's own verdict is kept beside it in the
record's nested document rather than destroyed. The same is true of a Speech
trial whose presentation a presenter says did not happen.

### Where the records go

An abnormal condition is written as a `faults` record whose `code` is
`abnormal_condition_name()` of the condition. There is **no new record kind and
no new body shape**: the recording layer's finalizer accepts three, and an
abnormal condition is a fault by every reading of what that column means.

The code is a *name* rather than a number, unlike a paradigm's own state, cause,
and reason enumerators, which the records carry as opaque integers precisely
because this layer does not own them. An abnormal condition is shared contract
vocabulary, so a reader looking for "what went wrong in this run" does not have
to know which paradigm produced the record in order to find it.

The nested document carries the condition, the policy, the response, the
paradigm, the reporting domain's own status enumerator as a number, and the
trial when there is one. A gap's own record carries the decision taken about it:
one record for one event, because two records would let a reader find the gap
without the decision, or the decision without what caused it.

### Center-Out

The paradigm with a realtime input path, so the invariants are strictest here.

A decoded value that is not finite **never reaches** the cursor, the guidance
blend, or the machine. It is screened in `consume()`, before a row is processed,
and under the default policy the trial it would have belonged to ends too: a
reach whose trajectory has a hole in it is not a reach anyone can score, and
continuing produces a longer trial rather than a better one. A run that wants
the alternative — the whole run down — configures
`decoded_command_invalid = abort_session`, and gets it because it asked rather
than because it was the only thing available.

A frame that is not the stream the run prepared against is
`input_schema_mismatch`, with a floor of `abort_session`. Every frame after it is
wrong in the same way, and a task that restarted its segment per malformed frame
would restart forever while recording nothing usable. Time that moves backwards
is the same condition: a malformed stream, not a decoder producing something odd.

A source discontinuity carries a **floor of `abort_trial`, and the configuration
does not get to lower it**. A hold and a reach are accumulated across consecutive
observations, and there are no consecutive observations across a gap. What the
configuration still chooses is whether the run continues at all. Aborting the
trial is not a new mechanism: it is the machine reset and segment restart the
integration already performs, now with the ended trial written down as a
trial with
`TrialOutcome::aborted` rather than left as a hole in the trial sequence. The
interval on that record is the machine's own — latched from its
`ExperimentEventKind::trial_start`, never guessed at, because a trial record
built from a guessed start is a measurement this layer invented.

`CenterOutControllerConfig::max_observation_interval_ns` is the paradigm's own
bound on a stale command. The cursor is advanced by `velocity * dt`, so a gap
nobody declared turns one decoded sample into a movement over an interval it was
never observed over — which is the shape "a stale command kept a trial alive"
actually takes here. It is not a command reissued; it is one command asked to
stand for a stretch of time the decoder said nothing about. Past the bound the
*interval* is refused rather than the observation: the sample becomes the first
of a new segment, with `dt` zero. Zero means the caller declares no bound, and
for an attached runtime the runtime's own staleness deadlines still apply.

The stale decision travels **on the observation record it was taken about**, not
in a record beside it — the same "one event, one record" rule a discontinuity
follows, and for a second reason as well. `consume()` reserves one trace slot per
row of a frame *before* it processes any of them, which is what makes a frame
either happen or not happen. A row that produced two records would spend a slot
the reservation never covered, and a later row of the same frame would be the one
refused — after the machine had already been reset and restarted for the stale
row. One row, one slot, whatever it had to decide. The only exception is a stale
condition configured to end the run, which has no observation following it to
carry it and takes the row's own slot.

A command that has passed its `valid_until_ns` never reaches the task at all.
`NativeActuator::submit` enforces expiry before entering the device-specific
write, so the task is not asked, produces no trace, and latches nothing — and
the expired command is not reissued to keep the trial moving. The next
observation the task sees is the next one that actually arrived. See the known
gap below about what that costs.

### WebGrid

WebGrid measures a *duration*: time from target onset to selection. A gap does
not corrupt a position here, it corrupts that number.

An input that crosses a gap produces two records — the gap and the step — and
`process()` reserves room for both **before it changes anything**. Reserving one
would let the gap be reported, counted, and the target marked inadmissible, and
then refuse the step for want of room: the caller would be told the input failed
for an input the run had already reacted to, and a retry would report the same
gap a second time, because nothing that advances the pointer clock had run yet.

`WebGridControllerConfig::max_pointer_interval_ns` is checked on every step, and
`note_input_gap()` is how the caller — which owns the pointer source and is the
only thing that can know its data stopped arriving — reports one it detected
itself. Either way the target in flight is marked inadmissible: every record it
produces from then on carries `acquisition_timing_valid = false`, and the trial
record written when the machine finally decides it carries
`TrialOutcome::aborted`.

`note_observer_drop()` is a different call, deliberately, and not a flag on the
first one. It records and it changes nothing, whatever the configuration says.

The raw selection is kept either way. Data is not deleted because it turned out
to be unusable; it is recorded as unusable, which is what makes offline
recomputation of the metric from records possible at all. The summary reports
`invalidated_targets` and `metric_includes_invalidated` beside the machine's own
metric, and does **not** quietly recompute that metric: `WebGridMachine` owns
the formula, and a summary that silently disagreed with the records it was
derived from would be the harder of the two to trust.

A repeated or regressed selection identity is rejected by `WebGridMachine`, as it
always was, and this layer adds no second opinion about it: a duplicate is a
contract violation of the selection stream, not an abnormal condition of the
run.

### Speech

**A semantic time jump is not a fault.** Speech time is semantic: the machine
advances through half-open phases from explicit instants, and a caller that
advances by a whole trial's worth of nanoseconds gets every phase boundary in
between, in order, exactly as if it had been stepped through them. A run that
reported a coarse advance as a condition would fault on every deliberately
coarse advance, and nothing here does.

**A presenter's report is only evidence about a trial when it names the request
that trial actually made.** The writer remembers the identity of each trial's
emitted CONTENT request — its sequence, the instant it was made at, the stimulus,
and the trial — and a report has to agree with all of it, pass
`validate(PresentationOutcome)`, and be the first report on that request. A
report that does not is `presentation_report_unmatched`: recorded, counted, and
changing no trial's evidence in either direction. Attribution by trial ordinal
alone would let a presenter decide, by naming a legal ordinal, whether a trial
the run never asked it about counts as presented — or as failed. The outcome
record carries `matched_request` so a reader can tell a measurement from a report
the run kept but could not attribute.

A matched report of `skipped` or `expired` is `presentation_failed`, named
against the trial it belongs to, and under the default policy that trial becomes
inadmissible.

`SpeechTraceWriter::emitted_content_request()` is how a bridge correlating
reports back — and a test standing in for one — learns what the run asked for.

**The evidence and request identities are sized at construction, not by an extra
call a caller can forget.** `SpeechTraceWriter`'s constructor takes a scheduler
that is already prepared — the supported order — and sizes its per-trial arrays
from that schedule there and then. `prepare_outcomes()` remains, but only to
choose how many presenter reports may be in flight at once; nothing about
matching a report to a request depends on calling it. A writer built *before* its
scheduler was prepared has no arrays to size against, and `validate()` refuses
such a run with `identity_missing` before anything is armed. The failure it
replaces was silent and severe: a run with no evidence array recorded
`presentation_evidence_required: true` beside zero trials missing evidence,
having tracked none of them, and filed every correct presenter report as
unattributable because there was no request identity to match it against.

`prepare_outcomes()` is legal before the session starts and only then, and that
precondition is now answered rather than written down: called on a live run it
returns `already_running` and changes nothing. It replaces the queue rather than
resizing it, and a presenter pushing into the old one from its own thread while
the bridge drains it is a use-after-free no ordering inside the writer could fix.

**Missing presentation evidence is recorded as missing.** `presented_ns` is
meaningful only for `PresentationStatus::presented`; for any other status it is
written as `null` and the record is stamped with the *requested* instant
instead. Writing the field's zero — or, worse, the intended onset — would put an
instant in a column a reader is entitled to read as a measurement, for a
presentation that never happened. The intended onset is the very thing a timing
claim would be checking *against*.

`SpeechTraceWriter::require_presentation_evidence()` is how a caller whose
claim requires presenter-reported software timing says so. Physical-onset claims
require separate hardware evidence and are not satisfied by this flag. It is off
by default, because a semantics-only session has no presenter and every trial
would report missing evidence. When it is on, the run counts the trials that
asked for
their content to be presented and for which no presenter ever said it was, and
reports the count once at summary time — at the end, because a report may legitimately
arrive after the trial it is about has completed, and deciding per trial as it
ended would report a missing report that was merely late.

The Speech side has two reporters rather than one, and merges them: the
scheduler meets its own conditions on whichever thread advances it, and a
presenter's reports are drained on the bridge thread. One shared reporter would
need a lock on a path a realtime producer uses.

### Emergency stop

`ExperimentSession::emergency_stop()` is a separate call from `abort()`, and the
difference is the order.

1. The **paradigm is halted first**, before anything is asked to wind down. An
   emergency stop is a statement about command application; halting the task is
   what stops the next command from being produced, and it takes effect the
   moment the latch is visible. Stopping the runtime first would leave the task
   producing commands for as long as the wind-down took.
2. Then the **runtime is aborted** — which is also what inhibits the actuator
   path, by the runtime's own decision.
3. Then the session shuts down as an abort, with a summary.

`halt()` on a controller **only latches**, and that is not an economy. While the
run is live the trace queue has exactly one producer, and that producer is
whichever thread is driving the paradigm — for Center-Out, the runtime's own.
`halt()` may be called from another thread entirely, so it sets an atomic and
nothing more. The record it owes is produced by `finish_halt()`, called from
`stop_execution()`, which is shutdown step 3 — after the runtime's workers have
been joined, when the producer is provably quiet. Before the join only the
producer writes; after it only the control thread does. That phase separation is
what makes the record safe rather than lucky.

Two records, two codes, and deliberately not one. The session writes
`emergency-stop-requested` carrying the reason the operator gave; the paradigm's
own `AbnormalEvent` is written under `emergency-stop` and says what the task did
about it — which trial it ended, under which policy, with which response. Sharing
one code would make "how many emergency stops did this run have" unanswerable
from the recording, and the request is written even when the paradigm's own
record cannot be, which is why it is not folded into it.

Task input that arrives after a halt is refused and recorded **once**, however
many refusals there are. A runtime that keeps delivering frames after a stop
would otherwise fill the trace queue with identical records and turn the stop
into a trace loss.

### The primary fault survives the shutdown that reacted to it

A runtime that faulted is stopped *because* it faulted, and the stop reports on
the stop. A session that overwrote what it already knew would hand back "the
abort succeeded" as the explanation for a run that failed.

So `SessionOutcome::runtime_status` and `recorder_status` keep the **first**
non-`ok` status, and `SessionOutcome::runtime_fault` carries the runtime's own
`primary_fault()`, sampled before the stop and again after it and kept only the
first time. That fault is also written into the recording, as a `faults` record
under stage `runtime` rather than `experiment` — a recording holding the
experiment's view of a failed run and not the runtime's is missing the half that
says why.

### How a paradigm ends a run it cannot continue

Through the summary the session samples, not through a status returned to a
caller. `ExperimentTraceSource::abnormal_summary()` is a lifetime counter read
the same way `dropped_trace_count()` is, with the session taking a baseline at
`start()` — a controller reset and started again carries every condition of
every run it has had, and a session that measured from zero would open by
charging a clean run for a previous run's conditions.

It has to work that way for the same reason the drop counter does: a controller
running inside a `NativeStreamRunner` reports what it decided to the *runtime*,
not to anyone who could act on it here. A session that learned about abnormal
conditions only from what a caller passed back would report a clean run for one
that ended a trial on every frame.

A latched `session_aborted` escalates through the runtime rather than through
the session's own shutdown, exactly as `TraceLossPolicy::fault` does: it can be
observed on the bridge thread while a caller holds the lifecycle lock, and a
bridge that waited for that lock would stop draining when the shutdown needs it
most. A later `stop()` after such a latch is an abort whatever the caller asked
for.

The session's reaction is written under its own code, `experiment-session-aborted`,
and **not** under the condition's name. The condition already has a record — the
paradigm's own `AbnormalEvent`, which names it, times it, and says which trial it
ended — and a second record sharing that name would make a reader counting
`presentation-failed` records count one failed presentation as two. The primary
condition travels in the reaction's fields (`primary_condition` as its contract
name, `condition` as its number), where it is data rather than an identity. This
is the same split, for the same reason, as `emergency-stop-requested` against
`emergency-stop`.

**The run ends at the latch, not at the returned status.** Every path that
produces a `session_aborted` response also calls `stop_accepting()` on the
controller that produced it, before the record is even queued. The fatal
`StreamStatus` those paths return is how a *caller* learns about it, and under an
attached runtime that status does take the edge down — but a caller-stepped run
has no runtime to take down, and a controller that only returned a status would
keep accepting input after having itself decided it could not continue. So:

- Center-Out under `decoded_command_invalid = abort_session` refuses the next
  `submit()` with `stopped`.
- WebGrid under `input_discontinuity = abort_session` refuses the next
  `process()` with `stopped` — including one carrying a selection, which would
  otherwise be scored and recorded as though the run were still the run.
- Speech under `presentation_failed = abort_session` refuses the next
  `advance()` with `stopped`. Speech is the case that needs saying out loud,
  because the condition is met on the bridge thread, inside `SpeechTraceWriter`,
  while the run that has to stop is the scheduler's; the writer latches it
  through `SpeechHeadlessScheduler::stop_accepting()`.

`stop_accepting()` is deliberately not `halt()`. `halt()` is a stop decided
*outside* the task and owes a record of its own, which `finish_halt()` writes.
`stop_accepting()` owes none: the abnormal event that reached `session_aborted`
**is** that record, and calling `halt()` here would produce a second one naming
the same condition. What the next input still gets is the ordinary
`input_after_terminal` refusal, recorded once, exactly as after a halt.

### Known gaps

An expired actuator command is enforced above the controller —
`NativeActuator::submit` returns before `write()` — so the task cannot see one
and cannot record it. What is recorded is the runtime's own `actuator_deadline`
fault, which the session now preserves. There is no experiment-side
`deadline_missed` record for that case, and there will not be one without a
change to the actuator interface.

WebGrid and Speech cannot *end* a trial in flight. `WebGridMachine` owns when a
target ends and `SpeechMachine` owns when a trial ends, and neither offers a way
to abandon one. `abort_trial` therefore invalidates rather than terminates
there, and the record says so. Making it a real termination is an extension of
those machines, which the current design does not make.

A presenter's report that arrives after the trial it names has already been
written reaches the recording as its own abnormal record, joined to the trial by
identity, but the trial record itself was already written and is not amended. A
reader joining on trial identity sees both; one reading only the trials stream
does not.

`AbnormalPolicySet::acquisition_fault` is inert on the shipped paths. No
current paradigm reports `recorder_fault`, `actuator_fault`, `runtime_fault`,
or
`deadline_missed` as an abnormal condition of its own; the runtime's fault path
and `TraceLossPolicy` own those reactions and are not configurable through this
set. The field is declared because the conditions it maps are part of the
vocabulary, and it is exercised only by test doubles.

A `session_aborted` latched by a condition whose record could not be queued —
the trace queue was full at that instant — still ends the run, but the recording
holds only the trace-loss fault and the session's `experiment-session-aborted`
row, not the condition's own record. The counters in `abnormal_summary()` are
correct either way; what is missing is the detail. Reserving a slot for a
condition that may never occur is the alternative, and it would cost every run a
slot to protect the case where the queue is already lost.

A presenter that reports on one trial's request twice has the second report
rejected as `presentation_report_unmatched`, which is correct but coarse: it
reads the same as a report naming a request the run never made. The record
carries the reported identity, so the two are distinguishable by a reader, but
not by the condition alone.

Nothing here has been measured. The abnormal path allocates nothing — the
controller tests assert that around the conditions they provoke — but no latency
or throughput measurement has been executed on it, so nothing in this section is
a realtime or performance claim.

## Deterministic experiment semantic replay

A recorded run is only evidence if it can be re-executed. Deterministic
semantic replay makes the claim explicit and checkable:

```text
same recorded semantic inputs
  + same configuration
  + same schedule and seed state
  + same paradigm version
    -> the same semantic outputs
```

The engines are `CenterOutReplay`, `WebGridReplay`, and `SpeechReplay` in
`neurale_experiments` -- the value contract, which links nothing. A replay that
needed the streaming runtime or the recorder would be re-executing something
other than the paradigm's own decisions.

### This is not recording replay

The recording layer replays the **data plane**: recorded frames and
discontinuities, back through a stream. Semantic replay replays the
**semantics**: the state machine, the schedule draws, the guidance and
assistance transforms, and the decisions taken about abnormal conditions. The
two are stacked, not alternatives, and neither subsumes the other.

Nothing here submits an actuator command, sleeps, waits on a clock, opens a
window, or reads a device. A replay of an hour-long session takes as long as the
arithmetic takes.

### Two things are inputs, and are never regenerated

**Conditions of a stream that no longer exists.** A gap in a decoded stream, an
interval too long to integrate one command across, an operator's stop. The run
decided these by looking at something a replay does not have. So the condition
enters as a recorded input, and what is regenerated is the *response* -- from
the configured `AbnormalPolicySet`, through the same `abnormal_response_for()`
the live run used -- together with everything the response changed: the trial it
ended, the segment that restarted, the acquisition interval it invalidated.

That is why a recording replayed under a **different severity disagrees**. The
abnormal policies are part of the experiment and the fingerprints do not cover
them, so re-deriving the response is the only thing that makes the policy
replayable at all.

**What an external party measured.** A presenter's documented software time is
the presentation layer's measurement; it is not physical display onset. The
experiments domain has nothing to regenerate it from, and a replay that
reproduced `presented_ns` would be reproducing the number it was handed. A
presenter's report therefore enters as a recorded input, and what is
regenerated is everything the experiments domain decided *about* it: whether
the run could attribute the report to
a CONTENT request the run itself emitted, which condition that produced, and
which response the configured severity gave the condition. Moving the reported
instant changes nothing a replay compares. Changing what the report claims to
answer changes the decision, and the replay says so.

### The answer is a verdict, not a boolean

`ReplayVerdict` has four values, and `match` is only one of them:

| Verdict | What it means |
| --- | --- |
| `match` | Every regenerated output agreed, over evidence that covered the run. |
| `mismatch` | An output differed. `first_mismatch` names the earliest, in replay order. |
| `incomplete` | Everything compared agreed, and the evidence did not cover the run. |
| `rejected` | Replay could not complete a semantic comparison. `rejection` says why; an unsupported operation may be discovered after a verified prefix. |

`incomplete` is the value the whole vocabulary exists for. A recording missing
the stream a replay would have disagreed with looks exactly like a recording of
a run that agreed, and a boolean answer reports the first as the second. A
default-constructed `ReplayReport` is `rejected` for the same reason: forgetting
to run a replay must not read as having run one.

`ReplayCompleteness` says what was missing -- an absent stream, recorded trace
loss, no recorded run end -- and `ReplayIncompletePolicy` says what to do about
it: verify what is available and answer `incomplete` anyway, or compare nothing
at all for a caller whose question is "is this recording replayable". There is
no `inputs_truncated` value in v1 because the recording model has no
input-timeline end anchor from which an engine could establish that fact.

`verify_available` is per stream. An absent stream remains unprovable, but every
stream whose `has_*` flag is true must still have exactly the regenerated
cardinality. Missing one stream therefore cannot hide a short or extra record in
another present stream. Declared trace loss is the exception for a short stream:
the missing item is the loss the recording already reports.

### Provenance identity is checked before anything is regenerated

`check_provenance()` compares paradigm, record version, seed, sampler version,
configuration fingerprint, schedule fingerprint, realized-schedule fingerprint,
metric version, and policy version, in that order, and answers with the first
disagreement. Nothing is replayed until it passes: a mismatch report about a
comparison against a different experiment would describe nothing.

One case is not a disagreement and is handled as itself. A provenance that
names no paradigm or no sampler is `provenance_incomplete` -- two such
provenances agree field by field, and letting that through would run a replay
against provenance that identifies nothing.

`check_provenance()` deliberately does not infer sampler executability from a
zero realized-schedule fingerprint. Whether a draw is needed is paradigm and
schedule-policy specific: Center-Out `repeat_until_success` and an explicit
WebGrid sequence execute no sampler, while their provenance still has no frozen
realized-schedule digest. The concrete replay engine rejects
`sampler_version_unsupported` only on a path that actually requires a draw and
has no authoritative recorded schedule.

WebGrid likewise distinguishes an unknown metric formula from an invalid
configuration. `metric_version_unsupported` means the nonzero version is a
structurally valid persisted identity that this build cannot execute; it is not
reported as `configuration_invalid`.

Speech has three concrete schedule sources. An explicit configuration reads
`explicit_schedule[ordinal]` and executes no sampler. A supported seeded
configuration regenerates all schedules from its determining tuple. In both
cases the config-derived complete schedule supplies candidate provenance, while
the recorded schedule is independently checked as an output stream. An absent
schedule stream therefore gives `incomplete`, not a sampler rejection.

Only a seeded configuration under an unsupported sampler uses recorded
schedules as fallback input. It requests one entry when each trial actually
starts; a terminal input before a future trial does not require that future
entry. Session orchestration provenance fingerprints the complete schedule
prepared by the original build, while its `speech.schedule` stream contains
one entry per trial
that actually started. For a partial fallback stream, this build cannot
independently reconstruct the unseen digest suffix; it checks the remaining
provenance identity and lets `SpeechMachine` validate every fallback entry it
actually consumes.

Consequently a present schedule stream has exactly the replay's started-trial
count, not configured `trial_count`. Missing a schedule for a trial replay has
reached is `sampler_version_unsupported`; schedules for trials never started are
neither required nor synthesized.

Speech keeps one emitted CONTENT-request state per configured trial, as the live
`SpeechTraceWriter` does. A valid presenter report may therefore arrive after
more than 64 later trials; replay introduces no private report-latency window.

### The first difference is the one the run computed first

Comparison stops at the first difference, and within one input the outputs are
compared **in the order the run produced them**: for Center-Out, the cursor,
then the applied velocity, then the guidance sample, then the transitions,
events, target, and trial. A cursor that drifted makes every transition after it
differ, and a report naming the transition would leave a reader to work
backwards to the cursor. Across inputs the order is the run's own, so the
earliest differing transition or event is still the one reported when the
values that produced it agreed.

Real fields are compared **by bit pattern**, not by `==`. A replay is a
determinism claim, and one that admitted two different doubles as equal would be
claiming something weaker than it says. Recorded doubles round-trip exactly --
`std::to_chars` writes the shortest form that reads back unchanged -- so a
correct recording passes the strict comparison.

A stream that runs out while the replay is still producing, or holds items the
replay never produced, is a `stream_length` mismatch: there is no item to say it
on. The exception is a recording that reports trace loss, where a short stream
is the loss it already declared, and the verdict is `incomplete` instead.

Before replay, each input is also checked against its paradigm-owned domain.
Ordinary observations carry no abnormal condition, decision inputs carry a
declared non-`unspecified` condition and declared response, WebGrid
`pointer_gap` and `observer_drop` pair specifically with `input_gap` and
`observer_frame_drop`, and Speech `halt` carries a declared condition. Evidence
that cannot be emitted by the corresponding recording bridge is rejected rather
than downgraded to a semantic mismatch.

### Shared replay rules

Live execution and replay use `abnormal_response_for()` for severity-to-response
mapping and `center_out_condition_handling()` for Center-Out condition policy.
Both use `kNanosecondsPerSecond` for velocity integration. The trace writers
and replay engines share `kCenterOutRecordVersion`, `kWebGridRecordVersion`,
and `kSpeechRecordVersion` so recorded and expected layouts agree.

### What a replay reads, and who builds it

A `*ReplayRecording` contains provenance, an origin instant, semantic inputs,
and recorded output streams with presence flags. Replay engines consume that
model; constructing it from an NRF file is not currently provided.

### Known gaps

The evidence model is native and so is every driver of it. There is no reader
that builds a `*ReplayRecording` from an NRF file, and no Python entry point
that runs a replay; what Python has is the verdict vocabulary, so a report can
be read where reports are read. A caller that wants to replay a recorded session
today has to recover the streams itself.

Center-Out is the only paradigm whose replay is checked against the *real*
controller's trace. WebGrid and Speech are checked against test recorders that
drive their machines the way a live run does -- two independently written
drivers of one contract -- which establishes that the replay reproduces the
pipeline, not that the bridge records every field it needs to.

A recording is replayed as one artefact. The session orchestration note about a
data-plane recording and an experiment-control recording being two NRF sessions
with no identity linking them is unchanged by replay: semantic replay needs
neither the neural streams nor a link to them, so replay did not have to settle
it, and did not.

The benchmark measures representative Center-Out, WebGrid, and Speech semantic
replay sessions in `neurale_experiments_benchmark`. The benchmark intentionally
marks
its checked recordings `incomplete/stream_absent`: it times regeneration of the
recorded semantic inputs while refusing to call absent expected streams a
verified match. Center-Out and WebGrid hold one machine and a cursor into each
recorded stream. Speech additionally allocates one request state and one
schedule entry per configured trial during `prepare()`; the measured `run()`
path does not resize them. The JSONL rows report p50/p95/p99, input count,
allocation backend, exact command, build, OS, and CPU. They characterize only
the recorded environment and are not a replay service-level objective.

## Offline traceability and provenance resolution

`neurale.experiments.traceability` owns the offline join from one experiment
outcome back to the evidence that was actually recorded. It is a
dependency-light Python module: it imports neither streaming nor recording/NRF,
and receives a reader-derived `RecordingEvidence` snapshot instead. This keeps
session completeness with the recording layer while making its verdict
participate in every query.

Evidence identity is the session-scoped stable compact pair
`(EvidenceKind, ordinal)`. The enclosing `RecordingEvidence.session_id` gives
that pair its global scope. `TrialIdentityEvidence` separately preserves all
five fields of the native `TrialIdentity`: ordinal, key, block, target, and
stimulus. Trial membership follows the native `same_trial` rule (ordinal plus
block); duplicated key/target/stimulus fields are retained and checked where a
writer persisted them as corroborating evidence. Existing trial and acquisition
evidence are checked against each other even when their schedule parent is
missing. Joins use identities, never timestamps. A source dependency is a half-open
`[sample_start, sample_stop)` interval and a feature may name more than one;
`first_frame_sequence` and `last_frame_sequence` are optional together because
the current pipeline does not promise that a window has one parent frame. An
unrecorded raw stream produces `RAW_STREAM_NOT_RECORDED` and is not returned as
supporting source evidence.

The three indexes are deliberately paradigm-specific:

- `CenterOutProvenanceIndex` follows feature observation, decoder output,
  applicable guidance, configured assistance, experiment state/target, final
  command request, and synchronous application result. `application_code` is
  the existing `CommandApplication` value and `status_code` is an opaque
  synchronous status. The current writer's `application_scope` is
  `center_out.headless_cursor`: it proves that `CenterOutController` integrated
  the final velocity into its headless cursor, not that a `NativeActuator` or
  vendor device accepted or acknowledged it. The linked experiment-state row
  describes the pre-step state, while assistance/command/application repeat the
  writer's post-step snapshot identity; those two identities are retained but
  are not falsely required to be equal on a step that advances a trial.
- `WebGridProvenanceIndex` follows the pointer source/update, active target,
  `SelectionEvent.id`, selected cell, correctness, the completed trial only for
  a correct selection, and that selection's raw metric contribution. A
  misselection does not fabricate a successful trial. The trial stored on a
  `webgrid.pointer` row is the writer's post-step snapshot and is preserved as
  such; it is not substituted for the selection event's own trial/target parent
  when a correct selection advances the schedule in that same step.
- `SpeechProvenanceIndex` follows one trial ordinal and block through its realized
  schedule, ordered semantic phases, presentation requests, optional presenter
  reports, acquisition interval/markers, and final trial. BLACK and CROSS use
  the unset stimulus ID, CONTENT uses the scheduled stimulus, and INTER_TRIAL
  emits no request. Requested time and intended onset stay on the request while
  presenter-reported software time stays on the outcome. Requests and presenter reports
  retain the writer's complete trial identity. BLACK, optional CROSS, and
  CONTENT each require their semantic presentation request, but the current
  writer attributes presenter reports only to the CONTENT request. Therefore
  only CONTENT requires an outcome persisted with `matched_request=true` for a
  complete trace. BLACK/CROSS reports are not required; an unmatched report for
  either remains auditable input evidence but is not returned as supporting
  trial-presentation evidence.
  A complete trace must have exactly BLACK, optional CROSS, CONTENT, and
  optional INTER_TRIAL in that order. An invalidated trial must be recorded as
  ABORTED; a contradictory successful outcome is rejected.

The result is `COMPLETE` only with verified complete session accounting, a
complete experiment trace, and every required/applicable link present.
Unverified or omitted evidence is `PARTIAL`; recorded session/trace loss is
`INCOMPLETE`; a fully supported command whose synchronous application faulted
is `FAULTED`. Every partial or incomplete answer includes structured
`EvidenceGap` values, so absence is never represented by a guessed parent.

`center_out_provenance_from_records` and
`webgrid_provenance_from_records` consume committed NRF control rows returned by
`NrfReader.iter_records`. They parse only experiment-domain record bodies and
join only the explicit ordinals persisted by the trace writers. Center-Out
source/feature/decoder evidence is supplied separately because those records
belong to their data-plane owners. No row position or timestamp creates a
parent. The builder verifies the supplied decoder output's frame sequence,
absolute sample index, and time against `center_out.observation_provenance`,
and checks command/outcome row times and complete trial identities; the
experiment-local decoder ordinal alone is not upstream identity. A WebGrid
correct selection whose acquisition interval was invalidated
keeps its raw metric contribution and is linked to the recorded aborted trial;
an incorrect selection cannot carry a completed trial. WebGrid also checks the
trial row's repeated target parent, selected/intended cells, correctness,
target onset, acquisition duration/validity, paradigm, and trial identity
against the selection and target records instead of discarding them. Trial
acquisition duration is validated directly from selection time minus target
onset even when metric evidence is absent. The builder also rejects a
`webgrid.selection_correct`/`webgrid.selection_incorrect` name that contradicts
the persisted `correct` field, and validates the persisted elapsed field for
both correct and incorrect selections. Embedded pointer sources use an empty
stream and `webgrid.pointer`; external sources use a non-empty stream and
`webgrid.pointer_reference`. The source declaration, row name, and row stream
must all agree. Center-Out guidance and assistance rows self-identify with their
own `observation_ordinal`; the corresponding parent link must agree and cannot
rename the child evidence.

## Public surface and evidence

The public API exposes shared experiment values and concrete paradigm modules;
the headless controllers, trace writers, and `ExperimentSession` remain private.
The {doc}`presentation architecture <experiment_presentation>` describes their
display boundary. Benchmark entry points, measured scope, and interpretation
rules live in {doc}`benchmarks`; benchmark results are not portable deadlines.

## Unique-owner table

Each row has exactly one owner.

| Capability | Owner |
| --- | --- |
| Trial and phase state transitions | the concrete paradigm |
| Semantic target, cell, and cue identity | the concrete paradigm |
| Hit, containment, dwell, and selection rules | the concrete paradigm |
| Task metrics (success rate, time to target, throughput, misclicks) | the concrete paradigm |
| Deterministic per-trial schedule sampling | shared experiment schedule contract |
| The stable sampler: pseudorandom algorithm, bounded-integer mapping, sampler version | shared experiment schedule contract |
| Semantic stimulus catalog (`stimulus_id` to content, label, metadata) | `neurale.experiments.speech` |
| Experiment time, trial identity, events, outcomes, bounded trace | shared experiment value contract |
| Presentation and command requests | shared experiment value contract |
| Task-specific reference/guidance velocity | the concrete paradigm (Center-Out for `CenterOutGuidance`) |
| Guidance precision and other guidance parameters | the concrete paradigm's guidance configuration |
| Acceptance region and cursor extent used by containment | the concrete paradigm |
| Rendered target and cursor radii | experiment presentation concrete configuration |
| Generic velocity assistance and shared control, including `OrthoImpedance` | `neurale.experiments.assistance` |
| Startup inhibition, deadlines, expiry, watchdog, fault propagation | `neurale.streaming` |
| Actuator inhibition and release, under every condition | `neurale.streaming` and its `SafetyController`; the experiments domain has no second opinion |
| What an abnormal condition *is*, and how policy, severity and response are named | shared experiment value contract (`abnormal.h`) |
| What a run does to a trial after a fault or discontinuity | the concrete paradigm's controller, under its configured `AbnormalPolicySet` |
| Whether a trial an abnormal condition touched may still be reported a success | shared experiment value contract; the trace writers enforce it |
| Actuator submission and its synchronous application status | `neurale.streaming` |
| Device-specific asynchronous hardware acknowledgement, if a device has one | that device's adapter, in Devices |
| Acquisition and actuator hardware adapters | Devices |
| Feature extraction | `neurale.features` and the internal pipeline |
| Model inference and decoder state | `neurale.decoding` and the internal pipeline |
| Session persistence, completeness, accounting | `neurale.recording` and NRF |
| Session bring-up and shutdown ordering across the runtime and the recorder | private `neurale_execution` |
| Which existing NRF control stream each experiment value belongs in | private `neurale_execution` |
| Whether the *experiment trace* is complete | private `neurale_execution`; whether the *session* is complete stays the recording layer's |
| Offline experiment provenance resolution | `neurale.experiments.traceability`; the recording layer remains owner of recording/accounting facts supplied to it |
| Pixels, windows, monitors, input devices, render threads | experiment presentation |
