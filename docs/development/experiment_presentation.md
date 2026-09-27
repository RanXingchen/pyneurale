# Experiment presentation architecture

## Scope

This document defines the optional presentation boundary and the detailed
Center-Out, WebGrid, and Speech presenter/recording integrations. SSVEP also
uses the optional native presentation build, but its `SSVEPDisplay` and session
follow a separate task-specific path described in the {doc}`SSVEP guide
<../user_guide/ssvep>`. None of these presenters owns task semantics or
physical display-onset evidence.

The {doc}`experiments contract <experiments>` remains normative for task
meaning.

## Ownership boundary

The dependency and authority direction is:

```text
Experiments domain concrete paradigm semantics
    snapshots + PresentationRequest
                 |
                 v
Presentation concrete presenter/input integration
    software PresentationOutcome + timestamped input values
                 |
                 v
Experiments orchestration and recording/provenance
```

The following rules are frozen.

1. The experiments domain owns task semantics. Presentation owns presentation and input only.
2. Presentation is concrete-paradigm integration, not a generic experiment GUI
   framework.
3. Shared native code may contain only the window, rendering, text, and input
   primitives needed by concrete presenters. Those primitives remain private.
4. A renderer consumes experiments-domain geometry and snapshots. Its viewport, aspect-ratio,
   DPI, framebuffer, and drawing choices cannot change task coordinates,
   containment, target identity, selection correctness, or task-state truth.
5. Semantic pointer and selection input is converted into the existing experiments-domain input
   contracts where applicable. Window-close, ESC, fullscreen, and other
   presentation-control input remains presentation/orchestration control and
   is not promoted into task-semantic input. A presenter never calls or mutates
   an experiments-domain state machine directly: task-semantic input reaches the experiments domain only through the
   contracts the experiments domain already owns, and presentation control reaches orchestration as
   a session or stop request rather than through a task state machine.
6. A software presentation timestamp is not physical display onset.
7. Window event handling, drawing, buffer submission, and text shaping never
   execute on a streaming critical thread.
8. Initialization, render, input, deadline, and shutdown failures are observable
   by experiments-domain orchestration and its existing abnormal-condition policy, and can be
   persisted by the existing experiment/control recording path.
9. Presentation remains optional at build, package, and import boundaries.
   Headless experiments-domain execution remains supported even though presentation is required for a
   complete interactive experiment stack.
10. Speech stimulus text is canonical UTF-8.
11. HarfBuzz performs Unicode shaping; FreeType supplies glyph rasterization
    and metrics. Presentation does not implement per-codepoint text layout.
12. Font selection is presentation configuration, not Speech task semantics.
13. Prepared Speech presentation identifies the exact font bytes and face used.
    The font file is read once; the hash and FreeType both consume that one byte
    buffer, so the recorded hash is the hash of the bytes actually rasterized.
    Where reproducibility is required, the configuration records a stable hash
    of those bytes; the recorded identity uses SHA-256.
14. Preparation shapes the complete Speech stimulus catalog and verifies every
    required glyph before experiment execution. A missing glyph is a
    prepare-time presentation error.
15. Presentation neither substitutes a fallback font nor normalizes, rewrites, or
    replaces stimulus text. The semantic text supplied by the experiments domain is unchanged.

## Concrete integrations, not a framework

Each concrete presenter has a task-specific boundary:

- Center-Out draws the experiments-domain target/cursor snapshot and turns presentation/control
  input into typed values. The experiments domain remains the only owner of containment, hold,
  timeout, target schedule, guidance, assistance composition, and success.
- WebGrid draws the experiments-domain logical grid, active target, and pointer. It reports
  pointer updates and explicit selection events; the experiments domain remains the only owner of
  cell mapping, correct/misclick semantics, target retention, and metrics.
- Speech resolves a prepared `stimulus_id`, shapes its unchanged UTF-8 text,
  and presents BLACK, optional FIXATION_CROSS, and TEXT_CONTENT requests. The experiments domain
  remains the only owner of the phase sequence, intended onset, duration, trial
  schedule, and completion semantics.
- SSVEP's separate `SSVEPDisplay` draws a task snapshot with luminance phase
  derived from scheduled stimulation time. It reports software render times and
  stop input. The SSVEP machine owns phase and selection semantics; the session
  owns recording. Display refresh metadata and swap return are not physical
  timing measurements.

There is no public renderer, presenter, window, scene, widget, backend, plugin,
or scheduler abstraction. Private sharing is limited to proven common work:
GLFW context/window/event handling, an OpenGL 3.3-compatible Core Profile
subset, fixed drawing resources, prepared input handoff, and the
HarfBuzz/FreeType text pipeline. Concrete paradigm integrations own the mapping
from their approved experiments-domain values to those primitives.

## Threading, safety, failure, and recording

Presentation runs outside `NativeStreamRunner` critical producer, processor,
consumer, and actuator paths. Any handoff from a critical path is bounded and
prepared; renderer work, OS window calls, OpenGL calls, font access, allocation,
logging, and filesystem access happen on the presentation side. Presentation
backpressure or queue overflow is an explicit failure, never permission to
block a critical thread indefinitely.

Presentation does not own runtime safety. `NativeStreamRunner` and its
`SafetyController` retain deadline, watchdog, primary-fault, actuator
inhibition, cancellation, and release authority. A presenter failure becomes a
typed outcome or explicit failure delivered to experiments-domain orchestration. The existing
`presentation_failed` / `presentation_evidence_missing` policy determines the
trial or session consequence; presentation does not invent a second severity policy.

Presentation does not own a recording format or writer. It supplies bounded presentation
and input evidence carrying the existing request, trial, stimulus, sequence,
and time identities. The experiments-domain integration and recording path remain the owners
of trace encoding, completeness, faults, and persistence. Failure to preserve
required evidence must make the trace/session explicitly incomplete rather than
silently presenting a complete result.

## Time contract

Four instants must remain distinguishable:

```text
Experiments domain requested_ns / intended onset_ns
    != software render submission time
    != software present/swap completion time
    != physical pixel/photon onset
```

`PresentationRequest.sequence`, not a timestamp, identifies the request.
`PresentationOutcome.request_sequence` refers to that request and
`presented_ns` is presenter-reported experiment time only when status is
`presented`. The implementation must document the exact software event sampled
for that value and the clock conversion used to express it in experiment time.
Neither `glfwSwapBuffers` return, CPU submission, nor an estimated GPU
completion time is physical-onset evidence.

The private surface samples `submitted_renderer_ns` from `std::chrono::steady_clock`
immediately before uploading/drawing the prepared frame, and samples
`presented_renderer_ns` immediately after `glfwSwapBuffers` returns. One affine
origin mapping converts both samples and every input callback timestamp into experiments-domain
experiment nanoseconds. The mapper rejects renderer-time regression and integer
overflow. `requested_ns` and `intended_ns` pass through unchanged, so neither is
overwritten by either software sample. The field named `presented_ns` therefore
means **swap-return software time**, not GPU completion, scanout, pixel onset, or
photon onset.

Physical onset requires separately acquired evidence such as a photodiode or
display trigger. Such hardware enters through Devices/Recording and provenance;
presentation may correlate it, but cannot manufacture it. Semantic replay regenerates experiments-domain
decisions and consumes recorded presentation outcomes as external evidence.

## Prepared Speech text contract

Preparation receives the immutable experiments-domain stimulus catalog and presentation-only
font configuration. It reads the exact font file once into an owned byte buffer,
records its stable identity, face index, and SHA-256 of those bytes, then feeds
that same buffer to FreeType via `FT_New_Memory_Face` while HarfBuzz shapes every
catalog string against it. Because the hash and the rasterizer share one read of
one buffer, the recorded SHA-256 is always the SHA-256 of the bytes actually
shaped and rasterized; the path is never opened a second time between hashing and
glyph preparation, so a path that is a symlink or is replaced between two reads
cannot make provenance and the rendered font diverge. Glyph IDs, positions,
metrics, and render resources are prepared before the run, and the byte buffer
is released once the face and atlas are built.

The UTF-8 byte sequence remains the semantic stimulus. Shaping configuration
may state direction, script, language, and supported OpenType features, but may
not change the string. A catalog entry that cannot be decoded as UTF-8, shaped,
or fully resolved in the selected font fails preparation. There is no runtime
fallback glyph, fallback font, or unspecified normalization.

The private atlas takes one exact font path and face index, validates UTF-8,
shapes the complete supplied catalog with HarfBuzz, rejects glyph ID zero, and
rasterizes all required glyphs with FreeType during `prepare`. The prepared atlas,
glyph positions, and texture are reused by the render loop; that loop performs no
font discovery, filesystem search, shaping, or glyph rasterization. The later
concrete presentation configuration remains responsible for recording the exact
font identity and required SHA-256 alongside experiment evidence. The atlas derives
script and direction from the unchanged Unicode string and freezes shaping
language to BCP-47 `und`, so process locale cannot silently alter shaping.

## Private surface contract

Logical experiment coordinates are Cartesian (`x` right, `y` up). They map to
the top-left-origin framebuffer through either `fit_letterbox` or explicit
`stretch`. Window pointer coordinates first use the independent window-to-
framebuffer scale, which preserves HiDPI behavior. Inverse pointer mapping accepts
only the half-open viewport; letterbox pixels are outside the logical task space.
Resize callbacks refresh window and framebuffer dimensions without changing the
logical rectangle, and a resize is a coordinate-mapper refresh only -- it is
never enqueued, since no concrete presenter consumes it as orchestration
evidence.

One presentation thread owns GLFW, the OpenGL context, drawing, event pumping,
input draining, preparation, and close. Cross-thread cancellation is the only
cross-thread control operation: it sets an atomic request and touches nothing
else -- not the window, not GLFW, not lifecycle or status, which the owner
thread owns. The owner thread observes the request on its next `pump_events()`
and performs the GLFW state transition there. The runtime polls with
`glfwPollEvents()`, so there is no blocking event loop to wake; a future move to
`glfwWaitEvents()` would need a separate, designed wake/lifetime handshake
rather than a GLFW call made from the cancelling thread. Other cross-thread
surface use fails with `wrong_thread`. Close and destruction are owner-thread
work: a live surface closed or destroyed from the wrong thread is rejected as
`wrong_thread` and left for the owner to tear down, never calling GLFW
off-thread. A surface that was never opened, or is already closed, has no live
GLFW state, so closing it from any thread is a harmless no-op.

Preparation fixes vertex, draw-batch, input-event, text, glyph, and atlas bounds.
The OpenGL 3.3-compatible Core path uses one VAO, one reusable VBO, fixed shaders,
and a prepared atlas texture. Lines, polylines, rectangles, circles, fixation
crosses, and shaped-text quads append only within those reserved capacities;
capacity exhaustion and input overflow are explicit terminal surface failures.
There is no immediate-mode path, scene graph, recording/filesystem work, or
streaming dependency.

GLFW callbacks enter a fixed ring in callback order, but only after a
per-presenter `InputAdmission` policy admits the event kind. A single
`glfwPollEvents()` runs every registered callback synchronously, so a pointer
or keyboard burst could otherwise overflow the fixed ring (a sticky
`input_overflow` fault) before the presenter drains. Admission drops the kinds a
presenter never consumes right at the callback: Speech and Center-Out admit only
key and window-close (pointer motion and mouse buttons never reach the ring),
while WebGrid admits pointer, mouse, key, and window-close. Keyboard admission is
ESC-press-only: keyboard orchestration is just ESC, so only an ESC press is
ever admitted -- other keys and key release/repeat are dropped at the callback and
can never fill the ring. This is a fixed private policy, not a generic key
registry or per-presenter key routing framework. Each admitted event receives a
monotonic renderer timestamp, mapped experiment timestamp, and strictly
increasing ordinal. Pointer motion and mouse-button events remain distinct;
keyboard and close events retain their own kinds.

### Center-Out 2D presenter

The private `CenterOut2DPresenter` consumes the immutable experiments-domain
`CenterOutTask`, one `CenterOutSnapshot`, and an explicit cursor position.
Its fixed-capacity render plan draws the cursor and outlines every target
location. The current target is filled and highlighted: the center target
during the center leg, or the selected outward target during the outward leg.
Feedback dwell keeps that target highlighted and maps the already-decided
move, hold, success, and failure states to presentation-only colors. The public
configuration supplies only window intent and an optional colour theme. Session
preparation derives the native geometry unit, viewport, target radius, cursor
radius, aspect policy, and input capacity from the authoritative task. These
draw values never enter experiments-domain containment, hold, timeout,
scheduling, guidance, or assistance decisions.

Logical coordinates are sent to the shared presentation surface without clamping. Any
letterboxing or OpenGL clipping is visual only and is never returned as a task
position. Updates replace the previous unpublished snapshot when presentation
falls behind. ESC and window-close input produce timestamped presentation
control events for orchestration; neither event performs a semantic task
transition.

### WebGrid presenter and pointer input

The private `WebGridPresenter` draws the experiments-domain task bounds and the exact cell
rectangles returned by `webgrid::grid_cell()`. It highlights only the active
target already named by `WebGridSnapshot`; selection feedback coloring is
derived from the experiments domain's already-decided `WebGridSelectionRecord` (presentation never asserts
correctness itself). Presentation does not locate a pointer in a cell,
construct `SelectionEvent`, score correct/misclick, advance a target, or update
metrics.

`WebGridPointerInputAdapter` preserves the native callback ordinal and mapped
renderer/experiment timestamps. Pointer motion produces a pointer update only.
One configured mouse-button press produces at most one `selection_request`
containing the logical pointer at press time; release, repeat, and other buttons
produce none. The experiments-domain combined input contract then applies that pointer before
constructing/evaluating the discrete selection at the same timestamp.

The coordinate mapper retains an unclipped inverse result for letterbox and
outside-window positions while reporting viewport membership separately. This
lets the experiments domain apply its own half-open task/cell geometry even at exact edges; presentation never
turns a viewport flag into a cell decision. Rendering replacement or loss has no
pointer-discontinuity meaning.

### Speech cue presenter and outcome handoff

The private `SpeechCuePresenter` consumes only the BLACK, FIXATION_CROSS, and
TEXT_CONTENT `PresentationRequest` values emitted by the experiments-domain Speech machine. It
does not build phase intervals, enable/disable CROSS, sample durations, or
advance a trial. A disabled CROSS therefore produces no presenter input at all;
there is no zero-length presentation placeholder.

Preparation copies the immutable experiments-domain catalog, validates unchanged UTF-8 bytes,
shapes every single-line stimulus through HarfBuzz, rasterizes every required
glyph through FreeType from the single read font byte buffer, and uploads one
reusable atlas. Presentation deliberately has no multiline layout contract; carriage
return and line feed are rejected at prepare rather than interpreted
inconsistently. The exact font path, face, and SHA-256 of the bytes read once
by the presenter are retained as presentation identity, and the same bytes feed
`FT_New_Memory_Face`. Missing Chinese glyphs, invalid UTF-8, duplicate text
identity, and atlas capacity failure are explicit preparation failures; no
fallback font, replacement glyph, normalization, transliteration, or timed-path
font lookup is performed.

Speech has no pointer or selection semantics, so the presenter's
`InputAdmission` policy admits only key and window-close, and the surface's
fixed key policy further admits only an ESC press: pointer motion, mouse
buttons, non-ESC keys, and key release/repeat are dropped at the callback and
can never fill the input ring, regardless of how the orchestrator schedules
drains. `poll_control` drains whatever was admitted and translates ESC and
window-close into timestamped orchestration control events; it never mutates an
experiments-domain SpeechMachine.

Every accepted request produces a `SpeechPresentationEvidence` value containing
the original request, its `PresentationOutcome`, the surface status, and the
separate requested, intended, submitted, and swap-return software times. The
outcome is published through a prepared fixed-capacity SPSC handoff owned by
presentation, not through the experiments-domain semantic trace queue. Orchestration drains that handoff
and calls the existing `SpeechTraceWriter::report_presentation()` seam after
the corresponding semantic request has entered the trace writer. The trace
writer retains fixed BLACK/CROSS/CONTENT request slots per trial, so failures
in any presented phase are attributed to the request that actually caused
them; only a successfully presented CONTENT request satisfies the trial-level
content-evidence requirement. A full
handoff refuses the presentation before drawing, increments an explicit drop
count, and returns `presentation_handoff_overflow`; renderer failure yields a
`skipped` outcome. Both therefore enter the experiments domain's existing `presentation_failed` or
trace-incompleteness policy instead of becoming silent UI loss.

`attempt_ns >= valid_until_ns` produces an `expired` outcome without drawing.
If swap-return time crosses the deadline, the software timing remains in the
presentation evidence but the outcome is not reported as a valid presentation. Request
sequence is strictly increasing per presenter run, so duplicate or reordered
requests cannot create a second outcome. `presented_ns` remains the documented
swap-return software observation and is never physical display onset.

## Recording and provenance integration

Presentation recording is an optional internal target above both owners:
`neurale_experiment_presentation_recording` depends on the private presenter
target and the existing private experiment execution target. The presenter target
itself still has no recording dependency. Its concrete bridge functions copy
only compact prepared values into the three concrete trace writers; JSON
encoding, spool submission, NRF finalization, completeness, and recovery remain
experiments-domain work and never execute in a render callback.

Center-Out and WebGrid record lifecycle/software-timing reports in fixed-capacity
SPSC queues. A rendered report retains requested, intended, submit, and
swap-return instants plus both identities needed to interpret it: the
presenter-local update ordinal and an explicit experiments-domain source ordinal. The source
ordinal is assigned by the Center-Out/WebGrid controller when it accepts the
semantic update, carried in its trace, and persisted unchanged by the writer;
ordinal zero is valid, so absence is represented by `has_source_ordinal=false`.
Center-Out provenance joins a presented frame to the exact
`center_out.observation_provenance.observation_ordinal`; an unrelated or omitted
source does not satisfy another command's presentation requirement.

A render failure is recorded as two distinct facts: the concrete
`center_out.presentation` or `webgrid.presentation` implementation evidence,
and one `AbnormalCondition::presentation_failed` row describing the run's
decision. Frame evidence carries the exact trial identity. A fixed-capacity
handoff returns that failure to the concrete task-owner thread, where the
existing policy aborts/reset Center-Out or invalidates WebGrid as configured;
the recording-drain thread never mutates either state machine. Lifecycle faults
before or after an active run have no trial and are recorded without fabricating
one. A runtime surface/input fault while the run is active uses a distinct
bounded handoff and carries the exact trial identity captured by orchestration
when the fault occurs. The Center-Out/WebGrid task-owner requires that identity
to match its still-active trial. If the trial has advanced, delivery is rejected
and counted as trace loss; the fault is never transferred to a later trial.
Close/cancel lifecycle records do not replace a runtime or experiment primary
fault. Queue refusal or identity rejection increments the existing monotonic
drop count, so `ExperimentSession` applies its configured trace-loss policy.

Each concrete presenter also freezes its presentation provenance before the
experiment session writes configuration. `center_out.presentation_config`,
`webgrid.presentation_config`, and `speech.presentation_config` retain the
renderer-independent logical geometry and the visual parameters that explain
what was shown. Bounded companion records retain presentation-only colours and
window/fullscreen/monitor/swap requests; the WebGrid record also retains its
selection button and pointer radius. Configuration is accepted once and is
rejected after the session has written configuration, so an ambient presenter
change cannot reinterpret an already-started run. Pure capacity fields such as
input queue length, vertex limits, atlas dimensions, and glyph counts are not
scientific presentation provenance.

The configuration seams accept the prepared concrete presenter, not another
caller-owned copy of its configuration. Center-Out and WebGrid also read the experiments-domain
task configuration from the trace writer's actual controller and require its
fingerprint to match the task frozen by the presenter. Speech reads both the
presentation configuration and `FontResourceIdentity` from the prepared
`SpeechCuePresenter`. A caller therefore cannot record a different visual
configuration, task geometry, or synthetic font hash as though the presenter
used it.

Speech also freezes the fingerprint of the exact stimulus catalog it prepared
for shaping and rasterization. The configuration bridge requires it to equal
the fingerprint of the catalog frozen by the bound `SpeechHeadlessScheduler`.
Catalog content participates in that digest, so reusing the same stimulus ID
for different text is rejected before presentation configuration is recorded.

For WebGrid, the presentation callback ordinal and renderer timestamp travel
with the exact pointer value accepted by `WebGridHeadlessController`. The
resulting `webgrid.pointer` row stores that identity; the selection row retains
its existing `pointer_update_ordinal` link rather than duplicating the presentation
fields. Offline provenance therefore follows click -> accepted pointer ->
selection deterministically, without correlating equal timestamps. Headless
inputs explicitly carry no presentation identity.

Speech continues to record `speech.presentation_request` and
`speech.presentation_outcome` separately. The presentation bridge adds intended onset,
renderer/experiment submit time, renderer swap-return time, and the private
implementation status to the outcome record while experiments-domain policy still consumes
only `PresentationOutcome`. Missing or failed actual evidence remains missing
or failed: neither requested nor intended time is copied into `presented_ns`.
Offline traceability can require this software evidence and then reports a
partial verdict when it is absent.

Surface lifecycle failures use the separate bounded
`speech.presentation_lifecycle` seam. Pre/post-run failures carry no trial. A
runtime fault carries the exact active trial identity through a fixed-capacity
handoff to `SpeechHeadlessScheduler`. The scheduler validates it against its
producer-side current trial and applies the configured `presentation_failed`
policy on the task-owner thread; a foreign or stale identity is rejected and
counted as trace loss rather than transferred to a later trial. The lifecycle
record remains implementation evidence and does not create a second abnormal
decision after the scheduler accepts the handoff. Runtime failure remains
lifecycle evidence rather than fabricating a
`PresentationOutcome` for a request it did not answer. Request-specific
failures continue to use `PresentationOutcome` and are not duplicated as
lifecycle outcomes.

Every Speech `PresentationRequest` is also published into a prepare-sized
producer-side identity registry before the scheduler trace becomes visible.
Outcome matching and request queries use that registry, so a renderer outcome
that reaches the recording queue before the corresponding scheduler trace is
drained is still attributed deterministically. Duplicate outcomes are rejected
atomically.

Speech additionally records `speech.presentation_font` with the configured font
path, face index, and SHA-256 from `FontResourceIdentity`, plus pixel height in
`speech.presentation_config`. The bridge reads that identity directly from the
prepared presenter and verifies its path and face against the presenter's frozen
configuration; the hash is therefore
the identity of the same one-read byte buffer that FreeType consumed, not a
second filesystem observation or an unspecified fallback font.

The bridge is noncritical and bounded. It writes no filesystem or NRF data,
does not change task geometry, schedules, selection correctness, or Speech phase
timing, and adds no public presenter/recorder API.

## Optional build and import boundary

The private `neurale_experiment_presentation` target depends on GLFW, OpenGL,
FreeType, and HarfBuzz and is built only with
`NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON`. Those libraries do not become dependencies of the
`neurale_experiments` value target and do not become mandatory core-wheel
runtime dependencies. Build discovery follows the repository's explicit
optional-feature convention: an explicitly requested presentation build fails
when a required dependency is unavailable; it never silently produces a
headless fallback while claiming presentation support.

GLFW, FreeType, and HarfBuzz are first searched for as system packages
(`find_package`); only when a system package is not found does `FetchContent`
download and build a private static copy pinned by URL and SHA-256 inside the
build tree (`build/_deps`). OpenGL is the system graphics stack and is never
bundled. The three are linked statically into the private presentation target,
so a packaged wheel's runtime dependency on them is intended to be only the
system OpenGL and display runtime; verifying that with `auditwheel`/`delocate`/`delvewheel`
remains a packaging-step concern rather than something `FetchContent` settles.
The standard CMake offline switches are the escape hatch, with no second
PyNeurale option: `FETCHCONTENT_FULLY_DISCONNECTED=ON` forbids downloads, so a
missing system package plus a forbidden download is a configuration error, and
`FETCHCONTENT_SOURCE_DIR_<NAME>` points at a local unpacked source tree for
air-gapped or packager builds.

The approved public ownership is an optional
`neurale.experiments.presentation` boundary for concrete configuration/control.
Importing `neurale` or `neurale.experiments` must not import that submodule,
initialize GLFW/OpenGL, inspect monitors, create a window, load a font, or probe
input devices. Optional presentation bindings are compiled into the aggregate
`neurale._native` extension. Loading that extension registers types but does not
initialize graphics; window creation remains an explicit presenter operation.
The presentation layer exposes `CenterOutPresenter`, `WebGridPresenter`,
`SpeechPresenter`, and `SSVEPDisplay` with their configuration/value types.
The shared `PresentationSurface`, GLFW callbacks, OpenGL resources, shaders,
atlas, and input queue remain private;
there is no generic presenter or renderer base class.

Release support is narrower than successful dependency discovery. The current
supported presentation profiles are Windows 11 x86-64 desktop OpenGL and
Ubuntu 24.04 x86-64 X11/Xvfb with Mesa, as frozen in the repository-root
[release/support matrix](https://github.com/RanXingchen/pyneurale/blob/main/SUPPORT.md).
Native Wayland, macOS/Cocoa, other window systems/GPUs, and physical display
onset measurement are unavailable without executed release evidence.

## Benchmark evidence

`neurale_experiment_presentation_benchmark` is built only when presentation and
native benchmarks are both enabled. It opens hidden real OpenGL windows and
characterizes Center-Out snapshot update/render, a 12x12 WebGrid update/render,
WebGrid pointer-to-logical-event conversion, prepared ASCII/Chinese/mixed and
representative Speech text, Speech request-to-swap-return software latency,
catalog preparation, and a repeated 100,000-presentation resource-stability run.
HarfBuzz shaping time and FreeType glyph-cache construction time are measured
inside prepare and reported separately. Steady-state rows verify that prepared
text/glyph capacities do not change and report unexpected runtime glyph-cache
misses.

JSONL records the exact command, Release/compiler/OS/CPU, actual OpenGL vendor,
renderer and version strings, reported monitor refresh rate, window/framebuffer
size, monitor/fullscreen mode, swap interval, exact font paths, CPU use, RSS,
allocation count and p50/p95/p99. These are characterization values for the
executed host only. `presented_ns` is sampled after `glfwSwapBuffers` returns;
neither that value nor any benchmark percentile is physical display onset,
photodiode evidence, or a universal realtime guarantee.

This benchmark covers Center-Out, WebGrid, and Speech, not SSVEP. Its dated
Windows result is retained in {doc}`benchmarks`.

## Explicit non-goals

Presentation does not introduce scientific plotting, an offline plotting package, a live
signal viewer, browser/UI semantics, a generic renderer framework, a generic
window abstraction, a scene graph, a GUI plugin system, or a replacement task
runtime. Scientific visualization is not shipped: there
is no visualization package or extra, and presentation does not make one a core
dependency.
