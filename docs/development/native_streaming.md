# Native streaming runtime design

## Status and scope

This document defines the architecture and realtime contracts for PyNeurale's
native streaming runtime. The current implementation includes fixed frame
metadata, validated immutable schemas, bounded pools and SPSC rings,
allocation-free continuity checking, a bounded critical actuator edge, and
independent bounded observer edges owned by `NativeStreamRunner`, with a
minimal pybind11 control facade and platform realtime configuration. Vendor
device access belongs to `neurale.devices`, not the streaming core.

Python exposes one public `StreamRunner`, backed exclusively by the native C++
engine. The C++ `NativeStreamRunner` name is an internal implementation detail,
not a second public Python runner.

## Control plane and data plane

Python owns the control plane. Before realtime startup it will:

- validate and normalize user configuration;
- create native sources, processors, actuators, and observers;
- establish immutable schemas and clock domains;
- allocate every buffer pool and bounded queue;
- prepare processor state and device resources;
- start, stop, and inspect the native runtime;
- read snapshots of statistics and fixed fault records.

Native C++ owns the realtime data plane. After startup it moves preallocated
frame leases through a fixed topology without entering Python or acquiring the
GIL. No arbitrary Python processor or callback may be installed on this path.

```text
native source
    -> exclusive pool lease
    -> ordered native processors
    -> critical SPSC -> safety gate -> native actuator
                                  -> observer dispatcher
                                     -> independent observer edge/worker
    -> release lease
```

Configuration, allocation, schema changes, and component replacement require
the runtime to be stopped. The first implementation will not support hot
reconfiguration.

`prepare()` creates every pool, queue, ingress timestamp slot, continuity
record, and fault slot. It also places the native safety boundary in its
default inhibited state. `arm()` is the only operation that releases that
inhibit and is valid only in `Prepared`. Starting an inhibited runtime fails.
Neither a fault, stop, reset, nor a new generation releases safety
automatically. The public facade does not arm implicitly: callers must invoke
`prepare()`, explicitly verify readiness, invoke `arm()`, and then start. The
`realtime` profile additionally requires an explicit native safety controller
and strict platform configuration.

## Component boundaries

### Source

`NativeFrameSource` receives a caller-owned writable frame backed by a
preallocated pool slot. It writes samples and fixed metadata, then returns a
`StreamStatus`: `would_block` means a transient source stall, while
`end_of_stream` is terminal for the current session. Its `read()`, `cancel()`,
and `reset()` callbacks are `noexcept`; they do not allocate or retain a frame.
`cancel()` is idempotent, may run concurrently with `read()`, and must make a
blocked device read return promptly. A driver that can block but cannot be
cancelled is not a valid native source.

### Processor

`NativeFrameProcessor` receives exclusive writable access through a move-only
`FrameBorrow` and a callback-scoped `FrameEmitter`. The emitter may explicitly
transfer the input without copying or lend one emitter-owned, pool-backed
writable output borrow at a time. Processors never receive a `FrameLease`;
lease ownership remains inside the emitter or chain. The runner's
`TerminalFrameEmitter` enforces
separate process and flush output limits, publishes to the critical edge, and
reclaims every unpublished lease when the callback returns. Zero, one, and
bounded multiple outputs therefore share the same ownership path. Stateful
processors receive ordered discontinuities and provide bounded `flush()` and
`reset()` callbacks. Arbitrary Python callables and Python-owned mutable state
are not valid processors.

Each `FrameBorrow` captures a generation from emitter-owned borrow state.
Publishing the corresponding frame or returning from the callback invalidates
that state. A stale handle has `valid() == false` and `get() == nullptr`; its
direct frame accessors terminate rather than dereference a recycled pool slot.
`FrameBorrowScope` establishes the same invalidation boundary for direct native
processor calls outside the runner. A processor may move a handle, but moving
does not extend its callback lifetime.

The enforceable handle rule does not make arbitrary C++ aliases revocable.
Processors must not retain the emitter, a `FrameBorrow`, a `FrameView`, a
derived `std::span`, or a payload pointer after the callback. Generation checks
reliably reject stale handles, while a raw pointer or span deliberately copied
out of a valid borrow remains a contract violation that requires code review
and sanitizer testing; accessing such an alias after callback return may be
undefined behavior. Current pipeline adapters keep only prepared algorithm
state and owned workspace, not frame, emitter, view, span, or payload aliases.

Before those callbacks can run, `prepare()` requires the processor to return an
owning `PreparedProcessorContract`. It declares the accepted input schema,
output schema, maximum outputs per input, maximum flush outputs, input
forwarding capability, and required workspace/frame-pool bounds. The runner
checks the declaration against its input and configured capacities, retains it
across reset, and rejects a changed declaration on the next prepare. The
terminal emitter validates every published frame against the declared output
schema and enforces the declared process, flush, and forwarding bounds. These
violations are reported through the normal processor/output fault channel.

`LinearProcessorChain` composes a non-empty fixed list of native processors on
the processing thread. Publication from a stage immediately invokes the next
stage depth-first; only the last stage reaches the terminal emitter. Prepare
validates every adjacent schema, process fan-out, flush fan-out, arithmetic
overflow, forwarding rule, and resource bound, then allocates the bounded
shared intermediate pool and validators. Flush runs source-to-sink with each
stage's flush output processed by all downstream stages before the next flush;
discontinuity is an ordered barrier. Branches, merges, cycles, mutation,
per-stage threads, and a Python chain API are not provided.

### Compatibility consumer

`NativeFrameConsumer` remains as a compatibility adapter for the original
native facade. It adapts `consume(FrameView)` to `NativeActuator`; new native
code should implement `NativeActuator` directly. It is unrelated to the
Python `StreamRunner` sink contract.

### Critical actuator

A critical actuator is reached through its own bounded SPSC edge. It receives
an `ActuatorCommand` containing command id, frame sequence, generation time,
valid-until time, session id, runtime generation, and a read-only frame
payload. `NativeActuator::submit()` rejects an expired command before entering
the device-specific `write()` implementation. Queue overflow, expiry, write
failure, and flush failure are critical faults and inhibit safety.

### Observer

An observer is noncritical telemetry, recording, visualization, or monitoring.
After actuator acceptance, the dispatcher copies the frame into each active
edge's preallocated pool and performs a nonblocking enqueue. Every observer
has its own worker, queue, drop policy, drop history, and counters. Supported
policies are `drop_oldest`, `latest_value`, and `drop_newest`. A blocked or
detached observer cannot block processing or the actuator. An observer becomes
safety-critical only when explicitly registered as a critical recorder.

A critical observer implements the format-independent
`NativeCriticalObserver` lifecycle. Registration is explicit and accepts only
`ObserverDropPolicy::fault`; a generic or lossy observer cannot be upgraded by
setting a flag. `arm()` requires readiness, `start()` starts acceptance, and
the runtime polls bounded health. Successful insertion into the shared
observer-dispatch edge fixes one `RuntimeAcceptance` -- a global ordinal across
frames and discontinuities plus the host acceptance time -- before the
critical callback copies the item into storage it owns.

On stop, abort, or fault the dispatcher continues offering every already
accepted item to critical observers while noncritical observers retain their
best-effort abort behavior. It then delivers the primary fault at most once,
delivers one fixed terminal notice, and invokes bounded drain. A shutdown
deadline calls the observer's native cancellation boundary and returns
`deadline_exceeded`; it never waits indefinitely. A failure of fault or
terminal handling is recorded without publishing back into the same path.
The acceptance suite is
`cpp/tests/streaming_recording_lifecycle_characterization_test.cpp`; the native
recorder integration is `cpp/tests/recording_runtime_integration_test.cpp`.

## Frame and clock model

Realtime frames contain fixed metadata and spans into a pool slot. They do not
own dynamic containers. A frame header carries:

- session and immutable schema identifiers;
- monotonically increasing frame sequence;
- host receive time in monotonic nanoseconds;
- optional source tick and its fixed clock domain;
- an optional host-clock `valid_until` value for deadline-sensitive frames;
- the number of signal blocks in the frame.

Each signal block has its own signal identifier, absolute sample index, device
tick, optional feature observation start time, sample count, payload offset,
payload byte count, and fixed clock-sync
snapshot. The snapshot identifies the clock domain and calibration generation,
maps one device tick to one host-monotonic time, carries the exact rational tick
rate, and records synchronization uncertainty. Blocks describe one
contiguous, ordered partition of the used payload with no gaps or overlap. This
is the native multi-rate
contract: sample position is never inferred from a frame-level index. The
signal schema fixes dtype, channel count, rational sample rate, clock domain,
layout, nominal and maximum block samples, checked maximum block bytes, physical
unit, and stable channel-set, calibration, and reference identifiers before the
session starts. The identifiers refer to immutable control-plane registries;
channel names or mutable calibration maps are never embedded in realtime
headers. `StreamSchema` validates and owns a fixed copy of
the signal descriptors, so caller storage need not outlive the schema. Runtime
frame views still borrow bounded spans from their owning pool lease.

Feature signals add a stable feature-set identifier and regular-observation
timing to the fixed signal schema. Their payload is always
``(n_observations, n_features)`` in observation-major order, and the block
observation timestamp denotes the center of the first window. Immutable
session registries own feature names, stable unit identifiers and symbols,
source stream identity, window length, shift, algorithm identity, and timestamp
reference. Python control-plane helpers convert compatible observer frames to
and from ``neurale.data.FeatureMatrix``. Algorithm adapters remain outside the
streaming core. The internal ``neurale_pipeline`` target currently contains
private LMP, Hilbert-envelope, and multitaper-bandpower feature adapters and
two private decoder adapters: a Kalman decoder and a linear decoder. No other
feature or decoder adapters are part of the current target. See the {doc}`architecture overview
<../architecture/overview>` for the dependency boundary.

The public `neurale.pipeline` control plane selects and configures these fixed
built-ins using versioned immutable plans. Compilation returns one native
processor chain accepted by `StreamRunner`; it does not publish the private
adapter classes or create a second runtime implementation.


The internal linear decoder adapter decodes the same kind of feature stream with
an affine model, ``Y = X * coef^T + intercept``. It shares the fitted feature
contract, the feature selection, and the fitted scaler vocabulary with the
Kalman adapter, and differs in three ways that follow from the model.

Prediction is stateless, so ``reset``, a discontinuity, and a flush have nothing
to restore, discard, or emit; the same observations decode to the same values
whatever order the blocks arrive in and however the stream is chunked. A
non-finite value in a selected column is a fault rather than a missing
measurement: an affine predictor has no recursion to carry a state forward
through an absent row, so there is no value to put in its place. And the
arithmetic is ``LinearModel::predict``, called once for the whole block rather
than once per observation. Under MKL that reaches ``cblas_dgemm`` with the local
thread count pinned to one. Pinning is what makes the call admissible here: an
unpinned BLAS call enters MKL's shared pool and synchronizes it on every
invocation, and at a decoder-sized matrix the measured tail of that is worse
than the pinned kernel's entire runtime. It is also a reproducibility
requirement, because that target links libgomp while static MKL brings libiomp5
and a threaded kernel with two OpenMP runtimes can answer differently for the
same input. The Kalman adapter keeps its recursion scalar instead, for the
opposite reason: after the change described next its matrices are all
state-sized, small enough that a vendor kernel would be called for its overhead
rather than its throughput.

The prepared Kalman recursion runs the information form rather than the
covariance form. The covariance form solves an ``m x m`` system for the gain on
every step, where ``m`` is the number of selected neural features, which makes a
step cubic in the observation dimension: at 512 features one step measured
8.1 ms, past a 10 ms end-to-end budget on its own. Because ``R + jitter I`` and
``H`` are both fixed when the filter is prepared, ``A = (R + jitter I)^-1 H``,
``M = H^T A`` and ``G = A^T R A`` are computed once at construction -- that is
where the single remaining ``m x m`` factorization lives, and where allocating
and throwing are both allowed. Every step is then two ``O(mk)`` passes over the
observation and a handful of ``k x k`` operations; the gain itself is never
formed. The same 512-feature step measures 1.9 microseconds.

That form asks more of the model than the covariance form did. It needs
``R + jitter I`` to be invertible on its own, where the covariance form only
needed ``H P H^T + R + jitter I`` to be. A model with no observation noise and
no jitter is therefore refused when the filter is prepared, rather than failing
on the first frame; supplying an explicit jitter is what the contract offers for
that model. The offline ``LinearGaussianModel`` keeps the covariance form, both
because it is the path the Python decoder runs and because it is what the
real-time recursion is checked against.

Temporal context is deliberately not part of the adapter. Stacking neighbouring
observations rearranges the design matrix upstream; a fit that used it presents
its stacked columns as ordinary features, and the selection names them like any
others. That is what keeps the stage stateless.

Sparse spike output uses a dedicated fixed-byte schema kind rather than the
single-event schema. The internal pipeline detector publishes at most one
fixed-capacity block for each input frame, and each block may contain multiple
valid spikes plus bounded overflow metadata. Its waveform tail, refractory,
and alignment state are prepared in advance and cleared at discontinuity,
flush, or reset. Decoding a block into an offline ``SpikeWaveformBatch`` is an
allocating control-plane operation, not part of the native callback.
For a non-empty sparse block, ``SignalBlockHeader.sample_index_start`` and
``last_sample_index`` are the inclusive first and last valid event indices;
empty blocks keep ``last_sample_index`` zero. The continuity checker compares
the next non-empty block's first event with the previous block's last event,
allowing equality for multiple spikes aligned to the same sample. The sorting
block writer independently enforces the complete peak/group/channel/crossing
order within each valid prefix.

The decoder adapters are the pipeline stages whose input is a feature stream
rather than a sampled one, and they share the fitted-contract checking described
next; only the model differs. The Kalman adapter came first. It accepts exactly one regular
``SignalKind::feature`` signal and checks the full fitted feature-set contract,
not only the width or the feature-set id. A feature-set id is session-local and
a different session may reuse the same number for a different feature set, so
the adapter carries the content of the contract the decoder was fitted under --
complete feature names and order, canonical unit symbols, observation rate,
window length, shift, source stream, algorithm name and version, and timestamp
reference -- and compares every field against the input descriptor at prepare.
This is the same check the Python ``FeatureSchema`` enforces, applied to every
column and not only to the selected ones: a native adapter that was weaker than
the offline decoder would accept inputs the Python one refuses, and an
observation-rate or window-length mismatch would silently change the dynamics
the transition matrix describes. On top of the full contract, each selected
column must carry the feature name the decoder was fitted on, because a
selection of the right length pointing at other columns feeds the model
different neural features under the names the decoder goes on reporting, and
every matrix still multiplies. Its output is one fixed
sampled signal carrying the decoded target state -- one channel per state
dimension, on the input's clock domain and observation rate, with the
configured unit and channel set.

Every observation in an input block is decoded and the block leaves as at most
one output frame, which keeps the input block's observation index and
observation time, so the output timeline is the input timeline and decoding a
stream in chunks gives what decoding it whole gives. The first observation
after prepare, reset, or a discontinuity is corrected where the state stands,
because the fitted initial state estimates the state at the start of a segment
rather than the step before it; every later observation costs a transition and
a correction. A discontinuity therefore ends the segment instead of continuing
a trajectory across data that is not there.

Model matrices, scaler statistics, selection, initial state and covariance, and
every workspace are fixed during ``prepare``; artifact parsing and Python object
conversion happen wherever the configuration is built, never on the data plane.
The recursion is ``models::PreparedKalmanFilter``, which is the offline
``LinearGaussianModel`` arithmetic with its temporaries hoisted into the object
-- the adapter holds no Kalman arithmetic of its own, and both paths compute the
same bits because the recursion's matrix products and its Cholesky are scalar on
every build. A BLAS call on the real-time thread would enter the shared thread
pool and synchronize it on every step, which is the same tail-latency pathology
the small-FFT dispatch avoids.

Failures are return values on the normal fault path: an input whose schema,
timing, or payload shape does not match is ``invalid_frame``, and so is a
partially missing observation or an infinity, since only an entirely ``nan``
observation may mean "absent" and only when the configured missing policy says
so; an innovation covariance that cannot be factorized is ``processor_failure``;
a decoded block that will not fit the output frame is ``output_limit``.

A frame may contain any nonempty subset of the schema's signals. Continuity is
committed only for blocks actually present, allowing sparse trigger/event
signals and independent multi-rate packetization without fabricating samples.
Reappearance is validated against that signal's last sample and device-tick
anchor; absence alone is not a gap.

The realtime path does not store a floating-point timestamp per sample. Sample
time is derived off the hot path from the schema's rational sample rate,
absolute sample index, device tick, and clock-domain calibration. Clock reset or
loss of continuity is represented by a fixed discontinuity record, not by
rewriting previous timestamps.

## Buffer ownership

All payload and signal-block-header storage is allocated before startup by a
fixed-capacity `FramePool`. `MutableFrame` exposes writable storage but keeps
the backing spans private; `set_used_sizes()` selects the published prefix and
cannot rebind storage away from the pool. `BufferToken` identifies a slot and 63-bit
generation; move-only `FrameLease` represents exclusive writable ownership.
The intended ownership sequence is:

1. The runtime acquires one lease without blocking.
2. The source fills the caller-owned lease.
3. The current runner's single processor receives a generation-checked,
   callback-scoped mutable borrow; the runner retains the lease.
4. The critical edge owns the lease until the actuator command returns.
5. The actuator assigns runtime-acceptance metadata and transfers successful
   output to the bounded dispatcher.
6. The dispatcher offers the callback-scoped borrow to each critical observer,
   then copies into each noncritical observer's independent preallocated pool.
7. Destruction of each owning message returns its lease exactly once.

`StreamMessage` owns the lease while a frame is queued, so copying a fixed
header and spans never releases or copies the underlying payload. A temporary
`FrameView` is valid only while that owning message or lease remains alive.
`FramePool` must outlive all of its leases and every queue that may contain a
frame message. Destruction with an outstanding lease terminates immediately,
and `outstanding()` exposes the ownership count for shutdown diagnostics. No
callback may retain a borrow, view, span, or payload pointer beyond that
lifetime. Handle invalidation is checked, but independently retained raw aliases
cannot be revoked by C++. Pool
exhaustion is an explicit `buffer_exhausted` result. The runtime never grows a
pool, creates a fallback buffer, or transfers ownership into a Python object on
the realtime thread.

Generation and leased state share one lock-free atomic word. Acquire changes a
free generation to leased without changing its generation; successful release
atomically advances the generation. A stale or duplicate release cannot match
the current state. The final representable generation is never wrapped back to
zero, so exhaustion sacrifices that slot instead of permitting ABA.

## Bounded queues

The acquisition and processing threads communicate through one SPSC ingress
ring. Processing publishes to a separate critical SPSC ring consumed by the
actuator worker. Successful commands enter one noncritical dispatcher ring;
the dispatcher fans out to independent bounded observer queues. Queue
operations on the realtime path are
non-allocating and have a finite outcome:

- success;
- would block;
- queue overflow;
- stopped.

There is no unbounded queue and no implicit temporary expansion. Critical-edge
exhaustion is a runtime fault. Observer-edge exhaustion is recorded as an
observer overrun and follows an explicit drop policy without delaying the
critical path. Capacities are reported in runtime configuration and statistics
so memory and latency bounds can be audited before start.

The main frame pool is derived from the configured topology rather than a
hard-coded slot count:

```text
required pool capacity =
    source-owned
  + ingress capacity
  + processor-owned
  + critical edge capacity
  + actuator-owned
  + observer edge capacity
  + reserve
```

Each term is the maximum number of main-pool leases that the corresponding
stage can retain concurrently. In the current topology `observer edge
capacity` is the dispatcher ring capacity. Each observer additionally owns a
separate pool sized from its `ObserverEdgeConfig.capacity`. `reserve` is an
explicit control-plane choice for shutdown and ownership handoff margin, not an
implementation constant. Startup uses checked addition and rejects a capacity
that cannot be represented by `size_t`.

`SpscRing<T>` uses all `N` slots reported by `capacity()`; it does not reserve a
sentinel slot. The producer owns the write sequence and the consumer owns the
read sequence. Both are unsigned monotonic counters, so slot selection wraps by
`sequence % N` while bounded unsigned subtraction remains valid across counter
wraparound. The two counters occupy separate 64-byte cache lines.

The producer acquires the consumer sequence before reusing a slot and releases
its sequence after constructing a message. The consumer acquires the producer
sequence before reading and releases its sequence after destroying the message.
Loads of a thread's own sequence are relaxed. These are the only ordering edges
needed for payload ownership transfer.

`try_push()` returns `queue_overflow` when full and never drops, grows, or
blocks. `try_pop()` returns `would_block` for an open empty ring. `close()` is
the producer's final operation and must not race another producer call; the
consumer may drain all published messages and then receives `stopped`. Edge
drop, blocking, retry, and deadline policies remain outside the ring.
Construction allocates all slot storage on the control plane; destruction is
only valid after producer and consumer threads have stopped. A GCC or Clang
test build configured with `NEURALE_ENABLE_TSAN=ON` instruments the concurrent
ring stress test with ThreadSanitizer and rejects unsupported toolchains.

Observer edges use a separate bounded `ObserverQueue`. Each slot carries an
atomic sequence so the normal observer consumer and the dispatcher performing
a configured oldest-item eviction cannot own the same message concurrently.
This is not exposed as a general MPMC queue. `drop_newest` never touches queued
messages, `drop_oldest` evicts one oldest message only when capacity is needed,
and `latest_value` removes all pending values before publishing the newest.
All three policies return leases through the edge's preallocated pool.

## Continuity and ordered control messages

`ContinuityChecker` is configured from the immutable session schema before
startup and allocates fixed state and scratch records per signal at that point.
Recoverable records are copied into a preallocated `DiscontinuityPool`; the
resulting move-only lease remains owned by its `StreamMessage`. A normal check
emits one `Frame` message. A recoverable gap emits `Discontinuity` followed by
that `Frame`. Output discontinuities and frames enter the same critical edge;
the actuator worker forwards them to the observer dispatcher in causal order.
End-of-stream,
graceful shutdown, and abort are explicit `StreamMessage` alternatives rather
than out-of-band queue state.

Continuity is sample-centered and independent per signal. Frame sequence,
absolute sample index, and schema-declared sample-counter ticks are integer
domains; no floating-point time tolerance is involved. A frame sequence gap
with continuous samples reports zero missing samples. A forward sample jump
reports the exact missing count. Device-counter wrap is supported by unsigned
counter arithmetic, while a non-wrap rollback is reported as a device restart.

Every frame is fully validated before expected positions are committed.
Recoverable gaps commit the actual frame as the new anchor for all signals.
Duplicate or regressed frame sequences, session changes, sample-index rollback,
schema changes, malformed block sets, and integer overflow are fatal validation
results with no messages and no continuity-state commit. Failure to acquire a
discontinuity slot also leaves continuity state unchanged. Destroying or
overwriting the owning message releases its gap slot after the consumer has
finished reading the view.

## Realtime thread prohibitions

After a realtime thread starts, its callback path must not perform:

- heap allocation, container growth, or lazy initialization;
- Python calls, GIL acquisition, reference-count ownership transfer, or Python
  exception translation;
- disk I/O, network I/O, console output, logging, or formatting of messages;
- unbounded mutex acquisition, condition-variable waits, sleeps, or retries;
- dynamic library loading, device discovery, schema negotiation, or processor
  construction;
- per-sample floating-point timestamp creation;
- exceptions for ordinary runtime outcomes;
- implicit queue growth, fallback allocation, or hidden host/device transfer.

Platform calls required for scheduling or a native device must be prepared and
audited explicitly. Any operation not proven bounded remains outside the
realtime thread.

## Stage deadlines and watchdog

All deadline and heartbeat values use the runtime's monotonic integer
nanosecond domain. They never use wall-clock time or per-sample floating-point
timestamps. The public immutable `RealtimeConfig` carries only
`latency_budget_seconds`, `source_timeout_seconds`, and optional advanced
platform intent. During `prepare()`, the native runtime resolves its individual
source-stall, ingress-dwell, processor, output-age, actuator-command, shutdown,
and watchdog-period limits. Output age includes the prepared pipeline's output
cadence, so a valid slow feature cadence is not treated as a stale output.

Acquisition publishes its last completed `read()` heartbeat. Processing and
the actuator publish callback start and active state; successful actuator
completion publishes the last valid output time. Every ingress slot has a
separately preallocated atomic enqueue
timestamp. The watchdog reads SPSC producer/consumer sequences and those
timestamps; it never reads frame storage, becomes a queue consumer, acquires a
processor lock, or calls a processor.

The watchdog is independent of every data-plane worker. It detects blocked source reads,
processor or actuator callbacks that do not return, the oldest queued frame
exceeding its dwell limit, and stale output. Processing also checks exact dwell
on dequeue.
During stopping, both the watchdog and the control-plane bounded wait can
record one fixed shutdown-timeout fault. Clock waits are interruptible so
normal completion and abort wake the watchdog immediately.

`NativeClock` supplies `now_ns()`, interruptible deadline waiting, and wakeup.
Production uses `steady_clock`; native tests inject an atomic fake clock and
advance it deterministically without sleeping. Python cannot provide a clock
or callback implementation.

## Fault and safety-stop model

Construction and startup may throw because they run on the control plane.
Realtime callbacks are `noexcept` and return `StreamStatus`. The runtime stores
the first critical failure in a fixed-size `FaultRecord` containing status,
stage, component identifier, frame sequence, sample index, and device tick.
Human-readable formatting happens later on the control plane.

On the first source, processor, actuator, ownership, critical queue, or
critical-recorder fault,
the runtime will:

1. publish the first fault record exactly once;
2. invoke the independent native `SafetyController::inhibit()` boundary;
3. stop accepting source data;
4. release or account for every outstanding pool lease;
5. transition to a terminal failed state visible to Python.

Secondary faults increment fixed counters and may occupy a bounded diagnostic
ring; they cannot replace the primary fault. Observer overruns are noncritical
unless that edge was explicitly configured as a critical recorder. Safety
stop never waits for Python cleanup. The primary slot has an explicit
publishing state: a terminal reader that observes an in-progress first fault
waits for that fixed record to become visible instead of treating it as absent.

`SafetyController::inhibit(reason)` and `release()` are native, `noexcept`,
bounded, and allocation-free contracts. Runtime inhibition uses an atomic
latch independent of data queues. A deadline fault is recorded before inhibit,
and fixed `detected_at_ns` and `last_inhibit` values make fault-to-inhibit
latency measurable. Concurrent or cascading faults cannot invoke an effective
inhibit twice. If the controller reports failure, that failure enters bounded
fault history without replacing the triggering primary fault.

`stop()` and `abort()` request source cancellation, inhibit safety, and wait no
longer than `shutdown_deadline` for acquisition and processing completion. On
timeout they return `deadline_exceeded`, retain all worker-owned resources, and
leave the runtime stopping or failed. They never detach a thread or destroy a
pool under a live lease. The blocking native component must eventually return;
after it does, the control plane calls `join()` before reset or destruction.

## Runtime and public API

Headers under `cpp/include/neurale/streaming` expose the native contracts.
`NativeStreamRunner` owns the schema, ingress and critical rings, observer
dispatcher, per-observer edges and pools, continuity state, output port, fixed
atomic statistics, bounded fault channel, and all native workers. External
native source, processor, actuator, and observer objects must outlive it. This
class is an implementation detail at the Python boundary: the only public
runner is `neurale.streaming.StreamRunner`.

The explicit state machine is `Created -> Prepared -> Running -> Stopping ->
Stopped`, with `Failed` as the terminal fault state until `reset()`. `prepare()`
validates the schema and topology and creates every pool, ring, timestamp,
fault slot, and continuity resource. Explicit `arm()` releases startup safety
inhibition. `start()` then creates acquisition, processing, actuator,
observer-dispatch, observer, and watchdog threads; it does not create data
buffers. Every worker first parks behind a startup gate. Only after all worker
threads exist (and requested realtime configuration has completed) does the
runtime call `start_observing()` on critical recorders and then release the
workers together. If any thread creation throws, a critical recorder therefore
remains in its pre-start ready state; already-created workers are cancelled and
joined, and the recorder is closed through its bounded pre-start cancel path
before `start()` propagates the exception. Worker terminal paths do not
reacquire the lifecycle lock held by startup cleanup. `run()` is `start()`
followed by `join()`.
Only the ordered post-worker join transition publishes `Stopped` or `Failed`;
an observer fault cannot overwrite a terminal state with `Stopping` or leave a
joined runtime reporting success.
Graceful EOF and `stop()` drain accepted frames and perform bounded processor,
actuator, and observer flushes. `abort()` cancels acquisition and keeps the
existing immediate-cancel behavior for noncritical observers, while every
runtime-accepted critical item is drained before its terminal notice. A
successful `reset()` returns to `Created`, so a later session must call
`prepare()` again. A native critical recorder is single-use and must be
detached/replaced before a new session. Once the runner is terminal and all
workers are joined, detaching a critical recorder removes both its prepared
edge and its registration; `reset()` therefore does not call the detached
recorder and a replacement may reuse the observer ID after reset.

Ingress overflow never drops silently: acquisition fixes `QueueOverrun` as a
fault, requests source cancellation, and processing releases and counts every
queued lease as aborted. The source closes the ring only after its final push;
the consumer drains it before terminal state. The threads communicate only by
the ring, atomic lifecycle flags and counters, and fixed fault slots.

The first failure is retained as the primary fixed fault. Later cleanup or
callback failures enter a preallocated bounded history, with an explicit
dropped-history counter after capacity is exhausted. No diagnostic vector
grows in the data path. Fault records carry session, runtime generation,
schema, clock-domain, frame, sample, and device-tick fields where available.
Runtime statistics maintain fixed maximum ingress-dwell, processor-execution,
and source-to-actuator durations; detailed percentiles remain a benchmark or
observer responsibility outside the critical loop.

The existing `_native` extension exposes the engine privately as
`_native.streaming._NativeStreamRunner`. The public `StreamRunner` is a thin
control-plane facade: lifecycle, pools, queues, continuity, faults, statistics,
watchdog, safety, and output dispatch all remain owned by that one C++ engine.
Importing `neurale.streaming` is still lazy and does not load native, CUDA, or
device subsystems until a native object is requested. There is no public
`NativeStreamRunner`, implementation selector, or implicit fallback to a
Python runner.

`StreamRunner` has three explicit execution profiles:

- `realtime` accepts only final native source, processor, and actuator types,
  requires an explicit native safety controller and strict platform mode, and
  rejects every Python critical adapter during `prepare()`.
- `research` permits explicitly constructed `PythonSourceAdapter`,
  `PythonProcessorAdapter`, and `PythonSinkAdapter` components and makes no
  realtime guarantee.
- `offline` permits the same adapters for synchronous experiments and
  compatibility work and makes no realtime guarantee.

The adapters copy Python-owned frame snapshots at their boundary and acquire
the GIL only when invoking the wrapped Python component. They are never chosen
implicitly and cannot enter the `realtime` profile. The adapters do not create
a second queue, lifecycle, fault, or statistics implementation; their status
codes are consumed by the native runtime state machine. Schema,
configuration, stats, and fault records cross the binding as owned values
rather than borrowed internal pointers.

`run()`, `join()`, `stop()`, `abort()`, and `reset()` release the GIL for their
complete native call. `prepare()` and `start()` remain control-plane methods;
thread creation may throw at this boundary. Component lifetimes are retained
by the runner binding, while the C++ runtime itself stores only native
references. No pybind trampoline or Python callback enters any
realtime-profile worker.

### Python observer bridge

`PythonObserverBridge` is the one deliberate Python callback boundary. Its
constructor receives the observed `StreamSchema` and derives its native copy
storage from that schema. It is a
final `NativeObserver` and can only be registered on a noncritical observer
edge; registration rejects `critical_recorder=true`. Its topology is:

```text
native observer edge
  -> PythonObserverBridge::observe() native copy
  -> bounded bridge ObserverQueue
  -> dedicated non-realtime bridge thread
  -> Python-owned snapshot
  -> Python callback
```

The native observer callback does not acquire the GIL, allocate, call Python,
or transfer a runtime/observer-edge lease to Python. It copies into a separate
preallocated bridge `FramePool` or `DiscontinuityPool` and performs a
nonblocking enqueue. Bridge capacity and `drop_oldest`, `latest_value`, or
`drop_newest` policy are explicit constructor arguments. Pool exhaustion and
queue saturation increment bridge-local bounded drop statistics and return
success to the runtime observer edge, so they cannot inhibit safety or detach
unrelated observers.

Only the bridge thread acquires the GIL. It converts each native message into
`PythonObserverFrame` or `PythonObserverDiscontinuity`. Frame payload is copied
into a Python-owned one-dimensional `numpy.uint8` array. The native bridge
lease is released before the callback is invoked, so retaining the snapshot or
delaying Python GC cannot consume either the critical frame pool or the bridge
pool. Signal block offsets describe the typed regions in that byte array.

A callback exception is captured as `callback_error`, increments
`callback_errors`, closes only that bridge, and drains its remaining native
messages as drops. Runtime primary fault and safety state are unchanged.
`stats` and `drop_ranges` expose bridge-local queue pressure independently from
the runtime's per-observer edge statistics.

`close()` is idempotent, stops accepting messages, discards pending bridge
messages, releases the GIL while joining the bridge thread, and releases all
native leases before destroying either pool. Runtime EOF uses graceful bridge
flush so already accepted messages may drain. Destruction performs an aborting
close, which prevents a GIL/join inversion during ordinary interpreter
shutdown. Python cannot forcibly terminate arbitrary callback code: a callback
that never returns and never releases the GIL remains an application error;
the critical native path remains isolated, but orderly process shutdown still
requires callback cooperation.

The bridge thread is separate, non-realtime, and owned by
`PythonObserverBridge` rather than by the runtime. It remains the only Python
callback boundary permitted in the `realtime` profile.

### Realtime thread configuration

`RealtimeConfig()` is a valid public default. Frame-pool storage, ingress and
critical queues, processor leases, discontinuity storage, and observer dispatch
capacity are native implementation details derived during `prepare()` from the
input and output schemas, prepared processor resource contract, source, and
registered observers. `RealtimePlatformConfig` is an optional advanced request
with independent acquisition, processing, actuator, observer-dispatch, and
watchdog thread settings. Each `RealtimeThreadConfig` has a 64-bit CPU affinity
mask, scheduling policy, priority, optional stack-size request, and stack
prefault size. Process memory locking and pool prefaulting are configured once
for the runtime. Affinity and priority are never selected implicitly by the
library.

`realtime_platform_capabilities()` reports exactly which features the current
backend can apply. A requested but unsupported operation is not reported as
successful. The realtime execution profile always applies platform intent
strictly. Research and offline profiles apply an explicitly supplied platform
request as best effort, and otherwise leave platform configuration disabled.
Best effort retains unsupported or failed operations in
`realtime_configuration_status`; every profile keeps all workers behind the
startup gate until thread creation and critical-recorder startup commit. Strict
mode additionally inhibits safety, records a `RealtimeConfiguration` fault,
and fails startup before any worker enters its data loop.

Linux uses `pthread_setname_np`, `pthread_setaffinity_np`,
`pthread_setschedparam`, and `mlockall(MCL_CURRENT | MCL_FUTURE)`. Stack and
preallocated pool pages can be touched before processing starts. FIFO and
round-robin scheduling or memory locking commonly require elevated resource
limits or capabilities; permission-dependent deployment checks must therefore
be run on the target host. The current `std::thread` launcher cannot request a
custom native stack size, so `stack_size` is reported as unsupported rather
than silently ignored.

Windows supports thread descriptions, the process-group affinity mask,
Windows thread priority, stack prefaulting, and pool prefaulting. POSIX
scheduling policies, process-wide memory locking, and custom stack size are
reported as unsupported. Other platforms return the same explicit capability
and result structures instead of pretending that configuration succeeded.

These controls improve scheduling predictability but provide no hard realtime
guarantee. A production Linux deployment still owns its PREEMPT_RT kernel,
CPU isolation, interrupt affinity, power management, memory limits, driver
behavior, and end-to-end latency validation. The library does not modify those
system policies automatically.

The native runtime is compiled as the independent `neurale_streaming` target,
with the `neurale::streaming` CMake alias. It does not depend on
`neurale_runtime`.

## Supported scope

The public runtime supplies generic source, processor, consumer, actuator,
observer, safety, and lifecycle contracts. The repository ships a simulated
neural source and an optional LSL provider through `neurale.devices`; it does
not ship a vendor-specific physical device adapter or actuator.

Actual realtime guarantees still depend on platform scheduling, device driver
behavior, memory locking, and callback implementations. Those guarantees are
outside the library contract and must be measured before production use.

## Preloaded array replay

`neurale.streaming.ArrayReplaySource(schema, data, paced=True, session_id=1)`
owns a copy of a C-contiguous float64 `(samples, channels)` array. Its schema
must contain one sample-major sampled signal with sample-counter device ticks.
Input must contain complete nominal blocks; the rational sample rate must give
an integer block duration in nanoseconds. No conversion or file I/O occurs in
the source callback. Paced replay releases each block at its absolute block-end
deadline on the host steady clock. Delays do not shift subsequent deadlines.
`paced=False` is for bounded correctness checks and can overflow runtime queues.
Use this source with the default host-clock `StreamRunner`, not a custom clock.

`NativeResultSink(output_schema, capacity)` preallocates dense float64 result
storage. Capacity counts observations, not frames. Exhaustion returns
`OUTPUT_LIMIT`; a discontinuity faults instead of silently joining segments.
The sink timestamps delivery after copying each output frame into storage.
It does not perform a hardware action.

For a native empty-path baseline, construct
`NativeResultSink(input_schema, capacity, window_samples=256, hop_samples=40,
output_channels=5)` and use `IdentityNativeProcessor`. This mode validates
contiguous sampled input, skips payload computation and captures zeros when
each window ends. The hop must be at least the maximum input block size.
Snapshot indices count windows; source ticks identify the input block containing
the window end. This opt-in mode does not change ordinary value capture.

After `runner.run()` or `runner.join()`, `source.timings` copies rows containing
`sample_idx_start, planned_ns, ready_ns`. `sink.snapshot()` copies `(values,
timing)`, where timing columns are `observation_index, source_tick,
source_received_ns, delivered_ns`. All these timestamps use the native host
steady-clock domain; do not subtract Python clock values from them. Runtime
`source_received_ns` is sampled after source read completion; `ready_ns` is
sampled immediately after source payload copy. All rows of one output frame
share its delivery timestamp. Snapshots can read a published prefix during a
run, but reset must occur only after the runtime is quiescent. A source or sink
must not be shared by simultaneously running sessions.

Use `compile_pipeline(plan).output_schema(input_schema)` for the sink schema,
then `compiled.create_runner(..., profile="realtime",
safety_controller=RecordingNativeSafetyController())`. Python prepares, starts,
joins and exports results; acquisition, processing and capture execute in C++.
The PMTM feature supports `detrend="none" | "mean" | "linear"` and
`backend="auto" | "builtin"`. `detrend="mean"` subtracts each channel's window
mean before applying DPSS tapers. Explicit `backend="builtin"` selects the
native FFT implementation.
