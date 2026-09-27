# Adding a pipeline adapter

A native kernel is not reachable from the streaming runtime until an adapter in
`cpp/src/pipeline/` presents it as a
{cpp:class}`neurale::streaming::NativeFrameProcessor`. This page is the checklist
for that step: what to touch, which existing adapter to copy, and what has to
pass before the work is done.

It does not describe the runtime itself. For the data-plane contract the adapter
must honour, read
[Native streaming runtime design](native_streaming.md) first -- in particular the
rule that the real-time thread must not allocate, grow a container, call Python,
do I/O, take an unbounded lock, or throw for an ordinary result.

## Pick the archetype first

Most of the cost of a new adapter is decided by this question, so answer it
before writing anything.

| Your kernel | Archetype | Copy from | Shared infrastructure |
| --- | --- | --- | --- |
| Rewrites samples in place, same schema out, one frame in and the same frame forwarded | **Sampled in-place** | `sos_filter_adapter.cpp` | `adapter_support::sampled_inplace_input`, `forwarding_contract`, `inplace_adapter_stream.h` |
| Consumes a sliding window and emits one observation per hop | **Windowed feature** | `lmp_feature_adapter.cpp` | `adapter_support::feature_output`, `feature_adapter_stream.h` |
| Maps a fitted feature set to a decoded state | **Decoder** | `linear_decoder_adapter.cpp` | `fitted_feature_contract.h` |
| None of these | **Dedicated adapter** | the nearest neighbour, for style only | none yet |

If it is none of these, write a dedicated adapter and stop there. Do **not**
introduce a fourth generic archetype for a single implementation: the resampler,
the spike detector and the Kalman decoder each stayed dedicated for exactly this
reason. Abstract only once a second real implementation has shown which
semantics are actually shared.

## The nine places to touch

1. **`cpp/src/pipeline/<name>_adapter.h`** -- the class, deriving from
   `NativeFrameProcessor`, with a pimpl `struct Impl`. **This header must not
   include the kernel's private header.** `fir_filter_adapter.h` includes only
   `<neurale/streaming/processor.h>`; the kernel type appears in the `.cpp`
   alone. That is what keeps `cpp/src/signal/...` out of every consumer's
   translation unit, and it is why the adapters are classes rather than aliases
   for a template.
2. **`cpp/src/pipeline/<name>_adapter.cpp`** -- config validation in the
   constructor, geometry resolution in `prepare()`, the kernel call on the data
   plane. For an in-place adapter this is around sixty lines; see below.
3. **`cpp/src/pipeline/CMakeLists.txt`** -- one line in the `neurale_pipeline`
   source list.
4. **`cpp/tests/pipeline/<name>_adapter_test.cpp`** -- the numerical reference
   and the algorithm-specific refusals. Include the archetype's test-support
   header (`inplace_adapter_test_support.h` or `feature_adapter_test_support.h`)
   rather than re-deriving the schema, prepare context, frame filling,
   near-equality and terminal sink.
5. **`cpp/tests/CMakeLists.txt`** -- a `neurale_add_test(...)` block. In-place
   and feature adapters pass `INCLUDE_DIRS ${_neurale_pipeline_include_dirs}`.
6. **`cpp/tests/pipeline/strict_realtime_contract_test.cpp`** -- if the adapter
   belongs on one of the matrix axes, add one `StageOption` row to
   `reference_options`, `filter_options` or `feature_options`. That row is the
   whole change: the name, the factory, the case count and the failure message
   all follow from it. The binary prints the matrix size on success, so check
   the printed count grew as you expected.
7. **`cpp/benchmarks/pipeline/`** -- only if the adapter needs a dedicated
   latency study. **Do not add an axis to the physical-chain benchmark**; new
   stages go into the existing fourteen chains, and a new axis belongs in the
   contract test instead.
8. **Public control plane** -- if the adapter is a supported built-in stage,
   add only its private constructor/config binding in
   `cpp/bindings/pipeline_bindings.cpp`, its immutable specification and fixed
   document identifier in `neurale.pipeline`, and round-trip/compile tests.
   Do not publish the adapter class or add a plugin registry.
9. **Docs** -- [Native streaming runtime design](native_streaming.md) for a new
   contract or invariant, and
   [Native streaming benchmarks](native_streaming_benchmarks.md) if you produced
   numbers.

## What a sampled in-place adapter actually contains

Everything an in-place adapter shares with the other three is already written.
What remains is the algorithm:

```cpp
constexpr adapter_support::Checked kChecked{"notch"};

streaming::PreparedProcessorContract
NotchFilterAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto& signal_schema =
        adapter_support::sampled_inplace_input(kChecked, context, impl_->prepared_schema.get());
    // ... build the kernel, record impl_->validator / prepared_schema / n_channels ...
    return adapter_support::forwarding_contract(context.input_schema, workspace_bytes);
}

streaming::StreamStatus NotchFilterAdapter::process(streaming::FrameBorrow& frame,
                                                    streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_inplace_frame(
        *impl_, impl_->processor != nullptr, frame, emitter,
        [this](std::span<double> samples, std::size_t n_samples)
        { impl_->processor->process(samples, n_samples); });
}
```

`struct Impl` inherits `adapter_support::InplaceStreamState` so that the
validator, the accepted schema and the channel count are shared fields rather
than three more copies.

Two things stay yours:

* **The workspace bound.** `sampled_inplace_input` deliberately does not compute
  it -- what a workspace means differs per algorithm, and the FIR, IIR, SOS and
  common-reference adapters each derive a different number. Compute it in the
  adapter and pass it to `forwarding_contract`. Note that SOS and common
  referencing compute it *after* constructing the kernel, because the bound
  depends on what the kernel resolved; that ordering is allowed precisely
  because the bound is a parameter rather than a callback.
* **Refusals only you make.** SOS bounds the channel count at `INT_MAX` because
  only its MKL path takes an `int`; common referencing rejects a zero-channel
  signal and an out-of-range reference index. Keep those in your `prepare()`,
  after the shared call, with a comment saying why they are not shared.

## Three traps

* **A `noexcept` adapter must guard a kernel that can throw.** Neither
  `IirRealtimeProcessor::process` nor `SosRealtimeProcessor::process` is
  `noexcept`. `process_inplace_frame` wraps the invocable for you and reports
  `processor_failure`; if you write a dedicated adapter instead, do the same. An
  exception escaping `process()` terminates the process rather than failing the
  frame.
* **Pin MKL's local thread count.** Two OpenMP runtimes coexist in this process
  (`neurale_models` links libgomp, static MKL brings libiomp5), and a
  multi-threaded MKL kernel can return *different results for the same input*.
  `mkl::LocalThreadLimit` in `cpp/src/mkl_utils.h` is a correctness requirement,
  not only a latency one. It is also what makes a steady-state zero-allocation
  claim reproducible.
* **A per-sample loop may be load-bearing.** `FirFilterAdapter` drives its kernel
  one sample at a time on purpose:
  `FirRealtimeProcessor::process()` switches to a block path once
  `should_process_block(n_samples)` holds, and that path calls `resize()` on
  three workspaces *inside* `process()` -- a heap allocation on the real-time
  thread. Do not "optimise" it into a block call without first giving the kernel
  a prepare-time bounded workspace. The same shape can hide in any kernel whose
  fast path allocates.

## What has to pass

Configure a build per the repository's CMake instructions, then:

```bash
cmake --build build -j"$(nproc)" --target neurale_pipeline_<owner>_<name>_test
./build/cpp/tests/neurale_pipeline_<owner>_<name>_test

# Chain contract, fault injection and watchdog across the whole matrix.
cmake --build build -j"$(nproc)" --target neurale_pipeline_strict_realtime_contract_test
./build/cpp/tests/neurale_pipeline_strict_realtime_contract_test --contract-only

# Steady-state allocation gate. Any tracked allocation fails the run.
./build/cpp/tests/neurale_pipeline_strict_realtime_contract_test --allocation-only
```

The allocation gate degrades to `operator new` tracking (CTest label
`allocation-operator-new`) whenever a sanitizer is enabled. **Full
`--wrap=malloc` evidence (label `allocation-full`) requires a build directory
with no sanitizers**, so run the gate there before claiming zero allocations.

Do not claim a real-time or performance improvement before running the
physical-chain benchmark; see
[Native streaming benchmarks](native_streaming_benchmarks.md) for its geometry
and its flags, which differ from the contract test's.
