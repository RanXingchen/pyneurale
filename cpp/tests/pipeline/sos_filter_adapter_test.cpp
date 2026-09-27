/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_counter.h"
#include "check_returns.h"
#include "inplace_adapter_test_support.h"
#include "realtime_utils.h"
#include "sos_filter_adapter.h"
#include "sos_realtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <new>
#include <span>
#include <vector>

#include <neurale/signal/iir.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/linear_processor_chain.h>

void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

namespace
{

using namespace neurale::streaming;

constexpr std::array<double, 12> sos{
    0.5, 0.25, 0.125, 1.0, -0.2, 0.05, 0.75, -0.1, 0.025, 1.0, -0.3, 0.1,
};
constexpr std::size_t channels = 2;
constexpr std::size_t max_samples = 16;
constexpr std::size_t max_scalars = channels * max_samples;

namespace test_support = neurale::pipeline::test_support;

// 30 kHz and a one-output/one-lease context: this test's geometry was chosen
// against the SOS kernel's own dispatch thresholds, not shared with the FIR and
// IIR tests, which run at 1 kHz with eight.
constexpr test_support::InplaceGeometry base_geometry{
    .n_channels = static_cast<std::uint32_t>(channels),
    .nominal_block_samples = 4,
    .max_block_samples = static_cast<std::uint32_t>(max_samples),
    .fs = {30'000, 1},
    .host_time_reference_ns = 4'000,
};

[[nodiscard]] StreamSchema make_schema()
{
    return test_support::make_schema(base_geometry);
}

// Built once at static-initialisation time, ahead of any allocation snapshot:
// `fill_frame` runs inside the steady-state loops that assert zero allocations,
// and constructing a StreamSchema per call would be counted against the adapter.
const StreamSchema fixed_schema = make_schema();

[[nodiscard]] ProcessorPrepareContext make_context(const StreamSchema& schema) noexcept
{
    return test_support::make_context(schema, 1, 1, 0);
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::span<const double> values,
                                      std::uint64_t sequence = 7,
                                      SampleIndex sample_start = 100) noexcept
{
    return test_support::fill_frame(frame, fixed_schema, base_geometry, values, sequence,
                                    sample_start);
}

using test_support::frame_values;
using test_support::nearly_equal;

// A pool capacity of zero: nothing downstream of an SOS adapter in this test
// acquires, so an acquire attempt is a defect and must be reported as one.
class TerminalSinkEmitter final : public test_support::TerminalSink
{
  public:
    explicit TerminalSinkEmitter(const StreamSchema& schema)
        : test_support::TerminalSink(schema, 0, max_scalars)
    {
    }
};

[[nodiscard]] int test_single_frame_and_metadata()
{
    auto schema = make_schema();
    neurale::pipeline::SosFilterAdapter adapter{sos};
    const auto contract = adapter.prepare(make_context(schema));
    CHECK(contract.accepted_input_schema.equivalent(schema));
    CHECK(contract.output_schema.equivalent(schema));
    CHECK(contract.max_process_outputs_per_input == 1);
    CHECK(contract.max_flush_outputs == 0);
    CHECK(contract.can_forward_input);

    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const std::array input{1.0, -1.0, 0.5, 2.0, -0.5, 0.25, 3.0, -2.0};
    auto expected = input;
    neurale::signal::detail::SosRealtimeProcessor reference{sos, channels,
                                                            std::span<const double>{}};
    reference.process(expected, input.size() / channels);
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    const auto original_header = lease.frame().header();
    const auto original_block = lease.frame().blocks().front();

    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.status() == StreamStatus::ok);
    CHECK(sink.publication_count() == 1);
    CHECK(nearly_equal(sink.values(), expected));
    CHECK(sink.header().session_id == original_header.session_id);
    CHECK(sink.header().sequence == original_header.sequence);
    CHECK(sink.header().host_received_ns == original_header.host_received_ns);
    CHECK(sink.header().source_tick == original_header.source_tick);
    CHECK(sink.header().valid_until_ns == original_header.valid_until_ns);
    CHECK(sink.header().schema_id == original_header.schema_id);
    CHECK(sink.header().source_clock_domain == original_header.source_clock_domain);
    CHECK(sink.header().flags == original_header.flags);
    CHECK(sink.block().sample_idx_start == original_block.sample_idx_start);
    CHECK(sink.block().device_tick_start == original_block.device_tick_start);
    CHECK(sink.block().signal_id == original_block.signal_id);
    CHECK(sink.block().n_samples == original_block.n_samples);
    CHECK(sink.block().clock_sync.clock_domain == original_block.clock_sync.clock_domain);
    CHECK(sink.block().clock_sync.generation == original_block.clock_sync.generation);
    return 0;
}

[[nodiscard]] int test_design_parameters_are_resolved_during_prepare()
{
    auto schema = make_schema();
    neurale::pipeline::SosFilterAdapter adapter{neurale::pipeline::FilterDesign{
        .type = "lowpass",
        .cutoff_hz = {300.0},
        .order = 4,
        .kind = "butterworth",
    }};
    const auto contract = adapter.prepare(make_context(schema));
    CHECK(contract.accepted_input_schema.equivalent(schema));
    CHECK(contract.output_schema.equivalent(schema));
    CHECK(contract.required_resources.workspace_bytes > 0);

    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const std::array input{1.0, 0.0, 0.0, 0.0};
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.values().size() == input.size());
    for (const auto value : sink.values())
    {
        CHECK(std::isfinite(value));
    }
    return 0;
}

[[nodiscard]] int test_line_noise_harmonics_are_designed_during_prepare()
{
    auto schema = make_schema();
    neurale::pipeline::SosFilterAdapter fundamental{
        neurale::pipeline::LineNoiseFilterDesign{50.0, 2.0, 1, 2}};
    neurale::pipeline::SosFilterAdapter harmonics{
        neurale::pipeline::LineNoiseFilterDesign{50.0, 2.0, 3, 2}};
    const auto fundamental_contract = fundamental.prepare(make_context(schema));
    const auto harmonic_contract = harmonics.prepare(make_context(schema));
    CHECK(harmonic_contract.required_resources.workspace_bytes >
          fundamental_contract.required_resources.workspace_bytes);

    std::vector<double> expected_sos;
    for (const auto center : {50.0, 100.0, 150.0})
    {
        const std::array cutoffs{center - 1.0, center + 1.0};
        auto sections =
            neurale::signal::iir_sos("butterworth", 2, cutoffs, "bandstop", 30'000.0, 0.0, 0.0);
        expected_sos.insert(expected_sos.end(), sections.begin(), sections.end());
    }
    const std::array input{1.0, 0.5, -0.25, 2.0, -1.0, 0.75, 0.0, -0.5};
    auto expected = input;
    neurale::signal::detail::SosRealtimeProcessor reference{expected_sos, channels,
                                                            std::span<const double>{}};
    reference.process(expected, input.size() / channels);

    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(harmonics.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));

    constexpr test_support::InplaceGeometry low_rate_geometry{
        .n_channels = static_cast<std::uint32_t>(channels),
        .nominal_block_samples = 4,
        .max_block_samples = static_cast<std::uint32_t>(max_samples),
        .fs = {202, 1},
    };
    auto low_rate_schema = test_support::make_schema(low_rate_geometry);
    neurale::pipeline::SosFilterAdapter low_rate_fundamental{
        neurale::pipeline::LineNoiseFilterDesign{50.0, 2.0, 1, 2}};
    neurale::pipeline::SosFilterAdapter low_rate_harmonics{
        neurale::pipeline::LineNoiseFilterDesign{50.0, 2.0, 3, 2}};
    CHECK(low_rate_harmonics.prepare(make_context(low_rate_schema))
              .required_resources.workspace_bytes ==
          low_rate_fundamental.prepare(make_context(low_rate_schema))
              .required_resources.workspace_bytes);
    return 0;
}

template <std::size_t ChunkCount>
[[nodiscard]] int run_chunks(std::span<const double> input,
                             const std::array<std::size_t, ChunkCount>& chunk_samples,
                             std::span<double> output)
{
    auto schema = make_schema();
    neurale::pipeline::SosFilterAdapter adapter{sos};
    const auto contract = adapter.prepare(make_context(schema));
    static_cast<void>(contract);
    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    std::size_t scalar_offset = 0;
    SampleIndex sample_offset = 0;
    for (const auto n_samples : chunk_samples)
    {
        const auto n_scalars = n_samples * channels;
        CHECK(n_samples <= max_samples);
        CHECK(fill_frame(lease.frame(), input.subspan(scalar_offset, n_scalars), sample_offset + 1,
                         sample_offset) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        std::copy(sink.values().begin(), sink.values().end(),
                  output.begin() + static_cast<std::ptrdiff_t>(scalar_offset));
        scalar_offset += n_scalars;
        sample_offset += n_samples;
    }
    CHECK(scalar_offset == input.size());
    return 0;
}

[[nodiscard]] int test_variable_chunks_and_chunk_invariance()
{
    std::array<double, 26> input{};
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] = std::sin(static_cast<double>(i) * 0.21) + static_cast<double>(i % channels);
    }
    auto expected = input;
    neurale::signal::detail::SosRealtimeProcessor reference{sos, channels,
                                                            std::span<const double>{}};
    reference.process(expected, input.size() / channels);

    std::array<double, input.size()> first{};
    std::array<double, input.size()> second{};
    CHECK(run_chunks(input, std::array<std::size_t, 4>{1, 4, 2, 6}, first) == 0);
    CHECK(run_chunks(input, std::array<std::size_t, 3>{7, 1, 5}, second) == 0);
    CHECK(nearly_equal(first, expected));
    CHECK(nearly_equal(second, expected));
    CHECK(nearly_equal(first, second));
    return 0;
}

[[nodiscard]] int test_reset_discontinuity_and_empty_frame()
{
    auto schema = make_schema();
    const std::array warmup{2.0, -1.0, 1.0, 0.5};
    const std::array probe{0.25, 0.75, -0.5, 1.5, 2.0, -2.0};
    auto expected = probe;
    neurale::signal::detail::SosRealtimeProcessor reference{sos, channels,
                                                            std::span<const double>{}};
    reference.process(expected, probe.size() / channels);

    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    neurale::pipeline::SosFilterAdapter adapter{sos};
    const auto contract = adapter.prepare(make_context(schema));
    static_cast<void>(contract);

    CHECK(fill_frame(lease.frame(), warmup) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), probe) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));

    CHECK(fill_frame(lease.frame(), warmup) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    const Discontinuity discontinuity{
        .session_id = 5,
        .previous_frame_sequence = 10,
        .actual_frame_sequence = 12,
        .reason = GapReason::source_gap,
    };
    CHECK(adapter.handle_discontinuity(discontinuity) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), probe) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));

    CHECK(adapter.reset() == StreamStatus::ok);
    lease.frame().header() = FrameHeader{.schema_id = 29};
    CHECK(lease.frame().set_used_sizes(0, 0) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);
    CHECK(sink.publication_count() == 0);
    CHECK(fill_frame(lease.frame(), probe) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));
    return 0;
}

[[nodiscard]] int test_steady_state_allocation()
{
    auto schema = make_schema();
    neurale::pipeline::SosFilterAdapter adapter{sos};
    const auto contract = adapter.prepare(make_context(schema));
    static_cast<void>(contract);
    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const std::array input{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);

    const auto before = allocations.load(std::memory_order_relaxed);
    for (std::size_t iteration = 0; iteration < 128; ++iteration)
    {
        CHECK(fill_frame(lease.frame(), input, iteration + 1, iteration * 3) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    return 0;
}

[[nodiscard]] int test_empty_state_at_mkl_threshold()
{
    // Regression guard for the null-dereference crash in
    // SosRealtimeProcessor::copy_public_state. When the processor is constructed
    // with an empty initial-state span and the MKL BLAS df2t kernel is selected
    // (n_channels >= kMklRealtimeMinChannels), the MKL branch read a fixed
    // 2 * n_channels doubles per section from state.data() -- which is null for
    // an empty span -- and crashed on every MKL build at 128+ channels. The
    // adapter always supplies an empty initial state (sos_filter_adapter.cpp),
    // so this path is hit on the first frame of every high-channel-count stream.
    constexpr std::size_t threshold = neurale::signal::detail::kMklRealtimeMinChannels;
    neurale::signal::detail::SosRealtimeProcessor processor{sos, threshold,
                                                            std::span<const double>{}};

    // Construction must survive. The quiescent initial state must read back as
    // all zeros regardless of the active kernel (builtin or MKL BLAS df2t).
    std::vector<double> snapshot(processor.n_sections() * 2 * threshold, 0.5);
    processor.state(snapshot);
    for (const auto value : snapshot)
    {
        CHECK(value == 0.0);
    }

    // A steady frame must process without crashing and produce finite output.
    constexpr std::size_t samples = 4;
    std::vector<double> frame(samples * threshold);
    for (std::size_t i = 0; i < frame.size(); ++i)
    {
        frame[i] = std::sin(static_cast<double>(i) * 0.11);
    }
    processor.process(frame, samples);
    for (const auto value : frame)
    {
        CHECK(std::isfinite(value));
    }
    return 0;
}

[[nodiscard]] int test_linear_chain_terminal_sink()
{
    auto schema = make_schema();
    neurale::pipeline::SosFilterAdapter adapter{sos};
    std::array<NativeFrameProcessor*, 1> stages{&adapter};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(make_context(schema));
    CHECK(contract.accepted_input_schema.equivalent(schema));
    CHECK(contract.output_schema.equivalent(schema));

    const std::array input{1.0, -2.0, 3.0, -4.0};
    auto expected = input;
    neurale::signal::detail::SosRealtimeProcessor reference{sos, channels,
                                                            std::span<const double>{}};
    reference.process(expected, input.size() / channels);
    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(chain.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.publication_count() == 1);
    CHECK(nearly_equal(sink.values(), expected));
    CHECK(chain.flush(sink) == StreamStatus::ok);
    return 0;
}

} // namespace

int main()
{
    if (const auto status = test_single_frame_and_metadata(); status != 0)
    {
        return status;
    }
    if (const auto status = test_design_parameters_are_resolved_during_prepare(); status != 0)
    {
        return status;
    }
    if (const auto status = test_line_noise_harmonics_are_designed_during_prepare(); status != 0)
    {
        return status;
    }
    if (const auto status = test_variable_chunks_and_chunk_invariance(); status != 0)
    {
        return status;
    }
    if (const auto status = test_reset_discontinuity_and_empty_frame(); status != 0)
    {
        return status;
    }
    if (const auto status = test_steady_state_allocation(); status != 0)
    {
        return status;
    }
    if (const auto status = test_empty_state_at_mkl_threshold(); status != 0)
    {
        return status;
    }
    return test_linear_chain_terminal_sink();
}
