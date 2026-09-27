/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_counter.h"
#include "check_returns.h"
#include "inplace_adapter_test_support.h"
#include "spatial_realtime.h"
#include "spatial_reference_adapter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>

#include <neurale/signal/spatial.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/linear_processor_chain.h>

namespace
{

using namespace neurale::streaming;

constexpr std::uint32_t channels = 4;
constexpr std::size_t max_samples = 16;
constexpr std::size_t max_scalars = channels * max_samples;

namespace test_support = neurale::pipeline::test_support;

// 30 kHz and a one-output/one-lease context, matching the SOS test rather than
// the 1 kHz FIR and IIR ones: each geometry was chosen against what that
// adapter dispatches on.
constexpr test_support::InplaceGeometry base_geometry{
    .n_channels = channels,
    .nominal_block_samples = 4,
    .max_block_samples = static_cast<std::uint32_t>(max_samples),
    .fs = {30'000, 1},
    .host_time_reference_ns = 4'000,
};

[[nodiscard]] StreamSchema make_schema(std::uint32_t n_channels = channels,
                                       SignalDType dtype = SignalDType::float64,
                                       SignalLayout layout = SignalLayout::sample_major)
{
    auto geometry = base_geometry;
    geometry.n_channels = n_channels;
    geometry.dtype = dtype;
    geometry.layout = layout;
    return test_support::make_schema(geometry);
}

// Built once at static-initialisation time, ahead of any allocation snapshot:
// `fill_frame` runs inside the steady-state loops that assert zero allocations,
// and constructing a StreamSchema per call would be counted against the adapter.
const StreamSchema fixed_schema = make_schema();

/// A non-sampled signal the adapter must refuse. `event` rather than `feature`
/// because a feature signal is only a valid StreamSchema alongside a registered
/// feature-set descriptor, and the guard under test is `kind != sampled`.
[[nodiscard]] StreamSchema make_event_schema()
{
    const std::array signals{
        SignalSchema{
            11,
            SignalDType::float64,
            channels,
            1,
            1,
            {0, 1},
            41,
            SignalLayout::sample_major,
            DeviceTickTracking::sample_counter,
            PhysicalUnit::volts,
            17,
            19,
            23,
            SignalKind::event,
        },
    };
    return StreamSchema{29, signals};
}

[[nodiscard]] StreamSchema make_two_signal_schema()
{
    const auto single = make_schema();
    const std::array signals{single.signals().front(), single.signals().front()};
    std::array copies{signals[0], signals[1]};
    copies[1].id = 13;
    return StreamSchema{29, copies};
}

[[nodiscard]] ProcessorPrepareContext context(const StreamSchema& schema, std::size_t outputs = 1,
                                              std::size_t leases = 1) noexcept
{
    return test_support::make_context(schema, outputs, leases, 0);
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, std::span<const double> values,
                                      std::uint64_t sequence = 7,
                                      SampleIndex sample_start = 100) noexcept
{
    return test_support::fill_frame(frame, fixed_schema, base_geometry, values, sequence,
                                    sample_start);
}

using test_support::nearly_equal;

/// The expected output, computed here from the definition.
///
/// This used to delegate to the batch `common_reference` in
/// cpp/src/signal/filtering/spatial.cpp. That stopped being an oracle the
/// moment the batch entry point was reduced to a shell over the same kernel the
/// adapter drives -- it would now be comparing the kernel to itself, which
/// proves only that it is deterministic. So the expectation is spelled out:
/// a full `std::sort` rather than the kernel's `nth_element`, and the even-count
/// average written out, so an error in the selection cannot be mirrored here.
[[nodiscard]] std::vector<double> expected_reference(std::span<const double> input,
                                                     std::span<const std::size_t> positions,
                                                     std::string_view method,
                                                     std::span<double> reference = {})
{
    std::vector<std::size_t> selected(positions.begin(), positions.end());
    if (selected.empty())
    {
        selected.resize(channels);
        for (std::size_t i = 0; i < channels; ++i)
        {
            selected[i] = i;
        }
    }
    const auto n_samples = input.size() / channels;
    std::vector<double> output(input.size());
    std::vector<double> gathered(selected.size());
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        const auto offset = sample * channels;
        for (std::size_t i = 0; i < selected.size(); ++i)
        {
            gathered[i] = input[offset + selected[i]];
        }
        double value = 0.0;
        if (method == "mean")
        {
            for (const auto entry : gathered)
            {
                value += entry;
            }
            value /= static_cast<double>(gathered.size());
        }
        else
        {
            std::sort(gathered.begin(), gathered.end());
            const auto middle = gathered.size() / 2;
            value = gathered.size() % 2 == 0 ? 0.5 * (gathered[middle - 1] + gathered[middle])
                                             : gathered[middle];
        }
        if (!reference.empty())
        {
            reference[sample] = value;
        }
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            output[offset + channel] = input[offset + channel] - value;
        }
    }
    return output;
}

/// Convenience wrapper preserving the call shape the cases below already use.
[[nodiscard]] std::vector<double> batch_reference(std::span<const double> input,
                                                  std::span<const std::size_t> reference_channels,
                                                  std::string_view method = "mean")
{
    return expected_reference(input, reference_channels, method);
}

[[nodiscard]] std::vector<double> make_input(std::size_t n_samples)
{
    std::vector<double> values(n_samples * channels);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        // A per-channel offset on top of a shared drift, so a common average
        // that is dropped or mis-scaled cannot pass by accident.
        values[i] = std::sin(static_cast<double>(i) * 0.19) + static_cast<double>(i % channels) +
                    0.05 * static_cast<double>(i / channels);
    }
    return values;
}

// A pool capacity of zero: nothing downstream of a common-reference adapter in
// this test acquires, so an acquire attempt is a defect and reported as one.
class TerminalSinkEmitter final : public test_support::TerminalSink
{
  public:
    explicit TerminalSinkEmitter(const StreamSchema& schema)
        : test_support::TerminalSink(schema, 0, max_scalars)
    {
    }
};

template <typename Callable> [[nodiscard]] bool throws_invalid_argument(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

[[nodiscard]] int test_contract_metadata_and_batch_parity()
{
    auto schema = make_schema();
    neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}};
    const auto contract = adapter.prepare(context(schema));
    CHECK(contract.accepted_input_schema.equivalent(schema));
    // The output schema is the input schema, reference_id included: forwarding a
    // frame without copying requires exactly that (linear_processor_chain.cpp).
    CHECK(contract.output_schema.equivalent(schema));
    CHECK(contract.output_schema.signals().front().reference_id ==
          schema.signals().front().reference_id);
    CHECK(contract.max_process_outputs_per_input == 1);
    CHECK(contract.max_flush_outputs == 0);
    CHECK(contract.can_forward_input);
    CHECK(contract.required_resources.frame_pool_leases == 1);
    CHECK(contract.required_resources.workspace_bytes == 0);

    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const auto input = make_input(3);
    const auto expected = batch_reference(input, {});
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

    // Referencing against every channel leaves no common component behind.
    const auto values = sink.values();
    for (std::size_t sample = 0; sample < values.size() / channels; ++sample)
    {
        double total = 0.0;
        for (std::size_t channel = 0; channel < channels; ++channel)
        {
            total += values[sample * channels + channel];
        }
        CHECK(std::abs(total) < 1e-12);
    }
    return 0;
}

[[nodiscard]] int test_reference_subset()
{
    constexpr std::array<std::size_t, 2> subset{0, 2};
    auto schema = make_schema();
    neurale::pipeline::SpatialReferenceAdapter adapter{subset};
    const auto contract = adapter.prepare(context(schema));
    CHECK(contract.required_resources.workspace_bytes == subset.size() * sizeof(std::size_t));

    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const auto input = make_input(5);
    const auto expected = batch_reference(input, subset);
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));

    // A subset that happens to name every channel is the whole-set case; the
    // constructor's fast path must not change the arithmetic.
    constexpr std::array<std::size_t, channels> all{0, 1, 2, 3};
    neurale::pipeline::SpatialReferenceAdapter explicit_all{all};
    static_cast<void>(explicit_all.prepare(context(schema)));
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(explicit_all.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), batch_reference(input, {})));
    return 0;
}

template <std::size_t ChunkCount>
[[nodiscard]] int run_chunks(std::span<const double> input,
                             const std::array<std::size_t, ChunkCount>& chunk_samples,
                             std::span<double> output,
                             neurale::pipeline::SpatialReferenceStatistic statistic =
                                 neurale::pipeline::SpatialReferenceStatistic::mean)
{
    auto schema = make_schema();
    neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}, statistic};
    static_cast<void>(adapter.prepare(context(schema)));
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

[[nodiscard]] int test_chunk_invariance_reset_and_discontinuity()
{
    const auto input = make_input(13);
    const auto expected = batch_reference(input, {});
    std::vector<double> first(input.size());
    std::vector<double> second(input.size());
    CHECK(run_chunks(input, std::array<std::size_t, 4>{1, 4, 2, 6}, first) == 0);
    CHECK(run_chunks(input, std::array<std::size_t, 3>{7, 1, 5}, second) == 0);
    CHECK(nearly_equal(first, expected));
    CHECK(nearly_equal(second, expected));

    // The kernel is stateless, so reset and discontinuity must both be accepted
    // and must leave the next block's output untouched.
    auto schema = make_schema();
    neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}};
    static_cast<void>(adapter.prepare(context(schema)));
    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    const auto probe = make_input(3);
    const auto probe_expected = batch_reference(probe, {});

    CHECK(fill_frame(lease.frame(), probe) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(adapter.flush(sink) == StreamStatus::ok);
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
    CHECK(nearly_equal(sink.values(), probe_expected));

    // An empty frame carries no block for the kernel to reference against.
    CHECK(lease.frame().set_used_sizes(0, 0) == StreamStatus::ok);
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);
    CHECK(sink.publication_count() == 0);
    return 0;
}

[[nodiscard]] int test_median_parity_and_workspace()
{
    using neurale::pipeline::SpatialReferenceStatistic;
    auto schema = make_schema();
    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    const auto input = make_input(6);

    // Whole channel set: four channels, so the even branch that averages the
    // two middle values runs. That branch is the one that has to stay bit-equal
    // with the batch implementation.
    {
        neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{},
                                                           SpatialReferenceStatistic::median};
        const auto contract = adapter.prepare(context(schema));
        CHECK(contract.output_schema.equivalent(schema));
        CHECK(contract.can_forward_input);
        CHECK(contract.required_resources.workspace_bytes == channels * sizeof(double));
        CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(nearly_equal(sink.values(), batch_reference(input, {}, "median")));
        // A median reference is not a mean reference; without this the whole
        // suite would still pass if the statistic argument were ignored.
        CHECK(!nearly_equal(sink.values(), batch_reference(input, {}, "mean")));
    }

    // Odd count: no second pass, the selected element is the answer.
    {
        constexpr std::array<std::size_t, 3> odd_subset{0, 1, 2};
        neurale::pipeline::SpatialReferenceAdapter adapter{odd_subset,
                                                           SpatialReferenceStatistic::median};
        const auto contract = adapter.prepare(context(schema));
        CHECK(contract.required_resources.workspace_bytes ==
              odd_subset.size() * sizeof(std::size_t) + odd_subset.size() * sizeof(double));
        CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(nearly_equal(sink.values(), batch_reference(input, odd_subset, "median")));
    }

    // Two channels is the smallest even count: the lower half is a single
    // element and max_element runs over exactly one value.
    {
        constexpr std::array<std::size_t, 2> even_subset{0, 2};
        neurale::pipeline::SpatialReferenceAdapter adapter{even_subset,
                                                           SpatialReferenceStatistic::median};
        const auto contract = adapter.prepare(context(schema));
        CHECK(contract.required_resources.workspace_bytes ==
              even_subset.size() * sizeof(std::size_t) + even_subset.size() * sizeof(double));
        CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(nearly_equal(sink.values(), batch_reference(input, even_subset, "median")));
    }
    return 0;
}

[[nodiscard]] int test_median_chunk_invariance_and_allocation()
{
    using neurale::pipeline::SpatialReferenceStatistic;
    const auto input = make_input(13);
    const auto expected = batch_reference(input, {}, "median");
    std::vector<double> first(input.size());
    std::vector<double> second(input.size());
    CHECK(run_chunks(input, std::array<std::size_t, 4>{1, 4, 2, 6}, first,
                     SpatialReferenceStatistic::median) == 0);
    CHECK(run_chunks(input, std::array<std::size_t, 3>{7, 1, 5}, second,
                     SpatialReferenceStatistic::median) == 0);
    CHECK(nearly_equal(first, expected));
    CHECK(nearly_equal(second, expected));

    // The regression this whole change exists for: the batch median acquires a
    // workspace on every call, and moving it to construction is the only reason
    // this path is allowed on the data plane at all.
    auto schema = make_schema();
    neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{},
                                                       SpatialReferenceStatistic::median};
    static_cast<void>(adapter.prepare(context(schema)));
    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const auto probe = make_input(4);
    CHECK(fill_frame(lease.frame(), probe) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);

    const auto before = allocations.load(std::memory_order_relaxed);
    for (std::size_t iteration = 0; iteration < 128; ++iteration)
    {
        CHECK(fill_frame(lease.frame(), probe, iteration + 1, iteration * 4) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    return 0;
}

[[nodiscard]] int test_rejections_and_unprepared_state()
{
    auto schema = make_schema();
    TerminalSinkEmitter unprepared_sink{schema};
    neurale::pipeline::SpatialReferenceAdapter unprepared{std::span<const std::size_t>{}};
    CHECK(unprepared.reset() == StreamStatus::invalid_state);
    CHECK(unprepared.flush(unprepared_sink) == StreamStatus::invalid_state);
    const Discontinuity discontinuity{};
    CHECK(unprepared.handle_discontinuity(discontinuity) == StreamStatus::invalid_state);

    CHECK(throws_invalid_argument(
        []
        {
            constexpr std::array<std::size_t, 3> duplicated{1, 2, 1};
            neurale::pipeline::SpatialReferenceAdapter adapter{duplicated};
        }));
    CHECK(throws_invalid_argument(
        [&]
        {
            constexpr std::array<std::size_t, 1> out_of_range{channels};
            neurale::pipeline::SpatialReferenceAdapter adapter{out_of_range};
            static_cast<void>(adapter.prepare(context(schema)));
        }));

    auto float_schema = make_schema(channels, SignalDType::float32);
    auto channel_major = make_schema(channels, SignalDType::float64, SignalLayout::channel_major);
    auto event_schema = make_event_schema();
    auto two_signals = make_two_signal_schema();
    for (const auto* rejected : {&float_schema, &channel_major, &event_schema, &two_signals})
    {
        CHECK(throws_invalid_argument(
            [&]
            {
                neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}};
                static_cast<void>(adapter.prepare(context(*rejected)));
            }));
    }
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}};
            static_cast<void>(adapter.prepare(context(schema, 0, 1)));
        }));
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}};
            static_cast<void>(adapter.prepare(context(schema, 1, 0)));
        }));

    neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}};
    static_cast<void>(adapter.prepare(context(schema)));
    // Re-preparing on the same schema is how the runtime restarts a session.
    static_cast<void>(adapter.prepare(context(schema)));
    auto changed_channels = make_schema(channels + 1);
    CHECK(throws_invalid_argument(
        [&] { static_cast<void>(adapter.prepare(context(changed_channels))); }));

    // The kernel itself refuses a degenerate geometry, which is what keeps the
    // adapter's own guard from being the only thing standing between a zero
    // channel count and a division by zero.
    CHECK(throws_invalid_argument(
        []
        {
            neurale::signal::detail::CommonReferenceRealtimeProcessor processor{
                0, std::span<const std::size_t>{}};
        }));
    return 0;
}

[[nodiscard]] int test_steady_state_allocation()
{
    auto schema = make_schema();
    constexpr std::array<std::size_t, 2> subset{1, 3};
    neurale::pipeline::SpatialReferenceAdapter adapter{subset};
    static_cast<void>(adapter.prepare(context(schema)));
    FramePool pool{1, max_scalars * sizeof(double), 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const auto input = make_input(4);
    CHECK(fill_frame(lease.frame(), input) == StreamStatus::ok);
    TerminalSinkEmitter sink{schema};
    sink.begin(lease.frame());
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);

    const auto before = allocations.load(std::memory_order_relaxed);
    for (std::size_t iteration = 0; iteration < 128; ++iteration)
    {
        CHECK(fill_frame(lease.frame(), input, iteration + 1, iteration * 4) == StreamStatus::ok);
        sink.begin(lease.frame());
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    return 0;
}

[[nodiscard]] int test_batch_entry_point_shell()
{
    // spatial.cpp is now validation plus a copy plus one call into the same
    // kernel. Three things about that shell can break without any adapter test
    // noticing: the copy could be dropped, the reference span could be misfed,
    // and `x` could stop being const in practice. All three are checked here
    // because this is the only C++ suite that links both sides.
    const auto input = make_input(5);
    for (const auto* method : {"mean", "median"})
    {
        for (const auto& positions :
             {std::vector<std::size_t>{0, 1, 2, 3}, std::vector<std::size_t>{0, 2}})
        {
            const auto n_samples = input.size() / channels;
            auto guarded = input;
            std::vector<double> output(input.size(), 0.0);
            std::vector<double> reference(n_samples, 0.0);
            neurale::signal::common_reference(guarded, n_samples, channels, positions, method,
                                              output, reference);

            std::vector<double> expected_values(n_samples, 0.0);
            const auto expected = expected_reference(input, positions, method, expected_values);
            CHECK(nearly_equal(output, expected));
            CHECK(nearly_equal(reference, expected_values));
            // The published contract says the input is not modified. The copy
            // that makes the shared kernel usable here is what upholds it.
            CHECK(nearly_equal(guarded, input));
        }
    }

    // Zero samples is a shape the Python suite parametrises over, and it is the
    // one that reaches the kernel with an empty span whose data() may be null.
    constexpr std::array<std::size_t, 1> single{0};
    neurale::signal::common_reference(std::span<const double>{}, 0, channels, single, "median",
                                      std::span<double>{}, std::span<double>{});
    return 0;
}

[[nodiscard]] int test_linear_chain_terminal_sink()
{
    auto schema = make_schema();
    neurale::pipeline::SpatialReferenceAdapter adapter{std::span<const std::size_t>{}};
    std::array<NativeFrameProcessor*, 1> stages{&adapter};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(context(schema));
    CHECK(contract.accepted_input_schema.equivalent(schema));
    CHECK(contract.output_schema.equivalent(schema));
    CHECK(contract.can_forward_input);

    const auto input = make_input(2);
    const auto expected = batch_reference(input, {});
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
    if (const auto status = test_contract_metadata_and_batch_parity(); status != 0)
    {
        return status;
    }
    if (const auto status = test_reference_subset(); status != 0)
    {
        return status;
    }
    if (const auto status = test_chunk_invariance_reset_and_discontinuity(); status != 0)
    {
        return status;
    }
    if (const auto status = test_median_parity_and_workspace(); status != 0)
    {
        return status;
    }
    if (const auto status = test_median_chunk_invariance_and_allocation(); status != 0)
    {
        return status;
    }
    if (const auto status = test_rejections_and_unprepared_state(); status != 0)
    {
        return status;
    }
    if (const auto status = test_steady_state_allocation(); status != 0)
    {
        return status;
    }
    if (const auto status = test_batch_entry_point_shell(); status != 0)
    {
        return status;
    }
    return test_linear_chain_terminal_sink();
}
