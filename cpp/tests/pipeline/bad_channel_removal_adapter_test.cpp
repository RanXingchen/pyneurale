/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_counter.h"
#include "bad_channel_removal_adapter.h"
#include "check_returns.h"
#include "inplace_adapter_test_support.h"

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <neurale/streaming/buffer_pool.h>

namespace
{

using namespace neurale::streaming;
namespace support = neurale::pipeline::test_support;

constexpr support::InplaceGeometry geometry{
    .n_channels = 4,
    .nominal_block_samples = 4,
    .max_block_samples = 8,
    .fs = {1'000, 1},
};

[[nodiscard]] StreamSchema make_schema(bool with_names = true, bool with_impedances = true)
{
    const std::vector<std::string> names =
        with_names ? std::vector<std::string>{"C1", "C2", "C3", "C4"} : std::vector<std::string>{};
    const std::vector<double> impedances =
        with_impedances ? std::vector<double>{1'000.0, 5'000.0, 20'000.0, 100'000.0}
                        : std::vector<double>{};
    const std::array signals{
        SignalSchema{
            support::kSignalId,
            SignalDType::float64,
            geometry.n_channels,
            geometry.nominal_block_samples,
            geometry.max_block_samples,
            geometry.fs,
            support::kClockDomain,
            SignalLayout::sample_major,
            DeviceTickTracking::sample_counter,
            PhysicalUnit::volts,
            support::kChannelSetId,
            support::kCalibrationId,
            support::kReferenceId,
            SignalKind::sampled,
            0,
            ObservationTiming::not_applicable,
            0,
            names,
            impedances,
        },
    };
    return StreamSchema{support::kSchemaId, signals};
}

[[nodiscard]] ProcessorPrepareContext context(const StreamSchema& schema) noexcept
{
    return support::make_context(schema, 1, 1, 0);
}

[[nodiscard]] int test_named_removal_contract_and_samples()
{
    auto schema = make_schema();
    const std::array<std::string, 2> bad{"C2", "C4"};
    neurale::pipeline::BadChannelRemovalAdapter adapter{{}, bad};
    const auto contract = adapter.prepare(context(schema));
    const auto& output_signal = contract.output_schema.signals().front();
    CHECK(contract.accepted_input_schema.equivalent(schema));
    CHECK(contract.output_schema.id() == schema.id() + 1);
    CHECK(output_signal.id == schema.signals().front().id);
    CHECK(output_signal.n_channels == 2);
    CHECK(output_signal.channel_set_id == schema.signals().front().channel_set_id + 1);
    CHECK(output_signal.channel_names == std::vector<std::string>({"C1", "C3"}));
    CHECK(output_signal.channel_impedances_ohm == std::vector<double>({1'000.0, 20'000.0}));
    CHECK(output_signal.fs.numerator == schema.signals().front().fs.numerator);
    CHECK(output_signal.fs.denominator == schema.signals().front().fs.denominator);
    CHECK(!contract.can_forward_input);
    CHECK(contract.max_process_outputs_per_input == 1);
    CHECK(contract.max_flush_outputs == 0);
    CHECK(contract.required_resources.frame_pool_leases == 1);

    FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease input;
    CHECK(input_pool.try_acquire(input) == StreamStatus::ok);
    constexpr std::array values{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    CHECK(support::fill_frame(input.frame(), schema, geometry, values) == StreamStatus::ok);
    support::TerminalSink sink{contract.output_schema, 1, values.size()};
    sink.begin(input.frame());
    CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    constexpr std::array expected{1.0, 3.0, 5.0, 7.0};
    CHECK(support::nearly_equal(sink.values(), expected));
    CHECK(sink.header().schema_id == contract.output_schema.id());
    CHECK(sink.block().payload_offset == 0);
    CHECK(sink.block().payload_byte_count == expected.size() * sizeof(double));
    CHECK(sink.block().n_samples == 2);
    return 0;
}

[[nodiscard]] int test_impedance_removal_and_combined_rules()
{
    auto schema = make_schema();
    neurale::pipeline::BadChannelRemovalAdapter impedance_only{{}, {}, 5'000.0, 20'000.0};
    const auto impedance_contract = impedance_only.prepare(context(schema));
    CHECK(impedance_contract.output_schema.signals().front().channel_names ==
          std::vector<std::string>({"C2", "C3"}));
    CHECK(impedance_contract.output_schema.signals().front().channel_impedances_ohm ==
          std::vector<double>({5'000.0, 20'000.0}));
    auto changed_signal = schema.signals().front();
    changed_signal.channel_impedances_ohm[0] = 2'000.0;
    const std::array changed_signals{changed_signal};
    auto changed_schema = StreamSchema{schema.id(), changed_signals};
    CHECK(support::throws_invalid_argument(
        [&] { static_cast<void>(impedance_only.prepare(context(changed_schema))); }));

    const std::array<std::string, 1> named_bad{"C2"};
    neurale::pipeline::BadChannelRemovalAdapter combined{{}, named_bad, 1'000.0, 20'000.0};
    const auto combined_contract = combined.prepare(context(schema));
    CHECK(combined_contract.output_schema.signals().front().channel_names ==
          std::vector<std::string>({"C1", "C3"}));

    auto no_impedances = make_schema(true, false);
    CHECK(support::throws_invalid_argument(
        [&]
        {
            neurale::pipeline::BadChannelRemovalAdapter missing{{}, {}, std::nullopt, 20'000.0};
            static_cast<void>(missing.prepare(context(no_impedances)));
        }));
    CHECK(support::throws_invalid_argument(
        [] { neurale::pipeline::BadChannelRemovalAdapter invalid{{}, {}, 20'000.0, 5'000.0}; }));
    return 0;
}

[[nodiscard]] int test_index_removal_and_rejections()
{
    auto schema = make_schema();
    constexpr std::array<std::size_t, 2> bad{0, 2};
    neurale::pipeline::BadChannelRemovalAdapter adapter{bad, {}};
    const auto contract = adapter.prepare(context(schema));
    CHECK(contract.output_schema.signals().front().channel_names ==
          std::vector<std::string>({"C2", "C4"}));
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(adapter.handle_discontinuity({}) == StreamStatus::ok);

    auto unnamed = make_schema(false);
    const std::array<std::string, 1> named_bad{"C2"};
    CHECK(support::throws_invalid_argument(
        [&]
        {
            neurale::pipeline::BadChannelRemovalAdapter named{{}, named_bad};
            static_cast<void>(named.prepare(context(unnamed)));
        }));
    constexpr std::array<std::size_t, 4> all{0, 1, 2, 3};
    CHECK(support::throws_invalid_argument(
        [&]
        {
            neurale::pipeline::BadChannelRemovalAdapter remove_all{all, {}};
            static_cast<void>(remove_all.prepare(context(schema)));
        }));
    constexpr std::array<std::size_t, 1> out_of_range{4};
    CHECK(support::throws_invalid_argument(
        [&]
        {
            neurale::pipeline::BadChannelRemovalAdapter invalid{out_of_range, {}};
            static_cast<void>(invalid.prepare(context(schema)));
        }));
    return 0;
}

[[nodiscard]] int test_steady_state_allocation()
{
    auto schema = make_schema();
    constexpr std::array<std::size_t, 1> bad{1};
    neurale::pipeline::BadChannelRemovalAdapter adapter{bad, {}};
    const auto contract = adapter.prepare(context(schema));
    FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease input;
    CHECK(input_pool.try_acquire(input) == StreamStatus::ok);
    constexpr std::array values{1.0, 2.0, 3.0, 4.0};
    support::TerminalSink sink{contract.output_schema, 1, values.size()};

    CHECK(support::fill_frame(input.frame(), schema, geometry, values) == StreamStatus::ok);
    sink.begin(input.frame());
    CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    const auto before = allocations.load(std::memory_order_relaxed);
    for (std::size_t iteration = 0; iteration < 128; ++iteration)
    {
        CHECK(support::fill_frame(input.frame(), schema, geometry, values, iteration + 1,
                                  iteration) == StreamStatus::ok);
        sink.begin(input.frame());
        CHECK(adapter.process(input.frame(), sink) == StreamStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    return 0;
}

} // namespace

int main()
{
    if (const auto status = test_named_removal_contract_and_samples(); status != 0)
    {
        return status;
    }
    if (const auto status = test_index_removal_and_rejections(); status != 0)
    {
        return status;
    }
    if (const auto status = test_impedance_removal_and_combined_rules(); status != 0)
    {
        return status;
    }
    return test_steady_state_allocation();
}
