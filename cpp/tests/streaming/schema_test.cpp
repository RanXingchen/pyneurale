/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/frame.h>
#include <neurale/streaming/realtime_config.h>
#include <neurale/streaming/schema.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "check_returns.h"

namespace
{

template <typename Exception, typename Callable> [[nodiscard]] bool throws(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const Exception&)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

using neurale::streaming::ClockDomainId;
using neurale::streaming::ClockSyncFlags;
using neurale::streaming::ClockSyncSnapshot;
using neurale::streaming::DeviceTickTracking;
using neurale::streaming::FeatureSetDescriptor;
using neurale::streaming::FeatureSetDescriptorRegistry;
using neurale::streaming::FeatureTimestampReference;
using neurale::streaming::FrameFlags;
using neurale::streaming::FrameHeader;
using neurale::streaming::ObservationTiming;
using neurale::streaming::PhysicalUnit;
using neurale::streaming::PoolCapacityBudget;
using neurale::streaming::RationalRate;
using neurale::streaming::RealtimeConfig;
using neurale::streaming::SignalBlockHeader;
using neurale::streaming::SignalDType;
using neurale::streaming::SignalKind;
using neurale::streaming::SignalLayout;
using neurale::streaming::SignalSchema;
using neurale::streaming::StreamSchema;
using neurale::streaming::UnitDescriptor;
using neurale::streaming::UnitRegistry;

[[nodiscard]] SignalSchema make_signal(std::uint32_t id, RationalRate rate = {30'000, 1'001},
                                       ClockDomainId clock_domain = 3)
{
    return SignalSchema{
        id, SignalDType::float32, 64, 32, 128, rate, clock_domain, SignalLayout::sample_major,
    };
}

int test_valid_schema()
{
    const auto neural = make_signal(11);
    const SignalSchema stimulus{
        12,
        SignalDType::int16,
        2,
        1,
        8,
        {1'000, 1},
        9,
        SignalLayout::channel_major,
        DeviceTickTracking::sample_counter,
        PhysicalUnit::volts,
        101,
        202,
        303,
    };
    const SignalSchema labels{
        13, SignalDType::int32, 1, 4, 4, {10, 1}, 4,
    };
    const SignalSchema trigger{
        14,
        SignalDType::int32,
        1,
        1,
        1,
        {0, 1},
        5,
        SignalLayout::sample_major,
        DeviceTickTracking::unavailable,
        PhysicalUnit::dimensionless,
        0,
        0,
        0,
        SignalKind::event,
    };
    const std::array signals{neural, stimulus, labels, trigger};
    const StreamSchema schema{27, signals};

    CHECK(neural.max_block_bytes == 64U * 128U * 4U);
    CHECK(neural.fs.numerator == 30'000);
    CHECK(neural.fs.denominator == 1'001);
    CHECK(stimulus.max_block_bytes == 2U * 8U * 2U);
    CHECK(stimulus.physical_unit == PhysicalUnit::volts);
    CHECK(stimulus.channel_set_id == 101);
    CHECK(stimulus.calibration_id == 202);
    CHECK(stimulus.reference_id == 303);
    CHECK(schema.id() == 27);
    CHECK(schema.signals().data() != signals.data());
    CHECK(schema.signals()[0].id == signals[0].id);
    CHECK(labels.max_block_bytes == 4U * 4U);
    CHECK(trigger.kind == SignalKind::event);
    CHECK(schema.signals().size() == 4);
    return 0;
}

int test_schema_validation()
{
    CHECK(throws<std::invalid_argument>(
        [] { SignalSchema{1, static_cast<SignalDType>(255), 1, 1, 1, {1, 1}, 1}; }));
    CHECK(throws<std::invalid_argument>(
        []
        {
            SignalSchema{1, SignalDType::float32,          1, 1, 1, {1, 1},
                         1, static_cast<SignalLayout>(255)};
        }));
    CHECK(throws<std::invalid_argument>(
        []
        {
            SignalSchema{1,
                         SignalDType::float32,
                         1,
                         1,
                         1,
                         {1, 1},
                         1,
                         SignalLayout::sample_major,
                         static_cast<DeviceTickTracking>(255)};
        }));
    CHECK(throws<std::invalid_argument>(
        [] { SignalSchema{1, SignalDType::float32, 0, 1, 1, {1, 1}, 1}; }));
    CHECK(throws<std::invalid_argument>(
        [] { SignalSchema{1, SignalDType::float32, 1, 1, 1, {0, 1}, 1}; }));
    CHECK(throws<std::invalid_argument>(
        [] { SignalSchema{1, SignalDType::float32, 1, 1, 1, {1, 0}, 1}; }));
    CHECK(throws<std::invalid_argument>(
        [] { SignalSchema{1, SignalDType::float32, 1, 0, 1, {1, 1}, 1}; }));
    CHECK(throws<std::invalid_argument>(
        [] { SignalSchema{1, SignalDType::float32, 1, 2, 1, {1, 1}, 1}; }));
    auto mismatched_names = make_signal(1);
    mismatched_names.channel_names = {"only-one"};
    CHECK(throws<std::invalid_argument>(
        [&] { StreamSchema{1, std::span<const SignalSchema>{&mismatched_names, 1}}; }));
    auto duplicate_names = make_signal(1);
    duplicate_names.channel_names.assign(64, "duplicate");
    CHECK(throws<std::invalid_argument>(
        [&] { StreamSchema{1, std::span<const SignalSchema>{&duplicate_names, 1}}; }));
    auto mismatched_impedances = make_signal(1);
    mismatched_impedances.channel_impedances_ohm = {1'000.0};
    CHECK(throws<std::invalid_argument>(
        [&] { StreamSchema{1, std::span<const SignalSchema>{&mismatched_impedances, 1}}; }));
    auto invalid_impedances = make_signal(1);
    invalid_impedances.channel_impedances_ohm.assign(64, 1'000.0);
    invalid_impedances.channel_impedances_ohm[0] = -1.0;
    CHECK(throws<std::invalid_argument>(
        [&] { StreamSchema{1, std::span<const SignalSchema>{&invalid_impedances, 1}}; }));
    return 0;
}

int test_fixed_sparse_spike_schema()
{
    const SignalSchema spike{31,
                             SignalDType::float64,
                             4,
                             8,
                             8,
                             {0, 1},
                             7,
                             SignalLayout::sample_major,
                             DeviceTickTracking::unavailable,
                             PhysicalUnit::volts,
                             17,
                             0,
                             0,
                             SignalKind::spike,
                             0,
                             ObservationTiming::not_applicable,
                             4'096};
    CHECK(spike.max_block_bytes == 4'096);
    CHECK(spike.fixed_block_bytes == 4'096);
    CHECK(throws<std::invalid_argument>(
        []
        {
            SignalSchema{31,
                         SignalDType::float64,
                         4,
                         8,
                         8,
                         {0, 1},
                         7,
                         SignalLayout::sample_major,
                         DeviceTickTracking::unavailable,
                         PhysicalUnit::volts,
                         17,
                         0,
                         0,
                         SignalKind::spike};
        }));
    CHECK(throws<std::invalid_argument>(
        []
        {
            SignalSchema{31,
                         SignalDType::float64,
                         4,
                         8,
                         8,
                         {1, 1},
                         7,
                         SignalLayout::sample_major,
                         DeviceTickTracking::unavailable,
                         PhysicalUnit::volts,
                         17,
                         0,
                         0,
                         SignalKind::spike,
                         0,
                         ObservationTiming::not_applicable,
                         4'096};
        }));
    return 0;
}

int test_block_byte_boundaries()
{
    constexpr auto max_u32 = std::numeric_limits<std::uint32_t>::max();
    constexpr std::uint32_t max_samples_without_overflow = max_u32 / 8U;
    const SignalSchema boundary{
        1, SignalDType::float64, max_u32, 1, max_samples_without_overflow, {1, 1}, 1,
    };
    const auto expected = static_cast<std::uint64_t>(max_u32) * max_samples_without_overflow * 8U;
    CHECK(boundary.max_block_bytes == expected);

    CHECK(throws<std::overflow_error>(
        [=] { SignalSchema{1, SignalDType::float64, max_u32, 1, max_u32, {1, 1}, 1}; }));
    return 0;
}

int test_stream_schema_validation()
{
    const std::array duplicate{make_signal(4), make_signal(4, {1'000, 1}, 8)};
    CHECK(throws<std::invalid_argument>([&] { StreamSchema{1, duplicate}; }));

    const std::span<const SignalSchema> empty;
    CHECK(throws<std::invalid_argument>([&] { StreamSchema{1, empty}; }));

    auto stale = make_signal(8);
    stale.n_channels = 32;
    const std::array stale_signals{stale};
    CHECK(throws<std::invalid_argument>([&] { StreamSchema{1, stale_signals}; }));
    return 0;
}

int test_feature_schema_and_registries()
{
    const std::array units{
        UnitDescriptor{1, "V^2", "squared volts"},
        UnitDescriptor{2, "V^2/Hz", "power spectral density"},
    };
    const FeatureSetDescriptor descriptor{
        .id = 91,
        .feature_names = {"lmp", "beta_power"},
        .unit_ids = {1, 2},
        .source_stream_id = 17,
        .source_stream = "motor-cortex",
        .algorithm_name = "test-features",
        .algorithm_version = "1",
        .window_length_ns = 100'000'000,
        .shift_ns = 50'000'000,
        .timestamp_reference = FeatureTimestampReference::window_center,
    };
    const std::array descriptors{descriptor};
    const SignalSchema feature{
        21,
        SignalDType::float32,
        2,
        2,
        4,
        {20, 1},
        3,
        SignalLayout::sample_major,
        DeviceTickTracking::unavailable,
        PhysicalUnit::unspecified,
        0,
        0,
        0,
        SignalKind::feature,
        91,
        ObservationTiming::regular,
    };
    const std::array signals{feature};
    const StreamSchema schema{41, signals, descriptors, units};

    CHECK(feature.kind == SignalKind::feature);
    CHECK(feature.feature_set_id == 91);
    CHECK(feature.observation_timing == ObservationTiming::regular);
    const auto* stored = schema.feature_sets().find(91);
    CHECK(stored != nullptr);
    CHECK(stored->feature_names == descriptor.feature_names);
    CHECK(stored->unit_ids == descriptor.unit_ids);
    CHECK(stored->source_stream_id == 17);
    CHECK(stored->source_stream == "motor-cortex");
    CHECK(stored->window_length_ns == 100'000'000);
    CHECK(stored->shift_ns == 50'000'000);
    CHECK(stored->timestamp_reference == FeatureTimestampReference::window_center);
    CHECK(schema.units().find(1)->symbol == "V^2");
    CHECK(schema.clone().equivalent(schema));
    return 0;
}

int test_feature_registry_validation()
{
    const std::array units{UnitDescriptor{1, "V", "volts"}};
    const FeatureSetDescriptor descriptor{
        .id = 4,
        .feature_names = {"mean"},
        .unit_ids = {1},
        .source_stream_id = 2,
        .source_stream = "source",
        .window_length_ns = 100'000'000,
        .shift_ns = 50'000'000,
    };
    const std::array duplicate_descriptors{descriptor, descriptor};
    CHECK(throws<std::invalid_argument>(
        [&] { FeatureSetDescriptorRegistry registry{duplicate_descriptors}; }));
    const std::array duplicate_units{
        UnitDescriptor{1, "V", "volts"},
        UnitDescriptor{1, "mV", "millivolts"},
    };
    CHECK(throws<std::invalid_argument>([&] { UnitRegistry registry{duplicate_units}; }));

    const SignalSchema feature{
        7,
        SignalDType::float32,
        1,
        1,
        2,
        {20, 1},
        3,
        SignalLayout::sample_major,
        DeviceTickTracking::unavailable,
        PhysicalUnit::unspecified,
        0,
        0,
        0,
        SignalKind::feature,
        4,
        ObservationTiming::regular,
    };
    const std::array signals{feature};
    CHECK(throws<std::invalid_argument>([&] { StreamSchema{8, signals}; }));

    const std::array descriptors{descriptor};
    const std::span<const UnitDescriptor> no_units;
    CHECK(throws<std::invalid_argument>([&] { StreamSchema{8, signals, descriptors, no_units}; }));

    auto duplicate_names = descriptor;
    duplicate_names.feature_names = {"mean", "mean"};
    duplicate_names.unit_ids = {1, 1};
    const std::array invalid_descriptors{duplicate_names};
    CHECK(throws<std::invalid_argument>(
        [&] { FeatureSetDescriptorRegistry registry{invalid_descriptors}; }));

    const SignalSchema wrong_rate{
        7,
        SignalDType::float32,
        1,
        1,
        2,
        {10, 1},
        3,
        SignalLayout::sample_major,
        DeviceTickTracking::unavailable,
        PhysicalUnit::unspecified,
        0,
        0,
        0,
        SignalKind::feature,
        4,
        ObservationTiming::regular,
    };
    const std::array wrong_rate_signals{wrong_rate};
    CHECK(throws<std::invalid_argument>(
        [&] { StreamSchema{8, wrong_rate_signals, descriptors, units}; }));
    return 0;
}

int test_frame_headers()
{
    static_assert(sizeof(FrameHeader) == 56);
    static_assert(sizeof(ClockSyncSnapshot) == 56);
    static_assert(sizeof(SignalBlockHeader) == 112);
    static_assert(std::is_trivially_copyable_v<FrameHeader>);
    static_assert(std::is_trivially_copyable_v<SignalBlockHeader>);
    static_assert(std::is_trivially_move_constructible_v<FrameHeader>);
    static_assert(std::is_trivially_move_constructible_v<SignalBlockHeader>);
    static_assert(!std::is_copy_constructible_v<StreamSchema>);

    const FrameHeader frame{
        .session_id = 31,
        .sequence = 72,
        .host_received_ns = 10'000,
        .source_tick = 400,
        .valid_until_ns = 12'000,
        .schema_id = 5,
        .source_clock_domain = 3,
        .signal_block_count = 2,
        .flags = FrameFlags::source_tick | FrameFlags::valid_until,
    };
    const std::array blocks{
        SignalBlockHeader{
            .sample_idx_start = 1'024,
            .device_tick_start = 8'000,
            .payload_offset = 0,
            .payload_byte_count = 32U * 64U * 4U,
            .signal_id = 11,
            .n_samples = 32,
            .clock_sync =
                ClockSyncSnapshot{
                    .device_tick_reference = 8'000,
                    .host_time_reference_ns = 10'000,
                    .device_tick_rate = {30'000, 1'001},
                    .uncertainty_ns = 200,
                    .clock_domain = 3,
                    .generation = 4,
                    .flags = ClockSyncFlags::synchronized,
                },
        },
        SignalBlockHeader{
            .sample_idx_start = 12,
            .device_tick_start = 2'000,
            .payload_offset = 32U * 64U * 4U,
            .payload_byte_count = 2U * 2U,
            .signal_id = 12,
            .n_samples = 2,
        },
    };

    CHECK(neurale::streaming::has_flag(frame.flags, FrameFlags::source_tick));
    CHECK(neurale::streaming::has_flag(frame.flags, FrameFlags::valid_until));
    CHECK(blocks[0].sample_idx_start != blocks[1].sample_idx_start);
    CHECK(blocks[0].device_tick_start != blocks[1].device_tick_start);
    CHECK(blocks[0].clock_sync.clock_domain == 3);
    CHECK(blocks[0].clock_sync.generation == 4);
    CHECK(blocks.size() == frame.signal_block_count);

    const FrameHeader no_optional_fields{};
    CHECK(!neurale::streaming::has_flag(no_optional_fields.flags, FrameFlags::source_tick));
    CHECK(!neurale::streaming::has_flag(no_optional_fields.flags, FrameFlags::valid_until));
    return 0;
}

int test_pool_capacity_budget()
{
    const PoolCapacityBudget budget{
        .source_owned = 1,
        .ingress_capacity = 8,
        .processor_owned = 2,
        .critical_edge_capacity = 4,
        .actuator_owned = 1,
        .observer_edge_capacity = 16,
        .reserve = 3,
    };
    CHECK(budget.required_buffer_count() == 35);

    const RealtimeConfig config{
        .pool_capacity = budget,
        .buffer_size = 4'096,
        .max_signal_blocks = 3,
        .discontinuity_capacity = 4,
        .gaps_per_discontinuity = 3,
    };
    CHECK(config.required_buffer_count() == 35);
    config.validate();
    auto invalid_config = config;
    invalid_config.buffer_size = 0;
    CHECK(throws<std::invalid_argument>([&] { invalid_config.validate(); }));
    constexpr std::array deadline_fields{
        &RealtimeConfig::source_stall_timeout,
        &RealtimeConfig::max_ingress_dwell,
        &RealtimeConfig::processor_execution_deadline,
        &RealtimeConfig::max_source_to_actuator_age,
        &RealtimeConfig::max_output_age,
        &RealtimeConfig::actuator_deadline,
        &RealtimeConfig::shutdown_deadline,
        &RealtimeConfig::watchdog_period,
    };
    for (const auto field : deadline_fields)
    {
        invalid_config = config;
        invalid_config.*field = std::chrono::nanoseconds{0};
        CHECK(throws<std::invalid_argument>([&] { invalid_config.validate(); }));
    }
    invalid_config = config;
    invalid_config.max_process_outputs = 0;
    CHECK(throws<std::invalid_argument>([&] { invalid_config.validate(); }));
    invalid_config = config;
    invalid_config.pool_capacity.critical_edge_capacity = 0;
    CHECK(throws<std::invalid_argument>([&] { invalid_config.validate(); }));
    invalid_config = config;
    invalid_config.pool_capacity.actuator_owned = 0;
    CHECK(throws<std::invalid_argument>([&] { invalid_config.validate(); }));
    invalid_config = config;
    invalid_config.fault_history_capacity = 0;
    CHECK(throws<std::invalid_argument>([&] { invalid_config.validate(); }));

    const PoolCapacityBudget overflow{
        .source_owned = std::numeric_limits<std::size_t>::max(),
        .reserve = 1,
    };
    CHECK(
        throws<std::overflow_error>([&] { static_cast<void>(overflow.required_buffer_count()); }));
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_valid_schema,
             test_schema_validation,
             test_fixed_sparse_spike_schema,
             test_block_byte_boundaries,
             test_stream_schema_validation,
             test_feature_schema_and_registries,
             test_feature_registry_validation,
             test_frame_headers,
             test_pool_capacity_budget,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
