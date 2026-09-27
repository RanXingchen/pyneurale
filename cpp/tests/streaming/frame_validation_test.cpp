/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/schema.h>

#include <array>
#include <cstdint>
#include <limits>

#include "check_returns.h"

namespace
{

using namespace neurale::streaming;

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{
        SignalSchema{1, SignalDType::float32, 2, 4, 8, {1'000, 1}, 10},
        SignalSchema{2, SignalDType::int16, 1, 4, 8, {500, 1}, 11},
    };
    return StreamSchema{7, signals};
}

[[nodiscard]] FrameLease make_valid_frame(FramePool& pool)
{
    FrameLease lease;
    if (pool.try_acquire(lease) != StreamStatus::ok)
    {
        return {};
    }
    lease.frame().header().schema_id = 7;
    auto blocks = lease.frame().block_storage();
    blocks[0] = SignalBlockHeader{
        .sample_idx_start = 100,
        .device_tick_start = 200,
        .payload_offset = 0,
        .payload_byte_count = 32,
        .signal_id = 1,
        .n_samples = 4,
    };
    blocks[1] = SignalBlockHeader{
        .sample_idx_start = 50,
        .device_tick_start = 75,
        .payload_offset = 32,
        .payload_byte_count = 8,
        .signal_id = 2,
        .n_samples = 4,
    };
    if (lease.frame().set_used_sizes(2, 40) != StreamStatus::ok)
    {
        return {};
    }
    return lease;
}

int test_valid_frame_and_used_sizes()
{
    auto schema = make_schema();
    FrameValidator validator{schema};
    FramePool pool{1, 64, 2};
    auto frame = make_valid_frame(pool);
    CHECK(frame);
    CHECK(validator.validate(frame.view()) == FrameValidationError::none);

    CHECK(frame.frame().set_used_sizes(3, 40) == StreamStatus::invalid_frame);
    CHECK(frame.frame().blocks().size() == 2);
    CHECK(frame.frame().payload().size() == 40);
    CHECK(frame.frame().set_used_sizes(2, 65) == StreamStatus::invalid_frame);
    CHECK(frame.frame().blocks().size() == 2);
    CHECK(frame.frame().payload().size() == 40);
    return 0;
}

int test_schema_and_block_identity_errors()
{
    auto schema = make_schema();
    FrameValidator validator{schema};
    FramePool pool{1, 64, 2};
    auto frame = make_valid_frame(pool);

    frame.frame().header().schema_id = 8;
    CHECK(validator.validate(frame.view()) == FrameValidationError::schema_changed);
    frame.frame().header().schema_id = 7;

    frame.frame().block_storage()[1].signal_id = 1;
    CHECK(validator.validate(frame.view()) == FrameValidationError::duplicate_signal);
    frame.frame().block_storage()[1].signal_id = 99;
    CHECK(validator.validate(frame.view()) == FrameValidationError::unknown_signal);
    return 0;
}

int test_sample_and_payload_errors()
{
    auto schema = make_schema();
    FrameValidator validator{schema};
    FramePool pool{1, 64, 2};
    auto frame = make_valid_frame(pool);
    auto& first = frame.frame().block_storage()[0];
    auto& second = frame.frame().block_storage()[1];

    first.n_samples = 0;
    CHECK(validator.validate(frame.view()) == FrameValidationError::invalid_sample_count);
    first.n_samples = 4;
    first.sample_idx_start = std::numeric_limits<std::uint64_t>::max() - 3;
    CHECK(validator.validate(frame.view()) == FrameValidationError::sample_idx_overflow);
    first.sample_idx_start = 100;

    first.clock_sync = ClockSyncSnapshot{
        .device_tick_rate = {1'000, 1},
        .clock_domain = 99,
        .generation = 1,
        .flags = ClockSyncFlags::synchronized,
    };
    CHECK(validator.validate(frame.view()) == FrameValidationError::invalid_clock_sync);
    first.clock_sync = {};

    first.payload_byte_count = 31;
    CHECK(validator.validate(frame.view()) == FrameValidationError::payload_size_mismatch);
    first.payload_byte_count = 32;
    second.payload_offset = 33;
    CHECK(validator.validate(frame.view()) == FrameValidationError::payload_layout_invalid);
    second.payload_offset = 31;
    CHECK(validator.validate(frame.view()) == FrameValidationError::payload_layout_invalid);
    second.payload_offset = 32;
    CHECK(frame.frame().set_used_sizes(2, 41) == StreamStatus::ok);
    CHECK(validator.validate(frame.view()) == FrameValidationError::payload_layout_invalid);
    return 0;
}

int test_feature_payload_shape_and_window_center_time()
{
    const std::array units{UnitDescriptor{1, "a.u.", "normalized"}};
    const FeatureSetDescriptor descriptor{
        .id = 8,
        .feature_names = {"left", "right"},
        .unit_ids = {1, 1},
        .source_stream_id = 3,
        .source_stream = "source",
        .window_length_ns = 200'000'000,
        .shift_ns = 100'000'000,
    };
    const std::array descriptors{descriptor};
    const SignalSchema feature{
        9,
        SignalDType::float32,
        2,
        2,
        4,
        {10, 1},
        2,
        SignalLayout::sample_major,
        DeviceTickTracking::unavailable,
        PhysicalUnit::unspecified,
        0,
        0,
        0,
        SignalKind::feature,
        8,
        ObservationTiming::regular,
    };
    const std::array signals{feature};
    const StreamSchema schema{12, signals, descriptors, units};
    FrameValidator validator{schema};
    FramePool pool{1, 32, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    lease.frame().header().schema_id = 12;
    lease.frame().block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = 5,
        .observation_time_start_ns = 1'250'000'000,
        .payload_offset = 0,
        .payload_byte_count = 16,
        .signal_id = 9,
        .n_samples = 2,
    };
    CHECK(lease.frame().set_used_sizes(1, 16) == StreamStatus::ok);
    CHECK(validator.validate(lease.view()) == FrameValidationError::none);
    CHECK(lease.view().blocks[0].observation_time_start_ns == 1'250'000'000);
    lease.frame().block_storage()[0].observation_time_start_ns =
        std::numeric_limits<HostTimeNs>::max();
    CHECK(validator.validate(lease.view()) == FrameValidationError::invalid_feature_timing);
    lease.frame().block_storage()[0].observation_time_start_ns = 1'250'000'000;
    lease.frame().block_storage()[0].payload_byte_count = 12;
    CHECK(lease.frame().set_used_sizes(1, 12) == StreamStatus::ok);
    CHECK(validator.validate(lease.view()) == FrameValidationError::feature_payload_shape_mismatch);
    return 0;
}

int test_fixed_sparse_spike_payload_validation()
{
    const SignalSchema spike{9,
                             SignalDType::float64,
                             2,
                             4,
                             4,
                             {0, 1},
                             2,
                             SignalLayout::sample_major,
                             DeviceTickTracking::unavailable,
                             PhysicalUnit::volts,
                             3,
                             0,
                             0,
                             SignalKind::spike,
                             0,
                             ObservationTiming::not_applicable,
                             64};
    const std::array signals{spike};
    const StreamSchema schema{12, signals};
    FrameValidator validator{schema};
    FramePool pool{1, 64, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    lease.frame().header().schema_id = 12;
    lease.frame().block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = 5,
        .payload_offset = 0,
        .payload_byte_count = 64,
        .signal_id = 9,
        .n_samples = 0,
    };
    CHECK(lease.frame().set_used_sizes(1, 64) == StreamStatus::ok);
    CHECK(validator.validate(lease.view()) == FrameValidationError::none);
    lease.frame().block_storage()[0].n_samples = 4;
    lease.frame().block_storage()[0].last_sample_idx = 8;
    CHECK(validator.validate(lease.view()) == FrameValidationError::none);
    lease.frame().block_storage()[0].sample_idx_start = std::numeric_limits<SampleIndex>::max();
    lease.frame().block_storage()[0].last_sample_idx = std::numeric_limits<SampleIndex>::max();
    CHECK(validator.validate(lease.view()) == FrameValidationError::none);
    lease.frame().block_storage()[0].sample_idx_start = 5;
    lease.frame().block_storage()[0].last_sample_idx = 4;
    CHECK(validator.validate(lease.view()) == FrameValidationError::invalid_sparse_idx_range);
    lease.frame().block_storage()[0].last_sample_idx = 8;
    lease.frame().block_storage()[0].n_samples = 5;
    CHECK(validator.validate(lease.view()) == FrameValidationError::invalid_sample_count);
    lease.frame().block_storage()[0].n_samples = 1;
    lease.frame().block_storage()[0].payload_byte_count = 63;
    CHECK(lease.frame().set_used_sizes(1, 63) == StreamStatus::ok);
    CHECK(validator.validate(lease.view()) == FrameValidationError::payload_size_mismatch);
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_valid_frame_and_used_sizes,
             test_schema_and_block_identity_errors,
             test_sample_and_payload_errors,
             test_feature_payload_shape_and_window_center_time,
             test_fixed_sparse_spike_payload_validation,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
