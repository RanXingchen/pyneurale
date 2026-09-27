/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/runtime.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include "check_returns.h"

namespace
{

using namespace neurale::streaming;

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{
        SignalSchema{1, SignalDType::float32, 2, 4, 4, {1'000, 1}, 2},
    };
    return StreamSchema{7, signals};
}

[[nodiscard]] RealtimeConfig make_config(RealtimeConfigMode mode)
{
    RealtimeConfig config;
    config.platform.mode = mode;
    config.platform.processing.set_name("processing");
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 4;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = 4;
    config.pool_capacity.actuator_owned = 1;
    config.buffer_size = 32;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 2;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.fault_history_capacity = 4;
    return config;
}

class EofSource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame&) noexcept override
    {
        return StreamStatus::end_of_stream;
    }
    void cancel() noexcept override {}
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

class NoOutputProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .output_schema =
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
            .max_process_outputs_per_input = 0,
            .max_flush_outputs = 0,
            .can_forward_input = false,
            .required_resources = ProcessorResourceBounds{.frame_pool_leases = 1},
        };
    }
    StreamStatus process(FrameBorrow&, FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush(FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

class NullConsumer final : public NativeFrameConsumer
{
  public:
    StreamStatus consume(FrameView) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

class UnsupportedPlatform final : public RealtimePlatform
{
  public:
    [[nodiscard]] RealtimePlatformCapabilities capabilities() const noexcept override
    {
        return {};
    }

    [[nodiscard]] RealtimeApplyResult
    configure_process_memory(const RealtimePlatformConfig& config) noexcept override
    {
        memory_calls.fetch_add(1, std::memory_order_relaxed);
        if (!fail_memory || !config.lock_memory)
        {
            return {};
        }
        const auto feature = feature_mask(RealtimeFeature::memory_lock);
        return {.requested = feature, .failed = feature, .native_error = 5};
    }

    [[nodiscard]] RealtimeApplyResult
    configure_current_thread(const RealtimeThreadConfig& config) noexcept override
    {
        thread_calls.fetch_add(1, std::memory_order_relaxed);
        const auto requested = config.requested_features();
        return {.requested = requested, .unsupported = requested};
    }

    bool fail_memory{};
    std::atomic<std::uint32_t> memory_calls{};
    std::atomic<std::uint32_t> thread_calls{};
};

[[nodiscard]] int test_configuration_validation_and_capabilities()
{
    RealtimeThreadConfig thread;
    thread.stack_size = 1024;
    thread.prefault_stack_bytes = 2048;
    try
    {
        thread.validate();
        return __LINE__;
    }
    catch (const std::invalid_argument&)
    {
    }

    try
    {
        thread.set_name("name-that-is-far-too-long");
        return __LINE__;
    }
    catch (const std::invalid_argument&)
    {
    }

    const auto capabilities = realtime_platform_capabilities();
    CHECK(!capabilities.hard_realtime_guaranteed);
#if defined(_WIN32)
    CHECK(capabilities.windows);
    CHECK(!capabilities.supports(RealtimeFeature::memory_lock));
#elif defined(__linux__)
    CHECK(capabilities.linux);
    CHECK(capabilities.supports(RealtimeFeature::memory_lock));
#endif
    return 0;
}

[[nodiscard]] int test_best_effort_records_warning_and_runs()
{
    auto schema = make_schema();
    auto config = make_config(RealtimeConfigMode::best_effort);
    EofSource source;
    NoOutputProcessor processor;
    NullConsumer consumer;
    UnsupportedPlatform platform;
    NativeStreamRunner runner(std::move(schema), config, source, processor, consumer,
                              default_native_clock(), default_safety_controller(), platform);

    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.run() == StreamStatus::ok);
    CHECK(runner.state() == RuntimeState::stopped);
    CHECK(!runner.realtime_configuration_status().ok());
    CHECK(runner.realtime_configuration_status().warning_count() == 1);
    CHECK(platform.thread_calls.load(std::memory_order_relaxed) == 4);
    CHECK(!runner.primary_fault().has_value());
    return 0;
}

[[nodiscard]] int test_strict_thread_failure_prevents_data_loop()
{
    auto schema = make_schema();
    auto config = make_config(RealtimeConfigMode::strict);
    EofSource source;
    NoOutputProcessor processor;
    NullConsumer consumer;
    UnsupportedPlatform platform;
    NativeStreamRunner runner(std::move(schema), config, source, processor, consumer,
                              default_native_clock(), default_safety_controller(), platform);

    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.start() == StreamStatus::realtime_configuration_failed);
    CHECK(runner.state() == RuntimeState::failed);
    CHECK(runner.primary_fault().has_value());
    CHECK(runner.primary_fault()->code == FaultCode::realtime_configuration);
    CHECK(runner.primary_fault()->status == StreamStatus::realtime_configuration_failed);
    CHECK(platform.thread_calls.load(std::memory_order_relaxed) == 4);
    CHECK(runner.outstanding_frames() == 0);
    CHECK(runner.reset() == StreamStatus::ok);
    CHECK(runner.state() == RuntimeState::created);
    return 0;
}

[[nodiscard]] int test_strict_memory_failure_creates_no_threads()
{
    auto schema = make_schema();
    auto config = make_config(RealtimeConfigMode::strict);
    config.platform.lock_memory = true;
    EofSource source;
    NoOutputProcessor processor;
    NullConsumer consumer;
    UnsupportedPlatform platform;
    platform.fail_memory = true;
    NativeStreamRunner runner(std::move(schema), config, source, processor, consumer,
                              default_native_clock(), default_safety_controller(), platform);

    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.start() == StreamStatus::realtime_configuration_failed);
    CHECK(platform.memory_calls.load(std::memory_order_relaxed) == 1);
    CHECK(platform.thread_calls.load(std::memory_order_relaxed) == 0);
    CHECK(runner.realtime_configuration_status().memory.failed ==
          feature_mask(RealtimeFeature::memory_lock));
    return 0;
}

} // namespace

int main()
{
    if (const auto result = test_configuration_validation_and_capabilities())
    {
        return result;
    }
    if (const auto result = test_best_effort_records_warning_and_runs())
    {
        return result;
    }
    if (const auto result = test_strict_thread_failure_prevents_data_loop())
    {
        return result;
    }
    return test_strict_memory_failure_creates_no_threads();
}
