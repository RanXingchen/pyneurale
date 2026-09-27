/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_tracker.h"
#include "benchmark_report.h"

#include <neurale/runtime/runtime_info.h>
#include <neurale/streaming/actuator.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/clock.h>
#include <neurale/streaming/observer.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/realtime_config.h>
#include <neurale/streaming/runtime.h>
#include <neurale/streaming/safety.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/source.h>
#include <neurale/streaming/spsc_ring.h>
#include <neurale/streaming/stream_message.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <span>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

using namespace neurale::streaming;

namespace
{
using neurale::benchmark::emit_json_string;
using neurale::benchmark::platform_name;

using Clock = std::chrono::steady_clock;

[[nodiscard]] std::uint64_t now_ns() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

struct Options
{
    // Canonical real-time measurement workload. The benchmark's only job is
    // performance measurement, so the default IS the full workload; use
    // --iterations / --pipeline-iterations to shrink for ad-hoc local runs.
    std::size_t iterations{1'000'000};
    std::size_t pipeline_iterations{10'000};
    std::uint64_t deadline_ns{};
    unsigned producer_cpu{};
    unsigned consumer_cpu{1};
};

std::atomic<std::uint64_t> benchmark_sink{};

struct Summary
{
    std::uint64_t minimum{};
    std::uint64_t median{};
    std::uint64_t p99{};
    std::uint64_t p999{};
    std::uint64_t p9999{};
    std::uint64_t maximum{};
    std::uint64_t deadline_misses{};
    std::size_t n_samples{};
};

struct Result
{
    std::string_view benchmark;
    std::string_view metric;
    std::string_view unit{"ns"};
    Summary summary{};
    std::uint64_t allocations{};
    std::size_t payload_bytes{};
    std::size_t consumers{};
    std::size_t channels{};
    std::size_t samples_per_block{};
    bool affinity_requested{};
    bool affinity_applied{};
    int producer_cpu{-1};
    int consumer_cpu{-1};
    std::size_t warmups{};
    std::size_t repetitions{};
    std::string_view dtype{"opaque"};
    std::string_view provider{"native_cpu"};
    bool passed{true};
};

[[nodiscard]] std::size_t percentile_index(std::size_t size, std::uint64_t numerator) noexcept
{
    return size == 0 ? 0 : static_cast<std::size_t>(((size - 1) * numerator + 9'999) / 10'000);
}

[[nodiscard]] Summary summarize(std::vector<std::uint64_t>& samples, std::uint64_t deadline_ns)
{
    std::sort(samples.begin(), samples.end());
    if (samples.empty())
    {
        return {};
    }
    const auto misses = deadline_ns == 0
                            ? 0
                            : static_cast<std::uint64_t>(std::count_if(
                                  samples.begin(), samples.end(),
                                  [deadline_ns](auto value) { return value > deadline_ns; }));
    return Summary{
        .minimum = samples.front(),
        .median = samples[percentile_index(samples.size(), 5'000)],
        .p99 = samples[percentile_index(samples.size(), 9'900)],
        .p999 = samples[percentile_index(samples.size(), 9'990)],
        .p9999 = samples[percentile_index(samples.size(), 9'999)],
        .maximum = samples.back(),
        .deadline_misses = misses,
        .n_samples = samples.size(),
    };
}

void emit(const Result& result)
{
    static const auto build = neurale::runtime::build_info();
    static const auto cpu = neurale::runtime::cpu_info();
    static const auto threading = neurale::runtime::threading_info();
    const auto repetitions =
        result.repetitions == 0 ? result.summary.n_samples : result.repetitions;

    std::cout << "{\"schema_version\":1,\"benchmark\":\"" << result.benchmark << "\",\"metric\":\""
              << result.metric << "\",\"unit\":\"" << result.unit
              << "\",\"samples\":" << result.summary.n_samples
              << ",\"min\":" << result.summary.minimum << ",\"median\":" << result.summary.median
              << ",\"p99\":" << result.summary.p99 << ",\"p99_9\":" << result.summary.p999
              << ",\"p99_99\":" << result.summary.p9999 << ",\"max\":" << result.summary.maximum
              << ",\"deadline_misses\":" << result.summary.deadline_misses
              << ",\"allocations\":" << result.allocations << ",\"allocation_backend\":\""
              << neurale::benchmark::allocation_tracking_backend()
              << "\",\"payload_bytes\":" << result.payload_bytes
              << ",\"consumers\":" << result.consumers << ",\"channels\":" << result.channels
              << ",\"samples_per_block\":" << result.samples_per_block
              << ",\"warmups\":" << result.warmups << ",\"repetitions\":" << repetitions
              << ",\"dtype\":\"" << result.dtype << "\",\"provider\":\"" << result.provider
              << "\",\"platform\":\"" << platform_name() << "\",\"native_version\":";
    emit_json_string(std::cout, build.version);
    std::cout << ",\"compiler\":";
    emit_json_string(std::cout, build.compiler);
    std::cout << ",\"build_type\":";
    emit_json_string(std::cout, build.build_type);
    std::cout << ",\"cpu_math_backend\":";
    emit_json_string(std::cout, build.cpu_math_backend);
    std::cout << ",\"fft_backend\":";
    emit_json_string(std::cout, build.fft_backend);
    std::cout << ",\"cuda_compiled\":" << (build.cuda_compiled ? "true" : "false")
              << ",\"cuda_toolkit_version\":";
    if (build.cuda_toolkit_version.has_value())
    {
        emit_json_string(std::cout, *build.cuda_toolkit_version);
    }
    else
    {
        std::cout << "null";
    }
    std::cout << ",\"cpu_architecture\":";
    emit_json_string(std::cout, cpu.architecture);
    std::cout << ",\"cpu_model\":";
    if (cpu.model.has_value())
    {
        emit_json_string(std::cout, *cpu.model);
    }
    else
    {
        std::cout << "null";
    }
    std::cout << ",\"logical_cores\":" << cpu.logical_cores << ",\"threading_backend\":";
    emit_json_string(std::cout, threading.backend);
    std::cout << ",\"threads\":" << threading.num_threads
              << ",\"affinity_requested\":" << (result.affinity_requested ? "true" : "false")
              << ",\"affinity_applied\":" << (result.affinity_applied ? "true" : "false")
              << ",\"producer_cpu\":" << result.producer_cpu
              << ",\"consumer_cpu\":" << result.consumer_cpu
              << ",\"passed\":" << (result.passed ? "true" : "false") << "}\n";
}

[[nodiscard]] bool pin_current_thread(unsigned cpu) noexcept
{
#ifdef _WIN32
    if (cpu >= sizeof(DWORD_PTR) * 8)
    {
        return false;
    }
    return SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << cpu) != 0;
#elif defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    static_cast<void>(cpu);
    return false;
#endif
}

void wait_for(const std::atomic<bool>& value) noexcept
{
    while (!value.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
}

template <typename Predicate> [[nodiscard]] bool wait_until(Predicate predicate)
{
    for (std::size_t attempt = 0; attempt < 20'000'000; ++attempt)
    {
        if (predicate())
        {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

[[nodiscard]] bool benchmark_frame_pool(const Options& options)
{
    constexpr std::array<std::size_t, 3> payload_sizes{64, 4'096, 262'144};
    constexpr std::array<std::size_t, 2> consumer_counts{1, 4};
    bool passed = true;

    for (const auto payload_bytes : payload_sizes)
    {
        for (const auto consumers : consumer_counts)
        {
            std::uint64_t checksum{};
            FramePool pool{8, payload_bytes, 1};
            pool.prefault();
            FrameLease lease;
            for (std::size_t i = 0; i < 1'024; ++i)
            {
                if (pool.try_acquire(lease) != StreamStatus::ok)
                {
                    return false;
                }
                lease.frame().payload_storage().front() = std::byte{1};
                if (lease.frame().set_used_sizes(0, payload_bytes) != StreamStatus::ok ||
                    lease.reset() != StreamStatus::ok)
                {
                    return false;
                }
            }

            std::vector<std::uint64_t> samples(options.iterations);
            neurale::benchmark::AllocationScope allocations;
            for (std::size_t i = 0; i < options.iterations; ++i)
            {
                const auto started = now_ns();
                if (pool.try_acquire(lease) != StreamStatus::ok)
                {
                    passed = false;
                    break;
                }
                lease.frame().payload_storage().front() = static_cast<std::byte>(i & 0xffU);
                lease.frame().payload_storage()[payload_bytes - 1] = std::byte{1};
                if (lease.frame().set_used_sizes(0, payload_bytes) != StreamStatus::ok)
                {
                    passed = false;
                    break;
                }
                const auto view = lease.view();
                for (std::size_t consumer = 0; consumer < consumers; ++consumer)
                {
                    checksum += std::to_integer<std::uint8_t>(view.payload.front());
                    checksum += std::to_integer<std::uint8_t>(view.payload.back());
                }
                if (lease.reset() != StreamStatus::ok)
                {
                    passed = false;
                    break;
                }
                samples[i] = now_ns() - started;
            }
            allocations.stop();
            benchmark_sink.fetch_add(checksum, std::memory_order_relaxed);
            const auto allocation_count = allocations.count();
            const auto case_passed = passed && allocation_count == 0 && pool.outstanding() == 0;
            emit(Result{
                .benchmark = "frame_pool_acquire_publish_release",
                .metric = "operation_latency",
                .summary = summarize(samples, options.deadline_ns),
                .allocations = allocation_count,
                .payload_bytes = payload_bytes,
                .consumers = consumers,
                .warmups = 1'024,
                .dtype = "bytes",
                .passed = case_passed,
            });
            passed = passed && case_passed;
        }
    }
    return passed;
}

[[nodiscard]] bool allocation_tracker_self_test()
{
    neurale::benchmark::AllocationScope new_scope;
    void* ptr = ::operator new(64);
    ::operator delete(ptr);
    new_scope.stop();
    if (new_scope.count() == 0)
    {
        return false;
    }
#ifdef NEURALE_BENCHMARK_WRAP_MALLOC
    neurale::benchmark::AllocationScope malloc_scope;
    auto* allocated = std::malloc(64);
    std::free(allocated);
    malloc_scope.stop();
    return malloc_scope.count() != 0;
#else
    return true;
#endif
}

struct RingMessage
{
    std::uint64_t sequence{};
    std::uint64_t sent_at_ns{};
};

static_assert(std::is_nothrow_move_constructible_v<RingMessage>);
static_assert(std::is_nothrow_move_assignable_v<RingMessage>);

[[nodiscard]] bool benchmark_spsc_threaded(const Options& options, std::string_view name,
                                           std::size_t capacity, bool request_affinity)
{
    SpscRing<RingMessage> ring{capacity};
    ring.prefault();
    std::vector<std::uint64_t> samples(options.iterations);
    std::atomic<unsigned> ready{};
    std::atomic<bool> start{};
    std::atomic<bool> producer_done{};
    std::atomic<bool> consumer_done{};
    std::atomic<bool> affinity_producer{};
    std::atomic<bool> affinity_consumer{};
    std::atomic<bool> valid{true};

    std::thread producer{
        [&]
        {
            if (request_affinity)
            {
                affinity_producer.store(pin_current_thread(options.producer_cpu),
                                        std::memory_order_release);
            }
            ready.fetch_add(1, std::memory_order_release);
            wait_for(start);
            for (std::size_t i = 0; i < options.iterations; ++i)
            {
                RingMessage message{i, now_ns()};
                while (ring.try_push(std::move(message)) == StreamStatus::queue_overflow)
                {
                    std::this_thread::yield();
                }
            }
            producer_done.store(true, std::memory_order_release);
        }};
    std::thread consumer{[&]
                         {
                             if (request_affinity)
                             {
                                 affinity_consumer.store(pin_current_thread(options.consumer_cpu),
                                                         std::memory_order_release);
                             }
                             ready.fetch_add(1, std::memory_order_release);
                             wait_for(start);
                             RingMessage message;
                             for (std::size_t i = 0; i < options.iterations;)
                             {
                                 if (ring.try_pop(message) != StreamStatus::ok)
                                 {
                                     std::this_thread::yield();
                                     continue;
                                 }
                                 if (message.sequence != i)
                                 {
                                     valid.store(false, std::memory_order_relaxed);
                                 }
                                 samples[i] = now_ns() - message.sent_at_ns;
                                 ++i;
                             }
                             consumer_done.store(true, std::memory_order_release);
                         }};

    while (ready.load(std::memory_order_acquire) != 2)
    {
        std::this_thread::yield();
    }
    neurale::benchmark::AllocationScope allocations;
    const auto run_started = now_ns();
    start.store(true, std::memory_order_release);
    while (!producer_done.load(std::memory_order_acquire) ||
           !consumer_done.load(std::memory_order_acquire))
    {
        std::this_thread::yield();
    }
    const auto run_duration = now_ns() - run_started;
    allocations.stop();
    producer.join();
    consumer.join();

    const auto count = allocations.count();
    const auto affinity_applied =
        !request_affinity || (affinity_producer.load() && affinity_consumer.load());
    const auto passed = valid.load() && count == 0 && ring.empty();
    emit(Result{
        .benchmark = name,
        .metric = "message_latency",
        .summary = summarize(samples, options.deadline_ns),
        .allocations = count,
        .affinity_requested = request_affinity,
        .affinity_applied = affinity_applied,
        .producer_cpu = request_affinity ? static_cast<int>(options.producer_cpu) : -1,
        .consumer_cpu = request_affinity ? static_cast<int>(options.consumer_cpu) : -1,
        .dtype = "RingMessage",
        .passed = passed,
    });
    std::vector<std::uint64_t> throughput{run_duration == 0
                                              ? 0
                                              : static_cast<std::uint64_t>(options.iterations) *
                                                    1'000'000'000 / run_duration};
    emit(Result{
        .benchmark = name,
        .metric = "throughput",
        .unit = "messages_per_second",
        .summary = summarize(throughput, 0),
        .allocations = count,
        .affinity_requested = request_affinity,
        .affinity_applied = affinity_applied,
        .producer_cpu = request_affinity ? static_cast<int>(options.producer_cpu) : -1,
        .consumer_cpu = request_affinity ? static_cast<int>(options.consumer_cpu) : -1,
        .dtype = "RingMessage",
        .passed = passed,
    });
    return passed;
}

[[nodiscard]] bool benchmark_spsc_near_full(const Options& options)
{
    constexpr std::size_t capacity = 1'024;
    SpscRing<RingMessage> ring{capacity};
    ring.prefault();
    for (std::size_t i = 0; i < capacity - 1; ++i)
    {
        RingMessage message{i, 0};
        if (ring.try_push(std::move(message)) != StreamStatus::ok)
        {
            return false;
        }
    }
    std::vector<std::uint64_t> samples(options.iterations);
    RingMessage message;
    bool valid = true;
    neurale::benchmark::AllocationScope allocations;
    for (std::size_t i = 0; i < options.iterations; ++i)
    {
        const auto started = now_ns();
        valid = valid && ring.try_pop(message) == StreamStatus::ok;
        RingMessage replacement{capacity + i, started};
        valid = valid && ring.try_push(std::move(replacement)) == StreamStatus::ok;
        samples[i] = now_ns() - started;
    }
    allocations.stop();
    const auto count = allocations.count();
    const auto passed = valid && count == 0 && ring.approximate_size() == capacity - 1;
    emit(Result{
        .benchmark = "spsc_near_full",
        .metric = "pop_push_latency",
        .summary = summarize(samples, options.deadline_ns),
        .allocations = count,
        .dtype = "RingMessage",
        .passed = passed,
    });
    return passed;
}

[[nodiscard]] bool benchmark_spsc(const Options& options)
{
    auto passed = benchmark_spsc_threaded(options, "spsc_single_message", 1, false);
    passed = benchmark_spsc_threaded(options, "spsc_burst", 1'024, false) && passed;
    passed = benchmark_spsc_threaded(options, "spsc_cross_cpu", 1'024, true) && passed;
    return benchmark_spsc_near_full(options) && passed;
}

struct PipelineSamples
{
    explicit PipelineSamples(std::size_t count)
        : acquisition_interval(count), ingress_dwell(count), process_execution(count),
          source_to_output(count), source_to_actuator(count)
    {
    }

    std::vector<std::uint64_t> acquisition_interval;
    std::vector<std::uint64_t> ingress_dwell;
    std::vector<std::uint64_t> process_execution;
    std::vector<std::uint64_t> source_to_output;
    std::vector<std::uint64_t> source_to_actuator;
};

class BenchmarkSource final : public NativeFrameSource
{
  public:
    BenchmarkSource(std::size_t warmup, std::size_t measured, std::size_t payload_bytes,
                    std::uint32_t n_samples, PipelineSamples& samples,
                    const std::atomic<std::uint64_t>& consumed)
        : warmup_(warmup), total_(warmup + measured), payload_bytes_(payload_bytes),
          n_samples_(n_samples), samples_(samples), consumed_(consumed)
    {
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (next_ == warmup_ && !measurement_released_.load(std::memory_order_acquire))
        {
            warmup_ready_.store(true, std::memory_order_release);
            return StreamStatus::would_block;
        }
        if (next_ >= total_)
        {
            return StreamStatus::end_of_stream;
        }
        if (next_ > consumed_.load(std::memory_order_acquire) + 16)
        {
            return StreamStatus::would_block;
        }

        const auto received = now_ns();
        if (next_ > warmup_)
        {
            samples_.acquisition_interval[next_ - warmup_] = received - previous_received_;
        }
        previous_received_ = received;
        frame.header() = FrameHeader{
            .session_id = SessionId{1},
            .sequence = next_,
            .host_received_ns = received,
            .schema_id = SchemaId{1},
            .source_clock_domain = ClockDomainId{1},
        };
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = next_ * n_samples_,
            .device_tick_start = next_ * n_samples_,
            .payload_offset = 0,
            .payload_byte_count = payload_bytes_,
            .signal_id = SignalId{1},
            .n_samples = n_samples_,
        };
        frame.payload_storage().front() = std::byte{1};
        frame.payload_storage()[payload_bytes_ - 1] = std::byte{1};
        if (frame.set_used_sizes(1, payload_bytes_) != StreamStatus::ok)
        {
            return StreamStatus::invalid_frame;
        }
        ++next_;
        return StreamStatus::ok;
    }

    void cancel() noexcept override
    {
        measurement_released_.store(true);
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
    [[nodiscard]] bool warmup_ready() const noexcept
    {
        return warmup_ready_.load(std::memory_order_acquire);
    }
    void release_measurement() noexcept
    {
        measurement_released_.store(true, std::memory_order_release);
    }

  private:
    std::size_t warmup_{};
    std::size_t total_{};
    std::size_t payload_bytes_{};
    std::uint32_t n_samples_{};
    PipelineSamples& samples_;
    const std::atomic<std::uint64_t>& consumed_;
    std::size_t next_{};
    std::uint64_t previous_received_{};
    std::atomic<bool> warmup_ready_{};
    std::atomic<bool> measurement_released_{};
};

class BenchmarkProcessor final : public NativeFrameProcessor
{
  public:
    BenchmarkProcessor(std::size_t warmup, PipelineSamples& samples)
        : warmup_(warmup), samples_(samples)
    {
    }

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {StreamSchema{context.input_schema.id(), context.input_schema.signals()},
                StreamSchema{context.input_schema.id(), context.input_schema.signals()},
                1,
                0,
                true,
                ProcessorResourceBounds{.frame_pool_leases = 1}};
    }

    StreamStatus process(FrameBorrow& frame, FrameEmitter& output) noexcept override
    {
        const auto started = now_ns();
        const auto sequence = static_cast<std::size_t>(frame.header().sequence);
        if (sequence >= warmup_)
        {
            samples_.ingress_dwell[sequence - warmup_] = started - frame.header().host_received_ns;
        }
        const auto status = output.publish_input();
        if (sequence >= warmup_)
        {
            samples_.process_execution[sequence - warmup_] = now_ns() - started;
        }
        return status;
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

  private:
    std::size_t warmup_{};
    PipelineSamples& samples_;
};

class BenchmarkActuator final : public NativeActuator
{
  public:
    BenchmarkActuator(std::size_t warmup, PipelineSamples& samples,
                      std::size_t fail_at = std::numeric_limits<std::size_t>::max())
        : warmup_(warmup), samples_(samples), fail_at_(fail_at)
    {
    }

    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
    void cancel() noexcept override {}
    [[nodiscard]] const std::atomic<std::uint64_t>& count() const noexcept
    {
        return count_;
    }
    [[nodiscard]] std::uint64_t failed_at_ns() const noexcept
    {
        return failed_at_ns_.load(std::memory_order_acquire);
    }

  private:
    StreamStatus write(const ActuatorCommand& command) noexcept override
    {
        const auto applied_at = now_ns();
        const auto sequence = static_cast<std::size_t>(command.sequence);
        if (sequence >= warmup_ && sequence - warmup_ < samples_.source_to_output.size())
        {
            const auto idx = sequence - warmup_;
            samples_.source_to_output[idx] =
                command.generated_at_ns - command.payload.header.host_received_ns;
            samples_.source_to_actuator[idx] = applied_at - command.payload.header.host_received_ns;
        }
        count_.fetch_add(1, std::memory_order_release);
        if (sequence == fail_at_)
        {
            failed_at_ns_.store(applied_at, std::memory_order_release);
            return StreamStatus::actuator_failure;
        }
        return StreamStatus::ok;
    }

    std::size_t warmup_{};
    PipelineSamples& samples_;
    std::size_t fail_at_{};
    std::atomic<std::uint64_t> count_{};
    std::atomic<std::uint64_t> failed_at_ns_{};
};

class BlockedObserver final : public NativeObserver
{
  public:
    StreamStatus observe(FrameView) noexcept override
    {
        entered_.store(true, std::memory_order_release);
        released_.wait(false, std::memory_order_acquire);
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
    void cancel() noexcept override
    {
        release();
    }
    void release() noexcept
    {
        released_.store(true, std::memory_order_release);
        released_.notify_all();
    }
    [[nodiscard]] bool entered() const noexcept
    {
        return entered_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> entered_{};
    std::atomic<bool> released_{};
};

class BenchmarkSafety final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason) noexcept override
    {
        inhibited_at_ns_.store(now_ns(), std::memory_order_release);
        inhibited_.store(true, std::memory_order_release);
        return StreamStatus::ok;
    }
    StreamStatus release() noexcept override
    {
        inhibited_.store(false, std::memory_order_release);
        inhibited_at_ns_.store(0, std::memory_order_release);
        return StreamStatus::ok;
    }
    [[nodiscard]] bool inhibited() const noexcept
    {
        return inhibited_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t inhibited_at_ns() const noexcept
    {
        return inhibited_at_ns_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> inhibited_{true};
    std::atomic<std::uint64_t> inhibited_at_ns_{};
};

[[nodiscard]] StreamSchema make_schema(std::size_t channels, std::size_t samples_per_block)
{
    const SignalSchema signal{SignalId{1},
                              SignalDType::float64,
                              static_cast<std::uint32_t>(channels),
                              static_cast<std::uint32_t>(samples_per_block),
                              static_cast<std::uint32_t>(samples_per_block),
                              {1'000, 1},
                              ClockDomainId{1}};
    return StreamSchema{SchemaId{1}, std::span{&signal, 1}};
}

[[nodiscard]] RealtimeConfig make_config(std::size_t payload_bytes)
{
    RealtimeConfig config;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 64;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = 64;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.observer_edge_capacity = 64;
    config.pool_capacity.reserve = 4;
    config.buffer_size = payload_bytes;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 4;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.max_flush_outputs = 1;
    config.source_stall_timeout = RealtimeDuration{60'000'000'000};
    config.max_ingress_dwell = RealtimeDuration{60'000'000'000};
    config.processor_execution_deadline = RealtimeDuration{60'000'000'000};
    config.max_output_age = RealtimeDuration{60'000'000'000};
    config.actuator_deadline = RealtimeDuration{60'000'000'000};
    config.shutdown_deadline = RealtimeDuration{60'000'000'000};
    return config;
}

struct PipelineResult
{
    Summary source_to_actuator{};
    std::uint64_t allocations{};
    bool passed{};
};

[[nodiscard]] PipelineResult run_pipeline_case(const Options& options, std::size_t channels,
                                               std::size_t samples_per_block, bool blocked_observer)
{
    constexpr std::size_t warmup = 32;
    const auto measured = options.pipeline_iterations;
    const auto payload_bytes = channels * samples_per_block * sizeof(double);
    PipelineSamples samples{measured};
    BenchmarkActuator actuator{warmup, samples};
    BenchmarkSource source{warmup,        measured,
                           payload_bytes, static_cast<std::uint32_t>(samples_per_block),
                           samples,       actuator.count()};
    BenchmarkProcessor processor{warmup, samples};
    BlockedObserver observer;
    NativeStreamRunner runtime{make_schema(channels, samples_per_block), make_config(payload_bytes),
                               source, processor, actuator};
    if (blocked_observer)
    {
        const ObserverEdgeConfig edge{1, 8, 8, ObserverDropPolicy::drop_newest, false};
        if (runtime.add_observer(observer, edge) != StreamStatus::ok)
        {
            return {};
        }
    }
    if (runtime.prepare() != StreamStatus::ok || runtime.arm() != StreamStatus::ok ||
        runtime.start() != StreamStatus::ok)
    {
        return {};
    }
    if (!wait_until(
            [&]
            {
                return source.warmup_ready() &&
                       actuator.count().load(std::memory_order_acquire) == warmup &&
                       (!blocked_observer || observer.entered());
            }))
    {
        static_cast<void>(runtime.abort());
        static_cast<void>(runtime.join());
        return {};
    }

    neurale::benchmark::AllocationScope allocations;
    source.release_measurement();
    const auto completed = wait_until(
        [&] { return actuator.count().load(std::memory_order_acquire) == warmup + measured; });
    allocations.stop();
    if (blocked_observer)
    {
        observer.release();
    }
    const auto status = runtime.join();
    const auto allocation_count = allocations.count();
    if (!samples.acquisition_interval.empty())
    {
        samples.acquisition_interval.erase(samples.acquisition_interval.begin());
    }
    auto source_to_actuator = summarize(samples.source_to_actuator, options.deadline_ns);
    const auto passed = completed && status == StreamStatus::ok && allocation_count == 0 &&
                        runtime.outstanding_frames() == 0;

    emit(Result{.benchmark = blocked_observer ? "pipeline_blocked_observer" : "pipeline_identity",
                .metric = "acquisition_interval",
                .summary = summarize(samples.acquisition_interval, options.deadline_ns),
                .allocations = allocation_count,
                .payload_bytes = payload_bytes,
                .channels = channels,
                .samples_per_block = samples_per_block,
                .warmups = warmup,
                .dtype = "float64",
                .passed = passed});
    emit(Result{.benchmark = blocked_observer ? "pipeline_blocked_observer" : "pipeline_identity",
                .metric = "ingress_dwell",
                .summary = summarize(samples.ingress_dwell, options.deadline_ns),
                .allocations = allocation_count,
                .payload_bytes = payload_bytes,
                .channels = channels,
                .samples_per_block = samples_per_block,
                .warmups = warmup,
                .dtype = "float64",
                .passed = passed});
    emit(Result{.benchmark = blocked_observer ? "pipeline_blocked_observer" : "pipeline_identity",
                .metric = "process_execution",
                .summary = summarize(samples.process_execution, options.deadline_ns),
                .allocations = allocation_count,
                .payload_bytes = payload_bytes,
                .channels = channels,
                .samples_per_block = samples_per_block,
                .warmups = warmup,
                .dtype = "float64",
                .passed = passed});
    emit(Result{.benchmark = blocked_observer ? "pipeline_blocked_observer" : "pipeline_identity",
                .metric = "source_to_output",
                .summary = summarize(samples.source_to_output, options.deadline_ns),
                .allocations = allocation_count,
                .payload_bytes = payload_bytes,
                .channels = channels,
                .samples_per_block = samples_per_block,
                .warmups = warmup,
                .dtype = "float64",
                .passed = passed});
    emit(Result{.benchmark = blocked_observer ? "pipeline_blocked_observer" : "pipeline_identity",
                .metric = "source_to_actuator",
                .summary = source_to_actuator,
                .allocations = allocation_count,
                .payload_bytes = payload_bytes,
                .channels = channels,
                .samples_per_block = samples_per_block,
                .warmups = warmup,
                .dtype = "float64",
                .passed = passed});
    return PipelineResult{source_to_actuator, allocation_count, passed};
}

[[nodiscard]] bool benchmark_emergency_inhibit(const Options& options)
{
    constexpr std::size_t warmup = 32;
    constexpr std::size_t measured = 32;
    constexpr std::size_t fail_at = warmup + 16;
    PipelineSamples samples{measured};
    BenchmarkActuator actuator{warmup, samples, fail_at};
    BenchmarkSource source{warmup, measured, sizeof(double), 1, samples, actuator.count()};
    BenchmarkProcessor processor{warmup, samples};
    BenchmarkSafety safety;
    NativeStreamRunner runtime{make_schema(1, 1), make_config(sizeof(double)), source, processor,
                               actuator,          default_native_clock(),      safety};
    if (runtime.prepare() != StreamStatus::ok || runtime.arm() != StreamStatus::ok ||
        runtime.start() != StreamStatus::ok)
    {
        return false;
    }
    if (!wait_until(
            [&]
            {
                return source.warmup_ready() &&
                       actuator.count().load(std::memory_order_acquire) == warmup;
            }))
    {
        return false;
    }
    neurale::benchmark::AllocationScope allocations;
    source.release_measurement();
    const auto inhibited =
        wait_until([&] { return safety.inhibited() && actuator.failed_at_ns() != 0; });
    allocations.stop();
    const auto status = runtime.join();
    const auto failed_at = actuator.failed_at_ns();
    const auto inhibited_at = safety.inhibited_at_ns();
    std::vector<std::uint64_t> sample{inhibited_at >= failed_at ? inhibited_at - failed_at : 0};
    const auto count = allocations.count();
    const auto passed = inhibited && status == StreamStatus::actuator_failure &&
                        inhibited_at >= failed_at && count == 0;
    emit(Result{
        .benchmark = "pipeline_emergency_inhibit",
        .metric = "emergency_inhibit_latency",
        .summary = summarize(sample, options.deadline_ns),
        .allocations = count,
        .payload_bytes = sizeof(double),
        .channels = 1,
        .samples_per_block = 1,
        .warmups = warmup,
        .dtype = "float64",
        .passed = passed,
    });
    return passed;
}

[[nodiscard]] bool benchmark_pipeline(const Options& options)
{
    constexpr std::array<std::array<std::size_t, 2>, 3> shapes{
        std::array<std::size_t, 2>{8, 32},
        std::array<std::size_t, 2>{64, 64},
        std::array<std::size_t, 2>{256, 128},
    };
    bool passed = true;
    for (const auto shape : shapes)
    {
        const auto baseline = run_pipeline_case(options, shape[0], shape[1], false);
        passed = passed && baseline.passed;
    }

    const auto baseline = run_pipeline_case(options, 64, 64, false);
    const auto blocked = run_pipeline_case(options, 64, 64, true);
    const auto relative_limit = baseline.source_to_actuator.median * 3 + 50'000;
    const auto observer_isolated =
        baseline.passed && blocked.passed && blocked.source_to_actuator.median <= relative_limit;
    std::vector<std::uint64_t> ratio_sample{baseline.source_to_actuator.median == 0
                                                ? 0
                                                : blocked.source_to_actuator.median * 1'000 /
                                                      baseline.source_to_actuator.median};
    emit(Result{
        .benchmark = "pipeline_observer_isolation",
        .metric = "blocked_to_baseline_ratio_x1000",
        .unit = "ratio_x1000",
        .summary = summarize(ratio_sample, 0),
        .allocations = baseline.allocations + blocked.allocations,
        .channels = 64,
        .samples_per_block = 64,
        .warmups = 32,
        .dtype = "ratio",
        .passed = observer_isolated,
    });
    return benchmark_emergency_inhibit(options) && observer_isolated && passed;
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        if (argument == "--iterations" && i + 1 < argc)
        {
            options.iterations = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--pipeline-iterations" && i + 1 < argc)
        {
            options.pipeline_iterations = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--deadline-ns" && i + 1 < argc)
        {
            options.deadline_ns = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--producer-cpu" && i + 1 < argc)
        {
            options.producer_cpu = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (argument == "--consumer-cpu" && i + 1 < argc)
        {
            options.consumer_cpu = static_cast<unsigned>(std::strtoul(argv[++i], nullptr, 10));
        }
    }
    return options;
}

} // namespace

int main(int argc, char** argv)
{
    const auto options = parse_options(argc, argv);
    if (options.iterations == 0 || options.pipeline_iterations == 0)
    {
        std::cerr << "iteration counts must be positive\n";
        return 2;
    }
    if (options.producer_cpu == options.consumer_cpu)
    {
        std::cerr << "producer and consumer CPUs must differ\n";
        return 2;
    }
    if (!allocation_tracker_self_test())
    {
        std::cerr << "allocation tracker self-test failed\n";
        return 2;
    }

    auto passed = benchmark_frame_pool(options);
    passed = benchmark_spsc(options) && passed;
    passed = benchmark_pipeline(options) && passed;
    return passed ? 0 : 1;
}
