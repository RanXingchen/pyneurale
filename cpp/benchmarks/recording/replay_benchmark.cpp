/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Native half of the record/replay characterization.
///
/// The public entry point is ``benchmarks/benchmark_record_replay.py``.  It
/// builds the recording plan, invokes this target's deterministic physical source,
/// measures the Python finalizer/recovery baseline in isolated processes, and
/// writes the aggregate JSONL artifact.  Keeping native callback and replay
/// timing here avoids a Python callback or GIL in either measured path.

#include "benchmark_report.h"
#include "byte_order.h"
#include "core/recorder.h"
#include "memory_spool_file.h"
#include "replay/source.h"
#include "sha256.h"
#include "spool_file.h"
#include "spool_layout.h"

#include <neurale/runtime/runtime_info.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/critical_observer.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <Windows.h>

// Its own block on purpose: Psapi.h requires Windows.h first, and clang-format
// sorts within a block but never across one.
#include <Psapi.h>
#else
#include <sys/resource.h>
#endif

namespace
{
using neurale::benchmark::bool_field;
using neurale::benchmark::emit_json_string;
using neurale::benchmark::summarize;
using neurale::benchmark::Summary;

using Clock = std::chrono::steady_clock;
using namespace neurale::recording;
using namespace neurale::streaming;

constexpr SchemaId kSchemaId = 613;
constexpr SessionId kReplaySessionId = 0x4D362D31332D5250ULL;
constexpr std::uint32_t kNeural = 1;
constexpr std::uint32_t kCursor = 2;
constexpr std::uint32_t kAudio = 3;
constexpr std::uint32_t kBandpower = 4;
constexpr std::uint64_t kFeatureSet = 61301;
constexpr std::uint64_t kMaxFramePayload =
    4ULL * 256ULL * 2ULL + 2ULL * 4ULL + 441ULL * 2ULL + 256ULL * 8ULL;

[[nodiscard]] std::uint64_t counter_word(std::uint64_t seed, std::uint64_t stream_id,
                                         std::uint64_t idx) noexcept
{
    auto value = seed ^ (stream_id * 0xD1B54A32D192ED03ULL) ^ (idx * 0x9E3779B97F4A7C15ULL);
    value ^= value >> 30U;
    value *= 0xBF58476D1CE4E5B9ULL;
    value ^= value >> 27U;
    value *= 0x94D049BB133111EBULL;
    value ^= value >> 31U;
    return value;
}

template <typename Value> [[nodiscard]] std::string sha256_hex(std::span<const Value> values)
{
    constexpr char digits[] = "0123456789abcdef";
    const auto digest = sha256(std::as_bytes(values));
    std::string result(digest.size() * 2, '0');
    for (std::size_t i = 0; i < digest.size(); ++i)
    {
        result[i * 2] = digits[digest[i] >> 4U];
        result[i * 2 + 1] = digits[digest[i] & 0x0FU];
    }
    return result;
}

[[nodiscard]] std::uint64_t now_ns() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

[[nodiscard]] bool read_bytes(const std::string& path, std::vector<std::byte>& out)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        return false;
    }
    const auto size = input.tellg();
    if (size < 0)
    {
        return false;
    }
    out.resize(static_cast<std::size_t>(size));
    input.seekg(0);
    return out.empty() || static_cast<bool>(input.read(reinterpret_cast<char*>(out.data()),
                                                       static_cast<std::streamsize>(out.size())));
}

[[nodiscard]] bool write_bytes(const std::string& path, std::span<const std::byte> bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    return output && (bytes.empty() ||
                      static_cast<bool>(output.write(reinterpret_cast<const char*>(bytes.data()),
                                                     static_cast<std::streamsize>(bytes.size()))));
}

void field(std::string_view name, std::string_view value, bool first = false)
{
    std::cout << (first ? "" : ",");
    emit_json_string(name);
    std::cout << ':';
    emit_json_string(value);
}

template <typename Value>
    requires std::is_arithmetic_v<Value>
void field(std::string_view name, Value value, bool first = false)
{
    std::cout << (first ? "" : ",");
    emit_json_string(name);
    std::cout << ':' << value;
}

void emit_summary(std::string_view prefix, const Summary& summary)
{
    field(std::string{prefix} + "_samples", summary.samples);
    field(std::string{prefix} + "_min_ns", summary.minimum);
    field(std::string{prefix} + "_median_ns", summary.median);
    field(std::string{prefix} + "_p95_ns", summary.p95);
    field(std::string{prefix} + "_p99_ns", summary.p99);
    field(std::string{prefix} + "_max_ns", summary.maximum);
}

struct ProcessCounters
{
    double cpu_seconds{};
    std::uint64_t peak_rss_bytes{};
    std::uint64_t read_bytes{};
    std::uint64_t write_bytes{};
    bool io_supported{};
};

[[nodiscard]] ProcessCounters process_counters() noexcept
{
    ProcessCounters result{};
#if defined(_WIN32)
    FILETIME create{}, exit{}, kernel{}, user{};
    if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user))
    {
        ULARGE_INTEGER kernel_value{}, user_value{};
        kernel_value.LowPart = kernel.dwLowDateTime;
        kernel_value.HighPart = kernel.dwHighDateTime;
        user_value.LowPart = user.dwLowDateTime;
        user_value.HighPart = user.dwHighDateTime;
        result.cpu_seconds =
            static_cast<double>(kernel_value.QuadPart + user_value.QuadPart) * 1e-7;
    }
    PROCESS_MEMORY_COUNTERS_EX memory{};
    memory.cb = sizeof(memory);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory), sizeof(memory)))
    {
        result.peak_rss_bytes = memory.PeakWorkingSetSize;
    }
    IO_COUNTERS io{};
    if (GetProcessIoCounters(GetCurrentProcess(), &io))
    {
        result.read_bytes = io.ReadTransferCount;
        result.write_bytes = io.WriteTransferCount;
        result.io_supported = true;
    }
#else
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0)
    {
        result.cpu_seconds =
            static_cast<double>(usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) +
            static_cast<double>(usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1e6;
#if defined(__APPLE__)
        result.peak_rss_bytes = static_cast<std::uint64_t>(usage.ru_maxrss);
#else
        result.peak_rss_bytes = static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
    }
#if defined(__linux__)
    std::ifstream input("/proc/self/io");
    std::string key;
    std::uint64_t value = 0;
    while (input >> key >> value)
    {
        if (key == "read_bytes:")
        {
            result.read_bytes = value;
        }
        else if (key == "write_bytes:")
        {
            result.write_bytes = value;
        }
    }
    result.io_supported = static_cast<bool>(input.eof());
#endif
#endif
    return result;
}

void emit_process_delta(const ProcessCounters& before, const ProcessCounters& after)
{
    field("process_cpu_seconds", std::max(0.0, after.cpu_seconds - before.cpu_seconds));
    field("peak_resident_memory_bytes", after.peak_rss_bytes);
    bool_field("process_io_supported", before.io_supported && after.io_supported);
    field("process_read_bytes",
          after.read_bytes >= before.read_bytes ? after.read_bytes - before.read_bytes : 0);
    field("process_write_bytes",
          after.write_bytes >= before.write_bytes ? after.write_bytes - before.write_bytes : 0);
}

void emit_build()
{
    const auto build = neurale::runtime::build_info();
    const auto cpu = neurale::runtime::cpu_info();
    const auto threading = neurale::runtime::threading_info();
    field("native_version", build.version);
    field("native_abi_version", build.abi_version);
    field("compiler", build.compiler);
    field("build_type", build.build_type);
    field("cpu_math_backend", build.cpu_math_backend);
    field("fft_backend", build.fft_backend);
    bool_field("cuda_compiled", build.cuda_compiled);
    field("cpu_architecture", cpu.architecture);
    field("cpu_vendor", cpu.vendor.value_or(""));
    field("cpu_model", cpu.model.value_or(""));
    field("logical_cores", cpu.logical_cores);
    field("threading_backend", threading.backend);
    field("native_threads", threading.num_threads);
}

class BenchmarkUnixClock final : public UnixClock
{
  public:
    [[nodiscard]] std::uint64_t unix_nanos() noexcept override
    {
        return next_.fetch_add(1'000, std::memory_order_relaxed);
    }

  private:
    std::atomic<std::uint64_t> next_{1'786'384'800'000'000'000ULL};
};

/// Measure the worker-owned store without changing the store or writer.
/// Storage for timings is fixed before prepare, so this wrapper adds no worker
/// allocation.  Checkpoint transactions are recognized from the native spool's
/// fixed record-kind field.
class TimedSpool final : public SpoolFile
{
  public:
    TimedSpool(SpoolFile& inner, std::size_t max_calls)
        : inner_(inner), append_ns_(max_calls), checkpoint_ns_(max_calls), sync_ns_(max_calls)
    {
    }

    SpoolIoResult append(std::span<const std::byte> bytes) noexcept override
    {
        const auto start = now_ns();
        const auto result = inner_.append(bytes);
        const auto elapsed = now_ns() - start;
        store(append_ns_, append_count_, append_dropped_, elapsed);
        if (is_checkpoint(bytes))
        {
            store(checkpoint_ns_, checkpoint_count_, checkpoint_dropped_, elapsed);
        }
        return result;
    }

    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override
    {
        return inner_.read_at(offset, out);
    }

    SpoolIoResult sync() noexcept override
    {
        const auto start = now_ns();
        const auto result = inner_.sync();
        store(sync_ns_, sync_count_, sync_dropped_, now_ns() - start);
        return result;
    }

    SpoolIoResult truncate(std::uint64_t bytes) noexcept override
    {
        return inner_.truncate(bytes);
    }

    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        return inner_.size();
    }
    void request_cancel() noexcept override
    {
        inner_.request_cancel();
    }
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return inner_.supports_bounded_cancel();
    }
    [[nodiscard]] bool supports_durability_policy(DurabilityPolicy policy) const noexcept override
    {
        return inner_.supports_durability_policy(policy);
    }

    [[nodiscard]] Summary append_summary() const
    {
        return summarize(std::span<const std::uint64_t>{append_ns_}.first(append_count_));
    }
    [[nodiscard]] Summary checkpoint_summary() const
    {
        return summarize(std::span<const std::uint64_t>{checkpoint_ns_}.first(checkpoint_count_));
    }
    [[nodiscard]] Summary sync_summary() const
    {
        return summarize(std::span<const std::uint64_t>{sync_ns_}.first(sync_count_));
    }
    [[nodiscard]] std::size_t append_dropped() const noexcept
    {
        return append_dropped_;
    }
    [[nodiscard]] std::size_t checkpoint_dropped() const noexcept
    {
        return checkpoint_dropped_;
    }
    [[nodiscard]] std::size_t sync_dropped() const noexcept
    {
        return sync_dropped_;
    }

  private:
    static void store(std::vector<std::uint64_t>& values, std::size_t& count, std::size_t& dropped,
                      std::uint64_t value) noexcept
    {
        if (count < values.size())
        {
            values[count++] = value;
            return;
        }
        ++dropped;
    }

    [[nodiscard]] static bool is_checkpoint(std::span<const std::byte> bytes) noexcept
    {
        if (bytes.size() < kTransactionHeaderBytes + kRecordHeaderBytes ||
            std::memcmp(bytes.data(), kTransactionBeginMagic, sizeof(kTransactionBeginMagic)) != 0)
        {
            return false;
        }
        const auto records = load_u32le(bytes, transaction_header::kRecordCount);
        return records == 1 &&
               load_u16le(bytes, kTransactionHeaderBytes + record_header::kRecordKind) ==
                   static_cast<std::uint16_t>(RecordKind::checkpoint);
    }

    SpoolFile& inner_;
    std::vector<std::uint64_t> append_ns_;
    std::vector<std::uint64_t> checkpoint_ns_;
    std::vector<std::uint64_t> sync_ns_;
    std::size_t append_count_{};
    std::size_t checkpoint_count_{};
    std::size_t sync_count_{};
    std::size_t append_dropped_{};
    std::size_t checkpoint_dropped_{};
    std::size_t sync_dropped_{};
};

class IdentityProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        const auto make_schema = [&]()
        {
            return StreamSchema{context.input_schema.id(), context.input_schema.signals(),
                                context.input_schema.feature_sets().descriptors(),
                                context.input_schema.units().units()};
        };
        return {make_schema(),
                make_schema(),
                1,
                0,
                true,
                ProcessorResourceBounds{.frame_pool_leases = 1}};
    }
    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        return output.publish_input();
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

class CountingConsumer final : public NativeFrameConsumer
{
  public:
    StreamStatus consume(FrameView frame) noexcept override
    {
        frames_.fetch_add(1, std::memory_order_relaxed);
        bytes_.fetch_add(frame.payload.size(), std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        discontinuities_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    StreamStatus reset() noexcept override
    {
        frames_.store(0, std::memory_order_relaxed);
        discontinuities_.store(0, std::memory_order_relaxed);
        bytes_.store(0, std::memory_order_relaxed);
        return StreamStatus::ok;
    }
    [[nodiscard]] std::uint64_t frames() const noexcept
    {
        return frames_.load();
    }
    [[nodiscard]] std::uint64_t bytes() const noexcept
    {
        return bytes_.load();
    }

  private:
    std::atomic<std::uint64_t> frames_{};
    std::atomic<std::uint64_t> discontinuities_{};
    std::atomic<std::uint64_t> bytes_{};
};

/// Benchmark-only decorator.  It observes the exact generic critical-recorder
/// callback boundary and delegates every lifecycle decision unchanged.
class TimedCriticalObserver final : public NativeCriticalObserver
{
  public:
    TimedCriticalObserver(NativeCriticalObserver& inner, std::size_t max_items)
        : inner_(inner), callback_ns_(max_items)
    {
    }

    StreamStatus ready_for_runtime() noexcept override
    {
        return inner_.ready_for_runtime();
    }
    StreamStatus start_observing() noexcept override
    {
        return inner_.start_observing();
    }
    StreamStatus health() const noexcept override
    {
        return inner_.health();
    }
    StreamStatus observe_accepted(FrameView frame, RuntimeAcceptance acceptance) noexcept override
    {
        const auto start = now_ns();
        const auto result = inner_.observe_accepted(frame, acceptance);
        store(now_ns() - start);
        return result;
    }
    StreamStatus handle_accepted_discontinuity(const Discontinuity& discontinuity,
                                               RuntimeAcceptance acceptance) noexcept override
    {
        const auto start = now_ns();
        const auto result = inner_.handle_accepted_discontinuity(discontinuity, acceptance);
        store(now_ns() - start);
        return result;
    }
    void note_rejected_before_acceptance(RejectedMessage message) noexcept override
    {
        inner_.note_rejected_before_acceptance(message);
    }
    void publish_primary_fault(const FaultRecord& fault) noexcept override
    {
        inner_.publish_primary_fault(fault);
    }
    StreamStatus drain(RuntimeTerminalNotice terminal) noexcept override
    {
        const auto start = now_ns();
        const auto result = inner_.drain(terminal);
        drain_ns_ = now_ns() - start;
        return result;
    }
    void cancel() noexcept override
    {
        inner_.cancel();
    }
    StreamStatus reset() noexcept override
    {
        return inner_.reset();
    }
    [[nodiscard]] Summary callback_summary() const
    {
        return summarize(std::span<const std::uint64_t>{callback_ns_}.first(callback_count_));
    }
    [[nodiscard]] std::uint64_t drain_ns() const noexcept
    {
        return drain_ns_;
    }

  private:
    void store(std::uint64_t value) noexcept
    {
        if (callback_count_ < callback_ns_.size())
        {
            callback_ns_[callback_count_++] = value;
        }
    }
    NativeCriticalObserver& inner_;
    std::vector<std::uint64_t> callback_ns_;
    std::size_t callback_count_{};
    std::uint64_t drain_ns_{};
};

[[nodiscard]] StreamSchema physical_schema()
{
    std::vector<SignalSchema> signals;
    signals.emplace_back(kNeural, SignalDType::int16, 256, 4, 4, RationalRate{4'000, 1}, 1,
                         SignalLayout::sample_major, DeviceTickTracking::sample_counter,
                         PhysicalUnit::volts, 0, 0, 0, SignalKind::sampled, 0,
                         ObservationTiming::not_applicable, 2'048);
    signals.emplace_back(kCursor, SignalDType::float32, 2, 1, 1, RationalRate{100, 1}, 2,
                         SignalLayout::sample_major, DeviceTickTracking::sample_counter,
                         PhysicalUnit::unspecified, 0, 0, 0, SignalKind::sampled, 0,
                         ObservationTiming::not_applicable, 8);
    signals.emplace_back(kAudio, SignalDType::int16, 1, 441, 441, RationalRate{44'100, 1}, 3,
                         SignalLayout::sample_major, DeviceTickTracking::sample_counter,
                         PhysicalUnit::unspecified, 0, 0, 0, SignalKind::sampled, 0,
                         ObservationTiming::not_applicable, 882);
    signals.emplace_back(kBandpower, SignalDType::float64, 256, 1, 1, RationalRate{100, 1}, 1,
                         SignalLayout::sample_major, DeviceTickTracking::sample_counter,
                         PhysicalUnit::unspecified, 0, 0, 0, SignalKind::feature, kFeatureSet,
                         ObservationTiming::regular, 2'048);
    std::vector<std::string> names;
    names.reserve(256);
    for (std::size_t channel = 0; channel < 256; ++channel)
    {
        names.push_back("bandpower:channel-" + std::to_string(channel));
    }
    FeatureSetDescriptor descriptor{.id = kFeatureSet,
                                    .feature_names = std::move(names),
                                    .unit_ids = std::vector<UnitId>(256, 1),
                                    .source_stream_id = kNeural,
                                    .source_stream = "neural",
                                    .algorithm_name = "multitaper-bandpower",
                                    .algorithm_version = "1",
                                    .window_length_ns = 100'000'000,
                                    .shift_ns = 10'000'000,
                                    .timestamp_reference =
                                        FeatureTimestampReference::window_center};
    const std::array descriptors{descriptor};
    const std::array units{UnitDescriptor{1, "V^2", "volt squared"}};
    return StreamSchema{kSchemaId, signals, descriptors, units};
}

class PhysicalSource final : public NativeFrameSource
{
  public:
    explicit PhysicalSource(std::size_t frames, std::uint64_t seed)
        : n_frames_(frames), neural_(4'000 * 256), cursor_(100 * 2), audio_(44'100),
          feature_(100 * 256)
    {
        for (std::size_t sample = 0; sample < 4'000; ++sample)
            for (std::size_t channel = 0; channel < 256; ++channel)
            {
                const auto idx = sample * 256 + channel;
                neural_[idx] = static_cast<std::int16_t>(
                    static_cast<std::int32_t>(counter_word(seed, kNeural, idx) & 0xFFFULL) - 2'048);
            }
        for (std::size_t sample = 0; sample < 100; ++sample)
        {
            for (std::size_t dim = 0; dim < 2; ++dim)
            {
                const auto idx = sample * 2 + dim;
                const auto integer =
                    static_cast<std::int32_t>(counter_word(seed, kCursor, idx) & 0xFFFFULL) -
                    32'768;
                cursor_[idx] = std::ldexp(static_cast<float>(integer), -17);
            }
            for (std::size_t channel = 0; channel < 256; ++channel)
            {
                const auto idx = sample * 256 + channel;
                const auto integer = 1'024ULL + (counter_word(seed, kBandpower, idx) & 0xFFFFULL);
                feature_[idx] = std::ldexp(static_cast<double>(integer), -36);
            }
        }
        for (std::size_t sample = 0; sample < 44'100; ++sample)
        {
            audio_[sample] = static_cast<std::int16_t>(
                static_cast<std::int32_t>(counter_word(seed, kAudio, sample) & 0x3FFFULL) - 8'192);
        }
    }

    [[nodiscard]] std::size_t item_count() const noexcept
    {
        return n_frames_;
    }
    [[nodiscard]] std::string neural_sha256() const
    {
        return sha256_hex(std::span{neural_});
    }
    [[nodiscard]] std::string cursor_sha256() const
    {
        return sha256_hex(std::span{cursor_});
    }
    [[nodiscard]] std::string audio_sha256() const
    {
        return sha256_hex(std::span{audio_});
    }
    [[nodiscard]] std::string bandpower_sha256() const
    {
        return sha256_hex(std::span{feature_});
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
            return StreamStatus::stopped;
        if (idx_ == n_frames_)
            return StreamStatus::end_of_stream;
        if (!started_.has_value())
            started_ = Clock::now();
        else
            std::this_thread::sleep_until(*started_ + std::chrono::milliseconds(idx_));

        const auto neural_start = idx_ * 4;
        frame.header() =
            FrameHeader{.session_id = kReplaySessionId,
                        .sequence = idx_,
                        .host_received_ns = idx_ * 1'000'000ULL,
                        .source_tick = neural_start,
                        .valid_until_ns = std::numeric_limits<HostTimeNs>::max(),
                        .schema_id = kSchemaId,
                        .source_clock_domain = 1,
                        .flags = FrameFlags::source_tick | FrameFlags::source_received};
        auto offset = std::size_t{0};
        auto n_blocks = std::size_t{0};
        const auto write_block = [&](SignalId signal_id, SampleIndex start, std::uint32_t samples,
                                     HostTimeNs observation_ns, const void* data, std::size_t bytes,
                                     ClockDomainId clock) noexcept
        {
            frame.block_storage()[n_blocks++] = SignalBlockHeader{
                .sample_idx_start = start,
                .device_tick_start = start,
                .observation_time_start_ns = observation_ns,
                .payload_offset = offset,
                .payload_byte_count = bytes,
                .signal_id = signal_id,
                .n_samples = samples,
                .clock_sync = ClockSyncSnapshot{.clock_domain = clock},
            };
            std::memcpy(frame.payload_storage().data() + offset, data, bytes);
            offset += bytes;
        };

        const auto neural_row = neural_start % 4'000;
        write_block(kNeural, neural_start, 4, neural_start * 1'000'000'000ULL / 4'000,
                    neural_.data() + neural_row * 256, 4 * 256 * sizeof(std::int16_t), 1);
        if (idx_ % 10 == 0)
        {
            const auto observation = idx_ / 10;
            const auto cursor_row = observation % 100;
            write_block(kCursor, observation, 1, observation * 10'000'000ULL,
                        cursor_.data() + cursor_row * 2, 2 * sizeof(float), 2);
            const auto audio_start = observation * 441;
            const auto audio_row = audio_start % 44'100;
            write_block(kAudio, audio_start, 441, audio_start * 1'000'000'000ULL / 44'100,
                        audio_.data() + audio_row, 441 * sizeof(std::int16_t), 3);
            write_block(kBandpower, observation, 1, 50'000'000ULL + observation * 10'000'000ULL,
                        feature_.data() + cursor_row * 256, 256 * sizeof(double), 1);
        }
        const auto status = frame.set_used_sizes(n_blocks, offset);
        if (status == StreamStatus::ok)
            ++idx_;
        return status;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }
    StreamStatus reset() noexcept override
    {
        idx_ = 0;
        started_.reset();
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::size_t n_frames_{};
    std::size_t idx_{};
    std::optional<Clock::time_point> started_;
    std::atomic<bool> cancelled_{};
    std::vector<std::int16_t> neural_;
    std::vector<float> cursor_;
    std::vector<std::int16_t> audio_;
    std::vector<double> feature_;
};

struct Options
{
    std::string operation;
    std::string image;
    std::string plan;
    std::string spool;
    std::string backend{"memory"};
    std::string pacing{"as_fast_as_possible"};
    std::size_t spool_capacity{256ULL << 20U};
    std::size_t queue_capacity{8'192};
    std::size_t checkpoint_interval{32};
    std::size_t frames{60'000};
    std::uint64_t seed{613};
    std::uint64_t worker_poll_ns{50'000};
    std::uint64_t drain_timeout_ns{30'000'000'000ULL};
    double speed_factor{1.0};
};

[[nodiscard]] bool parse(int argc, char** argv, Options& options)
{
    if (argc < 2)
    {
        return false;
    }
    options.operation = argv[1];
    for (int i = 2; i < argc; ++i)
    {
        const std::string_view argument = argv[i];
        if (i + 1 >= argc)
        {
            return false;
        }
        const std::string value = argv[++i];
        try
        {
            if (argument == "--image")
                options.image = value;
            else if (argument == "--plan")
                options.plan = value;
            else if (argument == "--spool")
                options.spool = value;
            else if (argument == "--backend")
                options.backend = value;
            else if (argument == "--pacing")
                options.pacing = value;
            else if (argument == "--spool-capacity")
                options.spool_capacity = std::stoull(value);
            else if (argument == "--queue-capacity")
                options.queue_capacity = std::stoull(value);
            else if (argument == "--checkpoint-interval")
                options.checkpoint_interval = std::stoull(value);
            else if (argument == "--frames")
                options.frames = std::stoull(value);
            else if (argument == "--seed")
                options.seed = std::stoull(value);
            else if (argument == "--worker-poll-ns")
                options.worker_poll_ns = std::stoull(value);
            else if (argument == "--drain-timeout-ns")
                options.drain_timeout_ns = std::stoull(value);
            else if (argument == "--speed-factor")
                options.speed_factor = std::stod(value);
            else
                return false;
        }
        catch (const std::exception&)
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] ReplayPacing pacing_value(std::string_view value)
{
    if (value == "as_fast_as_possible")
        return ReplayPacing::as_fast_as_possible;
    if (value == "recorded")
        return ReplayPacing::recorded;
    if (value == "step")
        return ReplayPacing::step;
    throw std::invalid_argument("unknown replay pacing");
}

[[nodiscard]] std::size_t payload_bound(const ReplayImageFile& image)
{
    std::uint64_t largest = 1;
    for (std::size_t i = 0; i < image.item_count(); ++i)
        largest = std::max(largest, image.item(i).payload_byte_count);
    return static_cast<std::size_t>(largest);
}

[[nodiscard]] std::size_t child_bound(const ReplayImageFile& image, ReplayItemKind kind)
{
    std::uint64_t largest = 1;
    for (std::size_t i = 0; i < image.item_count(); ++i)
    {
        const auto item = image.item(i);
        if (item.kind == kind)
            largest = std::max<std::uint64_t>(largest, item.n_children);
    }
    return static_cast<std::size_t>(largest);
}

[[nodiscard]] int replay(const Options& options)
{
    NativeReplaySource source;
    const auto opened = source.open(options.image.c_str());
    if (opened != ReplayImageStatus::ok)
    {
        std::cerr << "replay image open failed\n";
        return 2;
    }
    ReplaySourceConfig config;
    try
    {
        config.pacing = pacing_value(options.pacing);
    }
    catch (const std::exception&)
    {
        return 2;
    }
    config.speed_factor = options.speed_factor;
    config.blocking = true;
    config.wake_latency_ns = 1'000'000;
    if (source.prepare(config) != ReplayConfigStatus::ok)
    {
        std::cerr << "replay source prepare failed\n";
        return 3;
    }

    FramePool frames{2, payload_bound(source.image()),
                     child_bound(source.image(), ReplayItemKind::frame)};
    DiscontinuityPool discontinuities{2,
                                      child_bound(source.image(), ReplayItemKind::discontinuity)};
    frames.prefault();
    discontinuities.prefault();
    std::vector<std::uint64_t> read_ns(source.item_count());
    std::vector<std::uint64_t> step_ns(source.item_count());
    std::size_t emitted = 0;
    std::size_t n_steps = 0;
    const auto counters_before = process_counters();
    const auto started = now_ns();
    StreamStatus terminal = StreamStatus::invalid_state;
    for (;;)
    {
        FrameLease frame;
        DiscontinuityLease discontinuity;
        if (frames.try_acquire(frame) != StreamStatus::ok ||
            discontinuities.try_acquire(discontinuity) != StreamStatus::ok)
        {
            terminal = StreamStatus::buffer_exhausted;
            break;
        }
        std::uint64_t step_started = 0;
        if (config.pacing == ReplayPacing::step)
        {
            step_started = now_ns();
            if (source.advance(1) != 1)
            {
                terminal = StreamStatus::invalid_state;
                break;
            }
        }
        const auto read_started = now_ns();
        const auto status = source.read_message(frame.frame(), discontinuity);
        const auto read_elapsed = now_ns() - read_started;
        if (status == StreamStatus::ok || status == StreamStatus::discontinuity)
        {
            read_ns[emitted++] = read_elapsed;
            if (config.pacing == ReplayPacing::step)
                step_ns[n_steps++] = now_ns() - step_started;
            continue;
        }
        terminal = status;
        break;
    }
    const auto elapsed = now_ns() - started;
    const auto counters_after = process_counters();
    const auto stats = source.stats();
    const auto summary = summarize(std::span<const std::uint64_t>{read_ns}.first(emitted));
    const auto step_summary = summarize(std::span<const std::uint64_t>{step_ns}.first(n_steps));
    const auto ok = terminal == StreamStatus::end_of_stream && emitted == source.item_count();

    std::cout << '{';
    field("schema_version", 1, true);
    field("stage", "native_replay");
    field("status", ok ? "ok" : "failed");
    field("mode", source.image().string(source.image().summary().mode));
    field("pacing", options.pacing);
    field("speed_factor", options.speed_factor);
    field("elapsed_ns", elapsed);
    field("items", emitted);
    field("frames", stats.frames_emitted);
    field("discontinuities", stats.discontinuities_emitted);
    field("payload_bytes", source.image().summary().payload_byte_count);
    field("throughput_items_per_second",
          elapsed == 0 ? 0.0 : static_cast<double>(emitted) * 1e9 / elapsed);
    field("throughput_payload_bytes_per_second",
          elapsed == 0
              ? 0.0
              : static_cast<double>(source.image().summary().payload_byte_count) * 1e9 / elapsed);
    emit_summary("read_latency", summary);
    emit_summary("permit_to_emit_latency", step_summary);
    field("late_item_count", stats.late_item_count);
    field("max_lateness_ns", stats.max_lateness_ns);
    field("permits_granted", stats.permits_granted);
    field("permits_consumed", stats.permits_consumed);
    emit_process_delta(counters_before, counters_after);
    emit_build();
    std::cout << "}\n";
    return ok ? 0 : 4;
}

[[nodiscard]] int record(const Options& options)
{
    std::vector<std::byte> plan_document;
    if (!read_bytes(options.plan, plan_document) || plan_document.empty())
    {
        std::cerr << "recording plan read failed\n";
        return 2;
    }

    PhysicalSource source{options.frames, options.seed};
    auto schema = physical_schema();

    std::unique_ptr<MemorySpoolFile> memory;
    std::unique_ptr<BoundedMappedSpoolFile> mapped;
    SpoolFile* store = nullptr;
    if (options.backend == "memory")
    {
        memory = std::make_unique<MemorySpoolFile>(options.spool_capacity);
        store = memory.get();
    }
    else if (options.backend == "mapped")
    {
        mapped = std::make_unique<BoundedMappedSpoolFile>();
        const auto result = mapped->create(options.spool, options.spool_capacity);
        if (result.status != SpoolIoStatus::ok)
        {
            std::cerr << "mapped spool create failed: " << result.platform_error << '\n';
            return 3;
        }
        store = mapped.get();
    }
    else
    {
        std::cerr << "unsupported critical spool backend\n";
        return 2;
    }
    const auto checkpoint_capacity =
        options.checkpoint_interval == 0
            ? std::size_t{0}
            : options.frames / options.checkpoint_interval +
                  (options.frames % options.checkpoint_interval != 0 ? 1U : 0U);
    constexpr auto timing_margin = std::size_t{8};
    if (options.frames >
        std::numeric_limits<std::size_t>::max() - checkpoint_capacity - timing_margin)
    {
        std::cerr << "timing sample capacity overflow\n";
        return 2;
    }
    const auto timing_capacity = options.frames + checkpoint_capacity + timing_margin;
    TimedSpool timed_store{*store, timing_capacity};

    const std::array planned{
        PlannedSignalRecording{kNeural, 4ULL * 256ULL * 2ULL, 4},
        PlannedSignalRecording{kCursor, 2ULL * 4ULL, 1},
        PlannedSignalRecording{kAudio, 441ULL * 2ULL, 441},
        PlannedSignalRecording{kBandpower, 256ULL * 8ULL, 1},
    };
    NativeRecordingPlan plan{};
    plan.session_id = "018f4f30-6f9d-7b36-8c61-03f96cf5c613";
    plan.session_uuid = {0x01, 0x8f, 0x4f, 0x30, 0x6f, 0x9d, 0x7b, 0x36,
                         0x8c, 0x61, 0x03, 0xf9, 0x6c, 0xf5, 0xc6, 0x13};
    plan.created_unix_nanos = 1'786'384'800'000'000'000ULL;
    plan.plan_document = plan_document;
    plan.plan_fingerprint = sha256(plan_document);
    plan.native_session_id = kReplaySessionId;
    plan.native_schema_id = kSchemaId;
    plan.recorded_signals = planned;
    plan.frame_queue_capacity = options.queue_capacity;
    plan.control_queue_capacity = 64;
    plan.max_blocks_per_frame = 4;
    plan.max_frame_payload_bytes = kMaxFramePayload;
    plan.max_signal_gaps_per_discontinuity = 4;
    plan.max_control_payload_bytes = 4096;
    plan.max_records_per_transaction = 256;
    plan.max_transaction_bytes = 1ULL << 20U;
    plan.checkpoint_interval_transactions = options.checkpoint_interval;
    plan.worker_idle_poll_nanos = options.worker_poll_ns;
    plan.drain_timeout_nanos = options.drain_timeout_ns;
    plan.durability_policy = DurabilityPolicy::buffered;

    BenchmarkUnixClock unix_clock;
    NativeRecorderCore recorder;
    if (recorder.prepare(plan, timed_store, unix_clock) != RecorderStatusCode::ok)
    {
        std::cerr << "recorder prepare failed\n";
        return 3;
    }
    TimedCriticalObserver observer{recorder, source.item_count() + 8};

    RealtimeConfig realtime;
    realtime.pool_capacity.source_owned = 1;
    realtime.pool_capacity.ingress_capacity = 64;
    realtime.pool_capacity.processor_owned = 2;
    realtime.pool_capacity.critical_edge_capacity = options.queue_capacity;
    realtime.pool_capacity.actuator_owned = 1;
    realtime.pool_capacity.observer_edge_capacity = options.queue_capacity;
    realtime.buffer_size = static_cast<std::size_t>(kMaxFramePayload);
    realtime.max_signal_blocks = 4;
    realtime.discontinuity_capacity = 8;
    realtime.gaps_per_discontinuity = 4;
    realtime.max_process_outputs = 1;
    realtime.max_flush_outputs = 0;
    realtime.fault_history_capacity = 8;
    realtime.source_stall_timeout = RealtimeDuration{10'000'000'000ULL};
    realtime.max_ingress_dwell = RealtimeDuration{10'000'000'000ULL};
    realtime.processor_execution_deadline = RealtimeDuration{10'000'000'000ULL};
    realtime.max_source_to_actuator_age = RealtimeDuration{10'000'000'000ULL};
    realtime.max_output_age = RealtimeDuration{10'000'000'000ULL};
    realtime.actuator_deadline = RealtimeDuration{10'000'000'000ULL};
    realtime.shutdown_deadline = RealtimeDuration{5'000'000'000ULL};

    IdentityProcessor processor;
    CountingConsumer consumer;
    NativeStreamRunner runtime{std::move(schema), realtime, source, processor, consumer};
    const ObserverEdgeConfig edge{.id = 613,
                                  .capacity = options.queue_capacity,
                                  .drop_history_capacity = 8,
                                  .drop_policy = ObserverDropPolicy::fault,
                                  .critical_recorder = true};
    const auto add_status = runtime.add_critical_observer(observer, edge);
    const auto prepare_status = add_status == StreamStatus::ok ? runtime.prepare() : add_status;
    const auto arm_status = prepare_status == StreamStatus::ok ? runtime.arm() : prepare_status;
    if (add_status != StreamStatus::ok || prepare_status != StreamStatus::ok ||
        arm_status != StreamStatus::ok)
    {
        std::cerr << "runtime prepare/readiness failed\n";
        return 3;
    }

    const auto counters_before = process_counters();
    const auto started = now_ns();
    const auto start_status = runtime.start();
    const auto join_status = start_status == StreamStatus::ok ? runtime.join() : start_status;
    const auto elapsed = now_ns() - started;
    const auto recorder_status = recorder.status();
    const auto counters_after = process_counters();
    const auto runtime_fault = runtime.primary_fault();
    const auto timing_complete = timed_store.append_dropped() == 0 &&
                                 timed_store.checkpoint_dropped() == 0 &&
                                 timed_store.sync_dropped() == 0;
    const auto ok = start_status == StreamStatus::ok && join_status == StreamStatus::ok &&
                    !runtime_fault.has_value() &&
                    recorder_status.state == RecorderLifecycleState::stopped &&
                    recorder_status.runtime_accepted == source.item_count() &&
                    recorder_status.spool_committed == source.item_count() && timing_complete;

    bool spool_written = false;
    if (options.backend == "memory")
    {
        const auto bytes = memory->snapshot();
        spool_written = write_bytes(options.spool, bytes);
    }
    static_cast<void>(recorder.close());
    if (options.backend == "mapped")
    {
        const auto closed = mapped->close();
        spool_written = closed.status == SpoolIoStatus::ok;
    }
    const auto callback = observer.callback_summary();

    std::cout << '{';
    field("schema_version", 1, true);
    field("stage", "native_recording");
    field("status", ok && spool_written ? "ok" : "failed");
    field("spool_backend", options.backend);
    field("durability_policy", "buffered");
    field("source_pacing", "one_frame_per_millisecond");
    field("sync_scope", "readiness_superblock_only");
    field("elapsed_ns", elapsed);
    field("frames", consumer.frames());
    field("payload_bytes", recorder_status.payload_bytes_copied);
    field("spool_bytes", recorder_status.spool_committed_extent);
    field("runtime_accepted", recorder_status.runtime_accepted);
    field("recorder_accepted", recorder_status.recorder_accepted);
    field("spool_committed", recorder_status.spool_committed);
    field("committed_transactions", recorder_status.spool_committed_transactions);
    field("queue_capacity", recorder_status.data_queue_capacity);
    field("queue_high_water_mark", recorder_status.data_queue_high_water_mark);
    field("queue_pending", recorder_status.data_queue_pending);
    field("drain_ns", observer.drain_ns());
    field("throughput_payload_bytes_per_second",
          elapsed == 0 ? 0.0
                       : static_cast<double>(recorder_status.payload_bytes_copied) * 1e9 / elapsed);
    field("throughput_spool_bytes_per_second",
          elapsed == 0
              ? 0.0
              : static_cast<double>(recorder_status.spool_committed_extent) * 1e9 / elapsed);
    emit_summary("callback_enqueue_latency", callback);
    emit_summary("spool_append_latency", timed_store.append_summary());
    emit_summary("checkpoint_append_latency", timed_store.checkpoint_summary());
    emit_summary("spool_sync_latency", timed_store.sync_summary());
    field("spool_append_timing_dropped", timed_store.append_dropped());
    field("checkpoint_append_timing_dropped", timed_store.checkpoint_dropped());
    field("spool_sync_timing_dropped", timed_store.sync_dropped());
    field("timing_sample_capacity", timing_capacity);
    bool_field("spool_written", spool_written);
    bool_field("spool_materialization_outside_measured_interval", options.backend == "memory");
    bool_field("runtime_fault_present", runtime_fault.has_value());
    emit_process_delta(counters_before, counters_after);
    emit_build();
    std::cout << "}\n";
    return ok && spool_written ? 0 : 4;
}

[[nodiscard]] int metadata()
{
    std::cout << '{';
    field("schema_version", 1, true);
    field("stage", "native_helper_metadata");
    field("status", "ok");
    emit_build();
    std::cout << "}\n";
    return 0;
}

[[nodiscard]] int source_checksums(const Options& options)
{
    PhysicalSource source{11, options.seed};
    FramePool frames{1, kMaxFramePayload, 4};
    frames.prefault();

    std::uint64_t first_host_ns = 0;
    std::uint64_t second_host_ns = 0;
    std::uint64_t first_behavior_ns = 0;
    std::uint64_t second_behavior_ns = 0;
    std::uint32_t neural_samples = 0;
    std::uint32_t cursor_samples = 0;
    std::uint32_t audio_samples = 0;
    std::uint32_t bandpower_samples = 0;
    std::size_t neural_payload_bytes = 0;
    std::size_t multimodal_payload_bytes = 0;
    std::uint32_t neural_only_blocks = 0;
    std::uint32_t multimodal_blocks = 0;
    for (std::size_t i = 0; i <= 10; ++i)
    {
        FrameLease lease;
        if (frames.try_acquire(lease) != StreamStatus::ok ||
            source.read(lease.frame()) != StreamStatus::ok)
        {
            std::cerr << "source profile read failed\n";
            return 3;
        }
        const auto frame = lease.view();
        if (i == 0)
        {
            first_host_ns = frame.header.host_received_ns;
            multimodal_blocks = frame.header.signal_block_count;
            multimodal_payload_bytes = frame.payload.size();
            for (const auto& block : frame.blocks)
            {
                if (block.signal_id == kNeural)
                {
                    neural_samples = block.n_samples;
                    neural_payload_bytes = block.payload_byte_count;
                }
                else if (block.signal_id == kCursor)
                {
                    cursor_samples = block.n_samples;
                    first_behavior_ns = block.observation_time_start_ns;
                }
                else if (block.signal_id == kAudio)
                    audio_samples = block.n_samples;
                else if (block.signal_id == kBandpower)
                    bandpower_samples = block.n_samples;
            }
        }
        else if (i == 1)
        {
            second_host_ns = frame.header.host_received_ns;
            neural_only_blocks = frame.header.signal_block_count;
        }
        else if (i == 10)
        {
            for (const auto& block : frame.blocks)
                if (block.signal_id == kCursor)
                    second_behavior_ns = block.observation_time_start_ns;
        }
    }

    std::cout << '{';
    field("schema_version", 1, true);
    field("stage", "source_payload_checksums");
    field("status", "ok");
    field("payload_generator", "m6-13-counter-v1");
    field("seed", options.seed);
    field("neural_sha256", source.neural_sha256());
    field("cursor_sha256", source.cursor_sha256());
    field("audio_sha256", source.audio_sha256());
    field("bandpower_sha256", source.bandpower_sha256());
    field("frame_period_ns", second_host_ns - first_host_ns);
    field("behavior_block_period_ns", second_behavior_ns - first_behavior_ns);
    field("neural_frame_samples", neural_samples);
    field("cursor_block_samples", cursor_samples);
    field("audio_block_samples", audio_samples);
    field("bandpower_block_samples", bandpower_samples);
    field("neural_frame_payload_bytes", neural_payload_bytes);
    field("multimodal_frame_payload_bytes", multimodal_payload_bytes);
    field("neural_only_signal_blocks", neural_only_blocks);
    field("multimodal_signal_blocks", multimodal_blocks);
    std::cout << "}\n";
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse(argc, argv, options))
    {
        std::cerr << "usage: neurale_recording_replay_benchmark "
                     "<record|replay|metadata|source-checksums> "
                     "[--frames N --seed N --plan PATH --spool PATH --backend memory|mapped] "
                     "[--image PATH] "
                     "[--pacing as_fast_as_possible|recorded|step]\n";
        return 2;
    }
    if (options.operation == "record")
        return record(options);
    if (options.operation == "replay")
        return replay(options);
    if (options.operation == "metadata")
        return metadata();
    if (options.operation == "source-checksums")
        return source_checksums(options);
    return 2;
}
