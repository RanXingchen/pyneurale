/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Measured evidence for the simulation and simulated-device layer.
///
/// This target measures deterministic signal generation, the simulated
/// ``NativeFrameSource``, its pacing, cancellation, and scheduled loss/fault
/// handling, plus one small simulated-device -> native-sink integration. It
/// deliberately does not re-measure filters, features, decoders, or recording;
/// those keep their own owners in ``docs/development/benchmarks.md``. The
/// second integration case, simulated device -> SessionRecorder, lives in
/// ``benchmarks/benchmark_simulation.py`` because SessionRecorder is a
/// Python-owned control-plane object.
///
/// The default geometry is an NSP-class amplifier: 256 channels of sample-major
/// int16 at 30 kHz in 30-sample (1 ms) frames. The payload holds raw ADC counts,
/// so generator amplitudes are in counts and the schema declares dimensionless
/// rather than volts -- nothing here supplies a calibration.
///
/// Every row is one JSON Lines record carrying its own geometry, generator
/// configuration, pacing mode, clock parameters, and fault schedule, so a row
/// is interpretable without the invoking command line -- which is itself
/// recorded in the ``command`` field.

#include "allocation_tracker.h"
#include "benchmark_report.h"

#include <neurale/devices/simulation.h>
#include <neurale/runtime/runtime_info.h>
#include <neurale/signal/simulation.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/clock.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/realtime_config.h>
#include <neurale/streaming/runtime.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/source.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
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
using neurale::benchmark::platform_name;
using neurale::benchmark::summarize;
using neurale::benchmark::Summary;
using neurale::benchmark::text_field;

namespace ds = neurale::devices::simulation;
namespace ss = neurale::signal::simulation;
using namespace neurale::streaming;

using Clock = std::chrono::steady_clock;

/// The runtime stamps ``host_received_ns`` from the default steady clock, so
/// every timestamp taken here must come from the same epoch to be subtractable.
[[nodiscard]] std::uint64_t now_ns() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

struct Options
{
    std::uint32_t channels{256};
    std::uint64_t fs{30'000};
    std::uint32_t frame_samples{30};
    std::size_t generation_iterations{20'000};
    std::size_t read_iterations{20'000};
    std::size_t pipeline_frames{20'000};
    std::size_t warmup_frames{512};
    double paced_seconds{2.0};
    std::size_t cancel_repetitions{32};
    std::uint64_t loss_period_frames{64};
    std::uint64_t loss_samples{4};
};

struct ProcessCounters
{
    double cpu_seconds{};
    std::uint64_t peak_rss_bytes{};
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
#endif
    return result;
}

std::string g_command;

/// One emitted record. Fields that a case cannot measure keep their default and
/// are still written, so consumers never have to guess whether a column was
/// absent or genuinely zero.
struct Result
{
    std::string_view benchmark;
    std::string_view metric;
    std::string_view unit{"ns"};
    Summary summary{};
    double throughput_per_second{};
    std::string_view throughput_item{"none"};
    std::uint64_t allocations{};
    std::string_view allocation_tracking{"steady_state"};
    std::string_view generator_kind{"tones"};
    std::uint32_t channels{};
    std::uint64_t fs_num{};
    std::uint64_t fs_den{1};
    std::uint32_t frame_samples{};
    std::uint32_t max_frame_samples{};
    /// Payload element type of the measured row. The device ships int16 counts;
    /// the generator itself always produces float64, so the generation case
    /// reports what it actually wrote rather than what the device would ship.
    std::string_view dtype{"int16"};
    std::size_t frame_payload_bytes{};
    std::string_view pacing_mode{"unpaced"};
    bool device_ticks{true};
    std::int64_t clock_offset_ns{};
    std::int64_t clock_drift_ppm{};
    std::uint64_t clock_sync_uncertainty_ns{};
    std::string fault_schedule{"none"};
    std::size_t warmups{};
    std::size_t repetitions{};
    std::size_t frame_pool_capacity{};
    /// Payload bytes the frame pool reserves: capacity x buffer size. The pool
    /// also allocates block-header storage and its slot registry, so this is
    /// deliberately not a claim about the pool's whole resident footprint --
    /// `peak_resident_memory_bytes` is the process-level figure.
    std::size_t frame_pool_payload_bytes{};
    std::uint64_t ingress_high_water_mark{};
    std::size_t outstanding_frames{};
    std::uint64_t discontinuities{};
    double process_cpu_seconds{};
    std::uint64_t peak_resident_memory_bytes{};
    bool passed{true};
};

template <typename Value> void number_field(std::string_view name, Value value)
{
    std::cout << ',';
    emit_json_string(name);
    std::cout << ':' << value;
}

void emit(const Result& result)
{
    static const auto build = neurale::runtime::build_info();
    static const auto cpu = neurale::runtime::cpu_info();
    static const auto threading = neurale::runtime::threading_info();

    std::cout << "{\"schema_version\":1";
    text_field("benchmark", result.benchmark);
    text_field("metric", result.metric);
    text_field("unit", result.unit);
    number_field("samples", result.summary.samples);
    number_field("min_ns", result.summary.minimum);
    number_field("median_ns", result.summary.median);
    number_field("p95_ns", result.summary.p95);
    number_field("p99_ns", result.summary.p99);
    number_field("max_ns", result.summary.maximum);
    number_field("throughput_per_second", result.throughput_per_second);
    text_field("throughput_item", result.throughput_item);
    number_field("allocations", result.allocations);
    text_field("allocation_tracking", result.allocation_tracking);
    text_field("allocation_backend", neurale::benchmark::allocation_tracking_backend());
    text_field("generator_kind", result.generator_kind);
    number_field("channels", result.channels);
    number_field("fs_numerator", result.fs_num);
    number_field("fs_denominator", result.fs_den);
    text_field("dtype", result.dtype);
    number_field("frame_samples", result.frame_samples);
    number_field("max_frame_samples", result.max_frame_samples);
    number_field("frame_payload_bytes", result.frame_payload_bytes);
    text_field("pacing_mode", result.pacing_mode);
    bool_field("device_ticks", result.device_ticks);
    number_field("clock_offset_ns", result.clock_offset_ns);
    number_field("clock_drift_ppm", result.clock_drift_ppm);
    number_field("clock_sync_uncertainty_ns", result.clock_sync_uncertainty_ns);
    text_field("fault_schedule", result.fault_schedule);
    number_field("warmups", result.warmups);
    number_field("repetitions", result.repetitions);
    number_field("frame_pool_capacity", result.frame_pool_capacity);
    number_field("frame_pool_payload_bytes", result.frame_pool_payload_bytes);
    number_field("ingress_high_water_mark", result.ingress_high_water_mark);
    number_field("outstanding_frames", result.outstanding_frames);
    number_field("discontinuities", result.discontinuities);
    number_field("process_cpu_seconds", result.process_cpu_seconds);
    number_field("peak_resident_memory_bytes", result.peak_resident_memory_bytes);
    text_field("command", g_command);
    text_field("platform", platform_name());
    text_field("native_version", build.version);
    number_field("native_abi_version", build.abi_version);
    text_field("compiler", build.compiler);
    text_field("build_type", build.build_type);
    text_field("cpu_math_backend", build.cpu_math_backend);
    text_field("fft_backend", build.fft_backend);
    bool_field("cuda_compiled", build.cuda_compiled);
    text_field("cpu_architecture", cpu.architecture);
    text_field("cpu_vendor", cpu.vendor.value_or(""));
    text_field("cpu_model", cpu.model.value_or(""));
    number_field("logical_cores", cpu.logical_cores);
    text_field("threading_backend", threading.backend);
    number_field("threads", threading.num_threads);
    bool_field("passed", result.passed);
    std::cout << "}\n";
}

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

/// Raw 16-bit ADC counts, the wire format of an NSP-class amplifier. The
/// generator is configured in count units and the schema declares dimensionless
/// counts rather than volts, because nothing here supplies a calibration.
constexpr auto kSampleDtype = neurale::streaming::SignalDType::int16;
constexpr std::int64_t kClockOffsetNs = 250;
constexpr std::int64_t kClockDriftPpm = 100;
constexpr std::uint64_t kClockUncertaintyNs = 50;

[[nodiscard]] ss::SignalGenerator make_generator(std::string_view kind, const Options& options)
{
    const auto channels = static_cast<std::size_t>(options.channels);
    const auto rate = static_cast<double>(options.fs);
    if (kind == "zeros")
    {
        return ss::SignalGenerator::zeros(channels, rate);
    }
    if (kind == "constant")
    {
        std::vector<double> values(channels);
        for (std::size_t i = 0; i < channels; ++i)
        {
            values[i] = 100.0 + static_cast<double>(i);
        }
        return ss::SignalGenerator::constant(channels, rate, values);
    }
    if (kind == "noise")
    {
        return ss::SignalGenerator::noise(channels, rate, 0x5EED'1234'ABCDULL, -1'500.0, 1'500.0);
    }
    if (kind == "samples")
    {
        // One repeating frame of stored values: the memcpy-shaped generator,
        // measured at the same frame size every other case uses.
        std::vector<double> values(channels * options.frame_samples);
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            values[i] = static_cast<double>(i % 97) * 32.0;
        }
        return ss::SignalGenerator::samples(channels, rate, options.frame_samples, values, true);
    }
    // Two tones per channel, the default acquisition-shaped generator. All
    // amplitudes are in ADC counts and stay well inside the int16 range, so the
    // device's quantization rounds rather than saturates.
    std::vector<double> freqs(channels * 2);
    std::vector<double> amps(channels * 2);
    std::vector<double> phases(channels * 2);
    for (std::size_t i = 0; i < channels; ++i)
    {
        freqs[i] = 7.0 + static_cast<double>(i % 13);
        freqs[channels + i] = 83.0 + static_cast<double>(i % 29);
        amps[i] = 2'000.0;
        amps[channels + i] = 500.0;
        phases[i] = static_cast<double>(i) * 0.01;
        phases[channels + i] = -static_cast<double>(i) * 0.02;
    }
    return ss::SignalGenerator::tones(channels, rate, 2, freqs, amps, phases);
}

[[nodiscard]] ds::SimulatedNeuralSourceConfig
source_config(const Options& options, bool paced, std::vector<ds::AcquisitionEvent> events,
              std::optional<std::uint64_t> total = {})
{
    return {
        .session_id = 7,
        .schema_id = 707,
        .signal_id = 1,
        .clock_domain = 41,
        .nominal_samples_per_frame = options.frame_samples,
        .max_samples_per_frame = options.frame_samples,
        .fs = {options.fs, 1},
        .sample_dtype = kSampleDtype,
        .physical_unit = PhysicalUnit::dimensionless,
        .channel_set_id = 1,
        .calibration_id = 0,
        .reference_id = 0,
        .initial_sample_idx = 0,
        .total_sample_count = total,
        .device_ticks = true,
        .initial_device_tick = 0,
        .clock_offset_ns = kClockOffsetNs,
        .clock_drift_ppm = kClockDriftPpm,
        .clock_sync_uncertainty_ns = kClockUncertaintyNs,
        .paced = paced,
        .events = std::move(events),
    };
}

/// Fill in the geometry and clock fields every row shares, so no case can
/// describe the same workload differently. ``clock_configured`` is false for
/// the generation case, which has no device clock at all -- reporting the
/// simulator's clock parameters there would attribute settings to a
/// measurement they never touched.
void describe(Result& result, const Options& options, bool clock_configured = true)
{
    result.channels = options.channels;
    result.fs_num = options.fs;
    result.fs_den = 1;
    result.frame_samples = options.frame_samples;
    result.max_frame_samples = options.frame_samples;
    result.frame_payload_bytes = static_cast<std::size_t>(options.channels) *
                                 options.frame_samples *
                                 neurale::streaming::signal_dtype_size(kSampleDtype);
    result.device_ticks = clock_configured;
    result.clock_offset_ns = clock_configured ? kClockOffsetNs : 0;
    result.clock_drift_ppm = clock_configured ? kClockDriftPpm : 0;
    result.clock_sync_uncertainty_ns = clock_configured ? kClockUncertaintyNs : 0;
}

// ---------------------------------------------------------------------------
// Case 1: deterministic generation cost
// ---------------------------------------------------------------------------

[[nodiscard]] bool benchmark_generation(const Options& options)
{
    constexpr std::array<std::string_view, 5> kinds{"zeros", "constant", "tones", "noise",
                                                    "samples"};
    bool passed = true;
    for (const auto kind : kinds)
    {
        const auto generator = make_generator(kind, options);
        const auto values_per_frame =
            static_cast<std::size_t>(options.channels) * options.frame_samples;
        std::vector<double> output(values_per_frame);
        const auto before = process_counters();

        for (std::size_t i = 0; i < options.warmup_frames; ++i)
        {
            if (generator.generate(i * options.frame_samples, options.frame_samples, output) !=
                ss::GenerationStatus::ok)
            {
                return false;
            }
        }

        std::vector<std::uint64_t> timings(options.generation_iterations);
        bool valid = true;
        neurale::benchmark::AllocationScope allocations;
        for (std::size_t i = 0; i < options.generation_iterations; ++i)
        {
            const auto start = static_cast<std::uint64_t>(i) * options.frame_samples;
            const auto started = now_ns();
            valid = valid && generator.generate(start, options.frame_samples, output) ==
                                 ss::GenerationStatus::ok;
            timings[i] = now_ns() - started;
        }
        allocations.stop();
        const auto after = process_counters();
        const auto count = allocations.count();
        const auto summary = summarize(timings);
        const auto case_passed = valid && count == 0;
        passed = passed && case_passed;

        Result result{
            .benchmark = "signal_generation",
            .metric = "generate_frame",
            .summary = summary,
            .throughput_per_second = summary.median <= 0.0 ? 0.0
                                                           : static_cast<double>(values_per_frame) *
                                                                 1e9 / summary.median,
            .throughput_item = "channel_samples",
            .allocations = count,
            .generator_kind = kind,
            .pacing_mode = "not_applicable",
            .warmups = options.warmup_frames,
            .repetitions = options.generation_iterations,
            .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
            .peak_resident_memory_bytes = after.peak_rss_bytes,
            .passed = case_passed,
        };
        describe(result, options, false);
        // The generator is dtype-agnostic: it writes float64 into a caller
        // buffer, and the device converts afterwards. Reporting the device's
        // int16 geometry here would understate the bytes this case moved.
        result.dtype = "float64";
        result.frame_payload_bytes = values_per_frame * sizeof(double);
        emit(result);
    }
    return passed;
}

// ---------------------------------------------------------------------------
// Case 2: unpaced simulated-source read
// ---------------------------------------------------------------------------

[[nodiscard]] bool benchmark_source_read(const Options& options)
{
    const auto generator = make_generator("tones", options);
    ds::SimulatedNeuralSource source{generator, source_config(options, false, {})};
    const auto& signal = source.schema().signals().front();
    FramePool pool{2, signal.max_block_bytes, 1};
    pool.prefault();
    FrameLease lease;
    if (pool.try_acquire(lease) != StreamStatus::ok)
    {
        return false;
    }
    const auto before = process_counters();
    for (std::size_t i = 0; i < options.warmup_frames; ++i)
    {
        if (source.read(lease.frame()) != StreamStatus::ok)
        {
            return false;
        }
    }

    std::vector<std::uint64_t> timings(options.read_iterations);
    bool valid = true;
    neurale::benchmark::AllocationScope allocations;
    const auto run_started = now_ns();
    for (std::size_t i = 0; i < options.read_iterations; ++i)
    {
        const auto started = now_ns();
        valid = valid && source.read(lease.frame()) == StreamStatus::ok;
        timings[i] = now_ns() - started;
    }
    const auto run_duration = now_ns() - run_started;
    allocations.stop();
    const auto after = process_counters();
    const auto count = allocations.count();
    const auto released = lease.reset() == StreamStatus::ok;
    const auto case_passed = valid && released && count == 0 && pool.outstanding() == 0;

    const auto summary = summarize(timings);
    const auto frames_per_second = run_duration == 0
                                       ? 0.0
                                       : static_cast<double>(options.read_iterations) * 1e9 /
                                             static_cast<double>(run_duration);
    Result result{
        .benchmark = "simulated_source_read",
        .metric = "read_frame",
        .summary = summary,
        .throughput_per_second = frames_per_second,
        .throughput_item = "frames",
        .allocations = count,
        .warmups = options.warmup_frames,
        .repetitions = options.read_iterations,
        .frame_pool_capacity = 2,
        .frame_pool_payload_bytes = 2 * signal.max_block_bytes,
        .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(result, options);
    emit(result);

    // The same run expressed against the acquisition timeline it simulates: how
    // many times faster than real time one unpaced source can produce frames.
    std::vector<std::uint64_t> realtime_factor{static_cast<std::uint64_t>(
        frames_per_second * options.frame_samples * 1'000.0 / static_cast<double>(options.fs))};
    Result factor{
        .benchmark = "simulated_source_read",
        .metric = "realtime_factor",
        .unit = "factor_x1000",
        .summary = summarize(realtime_factor),
        .throughput_per_second = frames_per_second * options.frame_samples,
        .throughput_item = "samples",
        .allocations = count,
        .warmups = options.warmup_frames,
        .repetitions = options.read_iterations,
        .frame_pool_capacity = 2,
        .frame_pool_payload_bytes = 2 * signal.max_block_bytes,
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(factor, options);
    emit(factor);
    return case_passed;
}

// ---------------------------------------------------------------------------
// Case 3: paced acquisition timing error and jitter
// ---------------------------------------------------------------------------

[[nodiscard]] bool benchmark_pacing(const Options& options)
{
    const auto generator = make_generator("tones", options);
    ds::SimulatedNeuralSource source{generator, source_config(options, true, {})};
    const auto& signal = source.schema().signals().front();
    FramePool pool{2, signal.max_block_bytes, 1};
    pool.prefault();
    FrameLease lease;
    if (pool.try_acquire(lease) != StreamStatus::ok)
    {
        return false;
    }

    // Sizing only. Pacing follows the drifted device tick rate rather than the
    // nominal sample rate, so this decides how many frames fit in the requested
    // interval; it is deliberately not the oracle the error is measured against.
    const auto drifted_rate = static_cast<double>(options.fs) *
                              (1'000'000.0 + static_cast<double>(kClockDriftPpm)) / 1'000'000.0;
    const auto nominal_interval_ns =
        static_cast<double>(options.frame_samples) * 1e9 / drifted_rate;
    const auto frames = static_cast<std::size_t>(options.paced_seconds * 1e9 / nominal_interval_ns);
    if (frames < 2)
    {
        return false;
    }
    std::vector<std::uint64_t> error(frames - 1);
    std::vector<std::uint64_t> jitter(frames - 1);
    const auto before = process_counters();

    // Each frame's intended acquisition deadline comes from the simulator's own
    // clock provenance, not from a deadline the benchmark reconstructs. The
    // source writes `clock_sync.host_time_reference_ns = deadline +
    // clock_offset_ns`, so subtracting the configured offset recovers the exact
    // deadline it paced to. Anchoring instead to a host timestamp taken after
    // the first read would be late by that read's whole cost -- the epoch is
    // latched at the top of the read, before the frame is generated -- and at
    // the default geometry that bias is the same order as the metric itself.
    const auto deadline_of = [](const FrameView& view) noexcept
    {
        return view.blocks.front().clock_sync.host_time_reference_ns -
               static_cast<std::uint64_t>(kClockOffsetNs);
    };

    // Frame 0 is not a measured sample: its deadline is the epoch latched inside
    // that very read, so its "error" is by construction the read's own cost and
    // can never be early. It supplies the reference deadline and nothing else.
    if (source.read(lease.frame()) != StreamStatus::ok)
    {
        return false;
    }
    auto previous_completion = now_ns();
    auto previous_deadline = deadline_of(lease.view());
    const auto first_deadline = previous_deadline;
    auto last_deadline = previous_deadline;
    bool valid = true;
    neurale::benchmark::AllocationScope allocations;
    for (std::size_t i = 1; i < frames; ++i)
    {
        valid = valid && source.read(lease.frame()) == StreamStatus::ok;
        const auto completed = now_ns();
        const auto deadline = deadline_of(lease.view());
        error[i - 1] = completed > deadline ? completed - deadline : deadline - completed;
        const auto observed = completed - previous_completion;
        const auto intended = deadline - previous_deadline;
        jitter[i - 1] = observed > intended ? observed - intended : intended - observed;
        previous_completion = completed;
        previous_deadline = deadline;
        last_deadline = deadline;
    }
    allocations.stop();
    // Report the rate the simulator actually paced to, measured from its own
    // deadlines, rather than the nominal figure used to size the run.
    const auto paced_interval_ns =
        frames < 2
            ? 0.0
            : static_cast<double>(last_deadline - first_deadline) / static_cast<double>(frames - 1);
    const auto after = process_counters();
    const auto count = allocations.count();
    const auto released = lease.reset() == StreamStatus::ok;
    const auto case_passed = valid && released && count == 0 && pool.outstanding() == 0;

    Result timing{
        .benchmark = "simulated_source_paced",
        .metric = "pacing_absolute_error",
        .summary = summarize(error),
        .throughput_per_second = paced_interval_ns <= 0.0 ? 0.0 : 1e9 / paced_interval_ns,
        .throughput_item = "frames",
        .allocations = count,
        .pacing_mode = "paced_steady_clock",
        .warmups = 1,
        .repetitions = frames - 1,
        .frame_pool_capacity = 2,
        .frame_pool_payload_bytes = 2 * signal.max_block_bytes,
        .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(timing, options);
    emit(timing);

    Result interval{
        .benchmark = "simulated_source_paced",
        .metric = "inter_frame_jitter",
        .summary = summarize(jitter),
        .throughput_per_second = paced_interval_ns <= 0.0 ? 0.0 : 1e9 / paced_interval_ns,
        .throughput_item = "frames",
        .allocations = count,
        .pacing_mode = "paced_steady_clock",
        .warmups = 1,
        .repetitions = frames - 1,
        .frame_pool_capacity = 2,
        .frame_pool_payload_bytes = 2 * signal.max_block_bytes,
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(interval, options);
    emit(interval);
    return case_passed;
}

// ---------------------------------------------------------------------------
// Case 4: scheduled-loss discontinuity handling
// ---------------------------------------------------------------------------

[[nodiscard]] bool benchmark_discontinuity(const Options& options)
{
    const auto generator = make_generator("tones", options);
    std::vector<ds::AcquisitionEvent> events;
    const auto n_events = options.read_iterations / options.loss_period_frames;
    events.reserve(n_events);
    for (std::uint64_t i = 1; i <= n_events; ++i)
    {
        events.push_back({.frame_ordinal = i * options.loss_period_frames,
                          .kind = ds::AcquisitionEventKind::sample_loss,
                          .n_samples = options.loss_samples});
    }
    const std::string schedule = "sample_loss:every_" + std::to_string(options.loss_period_frames) +
                                 "_frames:" + std::to_string(options.loss_samples) + "_samples";

    ds::SimulatedNeuralSource source{generator, source_config(options, false, std::move(events))};
    const auto& signal = source.schema().signals().front();
    FramePool pool{2, signal.max_block_bytes, 1};
    pool.prefault();
    DiscontinuityPool gaps{2, 1};
    gaps.prefault();
    FrameLease lease;
    DiscontinuityLease gap;
    if (pool.try_acquire(lease) != StreamStatus::ok || gaps.try_acquire(gap) != StreamStatus::ok)
    {
        return false;
    }

    std::vector<std::uint64_t> data_timings;
    std::vector<std::uint64_t> gap_timings;
    data_timings.reserve(options.read_iterations);
    gap_timings.reserve(n_events + 1);
    bool valid = true;
    const auto before = process_counters();
    neurale::benchmark::AllocationScope allocations;
    for (std::size_t i = 0; i < options.read_iterations; ++i)
    {
        const auto started = now_ns();
        const auto status = source.read_message(lease.frame(), gap);
        const auto elapsed = now_ns() - started;
        if (status == StreamStatus::discontinuity)
        {
            gap_timings.push_back(elapsed);
            // Recycling the lease is the runtime's job in a real topology, so
            // it stays outside the measured interval here.
            valid = valid && gap.reset() == StreamStatus::ok &&
                    gaps.try_acquire(gap) == StreamStatus::ok;
            continue;
        }
        if (status != StreamStatus::ok)
        {
            valid = false;
            break;
        }
        data_timings.push_back(elapsed);
    }
    allocations.stop();
    const auto after = process_counters();
    const auto count = allocations.count();
    const auto released = gap.reset() == StreamStatus::ok && lease.reset() == StreamStatus::ok;
    const auto case_passed = valid && released && count == 0 && pool.outstanding() == 0 &&
                             gaps.outstanding() == 0 && !gap_timings.empty();

    Result data{
        .benchmark = "simulated_source_discontinuity",
        .metric = "data_read_under_schedule",
        .summary = summarize(data_timings),
        .allocations = count,
        .fault_schedule = schedule,
        .repetitions = data_timings.size(),
        .frame_pool_capacity = 2,
        .frame_pool_payload_bytes = 2 * signal.max_block_bytes,
        .discontinuities = gap_timings.size(),
        .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(data, options);
    emit(data);

    // A published gap costs one extra read that carries no frame. That whole
    // call is the handling overhead; it is not a slower data read.
    Result published{
        .benchmark = "simulated_source_discontinuity",
        .metric = "gap_publication",
        .summary = summarize(gap_timings),
        .allocations = count,
        .fault_schedule = schedule,
        .repetitions = gap_timings.size(),
        .frame_pool_capacity = 2,
        .frame_pool_payload_bytes = 2 * signal.max_block_bytes,
        .discontinuities = gap_timings.size(),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(published, options);
    emit(published);
    return case_passed;
}

// ---------------------------------------------------------------------------
// Case 5: cancellation of a blocked paced read
// ---------------------------------------------------------------------------

[[nodiscard]] bool benchmark_cancel(const Options& options)
{
    // A one-frame-per-second acquisition makes the second read block for a full
    // second, so the measured interval is cancellation alone and never a read
    // that was about to return anyway.
    Options slow = options;
    slow.fs = 1;
    slow.frame_samples = 1;
    const auto generator = make_generator("zeros", slow);
    ds::SimulatedNeuralSource source{generator, source_config(slow, true, {})};
    const auto& signal = source.schema().signals().front();
    FramePool pool{2, signal.max_block_bytes, 1};
    pool.prefault();
    FrameLease first;
    FrameLease blocked;
    if (pool.try_acquire(first) != StreamStatus::ok ||
        pool.try_acquire(blocked) != StreamStatus::ok)
    {
        return false;
    }

    std::vector<std::uint64_t> timings(options.cancel_repetitions);
    bool valid = true;
    const auto before = process_counters();
    for (std::size_t i = 0; i < options.cancel_repetitions; ++i)
    {
        valid = valid && source.read(first.frame()) == StreamStatus::ok;
        std::atomic<std::uint64_t> returned_at{};
        std::atomic<StreamStatus> status{StreamStatus::ok};
        std::thread reader{[&]
                           {
                               const auto result = source.read(blocked.frame());
                               returned_at.store(now_ns(), std::memory_order_release);
                               status.store(result, std::memory_order_release);
                           }};
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        const auto cancelled_at = now_ns();
        source.cancel();
        reader.join();
        const auto observed = returned_at.load(std::memory_order_acquire);
        valid = valid && status.load(std::memory_order_acquire) == StreamStatus::stopped &&
                observed >= cancelled_at;
        timings[i] = observed - cancelled_at;
        valid = valid && source.reset() == StreamStatus::ok;
    }
    const auto after = process_counters();
    const auto released = first.reset() == StreamStatus::ok && blocked.reset() == StreamStatus::ok;
    const auto case_passed = valid && released && pool.outstanding() == 0;

    Result result{
        .benchmark = "simulated_source_cancel",
        .metric = "cancel_to_read_return",
        .summary = summarize(timings),
        // A measurement thread is created per repetition, so this case cannot
        // make a steady-state allocation claim and does not pretend to.
        .allocation_tracking = "not_measured",
        .generator_kind = "zeros",
        .pacing_mode = "paced_steady_clock",
        .repetitions = options.cancel_repetitions,
        .frame_pool_capacity = 2,
        .frame_pool_payload_bytes = 2 * signal.max_block_bytes,
        .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(result, slow);
    emit(result);
    return case_passed;
}

// ---------------------------------------------------------------------------
// Case 6/7: simulated device -> native sink integration
// ---------------------------------------------------------------------------

/// Arms steady-state allocation tracking once the chain has warmed up, and
/// records when a scheduled fault first reached the acquisition loop.
class MeasuredSource final : public NativeFrameSource
{
  public:
    MeasuredSource(ds::SimulatedNeuralSource& source, std::size_t warmup_frames) noexcept
        : source_(source), warmup_frames_(warmup_frames)
    {
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        return finish(source_.read(frame));
    }

    StreamStatus read_message(MutableFrame& frame,
                              DiscontinuityLease& discontinuity) noexcept override
    {
        return finish(source_.read_message(frame, discontinuity));
    }

    [[nodiscard]] bool produces_discontinuities() const noexcept override
    {
        return source_.produces_discontinuities();
    }

    void cancel() noexcept override
    {
        source_.cancel();
    }

    StreamStatus reset() noexcept override
    {
        neurale::benchmark::set_allocation_tracking(false);
        successful_frames_ = 0;
        return source_.reset();
    }

    [[nodiscard]] std::uint64_t faulted_at_ns() const noexcept
    {
        return faulted_at_ns_.load(std::memory_order_acquire);
    }

  private:
    StreamStatus finish(StreamStatus status) noexcept
    {
        if (status == StreamStatus::ok && ++successful_frames_ == warmup_frames_)
        {
            neurale::benchmark::reset_allocation_count();
            neurale::benchmark::set_allocation_tracking(true);
        }
        if (status == StreamStatus::source_failure &&
            faulted_at_ns_.load(std::memory_order_relaxed) == 0)
        {
            faulted_at_ns_.store(now_ns(), std::memory_order_release);
        }
        return status;
    }

    ds::SimulatedNeuralSource& source_;
    std::size_t warmup_frames_{};
    std::size_t successful_frames_{};
    std::atomic<std::uint64_t> faulted_at_ns_{};
};

class PassthroughProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {context.input_schema.clone(),
                context.input_schema.clone(),
                1,
                0,
                true,
                {.workspace_bytes = 0, .frame_pool_leases = 1}};
    }

    StreamStatus process(FrameBorrow&, FrameEmitter& emitter) noexcept override
    {
        return emitter.publish_input();
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

/// Terminal sink recording acquisition-to-consumer latency into storage fixed
/// before the run, so the sink itself adds no steady-state allocation.
class LatencySink final : public NativeFrameConsumer
{
  public:
    LatencySink(std::size_t warmup_frames, std::size_t measured_frames)
        : warmup_frames_(warmup_frames), timings_(std::make_unique<std::vector<std::uint64_t>>())
    {
        timings_->reserve(measured_frames);
        timings_->resize(measured_frames);
    }

    StreamStatus consume(FrameView frame) noexcept override
    {
        const auto observed = now_ns();
        const auto idx = frames_++;
        if (idx >= warmup_frames_)
        {
            const auto measured = idx - warmup_frames_;
            if (measured < timings_->size())
            {
                const auto received = frame.header.host_received_ns;
                (*timings_)[measured] = observed > received ? observed - received : 0;
                ++measured_frames_;
            }
        }
        return StreamStatus::ok;
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        ++discontinuities_;
        return StreamStatus::ok;
    }

    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        frames_ = 0;
        measured_frames_ = 0;
        discontinuities_ = 0;
        return StreamStatus::ok;
    }

    [[nodiscard]] std::span<const std::uint64_t> timings() const noexcept
    {
        return {timings_->data(), measured_frames_};
    }
    [[nodiscard]] std::size_t frames() const noexcept
    {
        return frames_;
    }
    [[nodiscard]] std::uint64_t discontinuities() const noexcept
    {
        return discontinuities_;
    }

  private:
    std::size_t warmup_frames_{};
    std::unique_ptr<std::vector<std::uint64_t>> timings_;
    std::size_t frames_{};
    std::size_t measured_frames_{};
    std::uint64_t discontinuities_{};
};

[[nodiscard]] RealtimeConfig pipeline_config(std::size_t payload_bytes)
{
    RealtimeConfig config;
    config.platform.mode = RealtimeConfigMode::strict;
    config.platform.prefault_pools = true;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 64;
    config.pool_capacity.processor_owned = 8;
    config.pool_capacity.critical_edge_capacity = 64;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.reserve = 4;
    config.buffer_size = payload_bytes;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 8;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.max_flush_outputs = 1;
    config.fault_history_capacity = 8;
    config.source_stall_timeout = RealtimeDuration{10'000'000'000};
    config.max_ingress_dwell = RealtimeDuration{10'000'000'000};
    config.processor_execution_deadline = RealtimeDuration{10'000'000'000};
    config.max_source_to_actuator_age = RealtimeDuration{10'000'000'000};
    config.max_output_age = RealtimeDuration{10'000'000'000};
    config.actuator_deadline = RealtimeDuration{10'000'000'000};
    config.shutdown_deadline = RealtimeDuration{10'000'000'000};
    config.watchdog_period = RealtimeDuration{1'000'000};
    return config;
}

/// The runtime sizes its frame pool from `required_buffer_count()`, which also
/// reserves one buffer per ingress and critical-edge queue slot. Counting only
/// the stage-owned leases and the reserve reports roughly a tenth of what is
/// actually allocated, so the budget's own accessor is the only correct source.
[[nodiscard]] std::size_t pool_payload_bytes(const RealtimeConfig& config)
{
    return config.required_buffer_count() * config.buffer_size;
}

[[nodiscard]] bool benchmark_native_sink(const Options& options)
{
    const auto generator = make_generator("tones", options);
    const auto total_frames = options.warmup_frames + options.pipeline_frames;
    const auto total_samples = static_cast<std::uint64_t>(total_frames) * options.frame_samples;
    ds::SimulatedNeuralSource simulated{generator,
                                        source_config(options, false, {}, total_samples)};
    MeasuredSource source{simulated, options.warmup_frames};
    PassthroughProcessor processor;
    LatencySink sink{options.warmup_frames, options.pipeline_frames};
    // Take the buffer size from the schema the source froze, not from a
    // recomputed float64 geometry: the device's dtype decides the payload, and
    // a stale scalar size here silently over-allocates the whole frame pool and
    // misreports what the runtime reserved.
    const auto payload_bytes =
        static_cast<std::size_t>(simulated.schema().signals().front().max_block_bytes);
    auto config = pipeline_config(payload_bytes);
    const auto reserved_bytes = pool_payload_bytes(config);
    NativeStreamRunner runner{simulated.schema().clone(), config, source, processor, sink};

    const auto before = process_counters();
    if (runner.prepare() != StreamStatus::ok || runner.arm() != StreamStatus::ok)
    {
        return false;
    }
    const auto run_started = now_ns();
    if (runner.start() != StreamStatus::ok)
    {
        return false;
    }
    const auto status = runner.join();
    const auto run_duration = now_ns() - run_started;
    neurale::benchmark::set_allocation_tracking(false);
    const auto allocations = neurale::benchmark::allocation_count();
    const auto after = process_counters();
    const auto stats = runner.stats();
    const auto timings = sink.timings();
    const auto case_passed = status == StreamStatus::ok && sink.frames() == total_frames &&
                             timings.size() == options.pipeline_frames && allocations == 0 &&
                             runner.outstanding_frames() == 0 && stats.queue_overruns == 0 &&
                             stats.pool_exhaustions == 0;

    const auto summary = summarize(timings);
    const auto frames_per_second = run_duration == 0 ? 0.0
                                                     : static_cast<double>(total_frames) * 1e9 /
                                                           static_cast<double>(run_duration);
    Result latency{
        .benchmark = "simulated_device_to_native_sink",
        .metric = "acquisition_to_consumer",
        .summary = summary,
        .throughput_per_second = frames_per_second,
        .throughput_item = "frames",
        .allocations = allocations,
        .warmups = options.warmup_frames,
        .repetitions = options.pipeline_frames,
        .frame_pool_capacity = config.required_buffer_count(),
        .frame_pool_payload_bytes = reserved_bytes,
        .ingress_high_water_mark = stats.ingress_high_water_mark,
        .outstanding_frames = runner.outstanding_frames(),
        .discontinuities = sink.discontinuities(),
        .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(latency, options);
    emit(latency);

    Result throughput{
        .benchmark = "simulated_device_to_native_sink",
        .metric = "realtime_factor",
        .unit = "factor_x1000",
        .summary = summarize(std::vector<std::uint64_t>{
            static_cast<std::uint64_t>(frames_per_second * options.frame_samples * 1'000.0 /
                                       static_cast<double>(options.fs))}),
        .throughput_per_second = frames_per_second * options.frame_samples,
        .throughput_item = "samples",
        .allocations = allocations,
        .warmups = options.warmup_frames,
        .repetitions = options.pipeline_frames,
        .frame_pool_capacity = latency.frame_pool_capacity,
        .frame_pool_payload_bytes = reserved_bytes,
        .ingress_high_water_mark = stats.ingress_high_water_mark,
        .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(throughput, options);
    emit(throughput);
    return case_passed;
}

/// Cost of the generic fault path when the simulated device fails mid-session:
/// from the acquisition read that reported ``source_failure`` to the runner
/// having stopped and latched its primary fault.
[[nodiscard]] bool benchmark_fault_stop(const Options& options)
{
    const auto generator = make_generator("tones", options);
    const auto fault_ordinal = std::max<std::uint64_t>(options.warmup_frames, 64);
    std::vector<ds::AcquisitionEvent> events{
        {.frame_ordinal = fault_ordinal, .kind = ds::AcquisitionEventKind::source_fault}};
    const std::string schedule = "source_fault:frame_" + std::to_string(fault_ordinal);
    ds::SimulatedNeuralSource simulated{generator,
                                        source_config(options, false, std::move(events))};
    MeasuredSource source{simulated, 0};
    PassthroughProcessor processor;
    LatencySink sink{0, 1};
    const auto payload_bytes =
        static_cast<std::size_t>(simulated.schema().signals().front().max_block_bytes);
    auto config = pipeline_config(payload_bytes);
    NativeStreamRunner runner{simulated.schema().clone(), config, source, processor, sink};

    const auto before = process_counters();
    if (runner.prepare() != StreamStatus::ok || runner.arm() != StreamStatus::ok ||
        runner.start() != StreamStatus::ok)
    {
        return false;
    }
    const auto status = runner.join();
    const auto stopped_at = now_ns();
    const auto after = process_counters();
    const auto fault = runner.primary_fault();
    const auto raised_at = source.faulted_at_ns();
    const auto case_passed = status == StreamStatus::source_failure && fault.has_value() &&
                             raised_at != 0 && stopped_at >= raised_at &&
                             runner.outstanding_frames() == 0;
    std::vector<std::uint64_t> sample{case_passed ? stopped_at - raised_at : 0};

    Result result{
        .benchmark = "simulated_device_fault_stop",
        .metric = "source_failure_to_runner_stopped",
        .summary = summarize(sample),
        .allocation_tracking = "not_measured",
        .fault_schedule = schedule,
        .repetitions = 1,
        .frame_pool_capacity = config.required_buffer_count(),
        .frame_pool_payload_bytes = pool_payload_bytes(config),
        .ingress_high_water_mark = runner.stats().ingress_high_water_mark,
        .outstanding_frames = runner.outstanding_frames(),
        .process_cpu_seconds = std::max(0.0, after.cpu_seconds - before.cpu_seconds),
        .peak_resident_memory_bytes = after.peak_rss_bytes,
        .passed = case_passed,
    };
    describe(result, options);
    emit(result);
    return case_passed;
}

// ---------------------------------------------------------------------------

[[nodiscard]] bool parse_options(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        const auto value = [&]() -> std::string_view
        { return i + 1 < argc ? std::string_view{argv[i + 1]} : std::string_view{}; };
        if (argument == "--channels" && !value().empty())
        {
            options.channels = static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (argument == "--fs" && !value().empty())
        {
            options.fs = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--frame-samples" && !value().empty())
        {
            options.frame_samples =
                static_cast<std::uint32_t>(std::strtoul(argv[++i], nullptr, 10));
        }
        else if (argument == "--generation-iterations" && !value().empty())
        {
            options.generation_iterations = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--read-iterations" && !value().empty())
        {
            options.read_iterations = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--pipeline-frames" && !value().empty())
        {
            options.pipeline_frames = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--warmup-frames" && !value().empty())
        {
            options.warmup_frames = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--paced-seconds" && !value().empty())
        {
            options.paced_seconds = std::strtod(argv[++i], nullptr);
        }
        else if (argument == "--cancel-repetitions" && !value().empty())
        {
            options.cancel_repetitions = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--loss-period-frames" && !value().empty())
        {
            options.loss_period_frames = std::strtoull(argv[++i], nullptr, 10);
        }
        else if (argument == "--loss-samples" && !value().empty())
        {
            options.loss_samples = std::strtoull(argv[++i], nullptr, 10);
        }
        else
        {
            std::cerr << "unknown simulation benchmark option: " << argument << '\n';
            return false;
        }
    }
    return options.channels != 0 && options.fs != 0 && options.frame_samples != 0 &&
           options.generation_iterations != 0 && options.read_iterations != 0 &&
           options.pipeline_frames != 0 && options.paced_seconds > 0.0 &&
           options.cancel_repetitions != 0 && options.loss_period_frames != 0 &&
           options.loss_samples != 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parse_options(argc, argv, options))
    {
        return 2;
    }
    for (int i = 0; i < argc; ++i)
    {
        if (i != 0)
        {
            g_command.push_back(' ');
        }
        g_command.append(argv[i]);
    }

    auto passed = benchmark_generation(options);
    passed = benchmark_source_read(options) && passed;
    passed = benchmark_pacing(options) && passed;
    passed = benchmark_discontinuity(options) && passed;
    passed = benchmark_cancel(options) && passed;
    passed = benchmark_native_sink(options) && passed;
    passed = benchmark_fault_stop(options) && passed;
    return passed ? 0 : 1;
}
