/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "benchmark_report.h"
#include "fft_dispatch.h"
#include "fir_filter_adapter.h"
#include "hilbert_envelope_adapter.h"
#include "iir_filter_adapter.h"
#include "lmp_feature_adapter.h"
#include "multitaper_bandpower_adapter.h"
#include "multitaper_internal.h"
#include "resampler_adapter.h"
#include "sos_filter_adapter.h"
#include "spatial_reference_adapter.h"

#include <neurale/runtime/runtime_info.h>
#include <neurale/signal/fir.h>
#include <neurale/signal/iir.h>
#include <neurale/signal/representations.h>
#include <neurale/streaming/linear_processor_chain.h>
#include <neurale/streaming/runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{
using neurale::benchmark::emit_json_string;

using namespace neurale::streaming;
using Clock = std::chrono::steady_clock;

constexpr std::size_t source_rate_hz = 4'000;
constexpr std::size_t resampled_rate_hz = 1'000;
constexpr std::size_t channels = 256;
constexpr std::size_t frame_period_us = 1'000;
constexpr std::size_t samples_per_frame = 4;
constexpr std::size_t frames_per_second = 1'000;
constexpr std::size_t feature_window_ms = 100;
constexpr std::size_t feature_shift_ms = 10;
constexpr std::size_t iir_design_order = 2;
constexpr std::size_t fir_order = 24;
constexpr std::size_t fir_taps = fir_order + 1;
constexpr std::size_t notch_sections = 3;
constexpr double notch_half_width_hz = 1.0;
constexpr double bandpower_time_bandwidth = 4.0;
constexpr std::size_t bandpower_tapers = 7;
constexpr std::size_t resampler_filter_order = 64;
constexpr double resampler_cutoff_hz = 450.0;
constexpr std::array<double, 3> notch_centers_hz{100.0, 150.0, 200.0};
constexpr std::array<double, 6> identity_sos{1.0, 0.0, 0.0, 1.0, 0.0, 0.0};

enum class FilterKind
{
    none,
    sos,
    iir,
    fir
};

enum class FeatureKind
{
    lmp,
    hilbert,
    bandpower
};

struct ChainCase
{
    FilterKind bandpass{};
    FeatureKind feature{};
    bool resample{};
};

struct Options
{
    std::size_t warmup_frames{frames_per_second};
    std::size_t measurement_frames{60 * frames_per_second};
    std::size_t runs{3};
};

struct LatencySummary
{
    std::uint64_t minimum{};
    std::uint64_t median{};
    std::uint64_t p95{};
    std::uint64_t p99{};
    std::uint64_t maximum{};
};

struct CaseResult
{
    ChainCase chain{};
    std::size_t run_idx{};
    LatencySummary latency{};
    std::uint64_t elapsed_ns{};
    std::size_t input_frames{};
    std::size_t output_frames{};
    std::size_t output_observations{};
    std::size_t window_samples{};
    std::size_t shift_samples{};
    std::size_t fft_length{};
    std::size_t max_fan_out{};
    std::size_t frame_pool_bound{};
    std::uint64_t digest{};
};

[[nodiscard]] std::uint64_t now_ns() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

[[nodiscard]] std::string_view filter_name(FilterKind kind) noexcept
{
    switch (kind)
    {
    case FilterKind::none:
        return "none";
    case FilterKind::sos:
        return "sos";
    case FilterKind::iir:
        return "iir";
    case FilterKind::fir:
        return "fir";
    }
    return "unknown";
}

[[nodiscard]] std::string_view feature_name(FeatureKind kind) noexcept
{
    switch (kind)
    {
    case FeatureKind::lmp:
        return "lmp";
    case FeatureKind::hilbert:
        return "hilbert";
    case FeatureKind::bandpower:
        return "bandpower";
    }
    return "unknown";
}

[[nodiscard]] std::array<double, 2> feature_band(FeatureKind kind) noexcept
{
    return kind == FeatureKind::lmp ? std::array<double, 2>{1.0, 5.0}
                                    : std::array<double, 2>{70.0, 200.0};
}

[[nodiscard]] std::size_t next_power_of_two(std::size_t value)
{
    std::size_t result = 1;
    while (result < value)
    {
        if (result > std::numeric_limits<std::size_t>::max() / 2)
        {
            throw std::overflow_error("feature FFT length overflows size_t");
        }
        result *= 2;
    }
    return result;
}

[[nodiscard]] std::vector<std::string> make_channel_names()
{
    std::vector<std::string> result;
    result.reserve(channels);
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
        result.push_back("channel-" + std::to_string(channel));
    }
    return result;
}

[[nodiscard]] std::vector<std::string> make_feature_names(std::string_view prefix)
{
    std::vector<std::string> result;
    result.reserve(channels);
    for (std::size_t channel = 0; channel < channels; ++channel)
    {
        result.push_back(std::string{prefix} + ":channel-" + std::to_string(channel));
    }
    return result;
}

[[nodiscard]] std::vector<double> make_notch_sos()
{
    std::vector<double> result(notch_sections * 6);
    constexpr auto nyquist = static_cast<double>(source_rate_hz) / 2.0;
    constexpr auto normalized_bandwidth = 2.0 * notch_half_width_hz / nyquist;
    for (std::size_t i = 0; i < notch_centers_hz.size(); ++i)
    {
        const auto zpk =
            neurale::signal::notch_zpk(notch_centers_hz[i] / nyquist, normalized_bandwidth);
        neurale::signal::zpk2sos(zpk.z, zpk.p, zpk.k.real(),
                                 std::span<double>{result}.subspan(i * 6, 6));
    }
    return result;
}

[[nodiscard]] std::vector<double> make_resampler_filter()
{
    std::vector<double> result(resampler_filter_order + 1);
    const std::array cutoff{resampler_cutoff_hz / (static_cast<double>(source_rate_hz) / 2.0)};
    neurale::signal::firwin(resampler_filter_order, cutoff, "lowpass", "hamming", true, result);
    return result;
}

[[nodiscard]] StreamSchema make_schema()
{
    const std::array signals{SignalSchema{
        11,
        SignalDType::float64,
        channels,
        samples_per_frame,
        samples_per_frame,
        {source_rate_hz, 1},
        41,
        SignalLayout::sample_major,
        DeviceTickTracking::sample_counter,
        PhysicalUnit::volts,
        17,
        19,
        23,
    }};
    return StreamSchema{29, signals};
}

[[nodiscard]] RealtimeConfig make_config()
{
    RealtimeConfig config;
    config.platform.mode = RealtimeConfigMode::strict;
    config.platform.prefault_pools = true;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 8;
    config.pool_capacity.processor_owned = 4;
    config.pool_capacity.critical_edge_capacity = 8;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.reserve = 4;
    config.buffer_size = samples_per_frame * channels * sizeof(double);
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 2;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 3;
    config.max_flush_outputs = 32;
    config.fault_history_capacity = 8;
    config.source_stall_timeout = RealtimeDuration{60'000'000'000};
    config.max_ingress_dwell = RealtimeDuration{60'000'000'000};
    config.processor_execution_deadline = RealtimeDuration{60'000'000'000};
    config.max_source_to_actuator_age = RealtimeDuration{60'000'000'000};
    config.max_output_age = RealtimeDuration{60'000'000'000};
    config.actuator_deadline = RealtimeDuration{60'000'000'000};
    config.shutdown_deadline = RealtimeDuration{60'000'000'000};
    config.watchdog_period = RealtimeDuration{10'000'000};
    return config;
}

class BenchmarkSafety final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason) noexcept override
    {
        inhibited_.store(true, std::memory_order_release);
        return StreamStatus::ok;
    }

    StreamStatus release() noexcept override
    {
        inhibited_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::atomic<bool> inhibited_{true};
};

class SyntheticSource final : public NativeFrameSource
{
  public:
    SyntheticSource(std::size_t warmup_frames, std::size_t measurement_frames,
                    std::atomic<std::size_t>& processed, std::atomic<bool>& measurement_started,
                    std::atomic<std::uint64_t>& measurement_start_ns)
        : warmup_frames_(warmup_frames), n_frames_(warmup_frames + measurement_frames),
          processed_(processed), measurement_started_(measurement_started),
          measurement_start_ns_(measurement_start_ns), samples_(samples_per_frame * channels)
    {
        for (std::size_t sample = 0; sample < samples_per_frame; ++sample)
        {
            for (std::size_t channel = 0; channel < channels; ++channel)
            {
                samples_[sample * channels + channel] =
                    std::sin(static_cast<double>(sample) * 0.17) +
                    static_cast<double>(channel % 16) * 0.01;
            }
        }
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        while (idx_ != 0 && processed_.load(std::memory_order_acquire) < idx_ &&
               !cancelled_.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        if (idx_ == n_frames_)
        {
            return StreamStatus::end_of_stream;
        }
        if (idx_ == warmup_frames_)
        {
            measurement_start_ns_.store(now_ns(), std::memory_order_release);
            measurement_started_.store(true, std::memory_order_release);
        }

        const auto sample_start = static_cast<SampleIndex>(idx_ * samples_per_frame);
        const auto received = now_ns();
        frame.header() = FrameHeader{
            .session_id = 5,
            .sequence = idx_,
            .host_received_ns = received,
            .source_tick = sample_start,
            .valid_until_ns = std::numeric_limits<HostTimeNs>::max(),
            .schema_id = 29,
            .source_clock_domain = 41,
            .flags =
                FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
        };
        constexpr auto n_scalars = samples_per_frame * channels;
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = sample_start,
            .device_tick_start = sample_start,
            .payload_offset = 0,
            .payload_byte_count = n_scalars * sizeof(double),
            .signal_id = 11,
            .n_samples = samples_per_frame,
            .clock_sync =
                ClockSyncSnapshot{
                    .device_tick_reference = sample_start,
                    .host_time_reference_ns = received,
                    .device_tick_rate = {source_rate_hz, 1},
                    .uncertainty_ns = 1,
                    .clock_domain = 41,
                    .generation = 1,
                    .flags = ClockSyncFlags::synchronized,
                },
        };
        std::memcpy(frame.payload_storage().data(), samples_.data(), n_scalars * sizeof(double));
        const auto status = frame.set_used_sizes(1, n_scalars * sizeof(double));
        if (status == StreamStatus::ok)
        {
            ++idx_;
        }
        return status;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }

    StreamStatus reset() noexcept override
    {
        idx_ = 0;
        processed_.store(0, std::memory_order_release);
        measurement_started_.store(false, std::memory_order_release);
        measurement_start_ns_.store(0, std::memory_order_release);
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::size_t warmup_frames_{};
    std::size_t n_frames_{};
    std::atomic<std::size_t>& processed_;
    std::atomic<bool>& measurement_started_;
    std::atomic<std::uint64_t>& measurement_start_ns_;
    std::vector<double> samples_;
    std::size_t idx_{};
    std::atomic<bool> cancelled_{};
};

class CountingProcessor final : public NativeFrameProcessor
{
  public:
    CountingProcessor(NativeFrameProcessor& delegate, std::atomic<std::size_t>& processed) noexcept
        : delegate_(delegate), processed_(processed)
    {
    }

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return delegate_.prepare(context);
    }

    StreamStatus process(FrameBorrow& frame, FrameEmitter& emitter) noexcept override
    {
        const auto status = delegate_.process(frame, emitter);
        processed_.fetch_add(1, std::memory_order_release);
        return status;
    }

    StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept override
    {
        return delegate_.handle_discontinuity(discontinuity);
    }

    StreamStatus flush(FrameEmitter& emitter) noexcept override
    {
        return delegate_.flush(emitter);
    }

    StreamStatus reset() noexcept override
    {
        processed_.store(0, std::memory_order_release);
        return delegate_.reset();
    }

  private:
    NativeFrameProcessor& delegate_;
    std::atomic<std::size_t>& processed_;
};

class MeasuringConsumer final : public NativeFrameConsumer
{
  public:
    MeasuringConsumer(std::size_t capacity, std::atomic<bool>& measurement_started)
        : measurement_started_(measurement_started), latencies_(capacity)
    {
    }

    StreamStatus consume(FrameView frame) noexcept override
    {
        for (const auto value : frame.payload)
        {
            digest_ ^= static_cast<std::uint8_t>(value);
            digest_ *= std::uint64_t{1'099'511'628'211};
        }
        if (measurement_started_.load(std::memory_order_acquire) && measured_ < latencies_.size())
        {
            const auto current = now_ns();
            latencies_[measured_++] = current >= frame.header.host_received_ns
                                          ? current - frame.header.host_received_ns
                                          : 0;
            if (!frame.blocks.empty())
            {
                observations_ += frame.blocks.front().n_samples;
            }
        }
        ++frames_;
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
        frames_ = 0;
        measured_ = 0;
        observations_ = 0;
        digest_ = 1'469'598'103'934'665'603ULL;
        std::fill(latencies_.begin(), latencies_.end(), 0);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::span<std::uint64_t> latencies() noexcept
    {
        return {latencies_.data(), measured_};
    }
    [[nodiscard]] std::size_t frames() const noexcept
    {
        return frames_;
    }
    [[nodiscard]] std::size_t measured() const noexcept
    {
        return measured_;
    }
    [[nodiscard]] std::size_t observations() const noexcept
    {
        return observations_;
    }
    [[nodiscard]] std::uint64_t digest() const noexcept
    {
        return digest_;
    }

  private:
    std::atomic<bool>& measurement_started_;
    std::vector<std::uint64_t> latencies_;
    std::size_t frames_{};
    std::size_t measured_{};
    std::size_t observations_{};
    std::uint64_t digest_{1'469'598'103'934'665'603ULL};
};

struct ChainFixture
{
    std::unique_ptr<neurale::pipeline::SpatialReferenceAdapter> reference;
    std::unique_ptr<neurale::pipeline::SosFilterAdapter> notch;
    std::unique_ptr<NativeFrameProcessor> bandpass;
    std::unique_ptr<neurale::pipeline::ResamplerAdapter> resampler;
    std::unique_ptr<NativeFrameProcessor> feature;
    std::vector<NativeFrameProcessor*> stages;
    std::unique_ptr<LinearProcessorChain> chain;
    std::size_t window_samples{};
    std::size_t shift_samples{};
    std::size_t fft_length{};
};

[[nodiscard]] ChainFixture make_chain(const ChainCase& test_case)
{
    ChainFixture result;
    // Every chain is referenced first. A physical acquisition chain re-references
    // before it filters, and keeping it out of the case axis is deliberate: the
    // stage is unconditional here, so these numbers are not comparable with
    // JSONL from before it was added.
    result.reference = std::make_unique<neurale::pipeline::SpatialReferenceAdapter>(
        std::span<const std::size_t>{});
    result.stages.push_back(result.reference.get());
    result.notch = std::make_unique<neurale::pipeline::SosFilterAdapter>(make_notch_sos());
    result.stages.push_back(result.notch.get());

    if (test_case.bandpass != FilterKind::none)
    {
        const auto band = feature_band(test_case.feature);
        switch (test_case.bandpass)
        {
        case FilterKind::sos:
            result.bandpass = std::make_unique<neurale::pipeline::SosFilterAdapter>(
                neurale::signal::iir_sos("butterworth", iir_design_order, band, "bandpass",
                                         static_cast<double>(source_rate_hz), 1.0, 40.0));
            break;
        case FilterKind::iir:
        {
            auto coefs = neurale::signal::iir_tf("butterworth", iir_design_order, band, "bandpass",
                                                 static_cast<double>(source_rate_hz), 1.0, 40.0);
            result.bandpass =
                std::make_unique<neurale::pipeline::IirFilterAdapter>(coefs.num, coefs.den);
            break;
        }
        case FilterKind::fir:
        {
            std::vector<double> taps(fir_taps);
            const std::array normalized{band[0] / (static_cast<double>(source_rate_hz) / 2.0),
                                        band[1] / (static_cast<double>(source_rate_hz) / 2.0)};
            neurale::signal::firwin(fir_order, normalized, "bandpass", "hamming", true, taps);
            result.bandpass = std::make_unique<neurale::pipeline::FirFilterAdapter>(
                std::span<const double>{taps});
            break;
        }
        case FilterKind::none:
            break;
        }
        result.stages.push_back(result.bandpass.get());
    }

    if (test_case.resample)
    {
        const auto filter = make_resampler_filter();
        result.resampler = std::make_unique<neurale::pipeline::ResamplerAdapter>(31, 1, 4, filter);
        result.stages.push_back(result.resampler.get());
    }

    const auto feature_rate = test_case.resample ? resampled_rate_hz : source_rate_hz;
    result.window_samples = feature_rate * feature_window_ms / 1'000;
    result.shift_samples = feature_rate * feature_shift_ms / 1'000;
    result.fft_length = next_power_of_two(result.window_samples);
    auto channel_names = make_channel_names();
    switch (test_case.feature)
    {
    case FeatureKind::lmp:
        result.feature = std::make_unique<neurale::pipeline::LmpFeatureAdapter>(
            neurale::pipeline::LmpFeatureAdapterConfig{
                .output_schema_id = 37,
                .output_signal_id = 43,
                .feature_set_id = 47,
                .feature_unit = {53, "V", "volts"},
                .feature_names = make_feature_names("lmp-1-5Hz"),
                .source_stream = "synthetic-voltage",
                .algorithm_version = "physical-benchmark-1",
                .window_samples = result.window_samples,
                .shift_samples = result.shift_samples,
                .sos = {identity_sos.begin(), identity_sos.end()},
                .sos_sections = 1,
            });
        break;
    case FeatureKind::hilbert:
        result.feature = std::make_unique<neurale::pipeline::HilbertEnvelopeAdapter>(
            neurale::pipeline::HilbertEnvelopeAdapterConfig{
                .output_schema_id = 37,
                .output_signal_id = 43,
                .feature_set_id = 47,
                .feature_unit_id = 53,
                .band_names = {"70-200Hz"},
                .channel_names = std::move(channel_names),
                .source_stream = "synthetic-voltage",
                .algorithm_version = "physical-benchmark-1",
                .window_samples = result.window_samples,
                .shift_samples = result.shift_samples,
                .sos = {identity_sos.begin(), identity_sos.end()},
                .n_bands = 1,
                .sos_sections = 1,
                .fft_length = result.fft_length,
            });
        break;
    case FeatureKind::bandpower:
        result.feature = std::make_unique<neurale::pipeline::MultitaperBandpowerAdapter>(
            neurale::pipeline::MultitaperBandpowerAdapterConfig{
                .output_schema_id = 37,
                .output_signal_id = 43,
                .feature_set_id = 47,
                .feature_unit_id = 53,
                .bands = {{"70-200Hz", 70.0, 200.0}},
                .channel_names = std::move(channel_names),
                .source_stream = "synthetic-voltage",
                .algorithm_version = "physical-benchmark-1",
                .window_samples = result.window_samples,
                .shift_samples = result.shift_samples,
                .time_bandwidth = bandpower_time_bandwidth,
                .n_tapers = bandpower_tapers,
                .fft_length = result.fft_length,
                .weighting = neurale::signal::MultitaperWeighting::adaptive,
            });
        break;
    }
    result.stages.push_back(result.feature.get());
    result.chain = std::make_unique<LinearProcessorChain>(result.stages);
    return result;
}

[[nodiscard]] std::size_t percentile_index(std::size_t size, std::size_t numerator) noexcept
{
    return ((size - 1) * numerator + 9'999) / 10'000;
}

[[nodiscard]] LatencySummary summarize(std::span<std::uint64_t> samples)
{
    if (samples.empty())
    {
        throw std::runtime_error("physical benchmark emitted no measured feature frames");
    }
    std::sort(samples.begin(), samples.end());
    return {
        .minimum = samples.front(),
        .median = samples[percentile_index(samples.size(), 5'000)],
        .p95 = samples[percentile_index(samples.size(), 9'500)],
        .p99 = samples[percentile_index(samples.size(), 9'900)],
        .maximum = samples.back(),
    };
}

[[nodiscard]] CaseResult run_case(const ChainCase& test_case, const Options& options,
                                  std::size_t run_idx)
{
    auto fixture = make_chain(test_case);
    auto schema = make_schema();
    auto config = make_config();
    const auto contract = fixture.chain->prepare({
        .input_schema = schema,
        .max_process_outputs = config.max_process_outputs,
        .max_flush_outputs = config.max_flush_outputs,
        .available_frame_pool_leases = config.required_buffer_count(),
    });
    std::atomic<std::size_t> processed{};
    std::atomic<bool> measurement_started{};
    std::atomic<std::uint64_t> measurement_start_ns{};
    SyntheticSource source{options.warmup_frames, options.measurement_frames, processed,
                           measurement_started, measurement_start_ns};
    CountingProcessor processor{*fixture.chain, processed};
    MeasuringConsumer consumer{options.measurement_frames + config.max_flush_outputs,
                               measurement_started};
    SteadyNativeClock clock;
    BenchmarkSafety safety;
    NativeStreamRunner runtime{schema.clone(), config, source, processor, consumer, clock, safety};
    if (runtime.prepare() != StreamStatus::ok || runtime.arm() != StreamStatus::ok)
    {
        throw std::runtime_error("physical benchmark failed to prepare or arm");
    }
    const auto status = runtime.run();
    const auto end_ns = now_ns();
    const auto start_ns = measurement_start_ns.load(std::memory_order_acquire);
    if (status != StreamStatus::ok || start_ns == 0 || end_ns <= start_ns ||
        runtime.outstanding_frames() != 0 || runtime.outstanding_discontinuities() != 0 ||
        runtime.primary_fault().has_value())
    {
        throw std::runtime_error("physical benchmark chain execution failed");
    }
    return {
        .chain = test_case,
        .run_idx = run_idx,
        .latency = summarize(consumer.latencies()),
        .elapsed_ns = end_ns - start_ns,
        .input_frames = options.measurement_frames,
        .output_frames = consumer.measured(),
        .output_observations = consumer.observations(),
        .window_samples = fixture.window_samples,
        .shift_samples = fixture.shift_samples,
        .fft_length = fixture.fft_length,
        .max_fan_out = contract.max_process_outputs_per_input,
        .frame_pool_bound = contract.required_resources.frame_pool_leases,
        .digest = consumer.digest(),
    };
}

[[nodiscard]] std::string_view feature_fft_backend_name(const CaseResult& result) noexcept
{
    switch (result.chain.feature)
    {
    case FeatureKind::lmp:
        return "none";
    case FeatureKind::hilbert:
        return neurale::signal::detail::fft_kernel_name(
            neurale::signal::detail::select_fft_kernel(result.fft_length));
    case FeatureKind::bandpower:
        return neurale::signal::detail::fft_kernel_name(
            neurale::signal::detail::select_multitaper_fft_kernel(
                result.window_samples, result.fft_length, true,
                neurale::signal::SpectralBackend::automatic));
    }
    return "unknown";
}

void emit_json(const CaseResult& result, const Options& options)
{
    static const auto build = neurale::runtime::build_info();
    static const auto cpu = neurale::runtime::cpu_info();
    static const auto threading = neurale::runtime::threading_info();
    const auto band = feature_band(result.chain.feature);
    const auto simulated_ns = static_cast<double>(result.input_frames) * frame_period_us * 1'000.0;
    const auto realtime_factor = simulated_ns / static_cast<double>(result.elapsed_ns);
    const auto frames_per_second_result =
        static_cast<double>(result.input_frames) * 1.0e9 / static_cast<double>(result.elapsed_ns);
    const auto scalar_samples_per_second =
        frames_per_second_result * static_cast<double>(samples_per_frame * channels);
    std::cout
        << "{\"schema_version\":3,"
        << "\"benchmark\":\"pipeline_realtime_physical_chain\","
        << "\"scenario\":\"acquisition_4khz_256ch_1ms\","
        << "\"run\":" << result.run_idx << ",\"runs\":" << options.runs << ",\"feature\":\""
        << feature_name(result.chain.feature) << "\",\"bandpass\":\""
        << filter_name(result.chain.bandpass) << "\",\"notch_representation\":\"sos\","
        << "\"reference\":\"car_mean_all_channels\","
        << "\"resampler\":" << (result.chain.resample ? "true" : "false")
        << ",\"source_dtype\":\"float64\",\"input_conversion_included\":false,"
        << "\"layout\":\"sample_major\",\"shape\":[4,256],"
        << "\"source_rate_hz\":4000,\"frame_period_us\":1000,"
        << "\"duration_seconds\":"
        << static_cast<double>(options.measurement_frames) / frames_per_second
        << ",\"warmup_seconds\":" << static_cast<double>(options.warmup_frames) / frames_per_second
        << ",\"input_frames\":" << result.input_frames
        << ",\"notch_centers_hz\":[100,150,200],\"notch_half_width_hz\":1,"
        << "\"notch_sections\":3,\"feature_band_hz\":[" << band[0] << ',' << band[1]
        << "],\"bandpass_design_order\":"
        << (result.chain.bandpass == FilterKind::none  ? 0
            : result.chain.bandpass == FilterKind::fir ? fir_order
                                                       : iir_design_order)
        << ",\"bandpass_effective_order\":"
        << (result.chain.bandpass == FilterKind::none  ? 0
            : result.chain.bandpass == FilterKind::fir ? fir_order
                                                       : 2 * iir_design_order)
        << ",\"bandpass_sos_sections\":"
        << (result.chain.bandpass == FilterKind::sos ? iir_design_order : 0)
        << ",\"bandpass_numerator_count\":"
        << (result.chain.bandpass == FilterKind::iir ? 2 * iir_design_order + 1 : 0)
        << ",\"bandpass_denominator_count\":"
        << (result.chain.bandpass == FilterKind::iir ? 2 * iir_design_order + 1 : 0)
        << ",\"fir_order\":" << fir_order << ",\"fir_taps\":" << fir_taps
        << ",\"resampled_rate_hz\":" << (result.chain.resample ? resampled_rate_hz : source_rate_hz)
        << ",\"resampler_ratio\":\"" << (result.chain.resample ? "1/4" : "1/1")
        << "\",\"resampler_filter_order\":" << (result.chain.resample ? resampler_filter_order : 0)
        << ",\"resampler_cutoff_hz\":" << (result.chain.resample ? resampler_cutoff_hz : 0.0)
        << ",\"window_ms\":100,\"shift_ms\":10,\"window_samples\":" << result.window_samples
        << ",\"shift_samples\":" << result.shift_samples << ",\"fft_length\":" << result.fft_length
        << ",\"bandpower_nw\":4,\"bandpower_tapers\":7,"
        << "\"bandpower_weighting\":\"adaptive\",\"latency_unit\":\"ns\","
        << "\"latency_samples\":" << result.output_frames << ",\"min\":" << result.latency.minimum
        << ",\"median\":" << result.latency.median << ",\"p95\":" << result.latency.p95
        << ",\"p99\":" << result.latency.p99 << ",\"max\":" << result.latency.maximum
        << ",\"elapsed_ns\":" << result.elapsed_ns
        << ",\"frames_per_second\":" << frames_per_second_result
        << ",\"scalar_samples_per_second\":" << scalar_samples_per_second
        << ",\"realtime_factor\":" << realtime_factor
        << ",\"output_frames\":" << result.output_frames
        << ",\"output_observations\":" << result.output_observations
        << ",\"max_fan_out\":" << result.max_fan_out
        << ",\"frame_pool_bound\":" << result.frame_pool_bound << ",\"digest\":" << result.digest
        << ",\"allocation_tracking\":\"not_measured\","
        << "\"provider_request\":\"automatic\",\"feature_fft_backend\":\""
        << feature_fft_backend_name(result) << "\",\"build_fft_backend\":\"" << build.fft_backend
        << "\",\"cpu_math_backend\":\"" << build.cpu_math_backend << "\",\"build_type\":\""
        << build.build_type << "\",\"native_version\":";
    emit_json_string(std::cout, build.version);
    std::cout << ",\"compiler\":";
    emit_json_string(std::cout, build.compiler);
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
              << ",\"python_callback\":false,\"gil_required\":false}\n";
}

[[nodiscard]] Options parse_options(int argc, char** argv)
{
    Options result;
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument{argv[i]};
        if (argument == "--warmup-seconds" && i + 1 < argc)
        {
            result.warmup_frames = std::strtoull(argv[++i], nullptr, 10) * frames_per_second;
        }
        else if (argument == "--duration-seconds" && i + 1 < argc)
        {
            result.measurement_frames = std::strtoull(argv[++i], nullptr, 10) * frames_per_second;
        }
        else if (argument == "--runs" && i + 1 < argc)
        {
            result.runs = std::strtoull(argv[++i], nullptr, 10);
        }
        else
        {
            throw std::invalid_argument("unknown physical-chain benchmark option");
        }
    }
    if (result.measurement_frames < frames_per_second || result.runs == 0)
    {
        throw std::invalid_argument(
            "physical-chain benchmark requires at least one second and one run");
    }
    return result;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        const auto options = parse_options(argc, argv);
        std::vector<ChainCase> cases;
        cases.reserve(14);
        for (const auto resample : {false, true})
        {
            for (const auto feature : {FeatureKind::lmp, FeatureKind::hilbert})
            {
                for (const auto filter : {FilterKind::sos, FilterKind::iir, FilterKind::fir})
                {
                    cases.push_back({filter, feature, resample});
                }
            }
            cases.push_back({FilterKind::none, FeatureKind::bandpower, resample});
        }
        for (std::size_t run = 0; run < options.runs; ++run)
        {
            for (const auto& test_case : cases)
            {
                emit_json(run_case(test_case, options, run), options);
            }
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
