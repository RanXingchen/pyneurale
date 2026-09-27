/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_tracker.h"
#include "feature_stack_adapter.h"
#include "fir_filter_adapter.h"
#include "hilbert_envelope_adapter.h"
#include "iir_filter_adapter.h"
#include "lmp_feature_adapter.h"
#include "multitaper_bandpower_adapter.h"
#include "resampler_adapter.h"
#include "sos_filter_adapter.h"
#include "spatial_reference_adapter.h"

// Private signal headers, included only to assert backend dispatch coverage --
// see verify_geometry_dispatch(). Nothing on the data plane uses them.
#include "fft_dispatch.h"
#include "realtime_utils.h"

#include <neurale/streaming/linear_processor_chain.h>
#include <neurale/streaming/runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace
{

using namespace neurale::streaming;
using Clock = std::chrono::steady_clock;

constexpr std::array<double, 6> filter_sos{0.8, 0.1, 0.05, 1.0, -0.1, 0.02};
constexpr std::array<double, 2> filter_b{0.8, 0.1};
constexpr std::array<double, 2> filter_a{1.0, -0.1};
constexpr std::array<double, 3> filter_fir{0.25, 0.5, 0.25};
constexpr std::array<double, 5> resampler_filter{0.05, 0.2, 0.5, 0.2, 0.05};
constexpr std::array<double, 6> feature_sos{1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
constexpr std::array<double, 12> hilbert_sos{1.0, 0.0,  0.0,   1.0, 0.0,  0.0,
                                             0.5, 0.25, 0.125, 1.0, -0.2, 0.05};

/// Fixed chain geometry. Everything that changes with channel count, frame
/// size, or transform length is derived from one of these, so a geometry can be
/// added without touching the schema, source, or adapter construction.
struct Geometry
{
    std::string_view name;
    std::size_t channels;
    std::size_t samples_per_frame;
    std::size_t feature_window_samples;
    std::size_t feature_shift_samples;
    std::size_t feature_fft_length;
};

// Both geometries run the full case matrix. They are not "small and large" --
// they sit on opposite sides of every documented native backend dispatch
// threshold, because a gate that never crosses a threshold proves nothing about
// the kernel on the other side of it:
//
//   channels >= kMklRealtimeMinChannels (128)  -> SOS and direct-form IIR
//                                                 switch to mkl_blas_df2t
//                                                 (realtime_utils.h)
//   fft_length >= 64  -> select_fft_kernel() returns mkl_dfti, which is what
//                        moves AnalyticSignalProcessor (Hilbert) onto MKL
//                        (fft_dispatch.cpp, analytic.cpp)
//   window_samples <= fft_length -> can_use_mkl_multitaper_psd_backend()
//                        (multitaper_mkl.cpp; note it has no length threshold
//                        and deliberately does not consult select_fft_kernel)
//
// `narrow` keeps the original builtin-kernel coverage. `wide` is the only
// configuration in which the allocation gate observes the MKL kernels at all,
// and it is meaningful only in an MKL build -- see the CI allocation gates.
constexpr Geometry narrow_geometry{"narrow", 2, 64, 32, 16, 32};
constexpr Geometry wide_geometry{"wide", 128, 64, 64, 32, 64};
constexpr std::array geometries{narrow_geometry, wide_geometry};

/// One selectable stage on a chain axis: a name and a factory.
///
/// The reference, filter and feature axes were three enumerations, each needing
/// an enumerator, a `switch` in a name function, a `switch` in `make_chain` and
/// a brace-list in `main` kept in step. Missing one of the four is silent: the
/// matrix simply never builds that chain, and nothing fails. A row here is all
/// four at once, so adding an algorithm to an axis cannot half-happen.
///
/// A factory that returns `nullptr` is how an axis spells "this stage is
/// absent" -- `make_chain` skips a null stage, which is what makes
/// `reference=none` an ordinary row rather than a special case.
struct StageOption
{
    std::string_view name;
    std::unique_ptr<NativeFrameProcessor> (*make)(const Geometry&);
};

/// The resampler stays a `bool` rather than a fourth axis of `StageOption`:
/// it is not only a stage, it also changes the runtime configuration and the
/// expected fan-out (`make_config`, `expected_fan_out`).
struct ChainCase
{
    const StageOption* filter{};
    const StageOption* feature{};
    bool resample{};
    const StageOption* reference{};
};

struct Options
{
    std::size_t warmups{4};
    std::size_t repetitions{16};
};

struct ContractResult
{
    ChainCase chain{};
    std::string_view geometry;
    std::uint64_t allocations{};
    std::uint64_t frames_consumed{};
    std::uint64_t discontinuities{};
    std::uint64_t ingress_high_water_mark{};
    std::uint64_t actuator_high_water_mark{};
    std::size_t max_fan_out{};
    std::size_t frame_pool_bound{};
};

[[nodiscard]] std::uint64_t now_ns() noexcept
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
            .count());
}

[[nodiscard]] StreamSchema make_schema(const Geometry& geometry)
{
    std::array signals{SignalSchema{
        11,
        SignalDType::float64,
        static_cast<std::uint32_t>(geometry.channels),
        static_cast<std::uint32_t>(geometry.samples_per_frame),
        static_cast<std::uint32_t>(geometry.samples_per_frame),
        {1'000, 1},
        41,
        SignalLayout::sample_major,
        DeviceTickTracking::sample_counter,
        PhysicalUnit::volts,
        17,
        19,
        23,
    }};
    for (std::size_t channel = 0; channel < geometry.channels; ++channel)
    {
        signals.front().channel_names.push_back("ch" + std::to_string(channel));
    }
    return StreamSchema{29, signals};
}

/// Channel labels for the feature descriptors. One per channel, built during
/// construction rather than on the data plane.
[[nodiscard]] std::vector<std::string> make_channel_names(const Geometry& geometry)
{
    std::vector<std::string> result;
    result.reserve(geometry.channels);
    for (std::size_t channel = 0; channel < geometry.channels; ++channel)
    {
        result.push_back("ch" + std::to_string(channel));
    }
    return result;
}

[[nodiscard]] std::vector<std::string> make_lmp_feature_names(const Geometry& geometry)
{
    std::vector<std::string> result;
    result.reserve(geometry.channels);
    for (std::size_t channel = 0; channel < geometry.channels; ++channel)
    {
        result.push_back("lmp:ch" + std::to_string(channel));
    }
    return result;
}

[[nodiscard]] RealtimeConfig make_config(const Geometry& geometry, bool resample)
{
    RealtimeConfig config;
    config.platform.mode = RealtimeConfigMode::strict;
    config.platform.prefault_pools = true;
    config.pool_capacity.source_owned = 1;
    config.pool_capacity.ingress_capacity = 8;
    config.pool_capacity.processor_owned = 4;
    config.pool_capacity.critical_edge_capacity = 8;
    config.pool_capacity.actuator_owned = 1;
    config.pool_capacity.reserve = 2;
    // Four times one raw input frame. The factor covers the widest intermediate
    // in the matrix: the 2/1 resampler doubles the sample count, and every
    // feature frame is far smaller than a resampled sampled-signal frame.
    config.buffer_size =
        geometry.samples_per_frame * geometry.channels * sizeof(double) * std::size_t{4};
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 4;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = resample ? 2 : 1;
    config.max_flush_outputs = resample ? 2 : 1;
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

class RecordingSafety final : public SafetyController
{
  public:
    StreamStatus inhibit(SafetyReason reason) noexcept override
    {
        last_reason_.store(reason, std::memory_order_relaxed);
        inhibited_.store(true, std::memory_order_release);
        inhibit_count_.fetch_add(1, std::memory_order_relaxed);
        return StreamStatus::ok;
    }

    StreamStatus release() noexcept override
    {
        inhibited_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] bool inhibited() const noexcept
    {
        return inhibited_.load(std::memory_order_acquire);
    }

    [[nodiscard]] SafetyReason last_reason() const noexcept
    {
        return last_reason_.load(std::memory_order_relaxed);
    }

  private:
    std::atomic<bool> inhibited_{true};
    std::atomic<SafetyReason> last_reason_{SafetyReason::startup};
    std::atomic<std::size_t> inhibit_count_{};
};

class SyntheticSource final : public NativeFrameSource
{
  public:
    SyntheticSource(const Geometry& geometry, std::size_t n_frames, std::size_t allocation_warmups,
                    bool track_allocations, std::atomic<std::size_t>& consumed,
                    std::atomic<bool>& measurement_started, bool inject_gap = true) noexcept
        : geometry_(geometry), n_frames_(n_frames), allocation_warmups_(allocation_warmups),
          track_allocations_(track_allocations), consumed_(consumed),
          measurement_started_(measurement_started), inject_gap_(inject_gap)
    {
    }

    StreamStatus read(MutableFrame& frame) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        while (idx_ != 0 && consumed_.load(std::memory_order_acquire) < idx_ &&
               !cancelled_.load(std::memory_order_acquire))
        {
            std::this_thread::yield();
        }
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        if (fail_at_.has_value() && idx_ == *fail_at_)
        {
            return StreamStatus::source_failure;
        }
        if (stall_at_.has_value() && idx_ == *stall_at_)
        {
            while (!cancelled_.load(std::memory_order_acquire))
            {
                std::this_thread::yield();
            }
            return StreamStatus::stopped;
        }
        if (idx_ == n_frames_)
        {
            return StreamStatus::end_of_stream;
        }
        if (idx_ == allocation_warmups_)
        {
            if (track_allocations_)
            {
                neurale::benchmark::reset_allocation_count();
                neurale::benchmark::set_allocation_tracking(true);
            }
            measurement_started_.store(true, std::memory_order_release);
        }

        const auto gap = inject_gap_ && idx_ >= 4 ? std::size_t{1} : 0;
        const auto sequence = idx_ + gap;
        const auto sample_start =
            static_cast<SampleIndex>((idx_ + gap) * geometry_.samples_per_frame);
        frame.header() = FrameHeader{
            .session_id = 5,
            .sequence = sequence,
            .host_received_ns = now_ns(),
            .source_tick = sample_start,
            .valid_until_ns = std::numeric_limits<HostTimeNs>::max(),
            .schema_id = 29,
            .source_clock_domain = 41,
            .flags =
                FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
        };
        const auto n_scalars = geometry_.samples_per_frame * geometry_.channels;
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = sample_start,
            .device_tick_start = sample_start,
            .payload_offset = 0,
            .payload_byte_count = n_scalars * sizeof(double),
            .signal_id = 11,
            .n_samples = static_cast<std::uint32_t>(geometry_.samples_per_frame),
            .clock_sync =
                ClockSyncSnapshot{
                    .device_tick_reference = sample_start,
                    .host_time_reference_ns = frame.header().host_received_ns,
                    .device_tick_rate = {1'000, 1},
                    .uncertainty_ns = 1,
                    .clock_domain = 41,
                    .generation = 1,
                    .flags = ClockSyncFlags::synchronized,
                },
        };
        auto* output = reinterpret_cast<double*>(frame.payload_storage().data());
        for (std::size_t sample = 0; sample < geometry_.samples_per_frame; ++sample)
        {
            for (std::size_t channel = 0; channel < geometry_.channels; ++channel)
            {
                const auto phase = static_cast<double>(sample_start + sample) * 0.013;
                output[sample * geometry_.channels + channel] =
                    std::sin(phase) + static_cast<double>(channel) * 0.25;
            }
        }
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
        neurale::benchmark::set_allocation_tracking(false);
        idx_ = 0;
        consumed_.store(0, std::memory_order_release);
        measurement_started_.store(false, std::memory_order_release);
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    void fail_at(std::size_t value) noexcept
    {
        fail_at_ = value;
    }
    void stall_at(std::size_t value) noexcept
    {
        stall_at_ = value;
    }

  private:
    const Geometry& geometry_;
    std::size_t n_frames_{};
    std::size_t allocation_warmups_{};
    bool track_allocations_{};
    std::atomic<std::size_t>& consumed_;
    std::atomic<bool>& measurement_started_;
    bool inject_gap_{};
    std::size_t idx_{};
    std::optional<std::size_t> fail_at_;
    std::optional<std::size_t> stall_at_;
    std::atomic<bool> cancelled_{};
};

class MeasuringConsumer final : public NativeFrameConsumer
{
  public:
    MeasuringConsumer(std::size_t latency_capacity, bool stop_allocation_tracking,
                      std::atomic<std::size_t>& consumed, std::atomic<bool>& measurement_started)
        : stop_allocation_tracking_(stop_allocation_tracking), consumed_(consumed),
          measurement_started_(measurement_started), latencies_(latency_capacity)
    {
    }

    StreamStatus consume(FrameView frame) noexcept override
    {
        if (fail_at_.has_value() && frame_count_ == *fail_at_)
        {
            return StreamStatus::consumer_failure;
        }
        for (const auto value : frame.payload)
        {
            digest_ ^= static_cast<std::uint8_t>(value);
            digest_ *= std::uint64_t{1'099'511'628'211};
        }
        if (measurement_started_.load(std::memory_order_acquire) && measured_ < latencies_.size())
        {
            const auto received = frame.header.host_received_ns;
            const auto now = now_ns();
            latencies_[measured_++] = now >= received ? now - received : 0;
        }
        ++frame_count_;
        consumed_.store(frame_count_, std::memory_order_release);
        return StreamStatus::ok;
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        ++discontinuities_;
        return StreamStatus::ok;
    }

    StreamStatus flush() noexcept override
    {
        if (stop_allocation_tracking_)
        {
            neurale::benchmark::set_allocation_tracking(false);
        }
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        frame_count_ = 0;
        consumed_.store(0, std::memory_order_release);
        discontinuities_ = 0;
        measured_ = 0;
        digest_ = 1'469'598'103'934'665'603ULL;
        std::fill(latencies_.begin(), latencies_.end(), 0);
        return StreamStatus::ok;
    }

    void fail_at(std::size_t value) noexcept
    {
        fail_at_ = value;
    }

    [[nodiscard]] std::size_t frame_count() const noexcept
    {
        return frame_count_;
    }
    [[nodiscard]] std::size_t discontinuities() const noexcept
    {
        return discontinuities_;
    }
    [[nodiscard]] std::size_t measured() const noexcept
    {
        return measured_;
    }
    [[nodiscard]] std::uint64_t digest() const noexcept
    {
        return digest_;
    }
    [[nodiscard]] std::span<std::uint64_t> latencies() noexcept
    {
        return {latencies_.data(), measured_};
    }

  private:
    bool stop_allocation_tracking_{};
    std::atomic<std::size_t>& consumed_;
    std::atomic<bool>& measurement_started_;
    std::vector<std::uint64_t> latencies_;
    std::optional<std::size_t> fail_at_;
    std::size_t frame_count_{};
    std::size_t discontinuities_{};
    std::size_t measured_{};
    std::uint64_t digest_{1'469'598'103'934'665'603ULL};
};

class FaultingProcessor final : public NativeFrameProcessor
{
  public:
    FaultingProcessor(NativeFrameProcessor& delegate, std::size_t fail_at) noexcept
        : delegate_(delegate), fail_at_(fail_at)
    {
    }

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return delegate_.prepare(context);
    }

    StreamStatus process(FrameBorrow& frame, FrameEmitter& emitter) noexcept override
    {
        if (count_++ == fail_at_)
        {
            return StreamStatus::processor_failure;
        }
        return delegate_.process(frame, emitter);
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
        count_ = 0;
        return delegate_.reset();
    }

  private:
    NativeFrameProcessor& delegate_;
    std::size_t fail_at_{};
    std::size_t count_{};
};

struct ChainFixture
{
    std::unique_ptr<NativeFrameProcessor> reference;
    std::unique_ptr<NativeFrameProcessor> filter;
    std::unique_ptr<neurale::pipeline::ResamplerAdapter> resampler;
    std::unique_ptr<NativeFrameProcessor> feature;
    std::vector<NativeFrameProcessor*> stages;
    std::unique_ptr<LinearProcessorChain> chain;
};

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_no_reference(const Geometry&)
{
    return nullptr;
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_mean_reference(const Geometry&)
{
    return std::make_unique<neurale::pipeline::SpatialReferenceAdapter>(
        std::span<const std::size_t>{}, neurale::pipeline::SpatialReferenceStatistic::mean);
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_median_reference(const Geometry&)
{
    return std::make_unique<neurale::pipeline::SpatialReferenceAdapter>(
        std::span<const std::size_t>{}, neurale::pipeline::SpatialReferenceStatistic::median);
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_sos_filter(const Geometry&)
{
    return std::make_unique<neurale::pipeline::SosFilterAdapter>(filter_sos);
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_iir_filter(const Geometry&)
{
    return std::make_unique<neurale::pipeline::IirFilterAdapter>(filter_b, filter_a);
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_fir_filter(const Geometry&)
{
    return std::make_unique<neurale::pipeline::FirFilterAdapter>(filter_fir);
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_lmp_feature(const Geometry& geometry)
{
    return std::make_unique<neurale::pipeline::LmpFeatureAdapter>(
        neurale::pipeline::LmpFeatureAdapterConfig{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .feature_set_id = 47,
            .feature_unit = {53, "V", "volts"},
            .feature_names = make_lmp_feature_names(geometry),
            .source_stream = "synthetic-voltage",
            .algorithm_version = "strict-validation-1",
            .window_samples = geometry.feature_window_samples,
            .shift_samples = geometry.feature_shift_samples,
            .sos = {feature_sos.begin(), feature_sos.end()},
            .sos_sections = 1,
        });
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_hilbert_feature(const Geometry& geometry)
{
    return std::make_unique<neurale::pipeline::HilbertEnvelopeAdapter>(
        neurale::pipeline::HilbertEnvelopeAdapterConfig{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .feature_set_id = 47,
            .feature_unit_id = 53,
            .band_names = {"low", "high"},
            .channel_names = make_channel_names(geometry),
            .source_stream = "synthetic-voltage",
            .algorithm_version = "strict-validation-1",
            .window_samples = geometry.feature_window_samples,
            .shift_samples = geometry.feature_shift_samples,
            .sos = {hilbert_sos.begin(), hilbert_sos.end()},
            .n_bands = 2,
            .sos_sections = 1,
            .fft_length = geometry.feature_fft_length,
        });
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_bandpower_feature(const Geometry& geometry)
{
    return std::make_unique<neurale::pipeline::MultitaperBandpowerAdapter>(
        neurale::pipeline::MultitaperBandpowerAdapterConfig{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .feature_set_id = 47,
            .feature_unit_id = 53,
            .bands = {{"low", 0.0, 100.0}, {"high", 100.0, 400.0}},
            .channel_names = make_channel_names(geometry),
            .source_stream = "synthetic-voltage",
            .algorithm_version = "strict-validation-1",
            .window_samples = geometry.feature_window_samples,
            .shift_samples = geometry.feature_shift_samples,
            .time_bandwidth = 3.5,
            .n_tapers = 5,
            .fft_length = geometry.feature_fft_length,
            .weighting = neurale::signal::MultitaperWeighting::adaptive,
        });
}

[[nodiscard]] std::unique_ptr<NativeFrameProcessor> make_feature_stack(const Geometry& geometry)
{
    neurale::pipeline::LmpBranchConfig lmp{
        .cutoff_hz = 4.0,
        .filter_order = 2,
    };
    neurale::pipeline::HilbertEnvelopeBranchConfig hilbert{
        .bands = {{"low", 8.0, 30.0}, {"high", 30.0, 100.0}},
        .filter_order = 2,
    };
    neurale::pipeline::MultitaperBandpowerBranchConfig pmtm{
        .bands = {{"low", 0.0, 100.0}, {"high", 100.0, 400.0}},
        .time_bandwidth = 3.5,
        .n_tapers = 5,
        .weighting = neurale::signal::MultitaperWeighting::adaptive,
    };
    return std::make_unique<neurale::pipeline::FeatureStackAdapter>(
        neurale::pipeline::FeatureStackAdapterConfig{
            .algorithm_version = "strict-feature-stack-1",
            // The optional 2x resampler is the highest prepared input rate in this
            // matrix. Size the duration so both rates stay within the fixed FFT.
            .window_ns = geometry.feature_window_samples * 500'000ULL,
            .update_interval_ns = geometry.feature_shift_samples * 500'000ULL,
            .branches = {std::move(lmp), std::move(hilbert), std::move(pmtm)},
        });
}

// The three chain axes. Adding an algorithm to one of them is a row here and
// nothing else; the case count and the failure message follow.
constexpr std::array reference_options{
    StageOption{"none", &make_no_reference},
    StageOption{"mean", &make_mean_reference},
    StageOption{"median", &make_median_reference},
};
constexpr std::array filter_options{
    StageOption{"sos", &make_sos_filter},
    StageOption{"iir", &make_iir_filter},
    StageOption{"fir", &make_fir_filter},
};
constexpr std::array feature_options{
    StageOption{"lmp", &make_lmp_feature},
    StageOption{"hilbert", &make_hilbert_feature},
    StageOption{"bandpower", &make_bandpower_feature},
    StageOption{"feature-stack", &make_feature_stack},
};

[[nodiscard]] ChainFixture make_chain(const ChainCase& test_case, const Geometry& geometry)
{
    ChainFixture result;
    result.reference = test_case.reference->make(geometry);
    result.filter = test_case.filter->make(geometry);
    if (test_case.resample)
    {
        result.resampler =
            std::make_unique<neurale::pipeline::ResamplerAdapter>(31, 2, 1, resampler_filter);
    }
    result.feature = test_case.feature->make(geometry);
    // Common referencing sits ahead of the bandpass: that is the physiological
    // order, and it is what makes the filter's per-channel state see
    // already-referenced samples rather than the shared component.
    if (result.reference != nullptr)
    {
        result.stages.push_back(result.reference.get());
    }
    result.stages.push_back(result.filter.get());
    if (result.resampler != nullptr)
    {
        result.stages.push_back(result.resampler.get());
    }
    result.stages.push_back(result.feature.get());
    result.chain = std::make_unique<LinearProcessorChain>(result.stages);
    return result;
}

[[nodiscard]] bool validate_run_state(const NativeStreamRunner& runtime,
                                      const MeasuringConsumer& consumer,
                                      std::size_t latency_capacity)
{
    const auto stats = runtime.stats();
    const auto valid = runtime.state() == RuntimeState::stopped &&
                       runtime.realtime_configuration_status().ok() &&
                       !runtime.primary_fault().has_value() && runtime.outstanding_frames() == 0 &&
                       runtime.outstanding_discontinuities() == 0 && stats.queue_overruns == 0 &&
                       stats.pool_exhaustions == 0 && stats.ingress_high_water_mark <= 8 &&
                       stats.actuator_queue_high_water_mark <= 8 && consumer.measured() != 0 &&
                       consumer.measured() <= latency_capacity && consumer.discontinuities() == 1 &&
                       stats.discontinuities == 1;
    if (!valid)
    {
        std::cerr << "run state: state=" << static_cast<int>(runtime.state())
                  << " realtime=" << runtime.realtime_configuration_status().ok()
                  << " fault=" << runtime.primary_fault().has_value()
                  << " outstanding_frames=" << runtime.outstanding_frames()
                  << " outstanding_discontinuities=" << runtime.outstanding_discontinuities()
                  << " overruns=" << stats.queue_overruns
                  << " pool_exhaustions=" << stats.pool_exhaustions
                  << " ingress_hwm=" << stats.ingress_high_water_mark
                  << " actuator_hwm=" << stats.actuator_queue_high_water_mark
                  << " measured=" << consumer.measured() << " capacity=" << latency_capacity
                  << " discontinuities=" << consumer.discontinuities() << '\n';
    }
    return valid;
}

[[nodiscard]] std::optional<ContractResult> run_steady_case(const ChainCase& test_case,
                                                            const Geometry& geometry,
                                                            const Options& options,
                                                            bool track_allocations)
{
    neurale::benchmark::set_allocation_tracking(false);
    neurale::benchmark::reset_allocation_count();
    auto fixture = make_chain(test_case, geometry);
    auto schema = make_schema(geometry);
    auto config = make_config(geometry, test_case.resample);
    const auto contract = fixture.chain->prepare({
        .input_schema = schema,
        .max_process_outputs = config.max_process_outputs,
        .max_flush_outputs = config.max_flush_outputs,
        .available_frame_pool_leases = config.required_buffer_count(),
    });
    const auto expected_fan_out = test_case.resample ? std::size_t{2} : 1;
    const auto latency_capacity = options.repetitions * expected_fan_out + config.max_flush_outputs;
    if (contract.max_process_outputs_per_input != expected_fan_out ||
        contract.max_flush_outputs > config.max_flush_outputs ||
        contract.required_resources.frame_pool_leases == 0 ||
        contract.required_resources.frame_pool_leases > config.required_buffer_count())
    {
        std::cerr << "prepared contract: process=" << contract.max_process_outputs_per_input
                  << " flush=" << contract.max_flush_outputs
                  << " leases=" << contract.required_resources.frame_pool_leases
                  << " available=" << config.required_buffer_count() << '\n';
        return std::nullopt;
    }

    std::atomic<std::size_t> consumed{};
    std::atomic<bool> measurement_started{};
    SyntheticSource source{geometry,        options.warmups + options.repetitions,
                           options.warmups, track_allocations,
                           consumed,        measurement_started};
    MeasuringConsumer consumer{latency_capacity, track_allocations, consumed, measurement_started};
    SteadyNativeClock clock;
    RecordingSafety safety;
    NativeStreamRunner runtime{schema.clone(), config, source, *fixture.chain,
                               consumer,       clock,  safety};
    const auto prepare_status = runtime.prepare();
    const auto arm_status = prepare_status == StreamStatus::ok ? runtime.arm() : prepare_status;
    const auto run_status = arm_status == StreamStatus::ok ? runtime.run() : arm_status;
    if (prepare_status != StreamStatus::ok || arm_status != StreamStatus::ok ||
        run_status != StreamStatus::ok || !validate_run_state(runtime, consumer, latency_capacity))
    {
        std::cerr << "run status: prepare=" << static_cast<int>(prepare_status)
                  << " arm=" << static_cast<int>(arm_status)
                  << " run=" << static_cast<int>(run_status) << '\n';
        const auto realtime = runtime.realtime_configuration_status();
        if (const auto fault = runtime.primary_fault(); fault.has_value())
        {
            std::cerr << "primary fault code=" << static_cast<int>(fault->code)
                      << " status=" << static_cast<int>(fault->status)
                      << " stage=" << static_cast<int>(fault->stage) << " detail=" << fault->detail
                      << '\n';
        }
        std::cerr << "realtime memory requested=" << realtime.memory.requested
                  << " applied=" << realtime.memory.applied
                  << " unsupported=" << realtime.memory.unsupported
                  << " failed=" << realtime.memory.failed << '\n';
        for (std::size_t role = 0; role < realtime.threads.size(); ++role)
        {
            const auto& value = realtime.threads[role];
            std::cerr << "realtime thread " << role << " requested=" << value.requested
                      << " applied=" << value.applied << " unsupported=" << value.unsupported
                      << " failed=" << value.failed << " native_error=" << value.native_error
                      << '\n';
        }
        neurale::benchmark::set_allocation_tracking(false);
        return std::nullopt;
    }
    neurale::benchmark::set_allocation_tracking(false);
    const auto allocations = neurale::benchmark::allocation_count();
    const auto first_digest = consumer.digest();
    const auto first_frames = consumer.frame_count();
    const auto first_measurements = consumer.measured();
    const auto first_discontinuities = consumer.discontinuities();
    const auto first_stats = runtime.stats();

    const auto reset_status = runtime.reset();
    const auto second_prepare_status =
        reset_status == StreamStatus::ok ? runtime.prepare() : reset_status;
    const auto second_arm_status =
        second_prepare_status == StreamStatus::ok ? runtime.arm() : second_prepare_status;
    const auto second_run_status =
        second_arm_status == StreamStatus::ok ? runtime.run() : second_arm_status;
    if (reset_status != StreamStatus::ok || second_prepare_status != StreamStatus::ok ||
        second_arm_status != StreamStatus::ok || second_run_status != StreamStatus::ok ||
        !validate_run_state(runtime, consumer, latency_capacity))
    {
        std::cerr << "repeat status: reset=" << static_cast<int>(reset_status)
                  << " prepare=" << static_cast<int>(second_prepare_status)
                  << " arm=" << static_cast<int>(second_arm_status)
                  << " run=" << static_cast<int>(second_run_status) << '\n';
        neurale::benchmark::set_allocation_tracking(false);
        return std::nullopt;
    }
    neurale::benchmark::set_allocation_tracking(false);
    if (consumer.digest() != first_digest || consumer.frame_count() != first_frames ||
        consumer.measured() != first_measurements ||
        consumer.discontinuities() != first_discontinuities ||
        neurale::benchmark::allocation_count() != allocations)
    {
        std::cerr << "repeat mismatch: digest=" << (consumer.digest() == first_digest)
                  << " frames=" << (consumer.frame_count() == first_frames)
                  << " discontinuities=" << (consumer.discontinuities() == first_discontinuities)
                  << " allocations=" << neurale::benchmark::allocation_count()
                  << " first_allocations=" << allocations << '\n';
        return std::nullopt;
    }

    return ContractResult{
        .chain = test_case,
        .geometry = geometry.name,
        .allocations = allocations,
        .frames_consumed = first_stats.frames_consumed,
        .discontinuities = first_stats.discontinuities,
        .ingress_high_water_mark = first_stats.ingress_high_water_mark,
        .actuator_high_water_mark = first_stats.actuator_queue_high_water_mark,
        .max_fan_out = contract.max_process_outputs_per_input,
        .frame_pool_bound = contract.required_resources.frame_pool_leases,
    };
}

enum class FaultRole
{
    source,
    processor,
    consumer
};

[[nodiscard]] bool validate_fault_case(const ChainCase& test_case, const Geometry& geometry,
                                       FaultRole role)
{
    neurale::benchmark::set_allocation_tracking(false);
    auto fixture = make_chain(test_case, geometry);
    auto schema = make_schema(geometry);
    auto config = make_config(geometry, test_case.resample);
    std::atomic<std::size_t> consumed{};
    std::atomic<bool> measurement_started{};
    SyntheticSource source{geometry, 16, 0, false, consumed, measurement_started, false};
    MeasuringConsumer consumer{16, false, consumed, measurement_started};
    std::unique_ptr<FaultingProcessor> faulting_processor;
    NativeFrameProcessor* processor = fixture.chain.get();
    if (role == FaultRole::source)
    {
        source.fail_at(3);
    }
    else if (role == FaultRole::processor)
    {
        faulting_processor = std::make_unique<FaultingProcessor>(*fixture.chain, 3);
        processor = faulting_processor.get();
    }
    else
    {
        consumer.fail_at(3);
    }
    SteadyNativeClock clock;
    RecordingSafety safety;
    NativeStreamRunner runtime{std::move(schema), config, source, *processor,
                               consumer,          clock,  safety};
    if (runtime.prepare() != StreamStatus::ok || runtime.arm() != StreamStatus::ok)
    {
        return false;
    }
    const auto status = runtime.run();
    const auto fault = runtime.primary_fault();
    const auto expected_status = role == FaultRole::source      ? StreamStatus::source_failure
                                 : role == FaultRole::processor ? StreamStatus::processor_failure
                                                                : StreamStatus::consumer_failure;
    const auto expected_stage = role == FaultRole::source      ? FaultStage::source
                                : role == FaultRole::processor ? FaultStage::processor
                                                               : FaultStage::actuator;
    const auto valid = status == expected_status && fault.has_value() &&
                       fault->stage == expected_stage && safety.inhibited() &&
                       runtime.outstanding_frames() == 0 &&
                       runtime.outstanding_discontinuities() == 0;
    if (!valid)
    {
        std::cerr << "fault validation role=" << static_cast<int>(role)
                  << " status=" << static_cast<int>(status)
                  << " expected=" << static_cast<int>(expected_status)
                  << " fault=" << fault.has_value()
                  << " stage=" << (fault.has_value() ? static_cast<int>(fault->stage) : -1)
                  << " expected_stage=" << static_cast<int>(expected_stage)
                  << " inhibited=" << safety.inhibited()
                  << " frames=" << runtime.outstanding_frames()
                  << " discontinuities=" << runtime.outstanding_discontinuities() << '\n';
    }
    return valid;
}

[[nodiscard]] bool validate_watchdog_case(const ChainCase& test_case, const Geometry& geometry)
{
    neurale::benchmark::set_allocation_tracking(false);
    auto fixture = make_chain(test_case, geometry);
    auto schema = make_schema(geometry);
    auto config = make_config(geometry, test_case.resample);
    config.source_stall_timeout = RealtimeDuration{20'000'000};
    config.watchdog_period = RealtimeDuration{1'000'000};
    std::atomic<std::size_t> consumed{};
    std::atomic<bool> measurement_started{};
    SyntheticSource source{geometry, 16, 0, false, consumed, measurement_started, false};
    source.stall_at(3);
    MeasuringConsumer consumer{16, false, consumed, measurement_started};
    SteadyNativeClock clock;
    RecordingSafety safety;
    NativeStreamRunner runtime{std::move(schema), config, source, *fixture.chain,
                               consumer,          clock,  safety};
    if (runtime.prepare() != StreamStatus::ok || runtime.arm() != StreamStatus::ok)
    {
        return false;
    }
    const auto status = runtime.run();
    const auto fault = runtime.primary_fault();
    const auto valid = status == StreamStatus::deadline_exceeded && fault.has_value() &&
                       fault->code == FaultCode::source_stall &&
                       fault->stage == FaultStage::source && safety.inhibited() &&
                       safety.last_reason() == SafetyReason::source_stall &&
                       runtime.stats().deadline_faults >= 1 && runtime.outstanding_frames() == 0 &&
                       runtime.outstanding_discontinuities() == 0;
    if (!valid)
    {
        std::cerr << "watchdog validation status=" << static_cast<int>(status)
                  << " fault=" << fault.has_value()
                  << " code=" << (fault.has_value() ? static_cast<int>(fault->code) : -1)
                  << " stage=" << (fault.has_value() ? static_cast<int>(fault->stage) : -1)
                  << " inhibited=" << safety.inhibited()
                  << " reason=" << static_cast<int>(safety.last_reason())
                  << " deadline_faults=" << runtime.stats().deadline_faults
                  << " frames=" << runtime.outstanding_frames()
                  << " discontinuities=" << runtime.outstanding_discontinuities() << '\n';
    }
    return valid;
}

/// Prove that the geometries still sit where they are supposed to sit relative
/// to the native dispatch thresholds, and report which kernels this build
/// actually covers.
///
/// This exists because the matrix passing is not by itself evidence about the
/// MKL kernels: before the wide geometry the gate ran only at two channels and
/// a 32-point transform, so it never constructed an MKL kernel at all and could
/// not have observed an allocation in one. The checks below call the same
/// predicates the kernels dispatch on, so weakening a geometry fails here
/// instead of silently narrowing coverage.
[[nodiscard]] bool verify_geometry_dispatch()
{
    using neurale::signal::detail::fft_kernel_available;
    using neurale::signal::detail::FftKernel;
    using neurale::signal::detail::select_fft_kernel;
    using neurale::signal::detail::use_mkl_realtime_blas;

    const auto mkl_build = fft_kernel_available(FftKernel::mkl_dfti);

    // The narrow geometry must stay entirely on the builtin kernels; it is the
    // regression guard for the scalar paths.
    if (use_mkl_realtime_blas(narrow_geometry.channels) ||
        select_fft_kernel(narrow_geometry.feature_fft_length) != FftKernel::builtin)
    {
        std::cerr << "narrow geometry no longer selects the builtin kernels\n";
        return false;
    }

    // The wide geometry must cross both thresholds. These are build-independent
    // properties of the geometry itself.
    if (!use_mkl_realtime_blas(wide_geometry.channels))
    {
        std::cerr << "wide geometry channel count " << wide_geometry.channels
                  << " no longer reaches the MKL realtime BLAS threshold\n";
        return false;
    }
    if (wide_geometry.feature_window_samples > wide_geometry.feature_fft_length)
    {
        std::cerr << "wide geometry window " << wide_geometry.feature_window_samples
                  << " exceeds fft length " << wide_geometry.feature_fft_length
                  << ", which disables the MKL multitaper backend\n";
        return false;
    }
    if (mkl_build && select_fft_kernel(wide_geometry.feature_fft_length) != FftKernel::mkl_dfti)
    {
        std::cerr << "wide geometry fft length " << wide_geometry.feature_fft_length
                  << " no longer selects mkl_dfti\n";
        return false;
    }

    std::cout << "geometry " << narrow_geometry.name << ": channels=" << narrow_geometry.channels
              << " fft=" << narrow_geometry.feature_fft_length << " sos/iir=scalar fft=builtin\n";
    std::cout << "geometry " << wide_geometry.name << ": channels=" << wide_geometry.channels
              << " fft=" << wide_geometry.feature_fft_length
              << " sos/iir=" << (mkl_build ? "mkl_blas_df2t" : "scalar")
              << " fft=" << (mkl_build ? "mkl_dfti" : "builtin")
              << " multitaper=" << (mkl_build ? "mkl" : "builtin") << '\n';
    if (!mkl_build)
    {
        std::cout << "note: this is not an MKL build, so no run here is evidence about the "
                     "MKL kernels; the MKL allocation gate must run in an MKL build\n";
    }
    return true;
}

[[nodiscard]] bool allocation_tracker_self_test()
{
    neurale::benchmark::AllocationScope scope;
    void* ptr = ::operator new(64);
    ::operator delete(ptr);
    scope.stop();
    return scope.count() != 0;
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        constexpr Options options{};
        auto allocation_only = false;
        if (argc == 2 && std::string_view{argv[1]} == "--allocation-only")
        {
            allocation_only = true;
        }
        else if (argc != 2 || std::string_view{argv[1]} != "--contract-only")
        {
            throw std::invalid_argument(
                "strict realtime contract test requires --contract-only or --allocation-only");
        }
        if (!verify_geometry_dispatch())
        {
            return 2;
        }
        if (allocation_only && !allocation_tracker_self_test())
        {
            std::cerr << "allocation tracker self-test failed\n";
            return 2;
        }
        std::vector<ContractResult> results;
        results.reserve(geometries.size() * reference_options.size() * filter_options.size() *
                        feature_options.size() * 2);
        for (const auto& geometry : geometries)
        {
            for (const auto resample : {false, true})
            {
                for (const auto& reference : reference_options)
                {
                    for (const auto& filter : filter_options)
                    {
                        for (const auto& feature : feature_options)
                        {
                            const ChainCase test_case{&filter, &feature, resample, &reference};
                            auto result =
                                run_steady_case(test_case, geometry, options, allocation_only);
                            if (!result.has_value() ||
                                (allocation_only && result->allocations != 0))
                            {
                                std::cerr
                                    << "strict-chain validation failed: geometry=" << geometry.name
                                    << ", reference=" << reference.name
                                    << ", filter=" << filter.name << ", resampler=" << resample
                                    << ", feature=" << feature.name << ", allocations="
                                    << (result.has_value() ? result->allocations : std::uint64_t{0})
                                    << '\n';
                                return 2;
                            }
                            results.push_back(*result);
                        }
                    }
                }
            }
        }

        auto fault_and_watchdog_valid = true;
        if (!allocation_only)
        {
            // Fault and watchdog behaviour is transport-level and geometry
            // independent, so it runs once per case on the narrow geometry
            // rather than once per (geometry, case) pair.
            for (const auto& result : results)
            {
                if (result.geometry != narrow_geometry.name)
                {
                    continue;
                }
                const auto& test_case = result.chain;
                const auto source_valid =
                    validate_fault_case(test_case, narrow_geometry, FaultRole::source);
                const auto processor_valid =
                    validate_fault_case(test_case, narrow_geometry, FaultRole::processor);
                const auto consumer_valid =
                    validate_fault_case(test_case, narrow_geometry, FaultRole::consumer);
                const auto watchdog_valid = validate_watchdog_case(test_case, narrow_geometry);
                fault_and_watchdog_valid = source_valid && processor_valid && consumer_valid &&
                                           watchdog_valid && fault_and_watchdog_valid;
            }
        }
        if (!fault_and_watchdog_valid)
        {
            std::cerr << "strict-chain fault/watchdog validation failed\n";
            return 3;
        }
        // Printed on success so that a change to the matrix is visible in the
        // log rather than only in the runtime. The axes multiply, so an
        // accidentally dropped row shows up here as a smaller count long before
        // anyone notices which chain stopped being covered.
        std::cout << "chains validated: " << results.size() << " (" << geometries.size()
                  << " geometry x " << reference_options.size() << " reference x "
                  << filter_options.size() << " filter x " << feature_options.size()
                  << " feature x 2 resample)\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        neurale::benchmark::set_allocation_tracking(false);
        std::cerr << error.what() << '\n';
        return 1;
    }
}
