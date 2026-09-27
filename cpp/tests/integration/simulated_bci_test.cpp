/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "allocation_tracker.h"
#include "check_returns.h"
#include "fir_filter_adapter.h"
#include "hilbert_envelope_adapter.h"
#include "iir_filter_adapter.h"
#include "kalman_decoder_adapter.h"
#include "lmp_feature_adapter.h"
#include "multitaper_bandpower_adapter.h"
#include "resampler_adapter.h"
#include "sos_filter_adapter.h"

#include <neurale/devices/simulation.h>
#include <neurale/signal/simulation.h>
#include <neurale/streaming/linear_processor_chain.h>
#include <neurale/streaming/runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace
{

namespace ds = neurale::devices::simulation;
namespace ss = neurale::signal::simulation;
using namespace neurale::streaming;

constexpr std::size_t channels = 2;
constexpr std::size_t total_samples = 512;
constexpr std::size_t capture_capacity = 32'768;
constexpr std::array<double, 6> filter_sos{0.8, 0.1, 0.05, 1.0, -0.1, 0.02};
constexpr std::array<double, 2> filter_b{0.8, 0.1};
constexpr std::array<double, 2> filter_a{1.0, -0.1};
constexpr std::array<double, 3> filter_fir{0.25, 0.5, 0.25};
constexpr std::array<double, 5> resampler_filter{0.05, 0.2, 0.5, 0.2, 0.05};
constexpr std::array<double, 6> feature_sos{1.0, 0.0, 0.0, 1.0, 0.0, 0.0};
constexpr std::array<double, 12> hilbert_sos{1.0, 0.0,  0.0,   1.0, 0.0,  0.0,
                                             0.5, 0.25, 0.125, 1.0, -0.2, 0.05};

[[nodiscard]] ss::SignalGenerator signal_generator()
{
    constexpr std::array<double, 4> freqs{7.0, 11.0, 83.0, 127.0};
    constexpr std::array<double, 4> amps{1.0, 0.75, 0.2, 0.3};
    constexpr std::array<double, 4> phases{0.0, 0.25, 0.5, -0.25};
    return ss::SignalGenerator::tones(channels, 1'000.0, 2, freqs, amps, phases);
}

[[nodiscard]] ds::SimulatedNeuralSourceConfig
source_config(std::uint32_t frame_samples, std::vector<ds::AcquisitionEvent> events = {})
{
    return {
        .session_id = 5,
        .schema_id = 29,
        .signal_id = 11,
        .clock_domain = 41,
        .nominal_samples_per_frame = frame_samples,
        .max_samples_per_frame = frame_samples,
        .fs = {1'000, 1},
        .physical_unit = PhysicalUnit::volts,
        .channel_set_id = 17,
        .calibration_id = 19,
        .reference_id = 23,
        .initial_sample_idx = 100,
        .total_sample_count = total_samples,
        .device_ticks = true,
        .initial_device_tick = 5'000,
        .clock_offset_ns = 250,
        .clock_drift_ppm = 100,
        .clock_sync_uncertainty_ns = 50,
        .paced = false,
        .events = std::move(events),
    };
}

[[nodiscard]] RealtimeConfig runtime_config(bool resample = false)
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
    config.buffer_size = 32'768;
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 8;
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

class CaptureConsumer final : public NativeFrameConsumer
{
  public:
    explicit CaptureConsumer(bool stop_tracking = false)
        : stop_tracking_(stop_tracking),
          values_(std::make_unique<std::array<double, capture_capacity>>()),
          sample_indices_(std::make_unique<std::array<SampleIndex, capture_capacity>>())
    {
    }

    StreamStatus consume(FrameView frame) noexcept override
    {
        if (frame.blocks.size() != 1)
        {
            return StreamStatus::consumer_failure;
        }
        const auto& block = frame.blocks.front();
        if (block.payload_byte_count % sizeof(double) != 0 ||
            value_count_ + block.payload_byte_count / sizeof(double) > values_->size() ||
            observation_count_ + block.n_samples > sample_indices_->size())
        {
            return StreamStatus::consumer_failure;
        }
        const auto count = block.payload_byte_count / sizeof(double);
        const auto* values = reinterpret_cast<const double*>(
            frame.payload.data() + static_cast<std::ptrdiff_t>(block.payload_offset));
        std::copy_n(values, count, values_->begin() + static_cast<std::ptrdiff_t>(value_count_));
        value_count_ += count;
        for (std::size_t i = 0; i < block.n_samples; ++i)
        {
            (*sample_indices_)[observation_count_++] = block.sample_idx_start + i;
        }
        if (frame_count_ < frame_sequences_.size())
        {
            frame_sequences_[frame_count_] = frame.header.sequence;
            block_starts_[frame_count_] = block.sample_idx_start;
            device_ticks_[frame_count_] = block.device_tick_start;
            sync_generations_[frame_count_] = block.clock_sync.generation;
        }
        ++frame_count_;
        return StreamStatus::ok;
    }

    StreamStatus handle_discontinuity(const Discontinuity& value) noexcept override
    {
        ++discontinuities_;
        last_gap_reason_ = value.reason;
        return StreamStatus::ok;
    }

    StreamStatus flush() noexcept override
    {
        if (stop_tracking_)
        {
            neurale::benchmark::set_allocation_tracking(false);
        }
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        value_count_ = 0;
        observation_count_ = 0;
        frame_count_ = 0;
        discontinuities_ = 0;
        last_gap_reason_ = GapReason::source_gap;
        std::fill(values_->begin(), values_->end(), 0.0);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::span<const double> values() const noexcept
    {
        return {values_->data(), value_count_};
    }
    [[nodiscard]] std::span<const SampleIndex> sample_indices() const noexcept
    {
        return {sample_indices_->data(), observation_count_};
    }
    [[nodiscard]] std::size_t frames() const noexcept
    {
        return frame_count_;
    }
    [[nodiscard]] std::size_t discontinuities() const noexcept
    {
        return discontinuities_;
    }
    [[nodiscard]] GapReason last_gap_reason() const noexcept
    {
        return last_gap_reason_;
    }
    [[nodiscard]] const auto& frame_sequences() const noexcept
    {
        return frame_sequences_;
    }
    [[nodiscard]] const auto& block_starts() const noexcept
    {
        return block_starts_;
    }
    [[nodiscard]] const auto& device_ticks() const noexcept
    {
        return device_ticks_;
    }
    [[nodiscard]] const auto& sync_generations() const noexcept
    {
        return sync_generations_;
    }

  private:
    bool stop_tracking_{};
    std::unique_ptr<std::array<double, capture_capacity>> values_;
    std::unique_ptr<std::array<SampleIndex, capture_capacity>> sample_indices_;
    std::array<std::uint64_t, 256> frame_sequences_{};
    std::array<SampleIndex, 256> block_starts_{};
    std::array<DeviceTick, 256> device_ticks_{};
    std::array<std::uint32_t, 256> sync_generations_{};
    std::size_t value_count_{};
    std::size_t observation_count_{};
    std::size_t frame_count_{};
    std::size_t discontinuities_{};
    GapReason last_gap_reason_{GapReason::source_gap};
};

class TrackingSource final : public NativeFrameSource
{
  public:
    TrackingSource(ds::SimulatedNeuralSource& source, std::size_t warmup_frames) noexcept
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
        successful_frames_ = 0;
        neurale::benchmark::set_allocation_tracking(false);
        return source_.reset();
    }

  private:
    StreamStatus finish(StreamStatus status) noexcept
    {
        if (status == StreamStatus::ok && ++successful_frames_ == warmup_frames_)
        {
            neurale::benchmark::reset_allocation_count();
            neurale::benchmark::set_allocation_tracking(true);
        }
        return status;
    }

    ds::SimulatedNeuralSource& source_;
    std::size_t warmup_frames_{};
    std::size_t successful_frames_{};
};

class PassthroughProcessor final : public NativeFrameProcessor
{
  public:
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema = context.input_schema.clone(),
            .output_schema = context.input_schema.clone(),
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = true,
            .required_resources = {.workspace_bytes = 0, .frame_pool_leases = 1},
        };
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

enum class ChainKind
{
    sos_lmp,
    fir_resampler_hilbert,
    iir_bandpower,
};

[[nodiscard]] std::vector<std::string> channel_names()
{
    return {"left", "right"};
}

struct ChainFixture
{
    std::unique_ptr<NativeFrameProcessor> filter;
    std::unique_ptr<neurale::pipeline::ResamplerAdapter> resampler;
    std::unique_ptr<NativeFrameProcessor> feature;
    std::vector<NativeFrameProcessor*> stages;
    std::unique_ptr<LinearProcessorChain> chain;
    bool uses_resampler{};
};

[[nodiscard]] ChainFixture make_feature_chain(ChainKind kind)
{
    ChainFixture result;
    if (kind == ChainKind::sos_lmp)
    {
        result.filter = std::make_unique<neurale::pipeline::SosFilterAdapter>(filter_sos);
        result.feature = std::make_unique<neurale::pipeline::LmpFeatureAdapter>(
            neurale::pipeline::LmpFeatureAdapterConfig{
                .output_schema_id = 37,
                .output_signal_id = 43,
                .feature_set_id = 47,
                .feature_unit = {53, "V", "volts"},
                .feature_names = {"lmp:left", "lmp:right"},
                .source_stream = "simulated-neural",
                .algorithm_version = "m7-06",
                .window_samples = 32,
                .shift_samples = 16,
                .sos = {feature_sos.begin(), feature_sos.end()},
                .sos_sections = 1,
            });
    }
    else if (kind == ChainKind::fir_resampler_hilbert)
    {
        result.filter = std::make_unique<neurale::pipeline::FirFilterAdapter>(filter_fir);
        result.resampler =
            std::make_unique<neurale::pipeline::ResamplerAdapter>(31, 2, 1, resampler_filter);
        result.uses_resampler = true;
        result.feature = std::make_unique<neurale::pipeline::HilbertEnvelopeAdapter>(
            neurale::pipeline::HilbertEnvelopeAdapterConfig{
                .output_schema_id = 37,
                .output_signal_id = 43,
                .feature_set_id = 47,
                .feature_unit_id = 53,
                .band_names = {"low", "high"},
                .channel_names = channel_names(),
                .source_stream = "simulated-neural",
                .algorithm_version = "m7-06",
                .window_samples = 64,
                .shift_samples = 32,
                .sos = {hilbert_sos.begin(), hilbert_sos.end()},
                .n_bands = 2,
                .sos_sections = 1,
                .fft_length = 64,
            });
    }
    else
    {
        result.filter = std::make_unique<neurale::pipeline::IirFilterAdapter>(filter_b, filter_a);
        result.feature = std::make_unique<neurale::pipeline::MultitaperBandpowerAdapter>(
            neurale::pipeline::MultitaperBandpowerAdapterConfig{
                .output_schema_id = 37,
                .output_signal_id = 43,
                .feature_set_id = 47,
                .feature_unit_id = 53,
                .bands = {{"low", 1.0, 70.0}, {"high", 70.0, 200.0}},
                .channel_names = channel_names(),
                .source_stream = "simulated-neural",
                .algorithm_version = "m7-06",
                .window_samples = 64,
                .shift_samples = 32,
                .time_bandwidth = 3.5,
                .n_tapers = 5,
                .fft_length = 64,
                .weighting = neurale::signal::MultitaperWeighting::adaptive,
            });
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

struct Capture
{
    std::vector<double> values;
    std::vector<SampleIndex> sample_indices;
    std::size_t discontinuities{};
};

/// Runs one prepared chain to end-of-stream. `track_allocations` swaps the
/// source for the tracking wrapper that arms the allocation counter once the
/// chain has reached steady state, so the count that survives is the one the
/// prepared topology produced, not its warm-up.
[[nodiscard]] Capture run_chain(ds::SimulatedNeuralSource& source, LinearProcessorChain& chain,
                                const RealtimeConfig& config, bool track_allocations,
                                std::string_view label)
{
    TrackingSource tracked{source, 4};
    CaptureConsumer consumer{track_allocations};
    NativeStreamRunner runner{source.schema().clone(), config,
                              track_allocations ? static_cast<NativeFrameSource&>(tracked)
                                                : static_cast<NativeFrameSource&>(source),
                              chain, consumer};
    if (runner.prepare() != StreamStatus::ok || runner.arm() != StreamStatus::ok ||
        runner.run() != StreamStatus::ok || runner.outstanding_frames() != 0 ||
        runner.outstanding_discontinuities() != 0 || runner.primary_fault().has_value())
    {
        std::cerr << "chain failed: " << label << " state=" << static_cast<int>(runner.state())
                  << " frames=" << runner.outstanding_frames()
                  << " gaps=" << runner.outstanding_discontinuities();
        if (const auto fault = runner.primary_fault(); fault.has_value())
        {
            std::cerr << " fault=" << static_cast<int>(fault->status)
                      << " detail=" << fault->detail;
        }
        std::cerr << '\n';
        neurale::benchmark::set_allocation_tracking(false);
        return {};
    }
    neurale::benchmark::set_allocation_tracking(false);
    return {
        .values = {consumer.values().begin(), consumer.values().end()},
        .sample_indices = {consumer.sample_indices().begin(), consumer.sample_indices().end()},
        .discontinuities = consumer.discontinuities(),
    };
}

[[nodiscard]] Capture run_feature_chain(ChainKind kind, std::uint32_t frame_samples,
                                        bool track_allocations = false,
                                        std::vector<ds::AcquisitionEvent> events = {})
{
    auto fixture = make_feature_chain(kind);
    ds::SimulatedNeuralSource source{signal_generator(),
                                     source_config(frame_samples, std::move(events))};
    constexpr std::array<std::string_view, 3> labels{"sos->lmp", "fir->resampler->hilbert",
                                                     "iir->bandpower"};
    return run_chain(source, *fixture.chain, runtime_config(fixture.uses_resampler),
                     track_allocations, labels[static_cast<std::size_t>(kind)]);
}

[[nodiscard]] bool nearly_equal(std::span<const double> left, std::span<const double> right)
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const auto scale = std::max({1.0, std::abs(left[i]), std::abs(right[i])});
        if (std::abs(left[i] - right[i]) > 1e-11 * scale)
        {
            return false;
        }
    }
    return true;
}

[[nodiscard]] neurale::models::LinearGaussianModelState model_state()
{
    neurale::models::LinearGaussianModelState state;
    state.state_dim = 2;
    state.observation_dim = 2;
    state.transition = {0.90, 0.05, -0.10, 0.95};
    state.transition_offset = {0.01, -0.02};
    state.observation = {1.0, 0.2, 0.5, -0.8};
    state.observation_offset = {0.05, -0.10};
    state.process_covariance = {0.020, 0.005, 0.005, 0.030};
    state.observation_covariance = {0.10, 0.01, 0.01, 0.15};
    state.initial_state = {0.10, -0.20};
    state.initial_covariance = {0.050, 0.010, 0.010, 0.060};
    return state;
}

[[nodiscard]] neurale::pipeline::KalmanDecoderAdapterConfig decoder_config()
{
    return {
        .output_schema_id = 57,
        .output_signal_id = 61,
        .output_channel_set_id = 67,
        .output_physical_unit = PhysicalUnit::dimensionless,
        .feature_set_id = 47,
        .selection = {0, 1},
        .selected_feature_names = {"lmp:left", "lmp:right"},
        .fitted_feature_contract =
            neurale::pipeline::FittedFeatureContract{
                .feature_names = {"lmp:left", "lmp:right"},
                .feature_unit_symbols = {"V", "V"},
                .observation_rate = {125, 2},
                .window_length_ns = 32'000'000,
                .shift_ns = 16'000'000,
                .algorithm_name = "lmp",
                .algorithm_version = "m7-06",
                .source_stream = "simulated-neural",
            },
        .scaling = neurale::pipeline::FeatureScaling::none,
        .model = model_state(),
        .innovation_jitter = 1e-9,
        .missing = neurale::pipeline::KalmanMissingPolicy::error,
    };
}

struct DecoderFixture
{
    std::unique_ptr<neurale::pipeline::LmpFeatureAdapter> feature;
    std::unique_ptr<neurale::pipeline::KalmanDecoderAdapter> decoder;
    std::array<NativeFrameProcessor*, 2> stages{};
    std::unique_ptr<LinearProcessorChain> chain;
};

[[nodiscard]] DecoderFixture make_decoder_chain()
{
    DecoderFixture result;
    result.feature = std::make_unique<neurale::pipeline::LmpFeatureAdapter>(
        neurale::pipeline::LmpFeatureAdapterConfig{
            .output_schema_id = 37,
            .output_signal_id = 43,
            .feature_set_id = 47,
            .feature_unit = {53, "V", "volts"},
            .feature_names = {"lmp:left", "lmp:right"},
            .source_stream = "simulated-neural",
            .algorithm_version = "m7-06",
            .window_samples = 32,
            .shift_samples = 16,
            .sos = {feature_sos.begin(), feature_sos.end()},
            .sos_sections = 1,
        });
    result.decoder = std::make_unique<neurale::pipeline::KalmanDecoderAdapter>(decoder_config());
    result.stages = {result.feature.get(), result.decoder.get()};
    result.chain = std::make_unique<LinearProcessorChain>(result.stages);
    return result;
}

[[nodiscard]] Capture run_decoder_chain(std::uint32_t frame_samples, bool track_allocations = false)
{
    auto fixture = make_decoder_chain();
    ds::SimulatedNeuralSource source{signal_generator(), source_config(frame_samples)};
    return run_chain(source, *fixture.chain, runtime_config(), track_allocations, "lmp->kalman");
}

[[nodiscard]] int test_chain_a_source_to_native_consumer()
{
    auto generator = signal_generator();
    auto config = source_config(37);
    config.total_sample_count = 37;
    ds::SimulatedNeuralSource source{generator, config};
    PassthroughProcessor processor;
    CaptureConsumer consumer;
    NativeStreamRunner runner{source.schema().clone(), runtime_config(), source, processor,
                              consumer};
    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.run() == StreamStatus::ok);
    CHECK(consumer.frames() == 1);
    CHECK(consumer.values().size() == 37 * channels);
    std::array<double, 37 * channels> expected{};
    CHECK(generator.generate(100, 37, expected) == ss::GenerationStatus::ok);
    CHECK(std::equal(expected.begin(), expected.end(), consumer.values().begin()));
    CHECK(consumer.frame_sequences()[0] == 0);
    CHECK(consumer.block_starts()[0] == 100);
    CHECK(consumer.device_ticks()[0] == 5'000);
    CHECK(consumer.sync_generations()[0] == 1);
    CHECK(runner.outstanding_frames() == 0);
    CHECK(runner.outstanding_discontinuities() == 0);

    const auto first = std::vector<double>{consumer.values().begin(), consumer.values().end()};
    CHECK(runner.reset() == StreamStatus::ok);
    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.run() == StreamStatus::ok);
    CHECK(std::equal(first.begin(), first.end(), consumer.values().begin()));
    return 0;
}

[[nodiscard]] int test_chain_b_pipeline_combinations()
{
    constexpr std::array kinds{ChainKind::sos_lmp, ChainKind::fir_resampler_hilbert,
                               ChainKind::iir_bandpower};
    for (const auto kind : kinds)
    {
        const auto chunks32 = run_feature_chain(kind, 32);
        // Repeatability before invariance. A chain whose second identical run
        // disagrees with its first is not producing a different answer for a
        // different frame size -- it is not producing an answer at all, and the
        // invariance check below would report that as a frame-size problem. The
        // multitaper chain failed exactly this way while MKL executed its
        // batched transform across threads.
        const auto repeat32 = run_feature_chain(kind, 32);
        CHECK(!chunks32.values.empty());
        CHECK(nearly_equal(chunks32.values, repeat32.values));
        CHECK(chunks32.sample_indices == repeat32.sample_indices);

        const auto chunks47 = run_feature_chain(kind, 47);
        CHECK(nearly_equal(chunks32.values, chunks47.values));
        CHECK(chunks32.sample_indices == chunks47.sample_indices);
        CHECK(chunks32.discontinuities == 0);
        CHECK(chunks47.discontinuities == 0);
    }

    std::vector<ds::AcquisitionEvent> events{
        {.frame_ordinal = 3, .kind = ds::AcquisitionEventKind::sample_loss, .n_samples = 7},
    };
    const auto with_gap = run_feature_chain(ChainKind::sos_lmp, 32, false, std::move(events));
    CHECK(!with_gap.values.empty());
    CHECK(with_gap.discontinuities == 1);
    return 0;
}

[[nodiscard]] int test_chain_c_feature_to_kalman()
{
    auto fixture = make_decoder_chain();
    ds::SimulatedNeuralSource source{signal_generator(), source_config(32)};
    CaptureConsumer consumer;
    NativeStreamRunner runner{source.schema().clone(), runtime_config(), source, *fixture.chain,
                              consumer};
    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.run() == StreamStatus::ok);
    CHECK(!consumer.values().empty());
    const auto first_values =
        std::vector<double>{consumer.values().begin(), consumer.values().end()};
    const auto first_indices = std::vector<SampleIndex>{consumer.sample_indices().begin(),
                                                        consumer.sample_indices().end()};
    CHECK(runner.reset() == StreamStatus::ok);
    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.run() == StreamStatus::ok);
    CHECK(nearly_equal(first_values, consumer.values()));
    CHECK(
        std::equal(first_indices.begin(), first_indices.end(), consumer.sample_indices().begin()));
    CHECK(runner.outstanding_frames() == 0);
    CHECK(runner.outstanding_discontinuities() == 0);
    return 0;
}

[[nodiscard]] int test_allocation_and_fault_cleanup()
{
    neurale::benchmark::set_allocation_tracking(false);
    const auto captured = run_feature_chain(ChainKind::sos_lmp, 16, true);
    CHECK(!captured.values.empty());
    CHECK(neurale::benchmark::allocation_count() == 0);

    // The decoder is a second terminal stage, not a variation of the first, so
    // the source -> feature -> decoder -> consumer chain has to produce its own
    // zero. Without this the evidence would cover only the feature path while
    // Chain C ran untracked.
    const auto decoded = run_decoder_chain(16, true);
    CHECK(!decoded.values.empty());
    CHECK(neurale::benchmark::allocation_count() == 0);

    auto fixture = make_decoder_chain();
    std::vector<ds::AcquisitionEvent> events{
        {.frame_ordinal = 4, .kind = ds::AcquisitionEventKind::source_fault},
    };
    ds::SimulatedNeuralSource source{signal_generator(), source_config(16, std::move(events))};
    CaptureConsumer consumer;
    NativeStreamRunner runner{source.schema().clone(), runtime_config(), source, *fixture.chain,
                              consumer};
    CHECK(runner.prepare() == StreamStatus::ok);
    CHECK(runner.arm() == StreamStatus::ok);
    CHECK(runner.run() == StreamStatus::source_failure);
    CHECK(runner.primary_fault().has_value());
    CHECK(runner.primary_fault()->status == StreamStatus::source_failure);
    CHECK(runner.outstanding_frames() == 0);
    CHECK(runner.outstanding_discontinuities() == 0);
    return 0;
}

} // namespace

int main()
{
    if (const auto status = test_chain_a_source_to_native_consumer(); status != 0)
    {
        return status;
    }
    if (const auto status = test_chain_b_pipeline_combinations(); status != 0)
    {
        return status;
    }
    if (const auto status = test_chain_c_feature_to_kalman(); status != 0)
    {
        return status;
    }
    return test_allocation_and_fault_cleanup();
}
