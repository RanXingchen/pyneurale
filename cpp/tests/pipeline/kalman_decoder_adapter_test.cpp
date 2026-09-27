/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

// What the native Kalman decoder adapter must agree with, and what it must
// refuse. The reference throughout is ``models::LinearGaussianModel`` driven in
// the order ``neurale.decoding.KalmanDecoder`` drives it -- select, scale,
// correct the first row of a segment in place, predict and correct every row
// after it -- because that is the decoder this adapter has to reproduce, and
// the Python one runs exactly this recursion through its bindings.

#include "allocation_tracker.h"
#include "check_returns.h"
#include "kalman_decoder_adapter.h"
#include "lmp_feature_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <neurale/features/online.h>
#include <neurale/models/state_space.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/linear_processor_chain.h>

// The shared tracker always covers global operator new and, on the strict
// Linux acceptance build, also wraps malloc/calloc/realloc. ASan/UBSan builds
// exercise the same adapter target without the linker wrapping.
namespace
{

using namespace neurale::streaming;
using neurale::models::LinearGaussianModel;
using neurale::models::LinearGaussianModelState;

constexpr std::uint32_t features = 4;
constexpr std::size_t state_dim = 2;
constexpr std::size_t observation_dim = 3;
constexpr std::uint32_t max_input_observations = 8;
constexpr std::size_t max_frames = 64;
constexpr std::size_t max_values = 4096;
constexpr double innovation_jitter = 1e-9;
constexpr std::uint64_t shift_ns = 2'000'000;
constexpr std::uint64_t first_observation_ns = 12'000'000;

/// Columns of the input feature vector the model consumes, out of order and
/// incomplete on purpose: a selection that is neither identity nor a prefix is
/// the only one whose misuse a width check could not catch.
constexpr std::array<std::size_t, observation_dim> selection{3, 0, 2};
constexpr std::array<double, observation_dim> scaler_center{0.5, -0.25, 1.0};
constexpr std::array<double, observation_dim> scaler_scale{2.0, 0.5, 1.5};

[[nodiscard]] LinearGaussianModelState model_state()
{
    LinearGaussianModelState state;
    state.state_dim = state_dim;
    state.observation_dim = observation_dim;
    state.transition = {0.90, 0.05, -0.10, 0.95};
    state.transition_offset = {0.01, -0.02};
    state.observation = {1.0, 0.2, 0.5, -0.8, -0.3, 0.6};
    state.observation_offset = {0.05, -0.10, 0.20};
    state.process_covariance = {0.020, 0.005, 0.005, 0.030};
    state.observation_covariance = {0.10, 0.01, 0.02, 0.01, 0.15, 0.03, 0.02, 0.03, 0.20};
    state.initial_state = {0.10, -0.20};
    state.initial_covariance = {0.050, 0.010, 0.010, 0.060};
    return state;
}

[[nodiscard]] neurale::pipeline::KalmanDecoderAdapterConfig decoder_config(
    neurale::pipeline::KalmanMissingPolicy missing = neurale::pipeline::KalmanMissingPolicy::error)
{
    return {
        .output_schema_id = 37,
        .output_signal_id = 43,
        .output_channel_set_id = 59,
        .output_physical_unit = PhysicalUnit::dimensionless,
        .feature_set_id = 47,
        .selection = {selection.begin(), selection.end()},
        .selected_feature_names = {"rate_3", "rate_0", "rate_2"},
        .fitted_feature_contract =
            neurale::pipeline::FittedFeatureContract{
                .feature_names = {"rate_0", "rate_1", "rate_2", "rate_3"},
                .feature_unit_symbols = {"Hz", "Hz", "Hz", "Hz"},
                .observation_rate = {500, 1},
                .window_length_ns = 4'000'000,
                .shift_ns = shift_ns,
                .algorithm_name = "rate",
                .algorithm_version = "1",
                .source_stream = "units",
            },
        .scaling = neurale::pipeline::FeatureScaling::standard,
        .scaler_center = {scaler_center.begin(), scaler_center.end()},
        .scaler_scale = {scaler_scale.begin(), scaler_scale.end()},
        .model = model_state(),
        .innovation_jitter = innovation_jitter,
        .missing = missing,
    };
}

[[nodiscard]] std::vector<FeatureSetDescriptor>
feature_sets(std::vector<std::string> names = {"rate_0", "rate_1", "rate_2", "rate_3"})
{
    return {FeatureSetDescriptor{
        .id = 47,
        .feature_names = std::move(names),
        .unit_ids = std::vector<UnitId>(features, 53),
        .source_stream_id = 11,
        .source_stream = "units",
        .algorithm_name = "rate",
        .algorithm_version = "1",
        .window_length_ns = 4'000'000,
        .shift_ns = shift_ns,
        .timestamp_reference = FeatureTimestampReference::window_center,
    }};
}

[[nodiscard]] std::vector<UnitDescriptor> units()
{
    return {UnitDescriptor{.id = 53, .symbol = "Hz", .description = "hertz"}};
}

[[nodiscard]] StreamSchema
make_schema(std::uint32_t channels = features, SignalKind kind = SignalKind::feature,
            SignalDType dtype = SignalDType::float64, FeatureSetId feature_set = 47,
            std::vector<std::string> names = {"rate_0", "rate_1", "rate_2", "rate_3"})
{
    const std::array signals{
        SignalSchema{
            11,
            dtype,
            channels,
            6,
            max_input_observations,
            {500, 1},
            41,
            SignalLayout::sample_major,
            DeviceTickTracking::sample_counter,
            PhysicalUnit::unspecified,
            17,
            19,
            23,
            kind,
            kind == SignalKind::feature ? feature_set : 0,
            kind == SignalKind::feature ? ObservationTiming::regular
                                        : ObservationTiming::not_applicable,
        },
    };
    auto descriptors = feature_sets(std::move(names));
    descriptors.front().id = feature_set == 0 ? 47 : feature_set;
    return StreamSchema{29, signals, descriptors, units()};
}

[[nodiscard]] ProcessorPrepareContext context(const StreamSchema& schema) noexcept
{
    return {
        .input_schema = schema,
        .max_process_outputs = 32,
        .max_flush_outputs = 32,
        .available_frame_pool_leases = 4,
    };
}

[[nodiscard]] StreamStatus fill_frame(MutableFrame& frame, const StreamSchema& schema,
                                      std::span<const double> values, std::uint64_t sequence,
                                      SampleIndex observation_start) noexcept
{
    const auto channels = schema.signals().front().n_channels;
    if (values.empty() || values.size() % channels != 0)
    {
        return StreamStatus::invalid_frame;
    }
    const auto observations = values.size() / channels;
    frame.header() = FrameHeader{
        .session_id = 5,
        .sequence = sequence,
        .host_received_ns = 100'000'000 + observation_start,
        .source_tick = 5'000 + observation_start,
        .valid_until_ns = 200'000'000 + observation_start,
        .schema_id = schema.id(),
        .source_clock_domain = 41,
        .flags = FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
    };
    frame.block_storage()[0] = SignalBlockHeader{
        .sample_idx_start = observation_start,
        .device_tick_start = 5'000 + observation_start,
        .observation_time_start_ns = first_observation_ns + observation_start * shift_ns,
        .payload_offset = 0,
        .payload_byte_count = values.size_bytes(),
        .signal_id = 11,
        .n_samples = static_cast<std::uint32_t>(observations),
        .clock_sync =
            ClockSyncSnapshot{
                .device_tick_reference = 5'000,
                .host_time_reference_ns = 10'000'000,
                .device_tick_rate = schema.signals().front().fs,
                .uncertainty_ns = 3,
                .clock_domain = 41,
                .generation = 2,
                .flags = ClockSyncFlags::synchronized,
            },
    };
    std::memcpy(frame.payload_storage().data(), values.data(), values.size_bytes());
    return frame.set_used_sizes(1, values.size_bytes());
}

[[nodiscard]] bool nearly_equal(std::span<const double> left, std::span<const double> right,
                                double tol = 1e-12) noexcept
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const auto scale = std::max({1.0, std::abs(left[i]), std::abs(right[i])});
        if (std::abs(left[i] - right[i]) > tol * scale)
        {
            return false;
        }
    }
    return true;
}

class CollectingEmitter final : public FrameEmitter
{
  public:
    CollectingEmitter(const StreamSchema& schema, std::size_t payload_bytes)
        : pool_(1, payload_bytes, 1), validator_(schema)
    {
    }

    explicit CollectingEmitter(const StreamSchema& schema)
        : CollectingEmitter(schema, schema.signals().front().max_block_bytes)
    {
    }

    void clear() noexcept
    {
        frame_count_ = 0;
        value_count_ = 0;
        static_cast<void>(acquired_.reset());
    }

    StreamStatus try_acquire_frame(MutableFrame*& frame) noexcept override
    {
        frame = nullptr;
        if (acquired_)
        {
            return StreamStatus::invalid_state;
        }
        const auto status = pool_.try_acquire(acquired_);
        if (status == StreamStatus::ok)
        {
            frame = &acquired_.frame();
        }
        return status;
    }

    StreamStatus publish_acquired_frame() noexcept override
    {
        return record(std::move(acquired_));
    }

    StreamStatus publish_input() noexcept override
    {
        return StreamStatus::invalid_state;
    }

    [[nodiscard]] std::size_t frame_count() const noexcept
    {
        return frame_count_;
    }
    [[nodiscard]] std::span<const double> values() const noexcept
    {
        return {values_.data(), value_count_};
    }
    [[nodiscard]] const SignalBlockHeader& block(std::size_t idx) const noexcept
    {
        return blocks_[idx];
    }
    [[nodiscard]] const FrameHeader& frame_header(std::size_t idx) const noexcept
    {
        return headers_[idx];
    }
    /// Whether the emitter is holding no lease, which is what every completed
    /// call must leave behind.
    [[nodiscard]] bool idle() const noexcept
    {
        return !acquired_;
    }

  private:
    StreamStatus publish_owned(FrameLease lease) noexcept override
    {
        return record(std::move(lease));
    }

    StreamStatus record(FrameLease lease) noexcept
    {
        if (!lease || frame_count_ == max_frames ||
            validator_.validate(lease.view()) != FrameValidationError::none)
        {
            return StreamStatus::invalid_frame;
        }
        const auto view = lease.view();
        const auto n_scalars = view.payload.size() / sizeof(double);
        if (n_scalars > max_values - value_count_)
        {
            return StreamStatus::invalid_frame;
        }
        headers_[frame_count_] = view.header;
        blocks_[frame_count_] = view.blocks.front();
        std::memcpy(values_.data() + value_count_, view.payload.data(), view.payload.size());
        value_count_ += n_scalars;
        ++frame_count_;
        return lease.reset();
    }

    FramePool pool_;
    FrameValidator validator_;
    FrameLease acquired_{};
    std::array<FrameHeader, max_frames> headers_{};
    std::array<SignalBlockHeader, max_frames> blocks_{};
    std::array<double, max_values> values_{};
    std::size_t frame_count_{};
    std::size_t value_count_{};
};

/// An emitter with no frames at all, for the exhausted-output fault path.
class ExhaustedEmitter final : public FrameEmitter
{
  public:
    StreamStatus try_acquire_frame(MutableFrame*&) noexcept override
    {
        return StreamStatus::buffer_exhausted;
    }
    StreamStatus publish_acquired_frame() noexcept override
    {
        return StreamStatus::invalid_state;
    }
    StreamStatus publish_input() noexcept override
    {
        return StreamStatus::invalid_state;
    }

  private:
    StreamStatus publish_owned(FrameLease) noexcept override
    {
        return StreamStatus::invalid_state;
    }
};

/// Decode with the offline model, in the order ``KalmanDecoder`` decodes.
///
/// ``n_features`` wide input, one row per observation. The first row of a
/// segment is corrected where the state stands, because the fitted initial
/// state estimates the state at the start of a segment rather than the step
/// before it; every later row costs a transition and a correction. A row that
/// is entirely ``nan`` runs the transition alone.
[[nodiscard]] std::size_t reference_decode(std::span<const double> input, std::size_t n_features,
                                           std::span<const std::size_t> columns,
                                           bool scale_features, std::span<double> output)
{
    const LinearGaussianModel model{model_state()};
    const auto parameters = model_state();
    std::vector<double> state{parameters.initial_state};
    std::vector<double> covariance{parameters.initial_covariance};
    std::vector<double> observation(columns.size(), 0.0);
    const auto rows = input.size() / n_features;
    bool at_segment_start = true;
    for (std::size_t row = 0; row < rows; ++row)
    {
        std::size_t missing = 0;
        for (std::size_t i = 0; i < columns.size(); ++i)
        {
            const double value = input[row * n_features + columns[i]];
            if (std::isnan(value))
            {
                // An absent value has nothing to scale, and scaling could not
                // turn it into a present one.
                observation[i] = value;
                ++missing;
            }
            else if (scale_features)
            {
                observation[i] = (value - scaler_center[i]) / scaler_scale[i];
            }
            else
            {
                observation[i] = value;
            }
        }
        if (!at_segment_start)
        {
            model.predict(state, covariance);
        }
        if (missing == 0)
        {
            model.update(state, covariance, observation, innovation_jitter);
        }
        at_segment_start = false;
        std::copy(state.begin(), state.end(),
                  output.begin() + static_cast<std::ptrdiff_t>(row * state_dim));
    }
    return rows * state_dim;
}

[[nodiscard]] std::vector<double> ramp(std::size_t observations, std::uint32_t channels = features)
{
    std::vector<double> values(observations * channels);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        values[i] =
            std::sin(static_cast<double>(i) * 0.31) + 0.25 * static_cast<double>(i % channels);
    }
    return values;
}

template <std::size_t ChunkCount>
[[nodiscard]] int run_decoder(std::span<const double> input,
                              const std::array<std::size_t, ChunkCount>& chunks,
                              std::span<double> output, std::span<SignalBlockHeader> blocks,
                              std::size_t& n_values, std::size_t& n_frames)
{
    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool input_pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(input_pool.try_acquire(lease) == StreamStatus::ok);
    std::size_t scalar_offset = 0;
    SampleIndex observation_offset = 0;
    for (const auto chunk : chunks)
    {
        const auto prior_frames = sink.frame_count();
        CHECK(fill_frame(lease.frame(), schema, input.subspan(scalar_offset, chunk * features),
                         observation_offset + 1, observation_offset) == StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(sink.frame_count() - prior_frames <= contract.max_process_outputs_per_input);
        CHECK(sink.idle());
        scalar_offset += chunk * features;
        observation_offset += chunk;
    }
    CHECK(scalar_offset == input.size());
    CHECK(sink.values().size() <= output.size());
    CHECK(sink.frame_count() <= blocks.size());
    std::copy(sink.values().begin(), sink.values().end(), output.begin());
    for (std::size_t i = 0; i < sink.frame_count(); ++i)
    {
        blocks[i] = sink.block(i);
    }
    n_values = sink.values().size();
    n_frames = sink.frame_count();
    return 0;
}

template <typename Callable> [[nodiscard]] bool rejects(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}

/// Whether a configuration is refused before an adapter exists at all.
[[nodiscard]] bool rejects_config(neurale::pipeline::KalmanDecoderAdapterConfig config)
{
    return rejects([&] { neurale::pipeline::KalmanDecoderAdapter adapter{std::move(config)}; });
}

/// Whether an input is refused, for an otherwise valid configuration.
///
/// The schema is built inside the guard rather than passed in: some inputs the
/// decoder must not accept are ones ``StreamSchema`` itself refuses to
/// construct, and those have to count as a rejection too.
template <typename MakeSchema>
[[nodiscard]] bool
rejects_schema(MakeSchema&& make_input,
               neurale::pipeline::KalmanDecoderAdapterConfig config = decoder_config())
{
    return rejects(
        [&]
        {
            const auto schema = make_input();
            neurale::pipeline::KalmanDecoderAdapter adapter{std::move(config)};
            static_cast<void>(adapter.prepare(context(schema)));
        });
}

[[nodiscard]] int test_contract_and_schema()
{
    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    const auto& output = contract.output_schema.signals().front();
    CHECK(output.kind == SignalKind::sampled);
    CHECK(output.id == 43);
    CHECK(output.n_channels == state_dim);
    CHECK(output.dtype == SignalDType::float64);
    CHECK(output.layout == SignalLayout::sample_major);
    CHECK(output.observation_timing == ObservationTiming::not_applicable);
    CHECK(output.feature_set_id == 0);
    CHECK(output.fs.numerator == 500);
    CHECK(output.fs.denominator == 1);
    CHECK(output.clock_domain == schema.signals().front().clock_domain);
    CHECK(output.physical_unit == PhysicalUnit::dimensionless);
    CHECK(output.channel_set_id == 59);
    CHECK(output.nominal_block_samples == schema.signals().front().nominal_block_samples);
    CHECK(output.max_block_samples == schema.signals().front().max_block_samples);
    CHECK(output.device_tick_tracking == DeviceTickTracking::unavailable);
    CHECK(contract.max_process_outputs_per_input == 1);
    CHECK(contract.max_flush_outputs == 0);
    CHECK(!contract.can_forward_input);
    CHECK(contract.required_resources.frame_pool_leases == 1);
    CHECK(contract.required_resources.workspace_bytes > 0);
    CHECK(contract.output_schema.id() == 37);
    return 0;
}

[[nodiscard]] int test_schema_rejection()
{
    // A sampled input, a float32 input, and an input whose feature set is not
    // the one the decoder was fitted on.
    CHECK(rejects_schema([] { return make_schema(features, SignalKind::sampled); }));
    CHECK(rejects_schema(
        [] { return make_schema(features, SignalKind::feature, SignalDType::float32); }));
    CHECK(rejects_schema(
        [] { return make_schema(features, SignalKind::feature, SignalDType::float64, 61); }));
    // The selection is in range and the right length, and every matrix would
    // still multiply -- but column 3 no longer carries the feature the model
    // was fitted on, so the decoder would consume one signal while reporting
    // another.
    CHECK(rejects_schema(
        []
        {
            return make_schema(features, SignalKind::feature, SignalDType::float64, 47,
                               {"rate_0", "rate_1", "rate_2", "renamed"});
        }));
    // A narrower input the selection points past.
    CHECK(rejects_schema(
        []
        {
            return make_schema(3, SignalKind::feature, SignalDType::float64, 47,
                               {"rate_0", "rate_1", "rate_2"});
        }));

    // Output identifiers that collide with the input it decodes.
    auto colliding = decoder_config();
    colliding.output_signal_id = 11;
    CHECK(rejects_schema([] { return make_schema(); }, std::move(colliding)));

    // A configuration whose selection and model disagree never reaches prepare.
    auto short_selection = decoder_config();
    short_selection.selection = {0, 1};
    short_selection.selected_feature_names = {"rate_0", "rate_1"};
    CHECK(rejects_config(std::move(short_selection)));

    auto repeated_selection = decoder_config();
    repeated_selection.selection = {1, 1, 2};
    CHECK(rejects_config(std::move(repeated_selection)));

    auto zero_scale = decoder_config();
    zero_scale.scaler_scale = {2.0, 0.0, 1.5};
    CHECK(rejects_config(std::move(zero_scale)));
    return 0;
}

/// Build a schema whose descriptor differs from the fitted contract in exactly
/// one field, so the rejection test names the field that changed.
[[nodiscard]] StreamSchema make_mismatched_schema(const std::string& mismatch,
                                                  std::vector<std::string> names = {
                                                      "rate_0", "rate_1", "rate_2", "rate_3"})
{
    // The two name mismatches rewrite the descriptor's columns, so they are
    // applied before it is built rather than to a reference into it.
    if (mismatch == "unselected_renamed")
    {
        // Column 1 is not selected; renaming it does not affect the selected
        // names but does violate the full schema contract.
        names[1] = "renamed";
    }
    else if (mismatch == "unselected_reordered")
    {
        // Swap two unselected columns: the selected names still resolve, but
        // the full feature-name order no longer matches the fitted contract.
        std::swap(names[0], names[1]);
    }

    auto descriptors = feature_sets(std::move(names));
    auto& descriptor = descriptors.front();
    auto unit_list = units();
    RationalRate rate{500, 1};

    if (mismatch == "unit")
    {
        unit_list = {UnitDescriptor{.id = 53, .symbol = "uV", .description = "microvolts"}};
    }
    else if (mismatch == "rate" || mismatch == "shift")
    {
        // The schema validates rate * shift_ns == denominator * 1e9, so the
        // observation rate and the shift can only move together: 1000 Hz with a
        // 1 ms shift is the valid pair nearest the fitted 500 Hz / 2 ms. Both
        // cases therefore build the same schema; each is listed separately so
        // that a failure names the field of the fitted contract it contradicts.
        descriptor.shift_ns = 1'000'000;
        rate = {1'000, 1};
    }
    else if (mismatch == "window")
    {
        descriptor.window_length_ns = 2'000'000;
    }
    else if (mismatch == "source_stream")
    {
        descriptor.source_stream = "other-signal";
    }
    else if (mismatch == "algorithm_name")
    {
        descriptor.algorithm_name = "other";
    }
    else if (mismatch == "algorithm_version")
    {
        descriptor.algorithm_version = "2";
    }

    const std::array signals{
        SignalSchema{
            11,
            SignalDType::float64,
            features,
            6,
            max_input_observations,
            rate,
            41,
            SignalLayout::sample_major,
            DeviceTickTracking::sample_counter,
            PhysicalUnit::unspecified,
            17,
            19,
            23,
            SignalKind::feature,
            47,
            ObservationTiming::regular,
        },
    };
    return StreamSchema{29, signals, descriptors, unit_list};
}

[[nodiscard]] int test_fitted_feature_contract_rejection()
{
    // Each case changes one field of the input descriptor while keeping the
    // feature-set id, the feature count, and the selected feature names the
    // same. The adapter must refuse the input because the model parameters and
    // scaler statistics were estimated on a different feature contract.
    const auto cases = {
        std::string{"rate"},
        std::string{"unit"},
        std::string{"window"},
        std::string{"shift"},
        std::string{"source_stream"},
        std::string{"algorithm_name"},
        std::string{"algorithm_version"},
        std::string{"unselected_renamed"},
        std::string{"unselected_reordered"},
    };
    for (const auto& mismatch : cases)
    {
        if (!rejects_schema([&] { return make_mismatched_schema(mismatch); }))
        {
            return __LINE__;
        }
    }
    return 0;
}

[[nodiscard]] int test_offline_parity_and_chunk_invariance()
{
    constexpr std::size_t observations = 24;
    const auto input = ramp(observations);
    std::array<double, max_values> expected{};
    const auto n_expected =
        reference_decode(input, features, selection, /*scale_features=*/true, expected);

    std::array<double, max_values> first{};
    std::array<double, max_values> second{};
    std::array<SignalBlockHeader, max_frames> first_blocks{};
    std::array<SignalBlockHeader, max_frames> second_blocks{};
    std::size_t first_count{};
    std::size_t second_count{};
    std::size_t first_frames{};
    std::size_t second_frames{};
    CHECK(run_decoder(input, std::array<std::size_t, 5>{1, 3, 8, 4, 8}, first, first_blocks,
                      first_count, first_frames) == 0);
    CHECK(run_decoder(input, std::array<std::size_t, 3>{8, 8, 8}, second, second_blocks,
                      second_count, second_frames) == 0);
    CHECK(first_count == n_expected);
    CHECK(second_count == n_expected);
    CHECK(first_frames == 5);
    CHECK(second_frames == 3);
    CHECK(nearly_equal({first.data(), first_count}, {expected.data(), n_expected}));
    // Chunking is not merely close: the recursion is the same arithmetic on the
    // same values in the same order however the stream was cut.
    CHECK(std::memcmp(first.data(), second.data(), first_count * sizeof(double)) == 0);

    // Observation index and observation time survive the decode, so the output
    // timeline is the input timeline.
    std::uint64_t observation_idx = 0;
    for (std::size_t i = 0; i < first_frames; ++i)
    {
        CHECK(first_blocks[i].sample_idx_start == observation_idx);
        CHECK(first_blocks[i].observation_time_start_ns ==
              first_observation_ns + observation_idx * shift_ns);
        CHECK(first_blocks[i].signal_id == 43);
        CHECK(first_blocks[i].payload_offset == 0);
        CHECK(first_blocks[i].payload_byte_count ==
              first_blocks[i].n_samples * state_dim * sizeof(double));
        observation_idx += first_blocks[i].n_samples;
    }
    CHECK(observation_idx == observations);
    return 0;
}

[[nodiscard]] int test_multi_observation_reset_and_discontinuity()
{
    constexpr std::size_t observations = 8;
    const auto input = ramp(observations);
    std::array<double, max_values> expected{};
    const auto n_expected = reference_decode(input, features, selection, true, expected);

    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);

    // One frame carrying every observation: the whole block is decoded and
    // leaves as a single output frame.
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(sink.block(0).n_samples == observations);
    CHECK(nearly_equal(sink.values(), {expected.data(), n_expected}));
    CHECK(sink.frame_header(0).schema_id == 37);
    CHECK(sink.frame_header(0).sequence == 1);

    // A reset restores the prepared initial state, so the same input decodes to
    // the same bytes -- and repeated runs are identical, not merely close.
    std::array<double, max_values> repeated{};
    std::copy(sink.values().begin(), sink.values().end(), repeated.begin());
    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, input, 2, 100) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(std::memcmp(repeated.data(), sink.values().data(), n_expected * sizeof(double)) == 0);

    // A discontinuity ends the segment: what follows starts from the prepared
    // state again rather than continuing a trajectory across missing data.
    const Discontinuity discontinuity{
        .session_id = 5,
        .previous_frame_sequence = 2,
        .actual_frame_sequence = 9,
        .reason = GapReason::source_gap,
    };
    CHECK(adapter.handle_discontinuity(discontinuity) == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, input, 9, 400) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(std::memcmp(repeated.data(), sink.values().data(), n_expected * sizeof(double)) == 0);
    CHECK(sink.block(0).sample_idx_start == 400);

    // A flush produces nothing: every observation left with the frame that
    // carried it.
    sink.clear();
    CHECK(adapter.flush(sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 0);
    return 0;
}

[[nodiscard]] int test_missing_observation_policy()
{
    constexpr std::size_t observations = 6;
    auto input = ramp(observations);
    // One row the policy may treat as absent, and it is not the first: the
    // predict-only step has to land in the middle of a running segment.
    for (std::uint32_t channel = 0; channel < features; ++channel)
    {
        input[2 * features + channel] = std::nan("");
    }
    std::array<double, max_values> expected{};
    const auto n_expected = reference_decode(input, features, selection, true, expected);

    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{
        decoder_config(neurale::pipeline::KalmanMissingPolicy::predict)};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), {expected.data(), n_expected}));

    // A row that is missing in only some of the selected columns is not a
    // measurement that was never taken; neither is an infinity.
    auto partial = input;
    partial[2 * features + selection[0]] = 1.0;
    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, partial, 2, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);

    auto infinite = ramp(observations);
    infinite[3 * features + selection[1]] = std::numeric_limits<double>::infinity();
    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, infinite, 3, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);

    // Under the default policy an absent row is a fault rather than a
    // prediction: whether a gap in the features means "predict" is the
    // decoder's configuration, not the adapter's guess.
    neurale::pipeline::KalmanDecoderAdapter strict{decoder_config()};
    const auto strict_contract = strict.prepare(context(schema));
    CollectingEmitter strict_sink{strict_contract.output_schema};
    CHECK(fill_frame(lease.frame(), schema, input, 4, 0) == StreamStatus::ok);
    CHECK(strict.process(lease.frame(), strict_sink) == StreamStatus::invalid_frame);
    CHECK(strict_sink.frame_count() == 0);

    // A column outside the selection may hold anything, including a nan: the
    // decoder consumes what it was fitted on and nothing else.
    auto untouched = ramp(observations);
    untouched[4 * features + 1] = std::nan("");
    std::array<double, max_values> unaffected{};
    const auto n_unaffected = reference_decode(untouched, features, selection, true, unaffected);
    neurale::pipeline::KalmanDecoderAdapter other{decoder_config()};
    const auto other_contract = other.prepare(context(schema));
    CollectingEmitter other_sink{other_contract.output_schema};
    CHECK(fill_frame(lease.frame(), schema, untouched, 5, 0) == StreamStatus::ok);
    CHECK(other.process(lease.frame(), other_sink) == StreamStatus::ok);
    CHECK(nearly_equal(other_sink.values(), {unaffected.data(), n_unaffected}));
    return 0;
}

[[nodiscard]] int test_training_capture_alignment_and_overflow()
{
    constexpr std::size_t observations = 2;
    constexpr SampleIndex sample_idx_start = 23;
    const auto input = ramp(observations);
    auto schema = make_schema();

    neurale::pipeline::KalmanDecoderAdapter adapter{decoder_config()};
    adapter.enable_training_capture(observations);
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input, 1, sample_idx_start) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);

    auto* capture = adapter.training_capture();
    CHECK(capture != nullptr);
    for (std::size_t row = 0; row < observations; ++row)
    {
        neurale::pipeline::DecoderTrainingObservation captured{};
        CHECK(capture->try_pop(captured) == StreamStatus::ok);
        CHECK(captured.sample_idx == sample_idx_start + row);
        CHECK(captured.n_features == observation_dim);
        for (std::size_t column = 0; column < observation_dim; ++column)
        {
            CHECK(captured.values[column] == input[row * features + selection[column]]);
        }
    }
    neurale::pipeline::DecoderTrainingObservation captured{};
    CHECK(capture->try_pop(captured) == StreamStatus::would_block);
    CHECK(capture->dropped() == 0);

    neurale::pipeline::KalmanDecoderAdapter overflow{decoder_config()};
    overflow.enable_training_capture(1);
    const auto overflow_contract = overflow.prepare(context(schema));
    CollectingEmitter overflow_sink{overflow_contract.output_schema};
    CHECK(fill_frame(lease.frame(), schema, input, 2, sample_idx_start) == StreamStatus::ok);
    CHECK(overflow.process(lease.frame(), overflow_sink) == StreamStatus::queue_overflow);
    CHECK(overflow.training_capture()->dropped() == observations);
    CHECK(overflow_sink.idle());
    return 0;
}

[[nodiscard]] int test_allocation_and_fault_paths()
{
    constexpr std::size_t observations = 8;
    const auto input = ramp(observations);
    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    // One warm pass first: what the gate is about is the steady state, not the
    // first touch of a page.
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();

    neurale::benchmark::reset_allocation_count();
    {
        neurale::benchmark::AllocationScope allocation_scope;
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(adapter.handle_discontinuity(Discontinuity{
                  .session_id = 5,
                  .previous_frame_sequence = 1,
                  .actual_frame_sequence = 3,
                  .reason = GapReason::source_gap,
              }) == StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(adapter.flush(sink) == StreamStatus::ok);
        CHECK(adapter.reset() == StreamStatus::ok);
        allocation_scope.stop();
        CHECK(allocation_scope.count() == 0);
    }
    CHECK(sink.idle());

    // An output the emitter cannot supply is a fault, not a partial decode.
    ExhaustedEmitter exhausted;
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), exhausted) == StreamStatus::buffer_exhausted);

    // An output frame too small for the decoded block is an output-capacity
    // violation rather than an overrun of the frame it was given.
    CollectingEmitter narrow{contract.output_schema, state_dim * sizeof(double)};
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), narrow) == StreamStatus::output_limit);
    CHECK(narrow.frame_count() == 0);

    // A frame that does not describe the stream it claims to belong to.
    CHECK(adapter.reset() == StreamStatus::ok);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, input, 2, 0) == StreamStatus::ok);
    lease.frame().header().schema_id = 999;
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);
    return 0;
}

[[nodiscard]] int test_numerical_failure_is_a_fault()
{
    // Nothing is uncertain anywhere -- no process noise, no observation noise,
    // no initial covariance and no jitter -- so no gain exists and the decoder
    // must say so rather than divide by it.
    //
    // The information-form recursion says so at prepare rather than at step.
    // Its gain is built from ``(R + jitter I)^-1 H``, which is a property of the
    // fitted model alone, so a model with no observation noise is refused
    // before a session starts instead of failing on the real-time thread once
    // data arrives. The covariance form could not tell the two apart: its
    // ``H P H^T + R`` mixed the model with the running covariance, so the same
    // defect only surfaced on the first frame.
    auto config = decoder_config();
    config.innovation_jitter = 0.0;
    config.model.process_covariance = {0.0, 0.0, 0.0, 0.0};
    config.model.observation_covariance =
        std::vector<double>(observation_dim * observation_dim, 0.0);
    config.model.initial_covariance = {0.0, 0.0, 0.0, 0.0};

    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{std::move(config)};
    bool refused = false;
    try
    {
        static_cast<void>(adapter.prepare(context(schema)));
    }
    catch (const std::invalid_argument&)
    {
        refused = true;
    }
    CHECK(refused);

    // A singular observation covariance that an explicit jitter regularizes is
    // still accepted, and still decodes: the jitter is what the contract offers
    // for exactly this model.
    auto regularized = decoder_config();
    regularized.innovation_jitter = 1e-6;
    regularized.model.observation_covariance =
        std::vector<double>(observation_dim * observation_dim, 0.0);
    neurale::pipeline::KalmanDecoderAdapter usable{std::move(regularized)};
    const auto contract = usable.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const auto input = ramp(4);
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    CHECK(usable.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    return 0;
}

[[nodiscard]] int test_overflow_leaves_the_filter_on_its_prior()
{
    // The one step-time failure the information form still has, and the one
    // that used to corrupt the filter.
    //
    // 1e308 is finite, so gather() accepts it -- the finiteness rule reads the
    // raw feature, and only nan means absent. Scaling then divides it by
    // scaler_scale[1] == 0.5 and overflows, so the corrected state comes out
    // non-finite while the posterior covariance, which does not depend on the
    // observation at all, stays finite. That asymmetry is exactly the
    // all_finite(state) half of the step's numerical_failure return.
    //
    // The adapter maps that to processor_failure and does not clear the stream
    // state, so a step that had already written the correction into the state
    // in place would leave every later frame decoding from an infinity. Both
    // halves of the update commit only after both have been checked, so the
    // filter is still holding its prior here.
    constexpr std::size_t observations = 1;
    auto poisoned = ramp(observations);
    poisoned[selection[1]] = 1e308;

    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, poisoned, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::processor_failure);
    CHECK(sink.frame_count() == 0);
    CHECK(sink.idle());

    // No reset in between. The next ordinary frame must decode what a filter
    // that had never seen the overflow decodes -- the failed step consumed
    // nothing, so this row is still the first of the segment.
    const auto ordinary = ramp(observations);
    std::array<double, max_values> expected{};
    const auto n_expected = reference_decode(ordinary, features, selection, true, expected);
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, ordinary, 2, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(nearly_equal(sink.values(), {expected.data(), n_expected}));
    return 0;
}

[[nodiscard]] int test_python_decoder_parity()
{
    // A fitted ``neurale.decoding.KalmanDecoder`` and what it decoded, recorded
    // once and compared against here. Everything below -- the fitted
    // parameters, the frozen scaler statistics, the input frames, and the
    // decoded states -- came out of this, printed with 17 significant digits:
    //
    //   rng = np.random.default_rng(20260804)
    //   position = np.cumsum(rng.normal(scale=0.05, size=(40, 2)), axis=0)
    //   data = position @ rng.normal(size=(2, 4)) + rng.normal(scale=0.1, size=(40, 4))
    //   features = FeatureMatrix(data=data, fs=50.0,
    //                            feature_names=[f"rate_{i}" for i in range(4)],
    //                            unit="Hz", shift=1 / 50.0, source_signal="units")
    //   target = SignalArray.from_array(position, fs=50.0,
    //                                   time=features.time.copy(),
    //                                   channel_names=["x", "y"],
    //                                   channel_types="behavior", units="m", name="cursor")
    //   decoder = KalmanDecoder(scaler=StandardScaler(),
    //                           feature_names=["rate_3", "rate_0", "rate_2"],
    //                           innovation_jitter=1e-9).fit(features, target)
    //   decoder.reset()
    //   decoded = decoder.predict(features[:12])
    //
    // The point is the seam a C++ reference cannot cover on its own: that the
    // adapter reproduces the *decoder*, selection order, frozen scaler and
    // segment-start alignment included, and not merely the recursion under it.
    constexpr std::size_t rows = 12;
    constexpr std::array<double, rows * features> recorded_features{
        0.031431165306515671,  -0.010217391733959025, 0.047199638681514215,   0.050887186355548911,
        -0.10713027552043233,  0.1979433867461835,    -0.014682306757752263,  -0.13875485871268459,
        -0.11100283998548961,  0.04575635939406246,   -0.053483740675962642,  -0.009136733109120939,
        0.041540999729078917,  -0.02440261049057682,  -0.017495605878396079,  -0.13830009261935314,
        -0.13954557727415579,  0.049721816991812398,  -0.20625251225568209,   0.027514205591731875,
        -0.16336821469270951,  -0.09293422504562604,  -0.15360337786317216,   -0.13798649446745853,
        0.11421309319190007,   0.081395545046138329,  -0.23436169846432561,   -0.13388782258054849,
        0.13060814469902918,   -0.066859094480024583, -0.42259049353098876,   -0.099741633846420458,
        0.0051902284006664218, -0.16088823413451081,  -0.039509337548081419,  -0.087753721327387721,
        0.096828562156045753,  0.012644803098126101,  -0.057577385892426697,  0.1302396177276508,
        0.057403831869349942,  -0.17209684682740081,  -0.0059247796266795917, -0.13152128857381881,
        0.33061316945877728,   -0.19891878091504944,  0.0042482367468367893,  -0.048906871996062258,
    };
    constexpr std::array<double, rows * state_dim> recorded_states{
        -0.028609877775662637, 0.060447481658100093,  0.024336771885304621,  0.045324001198462312,
        0.043903053245932924,  0.012903958095626129,  0.035267028137508566,  -0.021633452348075233,
        0.048680178646299303,  -0.04755071849219026,  0.088230708918966394,  -0.041671009693513977,
        0.022530131887533188,  -0.084214663434019593, -0.031131884842001318, -0.10609550220601678,
        0.015067742622636729,  -0.099479029818889619, -0.010461565582985171, -0.164306428139117,
        0.025784654695751648,  -0.15011586817570677,  -0.037806344324004204, -0.20930012907083467,
    };

    auto config = decoder_config();
    config.model.transition = {0.94098226222721837, 0.079962768325158307, -0.023630912957121775,
                               0.95245863502870687};
    config.model.transition_offset = {0.00331997891403309, -0.031527823930797116};
    config.model.observation = {-1.5745952958300045, -2.0608425730992579, -3.5929914333873705,
                                -1.3013425741565305, 2.9168155233395372,  -0.0036781094835975914};
    config.model.observation_offset = {-0.91476025726972876, -1.0360662258289712,
                                       0.51341247741680718};
    config.model.process_covariance = {0.0026048011737814995, 0.0004435916461299543,
                                       0.0004435916461299543, 0.0018181808011157781};
    config.model.observation_covariance = {
        0.53628483933385407,   -0.037053356050751995,  0.13509300309107239,
        -0.037053356050751995, 0.06239488485623107,    -0.0065195407075563649,
        0.13509300309107239,   -0.0065195407075563649, 0.59554444866241929};
    config.model.initial_state = {-0.028609877775662637, 0.060447481658100093};
    // A single fitted segment with no jitter leaves an exactly zero initial
    // covariance, so the first corrected row is the initial state itself.
    config.model.initial_covariance = {0.0, 0.0, 0.0, 0.0};
    config.scaler_center = {0.060676049849556088, 0.45205215913867969, -0.17364334784375604};
    config.scaler_scale = {0.12745530824642942, 0.42945235396824782, 0.13938127223119517};

    auto schema = make_schema();
    neurale::pipeline::KalmanDecoderAdapter adapter{std::move(config)};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);

    // Cut across the block boundary the Python decoder never saw: the same
    // decoder, chunked, still has to produce what it produced whole.
    std::size_t offset = 0;
    std::uint64_t sequence = 1;
    for (const auto chunk : std::array<std::size_t, 3>{5, 3, 4})
    {
        CHECK(fill_frame(lease.frame(), schema,
                         std::span<const double>{recorded_features}.subspan(offset * features,
                                                                            chunk * features),
                         sequence++, offset) == StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        offset += chunk;
    }
    CHECK(offset == rows);
    CHECK(sink.values().size() == recorded_states.size());
    // The two agree to about 1e-15 relative on this host; the tolerance is the
    // file's usual one, which leaves room for a different solve backend without
    // leaving room for a different decoder.
    CHECK(nearly_equal(sink.values(), recorded_states));
    return 0;
}

[[nodiscard]] int test_feature_to_decoder_chain()
{
    // The whole edge, in one runtime topology: a voltage stream becomes LMP
    // features, the features become decoded state, and a native consumer sees
    // the state. Nothing between the two adapters is arranged by the test --
    // the Kalman side reads the feature-set descriptor the LMP side declared.
    constexpr std::uint32_t chain_channels = 2;
    constexpr std::size_t source_samples = 48;
    constexpr std::array<double, 6> lmp_sos{0.5, 0.25, 0.125, 1.0, -0.2, 0.05};
    const std::array signals{
        SignalSchema{
            11,
            SignalDType::float64,
            chain_channels,
            12,
            12,
            {1'000, 1},
            41,
            SignalLayout::sample_major,
            DeviceTickTracking::sample_counter,
            PhysicalUnit::volts,
            17,
            19,
            23,
        },
    };
    const StreamSchema source_schema{29, signals};

    neurale::pipeline::LmpFeatureAdapterConfig lmp{
        .output_schema_id = 31,
        .output_signal_id = 13,
        .feature_set_id = 47,
        .feature_unit = UnitDescriptor{.id = 53, .symbol = "V", .description = "volts"},
        .feature_names = {"lmp:left", "lmp:right"},
        .source_stream = "electrode-voltage",
        .algorithm_version = "1",
        .window_samples = 4,
        .shift_samples = 2,
        .sos = {lmp_sos.begin(), lmp_sos.end()},
        .sos_sections = 1,
    };

    auto config = decoder_config();
    config.selection = {0, 1};
    config.selected_feature_names = {"lmp:left", "lmp:right"};
    config.fitted_feature_contract.feature_names = {"lmp:left", "lmp:right"};
    config.fitted_feature_contract.feature_unit_symbols = {"V", "V"};
    config.fitted_feature_contract.source_stream = "electrode-voltage";
    config.fitted_feature_contract.algorithm_name = "lmp";
    config.scaling = neurale::pipeline::FeatureScaling::none;
    config.scaler_center.clear();
    config.scaler_scale.clear();
    config.model.observation_dim = 2;
    config.model.observation = {1.0, 0.2, 0.5, -0.8};
    config.model.observation_offset = {0.05, -0.10};
    config.model.observation_covariance = {0.10, 0.01, 0.01, 0.15};

    neurale::pipeline::LmpFeatureAdapter lmp_adapter{lmp};
    neurale::pipeline::KalmanDecoderAdapter decoder{config};
    std::array<NativeFrameProcessor*, 2> stages{&lmp_adapter, &decoder};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(context(source_schema));
    CHECK(contract.output_schema.signals().front().kind == SignalKind::sampled);
    CHECK(contract.output_schema.signals().front().n_channels == state_dim);
    CollectingEmitter sink{contract.output_schema};

    std::vector<double> input(source_samples * chain_channels);
    for (std::size_t i = 0; i < input.size(); ++i)
    {
        input[i] =
            std::cos(static_cast<double>(i) * 0.07) + static_cast<double>(i % chain_channels);
    }

    // The reference runs the same two stages offline, in the same order.
    neurale::features::LmpProcessor lmp_reference{4, 2, chain_channels, lmp_sos, 1};
    const auto feature_rows = lmp_reference.output_count(source_samples);
    std::vector<double> feature_values(feature_rows * chain_channels, 0.0);
    lmp_reference.process(input, source_samples, feature_values);
    std::array<double, max_values> expected{};
    const LinearGaussianModel reference_model{config.model};
    std::vector<double> state{config.model.initial_state};
    std::vector<double> covariance{config.model.initial_covariance};
    for (std::size_t row = 0; row < feature_rows; ++row)
    {
        if (row != 0)
        {
            reference_model.predict(state, covariance);
        }
        reference_model.update(
            state, covariance,
            std::span<const double>{feature_values}.subspan(row * chain_channels, chain_channels),
            innovation_jitter);
        std::copy(state.begin(), state.end(),
                  expected.begin() + static_cast<std::ptrdiff_t>(row * state_dim));
    }

    FramePool pool{1, source_schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    std::size_t offset = 0;
    std::uint64_t sequence = 1;
    while (offset < source_samples)
    {
        const auto chunk = std::min<std::size_t>(12, source_samples - offset);
        auto& frame = lease.frame();
        frame.header() = FrameHeader{
            .session_id = 5,
            .sequence = sequence++,
            .host_received_ns = 100'000'000 + offset,
            .source_tick = 5'000 + offset,
            .valid_until_ns = 200'000'000 + offset,
            .schema_id = source_schema.id(),
            .source_clock_domain = 41,
            .flags =
                FrameFlags::source_tick | FrameFlags::valid_until | FrameFlags::source_received,
        };
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = offset,
            .device_tick_start = 5'000 + offset,
            .payload_offset = 0,
            .payload_byte_count = chunk * chain_channels * sizeof(double),
            .signal_id = 11,
            .n_samples = static_cast<std::uint32_t>(chunk),
            .clock_sync =
                ClockSyncSnapshot{
                    .device_tick_reference = 5'000,
                    .host_time_reference_ns = 10'000'000,
                    .device_tick_rate = {1'000, 1},
                    .uncertainty_ns = 3,
                    .clock_domain = 41,
                    .generation = 2,
                    .flags = ClockSyncFlags::synchronized,
                },
        };
        std::memcpy(frame.payload_storage().data(), input.data() + offset * chain_channels,
                    chunk * chain_channels * sizeof(double));
        CHECK(frame.set_used_sizes(1, chunk * chain_channels * sizeof(double)) == StreamStatus::ok);
        CHECK(chain.process(frame, sink) == StreamStatus::ok);
        offset += chunk;
    }
    CHECK(chain.flush(sink) == StreamStatus::ok);
    CHECK(sink.values().size() == feature_rows * state_dim);
    CHECK(nearly_equal(sink.values(), {expected.data(), feature_rows * state_dim}));
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 2 && std::string_view{argv[1]} == "--allocation-only")
    {
        return test_allocation_and_fault_paths();
    }
    if (argc != 1)
    {
        return __LINE__;
    }
    if (const auto status = test_contract_and_schema(); status != 0)
    {
        return status;
    }
    if (const auto status = test_schema_rejection(); status != 0)
    {
        return status;
    }
    if (const auto status = test_fitted_feature_contract_rejection(); status != 0)
    {
        return status;
    }
    if (const auto status = test_offline_parity_and_chunk_invariance(); status != 0)
    {
        return status;
    }
    if (const auto status = test_multi_observation_reset_and_discontinuity(); status != 0)
    {
        return status;
    }
    if (const auto status = test_missing_observation_policy(); status != 0)
    {
        return status;
    }
    if (const auto status = test_training_capture_alignment_and_overflow(); status != 0)
    {
        return status;
    }
    if (const auto status = test_allocation_and_fault_paths(); status != 0)
    {
        return status;
    }
    if (const auto status = test_numerical_failure_is_a_fault(); status != 0)
    {
        return status;
    }
    if (const auto status = test_overflow_leaves_the_filter_on_its_prior(); status != 0)
    {
        return status;
    }
    if (const auto status = test_python_decoder_parity(); status != 0)
    {
        return status;
    }
    return test_feature_to_decoder_chain();
}
