/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

// What the native linear decoder adapter must agree with, and what it must
// refuse. The reference throughout is the affine decode written out in
// ``offline_decode`` below, in the order ``neurale.decoding.LinearDecoder``
// drives it -- select, scale, then one affine product per observation --
// because that is the decoder this adapter has to reproduce.
//
// The oracle is written out here rather than taken from
// ``models::LinearModel::predict``, because that is the very function the
// adapter calls: using it would compare the model to itself. The comparison
// carries a tolerance, since under MKL the adapter reaches a vendor GEMM that
// may accumulate in another order than the plain loop below.

#include "allocation_tracker.h"
#include "check_returns.h"
#include "linear_decoder_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <neurale/models/linear_model.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/linear_processor_chain.h>

namespace
{

using namespace neurale::streaming;

constexpr std::uint32_t features = 4;
constexpr std::size_t n_model_features = 3;
constexpr std::size_t n_outputs = 2;
constexpr std::uint32_t max_input_observations = 8;
constexpr std::size_t max_frames = 64;
constexpr std::size_t max_values = 4096;
constexpr std::uint64_t shift_ns = 2'000'000;
constexpr std::uint64_t first_observation_ns = 12'000'000;

/// Neither identity nor a prefix on purpose: a selection whose misuse a width
/// check could not catch is the only one worth testing.
constexpr std::array<std::size_t, n_model_features> selection{3, 0, 2};
constexpr std::array<double, n_model_features> scaler_center{0.5, -0.25, 1.0};
constexpr std::array<double, n_model_features> scaler_scale{2.0, 0.5, 1.5};

[[nodiscard]] neurale::models::LinearModelState model_state()
{
    neurale::models::LinearModelState state;
    state.n_features = n_model_features;
    state.n_outputs = n_outputs;
    state.coef = {0.75, -0.20, 0.40, -0.15, 0.60, 0.05};
    state.intercept = {0.30, -0.45};
    return state;
}

[[nodiscard]] neurale::pipeline::LinearDecoderAdapterConfig decoder_config()
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
    };
}

[[nodiscard]] std::vector<FeatureSetDescriptor>
feature_sets(std::vector<std::string> names = {"rate_0", "rate_1", "rate_2", "rate_3"},
             std::uint64_t descriptor_shift_ns = shift_ns)
{
    const auto names_size = names.size();
    return {FeatureSetDescriptor{
        .id = 47,
        .feature_names = std::move(names),
        .unit_ids = std::vector<UnitId>(names_size, 53),
        .source_stream_id = 11,
        .source_stream = "units",
        .algorithm_name = "rate",
        .algorithm_version = "1",
        .window_length_ns = 4'000'000,
        .shift_ns = descriptor_shift_ns,
        .timestamp_reference = FeatureTimestampReference::window_center,
    }};
}

[[nodiscard]] std::vector<UnitDescriptor> units(std::string symbol = "Hz")
{
    return {UnitDescriptor{.id = 53, .symbol = std::move(symbol), .description = "hertz"}};
}

[[nodiscard]] StreamSchema
make_schema(std::uint32_t channels = features, SignalKind kind = SignalKind::feature,
            SignalDType dtype = SignalDType::float64, FeatureSetId feature_set = 47,
            std::vector<std::string> names = {"rate_0", "rate_1", "rate_2", "rate_3"},
            SignalLayout layout = SignalLayout::sample_major, RationalRate rate = {500, 1},
            std::vector<UnitDescriptor> registry = units(), std::uint32_t nominal = 6)
{
    // StreamSchema requires the descriptor's shift to be the reciprocal of the
    // observation rate, so a schema at another rate has to move both together.
    const auto descriptor_shift_ns = 1'000'000'000ULL * rate.denominator / rate.numerator;
    const std::array signals{
        SignalSchema{
            11,
            dtype,
            channels,
            nominal,
            max_input_observations,
            rate,
            41,
            layout,
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
    auto descriptors = feature_sets(std::move(names), descriptor_shift_ns);
    descriptors.front().id = feature_set == 0 ? 47 : feature_set;
    return StreamSchema{29, signals, descriptors, std::move(registry)};
}

[[nodiscard]] ProcessorPrepareContext context(const StreamSchema& schema, std::size_t outputs = 32,
                                              std::size_t leases = 4) noexcept
{
    return {
        .input_schema = schema,
        .max_process_outputs = outputs,
        .max_flush_outputs = 32,
        .available_frame_pool_leases = leases,
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

/// A ramp with a per-column offset, so a selection that reads the wrong column
/// produces visibly different values rather than the same ones shuffled.
[[nodiscard]] std::vector<double> ramp(std::size_t observations)
{
    std::vector<double> values(observations * features);
    for (std::size_t row = 0; row < observations; ++row)
    {
        for (std::size_t column = 0; column < features; ++column)
        {
            values[row * features + column] =
                0.1 * static_cast<double>(row) + static_cast<double>(column) + 0.5;
        }
    }
    return values;
}

/// The decoded values, computed here from the definition.
///
/// Deliberately not `models::LinearModel::predict`: the adapter now calls that
/// itself, so using it here would compare the model to itself and prove only
/// that it is deterministic. The expectation is spelled out instead -- select,
/// scale, then one dot product per output written in the plainest order -- so a
/// mistake in the selection, the scaling, or the coefficient layout cannot be
/// mirrored into the oracle.
///
/// The comparison carries a tolerance rather than asserting bits: under MKL the
/// adapter reaches a vendor GEMM, which is free to accumulate in another order
/// than this loop does.
[[nodiscard]] std::vector<double> offline_decode(std::span<const double> input)
{
    const auto observations = input.size() / features;
    std::vector<double> decoded(observations * n_outputs);
    const auto state = model_state();
    for (std::size_t row = 0; row < observations; ++row)
    {
        std::array<double, n_model_features> scaled{};
        for (std::size_t i = 0; i < n_model_features; ++i)
        {
            const auto value = input[row * features + selection[i]];
            scaled[i] = (value - scaler_center[i]) / scaler_scale[i];
        }
        for (std::size_t output = 0; output < n_outputs; ++output)
        {
            double total = state.intercept[output];
            for (std::size_t i = 0; i < n_model_features; ++i)
            {
                total += state.coef[output * n_model_features + i] * scaled[i];
            }
            decoded[row * n_outputs + output] = total;
        }
    }
    return decoded;
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

template <typename Callable> [[nodiscard]] bool throws_invalid_argument(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

[[nodiscard]] int test_lda_labels_and_allocation()
{
    auto schema = make_schema();
    for (const bool binary : {true, false})
    {
        auto config = decoder_config();
        config.scaling = neurale::pipeline::FeatureScaling::none;
        config.scaler_center.clear();
        config.scaler_scale.clear();
        config.classes = binary ? std::vector<double>{17, 42} : std::vector<double>{17, 42, 9};
        config.model.n_outputs = binary ? 1 : 3;
        config.model.coef =
            binary ? std::vector<double>{1, 0, 0} : std::vector<double>{-1, 0, 0, 1, 0, 0, 0, 0, 0};
        config.model.intercept.assign(config.model.n_outputs, 0);
        neurale::pipeline::LinearDecoderAdapter adapter{config};
        const auto contract = adapter.prepare(context(schema));
        CHECK(contract.output_schema.signals().front().n_channels == 1);
        CollectingEmitter sink{contract.output_schema};
        FramePool pool{1, schema.signals().front().max_block_bytes, 1};
        FrameLease lease;
        CHECK(pool.try_acquire(lease) == StreamStatus::ok);
        std::vector<double> input(3 * features, 0);
        input[selection[0]] = -1;
        input[2 * features + selection[0]] = 1;
        CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        const std::array<double, 3> expected{17, 17, 42};
        CHECK(nearly_equal(sink.values(), expected));
        sink.clear();
        neurale::benchmark::reset_allocation_count();
        {
            neurale::benchmark::AllocationScope scope;
            CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
            scope.stop();
            CHECK(scope.count() == 0);
        }
        CHECK(nearly_equal(sink.values(), expected));
    }
    return 0;
}

} // namespace

namespace
{

[[nodiscard]] int test_contract_and_schema()
{
    auto schema = make_schema();
    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    CHECK(contract.accepted_input_schema.equivalent(schema));
    CHECK(contract.max_process_outputs_per_input == 1);
    // Nothing is buffered across frames, so a flush can never owe an output.
    CHECK(contract.max_flush_outputs == 0);
    // The decoded payload is a different width from the features, so the input
    // frame cannot be forwarded.
    CHECK(!contract.can_forward_input);
    CHECK(contract.required_resources.frame_pool_leases == 1);
    // One block of gathered rows plus one block of decoded rows, and the flat
    // selection the data plane walks.
    const auto expected_workspace =
        max_input_observations * (n_model_features + n_outputs) * sizeof(double) +
        n_model_features * sizeof(std::size_t);
    CHECK(contract.required_resources.workspace_bytes == expected_workspace);

    const auto outputs = contract.output_schema.signals();
    CHECK(outputs.size() == 1);
    const auto& decoded = outputs.front();
    const auto& input = schema.signals().front();
    CHECK(contract.output_schema.id() == 37);
    CHECK(decoded.id == 43);
    CHECK(decoded.n_channels == n_outputs);
    CHECK(decoded.dtype == SignalDType::float64);
    CHECK(decoded.layout == SignalLayout::sample_major);
    // The decoded target is what the features measured, so it keeps their clock
    // and their observation rate rather than resampling them.
    CHECK(decoded.fs.numerator == input.fs.numerator);
    CHECK(decoded.fs.denominator == input.fs.denominator);
    CHECK(decoded.clock_domain == input.clock_domain);
    CHECK(decoded.nominal_block_samples == input.nominal_block_samples);
    CHECK(decoded.max_block_samples == input.max_block_samples);
    // A decoded state is a sampled signal, not a feature: it carries no
    // feature-set descriptor and no observation timing of its own.
    CHECK(decoded.kind == SignalKind::sampled);
    CHECK(decoded.feature_set_id == 0);
    CHECK(decoded.observation_timing == ObservationTiming::not_applicable);
    CHECK(decoded.physical_unit == PhysicalUnit::dimensionless);
    CHECK(decoded.channel_set_id == 59);
    CHECK(decoded.calibration_id == input.calibration_id);
    CHECK(decoded.reference_id == input.reference_id);
    CHECK(decoded.device_tick_tracking == DeviceTickTracking::unavailable);
    return 0;
}

[[nodiscard]] int test_configuration_rejection()
{
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.selection.clear();
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.selected_feature_names.pop_back();
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    // A selection that repeats a column would feed one feature into two model
    // positions and silently double its weight.
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.selection = {0, 2, 2};
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.model.coef.pop_back();
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.model.coef[2] = std::numeric_limits<double>::quiet_NaN();
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.model.intercept[1] = std::numeric_limits<double>::infinity();
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    // A zero scale would divide the feature away rather than standardize it.
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.scaler_scale[0] = 0.0;
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.scaler_center.pop_back();
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.fitted_feature_contract.observation_rate = {0, 1};
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    CHECK(throws_invalid_argument(
        []
        {
            auto config = decoder_config();
            config.fitted_feature_contract.feature_unit_symbols.pop_back();
            neurale::pipeline::LinearDecoderAdapter adapter{config};
        }));
    return 0;
}

[[nodiscard]] int test_schema_rejection()
{
    auto sampled = make_schema(features, SignalKind::sampled);
    auto float32 = make_schema(features, SignalKind::feature, SignalDType::float32);
    auto wrong_set = make_schema(features, SignalKind::feature, SignalDType::float64, 61);
    // A schema whose width the StreamSchema itself accepts -- descriptor and
    // signal agree -- but which is not the feature set the decoder was fitted
    // on. A mismatch between signal and descriptor cannot be built at all:
    // StreamSchema refuses it before any adapter sees it.
    auto narrow = make_schema(features - 1, SignalKind::feature, SignalDType::float64, 47,
                              {"rate_0", "rate_1", "rate_2"});
    for (const auto* rejected : {&sampled, &float32, &wrong_set, &narrow})
    {
        CHECK(throws_invalid_argument(
            [&]
            {
                neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
                static_cast<void>(adapter.prepare(context(*rejected)));
            }));
    }
    auto schema = make_schema();
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
            static_cast<void>(adapter.prepare(context(schema, 0, 4)));
        }));
    CHECK(throws_invalid_argument(
        [&]
        {
            neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
            static_cast<void>(adapter.prepare(context(schema, 32, 0)));
        }));
    // An output that reuses the input's schema or signal id would make the
    // decoded stream indistinguishable from the features it came from.
    CHECK(throws_invalid_argument(
        [&]
        {
            auto config = decoder_config();
            config.output_schema_id = 29;
            neurale::pipeline::LinearDecoderAdapter adapter{config};
            static_cast<void>(adapter.prepare(context(schema)));
        }));
    CHECK(throws_invalid_argument(
        [&]
        {
            auto config = decoder_config();
            config.output_signal_id = 11;
            neurale::pipeline::LinearDecoderAdapter adapter{config};
            static_cast<void>(adapter.prepare(context(schema)));
        }));
    // A selection pointing past the input, and one pointing at a column that
    // carries a different feature than the decoder was fitted on.
    CHECK(throws_invalid_argument(
        [&]
        {
            auto config = decoder_config();
            config.selection = {0, 1, 9};
            neurale::pipeline::LinearDecoderAdapter adapter{config};
            static_cast<void>(adapter.prepare(context(schema)));
        }));
    CHECK(throws_invalid_argument(
        [&]
        {
            auto config = decoder_config();
            config.selection = {0, 1, 2};
            neurale::pipeline::LinearDecoderAdapter adapter{config};
            static_cast<void>(adapter.prepare(context(schema)));
        }));

    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
    static_cast<void>(adapter.prepare(context(schema)));
    // Re-preparing on the same schema is how a session restarts.
    static_cast<void>(adapter.prepare(context(schema)));
    // Differs only in nominal block size, which the fitted contract says
    // nothing about -- so this reaches the "schema cannot change" guard rather
    // than being turned away by the contract comparison first.
    auto changed = make_schema(features, SignalKind::feature, SignalDType::float64, 47,
                               {"rate_0", "rate_1", "rate_2", "rate_3"}, SignalLayout::sample_major,
                               {500, 1}, units(), 5);
    CHECK(throws_invalid_argument([&] { static_cast<void>(adapter.prepare(context(changed))); }));
    return 0;
}

[[nodiscard]] int test_fitted_contract_rejection()
{
    // Each of these is a runtime feature stream that would multiply correctly
    // and decode to wrong numbers: same width, different provenance.
    auto renamed = make_schema(features, SignalKind::feature, SignalDType::float64, 47,
                               {"rate_0", "rate_1", "rate_9", "rate_3"});
    auto rerated =
        make_schema(features, SignalKind::feature, SignalDType::float64, 47,
                    {"rate_0", "rate_1", "rate_2", "rate_3"}, SignalLayout::sample_major, {250, 1});
    auto reunit = make_schema(features, SignalKind::feature, SignalDType::float64, 47,
                              {"rate_0", "rate_1", "rate_2", "rate_3"}, SignalLayout::sample_major,
                              {500, 1}, units("V"));
    for (const auto* rejected : {&renamed, &rerated, &reunit})
    {
        CHECK(throws_invalid_argument(
            [&]
            {
                neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
                static_cast<void>(adapter.prepare(context(*rejected)));
            }));
    }
    auto schema = make_schema();
    for (const auto& mutate :
         std::vector<void (*)(neurale::pipeline::FittedFeatureContract&)>{
             [](auto& c) { c.window_length_ns = 1; },
             [](auto& c) { c.shift_ns = 1; },
             [](auto& c) { c.algorithm_name = "other"; },
             [](auto& c) { c.algorithm_version = "2"; },
             [](auto& c) { c.source_stream = "other"; },
             // timestamp_reference has a single enumerator today, so a
             // mismatching one cannot be constructed. The comparison is still
             // made in match_fitted_contract, against the day it gains another.
         })
    {
        CHECK(throws_invalid_argument(
            [&]
            {
                auto config = decoder_config();
                mutate(config.fitted_feature_contract);
                neurale::pipeline::LinearDecoderAdapter adapter{config};
                static_cast<void>(adapter.prepare(context(schema)));
            }));
    }
    // An unreduced rate is the same rate: 1000/2 must be accepted where 500/1
    // was fitted.
    auto unreduced = make_schema(features, SignalKind::feature, SignalDType::float64, 47,
                                 {"rate_0", "rate_1", "rate_2", "rate_3"},
                                 SignalLayout::sample_major, {1'000, 2});
    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
    static_cast<void>(adapter.prepare(context(unreduced)));
    return 0;
}

[[nodiscard]] int test_offline_parity_and_chunk_invariance()
{
    constexpr std::size_t observations = 8;
    const auto input = ramp(observations);
    const auto expected = offline_decode(input);
    auto schema = make_schema();

    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(sink.frame_count() == 1);
    CHECK(sink.values().size() == observations * n_outputs);
    CHECK(nearly_equal(sink.values(), expected));
    // The decoded block keeps the observation index and observation time of the
    // features it decoded, so the output timeline is the input timeline.
    CHECK(sink.block(0).sample_idx_start == 0);
    CHECK(sink.block(0).observation_time_start_ns == first_observation_ns);
    CHECK(sink.block(0).signal_id == 43);
    CHECK(sink.block(0).n_samples == observations);
    CHECK(sink.frame_header(0).schema_id == 37);
    CHECK(sink.frame_header(0).sequence == 1);
    CHECK(sink.idle());

    // Chunking must not change a value: prediction is stateless, so a block
    // boundary is not an event the decoder can notice.
    for (const auto& chunks : std::vector<std::vector<std::size_t>>{{1, 3, 4}, {5, 1, 2}, {8}})
    {
        neurale::pipeline::LinearDecoderAdapter chunked{decoder_config()};
        const auto chunk_contract = chunked.prepare(context(schema));
        CollectingEmitter chunk_sink{chunk_contract.output_schema};
        std::size_t offset = 0;
        SampleIndex observation_start = 0;
        for (const auto rows : chunks)
        {
            CHECK(fill_frame(lease.frame(), schema,
                             std::span<const double>{input}.subspan(offset, rows * features),
                             observation_start + 1, observation_start) == StreamStatus::ok);
            CHECK(chunked.process(lease.frame(), chunk_sink) == StreamStatus::ok);
            offset += rows * features;
            observation_start += static_cast<SampleIndex>(rows);
        }
        CHECK(chunk_sink.values().size() == observations * n_outputs);
        CHECK(nearly_equal(chunk_sink.values(), expected));
    }
    return 0;
}

[[nodiscard]] int test_statelessness_and_non_finite_rows()
{
    constexpr std::size_t observations = 4;
    const auto input = ramp(observations);
    const auto expected = offline_decode(input);
    auto schema = make_schema();
    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
    const auto contract = adapter.prepare(context(schema));
    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);

    // reset, a gap, and a flush are all no-ops on a stateless decoder: the same
    // block decodes to the same values across every one of them.
    for (int round = 0; round < 3; ++round)
    {
        sink.clear();
        CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
        CHECK(nearly_equal(sink.values(), expected));
        CHECK(adapter.reset() == StreamStatus::ok);
        CHECK(adapter.handle_discontinuity(Discontinuity{
                  .session_id = 5,
                  .previous_frame_sequence = 1,
                  .actual_frame_sequence = 7,
                  .reason = GapReason::source_gap,
              }) == StreamStatus::ok);
        CHECK(adapter.flush(sink) == StreamStatus::ok);
        // A flush emits nothing: every observation left with the frame that
        // carried it.
        CHECK(sink.frame_count() == 1);
    }

    // A non-finite value in a *selected* column has no decoded value to stand
    // in its place -- an affine model has no state to carry through it.
    for (const double poison :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()})
    {
        auto broken = input;
        broken[2 * features + selection[1]] = poison;
        sink.clear();
        CHECK(fill_frame(lease.frame(), schema, broken, 1, 0) == StreamStatus::ok);
        CHECK(adapter.process(lease.frame(), sink) == StreamStatus::invalid_frame);
        CHECK(sink.frame_count() == 0);
        CHECK(sink.idle());
    }

    // A non-finite value in an *unselected* column is never read and must not
    // fail the frame.
    auto unselected = input;
    unselected[1 * features + 1] = std::numeric_limits<double>::quiet_NaN();
    sink.clear();
    CHECK(fill_frame(lease.frame(), schema, unselected, 1, 0) == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(nearly_equal(sink.values(), expected));
    return 0;
}

[[nodiscard]] int test_linear_chain_terminal_sink()
{
    constexpr std::size_t observations = 6;
    const auto input = ramp(observations);
    const auto expected = offline_decode(input);
    auto schema = make_schema();
    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
    std::array<NativeFrameProcessor*, 1> stages{&adapter};
    LinearProcessorChain chain{stages};
    const auto contract = chain.prepare(context(schema));
    CHECK(contract.accepted_input_schema.equivalent(schema));
    CHECK(!contract.can_forward_input);

    CollectingEmitter sink{contract.output_schema};
    FramePool pool{1, schema.signals().front().max_block_bytes, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    CHECK(fill_frame(lease.frame(), schema, input, 1, 0) == StreamStatus::ok);
    CHECK(chain.process(lease.frame(), sink) == StreamStatus::ok);
    CHECK(chain.flush(sink) == StreamStatus::ok);
    CHECK(sink.values().size() == observations * n_outputs);
    CHECK(nearly_equal(sink.values(), expected));
    return 0;
}

[[nodiscard]] int test_training_capture_alignment_and_overflow()
{
    constexpr std::size_t observations = 2;
    constexpr SampleIndex sample_idx_start = 17;
    const auto input = ramp(observations);
    auto schema = make_schema();

    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
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
        CHECK(captured.n_features == n_model_features);
        for (std::size_t column = 0; column < n_model_features; ++column)
        {
            CHECK(captured.values[column] == input[row * features + selection[column]]);
        }
    }
    neurale::pipeline::DecoderTrainingObservation captured{};
    CHECK(capture->try_pop(captured) == StreamStatus::would_block);
    CHECK(capture->dropped() == 0);

    neurale::pipeline::LinearDecoderAdapter overflow{decoder_config()};
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
    neurale::pipeline::LinearDecoderAdapter adapter{decoder_config()};
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
    CollectingEmitter narrow{contract.output_schema, n_outputs * sizeof(double)};
    CHECK(adapter.reset() == StreamStatus::ok);
    CHECK(adapter.process(lease.frame(), narrow) == StreamStatus::output_limit);

    // Before prepare, every entry point reports invalid_state rather than
    // touching a model that is not there.
    neurale::pipeline::LinearDecoderAdapter unprepared{decoder_config()};
    CHECK(unprepared.reset() == StreamStatus::invalid_state);
    CHECK(unprepared.flush(sink) == StreamStatus::invalid_state);
    CHECK(unprepared.handle_discontinuity(Discontinuity{}) == StreamStatus::invalid_state);
    CHECK(unprepared.process(lease.frame(), sink) == StreamStatus::invalid_state);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (const auto status = test_lda_labels_and_allocation(); status != 0)
        return status;
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
    if (const auto status = test_configuration_rejection(); status != 0)
    {
        return status;
    }
    if (const auto status = test_schema_rejection(); status != 0)
    {
        return status;
    }
    if (const auto status = test_fitted_contract_rejection(); status != 0)
    {
        return status;
    }
    if (const auto status = test_offline_parity_and_chunk_invariance(); status != 0)
    {
        return status;
    }
    if (const auto status = test_statelessness_and_non_finite_rows(); status != 0)
    {
        return status;
    }
    if (const auto status = test_linear_chain_terminal_sink(); status != 0)
    {
        return status;
    }
    if (const auto status = test_training_capture_alignment_and_overflow(); status != 0)
    {
        return status;
    }
    return test_allocation_and_fault_paths();
}
