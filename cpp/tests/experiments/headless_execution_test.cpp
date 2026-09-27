/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "adaptive_center_out_actuator.h"
#include "allocation_counter.h"
#include "center_out_controller.h"
#include "check_counts.h"
#include "speech_controller.h"
#include "webgrid_controller.h"

#include <neurale/experiments/speech_replay.h>
#include <neurale/experiments/webgrid_replay.h>

#include <neurale/streaming/processor.h>
#include <neurale/streaming/runtime.h>
#include <neurale/streaming/source.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace
{
using namespace neurale::experiments;
using namespace neurale::execution;
using namespace neurale::streaming;

int failures = 0;

class ShiftedClock final : public NativeClock
{
  public:
    static constexpr HostTimeNs epoch = 10'000;

    ShiftedClock() noexcept : base_epoch_(default_native_clock().now_ns()) {}

    [[nodiscard]] HostTimeNs now_ns() noexcept override
    {
        return epoch + (default_native_clock().now_ns() - base_epoch_);
    }

    void wait_until(HostTimeNs deadline_ns) noexcept override
    {
        const auto delta = deadline_ns > epoch ? deadline_ns - epoch : 0;
        default_native_clock().wait_until(base_epoch_ + delta);
    }

    void wake() noexcept override
    {
        default_native_clock().wake();
    }

  private:
    HostTimeNs base_epoch_{};
};

center_out::CenterOut2DConfig center_out_config() noexcept
{
    center_out::RadialLayoutRequest request{};
    request.radius = 0.5;
    request.count = 2;
    request.center_id = 1;
    request.ids[0] = 2;
    request.ids[1] = 3;
    request.spokes[0] = 0;
    request.spokes[1] = 4;

    center_out::CenterOut2DConfig config{};
    config.geometry_unit = center_out::GeometryUnit::normalized;
    CHECK(center_out::build_radial_layout(request, config.layout) == ContractStatus::ok);
    config.acceptance = center_out::AcceptanceRegion{0.1, 0.1};
    config.cursor = center_out::CursorGeometry{0.0};
    config.movement_timeout = center_out::PhaseDurations{2'000'000'000, 2'000'000'000};
    config.hold_ns = 0;
    config.reward_dwell = center_out::PhaseDurations{0, 0};
    config.punish_dwell = center_out::PhaseDurations{0, 0};
    config.selection = center_out::TargetSelectionPolicy::repeat_until_success;
    config.seed = 17;
    config.trial_limit = 2;
    return config;
}

CommandSpace velocity_space() noexcept
{
    CommandSpace space{};
    space.id = 11;
    space.dim = 2;
    space.frame = CommandFrame::workspace_2d;
    space.axes[0] = CommandAxis{CommandAxisName::x, CommandUnit::normalized};
    space.axes[1] = CommandAxis{CommandAxisName::y, CommandUnit::normalized};
    return space;
}

CenterOutControllerConfig center_controller_config(std::size_t capacity = 16) noexcept
{
    CenterOutControllerConfig config{};
    config.decoded_signal_id = 9;
    config.paradigm = 41;
    config.task = center_out_config();
    config.guidance = center_out::CenterOutGuidanceConfig{center_out::GeometryUnit::normalized, 1.0,
                                                          4.0, 4.0, 0.05};
    config.velocity_space = velocity_space();
    config.linear_assistance = assistance::LinearAssistance{0.25};
    config.initial_position = {0.0, 0.0};
    config.cursor_min = {-1.0, -1.0};
    config.cursor_max = {1.0, 1.0};
    config.trace_capacity = capacity;
    config.assistance_method = assistance::AssistanceMethod::linear_blend;
    return config;
}

StreamSchema decoded_schema(PhysicalUnit unit = PhysicalUnit::dimensionless)
{
    const SignalSchema signal{9,
                              SignalDType::float64,
                              2,
                              2,
                              4,
                              RationalRate{10, 1},
                              7,
                              SignalLayout::sample_major,
                              DeviceTickTracking::unavailable,
                              unit,
                              5};
    const std::array signals{signal};
    return StreamSchema{13, signals};
}

StreamSchema single_decoded_schema()
{
    const SignalSchema signal{9,
                              SignalDType::float64,
                              2,
                              1,
                              1,
                              RationalRate{10, 1},
                              7,
                              SignalLayout::sample_major,
                              DeviceTickTracking::unavailable,
                              PhysicalUnit::dimensionless,
                              5};
    const std::array signals{signal};
    return StreamSchema{13, signals};
}

StreamSchema adaptive_feature_schema(std::uint32_t n_features = 2)
{
    const SignalSchema signal{22,
                              SignalDType::float64,
                              n_features,
                              1,
                              1,
                              RationalRate{10, 1},
                              7,
                              SignalLayout::sample_major,
                              DeviceTickTracking::unavailable,
                              PhysicalUnit::unspecified,
                              23,
                              0,
                              0,
                              SignalKind::feature,
                              24,
                              ObservationTiming::regular};
    const std::array signals{signal};
    std::vector<std::string> feature_names;
    feature_names.reserve(n_features);
    for (std::uint32_t feature = 0; feature < n_features; ++feature)
        feature_names.push_back("f" + std::to_string(feature));
    const std::array feature_sets{
        FeatureSetDescriptor{.id = 24,
                             .feature_names = std::move(feature_names),
                             .unit_ids = std::vector<UnitId>(n_features, 25),
                             .source_stream_id = 22,
                             .source_stream = "simulated",
                             .algorithm_name = "test",
                             .algorithm_version = "1",
                             .window_length_ns = 100'000'000,
                             .shift_ns = 100'000'000,
                             .timestamp_reference = FeatureTimestampReference::window_center}};
    const std::array units{UnitDescriptor{25, "1", "dimensionless"}};
    return StreamSchema{21, signals, feature_sets, units};
}

class ConstantVelocityDecoder final : public NativeFrameProcessor
{
  public:
    explicit ConstantVelocityDecoder(std::array<double, 2> velocity) : velocity_(velocity) {}

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {.accepted_input_schema = context.input_schema.clone(),
                .output_schema = single_decoded_schema(),
                .max_process_outputs_per_input = 1,
                .max_flush_outputs = 0,
                .can_forward_input = false,
                .required_resources = ProcessorResourceBounds{.frame_pool_leases = 1}};
    }

    StreamStatus process(FrameBorrow& input, FrameEmitter& emitter) noexcept override
    {
        FrameBorrow output;
        auto status = emitter.try_acquire(output);
        if (status != StreamStatus::ok)
            return status;
        output.header() = input.header();
        output.header().schema_id = 13;
        output.header().signal_block_count = 1;
        const auto source = input.blocks().front();
        output.block_storage()[0] = source;
        output.block_storage()[0].signal_id = 9;
        output.block_storage()[0].payload_offset = 0;
        output.block_storage()[0].payload_byte_count = sizeof(velocity_);
        std::memcpy(output.payload_storage().data(), velocity_.data(), sizeof(velocity_));
        status = output.set_used_sizes(1, sizeof(velocity_));
        return status == StreamStatus::ok ? emitter.publish_acquired() : status;
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
    std::array<double, 2> velocity_{};
};

struct AdaptiveFrameStorage
{
    std::array<SignalBlockHeader, 1> blocks{};
    alignas(double) std::array<std::byte, 2 * sizeof(double)> payload{};
};

FrameView adaptive_feature_frame(AdaptiveFrameStorage& storage, SampleIndex sample_idx,
                                 HostTimeNs time_ns) noexcept
{
    const std::array values{1.0, 2.0};
    std::memcpy(storage.payload.data(), values.data(), sizeof(values));
    storage.blocks[0] = SignalBlockHeader{.sample_idx_start = sample_idx,
                                          .observation_time_start_ns = time_ns,
                                          .payload_offset = 0,
                                          .payload_byte_count = sizeof(values),
                                          .signal_id = 22,
                                          .n_samples = 1};
    return FrameView{.header = FrameHeader{.session_id = 3,
                                           .sequence = sample_idx,
                                           .schema_id = 21,
                                           .source_clock_domain = 7,
                                           .signal_block_count = 1},
                     .blocks = storage.blocks,
                     .payload = storage.payload};
}

struct FrameStorage
{
    std::array<SignalBlockHeader, 1> blocks{};
    alignas(double) std::array<std::byte, 4 * 2 * sizeof(double)> payload{};
};

FrameView decoded_frame(FrameStorage& storage, std::span<const double> values,
                        std::uint32_t observations, std::uint64_t sequence, SampleIndex sample_idx,
                        HostTimeNs observation_time_ns) noexcept
{
    std::memcpy(storage.payload.data(), values.data(), values.size_bytes());
    storage.blocks[0] = SignalBlockHeader{
        .sample_idx_start = sample_idx,
        .observation_time_start_ns = observation_time_ns,
        .payload_offset = 0,
        .payload_byte_count = values.size_bytes(),
        .signal_id = 9,
        .n_samples = observations,
    };
    return FrameView{
        .header = FrameHeader{.session_id = 3,
                              .sequence = sequence,
                              .schema_id = 13,
                              .source_clock_domain = 7,
                              .signal_block_count = 1},
        .blocks = storage.blocks,
        .payload = std::span<const std::byte>{storage.payload.data(), values.size_bytes()},
    };
}

class GappedDecodedSource final : public NativeFrameSource
{
  public:
    StreamStatus read(MutableFrame&) noexcept override
    {
        return StreamStatus::end_of_stream;
    }

    StreamStatus read_message(MutableFrame& frame,
                              DiscontinuityLease& discontinuity) noexcept override
    {
        if (cancelled_.load(std::memory_order_acquire))
            return StreamStatus::stopped;
        if (idx_ == 1)
        {
            ++idx_;
            const SignalGap gap{.expected_sample_idx = 2,
                                .actual_sample_idx = 4,
                                .missing_samples = 2,
                                .signal_id = 9,
                                .reason = GapReason::source_gap,
                                .flags = SignalGapFlags::missing_samples_known};
            const auto status = discontinuity.assign(3, 0, 2, GapReason::source_gap,
                                                     std::span<const SignalGap>(&gap, 1));
            return status == StreamStatus::ok ? StreamStatus::discontinuity : status;
        }
        if (idx_ == 3)
            return StreamStatus::end_of_stream;
        const std::array values{0.0, 0.0, 0.0, 0.0};
        std::memcpy(frame.payload_storage().data(), values.data(), sizeof(values));
        const auto first_sample = idx_ == 0 ? 0ULL : 4ULL;
        frame.header() = FrameHeader{.session_id = 3,
                                     .sequence = idx_ == 0 ? 0ULL : 2ULL,
                                     .host_received_ns = idx_ == 0 ? 1'000ULL : 400'001'000ULL,
                                     .schema_id = 13,
                                     .source_clock_domain = 7,
                                     .signal_block_count = 1};
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = first_sample,
            .observation_time_start_ns = idx_ == 0 ? 1'000ULL : 400'001'000ULL,
            .payload_offset = 0,
            .payload_byte_count = sizeof(values),
            .signal_id = 9,
            .n_samples = 2,
        };
        ++idx_;
        return frame.set_used_sizes(1, sizeof(values));
    }

    [[nodiscard]] bool produces_discontinuities() const noexcept override
    {
        return true;
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }

    StreamStatus reset() noexcept override
    {
        idx_ = 0;
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::size_t idx_{};
    std::atomic<bool> cancelled_{};
};

class ForwardDecodedProcessor final : public NativeFrameProcessor
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
            .required_resources = ProcessorResourceBounds{.frame_pool_leases = 1},
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

RealtimeConfig runtime_config() noexcept
{
    RealtimeConfig config{};
    config.pool_capacity.source_owned = 2;
    config.pool_capacity.ingress_capacity = 8;
    config.pool_capacity.processor_owned = 2;
    config.pool_capacity.critical_edge_capacity = 8;
    config.pool_capacity.actuator_owned = 2;
    config.buffer_size = 4 * 2 * sizeof(double);
    config.max_signal_blocks = 1;
    config.discontinuity_capacity = 2;
    config.gaps_per_discontinuity = 1;
    config.max_process_outputs = 1;
    config.max_flush_outputs = 0;
    config.fault_history_capacity = 4;
    return config;
}

void test_center_out_terminal_consumer()
{
    auto wrong_unit_schema = decoded_schema(PhysicalUnit::volts);
    CenterOutController wrong_unit{};
    CHECK(wrong_unit.prepare(wrong_unit_schema, center_controller_config()) ==
          StreamStatus::realtime_configuration_failed);

    auto schema = decoded_schema();
    CenterOutController controller{};
    CHECK(controller.prepare(schema, center_controller_config()) == StreamStatus::ok);
    const auto prepared_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(controller.start(1'000, 50) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == prepared_allocations);
    CenterOutControlTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::session_start);

    FrameStorage storage{};
    const std::array values{0.0, 0.0, 1.0, 0.0};
    const auto frame = decoded_frame(storage, values, 2, 1, 0, 1'100);
    ActuatorCommand command{};
    command.valid_until_ns = 10'000;
    command.payload = frame;

    const auto before_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before_allocations);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::observation);
    CHECK(trace.sample_idx == 0);
    CHECK(trace.dt_ns == 100);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.sample_idx == 1);
    CHECK(trace.dt_ns == 100'000'000);
    CHECK(trace.assisted.values[0] > 0.0);
    CHECK(controller.position().x > 0.0);
    const auto intent = controller.intent_context();
    CHECK(intent.intent.valid);
    CHECK(intent.intent.sequence == trace.observation_ordinal);
    CHECK(intent.intent.intent_x == trace.guidance.values[0]);
    CHECK(intent.intent.intent_y == trace.guidance.values[1]);
    CHECK(intent.intent.source_frame_sequence == trace.frame_sequence);
    CHECK(intent.intent.source_sample_index == trace.sample_idx);
    CHECK(intent.target_id == trace.step.snapshot.active_target);
    CHECK(intent.trial_key == trace.step.snapshot.trial.key);
    const auto first_run_position = controller.position();

    FrameStorage bad_storage{};
    const std::array bad_values{(std::numeric_limits<double>::quiet_NaN)(), 0.0};
    const auto bad_frame = decoded_frame(bad_storage, bad_values, 1, 2, 2, 200'001'100);
    command.payload = bad_frame;
    // A decoded value that is not a number never reaches the cursor. Under the
    // default policy it is the *trial* that ends, not the run: the frame is
    // handled, the run continues, and what is refused is the one observation.
    const auto nan_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == nan_allocations);
    CHECK(controller.position().x == first_run_position.x);
    CHECK(controller.position().y == first_run_position.y);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::abnormal);
    CHECK(trace.abnormal.condition == AbnormalCondition::decoded_command_invalid);
    CHECK(trace.abnormal.policy == AbnormalPolicy::abort_trial);
    CHECK(trace.abnormal.response == AbnormalResponse::trial_aborted);
    CHECK(trace.has_aborted_trial);
    CHECK(trace.aborted_trial.outcome == TrialOutcome::aborted);
    CHECK(!abnormal_response_admits_trial(trace.abnormal.response));
    CHECK(controller.abnormal_summary().observed == 1);
    CHECK(controller.abnormal_summary().trials_affected == 1);
    CHECK(!controller.abnormal_summary().session_aborted);

    // A run that treats the same condition as fatal gets the old behaviour, and
    // gets it because it asked for it rather than because it is the only one
    // available.
    auto fatal_config = center_controller_config();
    fatal_config.abnormal.decoded_command_invalid = AbnormalPolicy::abort_session;
    CenterOutController fatal{};
    CHECK(fatal.prepare(schema, fatal_config) == StreamStatus::ok);
    CHECK(fatal.start(1'000, 50) == StreamStatus::ok);
    CHECK(fatal.try_pop_trace(trace) == StreamStatus::ok);
    ActuatorCommand fatal_command{};
    fatal_command.valid_until_ns = 10'000;
    fatal_command.payload = bad_frame;
    CHECK(fatal.submit(fatal_command, 2'000) == StreamStatus::invalid_frame);
    CHECK(fatal.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.abnormal.response == AbnormalResponse::session_aborted);
    CHECK(fatal.abnormal_summary().session_aborted);
    fatal.close();

    // A frame that is not the stream the run prepared against ends the run
    // whatever the configuration says about decoded values: every frame after
    // it is wrong in the same way. On a controller of its own, because a run
    // that ended stays ended -- and the rest of this test is about a run that
    // is still going.
    {
        auto wrong_clock = frame;
        wrong_clock.header.source_clock_domain = 99;
        CenterOutController mismatched{};
        CHECK(mismatched.prepare(schema, center_controller_config()) == StreamStatus::ok);
        CHECK(mismatched.start(1'000, 50) == StreamStatus::ok);
        CenterOutControlTrace mismatch_trace{};
        CHECK(mismatched.try_pop_trace(mismatch_trace) == StreamStatus::ok);
        ActuatorCommand mismatched_command{};
        mismatched_command.valid_until_ns = 10'000;
        mismatched_command.payload = wrong_clock;
        const auto resting = mismatched.position();
        CHECK(mismatched.submit(mismatched_command, 2'000) == StreamStatus::invalid_frame);
        CHECK(mismatched.position().x == resting.x);
        CHECK(mismatched.try_pop_trace(mismatch_trace) == StreamStatus::ok);
        CHECK(mismatch_trace.kind == CenterOutTraceKind::abnormal);
        CHECK(mismatch_trace.abnormal.condition == AbnormalCondition::input_schema_mismatch);
        CHECK(mismatch_trace.abnormal.response == AbnormalResponse::session_aborted);
        mismatched.close();
    }

    const SignalGap gap{.expected_sample_idx = 2,
                        .actual_sample_idx = 4,
                        .missing_samples = 2,
                        .signal_id = 9,
                        .reason = GapReason::source_gap,
                        .flags = SignalGapFlags::missing_samples_known};
    const std::array gaps{gap};
    const Discontinuity discontinuity{3, 1, 2, gaps, GapReason::source_gap};
    const auto before_gap_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(controller.handle_discontinuity(discontinuity) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before_gap_allocations);
    CHECK(controller.snapshot().state == center_out::CenterOutState::idle);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::discontinuity);
    // The gap and the decision about it are one record, not two.
    CHECK(trace.abnormal.condition == AbnormalCondition::source_discontinuity);
    CHECK(trace.abnormal.policy == AbnormalPolicy::abort_trial);

    const std::array after_gap{0.0, 0.0};
    FrameStorage gap_storage{};
    const auto next_frame = decoded_frame(gap_storage, after_gap, 1, 2, 4, 300'001'100);
    command.payload = next_frame;
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.restarted_after_discontinuity);
    CHECK(trace.dt_ns == 0);
    CHECK(trace.segment_start.snapshot.state == center_out::CenterOutState::move_to_center);
    CHECK(trace.step.snapshot.state != center_out::CenterOutState::hold_center);

    CHECK(controller.reset() == StreamStatus::ok);
    CHECK(controller.position().x == 0.0 && controller.position().y == 0.0);
    CHECK(controller.start(1'000, 50) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    command.payload = frame;
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(controller.position().x == first_run_position.x);
    CHECK(controller.position().y == first_run_position.y);
    controller.cancel();
    CHECK(controller.submit(command, 2'000) == StreamStatus::stopped);
    CHECK(controller.reset() == StreamStatus::ok);
    controller.close();
    controller.close();
    CHECK(controller.reset() == StreamStatus::invalid_state);

    CenterOutController bounded{};
    CHECK(bounded.prepare(schema, center_controller_config(2)) == StreamStatus::ok);
    CHECK(bounded.start(1'000, 50) == StreamStatus::ok);
    const auto bounded_snapshot = bounded.snapshot();
    CHECK(bounded.submit(command, 2'000) == StreamStatus::queue_overflow);
    CHECK(bounded.snapshot().time_ns == bounded_snapshot.time_ns);
    CHECK(bounded.position().x == 0.0 && bounded.position().y == 0.0);
}

void test_center_out_cursor_saturates_without_boundary_windup()
{
    auto config = center_controller_config(16);
    config.assistance_method = assistance::AssistanceMethod::none;
    config.linear_assistance = {};
    config.cursor_min = {-0.5, -0.5};
    config.cursor_max = {0.5, 0.5};
    config.training_capture_capacity = 4;
    config.presentation_state_capacity = 4;
    CenterOutController controller{};
    CHECK(controller.prepare(single_decoded_schema(), config) == StreamStatus::ok);
    CHECK(controller.start(1'000, 0) == StreamStatus::ok);
    CenterOutControlTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    FrameStorage first_storage{};
    const std::array outward{10.0, 0.0};
    CHECK(controller.consume(decoded_frame(first_storage, outward, 1, 0, 0, 100'001'000)) ==
          StreamStatus::ok);
    CHECK(controller.position().x == 0.5);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.position_before.x == 0.0);
    CHECK(trace.position_after.x == 0.5);

    CenterOutTrainingLabel label{};
    CHECK(controller.try_pop_training_label(label) == StreamStatus::ok);
    CHECK(label.cursor_position.x == 0.0);
    CenterOutPresentationState presentation{};
    CHECK(controller.try_pop_presentation_state(presentation) == StreamStatus::ok);
    CHECK(presentation.cursor.x == 0.5);

    FrameStorage pinned_storage{};
    CHECK(controller.consume(decoded_frame(pinned_storage, outward, 1, 1, 1, 200'001'000)) ==
          StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.position_before.x == 0.5);
    CHECK(trace.position_after.x == 0.5);
    CHECK(trace.position_after.x - trace.position_before.x == 0.0);
    CHECK(controller.try_pop_training_label(label) == StreamStatus::ok);
    CHECK(label.cursor_position.x == 0.5);
    CHECK(label.cursor_velocity.x == 0.0);
    CHECK(label.cursor_velocity.y == 0.0);
    CHECK(controller.try_pop_presentation_state(presentation) == StreamStatus::ok);
    CHECK(presentation.cursor.x == 0.5);

    FrameStorage inward_storage{};
    const std::array inward{-1.0, 0.0};
    CHECK(controller.consume(decoded_frame(inward_storage, inward, 1, 2, 2, 300'001'000)) ==
          StreamStatus::ok);
    CHECK(controller.position().x == 0.4);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.position_before.x == 0.5);
    CHECK(trace.position_after.x == 0.4);
    controller.close();
}

void test_center_out_runtime_terminal_discontinuity_delivery()
{
    auto schema = decoded_schema();
    CenterOutController controller{};
    CHECK(controller.prepare(schema, center_controller_config(32)) == StreamStatus::ok);
    CHECK(controller.start(0, 0) == StreamStatus::ok);
    GappedDecodedSource source{};
    ForwardDecodedProcessor processor{};
    ShiftedClock clock{};
    NativeStreamRunner runtime{
        decoded_schema(),           runtime_config(), source, processor, controller, clock,
        default_safety_controller()};
    CHECK(runtime.prepare() == StreamStatus::ok);
    CHECK(runtime.arm() == StreamStatus::ok);
    CHECK(runtime.run() == StreamStatus::ok);

    CenterOutControlTrace trace{};
    bool saw_discontinuity = false;
    bool saw_restart = false;
    bool saw_timing = false;
    while (controller.try_pop_trace(trace) == StreamStatus::ok)
    {
        saw_discontinuity = saw_discontinuity || trace.kind == CenterOutTraceKind::discontinuity;
        saw_restart = saw_restart || trace.restarted_after_discontinuity;
        if (trace.kind == CenterOutTraceKind::observation)
        {
            CHECK(trace.input_ready_ns >= ShiftedClock::epoch);
            CHECK(trace.decoded_ready_ns >= trace.input_ready_ns);
            CHECK(trace.decoded_ready_ns < ShiftedClock::epoch + 1'000'000'000);
            saw_timing = true;
        }
    }
    CHECK(saw_discontinuity);
    CHECK(saw_restart);
    CHECK(saw_timing);
    CHECK(runtime.stats().discontinuities >= 1);
}

void test_center_out_full_assistance_is_computer_controlled()
{
    auto schema = decoded_schema();
    auto config = center_controller_config(8);
    config.linear_assistance = assistance::LinearAssistance{1.0};
    CenterOutController controller{};
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CHECK(controller.start(1'000, 0) == StreamStatus::ok);

    CenterOutControlTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    FrameStorage storage{};
    const std::array decoded_away_from_target{10'000.0, -10'000.0, 10'000.0, -10'000.0};
    ActuatorCommand command{};
    command.valid_until_ns = 1'000'000'000;
    command.payload = decoded_frame(storage, decoded_away_from_target, 2, 1, 0, 1'100);
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    for (std::size_t observation = 0; observation < 2; ++observation)
    {
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(trace.assisted.values == trace.guidance.values);
    }
    CHECK(trace.decoded.values != trace.assisted.values);
}

void test_center_out_assistance_schedule_switches_after_trial_boundary()
{
    auto config = center_controller_config(256);
    config.assistance_blocks = {
        CenterOutAssistanceBlock{assistance::LinearAssistance{1.0}, 1},
        CenterOutAssistanceBlock{assistance::LinearAssistance{0.0}, 1},
    };
    auto schema = decoded_schema();
    CenterOutController controller{};
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CHECK(controller.start(1'000, 0) == StreamStatus::ok);
    CenterOutControlTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    FrameStorage storage{};
    const std::array decoded{-7.0, 3.0};
    bool completed_first_trial = false;
    bool checked_second_block = false;
    for (std::uint64_t sample = 0; sample < 200 && !checked_second_block; ++sample)
    {
        const auto time_ns = 1'000 + sample * 100'000'000;
        const auto frame = decoded_frame(storage, decoded, 1, sample + 1, sample, time_ns);
        CHECK(controller.consume(frame) == StreamStatus::ok);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        if (completed_first_trial)
        {
            CHECK(trace.linear_assistance.assistance == 0.0);
            CHECK(trace.assisted.values[0] == trace.decoded.values[0]);
            CHECK(trace.assisted.values[1] == trace.decoded.values[1]);
            checked_second_block = true;
            break;
        }
        CHECK(trace.linear_assistance.assistance == 1.0);
        CHECK(trace.assisted.values[0] == trace.guidance.values[0]);
        CHECK(trace.assisted.values[1] == trace.guidance.values[1]);
        for (std::uint8_t i = 0; i < trace.step.n_events; ++i)
        {
            completed_first_trial |= trace.step.events[i].kind == ExperimentEventKind::trial_stop;
        }
    }
    CHECK(completed_first_trial);
    CHECK(checked_second_block);
    controller.close();

    auto invalid = center_controller_config();
    invalid.assistance_blocks = {
        CenterOutAssistanceBlock{assistance::LinearAssistance{1.0}, 1},
    };
    CenterOutController rejected{};
    CHECK(rejected.prepare(schema, invalid) == StreamStatus::realtime_configuration_failed);
}

webgrid::WebGridConfig webgrid_config() noexcept
{
    webgrid::WebGridConfig config{};
    config.rows = 1;
    config.columns = 2;
    config.bounds = webgrid::TaskBounds{0.0, 2.0, 0.0, 1.0};
    config.n_candidates = 2;
    config.candidates[0] = 1;
    config.candidates[1] = 2;
    config.schedule = webgrid::TargetScheduleKind::explicit_sequence;
    config.immediate_repetition = webgrid::ImmediateRepetitionPolicy::forbid;
    config.correct_selection = webgrid::CorrectSelectionPolicy::advance_target;
    config.incorrect_selection = webgrid::IncorrectSelectionPolicy::keep_current_target;
    config.n_explicit = 2;
    config.explicit_targets[0] = 1;
    config.explicit_targets[1] = 2;
    config.initial_target = 1;
    config.target_count_limit = 2;
    config.metric_version = webgrid::kMetricVersion1;
    return config;
}

SelectionEvent selection(const WebGridHeadlessController& controller,
                         webgrid::PointerPosition pointer, ExperimentTimeNs time,
                         SequenceOrdinal sequence) noexcept
{
    SelectionEvent event{};
    const auto snapshot = controller.snapshot();
    CHECK(webgrid::make_selection_event(webgrid_config(), pointer, snapshot.active_target,
                                        snapshot.trial, 52, time, sequence,
                                        event) == ContractStatus::ok);
    return event;
}

void test_webgrid_headless_order_and_bounds()
{
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(webgrid_config(), 4) == StreamStatus::ok);
    const auto prepared_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(controller.start(52, 10) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == prepared_allocations);
    WebGridHeadlessTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    const webgrid::PointerPosition wrong{1.5, 0.5};
    const auto miss = selection(controller, wrong, 20, 1);
    const auto before_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(controller.process(20, wrong, miss) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before_allocations);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == WebGridTraceKind::selection);
    CHECK(trace.pointer.x == 1.5);
    CHECK(trace.step.selection_processed);
    CHECK(!trace.step.selection.event.correct);
    CHECK(trace.step.snapshot.active_target == 1);

    CHECK(controller.process(19, wrong) == StreamStatus::invalid_frame);
    CHECK(controller.snapshot().time_ns == 20);

    const webgrid::PointerPosition correct{0.5, 0.5};
    const auto hit = selection(controller, correct, 20, 2);
    CHECK(controller.process(20, correct, hit) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.step.selection.event.correct);
    CHECK(trace.step.trial_decided);
    CHECK(trace.step.snapshot.active_target == 2);

    CHECK(controller.reset() == StreamStatus::ok);
    CHECK(controller.start(52, 10) == StreamStatus::ok);
    CHECK(controller.snapshot().active_target == 1);
    controller.cancel();
    CHECK(controller.process(20, correct) == StreamStatus::stopped);
    CHECK(controller.reset() == StreamStatus::ok);
    controller.close();
    controller.close();
    CHECK(controller.reset() == StreamStatus::invalid_state);

    WebGridHeadlessController bounded{};
    CHECK(bounded.prepare(webgrid_config(), 1) == StreamStatus::ok);
    CHECK(bounded.start(52, 10) == StreamStatus::ok);
    CHECK(bounded.process(11, correct) == StreamStatus::queue_overflow);
    CHECK(bounded.snapshot().time_ns == 10);
}

speech::SpeechCatalog speech_catalog() noexcept
{
    speech::SpeechCatalog catalog{};
    catalog.count = 2;
    for (std::size_t i = 0; i < 2; ++i)
    {
        auto& item = catalog.entries[i];
        item.id = static_cast<StimulusId>(i + 1);
        item.label = static_cast<std::uint32_t>(i + 10);
        item.metadata = 100 + i;
        item.content = speech::SpeechContentKind::text;
        const char* text = i == 0 ? "left" : "right";
        item.text_length = static_cast<std::uint8_t>(std::strlen(text));
        std::memcpy(item.text.data(), text, item.text_length);
    }
    return catalog;
}

speech::SpeechCueConfig speech_config() noexcept
{
    speech::SpeechCueConfig config{};
    config.black_bound_ns = 100;
    config.cross_bound_ns = 0;
    config.content_bound_ns = 200;
    config.inter_trial_ns = 0;
    config.seed = 99;
    config.n_trials = 2;
    config.schedule = speech::SpeechScheduleKind::seeded;
    config.stimulus_order = speech::StimulusOrderPolicy::sequential;
    config.sampler_version = kCurrentSamplerVersion;
    config.cross_enabled = false;
    config.n_stimuli = 2;
    config.stimuli[0] = 1;
    config.stimuli[1] = 2;
    return config;
}

void test_speech_explicit_time_and_prepared_payloads()
{
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);

    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 4) == StreamStatus::ok);
    const auto prepared_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(scheduler.start(63, 1'000) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == prepared_allocations);
    SpeechHeadlessTrace trace{};
    CHECK(scheduler.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.step.n_requests == 1);
    CHECK(!trace.presentations[0].has_payload);

    CHECK(scheduler.advance(999) == StreamStatus::invalid_frame);
    CHECK(scheduler.snapshot().time_ns == 1'000);

    const auto first_end = scheduler.snapshot().timeline.trial.end_ns;
    const auto before_allocations = allocations.load(std::memory_order_relaxed);
    CHECK(scheduler.advance(first_end) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before_allocations);
    CHECK(scheduler.try_pop_trace(trace) == StreamStatus::ok);
    bool content_resolved = false;
    for (std::uint8_t i = 0; i < trace.n_presentations; ++i)
    {
        if (trace.presentations[i].request.stimulus_id != kUnsetStimulusId)
        {
            content_resolved = true;
            CHECK(trace.presentations[i].has_payload);
            CHECK(trace.presentations[i].payload.id == 1);
        }
    }
    CHECK(content_resolved);
    CHECK(trace.step.snapshot.trial.ordinal == 1);

    const auto second_end = scheduler.snapshot().timeline.trial.end_ns;
    CHECK(scheduler.advance(second_end) == StreamStatus::ok);
    CHECK(scheduler.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.step.snapshot.state == speech::SpeechState::complete);
    CHECK(scheduler.advance(second_end) == StreamStatus::stopped);

    CHECK(scheduler.reset() == StreamStatus::ok);
    CHECK(scheduler.start(63, 1'000) == StreamStatus::ok);
    CHECK(scheduler.snapshot().trial.ordinal == 0);
    CHECK(scheduler.snapshot().schedule.stimulus_id == schedules[0].stimulus_id);
    scheduler.cancel();
    CHECK(scheduler.advance(1'001) == StreamStatus::stopped);
    CHECK(scheduler.reset() == StreamStatus::ok);
    scheduler.close();
    scheduler.close();
    CHECK(scheduler.reset() == StreamStatus::invalid_state);

    SpeechHeadlessScheduler bounded{};
    CHECK(bounded.prepare(config, catalog, schedules, 1) == StreamStatus::ok);
    CHECK(bounded.start(63, 1'000) == StreamStatus::ok);
    const auto bounded_snapshot = bounded.snapshot();
    CHECK(bounded.advance(bounded_snapshot.timeline.trial.end_ns) == StreamStatus::queue_overflow);
    CHECK(bounded.snapshot().time_ns == bounded_snapshot.time_ns);
}

// --- Abnormal conditions at the controller boundary -------------------

void test_oversized_input_is_rejected_without_effect()
{
    // The invariant: a logical input either happens or does not. An
    // input that crosses a pointer gap produces two records, so reserving one
    // would let the gap be reported -- counted, and the target marked
    // inadmissible -- and then the step refused for want of room. The caller
    // would be told the input failed, retry it, and have the same gap reported
    // a second time, because nothing that advances the pointer clock had run.
    const auto build = [](std::size_t capacity)
    {
        WebGridControllerConfig config{};
        config.task = webgrid_config();
        config.trace_capacity = capacity;
        config.max_pointer_interval_ns = 500'000'000;
        return config;
    };
    const webgrid::PointerPosition inside{0.5, 0.5};

    // One slot short of what the input needs.
    {
        WebGridHeadlessController controller{};
        CHECK(controller.prepare(build(2)) == StreamStatus::ok);
        CHECK(controller.start(52, 10) == StreamStatus::ok);
        WebGridHeadlessTrace trace{};
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(controller.process(20, inside) == StreamStatus::ok);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

        // Two entries left free, then one of them filled, leaving exactly one.
        CHECK(controller.process(30, inside) == StreamStatus::ok);
        const auto snapshot_before = controller.snapshot();
        const auto before = controller.abnormal_summary();
        const auto dropped_before = controller.dropped_trace_count();

        CHECK(controller.process(1'000'000'030, inside) == StreamStatus::queue_overflow);

        // Nothing moved: no condition was counted, no target was invalidated,
        // and the machine is where it was.
        const auto after = controller.abnormal_summary();
        CHECK(after.observed == before.observed);
        CHECK(after.trials_affected == before.trials_affected);
        CHECK(controller.dropped_trace_count() == dropped_before + 2);
        CHECK(controller.snapshot().state == snapshot_before.state);
        CHECK(same_trial(controller.snapshot().trial, snapshot_before.trial));

        // And the retry, once there is room, reports the gap exactly once.
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(controller.process(1'000'000'030, inside) == StreamStatus::ok);
        CHECK(controller.abnormal_summary().observed == before.observed + 1);
        controller.close();
    }

    // Exactly what the input needs: the whole operation happens.
    {
        WebGridHeadlessController controller{};
        CHECK(controller.prepare(build(2)) == StreamStatus::ok);
        CHECK(controller.start(52, 10) == StreamStatus::ok);
        WebGridHeadlessTrace trace{};
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(controller.process(20, inside) == StreamStatus::ok);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

        CHECK(controller.process(1'000'000'020, inside) == StreamStatus::ok);
        CHECK(controller.abnormal_summary().observed == 1);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(trace.kind == WebGridTraceKind::abnormal);
        CHECK(trace.abnormal.condition == AbnormalCondition::input_gap);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(trace.kind == WebGridTraceKind::pointer_update);
        CHECK(!trace.acquisition_timing_valid);
        controller.close();
    }
}

void test_stale_observation_costs_one_queue_entry()
{
    // Center-Out reserves one slot per row of a frame before it processes any
    // of them. A stale row still has to fit in that reservation, which is why
    // its decision travels on the observation it was taken about rather than in
    // a record beside it: the alternative is a row that costs two slots, and a
    // later row of the same frame refused after the machine was already reset.
    auto schema = decoded_schema();
    auto config = center_controller_config(2);
    config.max_observation_interval_ns = 100'000'000;
    CenterOutController controller{};
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CHECK(controller.start(1'000, 50) == StreamStatus::ok);
    CenterOutControlTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    FrameStorage first_storage{};
    const std::array moving{1.0, 0.0};
    ActuatorCommand command{};
    command.valid_until_ns = 10'000;
    command.payload = decoded_frame(first_storage, moving, 1, 1, 0, 1'100);
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    // One free slot, and a row that has a decision to record as well as an
    // observation. Both fit, because they are one entry.
    FrameStorage late_storage{};
    command.payload = decoded_frame(late_storage, moving, 1, 2, 1, 1'000'001'100);
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(controller.dropped_trace_count() == 0);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::observation);
    CHECK(trace.has_abnormal);
    CHECK(trace.abnormal.condition == AbnormalCondition::input_stale);
    // And nothing else was queued behind it.
    CHECK(controller.try_pop_trace(trace) != StreamStatus::ok);
    controller.close();
}

void test_aborted_run_rejects_later_input()
{
    // `abort_session` is a statement about the run, not about the return value.
    // Both controllers return a fatal status when a condition reaches that
    // severity, but a status only ends a run if somebody acts on it, and a
    // caller stepping the controller directly -- headless, or from a harness
    // that logs statuses rather than obeying them -- is not the runtime. The
    // run has to be over from the instant it says so, and the record of it has
    // to stay one record.
    {
        auto schema = decoded_schema();
        auto config = center_controller_config(16);
        config.abnormal.decoded_command_invalid = AbnormalPolicy::abort_session;
        CenterOutController controller{};
        CHECK(controller.prepare(schema, config) == StreamStatus::ok);
        CHECK(controller.start(1'000, 50) == StreamStatus::ok);
        CenterOutControlTrace trace{};
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

        FrameStorage bad_storage{};
        const std::array broken{std::numeric_limits<double>::quiet_NaN(), 0.0};
        ActuatorCommand command{};
        command.valid_until_ns = 10'000;
        command.payload = decoded_frame(bad_storage, broken, 1, 1, 0, 1'100);
        CHECK(controller.submit(command, 2'000) != StreamStatus::ok);
        CHECK(controller.abnormal_summary().session_aborted);
        CHECK(controller.halted());
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(trace.kind == CenterOutTraceKind::abnormal);
        CHECK(trace.abnormal.condition == AbnormalCondition::decoded_command_invalid);
        CHECK(trace.abnormal.response == AbnormalResponse::session_aborted);

        // A perfectly good frame afterwards is refused, and refused as what it
        // is: input that arrived after the run reached a terminal state.
        FrameStorage good_storage{};
        const std::array moving{1.0, 0.0};
        command.payload = decoded_frame(good_storage, moving, 1, 2, 1, 1'200);
        CHECK(controller.submit(command, 2'000) == StreamStatus::stopped);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(trace.kind == CenterOutTraceKind::abnormal);
        CHECK(trace.abnormal.condition == AbnormalCondition::input_after_terminal);
        // Recorded once, however long the caller keeps pushing.
        CHECK(controller.submit(command, 2'000) == StreamStatus::stopped);
        CHECK(controller.try_pop_trace(trace) != StreamStatus::ok);
        controller.close();
    }

    {
        WebGridControllerConfig config{};
        config.task = webgrid_config();
        config.trace_capacity = 16;
        config.max_pointer_interval_ns = 500'000'000;
        config.abnormal.input_discontinuity = AbnormalPolicy::abort_session;
        WebGridHeadlessController controller{};
        CHECK(controller.prepare(config) == StreamStatus::ok);
        CHECK(controller.start(52, 10) == StreamStatus::ok);
        WebGridHeadlessTrace trace{};
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        const webgrid::PointerPosition inside{0.5, 0.5};
        CHECK(controller.process(20, inside) == StreamStatus::ok);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

        CHECK(controller.process(1'000'000'020, inside) != StreamStatus::ok);
        CHECK(controller.abnormal_summary().session_aborted);
        CHECK(controller.halted());
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(trace.kind == WebGridTraceKind::abnormal);
        CHECK(trace.abnormal.condition == AbnormalCondition::input_gap);
        CHECK(trace.abnormal.response == AbnormalResponse::session_aborted);

        // Including a selection. A selection accepted after the run ended would
        // be scored and recorded as though the run were still the run.
        SelectionEvent click{};
        click.time_ns = 1'000'000'030;
        click.kind = SelectionKind::discrete;
        CHECK(controller.process(1'000'000'030, inside, click) == StreamStatus::stopped);
        CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
        CHECK(trace.kind == WebGridTraceKind::abnormal);
        CHECK(trace.abnormal.condition == AbnormalCondition::input_after_terminal);
        CHECK(controller.process(1'000'000'040, inside) == StreamStatus::stopped);
        CHECK(controller.try_pop_trace(trace) != StreamStatus::ok);
        controller.close();
    }
}

/// Turns the controller's own trace records into a replay recording.
///
/// This is what a reader of the recording does, expressed against the trace
/// structs rather than against the encoded records -- the bridge writes exactly
/// these fields, so a harvester that works here is a statement about whether
/// the *fields* are sufficient for a replay.
class CenterOutTraceHarvester
{
  public:
    void take(const CenterOutControlTrace& trace)
    {
        switch (trace.kind)
        {
        case CenterOutTraceKind::session_start:
            keep_step(trace.step);
            keep_target(trace.step.snapshot, trace.time_ns);
            return;
        case CenterOutTraceKind::discontinuity:
        case CenterOutTraceKind::abnormal:
            keep_decision(trace);
            return;
        case CenterOutTraceKind::observation:
            break;
        }
        if (trace.has_abnormal)
        {
            keep_decision(trace);
        }
        if (trace.restarted_after_discontinuity)
        {
            forget_target();
            keep_step(trace.segment_start);
            keep_target(trace.segment_start.snapshot, trace.time_ns);
        }

        center_out::CenterOutReplayInput input{};
        input.kind = center_out::CenterOutReplayInputKind::observation;
        input.time_ns = trace.time_ns;
        input.decoded = {trace.decoded.values[0], trace.decoded.values[1]};
        input.frame_sequence = trace.frame_sequence;
        input.sample_idx = trace.sample_idx;
        inputs.push_back(input);

        center_out::CenterOutCursorSample cursor{};
        cursor.time_ns = trace.time_ns;
        cursor.before = trace.position_before;
        cursor.after = trace.position_after;
        cursor.dt_ns = trace.dt_ns;
        cursor.frame_sequence = trace.frame_sequence;
        cursor.sample_idx = trace.sample_idx;
        cursor.state = trace.step.snapshot.state;
        cursor.trial = trace.step.snapshot.trial;
        cursors.push_back(cursor);

        velocities.push_back(center_out::CenterOutVelocitySample{trace.time_ns, trace.decoded,
                                                                 trace.guidance, trace.assisted,
                                                                 trace.step.snapshot.trial});
        guidance.push_back(
            center_out::CenterOutGuidanceObservation{trace.time_ns, trace.guidance_sample});

        keep_step(trace.step);
        keep_target(trace.step.snapshot, trace.time_ns);
    }

    [[nodiscard]] center_out::CenterOutReplayRecording
    recording(const ReplayProvenance& provenance, ExperimentTimeNs origin_ns) const noexcept
    {
        center_out::CenterOutReplayRecording recording{};
        recording.provenance = provenance;
        recording.origin_ns = origin_ns;
        recording.inputs = inputs;
        recording.expected.transitions = transitions;
        recording.expected.events = events;
        recording.expected.trials = trials;
        recording.expected.aborts = aborts;
        recording.expected.targets = targets;
        recording.expected.cursor = cursors;
        recording.expected.vel = velocities;
        recording.expected.guidance = guidance;
        recording.expected.has_transitions = true;
        recording.expected.has_events = true;
        recording.expected.has_trials = true;
        recording.expected.has_aborts = true;
        recording.expected.has_targets = true;
        recording.expected.has_cursor = true;
        recording.expected.has_velocity = true;
        recording.expected.has_guidance = true;
        recording.run_end_recorded = true;
        return recording;
    }

    std::vector<center_out::CenterOutReplayInput> inputs{};
    std::vector<StateTransition> transitions{};
    std::vector<ExperimentEvent> events{};
    std::vector<center_out::CenterOutTrial> trials{};
    std::vector<center_out::CenterOutReplayAbort> aborts{};
    std::vector<center_out::CenterOutTargetOnset> targets{};
    std::vector<center_out::CenterOutCursorSample> cursors{};
    std::vector<center_out::CenterOutVelocitySample> velocities{};
    std::vector<center_out::CenterOutGuidanceObservation> guidance{};

  private:
    void keep_decision(const CenterOutControlTrace& trace)
    {
        center_out::CenterOutReplayInput input{};
        input.kind = center_out::CenterOutReplayInputKind::decision;
        input.time_ns = trace.abnormal.time_ns;
        input.condition = trace.abnormal.condition;
        input.response = trace.abnormal.response;
        inputs.push_back(input);
        if (trace.has_aborted_trial)
        {
            aborts.push_back(center_out::CenterOutReplayAbort{
                trace.aborted_trial, trace.abnormal.condition, trace.abnormal.response});
        }
        if (trace.abnormal.response == AbnormalResponse::trial_aborted ||
            trace.abnormal.response == AbnormalResponse::session_aborted)
        {
            forget_target();
        }
    }

    void keep_step(const center_out::CenterOutStepResult& step)
    {
        for (std::uint8_t i = 0; i < step.n_transitions; ++i)
        {
            transitions.push_back(step.transitions[i]);
        }
        for (std::uint8_t i = 0; i < step.n_events; ++i)
        {
            events.push_back(step.events[i]);
        }
        if (step.trial_decided)
        {
            trials.push_back(step.trial);
        }
    }

    void forget_target()
    {
        has_target_ = false;
        last_target_ = kUnsetTargetId;
    }

    void keep_target(const center_out::CenterOutSnapshot& snapshot, ExperimentTimeNs time_ns)
    {
        if (snapshot.active_target == kUnsetTargetId)
        {
            return;
        }
        if (has_target_ && snapshot.active_target == last_target_)
        {
            return;
        }
        has_target_ = true;
        last_target_ = snapshot.active_target;
        center_out::CenterOutTargetOnset onset{};
        onset.time_ns = time_ns;
        onset.target_id = snapshot.active_target;
        onset.pos = snapshot.active_position;
        onset.phase = snapshot.phase;
        onset.outward_target = snapshot.outward_target;
        onset.outward_idx = snapshot.outward_idx;
        onset.trial = snapshot.trial;
        targets.push_back(onset);
    }

    TargetId last_target_{kUnsetTargetId};
    bool has_target_{};
};

center_out::CenterOutReplayConfig replay_config(const CenterOutControllerConfig& config) noexcept
{
    center_out::CenterOutReplayConfig replay{};
    replay.paradigm = config.paradigm;
    replay.task = config.task;
    replay.guidance = config.guidance;
    replay.velocity_space = config.velocity_space;
    replay.linear_assistance = config.linear_assistance;
    replay.assistance_method = config.assistance_method;
    replay.initial_position = config.initial_position;
    replay.cursor_min = config.cursor_min;
    replay.cursor_max = config.cursor_max;
    replay.abnormal = config.abnormal;
    return replay;
}

void test_center_out_trace_is_enough_to_re_execute()
{
    // The evidence question here, put to the real controller rather than
    // to a test double: drive a run, keep only what its trace records carry,
    // and re-execute the paradigm from that alone. A field the controller does
    // not record is a field this replay cannot have, so a match here is a
    // statement about the recording and not only about the replay.
    auto schema = decoded_schema();
    auto config = center_controller_config(256);
    config.max_observation_interval_ns = 100'000'000;
    CenterOutController controller{};
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CHECK(controller.start(1'000, 1'000) == StreamStatus::ok);

    CenterOutTraceHarvester harvester;
    const auto harvest = [&harvester, &controller]()
    {
        CenterOutControlTrace trace{};
        while (controller.try_pop_trace(trace) == StreamStatus::ok)
        {
            harvester.take(trace);
        }
    };
    harvest();

    ActuatorCommand command{};
    command.valid_until_ns = 10'000;
    std::vector<FrameStorage> storage(40);
    std::uint64_t host = 1'000;
    for (std::uint64_t i = 0; i < 30; ++i)
    {
        // Aimed at whatever the controller says is up, so the run reaches its
        // targets rather than timing out on every one of them.
        const auto snapshot = controller.snapshot();
        const auto pos = controller.position();
        double vx = 0.0;
        double vy = 0.0;
        if (snapshot.active_target != kUnsetTargetId)
        {
            const double dx = snapshot.active_position.x - pos.x;
            const double dy = snapshot.active_position.y - pos.y;
            const double norm = std::sqrt(dx * dx + dy * dy);
            if (norm > 0.0)
            {
                vx = dx / norm;
                vy = dy / norm;
            }
        }
        // One deliberately long interval, so the recording holds a stale
        // decision, the trial it ended, and the segment that restarted after it.
        host += i == 12 ? 400'000'000 : 50'000'000;
        const std::array values{vx, vy};
        command.payload = decoded_frame(storage[i], values, 1, i + 1, i, host);
        const auto status = controller.submit(command, 2'000);
        CHECK(status == StreamStatus::ok || status == StreamStatus::stopped);
        harvest();
    }
    CHECK(controller.dropped_trace_count() == 0);
    CHECK(!harvester.trials.empty());
    CHECK(!harvester.aborts.empty());

    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(replay_config(config)) == ContractStatus::ok);
    const auto report = replay.run(harvester.recording(replay.provenance(), 1'000));
    if (report.verdict != ReplayVerdict::match)
    {
        std::cerr << "replay verdict " << replay_verdict_name(report.verdict) << " rejection "
                  << replay_rejection_name(report.rejection) << " item "
                  << replay_item_name(report.first_mismatch.item) << " field "
                  << report.first_mismatch.field << " index " << report.first_mismatch.idx << '\n';
    }
    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(report.items_compared > 50);

    // And the same recording replayed a second time answers the same, which is
    // what makes a replay a check rather than a measurement.
    CHECK(replay.run(harvester.recording(replay.provenance(), 1'000)).verdict ==
          ReplayVerdict::match);

    // The match above is only worth something if this replay would have noticed
    // a difference. One recorded cursor position, moved, and it does.
    CHECK(!harvester.cursors.empty());
    harvester.cursors[harvester.cursors.size() / 2].after.x += 1.0 / 1024.0;
    const auto disagreed = replay.run(harvester.recording(replay.provenance(), 1'000));
    CHECK(disagreed.verdict == ReplayVerdict::mismatch);
    CHECK(disagreed.first_mismatch.item == ReplayItem::cursor);
    CHECK(disagreed.first_mismatch.field == "after_x");
    controller.close();
}

// --- The real recording bridge's trace is enough to re-execute WebGrid --

/// Turns the WebGridHeadlessController's own trace records into a replay
/// recording, the WebGrid analogue of CenterOutTraceHarvester. A field the
/// controller does not record is a field this replay cannot have, so a match
/// here is a statement about the recording bridge and not only about the replay.
class WebGridTraceHarvester
{
  public:
    void take(const WebGridHeadlessTrace& trace)
    {
        switch (trace.kind)
        {
        case WebGridTraceKind::session_start:
            metrics_ = trace.step.snapshot.metrics;
            keep_target(trace.step.snapshot);
            return;
        case WebGridTraceKind::abnormal:
            keep_decision(trace);
            return;
        case WebGridTraceKind::pointer_update:
        case WebGridTraceKind::selection:
            break;
        }

        webgrid::WebGridReplayInput input{};
        input.kind = trace.kind == WebGridTraceKind::selection
                         ? webgrid::WebGridReplayInputKind::selection
                         : webgrid::WebGridReplayInputKind::pointer;
        input.time_ns = trace.step.snapshot.time_ns;
        input.pointer = trace.pointer;
        if (trace.kind == WebGridTraceKind::selection)
        {
            input.selection = trace.step.selection.event;
        }
        inputs.push_back(input);

        webgrid::WebGridPointerSample sample{};
        sample.time_ns = trace.step.snapshot.time_ns;
        sample.pointer = trace.pointer;
        sample.state = trace.step.snapshot.state;
        sample.active_target = trace.step.snapshot.active_target;
        sample.trial = trace.step.snapshot.trial;
        pointer.push_back(sample);
        metrics_ = trace.step.snapshot.metrics;

        if (trace.step.selection_processed)
        {
            webgrid::WebGridReplaySelection selection{};
            selection.record = trace.step.selection;
            selection.acquisition_timing_valid = trace.acquisition_timing_valid;
            selections.push_back(selection);
        }
        if (trace.step.trial_decided)
        {
            webgrid::WebGridReplayTrial trial{};
            trial.trial = trace.step.trial;
            trial.recorded_outcome =
                trace.trial_invalidated ? TrialOutcome::aborted : trace.step.trial.record.outcome;
            trial.acquisition_timing_valid = trace.acquisition_timing_valid;
            trials.push_back(trial);
        }
        keep_target(trace.step.snapshot);
    }

    [[nodiscard]] webgrid::WebGridReplayRecording
    recording(const ReplayProvenance& provenance, ExperimentTimeNs origin_ns) const noexcept
    {
        webgrid::WebGridReplayRecording recording{};
        recording.provenance = provenance;
        recording.origin_ns = origin_ns;
        recording.inputs = inputs;
        recording.expected.pointer = pointer;
        recording.expected.selections = selections;
        recording.expected.trials = trials;
        recording.expected.targets = targets;
        recording.expected.metrics = metrics_;
        recording.expected.invalidated_targets = invalidated_targets_;
        recording.expected.has_pointer = true;
        recording.expected.has_selections = true;
        recording.expected.has_trials = true;
        recording.expected.has_targets = true;
        recording.expected.has_metrics = true;
        recording.run_end_recorded = true;
        return recording;
    }

    [[nodiscard]] std::uint64_t invalidated_targets() const noexcept
    {
        return invalidated_targets_;
    }
    [[nodiscard]] webgrid::WebGridMetrics metrics() const noexcept
    {
        return metrics_;
    }

    std::vector<webgrid::WebGridReplayInput> inputs{};
    std::vector<webgrid::WebGridPointerSample> pointer{};
    std::vector<webgrid::WebGridReplaySelection> selections{};
    std::vector<webgrid::WebGridReplayTrial> trials{};
    std::vector<webgrid::WebGridTargetOnset> targets{};

  private:
    void keep_decision(const WebGridHeadlessTrace& trace)
    {
        webgrid::WebGridReplayInput input{};
        input.time_ns = trace.abnormal.time_ns;
        input.condition = trace.abnormal.condition;
        input.response = trace.abnormal.response;
        switch (trace.abnormal.condition)
        {
        case AbnormalCondition::input_gap:
            input.kind = webgrid::WebGridReplayInputKind::pointer_gap;
            break;
        case AbnormalCondition::observer_frame_drop:
            input.kind = webgrid::WebGridReplayInputKind::observer_drop;
            break;
        default:
            input.kind = webgrid::WebGridReplayInputKind::halt;
            break;
        }
        inputs.push_back(input);
        if (trace.abnormal.response == AbnormalResponse::trial_invalidated)
        {
            ++invalidated_targets_;
        }
    }

    void keep_target(const webgrid::WebGridSnapshot& snapshot)
    {
        if (snapshot.active_target == kUnsetTargetId)
        {
            return;
        }
        if (has_target_ && snapshot.active_target == last_target_ &&
            snapshot.target_onset_ns == last_onset_ns_)
        {
            return;
        }
        has_target_ = true;
        last_target_ = snapshot.active_target;
        last_onset_ns_ = snapshot.target_onset_ns;
        targets.push_back(webgrid::WebGridTargetOnset{
            snapshot.target_onset_ns, snapshot.active_target, snapshot.completed, snapshot.trial});
    }

    webgrid::WebGridMetrics metrics_{};
    std::uint64_t invalidated_targets_{};
    ExperimentTimeNs last_onset_ns_{};
    TargetId last_target_{kUnsetTargetId};
    bool has_target_{};
};

void test_webgrid_trace_is_enough_to_re_execute()
{
    // The WebGrid analogue of the Center-Out evidence test: drive the real
    // controller -- a pointer observation, a gap that invalidates the target it
    // crosses, the selection that closes the invalidated target, and a clean
    // second target -- keep only what its trace records carry, and re-execute
    // the paradigm from that alone. A field the controller does not record is a
    // field this replay cannot have, so a match here is a statement about the
    // recording bridge and not only about the replay.
    WebGridControllerConfig config{};
    config.task = webgrid_config();
    config.trace_capacity = 256;
    // Half a second without a pointer sample invalidates the acquisition it
    // crosses. The selection that closes such a target is still recorded -- raw
    // data is not deleted because it turned out to be unusable -- but it is
    // recorded as a measurement nobody may quote, which is the WebGrid field
    // most at risk of being missed by a recording bridge.
    config.max_pointer_interval_ns = 500'000'000;
    config.abnormal.input_discontinuity = AbnormalPolicy::abort_trial;

    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config) == StreamStatus::ok);
    CHECK(controller.start(52, 10) == StreamStatus::ok);

    WebGridTraceHarvester harvester;
    const auto harvest = [&harvester, &controller]()
    {
        WebGridHeadlessTrace trace{};
        while (controller.try_pop_trace(trace) == StreamStatus::ok)
        {
            harvester.take(trace);
        }
    };
    harvest();

    // Target 1: observe, then a gap that invalidates the acquisition, then the
    // selection that closes the invalidated target.
    CHECK(controller.process(20, webgrid::PointerPosition{0.5, 0.5}) == StreamStatus::ok);
    harvest();
    // A jump past the pointer-interval bound: the controller records the gap
    // (input_gap, trial_invalidated) before the observation that revealed it.
    CHECK(controller.process(1'000'000'040, webgrid::PointerPosition{0.5, 0.5}) ==
          StreamStatus::ok);
    harvest();
    {
        const auto event =
            selection(controller, webgrid::PointerPosition{0.5, 0.5}, 1'000'000'050, 1);
        CHECK(controller.process(1'000'000'050, webgrid::PointerPosition{0.5, 0.5}, event) ==
              StreamStatus::ok);
    }
    harvest();

    // Target 2: a clean acquisition.
    CHECK(controller.process(1'000'000'060, webgrid::PointerPosition{1.5, 0.5}) ==
          StreamStatus::ok);
    harvest();
    {
        const auto event =
            selection(controller, webgrid::PointerPosition{1.5, 0.5}, 1'000'000'070, 2);
        CHECK(controller.process(1'000'000'070, webgrid::PointerPosition{1.5, 0.5}, event) ==
              StreamStatus::ok);
    }
    harvest();

    CHECK(controller.dropped_trace_count() == 0);
    CHECK(harvester.trials.size() == 2);
    CHECK(harvester.invalidated_targets() == 1);
    // The invalidated target's trial is written as aborted beside the machine's
    // own success verdict, and the next target's is clean -- both of which a
    // replay has to reproduce separately.
    CHECK(harvester.trials[0].recorded_outcome == TrialOutcome::aborted);
    CHECK(harvester.trials[0].trial.record.outcome == TrialOutcome::success);
    CHECK(harvester.trials[1].recorded_outcome != TrialOutcome::aborted);
    CHECK(!harvester.selections.empty());
    CHECK(!harvester.pointer.empty());

    webgrid::WebGridReplay replay{};
    webgrid::WebGridReplayConfig replay_config{};
    replay_config.paradigm = 52;
    replay_config.task = controller.configuration();
    replay_config.abnormal = controller.controller_configuration().abnormal;
    CHECK(replay.prepare(replay_config) == ContractStatus::ok);
    const auto report = replay.run(harvester.recording(replay.provenance(), 10));
    if (report.verdict != ReplayVerdict::match)
    {
        std::cerr << "webgrid replay verdict " << replay_verdict_name(report.verdict)
                  << " rejection " << replay_rejection_name(report.rejection) << " item "
                  << replay_item_name(report.first_mismatch.item) << " field "
                  << report.first_mismatch.field << " index " << report.first_mismatch.idx << '\n';
    }
    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(report.items_compared > 10);

    // A replay is a check, not a measurement: the same recording answers twice.
    CHECK(replay.run(harvester.recording(replay.provenance(), 10)).verdict == ReplayVerdict::match);

    // And the match is only worth something if this replay would have noticed a
    // difference. One recorded pointer sample, moved, and it does.
    harvester.pointer[harvester.pointer.size() / 2].pointer.x += 0.25;
    const auto disagreed = replay.run(harvester.recording(replay.provenance(), 10));
    CHECK(disagreed.verdict == ReplayVerdict::mismatch);
    CHECK(disagreed.first_mismatch.item == ReplayItem::cursor);
    CHECK(disagreed.first_mismatch.field == "x");
    controller.close();
}

// --- The real recording bridge's trace is enough to re-execute Speech --

/// Turns the SpeechHeadlessScheduler's own trace records into a replay
/// recording, the Speech analogue of CenterOutTraceHarvester. The scheduler
/// emits one trace per advance, plus the opening trace its start produced; a
/// harvester that works here is a statement about whether the fields those
/// traces carry are sufficient for a replay.
class SpeechTraceHarvester
{
  public:
    void take(const SpeechHeadlessTrace& trace)
    {
        if (trace.is_abnormal)
        {
            speech::SpeechReplayInput input{};
            input.kind = speech::SpeechReplayInputKind::halt;
            input.time_ns = trace.abnormal.time_ns;
            input.condition = trace.abnormal.condition;
            input.response = trace.abnormal.response;
            inputs.push_back(input);
            return;
        }

        if (!seen_opening_)
        {
            // The opening trace is what start() produced. It contributes its
            // transitions, events, requests, and the first trial's schedule and
            // phases, but it is not itself a recorded semantic input: the run's
            // own clock begins with the first advance.
            seen_opening_ = true;
            origin_ns_ = trace.step.snapshot.time_ns;
        }
        else
        {
            speech::SpeechReplayInput input{};
            input.kind = speech::SpeechReplayInputKind::advance;
            input.time_ns = trace.step.snapshot.time_ns;
            inputs.push_back(input);
        }

        keep_step(trace.step);
        keep_trial_start(trace.step.snapshot);
    }

    [[nodiscard]] speech::SpeechReplayRecording
    recording(const ReplayProvenance& provenance) const noexcept
    {
        speech::SpeechReplayRecording recording{};
        recording.provenance = provenance;
        recording.origin_ns = origin_ns_;
        recording.inputs = inputs;
        recording.expected.transitions = transitions;
        recording.expected.events = events;
        recording.expected.requests = requests;
        recording.expected.phases = phases;
        recording.expected.schedules = schedules;
        recording.expected.trials = trials;
        recording.expected.reports = reports;
        recording.expected.has_transitions = true;
        recording.expected.has_events = true;
        recording.expected.has_requests = true;
        recording.expected.has_phases = true;
        recording.expected.has_schedules = true;
        recording.expected.has_trials = true;
        recording.expected.has_reports = true;
        recording.run_end_recorded = true;
        return recording;
    }

    [[nodiscard]] ExperimentTimeNs origin_ns() const noexcept
    {
        return origin_ns_;
    }

    std::vector<speech::SpeechReplayInput> inputs{};
    std::vector<StateTransition> transitions{};
    std::vector<ExperimentEvent> events{};
    std::vector<PresentationRequest> requests{};
    std::vector<speech::SpeechReplayPhase> phases{};
    std::vector<speech::SpeechTrialSchedule> schedules{};
    std::vector<speech::SpeechReplayTrial> trials{};
    std::vector<speech::SpeechReplayReportDecision> reports{};

  private:
    void keep_step(const speech::SpeechStepResult& step)
    {
        for (std::uint8_t i = 0; i < step.n_transitions; ++i)
        {
            transitions.push_back(step.transitions[i]);
        }
        for (std::uint8_t i = 0; i < step.n_events; ++i)
        {
            events.push_back(step.events[i]);
        }
        for (std::uint8_t i = 0; i < step.n_requests; ++i)
        {
            requests.push_back(step.requests[i]);
        }
        if (step.trial_decided)
        {
            speech::SpeechReplayTrial trial{};
            trial.trial = step.trial;
            // The scheduler-driven run meets no presenter reports, so no trial
            // is invalidated: the recording's own outcome column agrees with the
            // machine's. A presenter-driven run would set these through the
            // report path, which is not what this evidence test exercises.
            trial.recorded_outcome = step.trial.record.outcome;
            trial.invalidated = false;
            trials.push_back(trial);
        }
    }

    void keep_trial_start(const speech::SpeechSnapshot& snapshot)
    {
        if (snapshot.state == speech::SpeechState::idle ||
            snapshot.state == speech::SpeechState::complete)
        {
            return;
        }
        if (has_trial_ && snapshot.trial.ordinal == last_trial_)
        {
            return;
        }
        has_trial_ = true;
        last_trial_ = snapshot.trial.ordinal;
        schedules.push_back(snapshot.schedule);
        for (std::uint8_t i = 0; i < snapshot.timeline.count; ++i)
        {
            phases.push_back(
                speech::SpeechReplayPhase{snapshot.timeline.phases[i], snapshot.trial});
        }
    }

    ExperimentTimeNs origin_ns_{};
    TrialOrdinal last_trial_{};
    bool seen_opening_{};
    bool has_trial_{};
};

void test_speech_trace_is_enough_to_re_execute()
{
    // The Speech analogue of the Center-Out evidence test, put to the real
    // scheduler rather than to a test double: start a run, advance it through
    // every trial to completion, keep only what its trace records carry, and
    // re-execute the paradigm from that alone. A field the scheduler does not
    // record is a field this replay cannot have, so a match here is a statement
    // about the recording bridge and not only about the replay.
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);

    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 256) == StreamStatus::ok);
    CHECK(scheduler.start(63, 1'000) == StreamStatus::ok);

    SpeechTraceHarvester harvester;
    const auto harvest = [&harvester, &scheduler]()
    {
        SpeechHeadlessTrace trace{};
        while (scheduler.try_pop_trace(trace) == StreamStatus::ok)
        {
            harvester.take(trace);
        }
    };
    harvest();

    // Advance through both trials. Advancing to a trial's end instant decides
    // it and starts the next; the last advances the run to complete.
    for (int trial = 0; trial < 2; ++trial)
    {
        const auto end_ns = scheduler.snapshot().timeline.trial.end_ns;
        CHECK(scheduler.advance(end_ns) == StreamStatus::ok);
        harvest();
    }
    CHECK(scheduler.snapshot().state == speech::SpeechState::complete);
    CHECK(scheduler.dropped_trace_count() == 0);

    CHECK(harvester.trials.size() == 2);
    CHECK(harvester.schedules.size() == 2);
    CHECK(!harvester.phases.empty());
    CHECK(!harvester.requests.empty());
    CHECK(!harvester.transitions.empty());

    speech::SpeechReplay replay{};
    speech::SpeechReplayConfig replay_config{};
    replay_config.paradigm = 63;
    replay_config.task = scheduler.configuration();
    replay_config.catalog = scheduler.catalog();
    replay_config.abnormal = scheduler.abnormal_policies();
    CHECK(replay.prepare(replay_config) == ContractStatus::ok);
    CHECK(replay.authority() == ReplayAuthority::regenerate);
    // The session provenance fingerprints the full prepared schedule, which the
    // scheduler froze and exposes; the output stream carries only trials that
    // actually started, which the harvester collected.
    const auto report = replay.run(harvester.recording(replay.provenance(scheduler.schedules())));
    if (report.verdict != ReplayVerdict::match)
    {
        std::cerr << "speech replay verdict " << replay_verdict_name(report.verdict)
                  << " rejection " << replay_rejection_name(report.rejection) << " item "
                  << replay_item_name(report.first_mismatch.item) << " field "
                  << report.first_mismatch.field << " index " << report.first_mismatch.idx << '\n';
    }
    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(report.completeness == ReplayCompleteness::complete);
    CHECK(report.items_compared > 20);

    // A replay is a check, not a measurement: the same recording answers twice.
    CHECK(replay.run(harvester.recording(replay.provenance(scheduler.schedules()))).verdict ==
          ReplayVerdict::match);

    // And the match is only worth something if this replay would have noticed a
    // difference. One recorded phase boundary, moved, and it does.
    CHECK(!harvester.phases.empty());
    harvester.phases[0].phase.interval.end_ns += 1;
    const auto disagreed =
        replay.run(harvester.recording(replay.provenance(scheduler.schedules())));
    CHECK(disagreed.verdict == ReplayVerdict::mismatch);
    CHECK(disagreed.first_mismatch.item == ReplayItem::phase);
    CHECK(disagreed.first_mismatch.field == "end_ns");
    scheduler.close();
}
void test_pointer_gap_invalidates_crossed_acquisition()
{
    WebGridControllerConfig config{};
    config.task = webgrid_config();
    config.trace_capacity = 16;
    // Half a second without a pointer sample is not a pointer stream any more.
    config.max_pointer_interval_ns = 500'000'000;

    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config) == StreamStatus::ok);
    CHECK(controller.start(52, 10) == StreamStatus::ok);
    WebGridHeadlessTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == WebGridTraceKind::session_start);

    const webgrid::PointerPosition inside{0.5, 0.5};
    CHECK(controller.process(20, inside) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.acquisition_timing_valid);
    CHECK(!trace.trial_invalidated);

    // A drop on a path that carries no task input. It is recorded, and it
    // changes nothing -- which is the whole distinction this call exists for.
    const auto before_drop = allocations.load(std::memory_order_relaxed);
    CHECK(controller.note_observer_drop(30, 7) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before_drop);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == WebGridTraceKind::abnormal);
    CHECK(trace.abnormal.condition == AbnormalCondition::observer_frame_drop);
    CHECK(trace.abnormal.policy == AbnormalPolicy::record);
    CHECK(trace.abnormal.response == AbnormalResponse::recorded);
    CHECK(trace.acquisition_timing_valid);
    CHECK(controller.abnormal_summary().trials_affected == 0);

    CHECK(controller.process(40, inside) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    // Still a measurement: a dropped monitoring frame did not stop the pointer
    // from being observed.
    CHECK(trace.acquisition_timing_valid);

    // Now a real gap in the pointer data itself, detected from the interval.
    CHECK(controller.process(1'000'000'040, inside) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == WebGridTraceKind::abnormal);
    CHECK(trace.abnormal.condition == AbnormalCondition::input_gap);
    CHECK(trace.abnormal.policy == AbnormalPolicy::abort_trial);
    // WebGridMachine owns when a target ends and offers no way to abandon one,
    // so the target in flight is marked inadmissible rather than ended -- and
    // the record says which of the two happened rather than claiming the other.
    CHECK(trace.abnormal.response == AbnormalResponse::trial_invalidated);
    CHECK(!abnormal_response_admits_trial(trace.abnormal.response));
    CHECK(controller.abnormal_summary().trials_affected == 1);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == WebGridTraceKind::pointer_update);
    CHECK(!trace.acquisition_timing_valid);

    // The selection that closes the target still happens and is still recorded
    // -- raw data is not deleted because it turned out to be unusable -- but it
    // is recorded as a measurement nobody may use.
    const auto snapshot = controller.snapshot();
    const webgrid::PointerPosition on_target{0.5, 0.5};
    CHECK(snapshot.active_target == 1);
    const auto event = selection(controller, on_target, 1'000'000'050, 1);
    CHECK(controller.process(1'000'000'050, on_target, event) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == WebGridTraceKind::selection);
    CHECK(trace.step.trial_decided);
    CHECK(trace.trial_invalidated);
    CHECK(!trace.acquisition_timing_valid);
    // The machine's own verdict is untouched: it decided a correct selection,
    // and it was one. What the run adds is that the interval it was made over
    // is not a number anyone may quote.
    CHECK(trace.step.trial.record.outcome == TrialOutcome::success);

    // The next target starts clean.
    CHECK(controller.process(1'000'000'060, inside) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.acquisition_timing_valid);
    CHECK(!trace.trial_invalidated);
    controller.close();
}

void test_stale_or_repeated_selection_is_rejected()
{
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(webgrid_config(), 16) == StreamStatus::ok);
    CHECK(controller.start(52, 10) == StreamStatus::ok);
    WebGridHeadlessTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    const webgrid::PointerPosition on_target{0.5, 0.5};
    const auto first = selection(controller, on_target, 20, 1);
    CHECK(controller.process(20, on_target, first) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.step.selection_processed);
    CHECK(trace.step.trial_decided);
    const auto after_first = controller.snapshot().metrics.correct_selections;
    CHECK(after_first == 1);

    // The same selection identity again. WebGridMachine rejects a repeated or
    // regressed selection sequence, and this controller boundary check does
    // not add a second opinion about it: a duplicate is a contract violation
    // of the selection stream,
    // not an abnormal condition of the run.
    // Still running on the next target, so the refusal below is the machine
    // rejecting a repeated selection identity rather than a completed run
    // refusing everything.
    CHECK(controller.snapshot().state == webgrid::WebGridState::active_target);
    CHECK(controller.process(30, on_target, first) == StreamStatus::consumer_failure);
    CHECK(controller.snapshot().metrics.correct_selections == after_first);
    CHECK(controller.abnormal_summary().observed == 0);
    controller.close();
}

void test_input_after_halt_is_rejected_and_recorded_once()
{
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(webgrid_config(), 16) == StreamStatus::ok);
    CHECK(controller.start(52, 10) == StreamStatus::ok);
    WebGridHeadlessTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    const webgrid::PointerPosition on_target{0.5, 0.5};
    CHECK(controller.process(20, on_target) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    controller.halt(30, AbnormalCondition::emergency_stop);
    CHECK(controller.halted());
    // halt() only latches. It has to: while the run is live the trace queue has
    // one producer, and halt() may be called from another thread entirely.
    CHECK(controller.try_pop_trace(trace) == StreamStatus::would_block);

    // Every kind of task input is refused from here, selections included. A
    // selection accepted after a terminal condition would be scored.
    const auto after_halt = selection(controller, on_target, 40, 1);
    CHECK(controller.process(40, on_target) == StreamStatus::stopped);
    CHECK(controller.process(50, on_target, after_halt) == StreamStatus::stopped);
    CHECK(controller.snapshot().metrics.correct_selections == 0);

    // Recorded once, however many refusals there were. A run that kept
    // producing a record per refused frame would turn a stop into a trace loss.
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == WebGridTraceKind::abnormal);
    CHECK(trace.abnormal.condition == AbnormalCondition::input_after_terminal);
    CHECK(trace.abnormal.response == AbnormalResponse::input_refused);
    CHECK(trace.abnormal.policy == AbnormalPolicy::record);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::would_block);

    // The record the halt itself owes is produced once the producer is quiet.
    controller.finish_halt();
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.abnormal.condition == AbnormalCondition::emergency_stop);
    CHECK(trace.abnormal.policy == AbnormalPolicy::abort_session);
    CHECK(trace.abnormal.response == AbnormalResponse::session_aborted);
    CHECK(controller.abnormal_summary().session_aborted);
    // Idempotent: a second finish owes nothing.
    controller.finish_halt();
    CHECK(controller.try_pop_trace(trace) == StreamStatus::would_block);
    controller.close();
}

void test_semantic_time_jump_is_not_fault()
{
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);

    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 16) == StreamStatus::ok);
    CHECK(scheduler.start(11, 1'000) == StreamStatus::ok);
    SpeechHeadlessTrace trace{};
    CHECK(scheduler.try_pop_trace(trace) == StreamStatus::ok);

    // One advance across a whole trial's worth of nanoseconds. Speech time is
    // semantic: the machine crosses every phase boundary in between, in order,
    // exactly as if it had been stepped through them. Nothing about the size of
    // the step is abnormal, and a run that reported it as one would fault on
    // every deliberately coarse advance.
    CHECK(scheduler.advance(1'000 + 10'000'000) == StreamStatus::ok);
    CHECK(scheduler.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(!trace.is_abnormal);
    CHECK(trace.step.n_transitions > 0);
    CHECK(scheduler.abnormal_summary().observed == 0);
    CHECK(!scheduler.abnormal_summary().session_aborted);

    // A halt, on the other hand, refuses every further advance.
    scheduler.halt(2'000, AbnormalCondition::emergency_stop);
    CHECK(scheduler.advance(3'000) == StreamStatus::stopped);
    CHECK(scheduler.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.is_abnormal);
    CHECK(trace.abnormal.condition == AbnormalCondition::input_after_terminal);
    scheduler.finish_halt();
    CHECK(scheduler.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.abnormal.condition == AbnormalCondition::emergency_stop);
    scheduler.close();
}

void test_stale_command_is_rejected_across_gap()
{
    auto schema = decoded_schema();
    auto config = center_controller_config();
    // A tenth of a second is the longest interval this run will apply one
    // decoded command across.
    config.max_observation_interval_ns = 100'000'000;
    CenterOutController controller{};
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CHECK(controller.start(1'000, 50) == StreamStatus::ok);
    CenterOutControlTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    FrameStorage first_storage{};
    const std::array moving{1.0, 0.0};
    ActuatorCommand command{};
    command.valid_until_ns = 10'000;
    command.payload = decoded_frame(first_storage, moving, 1, 1, 0, 1'100);
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::observation);
    const auto after_first = controller.position();
    CHECK(after_first.x > 0.0);

    // A second sample a whole second later. Integrating it would move the
    // cursor by a second's worth of a velocity that was observed once -- the
    // shape a stale command actually takes here. The observation is not
    // refused; the interval before it is, and the segment restarts at dt zero.
    FrameStorage late_storage{};
    const auto before_stale = allocations.load(std::memory_order_relaxed);
    command.payload = decoded_frame(late_storage, moving, 1, 2, 1, 1'000'001'100);
    CHECK(controller.submit(command, 2'000) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before_stale);
    // One row of input, one queue entry -- the decision travels on the
    // observation it was taken about, exactly as a discontinuity's does. A
    // second entry would mean one row could cost two slots, and the frame
    // reserved one per row before it changed anything.
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::observation);
    CHECK(trace.has_abnormal);
    CHECK(trace.abnormal.condition == AbnormalCondition::input_stale);
    CHECK(trace.abnormal.response == AbnormalResponse::trial_aborted);
    CHECK(trace.has_aborted_trial);
    CHECK(trace.aborted_trial.outcome == TrialOutcome::aborted);
    // The trial's interval is the machine's own, from its TRIAL_START, not a
    // number this layer guessed at.
    CHECK(trace.aborted_trial.interval.start_ns == 50);
    CHECK(trace.aborted_trial.interval.end_ns == 1'000'000'150);
    CHECK(trace.restarted_after_discontinuity);
    // dt zero: nothing was integrated across the interval nobody observed.
    CHECK(trace.dt_ns == 0);
    CHECK(trace.position_after.x == after_first.x);
    CHECK(trace.position_after.y == after_first.y);
    controller.close();
}

void test_expired_command_never_reaches_task()
{
    auto schema = decoded_schema();
    CenterOutController controller{};
    CHECK(controller.prepare(schema, center_controller_config()) == StreamStatus::ok);
    CHECK(controller.start(1'000, 50) == StreamStatus::ok);
    CenterOutControlTrace trace{};
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);

    FrameStorage storage{};
    const std::array moving{1.0, 0.0};
    ActuatorCommand command{};
    command.generated_at_ns = 1'000;
    command.valid_until_ns = 2'000;
    command.payload = decoded_frame(storage, moving, 1, 1, 0, 1'100);

    // Submitted after its validity window. NativeActuator::submit enforces
    // expiry before entering the device-specific write, so the task is not even
    // asked -- which is the guarantee, not an optimisation.
    CHECK(controller.submit(command, 2'000) == StreamStatus::deadline_exceeded);
    CHECK(controller.position().x == 0.0);
    CHECK(controller.position().y == 0.0);
    // No trace either. A command that was never applied did not produce an
    // observation, and a record of one would be a record of something that did
    // not happen.
    CHECK(controller.try_pop_trace(trace) == StreamStatus::would_block);
    // And nothing was latched: an expired command is the runtime's deadline
    // being enforced, not the task deciding anything.
    CHECK(controller.abnormal_summary().observed == 0);
    CHECK(!controller.halted());

    // The expired command is not reissued to keep the trial moving. The next
    // observation the task sees is the next one that actually arrived, and the
    // interval in between contributed no motion at all.
    ActuatorCommand later{};
    later.generated_at_ns = 2'100;
    later.valid_until_ns = 10'000;
    FrameStorage next_storage{};
    later.payload = decoded_frame(next_storage, moving, 1, 2, 1, 1'200);
    CHECK(controller.submit(later, 2'200) == StreamStatus::ok);
    CHECK(controller.try_pop_trace(trace) == StreamStatus::ok);
    CHECK(trace.kind == CenterOutTraceKind::observation);
    // The interval runs from the last *accepted* observation -- the session
    // epoch -- to this one, so it spans the expired command's slot as well. The
    // expired command's own velocity is not what fills it: this one is, and
    // that is precisely the exposure ::CenterOutControllerConfig's
    // `max_observation_interval_ns` bounds when a caller declares a bound.
    CHECK(trace.dt_ns == 200);
    CHECK(trace.decoded.values[0] == 1.0);
    controller.close();
}

void test_adaptive_decoder_activates_only_after_trial_stop()
{
    auto feature_schema = adaptive_feature_schema();
    auto config = center_controller_config(256);
    config.linear_assistance = assistance::LinearAssistance{1.0};
    config.training_capture_capacity = 256;
    config.presentation_state_capacity = 256;
    ConstantVelocityDecoder initial{{0.0, 0.0}};
    ConstantVelocityDecoder candidate{{-1.0, 0.0}};
    AdaptiveCenterOutActuator actuator{};
    ShiftedClock clock{};
    actuator.bind_runtime_clock(clock);
    CHECK(actuator.prepare(feature_schema, initial, 1, config) == StreamStatus::ok);
    CHECK(actuator.prepare_candidate(candidate, 2) == StreamStatus::ok);
    PreparedDecoderCandidate prepared{&candidate, 2};
    CHECK(actuator.publish_candidate(prepared) == StreamStatus::ok);
    CHECK(actuator.prepare_candidate(candidate, 3) == StreamStatus::invalid_state);
    actuator.discard_pending_candidate();
    CHECK(actuator.active_decoder_version() == 1);
    CHECK(actuator.prepare_candidate(candidate, 2) == StreamStatus::ok);
    CHECK(actuator.publish_candidate(prepared) == StreamStatus::ok);
    CHECK(actuator.controller().start(1'000, 0) == StreamStatus::ok);
    CHECK(!actuator.controller().complete());
    CenterOutControlTrace trace{};
    CHECK(actuator.controller().try_pop_trace(trace) == StreamStatus::ok);

    SampleIndex sample_idx = 0;
    while (actuator.controller().completed_trial_count() == 0 && sample_idx < 100)
    {
        AdaptiveFrameStorage storage{};
        CHECK(actuator.consume(adaptive_feature_frame(
                  storage, sample_idx, 100'000'000ULL * (sample_idx + 1))) == StreamStatus::ok);
        ++sample_idx;
    }
    CHECK(actuator.controller().completed_trial_count() == 1);
    CHECK(actuator.active_decoder_version() == 2);
    CHECK(actuator.activated_decoder_count() == 1);

    AdaptiveFrameStorage storage{};
    CHECK(actuator.consume(adaptive_feature_frame(
              storage, sample_idx, 100'000'000ULL * (sample_idx + 1))) == StreamStatus::ok);
    bool saw_candidate_observation = false;
    while (actuator.controller().try_pop_trace(trace) == StreamStatus::ok)
    {
        if (trace.kind == CenterOutTraceKind::observation && trace.sample_idx == sample_idx)
        {
            CHECK(trace.decoder_version == 2);
            CHECK(trace.decoded_ready_ns >= ShiftedClock::epoch);
            CHECK(trace.decoded_ready_ns < ShiftedClock::epoch + 1'000'000'000);
            saw_candidate_observation = true;
        }
    }
    CHECK(saw_candidate_observation);

    AdaptiveFeatureObservation feature{};
    CenterOutTrainingLabel label{};
    CHECK(actuator.try_pop_feature(feature) == StreamStatus::ok);
    CHECK(actuator.controller().try_pop_training_label(label) == StreamStatus::ok);
    CHECK(feature.sample_idx == label.sample_idx);
    CHECK(feature.values.size() == 2);
    CHECK(feature.values[0] == 1.0);
    CHECK(feature.values[1] == 2.0);
    CHECK(actuator.dropped_feature_count() == 0);
    CHECK(actuator.controller().dropped_training_label_count() == 0);
    CenterOutPresentationState presentation{};
    CHECK(actuator.controller().try_pop_presentation_state(presentation) == StreamStatus::ok);
    CHECK(presentation.source_ordinal == 0);
    CHECK(presentation.time_ns == 99'999'000ULL);
    CHECK(presentation.cursor.x == 0.0);
    CHECK(presentation.cursor.y == 0.0);
    CHECK(actuator.controller().dropped_presentation_state_count() == 0);
    ++sample_idx;
    while (!actuator.controller().complete() && sample_idx < 200)
    {
        AdaptiveFrameStorage remaining_storage{};
        CHECK(actuator.consume(adaptive_feature_frame(remaining_storage, sample_idx,
                                                      100'000'000ULL * (sample_idx + 1))) ==
              StreamStatus::ok);
        ++sample_idx;
    }
    CHECK(actuator.controller().complete());
    CHECK(actuator.controller().reset() == StreamStatus::ok);
    CHECK(!actuator.controller().complete());
    actuator.controller().close();
}

void test_adaptive_decoder_accepts_schema_sized_feature_capture()
{
    constexpr auto n_features = std::uint32_t{768};
    auto feature_schema = adaptive_feature_schema(n_features);
    auto config = center_controller_config(8);
    config.linear_assistance = assistance::LinearAssistance{1.0};
    config.training_capture_capacity = 4;
    config.presentation_state_capacity = 4;
    ConstantVelocityDecoder decoder{{0.0, 0.0}};
    AdaptiveCenterOutActuator actuator{};
    CHECK(actuator.prepare(feature_schema, decoder, 1, config) == StreamStatus::ok);
    CHECK(actuator.controller().start(1'000, 0) == StreamStatus::ok);

    std::vector<double> values(n_features);
    for (std::size_t feature = 0; feature < values.size(); ++feature)
        values[feature] = static_cast<double>(feature);
    std::array<SignalBlockHeader, 1> blocks{};
    std::vector<std::byte> payload(values.size() * sizeof(double));
    std::memcpy(payload.data(), values.data(), payload.size());
    blocks[0] = SignalBlockHeader{.sample_idx_start = 0,
                                  .observation_time_start_ns = 100'000'000,
                                  .payload_offset = 0,
                                  .payload_byte_count = payload.size(),
                                  .signal_id = 22,
                                  .n_samples = 1};
    const FrameView frame{.header = FrameHeader{.session_id = 3,
                                                .sequence = 0,
                                                .schema_id = 21,
                                                .source_clock_domain = 7,
                                                .signal_block_count = 1},
                          .blocks = blocks,
                          .payload = payload};
    const auto before_consume = allocations.load(std::memory_order_relaxed);
    CHECK(actuator.consume(frame) == StreamStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == before_consume);

    AdaptiveFeatureObservation observation{};
    CHECK(actuator.try_pop_feature(observation) == StreamStatus::ok);
    CHECK(observation.values.size() == n_features);
    CHECK(observation.values.front() == 0.0);
    CHECK(observation.values.back() == 767.0);
    actuator.controller().close();
}

} // namespace

int main()
{
    test_center_out_terminal_consumer();
    test_center_out_cursor_saturates_without_boundary_windup();
    test_center_out_full_assistance_is_computer_controlled();
    test_center_out_assistance_schedule_switches_after_trial_boundary();
    test_center_out_runtime_terminal_discontinuity_delivery();
    test_webgrid_headless_order_and_bounds();
    test_speech_explicit_time_and_prepared_payloads();
    test_oversized_input_is_rejected_without_effect();
    test_stale_observation_costs_one_queue_entry();
    test_aborted_run_rejects_later_input();
    test_center_out_trace_is_enough_to_re_execute();
    test_webgrid_trace_is_enough_to_re_execute();
    test_speech_trace_is_enough_to_re_execute();
    test_pointer_gap_invalidates_crossed_acquisition();
    test_stale_or_repeated_selection_is_rejected();
    test_input_after_halt_is_rejected_and_recorded_once();
    test_semantic_time_jump_is_not_fault();
    test_stale_command_is_rejected_across_gap();
    test_expired_command_never_reaches_task();
    test_adaptive_decoder_activates_only_after_trial_stop();
    test_adaptive_decoder_accepts_schema_sized_feature_capture();
    return failures == 0 ? 0 : 1;
}
