// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#include "allocation_tracker.h"
#include "check_returns.h"
#include "interval_runner.h"
#include "ssvep_controller.h"
#include <array>
#include <cstring>
#include <limits>
#include <neurale/streaming/linear_processor_chain.h>

using namespace neurale::streaming;
using namespace neurale::pipeline;
using namespace neurale::execution;

StreamSchema schema()
{
    const SignalSchema signal{1,
                              SignalDType::float64,
                              2,
                              1,
                              1,
                              {100000000, 1},
                              1,
                              SignalLayout::sample_major,
                              DeviceTickTracking::unavailable,
                              PhysicalUnit::unspecified,
                              0,
                              0,
                              0,
                              SignalKind::feature,
                              1,
                              ObservationTiming::regular};
    const FeatureSetDescriptor feature{.id = 1,
                                       .feature_names = {"a", "b"},
                                       .unit_ids = {1, 1},
                                       .source_stream_id = 1,
                                       .source_stream = "test",
                                       .algorithm_name = "test",
                                       .algorithm_version = "1",
                                       .window_length_ns = 20,
                                       .shift_ns = 10};
    return {1, std::array{signal}, std::array{feature},
            std::array{UnitDescriptor{1, "1", "dimensionless"}}};
}

// Algorithm-independent test processor; no experiment and no LDA dependency.
class Classifier final : public NativeFrameProcessor
{
  public:
    bool corrupt_identity{};
    std::uint32_t output_channels{1};
    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        if (context.input_schema.signals()[0].observation_timing != ObservationTiming::irregular ||
            context.input_schema.signals()[0].fs.numerator != 0 ||
            context.input_schema.feature_sets().descriptors()[0].shift_ns != 0)
            throw std::invalid_argument("expected an interval observation");
        const SignalSchema output{99,
                                  SignalDType::float64,
                                  output_channels,
                                  1,
                                  1,
                                  {0, 1},
                                  1,
                                  SignalLayout::sample_major,
                                  DeviceTickTracking::unavailable,
                                  PhysicalUnit::dimensionless,
                                  0,
                                  0,
                                  0,
                                  SignalKind::event};
        return {context.input_schema.clone(),
                StreamSchema{99, std::array{output}},
                1,
                0,
                false,
                {0, 1}};
    }
    StreamStatus process(FrameBorrow& input, FrameEmitter& emitter) noexcept override
    {
        FrameBorrow output;
        auto status = emitter.try_acquire(output);
        if (status != StreamStatus::ok)
            return status;
        double value;
        std::memcpy(&value, input.payload().data(), sizeof(value));
        const double values[]{value + 100, value + 200};
        const auto bytes = output_channels * sizeof(double);
        output.header() = input.header();
        if (corrupt_identity)
            ++output.header().sequence;
        output.header().schema_id = 99;
        output.block_storage()[0] = input.blocks()[0];
        output.block_storage()[0].signal_id = 99;
        output.block_storage()[0].payload_byte_count = bytes;
        std::memcpy(output.payload_storage().data(), values, bytes);
        status = output.set_used_sizes(1, bytes);
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
};

StreamStatus feed(IntervalRunner& pipeline, FramePool& pool, std::uint64_t center, double value,
                  std::uint64_t received = 0)
{
    FrameLease lease;
    auto status = pool.try_acquire(lease);
    if (status != StreamStatus::ok)
        return status;
    auto& frame = lease.frame();
    frame.header() = {
        .session_id = 7, .host_received_ns = received ? received : center + 10, .schema_id = 1};
    frame.block_storage()[0] = {.sample_idx_start = center / 10,
                                .observation_time_start_ns = center,
                                .payload_byte_count = 16,
                                .signal_id = 1,
                                .n_samples = 1};
    const double values[]{value, 2 * value};
    std::memcpy(frame.payload_storage().data(), values, sizeof(values));
    status = frame.set_used_sizes(1, sizeof(values));
    return status == StreamStatus::ok ? pipeline.consume(frame.view()) : status;
}

int main()
{
    auto input = schema();
    IntervalRunner pipeline(input, 100, 4);
    Classifier classifier;
    pipeline.publish_decoder(classifier);
    FramePool pool(1, 16, 1);
    CHECK(pipeline.schedule({1, 100, 200}) == StreamStatus::ok);
    CHECK(feed(pipeline, pool, 100, 1000) == StreamStatus::ok); // crosses start
    CHECK(feed(pipeline, pool, 110, 2) == StreamStatus::ok);
    CHECK(feed(pipeline, pool, 120, 4) == StreamStatus::ok);
    CHECK(feed(pipeline, pool, 120, 1000) == StreamStatus::ok);      // duplicate
    CHECK(feed(pipeline, pool, 130, 1000, 201) == StreamStatus::ok); // late arrival
    CHECK(feed(pipeline, pool, 190, 6) == StreamStatus::ok);         // exact end
    CHECK(feed(pipeline, pool, 191, 1000) == StreamStatus::ok);      // crosses end
    IntervalResult result;
    CHECK(pipeline.advance_time(199) == StreamStatus::ok);
    CHECK(pipeline.pop_result(result) == StreamStatus::would_block);
    neurale::benchmark::reset_allocation_count();
    {
        neurale::benchmark::AllocationScope scope;
        // No new frame: the clock alone must complete aggregation and decoding.
        CHECK(pipeline.advance_time(200) == StreamStatus::ok);
        scope.stop();
        CHECK(scope.count() == 0);
    }
    CHECK(pipeline.pop_result(result) == StreamStatus::ok);
    double prediction{};
    CHECK(result.decoded);
    std::memcpy(&prediction, result.decoded.view().payload.data(), sizeof(prediction));
    CHECK(result.interval.key == 1 && result.center_ns == 150 && result.windows == 3 &&
          prediction == 104);
    CHECK(pipeline.features(0)[0] == 4 && pipeline.features(0)[1] == 8);
    CHECK(pipeline.advance_time(210) == StreamStatus::ok);
    CHECK(pipeline.pop_result(result) == StreamStatus::would_block); // once only
    CHECK(pipeline.schedule({2, 200, 300}) == StreamStatus::ok);
    CHECK(feed(pipeline, pool, 190, 1000) == StreamStatus::ok); // previous interval
    CHECK(pipeline.advance_time(300) == StreamStatus::ok);
    CHECK(pipeline.pop_result(result) == StreamStatus::ok && result.windows == 0 &&
          !result.decoded);
    CHECK(pipeline.schedule({3, 300, 400}) == StreamStatus::ok);
    CHECK(feed(pipeline, pool, 320, 8) == StreamStatus::ok);
    CHECK(pipeline.advance_time(400) == StreamStatus::ok);
    CHECK(pipeline.pop_result(result) == StreamStatus::ok && result.windows == 1 && result.decoded);
    CHECK(pipeline.features(2)[0] == 8);
    CHECK(pipeline.schedule({4, 400, 500}) == StreamStatus::ok);
    pipeline.cancel();
    CHECK(pipeline.advance_time(500) == StreamStatus::stopped);
    CHECK(pipeline.pop_result(result) == StreamStatus::would_block);

    IntervalRunner invalid(input, 100, 1);
    CHECK(invalid.schedule({1, 100, 201}) == StreamStatus::ok);
    CHECK(invalid.advance_time(100) == StreamStatus::invalid_state);
    IntervalRunner discontinuous(input, 100, 1);
    CHECK(discontinuous.handle_discontinuity({}) == StreamStatus::invalid_frame);
    CHECK(discontinuous.advance_time(200) == StreamStatus::invalid_frame);
    IntervalRunner nonfinite(input, 100, 1);
    CHECK(feed(nonfinite, pool, 110, std::numeric_limits<double>::infinity()) == StreamStatus::ok);
    CHECK(nonfinite.advance_time(120) == StreamStatus::invalid_frame);
    IntervalRunner mismatched(input, 100, 1);
    Classifier wrong_trial;
    wrong_trial.corrupt_identity = true;
    mismatched.publish_decoder(wrong_trial);
    CHECK(mismatched.schedule({1, 100, 200}) == StreamStatus::ok);
    CHECK(feed(mismatched, pool, 110, 2) == StreamStatus::ok);
    CHECK(mismatched.advance_time(200) == StreamStatus::invalid_frame);
    CHECK(mismatched.pop_result(result) == StreamStatus::would_block);

    // A second interval consumer needs neither SSVEP trial ordinals nor a decoder.
    IntervalRunner capture(input, 100, 1);
    CHECK(capture.schedule({7, 500, 600}) == StreamStatus::ok);
    CHECK(feed(capture, pool, 550, 3) == StreamStatus::ok);
    CHECK(capture.advance_time(600) == StreamStatus::ok);
    CHECK(capture.pop_result(result) == StreamStatus::ok && result.interval.key == 7 &&
          result.windows == 1 && !result.decoded);
    CHECK(capture.features(0)[0] == 3);

    // Decoder output stays a typed frame; it need not be a scalar target ID.
    IntervalRunner vector_decoding(input, 100, 1);
    Classifier vector_classifier;
    vector_classifier.output_channels = 2;
    vector_decoding.publish_decoder(vector_classifier);
    CHECK(vector_decoding.schedule({13, 600, 700}) == StreamStatus::ok);
    CHECK(feed(vector_decoding, pool, 650, 5) == StreamStatus::ok);
    CHECK(vector_decoding.advance_time(700) == StreamStatus::ok);
    CHECK(vector_decoding.pop_result(result) == StreamStatus::ok && result.decoded);
    const auto decoded = result.decoded.view();
    CHECK(decoded.blocks[0].payload_byte_count == 2 * sizeof(double));
    double decoded_values[2]{};
    std::memcpy(decoded_values, decoded.payload.data(), sizeof(decoded_values));
    CHECK(decoded_values[0] == 105 && decoded_values[1] == 205);
    CHECK(result.decoded.reset() == StreamStatus::ok);

    // A compiled chain transfers its own frame; the queued result must not
    // borrow storage from that chain after its lifetime ends.
    IntervalRunner chained(input, 100, 1);
    {
        Classifier stage;
        std::array<NativeFrameProcessor*, 1> stages{&stage};
        LinearProcessorChain chain{stages};
        chained.publish_decoder(chain);
        CHECK(chained.schedule({21, 800, 900}) == StreamStatus::ok);
        CHECK(feed(chained, pool, 850, 7) == StreamStatus::ok);
        CHECK(chained.advance_time(900) == StreamStatus::ok);
        CHECK(chained.pop_result(result) == StreamStatus::ok && result.decoded);
    }
    std::memcpy(&prediction, result.decoded.view().payload.data(), sizeof(prediction));
    CHECK(prediction == 107);
    CHECK(result.decoded.reset() == StreamStatus::ok);

    // A cached feature from before the session epoch must not wrap its trace timestamp.
    IntervalRunner trace_pipeline(input, 100, 5);
    neurale::experiments::ssvep::SSVEPConfig task{};
    task.n_targets = 2;
    task.targets[0] = {1, 8};
    task.targets[1] = {2, 10};
    task.stimulus_id = 5;
    task.n_trials = 5;
    task.seed = 42;
    task.cue_duration_ns = 10;
    task.stimulation_duration_ns = 100;
    task.decision_timeout_ns = 10;
    task.feedback_duration_ns = 5;
    task.inter_trial_ns = 5;
    SSVEPController controller(input, task, 4, trace_pipeline);
    controller.set_host_epoch(1000);
    ExperimentTraceSink sink;
    CHECK(feed(trace_pipeline, pool, 900, 1) == StreamStatus::ok);
    CHECK(trace_pipeline.advance_time(900) == StreamStatus::ok);
    CHECK(controller.drain(sink, 10) == 2);
    CHECK(sink.encoded() == 0);
    CHECK(feed(trace_pipeline, pool, 1200, 2) == StreamStatus::ok);
    CHECK(trace_pipeline.advance_time(1200) == StreamStatus::ok);
    CHECK(controller.drain(sink, 10) == 2);
    CHECK(sink.encoded() == 2 && sink.last_time_ns() == 200);
    return 0;
}
