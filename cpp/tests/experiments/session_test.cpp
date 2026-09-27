/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// \file
/// The session's bring-up and shutdown order, and the recording bridge.
///
/// Every assertion about "it was recorded" is made against the committed spool
/// prefix rather than against a counter, because what a reader promotes is the
/// only definition of recorded that survives a crash.

#include "center_out_trace_writer.h"
#include "session.h"
#include "speech_trace_writer.h"
#include "webgrid_trace_writer.h"

#include "check_counts.h"
#include "record_payloads.h"
#include "recording/recorder_test_support.h"
#include "spool_scanner.h"

#include <neurale/streaming/actuator.h>
#include <neurale/streaming/clock.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/runtime.h>
#include <neurale/streaming/safety.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/source.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

namespace
{
using namespace neurale::experiments;
using namespace neurale::execution;
using namespace neurale::recording;
using namespace neurale::recording::test;
using neurale::streaming::StreamStatus;

int failures = 0;

// --- reading the spool back -------------------------------------------------

/// One committed control record, with its body materialized.
struct ReadControl
{
    ControlPayloadFields fields{};
    std::string body{};
};

[[nodiscard]] std::vector<ReadControl> read_control(SpoolFile& file, SpoolScanReport& report)
{
    std::array<std::byte, 4096> scratch{};
    SpoolScanner scanner;
    report = scanner.scan(file, scratch);

    std::vector<ReadControl> out;
    SpoolRecordCursor cursor(file, report);
    SpoolRecordView view{};
    while (cursor.next(view))
    {
        if (static_cast<RecordKind>(view.kind) != RecordKind::control)
        {
            continue;
        }
        std::vector<std::byte> payload(view.payload_bytes);
        if (!cursor.read_payload(view, payload))
        {
            break;
        }
        ReadControl record{};
        if (!decode_control_payload(payload, record.fields))
        {
            break;
        }
        const auto* bytes = reinterpret_cast<const char*>(payload.data());
        record.body.assign(bytes + control_payload::kBody, bytes + payload.size());
        out.push_back(std::move(record));
    }
    return out;
}

/// The `name` a uniform body carries, or an empty view for another shape.
[[nodiscard]] std::string_view record_name(const ReadControl& record) noexcept
{
    static constexpr std::string_view kPrefix{"\"name\":\""};
    const auto start = record.body.find(kPrefix);
    if (start == std::string::npos)
    {
        return {};
    }
    const auto first = start + kPrefix.size();
    const auto last = record.body.find('"', first);
    if (last == std::string::npos)
    {
        return {};
    }
    return std::string_view{record.body}.substr(first, last - first);
}

[[nodiscard]] std::string_view record_field(const ReadControl& record,
                                            std::string_view prefix) noexcept
{
    const auto start = record.body.find(prefix);
    if (start == std::string::npos)
    {
        return {};
    }
    const auto first = start + prefix.size();
    const auto last = record.body.find('"', first);
    if (last == std::string::npos)
    {
        return {};
    }
    return std::string_view{record.body}.substr(first, last - first);
}

/// The `code` of a `faults`-shaped record.
///
/// Not the same field as record_name(): the uniform body carries `name`, and
/// the fault body carries `code` and `stage`. An abnormal condition is written
/// as a fault, with the contract's own condition name as the code.
[[nodiscard]] std::string_view record_code(const ReadControl& record) noexcept
{
    return record_field(record, "\"code\":\"");
}

[[nodiscard]] bool body_contains(const ReadControl& record, std::string_view needle) noexcept
{
    return record.body.find(needle) != std::string::npos;
}

/// Whether the record's *nested* document holds `"key":value`.
///
/// The nested `text`, `label`, and `outcome` documents travel as escaped
/// strings inside the body, so their quotes are backslash-escaped by the time
/// they reach the spool. Spelling the escaped form at every call site is how a
/// test ends up asserting a field it never actually looked for.
[[nodiscard]] bool nested_contains(const ReadControl& record, std::string_view key,
                                   std::string_view value)
{
    std::string needle;
    needle.append("\\\"").append(key).append("\\\":").append(value);
    return record.body.find(needle) != std::string::npos;
}

/// Read one unsigned field out of a record's nested document.
///
/// Tests that report on a presentation have to name the request the run
/// actually emitted, and the only place a test can learn its identity is the
/// recording -- which is also the only place a reader would learn it. Deriving
/// it a second time in the test would be the test agreeing with itself.
[[nodiscard]] bool nested_u64(const ReadControl& record, std::string_view key, std::uint64_t& value)
{
    std::string needle;
    needle.append("\\\"").append(key).append("\\\":");
    const auto at = record.body.find(needle);
    if (at == std::string::npos)
    {
        return false;
    }
    auto cursor = at + needle.size();
    if (cursor >= record.body.size() || record.body[cursor] < '0' || record.body[cursor] > '9')
    {
        return false;
    }
    std::uint64_t parsed = 0;
    while (cursor < record.body.size() && record.body[cursor] >= '0' && record.body[cursor] <= '9')
    {
        parsed = parsed * 10 + static_cast<std::uint64_t>(record.body[cursor] - '0');
        ++cursor;
    }
    value = parsed;
    return true;
}

[[nodiscard]] std::size_t count_coded(const std::vector<ReadControl>& records,
                                      std::string_view code) noexcept
{
    std::size_t seen = 0;
    for (const auto& record : records)
    {
        if (record_code(record) == code)
        {
            ++seen;
        }
    }
    return seen;
}

[[nodiscard]] std::size_t count_coded_at(const std::vector<ReadControl>& records,
                                         std::string_view code, ExperimentTimeNs time_ns) noexcept
{
    std::size_t seen = 0;
    for (const auto& record : records)
    {
        if (record_code(record) == code && record.fields.time_ns == time_ns)
        {
            ++seen;
        }
    }
    return seen;
}

// --- the pieces a session is given -----------------------------------------

/// A recorder and everything it borrows, kept together so a test can bring one
/// up in a line and still own every object the session only points at.
struct RecorderFixture
{
    PlanFixture plan_fixture{};
    RecorderMemorySpoolFile spool{};
    CountingUnixClock clock{};
    NativeRecorderCore recorder{};
    NativeRecordingPlan plan{};

    RecorderFixture()
    {
        plan_fixture.control_queue_capacity = 256;
        spool.reserve_bytes(1 << 20);
        plan = plan_fixture.plan();
        // The shared fixture's 128-byte control bound is sized for the recorder
        // core's own three-byte test bodies. An experiment record carries a
        // nested document, and a body over the plan's bound is a plan violation
        // that faults the recorder -- which is the recorder correctly refusing
        // a plan the caller did not size, not a bridge failure.
        plan.max_control_payload_bytes = 2048;
    }

    [[nodiscard]] RecorderAttachment attachment() noexcept
    {
        return RecorderAttachment{
            .recorder = &recorder, .plan = &plan, .spool = &spool, .clock = &clock};
    }
};

[[nodiscard]] ExperimentSessionConfig manual_session() noexcept
{
    ExperimentSessionConfig config{};
    // No bridge thread: a paradigm the test steps itself is pumped by the test,
    // which is what makes every ordering assertion below a fact rather than a
    // race the scheduler happened to win.
    config.bridge_poll_nanos = 0;
    config.drain_budget = 64;
    config.control_clock_domain = 5;
    config.max_control_body_bytes = 512;
    return config;
}

/// A session that owns its drain, which is what an attached runtime requires.
[[nodiscard]] ExperimentSessionConfig bridged_session() noexcept
{
    auto config = manual_session();
    // Short, because these tests wait for the bridge rather than for a clock:
    // an attached runtime produces on threads the test is not ordered against,
    // so the session -- not the test -- has to be the thing that drains.
    config.bridge_poll_nanos = 50'000;
    return config;
}

/// The paradigm identities these tests record under.
constexpr ParadigmId kWebGridParadigm = 71;
constexpr ParadigmId kSpeechParadigm = 63;
constexpr ParadigmId kCenterOutParadigm = 41;
/// The host instant the Center-Out sessions anchor experiment time zero at.
constexpr neurale::streaming::HostTimeNs kCenterOutHostEpoch = 1'000;

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

CenterOutControllerConfig center_out_config(std::size_t capacity = 64) noexcept
{
    CenterOutControllerConfig config{};
    config.decoded_signal_id = 9;
    config.paradigm = kCenterOutParadigm;
    center_out::RadialLayoutRequest request{};
    request.radius = 0.5;
    request.count = 2;
    request.center_id = 1;
    request.ids[0] = 2;
    request.ids[1] = 3;
    request.spokes[0] = 0;
    request.spokes[1] = 4;
    config.task.geometry_unit = center_out::GeometryUnit::normalized;
    CHECK(center_out::build_radial_layout(request, config.task.layout) == ContractStatus::ok);
    config.task.acceptance = center_out::AcceptanceRegion{0.1, 0.1};
    config.task.cursor = center_out::CursorGeometry{0.0};
    config.task.movement_timeout = center_out::PhaseDurations{2'000'000'000, 2'000'000'000};
    config.task.hold_ns = 0;
    config.task.reward_dwell = center_out::PhaseDurations{0, 0};
    config.task.punish_dwell = center_out::PhaseDurations{0, 0};
    config.task.selection = center_out::TargetSelectionPolicy::repeat_until_success;
    config.task.seed = 17;
    config.task.trial_limit = 2;
    config.guidance = center_out::CenterOutGuidanceConfig{center_out::GeometryUnit::normalized, 1.0,
                                                          4.0, 4.0, 0.05};
    config.velocity_space.id = 11;
    config.velocity_space.dim = 2;
    config.velocity_space.frame = CommandFrame::workspace_2d;
    config.velocity_space.axes[0] = CommandAxis{CommandAxisName::x, CommandUnit::normalized};
    config.velocity_space.axes[1] = CommandAxis{CommandAxisName::y, CommandUnit::normalized};
    config.linear_assistance = assistance::LinearAssistance{0.25};
    config.initial_position = {0.0, 0.0};
    config.cursor_min = {-1.0, -1.0};
    config.cursor_max = {1.0, 1.0};
    config.trace_capacity = capacity;
    config.assistance_method = assistance::AssistanceMethod::linear_blend;
    return config;
}

neurale::streaming::StreamSchema decoded_schema()
{
    using namespace neurale::streaming;
    const SignalSchema signal{9,
                              SignalDType::float64,
                              2,
                              2,
                              4,
                              RationalRate{10, 1},
                              7,
                              SignalLayout::sample_major,
                              DeviceTickTracking::unavailable,
                              PhysicalUnit::dimensionless,
                              5};
    const std::array signals{signal};
    return StreamSchema{13, signals};
}

struct FrameStorage
{
    std::array<neurale::streaming::SignalBlockHeader, 1> blocks{};
    alignas(double) std::array<std::byte, 4 * 2 * sizeof(double)> payload{};
};

neurale::streaming::FrameView decoded_frame(FrameStorage& storage, std::span<const double> values,
                                            std::uint32_t observations, std::uint64_t sequence,
                                            neurale::streaming::SampleIndex sample_idx,
                                            neurale::streaming::HostTimeNs time_ns) noexcept
{
    using namespace neurale::streaming;
    std::memcpy(storage.payload.data(), values.data(), values.size_bytes());
    storage.blocks[0] = SignalBlockHeader{
        .sample_idx_start = sample_idx,
        .observation_time_start_ns = time_ns,
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

/// Hand one decoded frame to a Center-Out controller as the runtime would.
[[nodiscard]] StreamStatus feed(CenterOutController& controller,
                                const neurale::streaming::FrameView& frame) noexcept
{
    neurale::streaming::ActuatorCommand command{};
    command.valid_until_ns = (std::numeric_limits<std::uint64_t>::max)();
    command.payload = frame;
    return controller.submit(command, 0);
}

/// Count the committed records carrying one record name.
[[nodiscard]] std::size_t count_named(const std::vector<ReadControl>& records,
                                      std::string_view name) noexcept
{
    std::size_t seen = 0;
    for (const auto& record : records)
    {
        if (record_name(record) == name)
        {
            ++seen;
        }
    }
    return seen;
}

/// Count the committed records carrying one record name at one instant.
[[nodiscard]] std::size_t count_at(const std::vector<ReadControl>& records, std::string_view name,
                                   ExperimentTimeNs time_ns) noexcept
{
    std::size_t seen = 0;
    for (const auto& record : records)
    {
        if (record_name(record) == name && record.fields.time_ns == time_ns)
        {
            ++seen;
        }
    }
    return seen;
}

/// Step a started WebGrid session through both of its targets.
///
/// It does not start the controller: the session already did, as bring-up step
/// 6. A caller that started it here would be starting it twice.
void run_webgrid(WebGridHeadlessController& controller, const webgrid::WebGridConfig& config,
                 ExperimentSession& session)
{
    ExperimentTimeNs time_ns = 1'000;
    for (int step = 0; step < 2; ++step)
    {
        const auto snapshot = controller.snapshot();
        const webgrid::PointerPosition pointer{snapshot.active_target == 1 ? 0.5 : 1.5, 0.5};
        SelectionEvent selection{};
        CHECK(webgrid::make_selection_event(config, pointer, snapshot.active_target, snapshot.trial,
                                            kWebGridParadigm, time_ns + 100,
                                            static_cast<SequenceOrdinal>(step),
                                            selection) == ContractStatus::ok);
        CHECK(session.note_step(controller.process(time_ns + 100, pointer, selection),
                                time_ns + 100) == StreamStatus::ok);
        static_cast<void>(session.pump());
        time_ns += 200;
    }
}

// --- an attached runtime ----------------------------------------------------

/// A source that publishes exactly *count* one-observation frames and ends.
///
/// One observation per frame is what makes the trace arithmetic below exact: a
/// frame costs a controller exactly one trace slot, so a queue of one slot is
/// either kept up with or overflowed, with nothing in between to explain a
/// result away.
class CountedDecodedSource final : public neurale::streaming::NativeFrameSource
{
  public:
    explicit CountedDecodedSource(std::size_t count) noexcept : remaining_(count) {}

    StreamStatus read(neurale::streaming::MutableFrame& frame) noexcept override
    {
        using namespace neurale::streaming;
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        if (remaining_ == 0)
        {
            return StreamStatus::end_of_stream;
        }
        const std::array values{0.0, 0.0};
        std::memcpy(frame.payload_storage().data(), values.data(), sizeof(values));
        frame.header() = FrameHeader{.session_id = 3,
                                     .sequence = sequence_,
                                     .host_received_ns = 1'000 + sequence_ * 100'000'000ULL,
                                     .schema_id = 13,
                                     .source_clock_domain = 7,
                                     .signal_block_count = 1};
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = sequence_,
            .observation_time_start_ns = 1'000 + sequence_ * 100'000'000ULL,
            .payload_offset = 0,
            .payload_byte_count = sizeof(values),
            .signal_id = 9,
            .n_samples = 1,
        };
        ++sequence_;
        --remaining_;
        return frame.set_used_sizes(1, sizeof(values));
    }

    void cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }

    StreamStatus reset() noexcept override
    {
        cancelled_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

  private:
    std::size_t remaining_;
    std::uint64_t sequence_{};
    std::atomic<bool> cancelled_{};
};

/// A source that publishes one frame and then holds the worker until the run
/// is cancelled.
///
/// For tests whose subject is what a *running* runtime does. Sizing a counted
/// source to "surely enough frames" does not make a runtime still running when
/// the assertion runs: the frames may all have been consumed, or the bridge may
/// have been outrun and the run ended itself under the loss policy, and either
/// way the test would be asserting about a runtime that had already stopped.
/// This one cannot end on its own, so there is nothing to be lucky about.
class HeldDecodedSource final : public neurale::streaming::NativeFrameSource
{
  public:
    StreamStatus read(neurale::streaming::MutableFrame& frame) noexcept override
    {
        using namespace neurale::streaming;
        if (published_)
        {
            std::unique_lock lock(mutex_);
            released_.wait(lock, [this] { return cancelled_.load(std::memory_order_acquire); });
            return StreamStatus::stopped;
        }
        if (cancelled_.load(std::memory_order_acquire))
        {
            return StreamStatus::stopped;
        }
        const std::array values{0.0, 0.0};
        std::memcpy(frame.payload_storage().data(), values.data(), sizeof(values));
        frame.header() = FrameHeader{.session_id = 3,
                                     .sequence = 0,
                                     .host_received_ns = 1'000,
                                     .schema_id = 13,
                                     .source_clock_domain = 7,
                                     .signal_block_count = 1};
        frame.block_storage()[0] = SignalBlockHeader{
            .sample_idx_start = 0,
            .observation_time_start_ns = 1'000,
            .payload_offset = 0,
            .payload_byte_count = sizeof(values),
            .signal_id = 9,
            .n_samples = 1,
        };
        published_ = true;
        return frame.set_used_sizes(1, sizeof(values));
    }

    void cancel() noexcept override
    {
        {
            const std::lock_guard lock(mutex_);
            cancelled_.store(true, std::memory_order_release);
        }
        released_.notify_all();
    }

    StreamStatus reset() noexcept override
    {
        cancelled_.store(false, std::memory_order_release);
        published_ = false;
        return StreamStatus::ok;
    }

  private:
    std::mutex mutex_{};
    std::condition_variable released_{};
    std::atomic<bool> cancelled_{};
    bool published_{};
};

class ForwardDecodedProcessor final : public neurale::streaming::NativeFrameProcessor
{
  public:
    neurale::streaming::PreparedProcessorContract
    prepare(const neurale::streaming::ProcessorPrepareContext& context) override
    {
        return {context.input_schema.clone(),
                context.input_schema.clone(),
                1,
                0,
                true,
                neurale::streaming::ProcessorResourceBounds{.frame_pool_leases = 1}};
    }

    StreamStatus process(neurale::streaming::FrameBorrow&,
                         neurale::streaming::FrameEmitter& emitter) noexcept override
    {
        return emitter.publish_input();
    }

    StreamStatus handle_discontinuity(const neurale::streaming::Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }

    StreamStatus flush(neurale::streaming::FrameEmitter&) noexcept override
    {
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }
};

/// A safety controller that remembers whether it is currently released.
///
/// It is the only way to see the thing that matters about an armed runtime that
/// never started: `arm()` releases safety and leaves the runtime in `prepared`,
/// so the runtime's own state says nothing about whether an actuator path is
/// still live.
class RecordingSafetyController final : public neurale::streaming::SafetyController
{
  public:
    StreamStatus inhibit(neurale::streaming::SafetyReason) noexcept override
    {
        released_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    StreamStatus release() noexcept override
    {
        released_.store(true, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] bool released() const noexcept
    {
        return released_.load(std::memory_order_acquire);
    }

  private:
    std::atomic<bool> released_{};
};

neurale::streaming::RealtimeConfig runtime_config() noexcept
{
    neurale::streaming::RealtimeConfig config{};
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

/// A paradigm whose configuration is valid and whose execution will not start.
///
/// It stands in for the real thing -- a schedule whose sampler version only
/// fails once the sampler is actually run -- because what is under test is what
/// the session does with the failure, not which paradigm produced it.
class UnstartableSource final : public ExperimentTraceSource
{
  public:
    [[nodiscard]] ContractStatus validate() const noexcept override
    {
        return ContractStatus::ok;
    }
    [[nodiscard]] StreamStatus start_execution(ExperimentTimeNs) noexcept override
    {
        return StreamStatus::realtime_configuration_failed;
    }
    void stop_execution(bool) noexcept override
    {
        ++stops_;
    }
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept override
    {
        return 0;
    }
    [[nodiscard]] SessionMetadata metadata() const noexcept override
    {
        SessionMetadata metadata{};
        metadata.experiment_type = "unstartable";
        return metadata;
    }
    void write_configuration(ExperimentTraceSink&, ExperimentTimeNs) noexcept override {}
    [[nodiscard]] std::size_t drain(ExperimentTraceSink&, std::size_t) noexcept override
    {
        return 0;
    }
    void write_summary(ExperimentTraceSink&, ExperimentTimeNs) noexcept override {}
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept override
    {
        return {};
    }
    void halt(ExperimentTimeNs, AbnormalCondition) noexcept override
    {
        ++halts_;
    }

    [[nodiscard]] std::size_t halts() const noexcept
    {
        return halts_;
    }
    [[nodiscard]] std::size_t stops() const noexcept
    {
        return stops_;
    }

  private:
    std::size_t stops_{};
    std::size_t halts_{};
};

/// A spool that cannot honor the recording plan's durability policy.
/// The readiness gate must reject it before committing a superblock or
/// starting the paradigm.
class UnsupportedDurabilitySpool final : public SpoolFile
{
  public:
    SpoolIoResult append(std::span<const std::byte> bytes) noexcept override
    {
        data_.insert(data_.end(), bytes.begin(), bytes.end());
        return SpoolIoResult{
            .status = SpoolIoStatus::ok, .transferred = bytes.size(), .platform_error = 0};
    }

    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override
    {
        if (offset >= data_.size())
        {
            return SpoolIoResult{
                .status = SpoolIoStatus::incomplete, .transferred = 0, .platform_error = 0};
        }
        const auto take =
            std::min<std::size_t>(out.size(), data_.size() - static_cast<std::size_t>(offset));
        std::copy_n(data_.begin() + static_cast<std::ptrdiff_t>(offset), take, out.begin());
        return SpoolIoResult{.status =
                                 take == out.size() ? SpoolIoStatus::ok : SpoolIoStatus::incomplete,
                             .transferred = take,
                             .platform_error = 0};
    }

    SpoolIoResult sync() noexcept override
    {
        return SpoolIoResult{};
    }

    SpoolIoResult truncate(std::uint64_t bytes) noexcept override
    {
        if (bytes < data_.size())
        {
            data_.resize(static_cast<std::size_t>(bytes));
        }
        return SpoolIoResult{};
    }

    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        return data_.size();
    }

    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return true;
    }

    [[nodiscard]] bool supports_durability_policy(DurabilityPolicy) const noexcept override
    {
        return false;
    }

  private:
    std::vector<std::byte> data_{};
};

/// A source that loses something, and only at the very end.
///
/// It exists because the interesting loss is the one discovered *during* the
/// shutdown drain: everything before that point has already been observed by a
/// pump the caller made, and a terminal intent settled before the drain would
/// close such a recording as though nothing had gone wrong.
class LateLossSource final : public ExperimentTraceSource
{
  public:
    enum class Loss : std::uint8_t
    {
        /// The producer's queue overflowed while the session was shutting down.
        dropped_during_final_drain,
        /// The summary itself did not fit, so the last record was never written.
        oversized_summary,
    };

    explicit LateLossSource(Loss loss) noexcept : loss_(loss) {}

    [[nodiscard]] ContractStatus validate() const noexcept override
    {
        return ContractStatus::ok;
    }
    [[nodiscard]] StreamStatus start_execution(ExperimentTimeNs) noexcept override
    {
        return StreamStatus::ok;
    }
    void stop_execution(bool) noexcept override
    {
        quiet_ = true;
    }
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept override
    {
        return dropped_;
    }
    [[nodiscard]] SessionMetadata metadata() const noexcept override
    {
        SessionMetadata metadata{};
        metadata.experiment_type = "late_loss";
        metadata.configuration_fingerprint = 7;
        return metadata;
    }
    void write_configuration(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept override
    {
        auto& text = sink.named(ControlKind::task_variables, "late_loss.configuration", time_ns);
        text.u64("value", 1);
        static_cast<void>(sink.commit());
    }
    [[nodiscard]] std::size_t drain(ExperimentTraceSink&, std::size_t) noexcept override
    {
        if (quiet_ && loss_ == Loss::dropped_during_final_drain)
        {
            dropped_ = 1;
        }
        return 0;
    }
    void write_summary(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept override
    {
        auto& text = sink.named(ControlKind::task_variables, "late_loss.summary", time_ns);
        if (loss_ == Loss::oversized_summary)
        {
            text.text("padding", std::string_view{std::string(600, 'x')});
        }
        text.u64("value", 2);
        static_cast<void>(sink.commit());
    }
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept override
    {
        return {};
    }
    void halt(ExperimentTimeNs, AbnormalCondition) noexcept override {}

  private:
    Loss loss_;
    bool quiet_{};
    std::uint64_t dropped_{};
};

/// A terminal consumer that fails once, on its first frame.
///
/// It stands in for the actuator or consumer edge going wrong under a run --
/// a device that refuses a write, a downstream stage that cannot accept -- and
/// it fails by returning a status rather than by throwing, because that is how
/// the runtime's own contract says an edge fails.
class FailingConsumer final : public neurale::streaming::NativeFrameConsumer
{
  public:
    [[nodiscard]] StreamStatus consume(neurale::streaming::FrameView) noexcept override
    {
        ++consumed_;
        return StreamStatus::consumer_failure;
    }
    [[nodiscard]] StreamStatus
    handle_discontinuity(const neurale::streaming::Discontinuity&) noexcept override
    {
        return StreamStatus::ok;
    }
    [[nodiscard]] StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }
    [[nodiscard]] StreamStatus reset() noexcept override
    {
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t consumed() const noexcept
    {
        return consumed_;
    }

  private:
    std::size_t consumed_{};
};

/// A paradigm that reports a condition demanding the run end, on request.
///
/// It reports it the way a real controller does -- through the summary the
/// session samples -- rather than by returning a status to the caller, because
/// a controller running inside a runtime has no caller to return one to.
class AbnormalSource final : public ExperimentTraceSource
{
  public:
    [[nodiscard]] ContractStatus validate() const noexcept override
    {
        return ContractStatus::ok;
    }
    [[nodiscard]] StreamStatus start_execution(ExperimentTimeNs) noexcept override
    {
        return StreamStatus::ok;
    }
    void stop_execution(bool aborting) noexcept override
    {
        aborting_ = aborting;
        // Whether the task had already been halted by the time the session got
        // round to stopping it. This is the ordering an emergency stop is: the
        // halt is what stops commands being produced, and it has to have
        // happened before anything begins winding down.
        halted_before_stop_ = halts_ > 0;
        ++stops_;
    }
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept override
    {
        return 0;
    }
    [[nodiscard]] SessionMetadata metadata() const noexcept override
    {
        SessionMetadata metadata{};
        metadata.experiment_type = "abnormal";
        return metadata;
    }
    void write_configuration(ExperimentTraceSink&, ExperimentTimeNs) noexcept override {}
    [[nodiscard]] std::size_t drain(ExperimentTraceSink&, std::size_t) noexcept override
    {
        return 0;
    }
    void write_summary(ExperimentTraceSink&, ExperimentTimeNs) noexcept override {}
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept override
    {
        return summary_;
    }
    void halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept override
    {
        ++halts_;
        halt_time_ns_ = time_ns;
        halt_condition_ = condition;
    }

    /// Report one condition that ends the run, exactly as a controller would.
    void demand_session_abort(AbnormalCondition condition) noexcept
    {
        ++summary_.observed;
        summary_.session_aborted = true;
        summary_.primary = condition;
        summary_.primary_policy = AbnormalPolicy::abort_session;
    }

    [[nodiscard]] std::size_t halts() const noexcept
    {
        return halts_;
    }
    [[nodiscard]] std::size_t stops() const noexcept
    {
        return stops_;
    }
    [[nodiscard]] bool stopped_as_abort() const noexcept
    {
        return aborting_;
    }
    [[nodiscard]] ExperimentTimeNs halt_time_ns() const noexcept
    {
        return halt_time_ns_;
    }
    [[nodiscard]] AbnormalCondition halt_condition() const noexcept
    {
        return halt_condition_;
    }
    [[nodiscard]] bool halted_before_stop() const noexcept
    {
        return halted_before_stop_;
    }

  private:
    AbnormalSummary summary_{};
    std::size_t halts_{};
    std::size_t stops_{};
    bool aborting_{};
    bool halted_before_stop_{};
    ExperimentTimeNs halt_time_ns_{};
    AbnormalCondition halt_condition_{AbnormalCondition::unspecified};
};

// --- the tests --------------------------------------------------------------

void test_recorded_webgrid_session_writes_metadata_first()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);

    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(fixture.recorder.state() == RecorderLifecycleState::recording);
    run_webgrid(controller, config, session);
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.state == SessionState::closed);
    CHECK(outcome.recorded);
    CHECK(outcome.experiment_trace_complete);
    CHECK(outcome.records_refused == 0);
    CHECK(outcome.trace_losses == 0);
    CHECK(outcome.records_recorded == outcome.records_encoded);
    // Finalization is a separate, offline step. The session says so rather
    // than claiming a sealed NRF session nothing wrote.
    CHECK(outcome.recorder.finalization_required);
    CHECK(outcome.recorder.nrf_committed == 0);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.session_end_present());
    CHECK(records.size() == outcome.records_recorded);
    CHECK(records.size() >= 4);
    if (records.size() < 4)
    {
        return;
    }
    // The opening records, in the order the session writes them: what the run
    // is, then what it was configured as, then everything the run produced.
    CHECK(record_name(records[0]) == "experiment.metadata");
    CHECK(records[0].fields.control_kind ==
          static_cast<std::uint32_t>(ProducerIdentityKind::task_variables));
    CHECK(records[0].fields.clock_domain == 5);
    CHECK(record_name(records[1]) == "webgrid.configuration");
    CHECK(record_name(records[2]) == "webgrid.pointer_source");

    // The metadata carries the fingerprints the caller fixed, not a rederived
    // one: a session whose metadata disagreed with its configuration record
    // would be a session nobody could match to a replay.
    const auto fingerprint = std::to_string(webgrid::configuration_fingerprint(config));
    CHECK(body_contains(records[0], fingerprint));
    CHECK(body_contains(records[0], "\\\"type\\\":\\\"webgrid\\\""));
    CHECK(body_contains(records[1], fingerprint));

    // Within a kind the identity is ascending and gapless, which is what makes
    // the order recoverable from the records rather than from their position.
    std::array<std::uint64_t, 12> expected{};
    for (const auto& record : records)
    {
        const auto kind = record.fields.control_kind;
        CHECK(kind >= 3 && kind <= 11);
        CHECK(record.fields.identity == expected[kind]);
        ++expected[kind];
    }

    bool saw_trial = false;
    bool saw_selection = false;
    bool saw_metrics = false;
    bool saw_pointer_identity = false;
    bool saw_target_identity = false;
    std::uint64_t last_pointer_ordinal = 0;
    std::uint64_t active_target_ordinal = 0;
    for (const auto& record : records)
    {
        if (record.fields.control_kind == static_cast<std::uint32_t>(ProducerIdentityKind::trials))
        {
            saw_trial = true;
            CHECK(body_contains(record, "\"start_ns\":"));
            CHECK(body_contains(record, "\"stop_ns\":"));
        }
        if (record_name(record) == "webgrid.selection_correct")
        {
            saw_selection = true;
            CHECK(body_contains(record, "\\\"pointer_update_ordinal\\\":"));
            CHECK(body_contains(record, "\\\"target_onset_ordinal\\\":"));
            std::uint64_t pointer_parent = 0;
            std::uint64_t target_parent = 0;
            CHECK(nested_u64(record, "pointer_update_ordinal", pointer_parent));
            CHECK(nested_u64(record, "target_onset_ordinal", target_parent));
            CHECK(pointer_parent == last_pointer_ordinal);
            CHECK(target_parent == active_target_ordinal);
        }
        if (record_name(record) == "webgrid.pointer")
        {
            saw_pointer_identity = true;
            CHECK(body_contains(record, "\\\"pointer_update_ordinal\\\":"));
            CHECK(nested_u64(record, "pointer_update_ordinal", last_pointer_ordinal));
        }
        if (record_name(record) == "webgrid.target_onset")
        {
            saw_target_identity = true;
            CHECK(body_contains(record, "\\\"target_onset_ordinal\\\":"));
            CHECK(nested_u64(record, "target_onset_ordinal", active_target_ordinal));
        }
        if (record_name(record) == "webgrid.metrics")
        {
            saw_metrics = true;
        }
    }
    CHECK(saw_trial);
    CHECK(saw_selection);
    CHECK(saw_metrics);
    CHECK(saw_pointer_identity);
    CHECK(saw_target_identity);
}

void test_recorder_changes_only_what_is_kept()
{
    const auto config = webgrid_config();

    webgrid::WebGridSnapshot recorded_snapshot{};
    std::uint64_t recorded_encoded = 0;
    {
        RecorderFixture fixture;
        WebGridHeadlessController controller{};
        CHECK(controller.prepare(config, 32) == StreamStatus::ok);
        WebGridTraceWriter writer{controller, kWebGridParadigm};
        ExperimentSession session{writer, manual_session()};
        CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
        CHECK(session.start(1'000) == StreamStatus::ok);
        run_webgrid(controller, config, session);
        CHECK(session.stop(5'000, "done") == StreamStatus::ok);
        CHECK(session.close() == RecorderStatusCode::ok);
        recorded_snapshot = controller.snapshot();
        recorded_encoded = session.outcome().records_encoded;
    }

    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.start(1'000) == StreamStatus::ok);
    run_webgrid(controller, config, session);
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(!outcome.recorded);
    CHECK(outcome.experiment_trace_complete);
    CHECK(outcome.records_recorded == 0);
    // The same records were built either way. A paradigm that behaved
    // differently because recording was off would be a paradigm whose recorded
    // runs did not describe its unrecorded ones.
    CHECK(outcome.records_encoded == recorded_encoded);
    const auto snapshot = controller.snapshot();
    CHECK(snapshot.completed == recorded_snapshot.completed);
    CHECK(snapshot.state == recorded_snapshot.state);
    CHECK(snapshot.metrics.correct_selections == recorded_snapshot.metrics.correct_selections);
}

void test_speech_session_keeps_intent_apart_from_report()
{
    RecorderFixture fixture;
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);

    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 16) == StreamStatus::ok);
    SpeechTraceWriter writer{scheduler, kSpeechParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);

    CHECK(session.start(1'000) == StreamStatus::ok);
    static_cast<void>(session.pump());
    for (int trial = 0; trial < 2; ++trial)
    {
        const auto end = scheduler.snapshot().timeline.trial.end_ns;
        CHECK(session.note_step(scheduler.advance(end), end) == StreamStatus::ok);
        static_cast<void>(session.pump());
    }

    // What a presenter reports arrives as its own record and never rewrites the
    // intended one.
    SpeechPresentationEvidence evidence{};
    auto& outcome = evidence.outcome;
    outcome.requested_ns = 1'000;
    outcome.presented_ns = 1'042;
    outcome.sequence = 900;
    outcome.request_sequence = 1;
    outcome.status = PresentationStatus::presented;
    outcome.stimulus_id = 1;
    evidence.software.event = PresentationLifecycleEvent::presented;
    evidence.software.time_ns = 1'042;
    evidence.software.requested_ns = 1'000;
    evidence.software.intended_ns = 1'010;
    evidence.software.submitted_renderer_ns = 50'000;
    evidence.software.submitted_ns = 1'030;
    evidence.software.presented_renderer_ns = 50'012;
    evidence.software.presented_ns = 1'042;
    evidence.software.implementation_status = 0;
    evidence.software.has_software_times = true;
    CHECK(writer.report_presentation(evidence));
    static_cast<void>(session.pump());

    CHECK(session.stop(9'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    CHECK(session.outcome().experiment_trace_complete);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);

    bool saw_schedule = false;
    bool saw_intended_phase = false;
    bool saw_request = false;
    bool saw_report = false;
    for (const auto& record : records)
    {
        const auto name = record_name(record);
        if (name == "speech.schedule")
        {
            saw_schedule = true;
            CHECK(body_contains(record, "black_duration_ns"));
            CHECK(body_contains(record, "content_duration_ns"));
            CHECK(body_contains(record, "sampler_version"));
        }
        if (name == "speech.intended_phase")
        {
            saw_intended_phase = true;
        }
        if (name == "speech.presentation_request")
        {
            saw_request = true;
            CHECK(record.fields.control_kind ==
                  static_cast<std::uint32_t>(ProducerIdentityKind::commands));
        }
        if (name == "speech.presentation_outcome")
        {
            saw_report = true;
            // Stamped with the reported instant, carrying the requested one
            // beside it. Both survive, which is the whole measurement.
            CHECK(record.fields.time_ns == 1'042);
            CHECK(body_contains(record, "\\\"requested_ns\\\":1000"));
            CHECK(body_contains(record, "\\\"request_sequence\\\":1"));
            CHECK(body_contains(record, "\\\"intended_ns\\\":1010"));
            CHECK(body_contains(record, "\\\"submitted_ns\\\":1030"));
            CHECK(body_contains(record, "\\\"submitted_renderer_ns\\\":50000"));
            CHECK(body_contains(record, "\\\"presented_renderer_ns\\\":50012"));
        }
    }
    CHECK(saw_schedule);
    CHECK(saw_intended_phase);
    CHECK(saw_request);
    CHECK(saw_report);

    // The prepared schedule's fingerprint is in the metadata, and it is the
    // contract's own, not one this layer invented.
    const auto identity = speech::schedule_identity(config, catalog);
    CHECK(body_contains(records[0], std::to_string(schedule_fingerprint(identity))));
    CHECK(body_contains(records[0], "\\\"type\\\":\\\"speech_cue\\\""));
}

void test_saturated_trace_queue_is_never_complete()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    // One slot. start() empties it of the paradigm's opening trace, so the
    // first step still fits; the second has nowhere to put its trace, because
    // the session is never pumped in between.
    CHECK(controller.prepare(config, 1) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    auto options = manual_session();
    options.trace_loss_policy = TraceLossPolicy::mark_incomplete;
    ExperimentSession session{writer, options};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    const webgrid::PointerPosition pointer{0.5, 0.5};
    CHECK(session.note_step(controller.process(1'050, pointer), 1'050) == StreamStatus::ok);
    CHECK(session.note_step(controller.process(1'100, pointer), 1'100) ==
          StreamStatus::queue_overflow);

    CHECK(!session.outcome().experiment_trace_complete);
    CHECK(session.outcome().trace_losses == 1);
    CHECK(!session.outcome().loss_faulted);

    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    const auto outcome = session.outcome();
    CHECK(!outcome.experiment_trace_complete);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    bool saw_loss = false;
    for (const auto& record : records)
    {
        if (record.fields.control_kind ==
                static_cast<std::uint32_t>(ProducerIdentityKind::faults) &&
            body_contains(record, "experiment-trace-loss"))
        {
            saw_loss = true;
            CHECK(body_contains(record, "trace_queue_overflow"));
            // Stamped with the instant the step happened at, not with zero: a
            // fault row before the session started is not a fault row.
            CHECK(record.fields.time_ns == 1'100);
        }
    }
    // The loss is a row in the recording, not only a field in a struct the
    // process took to its grave.
    CHECK(saw_loss);
}

void test_fault_policy_escalates_loss()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 1) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    const webgrid::PointerPosition pointer{0.5, 0.5};
    CHECK(session.note_step(controller.process(1'050, pointer), 1'050) == StreamStatus::ok);
    CHECK(session.note_step(controller.process(1'100, pointer), 1'100) ==
          StreamStatus::queue_overflow);

    CHECK(session.outcome().loss_faulted);
    CHECK(!session.outcome().experiment_trace_complete);

    // A stop after a latched loss is an abort, whatever the caller asked for.
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    SpoolScanReport report;
    static_cast<void>(read_control(fixture.spool, report));
    CHECK(report.has_requested_terminal_intent());
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
}

void test_stop_abort_and_close_are_idempotent()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    run_webgrid(controller, config, session);

    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    const auto after_first = session.outcome();
    CHECK(session.stop(6'000, "again") == StreamStatus::ok);
    CHECK(session.abort(7'000, "later") == StreamStatus::ok);
    const auto after_repeats = session.outcome();
    CHECK(after_repeats.records_encoded == after_first.records_encoded);
    CHECK(after_repeats.records_recorded == after_first.records_recorded);
    CHECK(after_repeats.state == SessionState::stopped);

    CHECK(session.close() == RecorderStatusCode::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    CHECK(session.outcome().state == SessionState::closed);

    SpoolScanReport report;
    static_cast<void>(read_control(fixture.spool, report));
    CHECK(report.status() == ScanStatus::ok);
    // One session end, however many times the caller asked for one.
    CHECK(report.session_end_present());
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::normal);
}

void test_refused_readiness_gate_starts_nothing()
{
    PlanFixture plan_fixture{};
    UnsupportedDurabilitySpool spool{};
    CountingUnixClock clock{};
    NativeRecorderCore recorder{};
    const auto plan = plan_fixture.plan();

    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(RecorderAttachment{
              .recorder = &recorder, .plan = &plan, .spool = &spool, .clock = &clock}) ==
          StreamStatus::ok);

    CHECK(session.start(1'000) == StreamStatus::consumer_failure);
    CHECK(session.state() == SessionState::failed);
    CHECK(session.outcome().recorder_status != RecorderStatusCode::ok);
    // Nothing was armed and nothing was recorded, so nothing was offered.
    CHECK(session.outcome().records_encoded == 0);
    CHECK(spool.size() == 0);
}

void test_invalid_paradigm_configuration_stops_before_recorder()
{
    RecorderFixture fixture;
    auto config = webgrid_config();
    config.rows = 0;
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::realtime_configuration_failed);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);

    CHECK(session.start(1'000) != StreamStatus::ok);
    CHECK(session.state() == SessionState::failed);
    // Step 1 runs before step 2, so a bad configuration never reaches a spool.
    CHECK(fixture.recorder.state() == RecorderLifecycleState::created);
    CHECK(fixture.spool.size() == 0);
}

void test_unprepared_paradigm_is_rejected_before_arming()
{
    RecorderFixture fixture;
    // A perfectly valid configuration exists, and nothing was told to run it.
    // The session validates the execution owner, not a configuration handed to
    // it, so there is nothing here for it to be satisfied by.
    WebGridHeadlessController controller{};
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);

    CHECK(session.start(1'000) != StreamStatus::ok);
    CHECK(session.state() == SessionState::failed);
    CHECK(fixture.recorder.state() == RecorderLifecycleState::created);
    CHECK(fixture.spool.size() == 0);
    CHECK(session.outcome().records_encoded == 0);
}

void test_session_start_starts_paradigm()
{
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};

    CHECK(controller.snapshot().state == webgrid::WebGridState::idle);
    CHECK(session.start(1'000) == StreamStatus::ok);
    // Running, without the caller having started anything: a start() that
    // returned ok while the paradigm was still idle would leave the first frame
    // of a runtime-driven session with nowhere to go.
    CHECK(controller.snapshot().state != webgrid::WebGridState::idle);
    CHECK(controller.start(kWebGridParadigm, 1'000) == StreamStatus::invalid_state);

    const webgrid::PointerPosition pointer{0.5, 0.5};
    CHECK(session.note_step(controller.process(1'100, pointer), 1'100) == StreamStatus::ok);

    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    // And stopped: the session made the paradigm quiet before it drained.
    CHECK(controller.process(5'100, pointer) == StreamStatus::stopped);
}

void test_failed_start_leaves_no_paradigm_running()
{
    PlanFixture plan_fixture{};
    UnsupportedDurabilitySpool spool{};
    CountingUnixClock clock{};
    NativeRecorderCore recorder{};
    const auto plan = plan_fixture.plan();

    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(RecorderAttachment{
              .recorder = &recorder, .plan = &plan, .spool = &spool, .clock = &clock}) ==
          StreamStatus::ok);

    // The readiness gate refuses its durability policy at step 4 -- before
    // the paradigm is started. Nothing to unwind, and nothing left running.
    CHECK(session.start(1'000) == StreamStatus::consumer_failure);
    CHECK(controller.snapshot().state == webgrid::WebGridState::idle);

    // Now the other side of it: a paradigm that is already running makes step 6
    // fail, and the session must not leave the recorder it started behind.
    RecorderFixture fixture;
    WebGridHeadlessController running{};
    CHECK(running.prepare(config, 32) == StreamStatus::ok);
    CHECK(running.start(kWebGridParadigm, 500) == StreamStatus::ok);
    WebGridTraceWriter second_writer{running, kWebGridParadigm};
    ExperimentSession second{second_writer, manual_session()};
    CHECK(second.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(second.start(1'000) == StreamStatus::invalid_state);
    CHECK(second.state() == SessionState::failed);
    CHECK(fixture.recorder.state() != RecorderLifecycleState::recording);
}

void test_speech_schedule_must_cover_configuration()
{
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 1> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);

    SpeechHeadlessScheduler scheduler{};
    // One trial's worth of schedule is not a schedule a two-trial session can
    // be replayed from, so the scheduler refuses to freeze it...
    CHECK(scheduler.prepare(config, catalog, schedules, 16) ==
          StreamStatus::realtime_configuration_failed);
    SpeechTraceWriter writer{scheduler, kSpeechParadigm};
    ExperimentSession session{writer, manual_session()};
    // ... and the session, which validates what the scheduler actually holds,
    // refuses before anything is prepared or armed.
    CHECK(session.start(1'000) != StreamStatus::ok);
    CHECK(session.state() == SessionState::failed);
}

void test_two_realizations_record_differently()
{
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> first{};
    CHECK(speech::prepare_trial(config, 0, first[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, first[1]) == ContractStatus::ok);

    // A second realization of the same configuration: one duration changed to
    // another value the bound admits. A seeded schedule is admitted on
    // membership, bounds, and sampler version alone, so this one is every bit
    // as legal as the first -- and a recording that cannot tell them apart
    // cannot say which run it is a recording of.
    auto second = first;
    second[0].black_duration_ns = first[0].black_duration_ns == 1 ? 2 : 1;
    CHECK(second[0].black_duration_ns != first[0].black_duration_ns);
    CHECK(speech::validate_against(second[0], config) == ContractStatus::ok);

    SpeechHeadlessScheduler first_scheduler{};
    CHECK(first_scheduler.prepare(config, catalog, first, 16) == StreamStatus::ok);
    SpeechTraceWriter first_writer{first_scheduler, kSpeechParadigm};

    SpeechHeadlessScheduler second_scheduler{};
    CHECK(second_scheduler.prepare(config, catalog, second, 16) == StreamStatus::ok);
    SpeechTraceWriter second_writer{second_scheduler, kSpeechParadigm};

    const auto left = first_writer.metadata();
    const auto right = second_writer.metadata();
    CHECK(left.configuration_fingerprint == right.configuration_fingerprint);
    // The identity is the same, because the identity is what produced them.
    CHECK(left.schedule_fingerprint == right.schedule_fingerprint);
    // What they realized is not.
    CHECK(left.realized_schedule_fingerprint != right.realized_schedule_fingerprint);
    CHECK(left.realized_schedule_fingerprint != 0);
}

void test_interrupted_session_leaves_recoverable_spool()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    run_webgrid(controller, config, session);

    // Ask for a checkpoint and wait for the worker to commit, then look at the
    // spool as a crash would leave it: records committed, no session end.
    CHECK(fixture.recorder.request_checkpoint());
    for (int attempt = 0;
         attempt < 100'000 && fixture.recorder.status().control_spool_committed == 0; ++attempt)
    {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    CHECK(fixture.recorder.status().control_spool_committed > 0);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(!report.session_end_present());
    CHECK(report.readable());
    // Recoverable means the committed prefix is promotable exactly as it
    // stands, which is what an interrupted session must leave behind.
    CHECK(report.finalizable());
    CHECK(!records.empty());
    CHECK(record_name(records[0]) == "experiment.metadata");

    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
}

void test_center_out_session_records_velocity_provenance()
{
    RecorderFixture fixture;
    auto schema = decoded_schema();
    CenterOutController controller{};
    auto config = center_out_config();
    config.guidance_resolver_version = 1;
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    writer.declare_schemas(0xABCDEF, 0x123456);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);

    // No runtime here: this test is about what the bridge records, and the
    // controller's own runtime integration is covered separately. The session still
    // starts the controller, so the frame below has somewhere to land.
    CHECK(session.start(1'000) == StreamStatus::ok);
    FrameStorage storage{};
    const std::array values{0.0, 0.0, 1.0, 0.0};
    CHECK(feed(controller, decoded_frame(storage, values, 2, 1, 0, 1'100)) == StreamStatus::ok);
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    bool saw_method = false;
    bool saw_guidance_config = false;
    bool saw_layout = false;
    bool saw_applied = false;
    bool saw_observation_provenance = false;
    bool saw_command = false;
    bool saw_command_outcome = false;
    std::uint64_t first_decision_state = (std::numeric_limits<std::uint64_t>::max)();
    std::uint64_t first_guidance_target = (std::numeric_limits<std::uint64_t>::max)();
    std::vector<std::pair<std::uint64_t, std::uint64_t>> transition_targets;
    for (const auto& record : records)
    {
        const auto name = record_name(record);
        if (name == "center_out.transition")
        {
            std::uint64_t sequence = 0;
            std::uint64_t target = 0;
            CHECK(nested_u64(record, "sequence", sequence));
            if (nested_u64(record, "active_target", target))
            {
                transition_targets.emplace_back(sequence, target);
            }
        }
        if (name == "center_out.assistance_method")
        {
            saw_method = true;
            CHECK(record.fields.control_kind ==
                  static_cast<std::uint32_t>(ProducerIdentityKind::assistance));
            CHECK(body_contains(record, "\\\"version\\\":1"));
        }
        if (name == "center_out.layout")
        {
            saw_layout = true;
        }
        if (name == "center_out.guidance_config")
        {
            saw_guidance_config = true;
            CHECK(nested_contains(record, "resolver_version", "1"));
        }
        if (name == "center_out.assisted_velocity")
        {
            saw_applied = true;
            // Who produced the velocity, what went in, what came out, and
            // whether it moved anything. A record with only the last of those
            // cannot say who is responsible for the motion.
            CHECK(body_contains(record, "\\\"decoded\\\":"));
            CHECK(body_contains(record, "\\\"guidance\\\":"));
            CHECK(body_contains(record, "\\\"assisted\\\":"));
            CHECK(body_contains(record, "\\\"applied\\\":"));
            CHECK(body_contains(record, "\\\"observation_ordinal\\\":"));
        }
        if (name == "center_out.observation_provenance")
        {
            saw_observation_provenance = true;
            CHECK(body_contains(record, "\\\"observation_ordinal\\\":"));
            CHECK(body_contains(record, "\\\"decoder_output_ordinal\\\":"));
            CHECK(body_contains(record, "\\\"frame_sequence\\\":"));
            CHECK(body_contains(record, "\\\"sample_index\\\":"));
            CHECK(body_contains(record, "\\\"state_sequence\\\":"));
            CHECK(body_contains(record, "\\\"guidance_ordinal\\\":"));
            CHECK(body_contains(record, "\\\"assistance_ordinal\\\":"));
            std::uint64_t observation = 0;
            std::uint64_t decoder = 0;
            CHECK(nested_u64(record, "observation_ordinal", observation));
            CHECK(nested_u64(record, "decoder_output_ordinal", decoder));
            CHECK(observation == decoder);
            if (observation == 0)
            {
                CHECK(nested_u64(record, "state_sequence", first_decision_state));
            }
        }
        if (name == "center_out.guidance_sample")
        {
            std::uint64_t observation = 0;
            CHECK(nested_u64(record, "observation_ordinal", observation));
            if (observation == 0)
            {
                CHECK(nested_u64(record, "state_target_id", first_guidance_target));
            }
        }
        if (name == "center_out.command")
        {
            saw_command = true;
            CHECK(record.fields.control_kind ==
                  static_cast<std::uint32_t>(ProducerIdentityKind::commands));
            CHECK(body_contains(record, "\\\"external\\\":"));
            CHECK(body_contains(record, "\\\"final\\\":"));
            CHECK(body_contains(record, "\\\"sequence\\\":"));
            CHECK(body_contains(record, "\\\"observation_ordinal\\\":"));
            std::uint64_t sequence = 0;
            std::uint64_t observation = 0;
            CHECK(nested_u64(record, "sequence", sequence));
            CHECK(nested_u64(record, "observation_ordinal", observation));
            CHECK(sequence == observation);
        }
        if (name == "center_out.command_outcome")
        {
            saw_command_outcome = true;
            CHECK(body_contains(record, "\\\"request_sequence\\\":"));
            CHECK(body_contains(record, "\\\"application\\\":1"));
            CHECK(body_contains(record, "\\\"status_code\\\":0"));
            CHECK(body_contains(record,
                                "\\\"application_scope\\\":\\\"center_out.headless_cursor\\\""));
            std::uint64_t sequence = 0;
            std::uint64_t request = 0;
            CHECK(nested_u64(record, "sequence", sequence));
            CHECK(nested_u64(record, "request_sequence", request));
            CHECK(sequence == request);
        }
    }
    CHECK(saw_method);
    CHECK(saw_guidance_config);
    CHECK(saw_layout);
    CHECK(saw_applied);
    CHECK(saw_observation_provenance);
    CHECK(saw_command);
    CHECK(saw_command_outcome);
    CHECK(first_decision_state != (std::numeric_limits<std::uint64_t>::max)());
    CHECK(first_guidance_target != (std::numeric_limits<std::uint64_t>::max)());
    bool linked_pre_step_state = false;
    bool saw_post_step_target = false;
    for (const auto& [sequence, target] : transition_targets)
    {
        if (sequence == first_decision_state)
        {
            linked_pre_step_state = target == first_guidance_target;
        }
        if (sequence > first_decision_state && target != first_guidance_target)
        {
            saw_post_step_target = true;
        }
    }
    // Observation zero acquires the centre and advances to the outward target
    // in the same machine step. Its decision records must still name the
    // centre state that produced them, not the outward state they caused.
    CHECK(linked_pre_step_state);
    CHECK(saw_post_step_target);
    // The schema fingerprints the caller declared travel in the metadata.
    CHECK(body_contains(records[0], std::to_string(0xABCDEF)));
    CHECK(body_contains(records[0], std::to_string(0x123456)));
    CHECK(body_contains(records[0], "\\\"type\\\":\\\"center_out_2d\\\""));
}

void test_center_out_records_every_segment_start()
{
    RecorderFixture fixture;
    auto schema = decoded_schema();
    auto config = center_out_config();
    // Off the centre, and fed a zero decoded velocity, so the machine stays in
    // move_to_centre for the whole run and the active target is the same cell
    // before and after the gap. That is what makes the second segment's target
    // onset a record a per-run dedupe would suppress.
    config.initial_position = {0.4, 0.0};
    CenterOutController controller{};
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    FrameStorage storage{};
    const std::array values{0.0, 0.0};
    CHECK(feed(controller, decoded_frame(storage, values, 1, 1, 0, 1'100)) == StreamStatus::ok);
    static_cast<void>(session.pump());

    // A gap resets the machine, and the observation after it opens a new
    // segment with its own start result.
    using namespace neurale::streaming;
    const SignalGap gap{.expected_sample_idx = 1,
                        .actual_sample_idx = 4,
                        .missing_samples = 3,
                        .signal_id = 9,
                        .reason = GapReason::source_gap,
                        .flags = SignalGapFlags::missing_samples_known};
    const std::array gaps{gap};
    const Discontinuity discontinuity{3, 1, 2, gaps, GapReason::source_gap};
    CHECK(controller.handle_discontinuity(discontinuity) == StreamStatus::ok);
    FrameStorage restart_storage{};
    CHECK(feed(controller, decoded_frame(restart_storage, values, 1, 2, 4, 300'001'100)) ==
          StreamStatus::ok);
    static_cast<void>(session.pump());

    CHECK(session.stop(400'000'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);

    // The run's opening: the machine's start result travels in the session-start
    // trace's `step`, and reading the empty `segment_start` instead loses the
    // SESSION_START transition and the target the run opens on entirely.
    CHECK(count_at(records, "center_out.transition", 1'000) >= 1);
    CHECK(count_at(records, "center_out.active_target", 1'000) == 1);

    // The gap itself, stamped with the last instant the run reached rather than
    // with zero, and named by the shared abnormal vocabulary rather than by a
    // paradigm-private record name: a reader looking for what went wrong in a
    // run must not have to know which paradigm produced the record first.
    CHECK(count_coded(records, "source-discontinuity") == 1);
    CHECK(count_coded_at(records, "source-discontinuity", 1'100) == 1);
    // And the trial the gap ended, written as a trial with an aborted outcome
    // rather than left as a hole in the trial sequence.
    std::size_t aborted_trials = 0;
    for (const auto& record : records)
    {
        if (body_contains(record, "center_out.trial_aborted"))
        {
            ++aborted_trials;
            CHECK(body_contains(record, "source-discontinuity"));
        }
    }
    CHECK(aborted_trials == 1);

    // The restarted segment's own start, which travels in `segment_start`
    // beside the observation that carried it. Its target is the same cell that
    // was up before the gap, so a dedupe that survives the reset suppresses the
    // one record proving the new segment had a target at all.
    CHECK(count_at(records, "center_out.transition", 300'001'100) >= 1);
    CHECK(count_at(records, "center_out.active_target", 300'001'100) == 1);
    CHECK(count_named(records, "center_out.active_target") == 2);
}

void test_center_out_records_configuration_sampler()
{
    RecorderFixture fixture;
    auto schema = decoded_schema();
    auto config = center_out_config();
    // A version this build cannot regenerate. The contract admits it, and
    // Center-Out runs under it because nothing here re-executes a sampler.
    config.task.sampler_version = 4242;
    CHECK(!sampler_version_supported(config.task.sampler_version));

    CenterOutController controller{};
    CHECK(controller.prepare(schema, config) == StreamStatus::ok);
    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(!records.empty());
    CHECK(record_name(records[0]) == "experiment.metadata");
    // The configuration's own sampler, and the schedule identity built from it.
    // Recording this build's version instead would make the metadata contradict
    // the configuration fingerprint beside it.
    CHECK(body_contains(records[0], "\\\"sampler_version\\\":4242"));
    const ScheduleIdentity identity{
        .seed = config.task.seed,
        .sampler_version = 4242,
        .configuration_fingerprint = center_out::configuration_fingerprint(config.task),
        .catalog_fingerprint = 0,
    };
    CHECK(body_contains(records[0], std::to_string(schedule_fingerprint(identity))));
}

void test_oversized_body_is_rejected_not_truncated()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    auto options = manual_session();
    // Smaller than any record this layer builds. Nothing may be offered under
    // it, and nothing may be silently shortened to fit.
    options.max_control_body_bytes = 8;
    options.trace_loss_policy = TraceLossPolicy::mark_incomplete;
    ExperimentSession session{writer, options};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    run_webgrid(controller, config, session);
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.records_recorded == 0);
    CHECK(outcome.records_refused > 0);
    CHECK(!outcome.experiment_trace_complete);
    CHECK(outcome.trace_losses > 0);
}

void test_loss_in_final_drain_ends_recording_as_abort()
{
    RecorderFixture fixture;
    LateLossSource source{LateLossSource::Loss::dropped_during_final_drain};
    ExperimentSession session{source, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // Nothing has gone wrong yet, and the caller asks for a graceful stop.
    CHECK(!session.outcome().loss_faulted);
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.loss_faulted);
    CHECK(!outcome.experiment_trace_complete);
    CHECK(outcome.trace_losses == 1);
    CHECK(outcome.producer_trace_drops == 1);
    // The intent the recorder was ended under, and the intent the recording
    // itself carries. A session that knows its trace has a hole must not leave
    // behind a recording that says it stopped cleanly.
    CHECK(outcome.terminal_abort);
    SpoolScanReport report;
    static_cast<void>(read_control(fixture.spool, report));
    CHECK(report.has_requested_terminal_intent());
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
}

void test_oversized_summary_is_a_loss()
{
    RecorderFixture fixture;
    LateLossSource source{LateLossSource::Loss::oversized_summary};
    ExperimentSession session{source, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    // The summary is the last record offered and the only one refused. It is
    // offered after the last pump, so a session that accounted refusals only
    // inside pump() would report a complete trace whose final record was never
    // written.
    CHECK(outcome.records_refused == 1);
    CHECK(outcome.trace_losses == 1);
    CHECK(!outcome.experiment_trace_complete);
    CHECK(outcome.terminal_abort);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
    CHECK(count_named(records, "late_loss.summary") == 0);
    bool saw_loss = false;
    for (const auto& record : records)
    {
        if (body_contains(record, "experiment-trace-loss"))
        {
            saw_loss = true;
        }
    }
    CHECK(saw_loss);
}

void test_unseen_producer_drop_is_still_a_loss()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    // One slot, which the session's own start leaves empty; the step below
    // fills it, and the step after that is the one with nowhere to go.
    CHECK(controller.prepare(config, 1) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    auto options = manual_session();
    options.trace_loss_policy = TraceLossPolicy::mark_incomplete;
    ExperimentSession session{writer, options};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // Deliberately discarded, as a runtime-driven controller's status is: the
    // runner sees it, and nobody who could call note_step() ever does.
    const webgrid::PointerPosition pointer{0.5, 0.5};
    CHECK(controller.process(1'050, pointer) == StreamStatus::ok);
    CHECK(controller.process(1'100, pointer) == StreamStatus::queue_overflow);
    CHECK(session.outcome().experiment_trace_complete);

    // The next drain samples the producer's own counter, which is where the
    // loss actually lives.
    static_cast<void>(session.pump());
    const auto after_pump = session.outcome();
    CHECK(!after_pump.experiment_trace_complete);
    CHECK(after_pump.producer_trace_drops == 1);
    CHECK(after_pump.trace_losses == 1);

    // And it is counted once, not once per observer of the same drop.
    static_cast<void>(session.pump());
    CHECK(session.outcome().trace_losses == 1);

    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    CHECK(!session.outcome().experiment_trace_complete);
}

void test_recorded_provenance_matches_running_configuration()
{
    RecorderFixture fixture;
    const auto running = webgrid_config();
    // A second configuration, equally valid, that nothing is going to execute.
    auto other = webgrid_config();
    other.rows = 2;
    other.bounds = webgrid::TaskBounds{0.0, 2.0, 0.0, 2.0};
    CHECK(webgrid::validate(other) == ContractStatus::ok);
    const auto running_fingerprint = webgrid::configuration_fingerprint(running);
    const auto other_fingerprint = webgrid::configuration_fingerprint(other);
    CHECK(running_fingerprint != other_fingerprint);

    WebGridHeadlessController controller{};
    CHECK(controller.prepare(running, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    run_webgrid(controller, running, session);
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(records.size() >= 2);
    // Read out of the controller, so there is no second configuration for the
    // recording to describe by mistake.
    CHECK(body_contains(records[0], std::to_string(running_fingerprint)));
    CHECK(!body_contains(records[0], std::to_string(other_fingerprint)));
    CHECK(body_contains(records[1], std::to_string(running_fingerprint)));
    CHECK(!body_contains(records[1], std::to_string(other_fingerprint)));
}

void test_second_session_does_not_inherit_drops()
{
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    // One slot, so the first session can be made to lose exactly one record.
    CHECK(controller.prepare(config, 1) == StreamStatus::ok);

    {
        RecorderFixture fixture;
        WebGridTraceWriter writer{controller, kWebGridParadigm};
        auto options = manual_session();
        options.trace_loss_policy = TraceLossPolicy::mark_incomplete;
        ExperimentSession first{writer, options};
        CHECK(first.enable_recording(fixture.attachment()) == StreamStatus::ok);
        CHECK(first.start(1'000) == StreamStatus::ok);
        const webgrid::PointerPosition pointer{0.5, 0.5};
        CHECK(first.note_step(controller.process(1'050, pointer), 1'050) == StreamStatus::ok);
        CHECK(first.note_step(controller.process(1'100, pointer), 1'100) ==
              StreamStatus::queue_overflow);
        CHECK(first.outcome().producer_trace_drops == 1);
        CHECK(first.stop(5'000, "done") == StreamStatus::ok);
        CHECK(first.close() == RecorderStatusCode::ok);
    }

    // The same controller, run again. Its counter is monotonic for its own
    // lifetime by design, so it still reads the first session's loss...
    CHECK(controller.reset() == StreamStatus::ok);
    CHECK(controller.dropped_trace_count() == 1);

    RecorderFixture fixture;
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession second{writer, manual_session()};
    CHECK(second.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(second.start(2'000) == StreamStatus::ok);
    static_cast<void>(second.pump());

    // ... and none of it belongs to this session, which lost nothing. A session
    // that measured from zero would open by accusing a run that has not yet
    // done anything.
    const auto during = second.outcome();
    CHECK(during.producer_trace_drops == 0);
    CHECK(during.trace_losses == 0);
    CHECK(during.experiment_trace_complete);

    CHECK(second.stop(6'000, "done") == StreamStatus::ok);
    CHECK(second.close() == RecorderStatusCode::ok);
    const auto outcome = second.outcome();
    CHECK(outcome.producer_trace_drops == 0);
    CHECK(outcome.trace_losses == 0);
    CHECK(outcome.experiment_trace_complete);
    // And it is ended as the clean stop it was.
    CHECK(!outcome.terminal_abort);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(count_named(records, "experiment.trace_loss") == 0);
    for (const auto& record : records)
    {
        CHECK(!body_contains(record, "trace_queue_overflow"));
    }
}

void test_runtime_needs_session_with_drain()
{
    CenterOutController consumer{};
    CHECK(consumer.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    CountedDecodedSource source{1};
    ForwardDecodedProcessor processor{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(), runtime_config(), source,
                                                   processor, consumer};

    UnstartableSource paradigm{};
    // manual_session() has no bridge: the caller pumps. That is a contract a
    // caller can keep for a paradigm it steps itself and cannot keep for a
    // runtime, which produces on threads no pump of theirs is ordered against.
    ExperimentSession session{paradigm, manual_session()};
    CHECK(session.attach_runtime(runtime) == StreamStatus::realtime_configuration_failed);
}

void test_failed_start_reinhibits_armed_runtime()
{
    RecorderFixture fixture;
    CenterOutController consumer{};
    CHECK(consumer.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    CountedDecodedSource source{1};
    ForwardDecodedProcessor processor{};
    RecordingSafetyController safety{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(),
                                                   runtime_config(),
                                                   source,
                                                   processor,
                                                   consumer,
                                                   neurale::streaming::default_native_clock(),
                                                   safety};

    UnstartableSource paradigm{};
    ExperimentSession session{paradigm, bridged_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.attach_runtime(runtime) == StreamStatus::ok);

    // The paradigm fails at step 6, which is after arm(). arm() is the only
    // thing that releases safety, and it leaves the runtime in `prepared` --
    // so a runtime that was armed and never started looks, by its own state,
    // exactly like one that was never armed.
    CHECK(session.start(1'000) == StreamStatus::realtime_configuration_failed);
    CHECK(session.state() == SessionState::failed);
    CHECK(runtime.state() == neurale::streaming::RuntimeState::stopped);
    // The part that matters: the actuator path is inhibited again. A failed
    // start that left it released would leave a released safety controller
    // behind a session the caller was told did not start.
    CHECK(!safety.released());
    CHECK(paradigm.stops() == 0);
    static_cast<void>(session.close());
}

void test_opening_trace_leaves_producer_before_start_returns()
{
    RecorderFixture fixture;
    CenterOutController controller{};
    // One slot, and the paradigm's own start fills it. Whether it is still full
    // when start() returns is the whole question: a runtime started at that
    // moment has nowhere to put its first frame's trace.
    CHECK(controller.prepare(decoded_schema(), center_out_config(1)) == StreamStatus::ok);
    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // No bridge and no runtime, so this is not a race the scheduler could have
    // won for us: an empty drain here means start() itself emptied the queue,
    // on the caller's thread, before it returned.
    CHECK(session.pump() == 0);

    // And it was emptied by being recorded, not by being discarded.
    FrameStorage storage{};
    const std::array values{0.0, 0.0};
    CHECK(feed(controller, decoded_frame(storage, values, 1, 1, 0, 1'100)) == StreamStatus::ok);
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    const auto outcome = session.outcome();
    CHECK(outcome.producer_trace_drops == 0);
    CHECK(outcome.experiment_trace_complete);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(count_at(records, "center_out.transition", 1'000) >= 1);
}

void test_runtime_never_produces_into_undrained_queue()
{
    RecorderFixture fixture;
    CenterOutController controller{};
    // One slot: the smallest capacity the controller allows, which is what turns the
    // window between "the paradigm has started" and "something is draining"
    // from a probability into an observation.
    CHECK(controller.prepare(decoded_schema(), center_out_config(1)) == StreamStatus::ok);
    CountedDecodedSource source{1};
    ForwardDecodedProcessor processor{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(), runtime_config(), source,
                                                   processor, controller};

    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    ExperimentSession session{writer, bridged_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.attach_runtime(runtime) == StreamStatus::ok);

    // start() starts the paradigm, whose own start already fills the single
    // slot, and only then starts the runtime -- with the slot emptied on this
    // thread first and the bridge already running.
    CHECK(session.start(1'000) == StreamStatus::ok);
    static_cast<void>(runtime.join());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.producer_trace_drops == 0);
    CHECK(outcome.trace_losses == 0);
    CHECK(outcome.experiment_trace_complete);
    CHECK(!outcome.terminal_abort);

    // Both halves reached the recording: the opening the paradigm produced
    // before the runtime existed, and the frame the runtime delivered after it.
    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(count_at(records, "center_out.transition", 1'000) >= 1);
    CHECK(count_named(records, "center_out.command") == 1);
}

void test_undrainable_session_is_rejected_early()
{
    RecorderFixture fixture;
    CenterOutController controller{};
    CHECK(controller.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    CountedDecodedSource source{1};
    ForwardDecodedProcessor processor{};
    RecordingSafetyController safety{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(),
                                                   runtime_config(),
                                                   source,
                                                   processor,
                                                   controller,
                                                   neurale::streaming::default_native_clock(),
                                                   safety};

    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    auto options = bridged_session();
    // Every drain loop is bounded by this budget, so zero is a session that
    // would report draining while moving nothing -- including the synchronous
    // drain that is supposed to empty the paradigm's opening trace.
    options.drain_budget = 0;
    ExperimentSession session{writer, options};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.attach_runtime(runtime) == StreamStatus::ok);

    CHECK(session.start(1'000) == StreamStatus::realtime_configuration_failed);
    CHECK(session.state() == SessionState::failed);

    // Nothing was touched: the refusal is step 0, before the recorder is
    // prepared, before the runtime is prepared or armed, and before the
    // paradigm is started.
    CHECK(fixture.recorder.state() == RecorderLifecycleState::created);
    CHECK(runtime.state() == neurale::streaming::RuntimeState::created);
    CHECK(!safety.released());
    CHECK(controller.snapshot().state == center_out::CenterOutState::idle);
}

void test_throwing_bring_up_unwinds_completely()
{
    RecorderFixture fixture;
    CenterOutController controller{};
    CHECK(controller.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    CountedDecodedSource source{1};
    ForwardDecodedProcessor processor{};
    RecordingSafetyController safety{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(),
                                                   runtime_config(),
                                                   source,
                                                   processor,
                                                   controller,
                                                   neurale::streaming::default_native_clock(),
                                                   safety};

    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    ExperimentSession session{writer, bridged_session()};
    SessionTestHooks hooks{};
    // The bridge thread is the one bring-up step that fails by throwing rather
    // than by returning a status, and it fails at the worst possible moment:
    // after arm(), after the paradigm started, after the recorder started.
    hooks.bridge_thread_fails = true;
    session.set_test_hooks(&hooks);
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.attach_runtime(runtime) == StreamStatus::ok);

    bool threw = false;
    try
    {
        static_cast<void>(session.start(1'000));
    }
    catch (const std::system_error&)
    {
        threw = true;
    }
    CHECK(threw);

    // The exception says what failed. The session still says what it left
    // behind, which is nothing.
    CHECK(session.state() == SessionState::failed);
    CHECK(runtime.state() == neurale::streaming::RuntimeState::stopped);
    CHECK(!safety.released());

    // The paradigm was stopped, so a frame arriving now is refused rather than
    // stepped into a run nobody is recording.
    FrameStorage storage{};
    const std::array values{0.0, 0.0};
    CHECK(feed(controller, decoded_frame(storage, values, 1, 1, 0, 1'100)) != StreamStatus::ok);

    // And the recording was ended as the abort it is, not as a run that
    // finished behind a start the caller was told had failed.
    static_cast<void>(session.close());
    SpoolScanReport report;
    static_cast<void>(read_control(fixture.spool, report));
    CHECK(report.has_requested_terminal_intent());
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
}

void test_destroying_running_session_ends_experiment()
{
    // A caller-stepped paradigm first: nothing but the session knows it is
    // running, so nothing but the session can stop it.
    {
        RecorderFixture fixture;
        const auto config = webgrid_config();
        WebGridHeadlessController controller{};
        CHECK(controller.prepare(config, 32) == StreamStatus::ok);
        const webgrid::PointerPosition pointer{0.5, 0.5};
        {
            WebGridTraceWriter writer{controller, kWebGridParadigm};
            ExperimentSession session{writer, manual_session()};
            CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
            CHECK(session.start(1'000) == StreamStatus::ok);
            CHECK(session.note_step(controller.process(1'100, pointer), 1'100) == StreamStatus::ok);
        }
        // Destroyed without stop() or close(). A session that only ended its
        // bridge would leave this stepping happily into nothing.
        CHECK(controller.process(1'200, pointer) != StreamStatus::ok);
        CHECK(fixture.recorder.state() != RecorderLifecycleState::recording);
        SpoolScanReport report;
        static_cast<void>(read_control(fixture.spool, report));
        CHECK(report.has_requested_terminal_intent());
        CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
    }

    // And an attached runtime, where leaving it running would leave it
    // producing into a queue whose only consumer had just been joined.
    RecorderFixture fixture;
    CenterOutController controller{};
    CHECK(controller.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    // A source that cannot end on its own, so the runtime is certainly still
    // running when the session goes out of scope. One that had already ended
    // would let a destructor which stops nothing pass this by luck.
    HeldDecodedSource source{};
    ForwardDecodedProcessor processor{};
    RecordingSafetyController safety{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(),
                                                   runtime_config(),
                                                   source,
                                                   processor,
                                                   controller,
                                                   neurale::streaming::default_native_clock(),
                                                   safety};
    {
        CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
        ExperimentSession session{writer, bridged_session()};
        CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
        CHECK(session.attach_runtime(runtime) == StreamStatus::ok);
        CHECK(session.start(1'000) == StreamStatus::ok);
    }
    // The runtime is down and safety is inhibited. Neither is evidence on its
    // own -- a runtime that outruns the bridge ends itself under the default
    // loss policy, and ending itself inhibits safety too -- which is why the
    // recording is what this half actually turns on: a destructor that stopped
    // only the bridge leaves a recorder that was never ended at all.
    CHECK(runtime.state() != neurale::streaming::RuntimeState::running);
    CHECK(!safety.released());
    CHECK(fixture.recorder.state() != RecorderLifecycleState::recording);
    SpoolScanReport runtime_report;
    static_cast<void>(read_control(fixture.spool, runtime_report));
    CHECK(runtime_report.has_requested_terminal_intent());
    CHECK(runtime_report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
}

// --- Safety, faults, and abnormal session semantics -------------------

void test_safety_is_inhibited_outside_armed_window()
{
    RecorderFixture fixture;
    CenterOutController controller{};
    CHECK(controller.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    CountedDecodedSource source{2};
    ForwardDecodedProcessor processor{};
    RecordingSafetyController safety{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(),
                                                   runtime_config(),
                                                   source,
                                                   processor,
                                                   controller,
                                                   neurale::streaming::default_native_clock(),
                                                   safety};

    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    ExperimentSession session{writer, bridged_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.attach_runtime(runtime) == StreamStatus::ok);

    // Nothing has armed anything, so the actuator path is inhibited. This is
    // the state a caller who constructed a runtime and a task, and started
    // neither, is entitled to.
    CHECK(!safety.released());

    CHECK(session.start(1'000) == StreamStatus::ok);
    // Released, and released by exactly one thing: NativeStreamRunner::arm().
    // The session never touches the SafetyController -- a second path to
    // release would be a second policy, and only one of them would be the one
    // that checks every critical edge first.
    CHECK(safety.released());

    static_cast<void>(runtime.join());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    // And inhibited again by the runtime's own shutdown, not by this layer.
    CHECK(!safety.released());
    CHECK(session.close() == RecorderStatusCode::ok);
}

void test_consumer_fault_survives_shutdown()
{
    RecorderFixture fixture;
    FailingConsumer consumer{};
    CountedDecodedSource source{4};
    ForwardDecodedProcessor processor{};
    RecordingSafetyController safety{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(),
                                                   runtime_config(),
                                                   source,
                                                   processor,
                                                   consumer,
                                                   neurale::streaming::default_native_clock(),
                                                   safety};

    AbnormalSource paradigm{};
    ExperimentSession session{paradigm, bridged_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.attach_runtime(runtime) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    static_cast<void>(runtime.join());

    // The terminal edge failed, so the runtime faulted and took itself down --
    // which is also what re-inhibits safety. That decision is the runtime's,
    // and the session does not have a second one.
    CHECK(consumer.consumed() >= 1);
    CHECK(!safety.released());

    // A stop issued *after* a fault reports on the stop. The session keeps what
    // it already knew instead: the fault the runtime recorded first.
    static_cast<void>(session.stop(2'000, "done"));
    CHECK(session.close() == RecorderStatusCode::ok);
    const auto outcome = session.outcome();
    CHECK(outcome.has_runtime_fault);
    // The terminal edge of this topology is the actuator, so that is the stage
    // the runtime recorded it under. What the session preserves is the
    // runtime's own account, not a re-derivation of it.
    CHECK(outcome.runtime_fault.code == neurale::streaming::FaultCode::actuator_write);
    CHECK(outcome.runtime_fault.status == StreamStatus::consumer_failure);
    CHECK(outcome.runtime_fault.stage == neurale::streaming::FaultStage::actuator);
    // And the status the session reports is the fault, not the shutdown that
    // reacted to it.
    CHECK(outcome.runtime_status == StreamStatus::consumer_failure);
    // The paradigm's execution was stopped as the abort it was, not as a clean
    // stop that happened to follow a failure.
    CHECK(paradigm.stops() == 1);

    // And the fault reached the recording, under the runtime's own stage rather
    // than the experiment's: a recording holding the experiment's view of a
    // failed run and not the runtime's is missing the half that says why.
    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(count_coded(records, "runtime-primary-fault") == 1);
    bool saw_stage = false;
    for (const auto& record : records)
    {
        if (record_code(record) == "runtime-primary-fault")
        {
            saw_stage = record_field(record, "\"stage\":\"") == "runtime";
        }
    }
    CHECK(saw_stage);
}

void test_paradigm_demanded_end_is_an_abort()
{
    RecorderFixture fixture;
    AbnormalSource paradigm{};
    ExperimentSession session{paradigm, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(!session.outcome().abnormal_session_aborted);

    // Reported through the summary the session samples on every drain, which is
    // the only channel a paradigm running inside a runtime has.
    paradigm.demand_session_abort(AbnormalCondition::recorder_fault);
    static_cast<void>(session.pump());

    const auto during = session.outcome();
    CHECK(during.abnormal_session_aborted);
    CHECK(during.abnormal_conditions == 1);
    CHECK(during.primary_abnormal == AbnormalCondition::recorder_fault);

    // A stop after that is an abort, whatever the caller asked for -- the same
    // rule a latched trace loss follows, for the same reason.
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    CHECK(session.outcome().terminal_abort);
    CHECK(paradigm.stopped_as_abort());

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
    // The session's reaction has a code of its own, and carries the condition's
    // own contract name as a field, so the reason is in the recording and not
    // only in a struct the process took to its grave. It is deliberately not
    // filed under `recorder-fault`: that name belongs to the condition, and a
    // reader counting conditions must not count reactions to them as well.
    CHECK(count_coded(records, "experiment-session-aborted") == 1);
    CHECK(count_coded(records, "recorder-fault") == 0);
    bool saw_condition = false;
    for (const auto& record : records)
    {
        if (record_code(record) == "experiment-session-aborted")
        {
            saw_condition = nested_contains(record, "primary_condition", "\\\"recorder-fault\\\"");
        }
    }
    CHECK(saw_condition);
}

void test_emergency_stop_halts_task_first()
{
    RecorderFixture fixture;
    AbnormalSource paradigm{};
    ExperimentSession session{paradigm, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(paradigm.halts() == 0);

    CHECK(session.emergency_stop(4'000, "operator") == StreamStatus::ok);

    // The task was halted, and halted first: an emergency stop is a statement
    // about command application, so the paradigm stops producing commands
    // before anything is asked to wind down. The ordering is visible here
    // because stop_execution() is shutdown step 3 and the halt is step zero.
    CHECK(paradigm.halts() == 1);
    CHECK(paradigm.halt_time_ns() == 4'000);
    CHECK(paradigm.halt_condition() == AbnormalCondition::emergency_stop);
    CHECK(paradigm.stopped_as_abort());
    // The ordering itself, not just that both happened: the task was already
    // refusing input by the time the session began winding anything down.
    CHECK(paradigm.halted_before_stop());

    const auto outcome = session.outcome();
    CHECK(outcome.state == SessionState::stopped);
    CHECK(outcome.terminal_abort);
    CHECK(outcome.abnormal_session_aborted);

    // Idempotent, and never upgraded back to a graceful stop.
    CHECK(session.emergency_stop(5'000, "again") == StreamStatus::ok);
    CHECK(session.stop(6'000, "please") == StreamStatus::ok);
    CHECK(paradigm.halts() == 1);
    CHECK(paradigm.stops() == 1);
    CHECK(session.close() == RecorderStatusCode::ok);
    CHECK(session.close() == RecorderStatusCode::ok);
    CHECK(session.outcome().state == SessionState::closed);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.session_end_present());
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
    // One record, however many times the caller asked for a stop -- and it is
    // the *request*, under its own code. What the task did about the stop is
    // the paradigm's own abnormal record under `emergency-stop`; this source is
    // a double that produces none, which is exactly why the two codes have to
    // be distinguishable rather than counted together.
    CHECK(count_coded(records, "emergency-stop-requested") == 1);
    CHECK(count_coded_at(records, "emergency-stop-requested", 4'000) == 1);
    CHECK(count_coded(records, "emergency-stop") == 0);
}

void test_emergency_stop_releases_nothing()
{
    RecorderFixture fixture;
    CenterOutController controller{};
    CHECK(controller.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    // A source that will not end on its own, so the run is genuinely still
    // running when the stop arrives. One that had already finished -- or had
    // outrun the bridge and ended itself -- would let a do-nothing emergency
    // stop pass by luck, and would make `safety.released()` below a race.
    HeldDecodedSource source{};
    ForwardDecodedProcessor processor{};
    RecordingSafetyController safety{};
    neurale::streaming::NativeStreamRunner runtime{decoded_schema(),
                                                   runtime_config(),
                                                   source,
                                                   processor,
                                                   controller,
                                                   neurale::streaming::default_native_clock(),
                                                   safety};

    CenterOutTraceWriter writer{controller, kCenterOutHostEpoch};
    ExperimentSession session{writer, bridged_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.attach_runtime(runtime) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(safety.released());

    CHECK(session.emergency_stop(9'000, "operator") == StreamStatus::ok);

    // The actuator path is inhibited -- by the runtime, which is the only thing
    // that decides it -- and the task refuses everything from here.
    CHECK(!safety.released());
    CHECK(controller.halted());
    FrameStorage storage{};
    const std::array values{0.0, 0.0};
    CHECK(feed(controller, decoded_frame(storage, values, 1, 1, 0, 1'100)) != StreamStatus::ok);

    CHECK(session.close() == RecorderStatusCode::ok);
    const auto outcome = session.outcome();
    CHECK(outcome.terminal_abort);
    CHECK(outcome.abnormal_session_aborted);
    CHECK(outcome.abnormal_conditions >= 1);
    CHECK(outcome.primary_abnormal == AbnormalCondition::emergency_stop);

    SpoolScanReport report;
    static_cast<void>(read_control(fixture.spool, report));
    CHECK(report.has_requested_terminal_intent());
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::aborted);
}

void test_absent_presentation_is_not_completed_trial()
{
    RecorderFixture fixture;
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);

    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 64) == StreamStatus::ok);
    SpeechTraceWriter writer{scheduler, kSpeechParadigm};
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    writer.require_presentation_evidence(true);
    SpeechPresentationConfigRecord presentation_config{};
    presentation_config.font_path = "configured-font.ttf";
    presentation_config.font_sha256[0] = 0xab;
    presentation_config.font_sha256[1] = 0xcd;
    presentation_config.face_idx = 2;
    presentation_config.pixel_height = 36;
    presentation_config.logical_left = -1.0;
    presentation_config.logical_bottom = -1.0;
    presentation_config.logical_width = 2.0;
    presentation_config.logical_height = 2.0;
    presentation_config.fixation_half_extent = 0.08;
    presentation_config.text_baseline_x = -0.8;
    presentation_config.window_width = 800;
    presentation_config.window_height = 600;
    presentation_config.monitor_idx = -1;
    presentation_config.swap_interval = 1;
    presentation_config.resizable = true;
    presentation_config.visible = true;
    CHECK(writer.report_presentation_config(std::move(presentation_config)));
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // Advance into trial zero's content phase and no further: the CONTENT
    // request has been made and recorded, and the trial has not been decided
    // yet. A presenter's report is about a request, so there has to be one
    // before there is anything to report on.
    static_cast<void>(session.note_step(scheduler.advance(1'150), 1'150));
    static_cast<void>(session.pump());

    // The identity of the request the run actually emitted, asked of the run
    // rather than assumed. A report that named a request nobody made is not
    // evidence about this trial, and a test that assumed one would be proving
    // the opposite of what it set out to.
    EmittedContentRequest emitted{};
    CHECK(writer.emitted_content_request(0, emitted));
    // The instant the seeded schedule put the content phase at, not a number
    // written here: the duration is drawn under the configured bound, so what
    // this can assert is the bound, and the rest comes from the run.
    CHECK(emitted.requested_ns > 1'000);
    CHECK(emitted.requested_ns <= 1'000 + config.black_bound_ns);
    CHECK(emitted.trial.ordinal == 0);

    // A presenter reporting that it did not present trial zero's content. The
    // controller layer never produces one of these; the shape exists so that a
    // presenter has somewhere to report without inventing a parallel vocabulary.
    PresentationOutcome failed{};
    failed.requested_ns = emitted.requested_ns;
    failed.presented_ns = 0;
    failed.sequence = 1;
    failed.request_sequence = emitted.sequence;
    failed.trial.ordinal = 0;
    failed.status = PresentationStatus::skipped;
    failed.stimulus_id = emitted.stimulus_id;
    CHECK(writer.report_presentation(failed));
    static_cast<void>(session.pump());

    // Run both trials to completion. The machine completes trial zero normally
    // -- it has no way not to -- which is exactly why the writer has to be the
    // thing that refuses to call it a success.
    for (ExperimentTimeNs time = 2'000; time <= 2'000 + 40'000'000; time += 1'000'000)
    {
        static_cast<void>(session.note_step(scheduler.advance(time), time));
        static_cast<void>(session.pump());
    }
    CHECK(session.stop(1'000 + 50'000'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.abnormal_conditions >= 1);
    CHECK(outcome.abnormal_trials_affected >= 1);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(count_coded(records, "presentation-failed") == 1);

    bool saw_report = false;
    bool saw_presentation_config = false;
    bool saw_presentation_font = false;
    bool saw_presentation_window = false;
    bool saw_presentation_style = false;
    bool saw_invalidated_trial = false;
    bool saw_admitted_trial = false;
    for (const auto& record : records)
    {
        saw_presentation_config =
            saw_presentation_config || record_name(record) == "speech.presentation_config";
        if (record_name(record) == "speech.presentation_font")
        {
            saw_presentation_font =
                body_contains(record, "configured-font.ttf") &&
                body_contains(record,
                              "abcd000000000000000000000000000000000000000000000000000000000000") &&
                nested_contains(record, "face_index", "2");
        }
        saw_presentation_window =
            saw_presentation_window || record_name(record) == "speech.presentation_window";
        saw_presentation_style =
            saw_presentation_style || record_name(record) == "speech.presentation_style";
        if (body_contains(record, "speech.presentation_outcome"))
        {
            saw_report = true;
            // Missing evidence is recorded as missing. Writing the field's zero
            // -- or the intended onset -- would put an instant in a column a
            // reader is entitled to read as a measurement, for a presentation
            // that never happened.
            CHECK(nested_contains(record, "presented_ns", "null"));
            // And the record is stamped with the instant that does exist.
            CHECK(record.fields.time_ns == emitted.requested_ns);
            // Attributed to a request the run made, and the record says so.
            CHECK(nested_contains(record, "matched_request", "true"));
        }
        if (record.fields.control_kind == static_cast<std::uint32_t>(ProducerIdentityKind::trials))
        {
            if (nested_contains(record, "invalidated", "true"))
            {
                saw_invalidated_trial = true;
                // TrialOutcome::aborted is 4. The machine's own verdict is kept
                // beside it rather than destroyed.
                CHECK(nested_contains(record, "outcome", "4"));
                CHECK(nested_contains(record, "task_outcome", "1"));
            }
            else
            {
                saw_admitted_trial = true;
            }
        }
    }
    CHECK(saw_presentation_config);
    CHECK(saw_presentation_font);
    CHECK(saw_presentation_window);
    CHECK(saw_presentation_style);
    CHECK(saw_report);
    CHECK(saw_invalidated_trial);
    // The other trial is untouched: one presentation failing does not make a
    // whole run inadmissible.
    CHECK(saw_admitted_trial);
}

void test_black_renderer_failure_names_its_request()
{
    RecorderFixture fixture;
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);
    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 64) == StreamStatus::ok);
    SpeechTraceWriter writer{scheduler, kSpeechParadigm};
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // The controller has emitted BLACK, but the recording consumer has not
    // drained that trace yet. A report arriving now must still match.
    EmittedPresentationRequest emitted{};
    CHECK(writer.emitted_presentation_request(0, speech::SpeechPhase::black, emitted));
    CHECK(emitted.cue == CueKind::black);
    CHECK(emitted.stimulus_id == kUnsetStimulusId);

    PresentationOutcome failed{};
    failed.requested_ns = emitted.requested_ns;
    failed.sequence = 1;
    failed.request_sequence = emitted.sequence;
    failed.trial = emitted.trial;
    failed.status = PresentationStatus::skipped;
    failed.stimulus_id = emitted.stimulus_id;
    CHECK(writer.report_presentation(failed));
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "renderer failed") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(count_coded(records, "presentation-failed") == 1);
    CHECK(count_coded(records, "presentation-report-unmatched") == 0);
    bool matched = false;
    for (const auto& record : records)
    {
        if (record_name(record) == "speech.presentation_outcome")
        {
            matched = nested_contains(record, "matched_request", "true");
        }
    }
    CHECK(matched);
}

/// Drive one Speech run far enough to have emitted trial zero's CONTENT
/// request, then hand the caller what it needs to report against it.
struct SpeechRun
{
    speech::SpeechCueConfig config{speech_config()};
    speech::SpeechCatalog catalog{speech_catalog()};
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    SpeechHeadlessScheduler scheduler{};
};

void test_unmatched_presenter_report_changes_no_trial()
{
    RecorderFixture fixture;
    SpeechRun run;
    CHECK(speech::prepare_trial(run.config, 0, run.schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(run.config, 1, run.schedules[1]) == ContractStatus::ok);
    CHECK(run.scheduler.prepare(run.config, run.catalog, run.schedules, 64) == StreamStatus::ok);
    SpeechTraceWriter writer{run.scheduler, kSpeechParadigm};
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    static_cast<void>(session.note_step(run.scheduler.advance(1'150), 1'150));
    static_cast<void>(session.pump());
    EmittedContentRequest emitted{};
    CHECK(writer.emitted_content_request(0, emitted));

    // A report that names a legal trial ordinal and a request this run never
    // made. Under attribution by ordinal alone it would have invalidated trial
    // zero; a presenter would then be able to decide, by naming an ordinal,
    // that a trial the run never asked it about did not happen.
    PresentationOutcome stray{};
    stray.requested_ns = emitted.requested_ns;
    stray.sequence = 1;
    stray.request_sequence = emitted.sequence + 1000;
    stray.trial.ordinal = 0;
    stray.status = PresentationStatus::skipped;
    stray.stimulus_id = emitted.stimulus_id;
    CHECK(writer.report_presentation(stray));
    static_cast<void>(session.pump());

    for (ExperimentTimeNs time = 2'000; time <= 2'000 + 40'000'000; time += 1'000'000)
    {
        static_cast<void>(session.note_step(run.scheduler.advance(time), time));
        static_cast<void>(session.pump());
    }
    CHECK(session.stop(1'000 + 50'000'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    // Recorded, and nothing more. No trial was affected by a report the run
    // could not attribute.
    CHECK(outcome.abnormal_conditions >= 1);
    CHECK(outcome.abnormal_trials_affected == 0);
    CHECK(!outcome.abnormal_session_aborted);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(count_coded(records, "presentation-report-unmatched") == 1);
    CHECK(count_coded(records, "presentation-failed") == 0);
    bool saw_report = false;
    for (const auto& record : records)
    {
        if (body_contains(record, "speech.presentation_outcome"))
        {
            saw_report = true;
            // The report is kept -- a presenter saying something unexpected is
            // worth having -- and the record says it was not attributed.
            CHECK(nested_contains(record, "matched_request", "false"));
        }
        if (record.fields.control_kind == static_cast<std::uint32_t>(ProducerIdentityKind::trials))
        {
            CHECK(!nested_contains(record, "invalidated", "true"));
        }
    }
    CHECK(saw_report);
}

void test_writer_without_trial_tracking_does_not_start()
{
    RecorderFixture fixture;
    SpeechRun run;
    CHECK(speech::prepare_trial(run.config, 0, run.schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(run.config, 1, run.schedules[1]) == ContractStatus::ok);
    // Built first and prepared second: the one order in which the writer cannot
    // size itself from the schedule it is about to record.
    SpeechTraceWriter writer{run.scheduler, kSpeechParadigm};
    CHECK(run.scheduler.prepare(run.config, run.catalog, run.schedules, 64) == StreamStatus::ok);
    writer.require_presentation_evidence(true);
    CHECK(writer.validate() == ContractStatus::identity_missing);

    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    // Refused before anything is armed, because the alternative is silent: a
    // run with no evidence array records `presentation_evidence_required`
    // beside a count of zero trials missing evidence, having tracked none of
    // them -- and files every correct presenter report as unattributable,
    // since there is no request identity to match it against.
    CHECK(session.start(1'000) != StreamStatus::ok);

    // Given the chance to size itself, the same writer is one this session runs.
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    CHECK(writer.validate() == ContractStatus::ok);
}

void test_outcome_queue_cannot_be_replaced_when_live()
{
    RecorderFixture fixture;
    SpeechRun run;
    CHECK(speech::prepare_trial(run.config, 0, run.schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(run.config, 1, run.schedules[1]) == ContractStatus::ok);
    CHECK(run.scheduler.prepare(run.config, run.catalog, run.schedules, 64) == StreamStatus::ok);
    SpeechTraceWriter writer{run.scheduler, kSpeechParadigm};
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // Replacing the queue here would free it under a presenter thread that may
    // be pushing into it, and under the bridge thread draining it. The
    // precondition used to be a sentence in the header; it is now an answer.
    CHECK(writer.prepare_outcomes(32) == ContractStatus::already_running);

    static_cast<void>(session.note_step(run.scheduler.advance(1'150), 1'150));
    static_cast<void>(session.pump());
    EmittedContentRequest emitted{};
    CHECK(writer.emitted_content_request(0, emitted));

    // The queue that was installed is still the queue, and still takes reports.
    PresentationOutcome presented{};
    presented.requested_ns = emitted.requested_ns;
    presented.presented_ns = emitted.requested_ns + 1;
    presented.sequence = 1;
    presented.request_sequence = emitted.sequence;
    presented.trial = emitted.trial;
    presented.status = PresentationStatus::presented;
    presented.stimulus_id = emitted.stimulus_id;
    CHECK(writer.report_presentation(presented));
    static_cast<void>(session.pump());

    for (ExperimentTimeNs time = 2'000; time <= 2'000 + 40'000'000; time += 1'000'000)
    {
        static_cast<void>(session.note_step(run.scheduler.advance(time), time));
        static_cast<void>(session.pump());
    }
    CHECK(session.stop(1'000 + 50'000'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    // Once the run is over there is no producer left to race, and sizing is
    // legal again.
    CHECK(writer.prepare_outcomes(32) == ContractStatus::ok);
}

void check_presentation_failure_policy(AbnormalPolicy policy, bool expect_trial_affected,
                                       bool expect_session_aborted)
{
    RecorderFixture fixture;
    SpeechRun run;
    CHECK(speech::prepare_trial(run.config, 0, run.schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(run.config, 1, run.schedules[1]) == ContractStatus::ok);
    AbnormalPolicySet policies{};
    policies.presentation_failed = policy;
    CHECK(run.scheduler.prepare(run.config, run.catalog, run.schedules, 64, policies) ==
          StreamStatus::ok);
    SpeechTraceWriter writer{run.scheduler, kSpeechParadigm};
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    static_cast<void>(session.note_step(run.scheduler.advance(1'150), 1'150));
    static_cast<void>(session.pump());
    EmittedContentRequest emitted{};
    CHECK(writer.emitted_content_request(0, emitted));

    PresentationOutcome failed{};
    failed.requested_ns = emitted.requested_ns;
    failed.sequence = 1;
    failed.request_sequence = emitted.sequence;
    failed.trial.ordinal = 0;
    failed.status = PresentationStatus::skipped;
    failed.stimulus_id = emitted.stimulus_id;
    CHECK(writer.report_presentation(failed));
    static_cast<void>(session.pump());

    // The first advance after the report is where an aborting severity has to
    // show itself. `abort_session` means the run ended, not that a status
    // saying so was handed to whoever happened to be holding it: a
    // caller-stepped Speech run has no runtime to take the edge down, so a
    // scheduler that kept advancing would go on producing trials for a run that
    // had already declared itself unable to continue.
    const auto after_report = run.scheduler.advance(2'000);
    CHECK((after_report == StreamStatus::stopped) == expect_session_aborted);
    CHECK(run.scheduler.halted() == expect_session_aborted);
    static_cast<void>(session.note_step(after_report, 2'000));
    static_cast<void>(session.pump());

    for (ExperimentTimeNs time = 1'002'000; time <= 2'000 + 40'000'000; time += 1'000'000)
    {
        const auto stepped = run.scheduler.advance(time);
        // And it stays ended. Nothing about a later instant makes an aborted
        // run admissible again.
        CHECK(!expect_session_aborted || stepped == StreamStatus::stopped);
        static_cast<void>(session.note_step(stepped, time));
        static_cast<void>(session.pump());
    }
    static_cast<void>(session.stop(1'000 + 50'000'000, "done"));
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.abnormal_conditions >= 1);
    CHECK((outcome.abnormal_trials_affected >= 1) == expect_trial_affected);
    CHECK(outcome.abnormal_session_aborted == expect_session_aborted);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    // Exactly one record naming the condition, whatever the severity. The
    // session's reaction to an aborting one is a second fact and gets a code of
    // its own, so that counting `presentation-failed` counts failed
    // presentations rather than failed presentations plus reactions to them.
    CHECK(count_coded(records, "presentation-failed") == 1);
    CHECK(count_coded(records, "experiment-session-aborted") == (expect_session_aborted ? 1U : 0U));
    bool saw_invalidated = false;
    for (const auto& record : records)
    {
        if (record.fields.control_kind ==
                static_cast<std::uint32_t>(ProducerIdentityKind::trials) &&
            nested_contains(record, "invalidated", "true"))
        {
            saw_invalidated = true;
        }
    }
    // A trial can only be marked inadmissible while the run is still writing
    // trials, which an aborting policy stops it doing.
    CHECK(saw_invalidated == (expect_trial_affected && !expect_session_aborted));
}

void test_presentation_failure_follows_declared_severity()
{
    // The same presenter report, three configurations, three different answers
    // -- which is the whole claim a configurable severity makes. A run whose
    // claims do not rest on display timing records the failure and carries on;
    // one whose claims do gives up the trial; one that cannot proceed without a
    // presenter gives up the run.
    check_presentation_failure_policy(AbnormalPolicy::record, false, false);
    check_presentation_failure_policy(AbnormalPolicy::abort_trial, true, false);
    check_presentation_failure_policy(AbnormalPolicy::abort_session, false, true);
}

void test_expired_after_swap_keeps_swap_return_time()
{
    // A swap that crossed the deadline is correctly reported expired: the
    // semantic presented_ns is null (it was never a valid presentation). But the
    // presentation-layer swap-return time still physically happened, and the
    // presenter exists to measure exactly that. The outcome record keeps the software swap-return
    // observation as software_presented_ns, distinct from the semantic
    // presented_ns, so the measurement is not dropped at the recording boundary.
    RecorderFixture fixture;
    SpeechRun run;
    CHECK(speech::prepare_trial(run.config, 0, run.schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(run.config, 1, run.schedules[1]) == ContractStatus::ok);
    CHECK(run.scheduler.prepare(run.config, run.catalog, run.schedules, 64) == StreamStatus::ok);
    SpeechTraceWriter writer{run.scheduler, kSpeechParadigm};
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    static_cast<void>(session.note_step(run.scheduler.advance(1'150), 1'150));
    static_cast<void>(session.pump());
    EmittedContentRequest emitted{};
    CHECK(writer.emitted_content_request(0, emitted));

    constexpr ExperimentTimeNs kSwapReturnNs = 1'200;
    neurale::execution::SpeechPresentationEvidence evidence{};
    evidence.outcome.requested_ns = emitted.requested_ns;
    evidence.outcome.sequence = 1;
    evidence.outcome.request_sequence = emitted.sequence;
    evidence.outcome.trial.ordinal = 0;
    evidence.outcome.status = PresentationStatus::expired;
    evidence.outcome.presented_ns = 0;
    evidence.outcome.stimulus_id = emitted.stimulus_id;
    evidence.software.event = neurale::execution::PresentationLifecycleEvent::presented;
    evidence.software.time_ns = emitted.requested_ns;
    evidence.software.requested_ns = emitted.requested_ns;
    evidence.software.intended_ns = emitted.requested_ns;
    evidence.software.submitted_renderer_ns = 1'100;
    evidence.software.submitted_ns = 1'100;
    evidence.software.presented_renderer_ns = 1'190;
    evidence.software.presented_ns = kSwapReturnNs;
    evidence.software.implementation_status = 9;
    evidence.software.has_software_times = true;
    CHECK(writer.report_presentation(evidence));
    static_cast<void>(session.pump());

    for (ExperimentTimeNs time = 2'000; time <= 2'000 + 40'000'000; time += 1'000'000)
    {
        static_cast<void>(session.note_step(run.scheduler.advance(time), time));
        static_cast<void>(session.pump());
    }
    static_cast<void>(session.stop(1'000 + 50'000'000, "done"));
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    bool saw_outcome = false;
    for (const auto& record : records)
    {
        if (body_contains(record, "speech.presentation_outcome") &&
            nested_contains(record, "matched_request", "true"))
        {
            saw_outcome = true;
            // The semantic presentation never happened: presented_ns is null. The
            // body carries the inner JSON as an escaped string, so the check goes
            // through nested_contains (which builds the escaped form) rather than a
            // raw substring search.
            CHECK(nested_contains(record, "presented_ns", "null"));
            // ...but the software swap-return time the run measured is kept.
            std::uint64_t software_presented_ns = 0;
            CHECK(nested_u64(record, "software_presented_ns", software_presented_ns));
            CHECK(software_presented_ns == kSwapReturnNs);
        }
    }
    CHECK(saw_outcome);
}

void test_controller_reused_after_emergency_stop_starts_clean()
{
    // The abort latch and the primary condition are about the run executing,
    // not about the controller's lifetime. A controller that ran once and was
    // stopped from outside must not report the next, entirely clean, run as
    // aborted by the previous run's stop -- and unlike the counts, neither
    // field can be recovered by subtracting a baseline.
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    {
        RecorderFixture fixture;
        WebGridTraceWriter writer{controller, kWebGridParadigm};
        ExperimentSession session{writer, manual_session()};
        CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
        CHECK(session.start(1'000) == StreamStatus::ok);
        CHECK(session.emergency_stop(2'000, "operator") == StreamStatus::ok);
        CHECK(session.close() == RecorderStatusCode::ok);
        const auto stopped = session.outcome();
        CHECK(stopped.abnormal_session_aborted);
        CHECK(stopped.primary_abnormal == AbnormalCondition::emergency_stop);
    }

    CHECK(controller.reset() == StreamStatus::ok);

    RecorderFixture fixture;
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    run_webgrid(controller, config, session);
    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(!outcome.abnormal_session_aborted);
    CHECK(outcome.primary_abnormal == AbnormalCondition::unspecified);
    CHECK(outcome.abnormal_conditions == 0);
    CHECK(!outcome.terminal_abort);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::normal);
    CHECK(count_coded(records, "emergency-stop") == 0);
    CHECK(count_coded(records, "emergency-stop-requested") == 0);
}

void test_run_reports_missing_presentation_evidence()
{
    RecorderFixture fixture;
    const auto config = speech_config();
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);

    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 64) == StreamStatus::ok);
    SpeechTraceWriter writer{scheduler, kSpeechParadigm};
    CHECK(writer.prepare_outcomes(16) == ContractStatus::ok);
    writer.require_presentation_evidence(true);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    for (ExperimentTimeNs time = 1'000; time <= 1'000 + 40'000'000; time += 1'000'000)
    {
        static_cast<void>(session.note_step(scheduler.advance(time), time));
        static_cast<void>(session.pump());
    }
    CHECK(session.stop(1'000 + 50'000'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    // No presenter ever reported anything, and the run said it needed one to.
    // Reported once, for the run, and not resolved: there is no instant to
    // supply for a presentation nobody observed, and the intended onset is not
    // one -- it is the very thing a timing claim would be checking against.
    CHECK(count_coded(records, "presentation-evidence-missing") == 1);
    bool saw_summary = false;
    for (const auto& record : records)
    {
        if (body_contains(record, "speech.summary"))
        {
            saw_summary = true;
            CHECK(nested_contains(record, "presentation_evidence_required", "true"));
            CHECK(nested_contains(record, "presentation_reports", "0"));
        }
    }
    CHECK(saw_summary);
}

void test_identical_runs_record_same_outcome()
{
    // Determinism of the recorded verdict, not of the bytes: two runs of one
    // configuration, stepped identically, must agree about how many records
    // they produced, whether the trace was complete, and how they ended. A
    // layer whose outcome depended on scheduling would make every one of the
    // assertions above a coincidence.
    const auto run_once = [](SessionOutcome& outcome, std::size_t& n_records)
    {
        RecorderFixture fixture;
        const auto config = webgrid_config();
        WebGridHeadlessController controller{};
        CHECK(controller.prepare(config, 32) == StreamStatus::ok);
        WebGridTraceWriter writer{controller, kWebGridParadigm};
        ExperimentSession session{writer, manual_session()};
        CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
        CHECK(session.start(1'000) == StreamStatus::ok);
        run_webgrid(controller, config, session);
        CHECK(session.stop(5'000, "done") == StreamStatus::ok);
        CHECK(session.close() == RecorderStatusCode::ok);
        outcome = session.outcome();
        SpoolScanReport report;
        n_records = read_control(fixture.spool, report).size();
        CHECK(report.status() == ScanStatus::ok);
    };

    SessionOutcome first{};
    SessionOutcome second{};
    std::size_t first_records = 0;
    std::size_t second_records = 0;
    run_once(first, first_records);
    run_once(second, second_records);

    CHECK(first_records == second_records);
    CHECK(first.records_encoded == second.records_encoded);
    CHECK(first.records_recorded == second.records_recorded);
    CHECK(first.records_refused == second.records_refused);
    CHECK(first.trace_losses == second.trace_losses);
    CHECK(first.abnormal_conditions == second.abnormal_conditions);
    CHECK(first.abnormal_trials_affected == second.abnormal_trials_affected);
    CHECK(first.experiment_trace_complete == second.experiment_trace_complete);
    CHECK(first.terminal_abort == second.terminal_abort);
    CHECK(first.state == second.state);
    CHECK(!first.has_runtime_fault);
    CHECK(first.primary_abnormal == AbnormalCondition::unspecified);
}
void test_gap_crossed_webgrid_target_records_reason()
{
    RecorderFixture fixture;
    WebGridControllerConfig options{};
    options.task = webgrid_config();
    options.trace_capacity = 64;
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(options) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // The pointer source stopped arriving. Only the caller can know that, so it
    // is reported to the task rather than detected by it.
    CHECK(session.note_step(controller.note_input_gap(1'100, 42), 1'100) == StreamStatus::ok);
    static_cast<void>(session.pump());

    // The target still completes, and completes correctly. What is no longer
    // true is that the interval it completed over is a measurement.
    {
        const auto snapshot = controller.snapshot();
        const webgrid::PointerPosition pointer{0.5, 0.5};
        SelectionEvent selection{};
        CHECK(webgrid::make_selection_event(options.task, pointer, snapshot.active_target,
                                            snapshot.trial, kWebGridParadigm, 1'200, 0,
                                            selection) == ContractStatus::ok);
        CHECK(session.note_step(controller.process(1'200, pointer, selection), 1'200) ==
              StreamStatus::ok);
        static_cast<void>(session.pump());
    }
    // The second target is clean.
    {
        const auto snapshot = controller.snapshot();
        const webgrid::PointerPosition pointer{1.5, 0.5};
        SelectionEvent selection{};
        CHECK(webgrid::make_selection_event(options.task, pointer, snapshot.active_target,
                                            snapshot.trial, kWebGridParadigm, 1'400, 1,
                                            selection) == ContractStatus::ok);
        CHECK(session.note_step(controller.process(1'400, pointer, selection), 1'400) ==
              StreamStatus::ok);
        static_cast<void>(session.pump());
    }

    CHECK(session.stop(5'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.abnormal_conditions == 1);
    CHECK(outcome.abnormal_trials_affected == 1);
    CHECK(outcome.primary_abnormal == AbnormalCondition::input_gap);
    // The run was not ended: `abort_trial` is about the trial, not the session.
    CHECK(!outcome.abnormal_session_aborted);
    CHECK(!outcome.terminal_abort);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(report.requested_terminal_intent() == RequestedTerminalIntent::normal);
    CHECK(count_coded(records, "input-gap") == 1);
    CHECK(count_coded_at(records, "input-gap", 1'100) == 1);

    std::size_t aborted = 0;
    std::size_t admitted = 0;
    for (const auto& record : records)
    {
        if (record.fields.control_kind != static_cast<std::uint32_t>(ProducerIdentityKind::trials))
        {
            continue;
        }
        // TrialOutcome::aborted is 4, success is 1. The machine decided both of
        // these targets were selected correctly; the run overrode the first
        // because the interval it was selected over is not a number anyone may
        // quote, and kept the machine's own verdict beside it.
        if (nested_contains(record, "acquisition_timing_valid", "false"))
        {
            ++aborted;
            CHECK(nested_contains(record, "outcome", "4"));
            CHECK(nested_contains(record, "task_outcome", "1"));
        }
        else
        {
            ++admitted;
            CHECK(nested_contains(record, "outcome", "1"));
        }
    }
    CHECK(aborted == 1);
    CHECK(admitted == 1);

    // And the raw selection is still there, marked rather than deleted -- which
    // is what makes recomputing the metric offline from records possible.
    std::size_t marked_selections = 0;
    for (const auto& record : records)
    {
        if (record_name(record) == "webgrid.selection_correct" &&
            nested_contains(record, "acquisition_timing_valid", "false"))
        {
            ++marked_selections;
        }
    }
    CHECK(marked_selections == 1);

    // The summary states that the machine's own metric still counts it, rather
    // than quietly recomputing a formula the machine owns.
    bool saw_summary = false;
    for (const auto& record : records)
    {
        if (record_name(record) == "webgrid.metrics")
        {
            saw_summary = true;
            CHECK(nested_contains(record, "invalidated_targets", "1"));
            CHECK(nested_contains(record, "metric_includes_invalidated", "true"));
        }
    }
    CHECK(saw_summary);
}

void test_webgrid_input_provenance_reaches_selection_record()
{
    RecorderFixture fixture;
    const auto config = webgrid_config();
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config, 32) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    writer.require_presentation_input_evidence(true);
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    static_cast<void>(session.pump());

    const auto snapshot = controller.snapshot();
    webgrid::GridCell cell{};
    CHECK(webgrid::grid_cell(config, snapshot.active_target, cell) == ContractStatus::ok);
    const webgrid::PointerPosition pointer{(cell.bounds.min_x + cell.bounds.max_x) * 0.5,
                                           (cell.bounds.min_y + cell.bounds.max_y) * 0.5};
    SelectionEvent selection{};
    CHECK(webgrid::make_selection_event(config, pointer, snapshot.active_target, snapshot.trial,
                                        kWebGridParadigm, 1'100, 7,
                                        selection) == ContractStatus::ok);
    WebGridPresentationInputEvidence input{};
    input.input_ordinal = 17;
    input.renderer_time_ns = 90'000;
    input.experiment_time_ns = 1'100;
    input.kind = 1;
    input.button = 0;
    input.modifiers = 2;
    input.inside_presentation = true;
    input.available = true;
    CHECK(session.note_step(controller.process(1'100, pointer, selection, input), 1'100) ==
          StreamStatus::ok);
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    bool pointer_recorded = false;
    bool selection_recorded = false;
    for (const auto& record : records)
    {
        if (record_name(record) == "webgrid.pointer")
        {
            pointer_recorded = true;
            CHECK(nested_contains(record, "presentation_input_available", "true"));
            CHECK(nested_contains(record, "presentation_input_ordinal", "17"));
            CHECK(nested_contains(record, "renderer_time_ns", "90000"));
            CHECK(nested_contains(record, "presentation_input_kind", "1"));
        }
        if (record_name(record) == "webgrid.selection_correct")
        {
            selection_recorded = true;
            // The selection points at this pointer row through its existing
            // pointer_update_ordinal; presentation evidence is not duplicated here.
            CHECK(nested_contains(record, "pointer_update_ordinal", "0"));
        }
    }
    CHECK(pointer_recorded);
    CHECK(selection_recorded);
}

void test_center_out_presentation_lifecycle_is_bounded()
{
    RecorderFixture fixture;
    CenterOutController controller{};
    CHECK(controller.prepare(decoded_schema(), center_out_config()) == StreamStatus::ok);
    CenterOutTraceWriter writer{controller, 5'000};
    CHECK(writer.set_host_epoch(5'000));
    constexpr std::string_view fingerprint{
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"};
    CHECK(!writer.report_decoder_publication(1, "too-short", 0, 0, 0, 1'000));
    CHECK(writer.report_decoder_publication(1, fingerprint, 0, 0, 0, 1'000));
    writer.require_presentation_evidence(true);
    CenterOutPresentationConfigRecord presentation_config{};
    presentation_config.geometry_unit = 1;
    presentation_config.logical_left = -1.0;
    presentation_config.logical_bottom = -1.0;
    presentation_config.logical_width = 2.0;
    presentation_config.logical_height = 2.0;
    presentation_config.target_radius = 0.05;
    presentation_config.cursor_radius = 0.025;
    presentation_config.circle_segments = 48;
    presentation_config.aspect_policy = 0;
    presentation_config.window_width = 800;
    presentation_config.window_height = 600;
    presentation_config.monitor_idx = -1;
    presentation_config.swap_interval = 1;
    presentation_config.resizable = true;
    presentation_config.visible = true;
    CHECK(writer.report_presentation_config(presentation_config));
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(!writer.set_host_epoch(6'000));
    CHECK(!writer.report_presentation_config(presentation_config));

    FrameStorage storage{};
    const std::array decoded{0.0, 0.0};
    CHECK(feed(controller, decoded_frame(storage, decoded, 1, 1, 0, 5'010)) == StreamStatus::ok);
    SequenceOrdinal source_ordinal{};
    CHECK(controller.latest_observation_ordinal(source_ordinal));
    const auto presented_trial = controller.snapshot().trial;
    static_cast<void>(session.pump());

    PresentationSoftwareEvidence presented{};
    presented.event = PresentationLifecycleEvent::presented;
    presented.time_ns = 1'020;
    presented.requested_ns = 1'000;
    presented.intended_ns = 1'010;
    presented.submitted_renderer_ns = 100;
    presented.submitted_ns = 1'015;
    presented.presented_renderer_ns = 105;
    presented.presented_ns = 1'020;
    presented.update_ordinal = 3;
    presented.source_ordinal = source_ordinal;
    presented.trial = presented_trial;
    presented.has_source_ordinal = true;
    presented.has_trial = true;
    presented.has_software_times = true;
    CHECK(writer.report_presentation(presented));
    static_cast<void>(session.pump());

    PresentationSoftwareEvidence closed{};
    closed.event = PresentationLifecycleEvent::closed;
    closed.time_ns = 1'025;
    CHECK(writer.report_presentation(closed));
    static_cast<void>(session.pump());

    PresentationSoftwareEvidence failed{};
    failed.event = PresentationLifecycleEvent::faulted;
    failed.time_ns = 1'030;
    failed.implementation_status = 9;
    // A runtime surface fault carries the trial identity captured when it
    // occurred; task-owner delivery rejects a stale identity rather than
    // transferring the fault to a later trial.
    failed.trial = controller.snapshot().trial;
    failed.has_trial = true;
    CHECK(writer.report_presentation(failed));
    CHECK(controller.enqueue_presentation_failure(
        PresentationFailureEvidence{failed.time_ns, failed.trial, failed.implementation_status}));
    CHECK(controller.apply_presentation_failures() == StreamStatus::ok);
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    bool saw_presented = false;
    bool saw_closed = false;
    bool saw_faulted = false;
    bool saw_presentation_config = false;
    bool saw_presentation_window = false;
    bool saw_presentation_style_1 = false;
    bool saw_presentation_style_2 = false;
    bool saw_decoder_publication = false;
    for (const auto& record : records)
    {
        if (record_name(record) == "center_out.presentation_config")
        {
            saw_presentation_config = nested_contains(record, "geometry_unit", "1") &&
                                      nested_contains(record, "target_radius", "0.05") &&
                                      nested_contains(record, "cursor_radius", "0.025");
        }
        saw_presentation_window =
            saw_presentation_window || record_name(record) == "center_out.presentation_window";
        saw_presentation_style_1 =
            saw_presentation_style_1 || record_name(record) == "center_out.presentation_style_1";
        saw_presentation_style_2 =
            saw_presentation_style_2 || record_name(record) == "center_out.presentation_style_2";
        if (record_name(record) == "center_out.decoder_publication")
        {
            saw_decoder_publication =
                nested_contains(record, "version", "1") &&
                nested_contains(record, "plan_fingerprint",
                                std::string{"\\\""}.append(fingerprint).append("\\\"")) &&
                nested_contains(record, "training_blocks", "0") &&
                nested_contains(record, "completed_trials", "0");
        }
        if (record_name(record) == "center_out.presentation")
        {
            if (nested_contains(record, "event", "2"))
            {
                saw_presented = true;
                CHECK(nested_contains(record, "requested_ns", "1000"));
                CHECK(nested_contains(record, "intended_ns", "1010"));
                CHECK(nested_contains(record, "presented_ns", "1020"));
                // The controller's source-identity join key is recorded alongside the
                // presenter-local update_ordinal, so an offline reader can join
                // this presentation frame to the controller observation it depicted.
                std::uint64_t source_ordinal = 0;
                CHECK(nested_u64(record, "source_ordinal", source_ordinal));
                CHECK(source_ordinal == 0);
            }
            saw_closed = saw_closed || nested_contains(record, "event", "4");
            // A faulted frame is still implementation evidence (event=faulted) and
            // is separate from the abnormal condition row below.
            saw_faulted = saw_faulted || nested_contains(record, "event", "5");
        }
    }
    CHECK(saw_presented);
    CHECK(saw_closed);
    CHECK(saw_faulted);
    CHECK(saw_presentation_config);
    CHECK(saw_presentation_window);
    CHECK(saw_presentation_style_1);
    CHECK(saw_presentation_style_2);
    CHECK(saw_decoder_publication);
    // The task-relevant failure is one abnormal condition row, not a generic
    // fault row, so it is counted once under its contract name.
    CHECK(count_coded(records, "presentation-failed") == 1);
    // The default abort_trial policy acts on the active trial; it does not
    // silently degrade to recorded merely because this was a surface-runtime
    // fault rather than one render(frame) result.
    CHECK(session.outcome().primary_abnormal == AbnormalCondition::presentation_failed);
    CHECK(session.outcome().abnormal_trials_affected == 1);
}

void test_speech_runtime_failure_invalidates_active_trial()
{
    RecorderFixture fixture;
    const auto config = speech_config();
    AbnormalPolicySet abnormal{};
    abnormal.presentation_failed = AbnormalPolicy::abort_trial;
    const auto catalog = speech_catalog();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    CHECK(speech::prepare_trial(config, 0, schedules[0]) == ContractStatus::ok);
    CHECK(speech::prepare_trial(config, 1, schedules[1]) == ContractStatus::ok);
    SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(config, catalog, schedules, 64, abnormal) == StreamStatus::ok);
    SpeechTraceWriter writer{scheduler, kSpeechParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    // Cross the trial boundary without letting the writer drain either trial.
    // Runtime attribution must follow the scheduler's producer-side current
    // trial rather than the writer's last consumed snapshot.
    const auto first_end = scheduler.snapshot().timeline.trial.end_ns;
    CHECK(scheduler.advance(first_end) == StreamStatus::ok);
    CHECK(scheduler.snapshot().trial.ordinal == 1);

    PresentationSoftwareEvidence fault{};
    fault.event = PresentationLifecycleEvent::faulted;
    fault.time_ns = first_end;
    fault.implementation_status = 9;
    fault.trial = scheduler.snapshot().trial;
    fault.has_trial = true;
    fault.policy_delegated = true;
    CHECK(scheduler.enqueue_presentation_failure(
        {fault.time_ns, fault.trial, fault.implementation_status}));
    CHECK(writer.report_lifecycle(fault));
    auto mismatched = fault;
    mismatched.time_ns = first_end + 1;
    ++mismatched.trial.ordinal;
    CHECK(scheduler.enqueue_presentation_failure(
        {mismatched.time_ns, mismatched.trial, mismatched.implementation_status}));
    CHECK(writer.report_lifecycle(mismatched));
    CHECK(scheduler.apply_presentation_failures() == StreamStatus::invalid_frame);
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "presentation fault") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.primary_abnormal == AbnormalCondition::presentation_failed);
    CHECK(outcome.abnormal_trials_affected == 1);
    CHECK(!outcome.abnormal_session_aborted);
    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(count_named(records, "speech.presentation_lifecycle") == 2);
    CHECK(count_coded(records, "presentation-failed") == 1);
    CHECK(count_coded(records, "presentation-report-unmatched") == 0);
    CHECK(scheduler.dropped_trace_count() == 1);
}

void test_webgrid_presentation_failure_is_bounded()
{
    // WebGrid mirrors Center-Out: a faulted presentation frame is implementation
    // evidence plus an abnormal condition row, and the failure reaches the
    // existing abnormal policy rather than a standalone generic fault row.
    RecorderFixture fixture;
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(webgrid_config(), 8) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    WebGridPresentationConfigRecord presentation_config{};
    presentation_config.logical_min_x = 0.0;
    presentation_config.logical_max_x = 2.0;
    presentation_config.logical_min_y = 0.0;
    presentation_config.logical_max_y = 1.0;
    presentation_config.pointer_radius = 0.03;
    presentation_config.circle_segments = 32;
    presentation_config.selection_button = 0;
    presentation_config.aspect_policy = 0;
    presentation_config.window_width = 800;
    presentation_config.window_height = 600;
    presentation_config.monitor_idx = -1;
    presentation_config.swap_interval = 1;
    presentation_config.resizable = true;
    presentation_config.visible = true;
    CHECK(writer.report_presentation_config(presentation_config));
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);
    CHECK(!writer.report_presentation_config(presentation_config));

    const auto initial = controller.snapshot();
    CHECK(controller.process(1'010, {0.25, 0.25}) == StreamStatus::ok);
    SequenceOrdinal source_ordinal{};
    CHECK(controller.latest_pointer_update_ordinal(source_ordinal));
    static_cast<void>(session.pump());

    PresentationSoftwareEvidence presented{};
    presented.event = PresentationLifecycleEvent::presented;
    presented.time_ns = 1'020;
    presented.requested_ns = 1'000;
    presented.intended_ns = 1'010;
    presented.submitted_renderer_ns = 100;
    presented.submitted_ns = 1'015;
    presented.presented_renderer_ns = 105;
    presented.presented_ns = 1'020;
    presented.update_ordinal = 3;
    presented.source_ordinal = source_ordinal;
    presented.trial = initial.trial;
    presented.has_source_ordinal = true;
    presented.has_trial = true;
    presented.has_software_times = true;
    CHECK(writer.report_presentation(presented));
    static_cast<void>(session.pump());

    PresentationSoftwareEvidence failed{};
    failed.event = PresentationLifecycleEvent::faulted;
    failed.time_ns = 1'030;
    failed.implementation_status = 9;
    failed.trial = controller.snapshot().trial;
    failed.has_trial = true;
    CHECK(writer.report_presentation(failed));
    CHECK(controller.enqueue_presentation_failure(
        PresentationFailureEvidence{failed.time_ns, failed.trial, failed.implementation_status}));
    CHECK(controller.apply_presentation_failures() == StreamStatus::ok);
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    bool saw_presented = false;
    bool saw_faulted = false;
    bool saw_presentation_config = false;
    bool saw_presentation_window = false;
    bool saw_presentation_style_1 = false;
    bool saw_presentation_style_2 = false;
    for (const auto& record : records)
    {
        if (record_name(record) == "webgrid.presentation_config")
        {
            saw_presentation_config = nested_contains(record, "logical_max_x", "2") &&
                                      nested_contains(record, "logical_max_y", "1") &&
                                      nested_contains(record, "pointer_radius", "0.03");
        }
        saw_presentation_window =
            saw_presentation_window || record_name(record) == "webgrid.presentation_window";
        saw_presentation_style_1 =
            saw_presentation_style_1 || record_name(record) == "webgrid.presentation_style_1";
        saw_presentation_style_2 =
            saw_presentation_style_2 || record_name(record) == "webgrid.presentation_style_2";
        if (record_name(record) == "webgrid.presentation")
        {
            saw_presented = saw_presented || nested_contains(record, "event", "2");
            saw_faulted = saw_faulted || nested_contains(record, "event", "5");
        }
    }
    CHECK(saw_presented);
    CHECK(saw_faulted);
    CHECK(saw_presentation_config);
    CHECK(saw_presentation_window);
    CHECK(saw_presentation_style_1);
    CHECK(saw_presentation_style_2);
    CHECK(count_coded(records, "presentation-failed") == 1);
    CHECK(session.outcome().primary_abnormal == AbnormalCondition::presentation_failed);
    CHECK(session.outcome().abnormal_trials_affected == 1);
}

void check_center_out_presentation_failure_policy(AbnormalPolicy policy, bool expect_trial_affected,
                                                  bool expect_session_aborted)
{
    RecorderFixture fixture;
    auto config = center_out_config();
    config.abnormal.presentation_failed = policy;
    CenterOutController controller{};
    CHECK(controller.prepare(decoded_schema(), config) == StreamStatus::ok);
    CenterOutTraceWriter writer{controller, 5'000};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    PresentationSoftwareEvidence failed{};
    failed.event = PresentationLifecycleEvent::faulted;
    failed.time_ns = 1'030;
    failed.implementation_status = 9;
    failed.trial = controller.snapshot().trial;
    failed.has_trial = true;
    CHECK(writer.report_presentation(failed));
    CHECK(controller.enqueue_presentation_failure(
        PresentationFailureEvidence{failed.time_ns, failed.trial, failed.implementation_status}));
    const auto apply_status = controller.apply_presentation_failures();
    CHECK(apply_status ==
          (expect_session_aborted ? StreamStatus::consumer_failure : StreamStatus::ok));
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.abnormal_conditions >= 1);
    CHECK(outcome.primary_abnormal == AbnormalCondition::presentation_failed);
    CHECK((outcome.abnormal_trials_affected != 0) == expect_trial_affected);
    CHECK(outcome.abnormal_session_aborted == expect_session_aborted);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(count_coded(records, "presentation-failed") == 1);
}

void check_webgrid_presentation_failure_policy(AbnormalPolicy policy, bool expect_trial_affected,
                                               bool expect_session_aborted)
{
    RecorderFixture fixture;
    WebGridControllerConfig config{};
    config.task = webgrid_config();
    config.abnormal.presentation_failed = policy;
    config.trace_capacity = 8;
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(config) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    PresentationSoftwareEvidence failed{};
    failed.event = PresentationLifecycleEvent::faulted;
    failed.time_ns = 1'030;
    failed.implementation_status = 9;
    failed.trial = controller.snapshot().trial;
    failed.has_trial = true;
    CHECK(writer.report_presentation(failed));
    CHECK(controller.enqueue_presentation_failure(
        PresentationFailureEvidence{failed.time_ns, failed.trial, failed.implementation_status}));
    const auto apply_status = controller.apply_presentation_failures();
    CHECK(apply_status ==
          (expect_session_aborted ? StreamStatus::consumer_failure : StreamStatus::ok));
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "done") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(outcome.abnormal_conditions >= 1);
    CHECK(outcome.primary_abnormal == AbnormalCondition::presentation_failed);
    CHECK((outcome.abnormal_trials_affected != 0) == expect_trial_affected);
    CHECK(outcome.abnormal_session_aborted == expect_session_aborted);

    SpoolScanReport report;
    const auto records = read_control(fixture.spool, report);
    CHECK(report.status() == ScanStatus::ok);
    CHECK(count_coded(records, "presentation-failed") == 1);
}

void test_center_out_and_webgrid_failure_follow_severity()
{
    // Frame failures carry the exact active trial to a bounded task-owner
    // handoff. record leaves it admissible; abort_trial truly aborts Center-Out
    // and invalidates WebGrid; abort_session ends the run.
    check_center_out_presentation_failure_policy(AbnormalPolicy::record, false, false);
    check_center_out_presentation_failure_policy(AbnormalPolicy::abort_trial, true, false);
    check_center_out_presentation_failure_policy(AbnormalPolicy::abort_session, false, true);
    check_webgrid_presentation_failure_policy(AbnormalPolicy::record, false, false);
    check_webgrid_presentation_failure_policy(AbnormalPolicy::abort_trial, true, false);
    check_webgrid_presentation_failure_policy(AbnormalPolicy::abort_session, false, true);
}

void test_presentation_queue_saturation_is_trace_loss()
{
    RecorderFixture fixture;
    WebGridHeadlessController controller{};
    CHECK(controller.prepare(webgrid_config(), 8) == StreamStatus::ok);
    WebGridTraceWriter writer{controller, kWebGridParadigm};
    ExperimentSession session{writer, manual_session()};
    CHECK(session.enable_recording(fixture.attachment()) == StreamStatus::ok);
    CHECK(session.start(1'000) == StreamStatus::ok);

    PresentationSoftwareEvidence evidence{};
    evidence.event = PresentationLifecycleEvent::opened;
    evidence.time_ns = 1'001;
    for (std::size_t i = 0; i < 64; ++i)
    {
        evidence.update_ordinal = i;
        CHECK(writer.report_presentation(evidence));
    }
    CHECK(!writer.report_presentation(evidence));
    static_cast<void>(session.pump());
    CHECK(session.stop(2'000, "saturated") == StreamStatus::ok);
    CHECK(session.close() == RecorderStatusCode::ok);

    const auto outcome = session.outcome();
    CHECK(!outcome.experiment_trace_complete);
    CHECK(outcome.producer_trace_drops == 1);
    CHECK(outcome.trace_losses == 1);
    CHECK(outcome.terminal_abort);
}
} // namespace

int main()
{
    test_recorded_webgrid_session_writes_metadata_first();
    test_recorder_changes_only_what_is_kept();
    test_speech_session_keeps_intent_apart_from_report();
    test_saturated_trace_queue_is_never_complete();
    test_fault_policy_escalates_loss();
    test_stop_abort_and_close_are_idempotent();
    test_refused_readiness_gate_starts_nothing();
    test_invalid_paradigm_configuration_stops_before_recorder();
    test_unprepared_paradigm_is_rejected_before_arming();
    test_session_start_starts_paradigm();
    test_failed_start_leaves_no_paradigm_running();
    test_speech_schedule_must_cover_configuration();
    test_two_realizations_record_differently();
    test_interrupted_session_leaves_recoverable_spool();
    test_center_out_session_records_velocity_provenance();
    test_center_out_records_every_segment_start();
    test_center_out_records_configuration_sampler();
    test_loss_in_final_drain_ends_recording_as_abort();
    test_oversized_summary_is_a_loss();
    test_unseen_producer_drop_is_still_a_loss();
    test_recorded_provenance_matches_running_configuration();
    test_oversized_body_is_rejected_not_truncated();
    test_second_session_does_not_inherit_drops();
    test_runtime_needs_session_with_drain();
    test_failed_start_reinhibits_armed_runtime();
    test_opening_trace_leaves_producer_before_start_returns();
    test_runtime_never_produces_into_undrained_queue();
    test_undrainable_session_is_rejected_early();
    test_throwing_bring_up_unwinds_completely();
    test_destroying_running_session_ends_experiment();
    test_safety_is_inhibited_outside_armed_window();
    test_consumer_fault_survives_shutdown();
    test_paradigm_demanded_end_is_an_abort();
    test_emergency_stop_halts_task_first();
    test_emergency_stop_releases_nothing();
    test_absent_presentation_is_not_completed_trial();
    test_black_renderer_failure_names_its_request();
    test_run_reports_missing_presentation_evidence();
    test_unmatched_presenter_report_changes_no_trial();
    test_writer_without_trial_tracking_does_not_start();
    test_outcome_queue_cannot_be_replaced_when_live();
    test_presentation_failure_follows_declared_severity();
    test_expired_after_swap_keeps_swap_return_time();
    test_controller_reused_after_emergency_stop_starts_clean();
    test_gap_crossed_webgrid_target_records_reason();
    test_webgrid_input_provenance_reaches_selection_record();
    test_center_out_presentation_lifecycle_is_bounded();
    test_speech_runtime_failure_invalidates_active_trial();
    test_webgrid_presentation_failure_is_bounded();
    test_center_out_and_webgrid_failure_follow_severity();
    test_presentation_queue_saturation_is_trace_loss();
    test_identical_runs_record_same_outcome();
    return failures == 0 ? 0 : 1;
}
