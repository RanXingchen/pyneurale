/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/center_out_replay.h>
#include <neurale/experiments/replay.h>
#include <neurale/experiments/speech_replay.h>
#include <neurale/experiments/webgrid_replay.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "check_counts.h"

namespace
{

using namespace neurale::experiments;

int failures = 0;

constexpr ParadigmId kCenterOutParadigm = 41;
constexpr ParadigmId kWebGridParadigm = 52;
constexpr ParadigmId kSpeechParadigm = 63;

// --- configurations ---------------------------------------------------------

center_out::CenterOut2DConfig center_out_task() noexcept
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

center_out::CenterOutReplayConfig center_out_config() noexcept
{
    center_out::CenterOutReplayConfig config{};
    config.paradigm = kCenterOutParadigm;
    config.task = center_out_task();
    config.guidance = center_out::CenterOutGuidanceConfig{center_out::GeometryUnit::normalized, 1.0,
                                                          4.0, 4.0, 0.05};
    config.velocity_space = velocity_space();
    config.linear_assistance = assistance::LinearAssistance{0.25};
    config.assistance_method = assistance::AssistanceMethod::linear_blend;
    config.initial_position = {0.0, 0.0};
    config.cursor_min = {-1.0, -1.0};
    config.cursor_max = {1.0, 1.0};
    return config;
}

webgrid::WebGridReplayConfig webgrid_config() noexcept
{
    webgrid::WebGridConfig task{};
    task.rows = 1;
    task.columns = 2;
    task.bounds = webgrid::TaskBounds{0.0, 2.0, 0.0, 1.0};
    task.n_candidates = 2;
    task.candidates[0] = 1;
    task.candidates[1] = 2;
    task.schedule = webgrid::TargetScheduleKind::explicit_sequence;
    task.immediate_repetition = webgrid::ImmediateRepetitionPolicy::forbid;
    task.correct_selection = webgrid::CorrectSelectionPolicy::advance_target;
    task.incorrect_selection = webgrid::IncorrectSelectionPolicy::keep_current_target;
    task.n_explicit = 2;
    task.explicit_targets[0] = 1;
    task.explicit_targets[1] = 2;
    task.initial_target = 1;
    task.target_count_limit = 2;
    task.metric_version = webgrid::kMetricVersion1;

    webgrid::WebGridReplayConfig config{};
    config.paradigm = kWebGridParadigm;
    config.task = task;
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

speech::SpeechReplayConfig speech_config() noexcept
{
    speech::SpeechCueConfig task{};
    task.black_bound_ns = 100;
    task.cross_bound_ns = 0;
    task.content_bound_ns = 200;
    task.inter_trial_ns = 0;
    task.seed = 99;
    task.n_trials = 2;
    task.schedule = speech::SpeechScheduleKind::seeded;
    task.stimulus_order = speech::StimulusOrderPolicy::sequential;
    task.sampler_version = kCurrentSamplerVersion;
    task.cross_enabled = false;
    task.n_stimuli = 2;
    task.stimuli[0] = 1;
    task.stimuli[1] = 2;

    speech::SpeechReplayConfig config{};
    config.paradigm = kSpeechParadigm;
    config.task = task;
    config.catalog = speech_catalog();
    return config;
}

speech::SpeechReplayConfig explicit_speech_config() noexcept
{
    auto config = speech_config();
    std::array<speech::SpeechTrialSchedule, 2> schedules{};
    for (TrialOrdinal ordinal = 0; ordinal < schedules.size(); ++ordinal)
    {
        CHECK(speech::prepare_trial(config.task, ordinal, schedules[ordinal]) ==
              ContractStatus::ok);
    }
    config.task.schedule = speech::SpeechScheduleKind::explicit_sequence;
    config.task.stimulus_order = speech::StimulusOrderPolicy::unspecified;
    config.task.n_stimuli = 0;
    config.task.stimuli = {};
    config.task.n_explicit = static_cast<TrialOrdinal>(schedules.size());
    for (std::size_t i = 0; i < schedules.size(); ++i)
    {
        config.task.explicit_schedule[i] = schedules[i];
    }
    return config;
}

// --- recorders --------------------------------------------------------------
//
// A recorder is what a live run and its recording bridge together produce: it
// drives the paradigm's own machine and collects the streams a reader would
// recover. It is deliberately not the replay engine -- the whole point is that
// two independently written drivers of the same contract agree -- and it is
// where a test mutates a recording to see the replay notice.

struct CenterOutRecording
{
    ExperimentTimeNs origin_ns{};
    std::vector<center_out::CenterOutReplayInput> inputs{};
    std::vector<StateTransition> transitions{};
    std::vector<ExperimentEvent> events{};
    std::vector<center_out::CenterOutTrial> trials{};
    std::vector<center_out::CenterOutReplayAbort> aborts{};
    std::vector<center_out::CenterOutTargetOnset> targets{};
    std::vector<center_out::CenterOutCursorSample> cursor{};
    std::vector<center_out::CenterOutVelocitySample> vel{};
    std::vector<center_out::CenterOutGuidanceObservation> guidance{};

    [[nodiscard]] center_out::CenterOutReplayRecording view(const ReplayProvenance& provenance,
                                                            bool complete = true) const noexcept
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
        recording.expected.cursor = cursor;
        recording.expected.vel = vel;
        recording.expected.guidance = guidance;
        recording.expected.has_transitions = complete;
        recording.expected.has_events = complete;
        recording.expected.has_trials = complete;
        recording.expected.has_aborts = complete;
        recording.expected.has_targets = complete;
        recording.expected.has_cursor = complete;
        recording.expected.has_velocity = complete;
        recording.expected.has_guidance = complete;
        recording.run_end_recorded = true;
        return recording;
    }
};

/// Drives the Center-Out chain the way a live run does, and keeps everything.
class CenterOutRecorder
{
  public:
    explicit CenterOutRecorder(const center_out::CenterOutReplayConfig& config) : config_(config)
    {
        CHECK(guidance_.configure(config.guidance) == ContractStatus::ok);
    }

    void start(ExperimentTimeNs origin_ns)
    {
        recording_.origin_ns = origin_ns;
        position_ = config_.initial_position;
        last_time_ns_ = origin_ns;
        has_last_time_ = true;
        running_ = true;
        center_out::CenterOutStepResult opening{};
        CHECK(begin_segment(origin_ns, opening) == ContractStatus::ok);
        keep_step(opening);
        keep_target(opening.snapshot, origin_ns);
    }

    void observe(ExperimentTimeNs time_ns, double vx, double vy, std::uint64_t frame,
                 std::uint64_t sample)
    {
        center_out::CenterOutReplayInput input{};
        input.kind = center_out::CenterOutReplayInputKind::observation;
        input.time_ns = time_ns;
        input.decoded = {vx, vy};
        input.frame_sequence = frame;
        input.sample_idx = sample;
        recording_.inputs.push_back(input);
        if (machine_.complete() || terminal_)
        {
            return;
        }

        if (pending_restart_)
        {
            center_out::CenterOutStepResult restart{};
            CHECK(begin_segment(time_ns, restart) == ContractStatus::ok);
            last_time_ns_ = time_ns;
            has_last_time_ = true;
            pending_restart_ = false;
            forget_target();
            keep_step(restart);
            keep_target(restart.snapshot, time_ns);
        }

        const DurationNs dt_ns = has_last_time_ ? time_ns - last_time_ns_ : 0;
        const auto before = position_;
        assistance::VelocityVector decoded{};
        decoded.space = config_.velocity_space.id;
        decoded.dim = 2;
        decoded.values[0] = vx;
        decoded.values[1] = vy;

        assistance::VelocityVector reference{};
        reference.space = config_.velocity_space.id;
        reference.dim = 2;
        center_out::CenterOutGuidanceSample sample_out{};
        const auto snapshot_before = machine_.snapshot();
        if (snapshot_before.active_target != kUnsetTargetId)
        {
            const center_out::TargetPlacement target{snapshot_before.active_target,
                                                     snapshot_before.active_position};
            CHECK(guidance_.update(target, position_, dt_ns, sample_out) == ContractStatus::ok);
            reference.values[0] = sample_out.vel.x;
            reference.values[1] = sample_out.vel.y;
        }
        else
        {
            guidance_.reset();
        }

        auto assisted = decoded;
        if (config_.assistance_method == assistance::AssistanceMethod::linear_blend)
        {
            CHECK(assistance::blend_velocity(config_.velocity_space, decoded, reference,
                                             config_.linear_assistance,
                                             assisted) == ContractStatus::ok);
        }
        const auto seconds =
            static_cast<double>(dt_ns) / static_cast<double>(kNanosecondsPerSecond);
        position_ =
            center_out::WorkspacePoint{std::clamp(position_.x + assisted.values[0] * seconds,
                                                  config_.cursor_min.x, config_.cursor_max.x),
                                       std::clamp(position_.y + assisted.values[1] * seconds,
                                                  config_.cursor_min.y, config_.cursor_max.y)};

        center_out::CenterOutStepResult step{};
        CHECK(machine_.step(time_ns, position_, step) == ContractStatus::ok);
        note_trial_boundaries(step);
        last_time_ns_ = time_ns;

        center_out::CenterOutCursorSample cursor{};
        cursor.time_ns = time_ns;
        cursor.before = before;
        cursor.after = position_;
        cursor.dt_ns = dt_ns;
        cursor.frame_sequence = frame;
        cursor.sample_idx = sample;
        cursor.state = step.snapshot.state;
        cursor.trial = step.snapshot.trial;
        recording_.cursor.push_back(cursor);

        center_out::CenterOutVelocitySample vel{};
        vel.time_ns = time_ns;
        vel.decoded = decoded;
        vel.guidance = reference;
        vel.assisted = assisted;
        vel.trial = step.snapshot.trial;
        recording_.vel.push_back(vel);

        recording_.guidance.push_back(
            center_out::CenterOutGuidanceObservation{time_ns, sample_out});

        keep_step(step);
        keep_target(step.snapshot, time_ns);
    }

    void decide(ExperimentTimeNs time_ns, AbnormalCondition condition)
    {
        const auto snapshot = machine_.snapshot();
        const bool in_trial = running_ && !machine_.complete();
        const auto handling = center_out::center_out_condition_handling(condition);
        const auto policy = escalate(policy_for(config_.abnormal, condition), handling.floor);
        const auto response =
            abnormal_response_for(policy, in_trial, /*can_end_trial=*/true, handling.input_refused);

        center_out::CenterOutReplayInput input{};
        input.kind = center_out::CenterOutReplayInputKind::decision;
        input.time_ns = time_ns;
        input.condition = condition;
        input.response = response;
        recording_.inputs.push_back(input);

        const bool ended = response == AbnormalResponse::trial_aborted ||
                           response == AbnormalResponse::session_aborted;
        if (ended && in_trial && has_trial_)
        {
            center_out::CenterOutReplayAbort abort{};
            abort.record.trial = snapshot.trial;
            abort.record.interval = TimeInterval{
                trial_started_ns_, time_ns < trial_started_ns_ ? trial_started_ns_ : time_ns};
            abort.record.paradigm = config_.paradigm;
            abort.record.outcome = TrialOutcome::aborted;
            abort.condition = condition;
            abort.response = response;
            recording_.aborts.push_back(abort);
        }
        if (response == AbnormalResponse::trial_aborted)
        {
            machine_.reset();
            guidance_.reset();
            has_last_time_ = false;
            has_trial_ = false;
            pending_restart_ = true;
            forget_target();
        }
        if (response == AbnormalResponse::session_aborted)
        {
            terminal_ = true;
            running_ = false;
            forget_target();
        }
    }

    /// One observation aimed at whatever target is up, at @p speed.
    ///
    /// A run that reaches its targets rather than timing out on all of them,
    /// which is what makes the recording hold decided trials to compare.
    void drive(ExperimentTimeNs time_ns, double speed, std::uint64_t frame, std::uint64_t sample)
    {
        const auto snapshot = machine_.snapshot();
        double vx = 0.0;
        double vy = 0.0;
        if (snapshot.active_target != kUnsetTargetId)
        {
            const double dx = snapshot.active_position.x - position_.x;
            const double dy = snapshot.active_position.y - position_.y;
            const double norm = std::sqrt(dx * dx + dy * dy);
            if (norm > 0.0)
            {
                vx = speed * dx / norm;
                vy = speed * dy / norm;
            }
        }
        observe(time_ns, vx, vy, frame, sample);
    }

    [[nodiscard]] const CenterOutRecording& recording() const noexcept
    {
        return recording_;
    }
    [[nodiscard]] CenterOutRecording& recording() noexcept
    {
        return recording_;
    }

  private:
    ContractStatus begin_segment(ExperimentTimeNs time_ns, center_out::CenterOutStepResult& result)
    {
        machine_.reset();
        guidance_.reset();
        has_trial_ = false;
        trial_started_ns_ = time_ns;
        const auto status = machine_.start(config_.paradigm, config_.task, time_ns, result);
        if (status == ContractStatus::ok)
        {
            note_trial_boundaries(result);
        }
        return status;
    }

    void note_trial_boundaries(const center_out::CenterOutStepResult& result)
    {
        for (std::uint8_t i = 0; i < result.n_events; ++i)
        {
            const auto& event = result.events[i];
            if (event.kind == ExperimentEventKind::trial_start)
            {
                trial_started_ns_ = event.time_ns;
                has_trial_ = true;
            }
            else if (event.kind == ExperimentEventKind::trial_stop)
            {
                has_trial_ = false;
            }
        }
    }

    void keep_step(const center_out::CenterOutStepResult& step)
    {
        for (std::uint8_t i = 0; i < step.n_transitions; ++i)
        {
            recording_.transitions.push_back(step.transitions[i]);
        }
        for (std::uint8_t i = 0; i < step.n_events; ++i)
        {
            recording_.events.push_back(step.events[i]);
        }
        if (step.trial_decided)
        {
            recording_.trials.push_back(step.trial);
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
        recording_.targets.push_back(onset);
    }

    center_out::CenterOutReplayConfig config_{};
    center_out::CenterOutMachine machine_{};
    center_out::CenterOutGuidance guidance_{};
    center_out::WorkspacePoint position_{};
    CenterOutRecording recording_{};
    ExperimentTimeNs last_time_ns_{};
    ExperimentTimeNs trial_started_ns_{};
    TargetId last_target_{kUnsetTargetId};
    bool has_last_time_{};
    bool has_trial_{};
    bool has_target_{};
    bool pending_restart_{};
    bool running_{};
    bool terminal_{};
};

/// A Center-Out run that reaches its targets: both trials are decided.
CenterOutRecorder record_center_out(const center_out::CenterOutReplayConfig& config)
{
    CenterOutRecorder recorder{config};
    recorder.start(1'000);
    for (std::uint64_t step = 1; step <= 60; ++step)
    {
        recorder.drive(1'000 + step * 50'000'000, 1.0, step / 4 + 1, step - 1);
    }
    return recorder;
}

struct WebGridRecording
{
    ExperimentTimeNs origin_ns{};
    std::vector<webgrid::WebGridReplayInput> inputs{};
    std::vector<webgrid::WebGridPointerSample> pointer{};
    std::vector<webgrid::WebGridReplaySelection> selections{};
    std::vector<webgrid::WebGridReplayTrial> trials{};
    std::vector<webgrid::WebGridTargetOnset> targets{};
    webgrid::WebGridMetrics metrics{};
    std::uint64_t invalidated_targets{};

    [[nodiscard]] webgrid::WebGridReplayRecording view(const ReplayProvenance& provenance,
                                                       bool complete = true) const noexcept
    {
        webgrid::WebGridReplayRecording recording{};
        recording.provenance = provenance;
        recording.origin_ns = origin_ns;
        recording.inputs = inputs;
        recording.expected.pointer = pointer;
        recording.expected.selections = selections;
        recording.expected.trials = trials;
        recording.expected.targets = targets;
        recording.expected.metrics = metrics;
        recording.expected.invalidated_targets = invalidated_targets;
        recording.expected.has_pointer = complete;
        recording.expected.has_selections = complete;
        recording.expected.has_trials = complete;
        recording.expected.has_targets = complete;
        recording.expected.has_metrics = complete;
        recording.run_end_recorded = true;
        return recording;
    }
};

class WebGridRecorder
{
  public:
    explicit WebGridRecorder(const webgrid::WebGridReplayConfig& config) : config_(config) {}

    void start(ExperimentTimeNs origin_ns)
    {
        recording_.origin_ns = origin_ns;
        webgrid::WebGridStepResult opening{};
        CHECK(machine_.start(config_.paradigm, config_.task, origin_ns, opening) ==
              ContractStatus::ok);
        running_ = true;
        recording_.metrics = opening.snapshot.metrics;
        keep_target(opening.snapshot);
    }

    void point(ExperimentTimeNs time_ns, webgrid::PointerPosition pointer)
    {
        step(time_ns, pointer, nullptr);
    }

    void select(ExperimentTimeNs time_ns, webgrid::PointerPosition pointer,
                SequenceOrdinal sequence)
    {
        const auto snapshot = machine_.snapshot();
        SelectionEvent event{};
        CHECK(webgrid::make_selection_event(config_.task, pointer, snapshot.active_target,
                                            snapshot.trial, config_.paradigm, time_ns, sequence,
                                            event) == ContractStatus::ok);
        step(time_ns, pointer, &event);
    }

    void gap(ExperimentTimeNs time_ns)
    {
        const bool in_trial = running_ && !machine_.complete();
        const auto policy = policy_for(config_.abnormal, AbnormalCondition::input_gap);
        const auto response = abnormal_response_for(policy, in_trial, /*can_end_trial=*/false,
                                                    /*input_refused=*/false);
        webgrid::WebGridReplayInput input{};
        input.kind = webgrid::WebGridReplayInputKind::pointer_gap;
        input.time_ns = time_ns;
        input.condition = AbnormalCondition::input_gap;
        input.response = response;
        recording_.inputs.push_back(input);
        if (response == AbnormalResponse::trial_invalidated)
        {
            target_invalidated_ = true;
            ++recording_.invalidated_targets;
        }
    }

    [[nodiscard]] WebGridRecording& recording() noexcept
    {
        return recording_;
    }

  private:
    void step(ExperimentTimeNs time_ns, webgrid::PointerPosition pointer,
              const SelectionEvent* selection)
    {
        webgrid::WebGridReplayInput input{};
        input.kind = selection == nullptr ? webgrid::WebGridReplayInputKind::pointer
                                          : webgrid::WebGridReplayInputKind::selection;
        input.time_ns = time_ns;
        input.pointer = pointer;
        if (selection != nullptr)
        {
            input.selection = *selection;
        }
        recording_.inputs.push_back(input);
        if (!running_ || machine_.complete())
        {
            return;
        }

        webgrid::WebGridStepResult step{};
        const auto status = selection == nullptr
                                ? machine_.step(time_ns, pointer, step)
                                : machine_.step(time_ns, pointer, *selection, step);
        CHECK(status == ContractStatus::ok);
        recording_.metrics = step.snapshot.metrics;

        webgrid::WebGridPointerSample sample{};
        sample.time_ns = step.snapshot.time_ns;
        sample.pointer = pointer;
        sample.state = step.snapshot.state;
        sample.active_target = step.snapshot.active_target;
        sample.trial = step.snapshot.trial;
        recording_.pointer.push_back(sample);

        if (step.selection_processed)
        {
            recording_.selections.push_back(
                webgrid::WebGridReplaySelection{step.selection, !target_invalidated_});
        }
        if (step.trial_decided)
        {
            webgrid::WebGridReplayTrial trial{};
            trial.trial = step.trial;
            trial.recorded_outcome =
                target_invalidated_ ? TrialOutcome::aborted : step.trial.record.outcome;
            trial.acquisition_timing_valid = !target_invalidated_;
            recording_.trials.push_back(trial);
            target_invalidated_ = false;
        }
        keep_target(step.snapshot);
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
        recording_.targets.push_back(webgrid::WebGridTargetOnset{
            snapshot.target_onset_ns, snapshot.active_target, snapshot.completed, snapshot.trial});
    }

    webgrid::WebGridReplayConfig config_{};
    webgrid::WebGridMachine machine_{};
    WebGridRecording recording_{};
    ExperimentTimeNs last_onset_ns_{};
    TargetId last_target_{kUnsetTargetId};
    bool target_invalidated_{};
    bool has_target_{};
    bool running_{};
};

WebGridRecorder record_webgrid(const webgrid::WebGridReplayConfig& config)
{
    WebGridRecorder recorder{config};
    recorder.start(10);
    recorder.point(20, webgrid::PointerPosition{0.5, 0.5});
    recorder.select(30, webgrid::PointerPosition{0.5, 0.5}, 1);
    recorder.point(40, webgrid::PointerPosition{1.5, 0.5});
    recorder.select(50, webgrid::PointerPosition{1.5, 0.5}, 2);
    return recorder;
}

struct SpeechRecording
{
    ExperimentTimeNs origin_ns{};
    std::vector<speech::SpeechReplayInput> inputs{};
    std::vector<StateTransition> transitions{};
    std::vector<ExperimentEvent> events{};
    std::vector<PresentationRequest> requests{};
    std::vector<speech::SpeechReplayPhase> phases{};
    std::vector<speech::SpeechTrialSchedule> schedules{};
    std::vector<speech::SpeechReplayTrial> trials{};
    std::vector<speech::SpeechReplayReportDecision> reports{};

    [[nodiscard]] speech::SpeechReplayRecording view(const ReplayProvenance& provenance,
                                                     bool complete = true) const noexcept
    {
        speech::SpeechReplayRecording recording{};
        recording.provenance = provenance;
        recording.origin_ns = origin_ns;
        recording.inputs = inputs;
        recording.expected.transitions = transitions;
        recording.expected.events = events;
        recording.expected.requests = requests;
        recording.expected.phases = phases;
        recording.expected.schedules = schedules;
        recording.expected.trials = trials;
        recording.expected.reports = reports;
        recording.expected.has_transitions = complete;
        recording.expected.has_events = complete;
        recording.expected.has_requests = complete;
        recording.expected.has_phases = complete;
        // The stream itself is recovered even when an early terminal input
        // means it holds only the trials that actually started, matching how
        // SpeechTraceWriter itself behaves rather than pre-filling future trials.
        recording.expected.has_schedules = true;
        recording.expected.has_trials = complete;
        recording.expected.has_reports = complete;
        recording.run_end_recorded = true;
        return recording;
    }
};

class SpeechRecorder
{
  public:
    explicit SpeechRecorder(const speech::SpeechReplayConfig& config)
        : config_(config), requests_(static_cast<std::size_t>(config.task.n_trials))
    {
        for (TrialOrdinal ordinal = 0; ordinal < config.task.n_trials; ++ordinal)
        {
            speech::SpeechTrialSchedule schedule{};
            CHECK(speech::prepare_trial(config.task, ordinal, schedule) == ContractStatus::ok);
            prepared_schedules_.push_back(schedule);
        }
    }

    void start(ExperimentTimeNs origin_ns)
    {
        recording_.origin_ns = origin_ns;
        speech::SpeechStepResult opening{};
        CHECK(machine_.start(config_.paradigm, config_.task, prepared_schedules_[0], origin_ns,
                             opening) == ContractStatus::ok);
        running_ = true;
        keep_step(opening);
        keep_trial_start(opening.snapshot);
    }

    void advance(ExperimentTimeNs time_ns)
    {
        speech::SpeechReplayInput input{};
        input.kind = speech::SpeechReplayInputKind::advance;
        input.time_ns = time_ns;
        recording_.inputs.push_back(input);
        if (!running_ || machine_.complete())
        {
            return;
        }
        const auto snapshot = machine_.snapshot();
        const auto next_ordinal = snapshot.trial.ordinal + 1;
        speech::SpeechStepResult step{};
        const auto status = next_ordinal < config_.task.n_trials
                                ? machine_.step(time_ns, prepared_schedules_[next_ordinal], step)
                                : machine_.step(time_ns, step);
        CHECK(status == ContractStatus::ok);
        keep_step(step);
        keep_trial_start(step.snapshot);
    }

    void halt(ExperimentTimeNs time_ns,
              AbnormalCondition condition = AbnormalCondition::emergency_stop)
    {
        speech::SpeechReplayInput input{};
        input.kind = speech::SpeechReplayInputKind::halt;
        input.time_ns = time_ns;
        input.condition = condition;
        input.response = abnormal_response_for(
            escalate(policy_for(config_.abnormal, condition), AbnormalPolicy::abort_trial),
            running_ && !machine_.complete(), /*can_end_trial=*/false,
            /*input_refused=*/true);
        recording_.inputs.push_back(input);
        running_ = false;
    }

    /// A presenter report, built against a request this run actually emitted.
    [[nodiscard]] PresentationOutcome report_for(TrialOrdinal ordinal, PresentationStatus status,
                                                 SequenceOrdinal sequence) const
    {
        PresentationOutcome outcome{};
        for (const auto& request : recording_.requests)
        {
            if (request.cue == CueKind::text_content && request.trial.ordinal == ordinal)
            {
                outcome.requested_ns = request.requested_ns;
                outcome.request_sequence = request.sequence;
                outcome.trial = request.trial;
                outcome.stimulus_id = request.stimulus_id;
            }
        }
        outcome.sequence = sequence;
        outcome.status = status;
        if (status == PresentationStatus::presented)
        {
            outcome.presented_ns = outcome.requested_ns + 7;
        }
        return outcome;
    }

    void report(const PresentationOutcome& outcome)
    {
        speech::SpeechReplayInput input{};
        input.kind = speech::SpeechReplayInputKind::presentation_report;
        input.outcome = outcome;
        recording_.inputs.push_back(input);

        const bool in_range = outcome.trial.ordinal < requests_.size();
        Slot* slot =
            in_range ? &requests_[static_cast<std::size_t>(outcome.trial.ordinal)] : nullptr;
        const bool addressable =
            slot != nullptr && slot->occupied && slot->ordinal == outcome.trial.ordinal;
        const bool matched =
            validate(outcome) == ContractStatus::ok && addressable && slot->emitted &&
            !slot->answered && outcome.request_sequence == slot->sequence &&
            outcome.requested_ns == slot->requested_ns &&
            outcome.stimulus_id == slot->stimulus_id && same_trial(outcome.trial, slot->trial);
        if (matched)
        {
            slot->answered = true;
        }
        AbnormalCondition condition{AbnormalCondition::presentation_report_unmatched};
        bool has_trial = false;
        if (matched && outcome.status == PresentationStatus::presented)
        {
            condition = AbnormalCondition::unspecified;
        }
        else if (matched)
        {
            condition = AbnormalCondition::presentation_failed;
            has_trial = true;
        }
        auto response = AbnormalResponse::recorded;
        if (condition != AbnormalCondition::unspecified)
        {
            response = abnormal_response_for(policy_for(config_.abnormal, condition), has_trial,
                                             /*can_end_trial=*/false, /*input_refused=*/!matched);
        }
        if (response == AbnormalResponse::trial_invalidated && addressable)
        {
            slot->invalidated = true;
        }
        recording_.reports.push_back(
            speech::SpeechReplayReportDecision{matched, condition, response});
    }

    [[nodiscard]] SpeechRecording& recording() noexcept
    {
        return recording_;
    }

    [[nodiscard]] const std::vector<speech::SpeechTrialSchedule>&
    prepared_schedules() const noexcept
    {
        return prepared_schedules_;
    }

  private:
    struct Slot
    {
        TrialOrdinal ordinal{};
        SequenceOrdinal sequence{};
        ExperimentTimeNs requested_ns{};
        TrialIdentity trial{};
        StimulusId stimulus_id{kUnsetStimulusId};
        bool occupied{};
        bool emitted{};
        bool answered{};
        bool invalidated{};
    };

    void keep_step(const speech::SpeechStepResult& step)
    {
        for (std::uint8_t i = 0; i < step.n_transitions; ++i)
        {
            recording_.transitions.push_back(step.transitions[i]);
        }
        for (std::uint8_t i = 0; i < step.n_events; ++i)
        {
            recording_.events.push_back(step.events[i]);
        }
        for (std::uint8_t i = 0; i < step.n_requests; ++i)
        {
            const auto& request = step.requests[i];
            if (request.cue == CueKind::text_content)
            {
                auto& slot = requests_[static_cast<std::size_t>(request.trial.ordinal)];
                slot.ordinal = request.trial.ordinal;
                slot.sequence = request.sequence;
                slot.requested_ns = request.requested_ns;
                slot.trial = request.trial;
                slot.stimulus_id = request.stimulus_id;
                slot.occupied = true;
                slot.emitted = true;
                slot.answered = false;
                slot.invalidated = false;
            }
            recording_.requests.push_back(request);
        }
        if (step.trial_decided)
        {
            const auto& slot = requests_[static_cast<std::size_t>(step.trial.record.trial.ordinal)];
            const bool invalidated = slot.occupied &&
                                     slot.ordinal == step.trial.record.trial.ordinal &&
                                     slot.invalidated;
            speech::SpeechReplayTrial trial{};
            trial.trial = step.trial;
            trial.recorded_outcome =
                invalidated ? TrialOutcome::aborted : step.trial.record.outcome;
            trial.invalidated = invalidated;
            recording_.trials.push_back(trial);
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
        recording_.schedules.push_back(snapshot.schedule);
        for (std::uint8_t i = 0; i < snapshot.timeline.count; ++i)
        {
            recording_.phases.push_back(
                speech::SpeechReplayPhase{snapshot.timeline.phases[i], snapshot.trial});
        }
    }

    speech::SpeechReplayConfig config_{};
    speech::SpeechMachine machine_{};
    SpeechRecording recording_{};
    std::vector<speech::SpeechTrialSchedule> prepared_schedules_{};
    std::vector<Slot> requests_{};
    TrialOrdinal last_trial_{};
    bool has_trial_{};
    bool running_{};
};

SpeechRecorder record_speech(const speech::SpeechReplayConfig& config)
{
    SpeechRecorder recorder{config};
    recorder.start(1'000);
    for (ExperimentTimeNs time = 1'050; time <= 1'000 + 2'000; time += 50)
    {
        recorder.advance(time);
    }
    return recorder;
}

// --- tests ------------------------------------------------------------------

void test_recorded_center_out_run_replays_identically()
{
    const auto config = center_out_config();
    auto recorder = record_center_out(config);

    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto report = replay.run(recorder.recording().view(replay.provenance()));

    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(replay_reproduced(report));
    CHECK(report.rejection == ReplayRejection::none);
    CHECK(report.completeness == ReplayCompleteness::complete);
    CHECK(report.inputs_replayed == recorder.recording().inputs.size());
    // A match over nothing is not a match. The counts are asserted so that a
    // comparison that silently stopped comparing cannot pass this test.
    CHECK(report.items_compared > 20);
    CHECK(report.fields_compared > 100);
    CHECK(!recorder.recording().trials.empty());
    CHECK(!recorder.recording().transitions.empty());
}

void test_center_out_boundary_saturation_replays_identically()
{
    auto config = center_out_config();
    config.assistance_method = assistance::AssistanceMethod::none;
    config.linear_assistance = {};
    config.cursor_min = {-0.5, -0.5};
    config.cursor_max = {0.5, 0.5};
    CenterOutRecorder recorder{config};
    recorder.start(1'000);
    recorder.observe(100'001'000, 10.0, 0.0, 0, 0);
    recorder.observe(200'001'000, 10.0, 0.0, 1, 1);
    recorder.observe(300'001'000, -1.0, 0.0, 2, 2);
    CHECK(recorder.recording().cursor[0].after.x == 0.5);
    CHECK(recorder.recording().cursor[1].before.x == 0.5);
    CHECK(recorder.recording().cursor[1].after.x == 0.5);
    CHECK(recorder.recording().cursor[2].after.x == 0.4);

    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    CHECK(replay.run(recorder.recording().view(replay.provenance())).verdict ==
          ReplayVerdict::match);
}

void test_repeated_replay_answers_identically()
{
    const auto config = center_out_config();
    auto recorder = record_center_out(config);
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto recording = recorder.recording().view(replay.provenance());

    const auto first = replay.run(recording);
    const auto second = replay.run(recording);
    // The point of reset() being run()'s own first act: a replay carries no
    // state from the run before it, so a caller that replays a recording twice
    // gets the same answer rather than a second one built on the first.
    CHECK(first.verdict == second.verdict);
    CHECK(first.items_compared == second.items_compared);
    CHECK(first.fields_compared == second.fields_compared);
    CHECK(first.inputs_replayed == second.inputs_replayed);
    CHECK(second.verdict == ReplayVerdict::match);

    // And an explicit reset between them changes nothing either.
    replay.reset();
    const auto third = replay.run(recording);
    CHECK(third.verdict == ReplayVerdict::match);
    CHECK(third.fields_compared == first.fields_compared);
}

void test_foreign_experiment_recording_is_rejected_early()
{
    const auto config = center_out_config();
    auto recorder = record_center_out(config);
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    struct Case
    {
        ReplayRejection expected;
        void (*mutate)(ReplayProvenance&);
    };
    const Case cases[] = {
        {ReplayRejection::paradigm_mismatch, [](ReplayProvenance& p) { p.paradigm += 1; }},
        {ReplayRejection::experiment_version_mismatch,
         [](ReplayProvenance& p) { p.experiment_version += 1; }},
        {ReplayRejection::seed_mismatch, [](ReplayProvenance& p) { p.seed += 1; }},
        {ReplayRejection::sampler_version_mismatch,
         [](ReplayProvenance& p) { p.sampler_version += 1; }},
        {ReplayRejection::configuration_fingerprint_mismatch,
         [](ReplayProvenance& p) { p.configuration_fingerprint += 1; }},
        {ReplayRejection::schedule_fingerprint_mismatch,
         [](ReplayProvenance& p) { p.schedule_fingerprint += 1; }},
        {ReplayRejection::realized_schedule_mismatch,
         [](ReplayProvenance& p) { p.realized_schedule_fingerprint += 1; }},
        {ReplayRejection::metric_version_mismatch,
         [](ReplayProvenance& p) { p.metric_version += 1; }},
        {ReplayRejection::policy_version_mismatch,
         [](ReplayProvenance& p) { p.policy_version += 1; }},
        {ReplayRejection::provenance_incomplete,
         [](ReplayProvenance& p) { p.paradigm = kUnsetParadigmId; }},
    };
    for (const auto& scenario : cases)
    {
        auto provenance = replay.provenance();
        scenario.mutate(provenance);
        const auto report = replay.run(recorder.recording().view(provenance));
        CHECK(report.verdict == ReplayVerdict::rejected);
        CHECK(report.rejection == scenario.expected);
        // Rejected before anything was replayed, which is the whole reason the
        // provenance check runs first: a mismatch report about a comparison
        // against a different experiment would describe nothing.
        CHECK(report.inputs_replayed == 0);
        CHECK(report.items_compared == 0);
        CHECK(report.first_mismatch.item == ReplayItem::none);
    }
}

void test_unrecorded_configuration_is_rejected()
{
    // Two configurations that differ in a value the fingerprint covers. The
    // recording is made under the first and replayed under the second, which is
    // how a caller actually gets this wrong: not by editing a digest, but by
    // replaying with the configuration they happen to have.
    const auto recorded_config = center_out_config();
    auto altered_config = center_out_config();
    altered_config.task.seed = recorded_config.task.seed + 1;

    auto recorder = record_center_out(recorded_config);
    center_out::CenterOutReplay recorded_replay{};
    CHECK(recorded_replay.prepare(recorded_config) == ContractStatus::ok);
    center_out::CenterOutReplay altered_replay{};
    CHECK(altered_replay.prepare(altered_config) == ContractStatus::ok);

    const auto report = altered_replay.run(recorder.recording().view(recorded_replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::rejected);
    CHECK(report.rejection == ReplayRejection::seed_mismatch);

    // The same recording under the configuration it was recorded with does
    // reproduce, so the rejection above is about the configuration and not
    // about the recording.
    CHECK(recorded_replay.run(recorder.recording().view(recorded_replay.provenance())).verdict ==
          ReplayVerdict::match);
}

void test_altered_input_is_reported_at_first_difference()
{
    const auto config = center_out_config();
    auto recorder = record_center_out(config);
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    auto& recording = recorder.recording();
    // One decoded component of one observation, changed by the smallest amount
    // that is still a different number.
    std::size_t altered_idx = 0;
    for (std::size_t i = 0; i < recording.inputs.size(); ++i)
    {
        if (recording.inputs[i].kind == center_out::CenterOutReplayInputKind::observation)
        {
            altered_idx = i;
            break;
        }
    }
    recording.inputs[altered_idx].decoded[0] += 0.5;

    const auto report = replay.run(recording.view(replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::mismatch);
    CHECK(report.rejection == ReplayRejection::none);
    // The cursor, not a transition: the cursor is what the changed input
    // changed first, and everything downstream of it is a consequence.
    CHECK(report.first_mismatch.item == ReplayItem::cursor);
    CHECK(report.first_mismatch.idx == 0);
    CHECK(report.first_mismatch.real_valued);
    CHECK(report.first_mismatch.recorded_real != report.first_mismatch.regenerated_real);
    CHECK(report.first_mismatch.field == "after_x");
}

void test_altered_transition_is_reported_as_itself()
{
    const auto config = center_out_config();
    auto recorder = record_center_out(config);
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    auto& recording = recorder.recording();
    CHECK(recording.transitions.size() > 2);
    const std::size_t idx = recording.transitions.size() - 1;
    recording.transitions[idx].cause += 1;

    const auto report = replay.run(recording.view(replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::mismatch);
    CHECK(report.first_mismatch.item == ReplayItem::transition);
    CHECK(report.first_mismatch.idx == idx);
    CHECK(report.first_mismatch.field == "cause");
    CHECK(!report.first_mismatch.real_valued);
    CHECK(report.first_mismatch.recorded == report.first_mismatch.regenerated + 1);
}

void test_recording_cardinality_mismatch_is_reported()
{
    const auto config = center_out_config();
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    {
        auto recorder = record_center_out(config);
        auto& recording = recorder.recording();
        recording.events.pop_back();
        const auto report = replay.run(recording.view(replay.provenance()));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::stream_length);
        CHECK(report.first_mismatch.field == "recorded_short");
    }
    {
        auto recorder = record_center_out(config);
        auto& recording = recorder.recording();
        recording.events.push_back(recording.events.back());
        const auto report = replay.run(recording.view(replay.provenance()));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::stream_length);
        CHECK(report.first_mismatch.field == "recorded_extra");
    }
}

void test_absent_stream_does_not_hide_cardinality_errors()
{
    const auto config = center_out_config();
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    {
        auto recorder = record_center_out(config);
        auto& recording = recorder.recording();
        recording.events.pop_back();
        auto view = recording.view(replay.provenance());
        view.expected.has_guidance = false;
        const auto report = replay.run(view);
        CHECK(report.completeness == ReplayCompleteness::stream_absent);
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::stream_length);
        CHECK(report.first_mismatch.field == "recorded_short");
    }
    {
        auto recorder = record_center_out(config);
        auto& recording = recorder.recording();
        recording.events.push_back(recording.events.back());
        auto view = recording.view(replay.provenance());
        view.expected.has_guidance = false;
        const auto report = replay.run(view);
        CHECK(report.completeness == ReplayCompleteness::stream_absent);
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::stream_length);
        CHECK(report.first_mismatch.field == "recorded_extra");
    }
}

void test_abnormally_ended_trial_replays_as_such()
{
    // A stale interval under the default severity ends the trial in flight and
    // restarts the segment. Everything after it -- the new segment's own start,
    // its target, its transitions -- is regenerated too, which is what makes
    // this a replay of a faulted run rather than of the part before the fault.
    auto config = center_out_config();
    config.abnormal.input_discontinuity = AbnormalPolicy::abort_trial;

    CenterOutRecorder recorder{config};
    recorder.start(1'000);
    recorder.observe(100'001'000, 0.6, 0.0, 1, 0);
    recorder.decide(200'001'000, AbnormalCondition::input_stale);
    recorder.observe(200'001'000, 0.6, 0.0, 2, 1);
    recorder.observe(300'001'000, -0.5, 0.0, 2, 2);
    recorder.decide(400'001'000, AbnormalCondition::source_discontinuity);
    recorder.observe(500'001'000, 0.2, 0.0, 5, 3);

    CHECK(!recorder.recording().aborts.empty());

    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto report = replay.run(recorder.recording().view(replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(report.items_compared > 10);
}

void test_replay_under_foreign_severity_disagrees()
{
    // The abnormal policies are part of the experiment, and the fingerprints do
    // not cover them. The response is therefore regenerated from the configured
    // severity rather than taken from the recording, so replaying a recorded
    // condition under a different severity disagrees about what the run did --
    // which is the only thing that makes the policy replayable at all.
    auto recorded_config = center_out_config();
    recorded_config.abnormal.input_discontinuity = AbnormalPolicy::abort_trial;
    CenterOutRecorder recorder{recorded_config};
    recorder.start(1'000);
    recorder.observe(100'001'000, 0.6, 0.0, 1, 0);
    recorder.decide(200'001'000, AbnormalCondition::input_stale);
    recorder.observe(300'001'000, 0.6, 0.0, 2, 1);

    auto altered_config = recorded_config;
    altered_config.abnormal.input_discontinuity = AbnormalPolicy::abort_session;

    center_out::CenterOutReplay recorded_replay{};
    CHECK(recorded_replay.prepare(recorded_config) == ContractStatus::ok);
    center_out::CenterOutReplay altered_replay{};
    CHECK(altered_replay.prepare(altered_config) == ContractStatus::ok);

    const auto recording = recorder.recording().view(recorded_replay.provenance());
    CHECK(recorded_replay.run(recording).verdict == ReplayVerdict::match);

    const auto report = altered_replay.run(recording);
    CHECK(report.verdict == ReplayVerdict::mismatch);
    CHECK(report.first_mismatch.item == ReplayItem::trial);
    CHECK(report.first_mismatch.recorded ==
          static_cast<std::uint64_t>(AbnormalResponse::trial_aborted));
    CHECK(report.first_mismatch.regenerated ==
          static_cast<std::uint64_t>(AbnormalResponse::session_aborted));
}

void test_incomplete_evidence_is_never_a_match()
{
    const auto config = center_out_config();
    auto recorder = record_center_out(config);
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    // Every stream present but one. The prefix agrees perfectly, and the answer
    // is still not `match`: what a recording did not keep cannot have agreed.
    {
        auto recording = recorder.recording().view(replay.provenance());
        recording.expected.has_transitions = false;
        const auto report = replay.run(recording);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(!replay_reproduced(report));
        CHECK(report.completeness == ReplayCompleteness::stream_absent);
        CHECK(report.first_mismatch.item == ReplayItem::none);
        // Everything that was there was still checked. An incomplete verdict is
        // a partial verification, not a refusal to do one.
        CHECK(report.items_compared > 0);
    }
    // And the dangerous case: the missing stream is exactly the one that would
    // have disagreed. Answering `match` here is the failure mode the whole
    // verdict vocabulary exists to prevent.
    {
        auto mutated = recorder.recording();
        mutated.transitions[mutated.transitions.size() - 1].cause += 1;
        auto recording = mutated.view(replay.provenance());
        recording.expected.has_transitions = false;
        const auto report = replay.run(recording);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(!replay_reproduced(report));
    }
    {
        auto recording = recorder.recording().view(replay.provenance());
        recording.run_end_recorded = false;
        const auto report = replay.run(recording);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(report.completeness == ReplayCompleteness::run_end_missing);
    }
    {
        auto recording = recorder.recording().view(replay.provenance());
        recording.trace_loss_recorded = true;
        const auto report = replay.run(recording);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(report.completeness == ReplayCompleteness::trace_loss_recorded);
    }
    // A recording that lost records is short, and a short recording is not
    // evidence of disagreement: the records a replay would have disagreed with
    // are the ones that might be missing.
    {
        auto mutated = recorder.recording();
        mutated.events.pop_back();
        auto recording = mutated.view(replay.provenance());
        recording.trace_loss_recorded = true;
        const auto report = replay.run(recording);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(report.completeness == ReplayCompleteness::trace_loss_recorded);
    }
    // The other policy compares nothing at all, for a caller whose question is
    // whether the recording is replayable rather than whether a prefix agrees.
    {
        auto recording = recorder.recording().view(replay.provenance());
        recording.expected.has_cursor = false;
        const auto report = replay.run(recording, ReplayIncompletePolicy::refuse);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(report.items_compared == 0);
        CHECK(report.inputs_replayed == 0);
    }
}

void test_unconfigured_replay_verifies_nothing()
{
    center_out::CenterOutReplay replay{};
    CHECK(!replay.prepared());
    center_out::CenterOutReplayRecording empty{};
    const auto report = replay.run(empty);
    CHECK(report.verdict == ReplayVerdict::rejected);
    CHECK(report.rejection == ReplayRejection::configuration_invalid);

    // A default-constructed report is rejected too, so a caller that forgot to
    // run the replay cannot read its answer as a reproduction.
    const ReplayReport untouched{};
    CHECK(untouched.verdict == ReplayVerdict::rejected);
    CHECK(!replay_reproduced(untouched));

    auto config = center_out_config();
    config.paradigm = kUnsetParadigmId;
    CHECK(replay.prepare(config) == ContractStatus::identity_missing);
    CHECK(!replay.prepared());
}

void test_impossible_inputs_are_rejected()
{
    const auto config = center_out_config();
    auto recorder = record_center_out(config);
    center_out::CenterOutReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    auto& recording = recorder.recording();
    // Time that moves backwards is not a difference between two runs; it is
    // evidence that cannot describe one, and is refused rather than compared.
    const auto first = recording.inputs[1].time_ns;
    recording.inputs[1].time_ns = recording.inputs[2].time_ns;
    recording.inputs[2].time_ns = first;
    const auto report = replay.run(recording.view(replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::rejected);
    CHECK(report.rejection == ReplayRejection::evidence_invalid);
    CHECK(report.inputs_replayed == 0);

    {
        auto altered = recorder.recording();
        altered.inputs.front().condition = AbnormalCondition::input_gap;
        CHECK(replay.run(altered.view(replay.provenance())).rejection ==
              ReplayRejection::evidence_invalid);
    }

    {
        auto web_config = webgrid_config();
        auto web_recorder = record_webgrid(web_config);
        webgrid::WebGridReplay web_replay{};
        CHECK(web_replay.prepare(web_config) == ContractStatus::ok);
        web_recorder.recording().inputs.front().kind = webgrid::WebGridReplayInputKind::pointer_gap;
        web_recorder.recording().inputs.front().condition = AbnormalCondition::observer_frame_drop;
        CHECK(web_replay.run(web_recorder.recording().view(web_replay.provenance())).rejection ==
              ReplayRejection::evidence_invalid);
    }

    {
        auto speech_config_value = speech_config();
        auto speech_recorder = record_speech(speech_config_value);
        speech::SpeechReplay speech_replay{};
        CHECK(speech_replay.prepare(speech_config_value) == ContractStatus::ok);
        speech_recorder.recording().inputs.front().condition = AbnormalCondition::emergency_stop;
        const auto provenance = speech_replay.provenance(speech_recorder.recording().schedules);
        CHECK(speech_replay.run(speech_recorder.recording().view(provenance)).rejection ==
              ReplayRejection::evidence_invalid);
    }
}

void test_sampler_support_required_only_on_draw()
{
    const auto unsupported = kCurrentSamplerVersion + 17;

    auto center_config = center_out_config();
    center_config.task.sampler_version = unsupported;
    auto center_recorder = record_center_out(center_config);
    center_out::CenterOutReplay center_replay{};
    CHECK(center_replay.prepare(center_config) == ContractStatus::ok);
    CHECK(center_replay.run(center_recorder.recording().view(center_replay.provenance())).verdict ==
          ReplayVerdict::match);

    auto web_config = webgrid_config();
    web_config.task.sampler_version = unsupported;
    auto web_recorder = record_webgrid(web_config);
    webgrid::WebGridReplay web_replay{};
    CHECK(web_replay.prepare(web_config) == ContractStatus::ok);
    CHECK(web_replay.run(web_recorder.recording().view(web_replay.provenance())).verdict ==
          ReplayVerdict::match);

    auto center_draw_config = center_out_config();
    center_draw_config.task.selection = center_out::TargetSelectionPolicy::sample_each_trial;
    auto center_draw_recorder = record_center_out(center_draw_config);
    center_draw_config.task.sampler_version = unsupported;
    center_out::CenterOutReplay center_draw_replay{};
    CHECK(center_draw_replay.prepare(center_draw_config) == ContractStatus::ok);
    const auto center_refused = center_draw_replay.run(
        center_draw_recorder.recording().view(center_draw_replay.provenance()));
    CHECK(center_refused.verdict == ReplayVerdict::rejected);
    CHECK(center_refused.rejection == ReplayRejection::sampler_version_unsupported);
    CHECK(center_refused.inputs_replayed == 0);

    auto web_draw_config = webgrid_config();
    web_draw_config.task.schedule = webgrid::TargetScheduleKind::seeded;
    web_draw_config.task.n_explicit = 0;
    web_draw_config.task.explicit_targets = {};
    auto web_draw_recorder = record_webgrid(web_draw_config);
    web_draw_config.task.sampler_version = unsupported;
    webgrid::WebGridReplay web_draw_replay{};
    CHECK(web_draw_replay.prepare(web_draw_config) == ContractStatus::ok);
    const auto web_refused =
        web_draw_replay.run(web_draw_recorder.recording().view(web_draw_replay.provenance()));
    CHECK(web_refused.verdict == ReplayVerdict::rejected);
    CHECK(web_refused.rejection == ReplayRejection::sampler_version_unsupported);
    CHECK(web_refused.inputs_replayed == 2);
}

void test_webgrid_names_unsupported_metric_formula()
{
    const auto recorded_config = webgrid_config();
    auto recorder = record_webgrid(recorded_config);
    auto future_config = recorded_config;
    future_config.task.metric_version = webgrid::kCurrentMetricVersion + 1;

    webgrid::WebGridReplay replay{};
    CHECK(replay.prepare(future_config) == ContractStatus::ok);
    const auto report = replay.run(recorder.recording().view(replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::rejected);
    CHECK(report.rejection == ReplayRejection::metric_version_unsupported);
    CHECK(report.inputs_replayed == 0);
    CHECK(report.items_compared == 0);
}

void test_recorded_webgrid_run_replays_identically()
{
    const auto config = webgrid_config();
    auto recorder = record_webgrid(config);
    webgrid::WebGridReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    const auto report = replay.run(recorder.recording().view(replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(report.completeness == ReplayCompleteness::complete);
    CHECK(report.items_compared > 5);
    CHECK(recorder.recording().selections.size() == 2);
    CHECK(!recorder.recording().trials.empty());
    // The derived metric set is part of what was reproduced, not something the
    // replay took on trust from the recording.
    CHECK(recorder.recording().metrics.correct_selections > 0);
}

void test_disagreeing_webgrid_selection_is_found()
{
    const auto config = webgrid_config();
    webgrid::WebGridReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    {
        auto recorder = record_webgrid(config);
        auto& recording = recorder.recording();
        recording.selections[0].record.event.correct =
            !recording.selections[0].record.event.correct;
        const auto report = replay.run(recording.view(replay.provenance()));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::selection);
        CHECK(report.first_mismatch.idx == 0);
        CHECK(report.first_mismatch.field == "correct");
    }
    {
        auto recorder = record_webgrid(config);
        auto& recording = recorder.recording();
        recording.metrics.correct_selections += 1;
        const auto report = replay.run(recording.view(replay.provenance()));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::metrics);
        CHECK(report.first_mismatch.field == "correct_selections");
    }
    {
        auto recorder = record_webgrid(config);
        auto& recording = recorder.recording();
        recording.pointer[0].pointer.x += 0.25;
        const auto report = replay.run(recording.view(replay.provenance()));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::cursor);
        CHECK(report.first_mismatch.field == "x");
    }
}

void test_gap_crossed_webgrid_target_replays_invalidated()
{
    auto config = webgrid_config();
    config.abnormal.input_discontinuity = AbnormalPolicy::abort_trial;

    WebGridRecorder recorder{config};
    recorder.start(10);
    recorder.point(20, webgrid::PointerPosition{0.5, 0.5});
    recorder.gap(25);
    recorder.select(30, webgrid::PointerPosition{0.5, 0.5}, 1);
    recorder.point(40, webgrid::PointerPosition{1.5, 0.5});
    recorder.select(50, webgrid::PointerPosition{1.5, 0.5}, 2);

    CHECK(recorder.recording().invalidated_targets == 1);
    CHECK(recorder.recording().trials.size() == 2);
    // The machine still called it a success; the recording still calls the
    // trial aborted. A replay has to reproduce both of those, separately.
    CHECK(recorder.recording().trials[0].recorded_outcome == TrialOutcome::aborted);
    CHECK(recorder.recording().trials[0].trial.record.outcome != TrialOutcome::aborted);
    CHECK(recorder.recording().trials[1].recorded_outcome != TrialOutcome::aborted);

    webgrid::WebGridReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto report = replay.run(recorder.recording().view(replay.provenance()));
    CHECK(report.verdict == ReplayVerdict::match);

    // And under a severity that would not have invalidated it, the run and the
    // recording disagree about whether the target's timing is a measurement.
    auto lenient = config;
    lenient.abnormal.input_discontinuity = AbnormalPolicy::record;
    webgrid::WebGridReplay lenient_replay{};
    CHECK(lenient_replay.prepare(lenient) == ContractStatus::ok);
    const auto lenient_report =
        lenient_replay.run(recorder.recording().view(lenient_replay.provenance()));
    CHECK(lenient_report.verdict == ReplayVerdict::mismatch);
}

void test_recorded_speech_run_replays_identically()
{
    const auto config = speech_config();
    auto recorder = record_speech(config);
    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    CHECK(replay.authority() == ReplayAuthority::regenerate);

    const auto provenance = replay.provenance(recorder.recording().schedules);
    const auto report = replay.run(recorder.recording().view(provenance));
    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(report.completeness == ReplayCompleteness::complete);
    CHECK(!recorder.recording().requests.empty());
    CHECK(!recorder.recording().phases.empty());
    CHECK(recorder.recording().trials.size() == config.task.n_trials);
    // The realized durations were drawn again rather than read back, so a
    // recording whose durations do not match the seed is caught.
    CHECK(recorder.recording().schedules.size() == config.task.n_trials);
    CHECK(recorder.recording().schedules[0].black_duration_ns > 0);
}

void test_disagreeing_speech_schedule_is_found()
{
    const auto config = speech_config();
    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    auto recorder = record_speech(config);
    auto& recording = recorder.recording();
    const auto provenance = replay.provenance(recording.schedules);
    // Candidate provenance belongs to the schedule regenerated from the seed.
    // The recorded schedule remains an output stream and is compared as one.
    recording.schedules[1].content_duration_ns += 1;
    const auto report = replay.run(recording.view(provenance));
    CHECK(report.verdict == ReplayVerdict::mismatch);
    CHECK(report.rejection == ReplayRejection::none);
    CHECK(report.first_mismatch.item == ReplayItem::schedule);
    CHECK(report.first_mismatch.field == "content_duration_ns");
}

void test_speech_schedule_stream_cardinality_and_absence_are_explicit()
{
    const auto config = speech_config();
    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);

    {
        auto recorder = record_speech(config);
        const auto provenance = replay.provenance(recorder.recording().schedules);
        auto view = recorder.recording().view(provenance);
        view.expected.has_schedules = false;
        view.expected.schedules = {};
        const auto report = replay.run(view);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(report.completeness == ReplayCompleteness::stream_absent);
        CHECK(report.rejection == ReplayRejection::none);
    }
    {
        auto recorder = record_speech(config);
        auto& schedules = recorder.recording().schedules;
        const auto provenance = replay.provenance(schedules);
        schedules.pop_back();
        const auto report = replay.run(recorder.recording().view(provenance));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.rejection == ReplayRejection::none);
        CHECK(report.first_mismatch.item == ReplayItem::stream_length);
        CHECK(report.first_mismatch.field == "recorded_short");
        CHECK(report.first_mismatch.recorded == schedules.size());
        CHECK(report.first_mismatch.regenerated == config.task.n_trials);
    }
    {
        auto recorder = record_speech(config);
        auto& schedules = recorder.recording().schedules;
        const auto provenance = replay.provenance(schedules);
        schedules.pop_back();
        auto view = recorder.recording().view(provenance);
        view.trace_loss_recorded = true;
        const auto report = replay.run(view);
        CHECK(report.verdict == ReplayVerdict::incomplete);
        CHECK(report.rejection == ReplayRejection::none);
        CHECK(report.completeness == ReplayCompleteness::trace_loss_recorded);
    }
    {
        auto recorder = record_speech(config);
        auto& schedules = recorder.recording().schedules;
        const auto provenance = replay.provenance(schedules);
        auto extra = schedules.back();
        extra.ordinal = static_cast<TrialOrdinal>(schedules.size());
        schedules.push_back(extra);
        const auto report = replay.run(recorder.recording().view(provenance));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.item == ReplayItem::stream_length);
        CHECK(report.first_mismatch.recorded == schedules.size());
        CHECK(report.first_mismatch.regenerated == config.task.n_trials);
    }
}

void test_explicit_speech_falls_back_to_configuration()
{
    auto config = explicit_speech_config();
    config.task.sampler_version = kCurrentSamplerVersion + 21;
    for (std::size_t i = 0; i < config.task.n_explicit; ++i)
    {
        config.task.explicit_schedule[i].sampler_version = config.task.sampler_version;
    }
    auto recorder = record_speech(config);
    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    CHECK(replay.authority() == ReplayAuthority::recorded_schedule);

    const auto provenance = replay.provenance(recorder.prepared_schedules());
    auto view = recorder.recording().view(provenance);
    view.expected.has_schedules = false;
    view.expected.schedules = {};
    const auto report = replay.run(view);
    CHECK(report.verdict == ReplayVerdict::incomplete);
    CHECK(report.rejection == ReplayRejection::none);
    CHECK(report.completeness == ReplayCompleteness::stream_absent);
    CHECK(report.inputs_replayed == recorder.recording().inputs.size());
}

void test_speech_schedule_cardinality_follows_started_trials()
{
    auto config = speech_config();
    config.task.n_trials = 4;
    SpeechRecorder recorder{config};
    recorder.start(1'000);
    recorder.halt(1'050);
    CHECK(recorder.recording().schedules.size() == 1);
    CHECK(recorder.prepared_schedules().size() == config.task.n_trials);

    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto provenance = replay.provenance(recorder.prepared_schedules());
    const auto report = replay.run(recorder.recording().view(provenance));
    CHECK(report.verdict == ReplayVerdict::match);
    CHECK(report.rejection == ReplayRejection::none);
    CHECK(report.completeness == ReplayCompleteness::complete);

    auto future_config = config;
    future_config.task.sampler_version = kCurrentSamplerVersion + 9;
    auto future_recording = recorder.recording();
    auto future_schedules = recorder.prepared_schedules();
    for (auto& schedule : future_schedules)
    {
        schedule.sampler_version = future_config.task.sampler_version;
    }
    for (auto& schedule : future_recording.schedules)
    {
        schedule.sampler_version = future_config.task.sampler_version;
    }

    speech::SpeechReplay fallback{};
    CHECK(fallback.prepare(future_config) == ContractStatus::ok);
    CHECK(fallback.authority() == ReplayAuthority::recorded_schedule);
    const auto future_provenance = fallback.provenance(future_schedules);
    const auto fallback_report = fallback.run(future_recording.view(future_provenance));
    CHECK(fallback_report.verdict == ReplayVerdict::match);
    CHECK(fallback_report.rejection == ReplayRejection::none);

    future_recording.schedules.clear();
    const auto missing = fallback.run(future_recording.view(future_provenance));
    CHECK(missing.verdict == ReplayVerdict::rejected);
    CHECK(missing.rejection == ReplayRejection::sampler_version_unsupported);
}

void test_speech_replay_imposes_no_latency_window()
{
    auto config = speech_config();
    config.task.n_trials = 70;
    SpeechRecorder recorder{config};
    recorder.start(1'000);

    ExperimentTimeNs time_ns = 1'000;
    while (recorder.recording().trials.size() < 66)
    {
        time_ns += 500;
        recorder.advance(time_ns);
    }
    const auto delayed = recorder.report_for(0, PresentationStatus::presented, 1);
    recorder.report(delayed);
    CHECK(recorder.recording().reports.back().matched);
    while (recorder.recording().trials.size() < config.task.n_trials)
    {
        time_ns += 500;
        recorder.advance(time_ns);
    }

    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto provenance = replay.provenance(recorder.recording().schedules);
    CHECK(replay.run(recorder.recording().view(provenance)).verdict == ReplayVerdict::match);
}

void test_disagreeing_speech_phase_is_found()
{
    const auto config = speech_config();
    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    auto recorder = record_speech(config);
    auto& recording = recorder.recording();
    const auto provenance = replay.provenance(recording.schedules);
    CHECK(!recording.phases.empty());
    recording.phases[0].phase.interval.end_ns += 1;
    const auto report = replay.run(recording.view(provenance));
    CHECK(report.verdict == ReplayVerdict::mismatch);
    CHECK(report.first_mismatch.item == ReplayItem::phase);
    CHECK(report.first_mismatch.field == "end_ns");
}

void test_presenter_report_is_evidence_not_replayed()
{
    // The presentation-time boundary. A presenter's `presented_ns` is never regenerated and
    // never compared against a regenerated instant; what a replay reproduces is
    // whether the run could attribute the report to a request it emitted, and
    // what the configured severity made of it.
    auto config = speech_config();
    config.abnormal.presentation_failed = AbnormalPolicy::abort_trial;

    SpeechRecorder recorder{config};
    recorder.start(1'000);
    recorder.advance(1'150);
    const auto failed = recorder.report_for(0, PresentationStatus::skipped, 1);
    CHECK(failed.request_sequence != 0);
    recorder.report(failed);
    for (ExperimentTimeNs time = 1'200; time <= 1'000 + 2'000; time += 50)
    {
        recorder.advance(time);
    }

    auto& recording = recorder.recording();
    CHECK(recording.reports.size() == 1);
    CHECK(recording.reports[0].matched);
    CHECK(recording.reports[0].condition == AbnormalCondition::presentation_failed);
    CHECK(recording.reports[0].response == AbnormalResponse::trial_invalidated);
    CHECK(recording.trials[0].invalidated);
    CHECK(recording.trials[0].recorded_outcome == TrialOutcome::aborted);

    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto provenance = replay.provenance(recording.schedules);
    CHECK(replay.run(recording.view(provenance)).verdict == ReplayVerdict::match);

    // Changing what the report *claims to answer* does change the decision, and
    // that decision belongs to the controller alone.
    {
        auto altered = recorder.recording();
        for (auto& input : altered.inputs)
        {
            if (input.kind == speech::SpeechReplayInputKind::presentation_report)
            {
                input.outcome.request_sequence += 1'000;
            }
        }
        const auto report = replay.run(altered.view(provenance));
        CHECK(report.verdict == ReplayVerdict::mismatch);
        CHECK(report.first_mismatch.field == "matched_request");
    }
    // And under a severity the run did not use, the response disagrees.
    {
        auto lenient = config;
        lenient.abnormal.presentation_failed = AbnormalPolicy::record;
        speech::SpeechReplay lenient_replay{};
        CHECK(lenient_replay.prepare(lenient) == ContractStatus::ok);
        const auto lenient_provenance = lenient_replay.provenance(recording.schedules);
        const auto report = lenient_replay.run(recording.view(lenient_provenance));
        CHECK(report.verdict == ReplayVerdict::mismatch);
    }
}

void test_presenter_reported_instant_is_never_regenerated()
{
    // The other half of the presentation-time boundary. A presenter-reported software
    // presentation instant is the one measurement the controller has no way to produce, so
    // moving it changes nothing a replay compares -- while every decision the
    // run took about the report is still reproduced exactly.
    const auto config = speech_config();
    SpeechRecorder recorder{config};
    recorder.start(1'000);
    recorder.advance(1'150);
    const auto presented = recorder.report_for(0, PresentationStatus::presented, 1);
    CHECK(presented.presented_ns > presented.requested_ns);
    recorder.report(presented);
    for (ExperimentTimeNs time = 1'200; time <= 1'000 + 2'000; time += 50)
    {
        recorder.advance(time);
    }

    auto& recording = recorder.recording();
    CHECK(recording.reports.size() == 1);
    CHECK(recording.reports[0].matched);
    CHECK(recording.reports[0].condition == AbnormalCondition::unspecified);
    CHECK(!recording.trials[0].invalidated);

    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    const auto provenance = replay.provenance(recording.schedules);
    CHECK(replay.run(recording.view(provenance)).verdict == ReplayVerdict::match);

    auto altered = recorder.recording();
    for (auto& input : altered.inputs)
    {
        if (input.kind == speech::SpeechReplayInputKind::presentation_report)
        {
            input.outcome.presented_ns += 1'000'000;
        }
    }
    CHECK(replay.run(altered.view(provenance)).verdict == ReplayVerdict::match);
}

void test_unregeneratable_sampler_falls_back_or_is_rejected()
{
    auto config = speech_config();
    auto recorder = record_speech(config);
    auto recording = recorder.recording();

    // A configuration naming a sampler this build does not implement is legal.
    // ScheduleIdentity says so, and the route through it is the recorded
    // realized schedule -- which is why a recording without one is refused
    // rather than replayed under whatever sampler happens to be current.
    config.task.sampler_version = kCurrentSamplerVersion + 7;
    for (auto& schedule : recording.schedules)
    {
        schedule.sampler_version = config.task.sampler_version;
    }
    speech::SpeechReplay replay{};
    CHECK(replay.prepare(config) == ContractStatus::ok);
    CHECK(replay.authority() == ReplayAuthority::recorded_schedule);

    auto view = recording.view(replay.provenance(recording.schedules));
    view.expected.has_schedules = false;
    const auto refused = replay.run(view);
    CHECK(refused.verdict == ReplayVerdict::rejected);
    CHECK(refused.rejection == ReplayRejection::sampler_version_unsupported);
}

void test_report_field_names_are_stable()
{
    CHECK(std::string_view{replay_verdict_name(ReplayVerdict::match)} == "match");
    CHECK(std::string_view{replay_verdict_name(ReplayVerdict::incomplete)} == "incomplete");
    CHECK(std::string_view{replay_rejection_name(ReplayRejection::seed_mismatch)} ==
          "seed-mismatch");
    CHECK(std::string_view{replay_rejection_name(ReplayRejection::metric_version_unsupported)} ==
          "metric-version-unsupported");
    CHECK(std::string_view{replay_item_name(ReplayItem::guidance_sample)} == "guidance-sample");
    CHECK(std::string_view{replay_completeness_name(ReplayCompleteness::stream_absent)} ==
          "stream-absent");
    // Every declared value has a name of its own, so a report cannot print two
    // findings the same way.
    for (std::uint8_t left = 0; left <= static_cast<std::uint8_t>(ReplayItem::stream_length);
         ++left)
    {
        for (std::uint8_t right = static_cast<std::uint8_t>(left + 1);
             right <= static_cast<std::uint8_t>(ReplayItem::stream_length); ++right)
        {
            CHECK(std::string_view{replay_item_name(static_cast<ReplayItem>(left))} !=
                  std::string_view{replay_item_name(static_cast<ReplayItem>(right))});
        }
    }
    CHECK(std::string_view{replay_verdict_name(static_cast<ReplayVerdict>(200))} == "undeclared");
    CHECK(std::string_view{replay_item_name(static_cast<ReplayItem>(200))} == "undeclared");
}

int run()
{
    test_recorded_center_out_run_replays_identically();
    test_center_out_boundary_saturation_replays_identically();
    test_repeated_replay_answers_identically();
    test_foreign_experiment_recording_is_rejected_early();
    test_unrecorded_configuration_is_rejected();
    test_altered_input_is_reported_at_first_difference();
    test_altered_transition_is_reported_as_itself();
    test_recording_cardinality_mismatch_is_reported();
    test_absent_stream_does_not_hide_cardinality_errors();
    test_abnormally_ended_trial_replays_as_such();
    test_replay_under_foreign_severity_disagrees();
    test_incomplete_evidence_is_never_a_match();
    test_unconfigured_replay_verifies_nothing();
    test_impossible_inputs_are_rejected();
    test_sampler_support_required_only_on_draw();
    test_webgrid_names_unsupported_metric_formula();
    test_recorded_webgrid_run_replays_identically();
    test_disagreeing_webgrid_selection_is_found();
    test_gap_crossed_webgrid_target_replays_invalidated();
    test_recorded_speech_run_replays_identically();
    test_disagreeing_speech_schedule_is_found();
    test_speech_schedule_stream_cardinality_and_absence_are_explicit();
    test_explicit_speech_falls_back_to_configuration();
    test_speech_schedule_cardinality_follows_started_trials();
    test_disagreeing_speech_phase_is_found();
    test_presenter_report_is_evidence_not_replayed();
    test_presenter_reported_instant_is_never_regenerated();
    test_speech_replay_imposes_no_latency_window();
    test_unregeneratable_sampler_falls_back_or_is_rejected();
    test_report_field_names_are_stable();
    return failures == 0 ? 0 : 1;
}

} // namespace

int main()
{
    return run();
}
