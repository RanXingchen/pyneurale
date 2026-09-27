/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "center_out_trace_writer.h"

#include <span>

#include <neurale/experiments/schedule.h>

namespace neurale::execution
{

using namespace neurale::experiments;
namespace
{
/// The version of the transform that produced the assisted velocity, by method.
/// The contract declares one supported version per method and refuses a record
/// naming another, so this is a lookup rather than a policy.
[[nodiscard]] assistance::AssistanceVersion
assistance_version(assistance::AssistanceMethod method) noexcept
{
    switch (method)
    {
    case assistance::AssistanceMethod::linear_blend:
        return assistance::kLinearBlendVersion1;
    case assistance::AssistanceMethod::ortho_impedance:
        return assistance::kOrthoImpedanceVersion1;
    default:
        return 0;
    }
}

void write_components(JsonWriter& text, std::string_view key,
                      const assistance::VelocityVector& vel) noexcept
{
    text.array_f64(key, std::span<const double>{vel.values.data(), vel.dim});
}
} // namespace

ContractStatus CenterOutTraceWriter::validate() const noexcept
{
    if (!controller_.prepared())
    {
        return ContractStatus::not_running;
    }
    if (const auto status = center_out::validate(config().task); status != ContractStatus::ok)
    {
        return status;
    }
    if (const auto status = experiments::validate(config().velocity_space);
        status != ContractStatus::ok)
    {
        return status;
    }
    if (config().paradigm == kUnsetParadigmId)
    {
        return ContractStatus::identity_missing;
    }
    if (config().assistance_method == assistance::AssistanceMethod::linear_blend)
    {
        if (const auto status = assistance::validate(config().linear_assistance);
            status != ContractStatus::ok)
        {
            return status;
        }
    }
    // Guidance is optional configuration, and an unconfigured guidance is not a
    // broken one: the machine simply has no reference velocity to blend.
    if (config().guidance.geometry_unit != center_out::GeometryUnit::unspecified)
    {
        if (const auto status = center_out::validate(config().guidance);
            status != ContractStatus::ok)
        {
            return status;
        }
        if (const auto status = center_out::validate_against(config().guidance, config().task);
            status != ContractStatus::ok)
        {
            return status;
        }
    }
    return ContractStatus::ok;
}

streaming::StreamStatus CenterOutTraceWriter::start_execution(ExperimentTimeNs time_ns) noexcept
{
    presentation_reports_ = 0;
    presentation_failures_ = 0;
    successful_presentations_ = 0;
    // Configure this writer's own reporter from the run's policy set, exactly as
    // the controller configured its own at prepare(). Presentation evidence is
    // drained here on the bridge thread, so the writer decides presentation
    // failures under the same policies rather than racing the controller's
    // non-atomic reporter.
    abnormal_.configure(controller_.configuration().abnormal, controller_.configuration().paradigm);
    abnormal_.restart();
    return controller_.start(host_epoch_ns_, time_ns);
}

bool CenterOutTraceWriter::report_presentation(
    const PresentationSoftwareEvidence& evidence) noexcept
{
    return presentations_.try_push(evidence) == streaming::StreamStatus::ok;
}

bool CenterOutTraceWriter::report_decoder_publication(
    std::uint64_t version, std::string_view plan_fingerprint, std::uint64_t training_blocks,
    std::uint64_t training_trials, TrialOrdinal completed_trials, ExperimentTimeNs time_ns) noexcept
{
    if (version == 0 || plan_fingerprint.size() != kDecoderPlanFingerprintBytes)
        return false;
    DecoderPublicationEvidence evidence{
        .version = version,
        .time_ns = time_ns,
        .training_blocks = training_blocks,
        .training_trials = training_trials,
        .completed_trials = completed_trials,
    };
    for (std::size_t i = 0; i < plan_fingerprint.size(); ++i)
    {
        const auto byte = plan_fingerprint[i];
        if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f')))
            return false;
        evidence.plan_fingerprint[i] = byte;
    }
    return decoder_publications_.try_push(evidence) == streaming::StreamStatus::ok;
}

bool CenterOutTraceWriter::report_presentation_config(
    const CenterOutPresentationConfigRecord& config) noexcept
{
    if (configuration_written_ || has_presentation_config_)
    {
        return false;
    }
    presentation_config_ = config;
    has_presentation_config_ = true;
    return true;
}

void CenterOutTraceWriter::stop_execution(bool) noexcept
{
    // The producer is quiet by the time the session calls this -- an attached
    // runtime has been stopped and joined -- which is what makes it safe to let
    // a latched halt produce its record here rather than where it was latched.
    static_cast<void>(controller_.apply_presentation_failures());
    controller_.finish_halt();
    controller_.cancel();
}

AbnormalSummary CenterOutTraceWriter::abnormal_summary() const noexcept
{
    return merge(controller_.abnormal_summary(), abnormal_.summary());
}

void CenterOutTraceWriter::halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept
{
    controller_.halt(time_ns, condition);
}

std::uint64_t CenterOutTraceWriter::dropped_trace_count() const noexcept
{
    return controller_.dropped_trace_count() + presentations_.dropped() +
           decoder_publications_.dropped();
}

SessionMetadata CenterOutTraceWriter::metadata() const noexcept
{
    const auto configuration = center_out::configuration_fingerprint(config().task);
    // The configuration's sampler version, not this build's. A configuration
    // may legally name a sampler this build cannot regenerate -- the contract
    // says so explicitly -- and Center-Out can still run under one, because
    // nothing here re-executes the sampler. Recording the build's version
    // instead would produce a session whose configuration fingerprint and whose
    // stated sampler contradict each other.
    const auto sampler = config().task.sampler_version;
    const ScheduleIdentity identity{
        .seed = config().task.seed,
        .sampler_version = sampler,
        .configuration_fingerprint = configuration,
        .catalog_fingerprint = 0,
    };
    return SessionMetadata{
        .experiment_type = kCenterOutExperimentType,
        .experiment_version = kCenterOutExperimentVersion,
        .configuration_fingerprint = configuration,
        .schedule_fingerprint = schedule_fingerprint(identity),
        // Center-Out draws its targets as the run goes rather than freezing a
        // schedule beforehand, so there is no realized schedule to digest. Zero
        // says that, and says it in the same field every paradigm uses.
        .realized_schedule_fingerprint = 0,
        .schedule_seed = config().task.seed,
        .sampler_version = sampler,
        .input_schema_fingerprint = input_schema_,
        .output_schema_fingerprint = output_schema_,
        .metric_version = 0,
        .policy_version = assistance_version(config().assistance_method),
    };
}

void CenterOutTraceWriter::write_configuration(ExperimentTraceSink& sink,
                                               ExperimentTimeNs time_ns) noexcept
{
    configuration_written_ = true;
    auto& text = sink.named(ControlKind::task_variables, "center_out.configuration", time_ns);
    text.u64("configuration_fingerprint", center_out::configuration_fingerprint(config().task));
    text.u64("layout_fingerprint", center_out::layout_fingerprint(config().task.layout));
    text.u64("seed", config().task.seed);
    text.u64("paradigm", config().paradigm);
    text.u64("decoded_signal_id", config().decoded_signal_id);
    text.u64("command_space", config().velocity_space.id);
    text.u64("command_dimension", config().velocity_space.dim);
    text.u64("command_frame", static_cast<std::uint64_t>(config().velocity_space.frame));
    text.f64("cursor_min_x", config().cursor_min.x);
    text.f64("cursor_min_y", config().cursor_min.y);
    text.f64("cursor_max_x", config().cursor_max.x);
    text.f64("cursor_max_y", config().cursor_max.y);
    static_cast<void>(sink.commit());

    // Assistance provenance, stated once as configuration and repeated per
    // observation as what was actually applied. The two have to agree, and a
    // reader that only has one of them cannot check that they do.
    auto& method = sink.named(ControlKind::assistance, "center_out.assistance_method", time_ns);
    method.u64("method", static_cast<std::uint64_t>(config().assistance_method));
    method.u64("version", assistance_version(config().assistance_method));
    method.f64("assistance", config().linear_assistance.assistance);
    method.u64("block_count", config().assistance_blocks.size());
    static_cast<void>(sink.value(config().linear_assistance.assistance));
    static_cast<void>(sink.commit());

    for (std::size_t i = 0; i < config().assistance_blocks.size(); ++i)
    {
        const auto& block = config().assistance_blocks[i];
        auto& schedule =
            sink.named(ControlKind::assistance, "center_out.assistance_block", time_ns);
        schedule.u64("index", i);
        schedule.u64("trials", block.trials);
        schedule.f64("assistance", block.linear.assistance);
        static_cast<void>(sink.commit());
    }

    auto& guidance = sink.named(ControlKind::assistance, "center_out.guidance_config", time_ns);
    guidance.u64("geometry_unit", static_cast<std::uint64_t>(config().guidance.geometry_unit));
    guidance.f64("max_speed", config().guidance.max_speed);
    guidance.f64("acceleration", config().guidance.acceleration);
    guidance.f64("deceleration", config().guidance.deceleration);
    guidance.f64("precision", config().guidance.precision);
    guidance.u64("resolver_version", config().guidance_resolver_version);
    static_cast<void>(sink.commit());

    auto& layout = sink.named(ControlKind::targets, "center_out.layout", time_ns);
    layout.f64("center_x", config().task.layout.center.pos.x);
    layout.f64("center_y", config().task.layout.center.pos.y);
    layout.u64("center_id", config().task.layout.center.id);
    layout.u64("surrounding_count", config().task.layout.count);
    static_cast<void>(sink.commit());

    for (std::uint8_t i = 0; i < config().task.layout.count; ++i)
    {
        const auto& target = config().task.layout.surrounding[i];
        auto& text_target = sink.named(ControlKind::targets, "center_out.target", time_ns);
        text_target.u64("index", i);
        text_target.u64("target_id", target.id);
        text_target.f64("x", target.pos.x);
        text_target.f64("y", target.pos.y);
        static_cast<void>(sink.value(static_cast<std::uint64_t>(target.id)));
        static_cast<void>(sink.commit());
    }

    write_presentation_config(sink, time_ns);
}

std::size_t CenterOutTraceWriter::drain(ExperimentTraceSink& sink, std::size_t budget) noexcept
{
    std::size_t taken = 0;
    PresentationSoftwareEvidence presentation{};
    while (taken < budget && presentations_.try_pop(presentation) == streaming::StreamStatus::ok)
    {
        write_presentation(sink, presentation);
        ++taken;
    }
    DecoderPublicationEvidence publication{};
    while (taken < budget &&
           decoder_publications_.try_pop(publication) == streaming::StreamStatus::ok)
    {
        write_decoder_publication(sink, publication);
        ++taken;
    }
    CenterOutControlTrace trace{};
    while (taken < budget && controller_.try_pop_trace(trace) == streaming::StreamStatus::ok)
    {
        write_trace(sink, trace);
        ++taken;
    }
    return taken;
}

void CenterOutTraceWriter::write_decoder_publication(
    ExperimentTraceSink& sink, const DecoderPublicationEvidence& evidence) noexcept
{
    auto& text =
        sink.named(ControlKind::task_variables, "center_out.decoder_publication", evidence.time_ns);
    text.u64("version", evidence.version);
    text.text("plan_fingerprint",
              std::string_view{evidence.plan_fingerprint.data(), evidence.plan_fingerprint.size()});
    text.u64("training_blocks", evidence.training_blocks);
    text.u64("training_trials", evidence.training_trials);
    text.u64("completed_trials", evidence.completed_trials);
    static_cast<void>(sink.commit());
}

void CenterOutTraceWriter::write_presentation(ExperimentTraceSink& sink,
                                              const PresentationSoftwareEvidence& evidence) noexcept
{
    const bool failed = evidence.event == PresentationLifecycleEvent::faulted;
    // The implementation evidence is always a named record: what the renderer did
    // (event, implementation_status, software timing) is separate from what the
    // run does about a failure. A faulted frame therefore produces two records --
    // this one and the task owner's abnormal-condition row -- rather than one generic
    // fault row that conflates "what happened" with "how the experiment handles
    // it". The abnormal row's code is the condition's contract name, so the two
    // can never collide as two records both named presentation-failed.
    auto& text =
        sink.named(ControlKind::experiment_states, "center_out.presentation", evidence.time_ns);
    text.u64("event", static_cast<std::uint64_t>(evidence.event));
    text.u64("implementation_status", evidence.implementation_status);
    text.u64("update_ordinal", evidence.update_ordinal);
    text.boolean("has_source_ordinal", evidence.has_source_ordinal);
    if (evidence.has_source_ordinal)
        text.u64("source_ordinal", evidence.source_ordinal);
    text.boolean("has_trial", evidence.has_trial);
    if (evidence.has_trial)
        text.trial("trial", evidence.trial);
    text.boolean("has_software_times", evidence.has_software_times);
    if (evidence.has_software_times)
    {
        text.u64("requested_ns", evidence.requested_ns);
        text.u64("intended_ns", evidence.intended_ns);
        text.u64("submitted_renderer_ns", evidence.submitted_renderer_ns);
        text.u64("submitted_ns", evidence.submitted_ns);
        text.u64("presented_renderer_ns", evidence.presented_renderer_ns);
        text.u64("presented_ns", evidence.presented_ns);
    }
    static_cast<void>(sink.commit());
    ++presentation_reports_;
    presentation_failures_ += static_cast<std::uint64_t>(failed);
    successful_presentations_ += static_cast<std::uint64_t>(
        evidence.event == PresentationLifecycleEvent::presented && evidence.has_software_times);
    if (!failed || evidence.has_trial)
    {
        return;
    }
    // Pre/post-run lifecycle failures have no task-owner callback to apply.
    // Frame and active runtime failures carry an exact trial and are handled by
    // the controller's task-owner entry point.
    const auto event = abnormal_.observe(AbnormalCondition::presentation_failed, evidence.time_ns,
                                         /*trial*/ nullptr, evidence.implementation_status,
                                         AbnormalPolicy::record, /*can_end_trial*/ false,
                                         /*input_refused*/ false);
    write_abnormal(sink, event);
    if (event.response == AbnormalResponse::session_aborted)
    {
        controller_.stop_accepting();
    }
}

void CenterOutTraceWriter::write_abnormal(ExperimentTraceSink& sink,
                                          const AbnormalEvent& event) noexcept
{
    auto& text = sink.abnormal(event);
    static_cast<void>(sink.commit());
    static_cast<void>(text);
}

void CenterOutTraceWriter::write_presentation_config(ExperimentTraceSink& sink,
                                                     ExperimentTimeNs time_ns) noexcept
{
    if (!has_presentation_config_)
    {
        return;
    }
    const auto& c = presentation_config_;
    auto& text = sink.named(ControlKind::task_variables, "center_out.presentation_config", time_ns);
    text.u64("geometry_unit", c.geometry_unit);
    text.f64("logical_left", c.logical_left);
    text.f64("logical_bottom", c.logical_bottom);
    text.f64("logical_width", c.logical_width);
    text.f64("logical_height", c.logical_height);
    text.f64("target_radius", c.target_radius);
    text.f64("cursor_radius", c.cursor_radius);
    text.u64("circle_segments", c.circle_segments);
    text.u64("aspect_policy", c.aspect_policy);
    static_cast<void>(sink.commit());

    // The scored geometry, and its relation to the drawn geometry above. The
    // two are independent by contract -- the hit test never reads a rendered
    // radius -- so this reconciles nothing; it makes "did the subject see what
    // the run scored" answerable from the recording instead of leaving it to be
    // rederived by joining the task configuration to this one. Its own record,
    // like the Speech font block, so both stay well inside the control-body
    // limit rather than growing toward it.
    auto& acceptance =
        sink.named(ControlKind::task_variables, "center_out.presentation_acceptance", time_ns);
    acceptance.f64("acceptance_half_extent_x", c.acceptance_half_extent_x);
    acceptance.f64("acceptance_half_extent_y", c.acceptance_half_extent_y);
    acceptance.f64("task_cursor_extent", c.task_cursor_extent);
    acceptance.f64("target_radius", c.target_radius);
    acceptance.f64("cursor_radius", c.cursor_radius);
    acceptance.boolean("target_circle_contains_acceptance", c.target_circle_contains_acceptance);
    acceptance.boolean("acceptance_contains_target_circle", c.acceptance_contains_target_circle);
    acceptance.boolean("cursor_radius_matches_task_extent", c.cursor_radius_matches_task_extent);
    static_cast<void>(sink.commit());

    auto& window =
        sink.named(ControlKind::task_variables, "center_out.presentation_window", time_ns);
    window.u64("window_width", static_cast<std::uint64_t>(c.window_width));
    window.u64("window_height", static_cast<std::uint64_t>(c.window_height));
    window.i64("monitor_index", c.monitor_idx);
    window.i64("swap_interval", c.swap_interval);
    window.boolean("fullscreen", c.fullscreen);
    window.boolean("resizable", c.resizable);
    window.boolean("visible", c.visible);
    static_cast<void>(sink.commit());

    const auto write_styles = [&sink, time_ns](std::string_view name,
                                               std::span<const PresentationConfigColour> colours,
                                               std::span<const std::string_view> names) noexcept
    {
        auto& style = sink.named(ControlKind::task_variables, name, time_ns);
        for (std::size_t i = 0; i < colours.size(); ++i)
        {
            const auto& colour = colours[i];
            const std::array<double, 4> components{colour.red, colour.green, colour.blue,
                                                   colour.alpha};
            style.array_f64(names[i], components);
        }
        static_cast<void>(sink.commit());
    };
    const std::array first_colours{c.background, c.center_target, c.outward_target, c.active_move};
    const std::array<std::string_view, 4> first_names{"background", "center_target",
                                                      "outward_target", "active_move"};
    write_styles("center_out.presentation_style_1", first_colours, first_names);
    const std::array second_colours{c.active_hold, c.success, c.failure, c.cursor};
    const std::array<std::string_view, 4> second_names{"active_hold", "success", "failure",
                                                       "cursor"};
    write_styles("center_out.presentation_style_2", second_colours, second_names);
}

void CenterOutTraceWriter::write_aborted_trial(ExperimentTraceSink& sink,
                                               const CenterOutControlTrace& trace) noexcept
{
    if (!trace.has_aborted_trial)
    {
        return;
    }
    // A trial the run ended is written as a trial, with TrialOutcome::aborted.
    // Leaving it out would be the silent version of the same fact: a reader
    // counting decided trials would find a gap and no reason for it.
    //
    // `reason` is the contract's zero rather than a CenterOutReason. That
    // enumeration is the machine's vocabulary for decisions the machine made,
    // and this is not one of them; the condition is in the abnormal record
    // written immediately before this one, stamped with the same instant and
    // carrying the same trial identity.
    auto& label = sink.trial(trace.aborted_trial);
    label.text("kind", "center_out.trial_aborted");
    label.text("condition", abnormal_condition_name(trace.abnormal.condition));
    label.u64("response", static_cast<std::uint64_t>(trace.abnormal.response));
    static_cast<void>(sink.commit());
    ++completed_;
}

void CenterOutTraceWriter::write_trace(ExperimentTraceSink& sink,
                                       const CenterOutControlTrace& trace) noexcept
{
    switch (trace.kind)
    {
    case CenterOutTraceKind::session_start:
        // The machine's start result -- the SESSION_START transition, the first
        // TRIAL_START event, and the target the run opens on -- travels in
        // `step`. `segment_start` is empty here, and writing that one instead
        // is how a recording loses the beginning of the run it records.
        write_step(sink, trace.step);
        write_target(sink, trace.step.snapshot, trace.time_ns);
        return;

    case CenterOutTraceKind::discontinuity:
    {
        auto& text = sink.abnormal(trace.abnormal);
        // A gap in the decoded stream is data about the run, not a verdict on
        // it: the record says a gap happened, what was decided about it, and
        // the session's own outcome says whether anything was lost.
        text.u64("frame_sequence", trace.frame_sequence);
        text.u64("gap_reason", static_cast<std::uint64_t>(trace.gap_reason));
        text.boolean("restarted", trace.restarted_after_discontinuity);
        static_cast<void>(sink.commit());
        write_aborted_trial(sink, trace);
        forget_target();
        return;
    }

    case CenterOutTraceKind::abnormal:
    {
        auto& text = sink.abnormal(trace.abnormal);
        text.f64("x", trace.position_after.x);
        text.f64("y", trace.position_after.y);
        static_cast<void>(sink.commit());
        write_aborted_trial(sink, trace);
        if (trace.abnormal.response == AbnormalResponse::trial_aborted ||
            trace.abnormal.response == AbnormalResponse::session_aborted)
        {
            // The segment this target belonged to is over. Forgetting it here
            // is what makes the next segment's opening target a new record
            // rather than a repeat suppressed as unchanged.
            forget_target();
        }
        return;
    }

    case CenterOutTraceKind::observation:
        break;
    }

    if (trace.has_abnormal)
    {
        // An observation that also carries a decision writes the decision
        // first: the interval that was refused ended before the observation
        // that revealed it, and a reader following the record order would
        // otherwise see the new segment begin before anything explained why.
        auto& text = sink.abnormal(trace.abnormal);
        text.f64("x", trace.position_before.x);
        text.f64("y", trace.position_before.y);
        static_cast<void>(sink.commit());
        write_aborted_trial(sink, trace);
    }

    if (trace.restarted_after_discontinuity)
    {
        // The first observation of a restarted segment carries that segment's
        // own start result beside its own step. Both belong in the recording:
        // without the first, the replay sees a segment that never started; with
        // only the first, it sees a start that produced no motion.
        forget_target();
        write_step(sink, trace.segment_start);
        write_target(sink, trace.segment_start.snapshot, trace.time_ns);
    }

    write_velocity(sink, trace);
    write_step(sink, trace.step);

    const auto& snapshot = trace.step.snapshot;
    auto& cursor = sink.named(ControlKind::labels, "center_out.cursor", trace.time_ns);
    cursor.f64("x", trace.position_after.x);
    cursor.f64("y", trace.position_after.y);
    cursor.f64("previous_x", trace.position_before.x);
    cursor.f64("previous_y", trace.position_before.y);
    cursor.u64("dt_ns", trace.dt_ns);
    cursor.u64("observation_ordinal", trace.observation_ordinal);
    cursor.u64("frame_sequence", trace.frame_sequence);
    cursor.u64("sample_index", trace.sample_idx);
    cursor.u64("state", static_cast<std::uint64_t>(snapshot.state));
    cursor.u64("phase", static_cast<std::uint64_t>(snapshot.phase));
    cursor.boolean("contained", snapshot.contained);
    cursor.interval("leg", snapshot.leg);
    cursor.interval("hold", snapshot.hold);
    cursor.interval("dwell", snapshot.dwell);
    cursor.trial("trial", snapshot.trial);
    static_cast<void>(sink.commit());

    write_target(sink, snapshot, trace.time_ns);

    if (trace.step.trial_decided)
    {
        const auto& trial = trace.step.trial;
        auto& label = sink.trial(trial.record);
        label.trial("trial", trial.record.trial);
        label.u64("paradigm", trial.record.paradigm);
        label.u64("outward_target", trial.outward_target);
        label.u64("outward_index", trial.outward_idx);
        label.u64("decided_phase", static_cast<std::uint64_t>(trial.decided_phase));
        label.u64("center_acquire_ns", trial.center_acquire_ns);
        label.u64("outward_acquire_ns", trial.outward_acquire_ns);
        static_cast<void>(sink.commit());
    }
    completed_ = snapshot.completed;
    successes_ = snapshot.successes;
}

void CenterOutTraceWriter::write_step(ExperimentTraceSink& sink,
                                      const center_out::CenterOutStepResult& step) noexcept
{
    for (std::uint8_t i = 0; i < step.n_transitions; ++i)
    {
        const auto& transition = step.transitions[i];
        auto& text =
            sink.named(ControlKind::experiment_states, "center_out.transition", transition.time_ns);
        text.u64("sequence", transition.sequence);
        text.u64("paradigm", transition.paradigm);
        text.u64("from_state", transition.from_state);
        text.u64("to_state", transition.to_state);
        text.u64("cause", transition.cause);
        text.trial("trial", transition.trial);
        if (i + 1 == step.n_transitions)
        {
            // A command links to the last transition emitted before it. The
            // transition's TrialIdentity names trial scheduling identity, not
            // necessarily the target active after all transitions in this
            // step, so persist that final state fact explicitly.
            text.u64("active_target", step.snapshot.active_target);
            text.u64("active_state", static_cast<std::uint64_t>(step.snapshot.state));
        }
        static_cast<void>(sink.value(static_cast<std::uint64_t>(transition.to_state)));
        static_cast<void>(sink.commit());
        active_state_sequence_ = transition.sequence;
    }
    for (std::uint8_t i = 0; i < step.n_events; ++i)
    {
        const auto& event = step.events[i];
        auto& text = sink.named(ControlKind::events, "center_out.event", event.time_ns);
        text.u64("sequence", event.sequence);
        text.u64("kind", static_cast<std::uint64_t>(event.kind));
        text.u64("paradigm", event.paradigm);
        text.u64("code", event.code);
        text.i64("value", event.value);
        text.trial("trial", event.trial);
        static_cast<void>(sink.value(static_cast<std::uint64_t>(event.kind)));
        static_cast<void>(sink.commit());
    }
}

void CenterOutTraceWriter::write_velocity(ExperimentTraceSink& sink,
                                          const CenterOutControlTrace& trace) noexcept
{
    // write_velocity() runs before write_step(). The decoded observation,
    // guidance, assistance, and command were all computed from that pre-step
    // state; transitions in trace.step are their effect and must not become
    // their parent.
    const auto state_sequence = active_state_sequence_;
    const bool guidance_applicable =
        config().guidance.geometry_unit != center_out::GeometryUnit::unspecified;
    const bool assistance_configured =
        config().assistance_method != assistance::AssistanceMethod::none;

    auto& provenance =
        sink.named(ControlKind::labels, "center_out.observation_provenance", trace.time_ns);
    provenance.u64("observation_ordinal", trace.observation_ordinal);
    provenance.u64("decoder_output_ordinal", trace.observation_ordinal);
    provenance.u64("decoder_version", trace.decoder_version);
    provenance.u64("frame_sequence", trace.frame_sequence);
    provenance.u64("sample_index", trace.sample_idx);
    provenance.u64("input_ready_ns", trace.input_ready_ns);
    provenance.u64("decoded_ready_ns", trace.decoded_ready_ns);
    provenance.u64("input_to_decoded_ns", trace.decoded_ready_ns >= trace.input_ready_ns
                                              ? trace.decoded_ready_ns - trace.input_ready_ns
                                              : 0);
    provenance.u64("state_sequence", state_sequence);
    provenance.u64("guidance_ordinal", trace.observation_ordinal);
    provenance.u64("assistance_ordinal", trace.observation_ordinal);
    provenance.boolean("guidance_applicable", guidance_applicable);
    provenance.boolean("assistance_configured", assistance_configured);
    static_cast<void>(sink.commit());

    auto& applied =
        sink.named(ControlKind::assistance, "center_out.assisted_velocity", trace.time_ns);
    applied.u64("observation_ordinal", trace.observation_ordinal);
    applied.u64("method", static_cast<std::uint64_t>(config().assistance_method));
    applied.u64("version", assistance_version(config().assistance_method));
    applied.f64("assistance", trace.linear_assistance.assistance);
    applied.u64("space", trace.assisted.space);
    applied.u64("dimension", trace.assisted.dim);
    write_components(applied, "decoded", trace.decoded);
    write_components(applied, "guidance", trace.guidance);
    write_components(applied, "assisted", trace.assisted);
    // What the applied velocity did to the cursor. The command and its effect
    // are one record on purpose: an assisted velocity that was computed and not
    // applied is a different fact from one that moved the cursor, and a reader
    // must not have to infer which happened from the next position record.
    applied.f64("applied_dx", trace.position_after.x - trace.position_before.x);
    applied.f64("applied_dy", trace.position_after.y - trace.position_before.y);
    const auto dt_seconds =
        static_cast<double>(trace.dt_ns) / static_cast<double>(kNanosecondsPerSecond);
    applied.f64("cursor_vx", dt_seconds == 0.0
                                 ? 0.0
                                 : (trace.position_after.x - trace.position_before.x) / dt_seconds);
    applied.f64("cursor_vy", dt_seconds == 0.0
                                 ? 0.0
                                 : (trace.position_after.y - trace.position_before.y) / dt_seconds);
    applied.u64("dt_ns", trace.dt_ns);
    applied.boolean("applied", trace.dt_ns != 0);
    applied.trial("trial", trace.step.snapshot.trial);
    static_cast<void>(sink.commit());

    const auto& sample = trace.guidance_sample;
    auto& guidance =
        sink.named(ControlKind::assistance, "center_out.guidance_sample", trace.time_ns);
    guidance.u64("observation_ordinal", trace.observation_ordinal);
    guidance.f64("vx", sample.vel.x);
    guidance.f64("vy", sample.vel.y);
    guidance.f64("distance", sample.distance);
    guidance.f64("speed_towards_target", sample.speed_towards_target);
    guidance.boolean("retargeted", sample.retargeted);
    guidance.u64("phase", static_cast<std::uint64_t>(sample.state.phase));
    guidance.u64("state_target_id", sample.state.target);
    static_cast<void>(sink.commit());

    auto& command = sink.named(ControlKind::commands, "center_out.command", trace.time_ns);
    command.u64("sequence", trace.observation_ordinal);
    command.u64("observation_ordinal", trace.observation_ordinal);
    command.u64("generated_ns", trace.time_ns);
    command.u64("valid_until_ns", kNoExpiryNs);
    command.u64("space", trace.assisted.space);
    command.u64("dimension", trace.assisted.dim);
    write_components(command, "external", trace.decoded);
    write_components(command, "guidance", trace.guidance);
    write_components(command, "final", trace.assisted);
    command.trial("trial", trace.step.snapshot.trial);
    static_cast<void>(sink.commit());

    auto& outcome = sink.named(ControlKind::commands, "center_out.command_outcome", trace.time_ns);
    outcome.u64("sequence", trace.observation_ordinal);
    outcome.u64("request_sequence", trace.observation_ordinal);
    outcome.u64("generated_ns", trace.time_ns);
    outcome.u64("submitted_ns", trace.time_ns);
    outcome.u64("application", static_cast<std::uint64_t>(CommandApplication::accepted));
    outcome.u64("status_code", static_cast<std::uint64_t>(streaming::StreamStatus::ok));
    // This is the synchronous application performed by CenterOutController
    // when it integrates the final velocity into its headless cursor. It is
    // neither a NativeActuator submission nor a device acknowledgement.
    outcome.text("application_scope", "center_out.headless_cursor");
    outcome.trial("trial", trace.step.snapshot.trial);
    static_cast<void>(sink.commit());
}

void CenterOutTraceWriter::forget_target() noexcept
{
    // A restarted segment begins from a machine that was reset, so whatever
    // target it puts up is a new onset -- even when it is the same cell that
    // was up before the gap. Keeping the cached identity would suppress exactly
    // that record and leave the replay with a segment that never had a target.
    has_target_ = false;
    last_target_ = kUnsetTargetId;
}

void CenterOutTraceWriter::write_target(ExperimentTraceSink& sink,
                                        const center_out::CenterOutSnapshot& snapshot,
                                        ExperimentTimeNs time_ns) noexcept
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
    auto& text = sink.named(ControlKind::targets, "center_out.active_target", time_ns);
    text.u64("target_id", snapshot.active_target);
    text.f64("x", snapshot.active_position.x);
    text.f64("y", snapshot.active_position.y);
    text.u64("phase", static_cast<std::uint64_t>(snapshot.phase));
    text.u64("outward_target", snapshot.outward_target);
    text.u64("outward_index", snapshot.outward_idx);
    text.interval("leg", snapshot.leg);
    text.trial("trial", snapshot.trial);
    static_cast<void>(sink.value(static_cast<std::uint64_t>(snapshot.active_target)));
    static_cast<void>(sink.commit());
}

void CenterOutTraceWriter::write_summary(ExperimentTraceSink& sink,
                                         ExperimentTimeNs time_ns) noexcept
{
    if (require_presentation_evidence_ && successful_presentations_ == 0)
    {
        // Route missing presentation evidence through the existing
        // presentation_evidence_missing condition rather than emitting a
        // same-named generic fault row. The coded name is unchanged, so a reader
        // counting presentation-evidence-missing records is unaffected; the
        // difference is that the condition is now counted and policy-aware.
        const auto event =
            abnormal_.observe(AbnormalCondition::presentation_evidence_missing, time_ns,
                              /*trial*/ nullptr, static_cast<std::uint32_t>(presentation_reports_),
                              AbnormalPolicy::record, /*can_end_trial*/ false,
                              /*input_refused*/ false);
        auto& missing = sink.abnormal(event);
        missing.u64("presentation_reports", presentation_reports_);
        static_cast<void>(sink.commit());
    }
    auto& text = sink.named(ControlKind::task_variables, "center_out.summary", time_ns);
    text.u64("completed_trials", completed_);
    text.u64("successful_trials", successes_);
    static_cast<void>(sink.commit());

    if (presentation_reports_ != 0 || require_presentation_evidence_)
    {
        auto& presentation =
            sink.named(ControlKind::experiment_states, "center_out.presentation_summary", time_ns);
        presentation.u64("presentation_reports", presentation_reports_);
        presentation.u64("presentation_failures", presentation_failures_);
        presentation.u64("successful_presentations", successful_presentations_);
        presentation.boolean("presentation_evidence_required", require_presentation_evidence_);
        static_cast<void>(sink.commit());
    }
}

} // namespace neurale::execution
