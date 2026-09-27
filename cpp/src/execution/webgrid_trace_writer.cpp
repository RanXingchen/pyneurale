/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "webgrid_trace_writer.h"

#include <array>
#include <span>

#include <neurale/experiments/schedule.h>

namespace neurale::execution
{

using namespace neurale::experiments;

bool WebGridTraceWriter::report_presentation(const PresentationSoftwareEvidence& evidence) noexcept
{
    return presentations_.try_push(evidence) == streaming::StreamStatus::ok;
}

bool WebGridTraceWriter::report_presentation_config(
    const WebGridPresentationConfigRecord& config) noexcept
{
    if (configuration_written_ || has_presentation_config_)
    {
        return false;
    }
    presentation_config_ = config;
    has_presentation_config_ = true;
    return true;
}

ContractStatus WebGridTraceWriter::validate() const noexcept
{
    // Not prepared means there is no frozen configuration to validate, and
    // validating the default-constructed one would pass or fail on a
    // configuration nothing is going to run.
    if (!controller_.prepared())
    {
        return ContractStatus::not_running;
    }
    if (paradigm_ == kUnsetParadigmId)
    {
        return ContractStatus::identity_missing;
    }
    return webgrid::validate(config());
}

streaming::StreamStatus WebGridTraceWriter::start_execution(ExperimentTimeNs time_ns) noexcept
{
    presentation_reports_ = 0;
    presentation_failures_ = 0;
    missing_presentation_inputs_ = 0;
    abnormal_.configure(controller_.controller_configuration().abnormal, paradigm_);
    abnormal_.restart();
    return controller_.start(paradigm_, time_ns);
}

void WebGridTraceWriter::stop_execution(bool) noexcept
{
    // cancel(), not close(): the controller is the caller's, the queued traces
    // are still owed to the drain that follows, and the difference between a
    // stop and an abort is the recording's, not the controller's.
    static_cast<void>(controller_.apply_presentation_failures());
    controller_.finish_halt();
    controller_.cancel();
}

AbnormalSummary WebGridTraceWriter::abnormal_summary() const noexcept
{
    return merge(controller_.abnormal_summary(), abnormal_.summary());
}

void WebGridTraceWriter::halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept
{
    controller_.halt(time_ns, condition);
}

std::uint64_t WebGridTraceWriter::dropped_trace_count() const noexcept
{
    return controller_.dropped_trace_count() + presentations_.dropped();
}

SessionMetadata WebGridTraceWriter::metadata() const noexcept
{
    const auto configuration = webgrid::configuration_fingerprint(config());
    const ScheduleIdentity identity{
        .seed = config().seed,
        .sampler_version = config().sampler_version,
        .configuration_fingerprint = configuration,
        // WebGrid presents cells rather than stimuli and has no catalog, which
        // the identity spells as zero rather than as an absent field.
        .catalog_fingerprint = 0,
    };
    return SessionMetadata{
        .experiment_type = kWebGridExperimentType,
        .experiment_version = kWebGridExperimentVersion,
        .configuration_fingerprint = configuration,
        .schedule_fingerprint = schedule_fingerprint(identity),
        .schedule_seed = config().seed,
        .sampler_version = config().sampler_version,
        .input_schema_fingerprint = 0,
        .output_schema_fingerprint = 0,
        .metric_version = config().metric_version,
        .policy_version = 0,
    };
}

void WebGridTraceWriter::write_configuration(ExperimentTraceSink& sink,
                                             ExperimentTimeNs time_ns) noexcept
{
    configuration_written_ = true;
    auto& text = sink.named(ControlKind::task_variables, "webgrid.configuration", time_ns);
    text.u64("fingerprint", webgrid::configuration_fingerprint(config()));
    text.u64("rows", config().rows);
    text.u64("columns", config().columns);
    text.u64("candidate_count", config().n_candidates);
    text.u64("explicit_count", config().n_explicit);
    text.u64("schedule", static_cast<std::uint64_t>(config().schedule));
    text.u64("immediate_repetition", static_cast<std::uint64_t>(config().immediate_repetition));
    text.u64("correct_selection", static_cast<std::uint64_t>(config().correct_selection));
    text.u64("incorrect_selection", static_cast<std::uint64_t>(config().incorrect_selection));
    text.u64("seed", config().seed);
    text.u64("sampler_version", config().sampler_version);
    text.u64("paradigm", paradigm_);
    text.u64("initial_target", config().initial_target);
    text.u64("session_duration_ns", config().session_duration_ns);
    text.u64("target_count_limit", config().target_count_limit);
    text.u64("metric_version", config().metric_version);
    static_cast<void>(sink.commit());

    // The pointer's provenance, stated once. Either the samples travel in this
    // trace or they are already in a named stream, and a replay has to be able
    // to tell which without inspecting how many pointer records it found.
    auto& pointer = sink.named(ControlKind::labels, "webgrid.pointer_source", time_ns);
    pointer.boolean("recorded_here", cursor_stream_.empty());
    pointer.text("stream", cursor_stream_);
    pointer.u64("source_ordinal", 0);
    static_cast<void>(sink.commit());

    write_presentation_config(sink, time_ns);
}

std::size_t WebGridTraceWriter::drain(ExperimentTraceSink& sink, std::size_t budget) noexcept
{
    std::size_t taken = 0;
    PresentationSoftwareEvidence presentation{};
    while (taken < budget && presentations_.try_pop(presentation) == streaming::StreamStatus::ok)
    {
        write_presentation(sink, presentation);
        ++taken;
    }
    WebGridHeadlessTrace trace{};
    while (taken < budget && controller_.try_pop_trace(trace) == streaming::StreamStatus::ok)
    {
        write_trace(sink, trace);
        ++taken;
    }
    return taken;
}

void WebGridTraceWriter::write_presentation(ExperimentTraceSink& sink,
                                            const PresentationSoftwareEvidence& evidence) noexcept
{
    const bool failed = evidence.event == PresentationLifecycleEvent::faulted;
    // Implementation evidence is always a named record; the task-relevant
    // failure is routed through the task owner's abnormal policy as a separate
    // condition row, so a faulted frame writes two records rather than one
    // generic fault row. See CenterOutTraceWriter::write_presentation for the
    // rationale.
    auto& text =
        sink.named(ControlKind::experiment_states, "webgrid.presentation", evidence.time_ns);
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
    if (!failed || evidence.has_trial)
    {
        return;
    }
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

void WebGridTraceWriter::write_abnormal(ExperimentTraceSink& sink,
                                        const AbnormalEvent& event) noexcept
{
    auto& text = sink.abnormal(event);
    static_cast<void>(sink.commit());
    static_cast<void>(text);
}

void WebGridTraceWriter::write_presentation_config(ExperimentTraceSink& sink,
                                                   ExperimentTimeNs time_ns) noexcept
{
    if (!has_presentation_config_)
    {
        return;
    }
    const auto& c = presentation_config_;
    auto& text = sink.named(ControlKind::task_variables, "webgrid.presentation_config", time_ns);
    text.f64("logical_min_x", c.logical_min_x);
    text.f64("logical_max_x", c.logical_max_x);
    text.f64("logical_min_y", c.logical_min_y);
    text.f64("logical_max_y", c.logical_max_y);
    text.f64("pointer_radius", c.pointer_radius);
    text.u64("circle_segments", c.circle_segments);
    text.i64("selection_button", c.selection_button);
    text.u64("aspect_policy", c.aspect_policy);
    static_cast<void>(sink.commit());

    auto& window = sink.named(ControlKind::task_variables, "webgrid.presentation_window", time_ns);
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
    const std::array first_colours{c.background, c.cell, c.active_target, c.correct_feedback};
    const std::array<std::string_view, 4> first_names{"background", "cell", "active_target",
                                                      "correct_feedback"};
    write_styles("webgrid.presentation_style_1", first_colours, first_names);
    const std::array second_colours{c.incorrect_feedback, c.grid_line, c.pointer};
    const std::array<std::string_view, 3> second_names{"incorrect_feedback", "grid_line",
                                                       "pointer"};
    write_styles("webgrid.presentation_style_2", second_colours, second_names);
}

void WebGridTraceWriter::write_trace(ExperimentTraceSink& sink,
                                     const WebGridHeadlessTrace& trace) noexcept
{
    if (trace.kind == WebGridTraceKind::abnormal)
    {
        auto& text = sink.abnormal(trace.abnormal);
        text.boolean("acquisition_timing_valid", trace.acquisition_timing_valid);
        static_cast<void>(sink.commit());
        if (trace.abnormal.response == AbnormalResponse::trial_invalidated)
        {
            ++invalidated_targets_;
        }
        return;
    }

    const auto& snapshot = trace.step.snapshot;
    metrics_ = snapshot.metrics;

    if (trace.kind != WebGridTraceKind::session_start)
    {
        if (require_presentation_input_evidence_ && !trace.presentation_input.available)
        {
            ++missing_presentation_inputs_;
        }
        auto& pointer =
            sink.named(ControlKind::labels,
                       cursor_stream_.empty() ? "webgrid.pointer" : "webgrid.pointer_reference",
                       snapshot.time_ns);
        pointer.u64("pointer_update_ordinal", trace.pointer_update_ordinal);
        pointer.u64("source_ordinal", 0);
        pointer.text("stream", cursor_stream_);
        if (cursor_stream_.empty())
        {
            pointer.f64("x", trace.pointer.x);
            pointer.f64("y", trace.pointer.y);
        }
        pointer.u64("state", static_cast<std::uint64_t>(snapshot.state));
        pointer.u64("active_target", snapshot.active_target);
        pointer.boolean("presentation_input_available", trace.presentation_input.available);
        if (trace.presentation_input.available)
        {
            pointer.u64("presentation_input_ordinal", trace.presentation_input.input_ordinal);
            pointer.u64("renderer_time_ns", trace.presentation_input.renderer_time_ns);
            pointer.u64("presentation_input_kind", trace.presentation_input.kind);
            pointer.boolean("inside_presentation", trace.presentation_input.inside_presentation);
            pointer.i64("button", trace.presentation_input.button);
            pointer.i64("modifiers", trace.presentation_input.modifiers);
        }
        pointer.trial("trial", snapshot.trial);
        static_cast<void>(sink.commit());
    }

    if (trace.step.selection_processed)
    {
        const auto& selection = trace.step.selection;
        auto& text = sink.named(ControlKind::events,
                                selection.event.correct ? "webgrid.selection_correct"
                                                        : "webgrid.selection_incorrect",
                                selection.event.time_ns);
        text.u64("sequence", selection.event.sequence);
        text.u64("pointer_update_ordinal", trace.pointer_update_ordinal);
        text.u64("target_onset_ordinal", active_target_onset_ordinal_);
        text.u64("kind", static_cast<std::uint64_t>(selection.event.kind));
        text.u64("paradigm", selection.event.paradigm);
        text.boolean("correct", selection.event.correct);
        text.u64("selected_id", selection.event.selected_id);
        text.u64("intended_id", selection.event.intended_id);
        text.u64("dwell_ns", selection.event.dwell_ns);
        text.u64("target_onset_ns", selection.target_onset_ns);
        text.u64("elapsed_since_target_onset_ns", selection.elapsed_since_target_onset_ns);
        // The raw selection is kept whatever happened -- data is not deleted
        // because it turned out to be unusable -- but the acquisition interval
        // beside it is marked for what it is. Recomputing the published metric
        // offline from records that say this is the whole point of keeping the
        // raw selections in the first place.
        text.boolean("acquisition_timing_valid", trace.acquisition_timing_valid);
        text.trial("trial", selection.event.trial);
        static_cast<void>(sink.value(static_cast<std::uint64_t>(selection.event.selected_id)));
        static_cast<void>(sink.commit());
    }

    if (trace.step.trial_decided)
    {
        const auto& completed = trace.step.trial;
        auto record = completed.record;
        if (trace.trial_invalidated)
        {
            // The machine decided this target was selected correctly, and it
            // was. What it cannot know is that the interval it was selected
            // over is not a measurement any more. Recording the machine's
            // verdict unchanged would be the silent success this layer exists
            // to prevent, so the outcome written is TrialOutcome::aborted and
            // the machine's own verdict is kept beside it rather than
            // destroyed.
            record.outcome = TrialOutcome::aborted;
        }
        auto& label = sink.trial(record);
        label.trial("trial", completed.record.trial);
        label.u64("paradigm", completed.record.paradigm);
        label.u64("acquisition_ns", completed.acquisition_ns);
        label.u64("selected_id", completed.selection.event.selected_id);
        label.u64("intended_id", completed.selection.event.intended_id);
        label.boolean("correct", completed.selection.event.correct);
        label.u64("target_onset_ns", completed.selection.target_onset_ns);
        label.u64("selection_sequence", completed.selection.event.sequence);
        label.u64("target_onset_ordinal", active_target_onset_ordinal_);
        label.boolean("acquisition_timing_valid", trace.acquisition_timing_valid);
        if (trace.trial_invalidated)
        {
            label.u64("task_outcome", static_cast<std::uint64_t>(completed.record.outcome));
            label.u64("task_reason", completed.record.reason);
        }
        static_cast<void>(sink.commit());
    }

    // The onset of whatever target is up now, written when it changes. A cell
    // that stayed up across a step has not had a second onset, and recording
    // one per step would make a replay see targets the run never presented.
    if (snapshot.active_target != kUnsetTargetId &&
        (!has_target_ || snapshot.active_target != last_target_ ||
         snapshot.target_onset_ns != last_onset_ns_))
    {
        auto& text =
            sink.named(ControlKind::targets, "webgrid.target_onset", snapshot.target_onset_ns);
        text.u64("target_id", snapshot.active_target);
        text.u64("target_onset_ordinal", target_onset_ordinal_);
        text.u64("rows", config().rows);
        text.u64("columns", config().columns);
        text.u64("completed", snapshot.completed);
        text.trial("trial", snapshot.trial);
        static_cast<void>(sink.value(static_cast<std::uint64_t>(snapshot.active_target)));
        static_cast<void>(sink.commit());
        has_target_ = true;
        last_target_ = snapshot.active_target;
        last_onset_ns_ = snapshot.target_onset_ns;
        active_target_onset_ordinal_ = target_onset_ordinal_;
        ++target_onset_ordinal_;
    }
}

void WebGridTraceWriter::write_summary(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept
{
    if (require_presentation_input_evidence_ && missing_presentation_inputs_ != 0)
    {
        const auto event = abnormal_.observe(
            AbnormalCondition::presentation_evidence_missing, time_ns, /*trial*/ nullptr,
            static_cast<std::uint32_t>(missing_presentation_inputs_), AbnormalPolicy::record,
            /*can_end_trial*/ false,
            /*input_refused*/ false);
        auto& missing = sink.abnormal(event);
        missing.u64("missing_inputs", missing_presentation_inputs_);
        static_cast<void>(sink.commit());
    }
    auto& text = sink.named(ControlKind::task_variables, "webgrid.metrics", time_ns);
    text.u64("metric_version", metrics_.metric_version);
    text.u64("correct_selections", metrics_.correct_selections);
    text.u64("incorrect_selections", metrics_.incorrect_selections);
    text.u64("elapsed_active_ns", metrics_.elapsed_active_ns);
    text.boolean("rates_defined", metrics_.rates_defined);
    text.f64("correct_targets_per_minute", metrics_.correct_targets_per_minute);
    text.f64("net_correct_targets_per_minute", metrics_.net_correct_targets_per_minute);
    text.u64("acquisition_count", metrics_.n_acquisitions);
    text.u64("total_acquisition_ns", metrics_.total_acquisition_ns);
    text.u64("minimum_acquisition_ns", metrics_.minimum_acquisition_ns);
    text.u64("maximum_acquisition_ns", metrics_.maximum_acquisition_ns);
    text.u64("mean_acquisition_ns", metrics_.mean_acquisition_ns);
    // The metric above is WebGridMachine's, unchanged, including any target an
    // abnormal condition invalidated. It is not silently recomputed here: the
    // machine owns that formula, and a summary that quietly disagreed with the
    // records it was derived from would be the harder of the two to trust. What
    // is added is the count a reader needs to know the metric is provisional,
    // and the per-selection records already say which ones to exclude.
    text.u64("invalidated_targets", invalidated_targets_);
    text.boolean("metric_includes_invalidated", invalidated_targets_ != 0);
    static_cast<void>(sink.commit());

    if (presentation_reports_ != 0 || missing_presentation_inputs_ != 0 ||
        require_presentation_input_evidence_)
    {
        auto& presentation =
            sink.named(ControlKind::experiment_states, "webgrid.presentation_summary", time_ns);
        presentation.u64("presentation_reports", presentation_reports_);
        presentation.u64("presentation_failures", presentation_failures_);
        presentation.u64("missing_presentation_inputs", missing_presentation_inputs_);
        presentation.boolean("presentation_input_evidence_required",
                             require_presentation_input_evidence_);
        static_cast<void>(sink.commit());
    }
}

} // namespace neurale::execution
