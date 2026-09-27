// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "recording_bridge.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace neurale::experiment_presentation
{
namespace
{
[[nodiscard]] execution::PresentationConfigColour colour(const Color& c) noexcept
{
    return {c.red, c.green, c.blue, c.alpha};
}

[[nodiscard]] execution::PresentationSoftwareEvidence
software_evidence(const SoftwarePresentationTimes& times, std::uint64_t update_ordinal,
                  std::uint64_t source_ordinal, const experiments::TrialIdentity* trial) noexcept
{
    return {
        .event = times.status == SurfaceStatus::ok
                     ? execution::PresentationLifecycleEvent::presented
                     : execution::PresentationLifecycleEvent::faulted,
        .time_ns = times.status == SurfaceStatus::ok ? times.presented_ns : times.requested_ns,
        .requested_ns = times.requested_ns,
        .intended_ns = times.intended_ns,
        .submitted_renderer_ns = times.submitted_renderer_ns,
        .submitted_ns = times.submitted_ns,
        .presented_renderer_ns = times.presented_renderer_ns,
        .presented_ns = times.presented_ns,
        .update_ordinal = update_ordinal,
        .source_ordinal = source_ordinal,
        .trial = trial != nullptr ? *trial : experiments::TrialIdentity{},
        .implementation_status = static_cast<std::uint32_t>(times.status),
        .has_source_ordinal = true,
        .has_trial = trial != nullptr,
        // A request rejected before PresentationSurface::present() has no
        // measured submit/present timestamps. Keep those fields unavailable
        // instead of turning their zero-initialized storage into evidence.
        .has_software_times = times.status == SurfaceStatus::ok,
    };
}

[[nodiscard]] execution::PresentationSoftwareEvidence
lifecycle_evidence(execution::PresentationLifecycleEvent event,
                   experiments::ExperimentTimeNs time_ns, SurfaceStatus status,
                   std::uint64_t update_ordinal,
                   const experiments::TrialIdentity* trial = nullptr) noexcept
{
    return {.event = event,
            .time_ns = time_ns,
            .update_ordinal = update_ordinal,
            .trial = trial != nullptr ? *trial : experiments::TrialIdentity{},
            .implementation_status = static_cast<std::uint32_t>(status),
            .has_trial = trial != nullptr,
            .has_software_times = false};
}
} // namespace

bool report_center_out_presentation(execution::CenterOutTraceWriter& writer,
                                    execution::CenterOutController& controller,
                                    const SoftwarePresentationTimes& times,
                                    std::uint64_t update_ordinal, std::uint64_t source_ordinal,
                                    const experiments::TrialIdentity& trial) noexcept
{
    const auto evidence = software_evidence(times, update_ordinal, source_ordinal, &trial);
    const bool recorded = writer.report_presentation(evidence);
    if (times.status == SurfaceStatus::ok)
        return recorded;
    const execution::PresentationFailureEvidence failure{evidence.time_ns, trial,
                                                         evidence.implementation_status};
    return controller.enqueue_presentation_failure(failure) && recorded;
}

bool report_center_out_lifecycle(execution::CenterOutTraceWriter& writer,
                                 execution::PresentationLifecycleEvent event,
                                 experiments::ExperimentTimeNs time_ns, SurfaceStatus status,
                                 std::uint64_t update_ordinal) noexcept
{
    return writer.report_presentation(lifecycle_evidence(event, time_ns, status, update_ordinal));
}

bool report_center_out_runtime_failure(execution::CenterOutTraceWriter& writer,
                                       execution::CenterOutController& controller,
                                       experiments::ExperimentTimeNs time_ns, SurfaceStatus status,
                                       const experiments::TrialIdentity& trial,
                                       std::uint64_t update_ordinal) noexcept
{
    if (status == SurfaceStatus::ok)
        return false;
    const auto evidence = lifecycle_evidence(execution::PresentationLifecycleEvent::faulted,
                                             time_ns, status, update_ordinal, &trial);
    const bool recorded = writer.report_presentation(evidence);
    const execution::PresentationFailureEvidence failure{time_ns, trial,
                                                         static_cast<std::uint32_t>(status)};
    return controller.enqueue_presentation_failure(failure) && recorded;
}

bool report_webgrid_presentation(execution::WebGridTraceWriter& writer,
                                 execution::WebGridHeadlessController& controller,
                                 const SoftwarePresentationTimes& times,
                                 std::uint64_t update_ordinal, std::uint64_t source_ordinal,
                                 const experiments::TrialIdentity& trial) noexcept
{
    const auto evidence = software_evidence(times, update_ordinal, source_ordinal, &trial);
    const bool recorded = writer.report_presentation(evidence);
    if (times.status == SurfaceStatus::ok)
        return recorded;
    const execution::PresentationFailureEvidence failure{evidence.time_ns, trial,
                                                         evidence.implementation_status};
    return controller.enqueue_presentation_failure(failure) && recorded;
}

bool report_webgrid_lifecycle(execution::WebGridTraceWriter& writer,
                              execution::PresentationLifecycleEvent event,
                              experiments::ExperimentTimeNs time_ns, SurfaceStatus status,
                              std::uint64_t update_ordinal) noexcept
{
    return writer.report_presentation(lifecycle_evidence(event, time_ns, status, update_ordinal));
}

bool report_webgrid_runtime_failure(execution::WebGridTraceWriter& writer,
                                    execution::WebGridHeadlessController& controller,
                                    experiments::ExperimentTimeNs time_ns, SurfaceStatus status,
                                    const experiments::TrialIdentity& trial,
                                    std::uint64_t update_ordinal) noexcept
{
    if (status == SurfaceStatus::ok)
        return false;
    const auto evidence = lifecycle_evidence(execution::PresentationLifecycleEvent::faulted,
                                             time_ns, status, update_ordinal, &trial);
    const bool recorded = writer.report_presentation(evidence);
    const execution::PresentationFailureEvidence failure{time_ns, trial,
                                                         static_cast<std::uint32_t>(status)};
    return controller.enqueue_presentation_failure(failure) && recorded;
}

execution::WebGridPresentationInputEvidence
webgrid_input_evidence(const WebGridPresentationInput& input) noexcept
{
    return {.input_ordinal = input.input_ordinal,
            .renderer_time_ns = input.renderer_time_ns,
            .experiment_time_ns = input.experiment_time_ns,
            .kind = static_cast<std::uint8_t>(input.kind),
            .button = input.button,
            .modifiers = input.modifiers,
            .inside_presentation = input.inside_presentation,
            .available = true};
}

streaming::StreamStatus process_webgrid_input(execution::WebGridHeadlessController& controller,
                                              const WebGridPresentationInput& input) noexcept
{
    if (input.kind != WebGridPresentationInputKind::pointer_update)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    return controller.process(input.experiment_time_ns, input.pointer,
                              webgrid_input_evidence(input));
}

streaming::StreamStatus
process_webgrid_selection(execution::WebGridHeadlessController& controller,
                          const WebGridPresentationInput& input,
                          const experiments::SelectionEvent& selection) noexcept
{
    if (input.kind != WebGridPresentationInputKind::selection_request ||
        selection.time_ns != input.experiment_time_ns)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    return controller.process(input.experiment_time_ns, input.pointer, selection,
                              webgrid_input_evidence(input));
}

bool report_speech_presentation(execution::SpeechTraceWriter& writer,
                                const SpeechPresentationEvidence& evidence) noexcept
{
    if (experiments::validate(evidence.request) != experiments::ContractStatus::ok ||
        experiments::validate(evidence.outcome) != experiments::ContractStatus::ok ||
        evidence.outcome.request_sequence != evidence.request.sequence ||
        evidence.outcome.requested_ns != evidence.request.requested_ns ||
        evidence.outcome.stimulus_id != evidence.request.stimulus_id ||
        !experiments::same_trial(evidence.outcome.trial, evidence.request.trial) ||
        evidence.software_times.requested_ns != evidence.request.requested_ns ||
        evidence.software_times.intended_ns != evidence.request.onset_ns)
    {
        return false;
    }
    // A presented outcome and its software evidence must agree on the one
    // swap-return experiment time. The presenter sets outcome.presented_ns from
    // the same SoftwarePresentationTimes it hands to the bridge; a mismatch here
    // would mean one evidence value carries two contradictory swap-return times.
    if (evidence.outcome.status == experiments::PresentationStatus::presented &&
        evidence.outcome.presented_ns != evidence.software_times.presented_ns)
    {
        return false;
    }
    execution::SpeechPresentationEvidence report{};
    report.outcome = evidence.outcome;
    // Speech joins an outcome to its request by request_sequence, which
    // write_outcome records directly; source_ordinal is the Center-Out/WebGrid
    // join key and is left at zero here.
    report.software = software_evidence(evidence.software_times, evidence.request.sequence, 0,
                                        &evidence.request.trial);
    report.software.has_source_ordinal = false;
    report.software.implementation_status = static_cast<std::uint32_t>(evidence.render_status);
    return writer.report_presentation(report);
}

bool report_speech_lifecycle(execution::SpeechTraceWriter& writer,
                             execution::PresentationLifecycleEvent event,
                             experiments::ExperimentTimeNs time_ns, SurfaceStatus status,
                             std::uint64_t update_ordinal) noexcept
{
    return writer.report_lifecycle(lifecycle_evidence(event, time_ns, status, update_ordinal));
}

bool report_speech_runtime_failure(execution::SpeechTraceWriter& writer,
                                   experiments::ExperimentTimeNs time_ns, SurfaceStatus status,
                                   const experiments::TrialIdentity& trial,
                                   std::uint64_t update_ordinal) noexcept
{
    if (status == SurfaceStatus::ok)
        return false;
    auto evidence = lifecycle_evidence(execution::PresentationLifecycleEvent::faulted, time_ns,
                                       status, update_ordinal, &trial);
    const execution::PresentationFailureEvidence failure{time_ns, trial,
                                                         static_cast<std::uint32_t>(status)};
    evidence.policy_delegated = true;
    const bool delegated = writer.enqueue_presentation_failure(failure);
    const bool recorded = writer.report_lifecycle(evidence);
    return delegated && recorded;
}

bool record_center_out_presentation_config(execution::CenterOutTraceWriter& writer,
                                           const CenterOut2DPresenter& presenter) noexcept
{
    if (!presenter.prepared())
        return false;
    const auto& config = presenter.configuration();
    const auto& task = writer.task_configuration();
    if (experiments::center_out::configuration_fingerprint(task) !=
            experiments::center_out::configuration_fingerprint(presenter.task_configuration()) ||
        validate_center_out_presentation(task, config) != SurfaceStatus::ok)
    {
        return false;
    }
    execution::CenterOutPresentationConfigRecord record{};
    record.geometry_unit = static_cast<std::uint64_t>(config.geometry_unit);
    record.logical_left = config.logical_space.left;
    record.logical_bottom = config.logical_space.bottom;
    record.logical_width = config.logical_space.width;
    record.logical_height = config.logical_space.height;
    record.target_radius = config.style.target_radius;
    record.cursor_radius = config.style.cursor_radius;
    // The scored geometry, recorded beside the drawn geometry. Both come from
    // the one task configuration validated above, so the comparison is between
    // this run's own numbers rather than an assumed pairing. The presenter draws
    // what it was told to draw either way -- these fields make a disagreement legible
    // offline instead of leaving it to be rederived from two records.
    record.acceptance_half_extent_x = task.acceptance.half_extent_x;
    record.acceptance_half_extent_y = task.acceptance.half_extent_y;
    record.task_cursor_extent = task.cursor.extent;
    const auto acceptance_reach =
        std::hypot(task.acceptance.half_extent_x, task.acceptance.half_extent_y);
    record.target_circle_contains_acceptance = config.style.target_radius >= acceptance_reach;
    record.acceptance_contains_target_circle =
        config.style.target_radius <=
        std::min(task.acceptance.half_extent_x, task.acceptance.half_extent_y);
    record.cursor_radius_matches_task_extent = config.style.cursor_radius == task.cursor.extent;
    record.circle_segments = config.style.circle_segments;
    record.background = colour(config.style.background);
    record.center_target = colour(config.style.center_target);
    record.outward_target = colour(config.style.outward_target);
    record.active_move = colour(config.style.active_move);
    record.active_hold = colour(config.style.active_hold);
    record.success = colour(config.style.success);
    record.failure = colour(config.style.failure);
    record.cursor = colour(config.style.cursor);
    record.aspect_policy = static_cast<std::uint64_t>(config.aspect_policy);
    record.window_width = config.window_size.width;
    record.window_height = config.window_size.height;
    record.monitor_idx = config.monitor_idx;
    record.swap_interval = config.swap_interval;
    record.fullscreen = config.fullscreen;
    record.resizable = config.resizable;
    record.visible = config.visible;
    return writer.report_presentation_config(record);
}

bool record_webgrid_presentation_config(execution::WebGridTraceWriter& writer,
                                        const WebGridPresenter& presenter) noexcept
{
    if (!presenter.prepared())
        return false;
    const auto& config = presenter.configuration();
    const auto& task = writer.task_configuration();
    if (experiments::webgrid::configuration_fingerprint(task) !=
            experiments::webgrid::configuration_fingerprint(presenter.task_configuration()) ||
        validate_webgrid_presentation(task, config) != SurfaceStatus::ok)
    {
        return false;
    }
    execution::WebGridPresentationConfigRecord record{};
    record.logical_min_x = task.bounds.min_x;
    record.logical_max_x = task.bounds.max_x;
    record.logical_min_y = task.bounds.min_y;
    record.logical_max_y = task.bounds.max_y;
    record.pointer_radius = config.style.pointer_radius;
    record.circle_segments = config.style.circle_segments;
    record.background = colour(config.style.background);
    record.cell = colour(config.style.cell);
    record.active_target = colour(config.style.active_target);
    record.correct_feedback = colour(config.style.correct_feedback);
    record.incorrect_feedback = colour(config.style.incorrect_feedback);
    record.grid_line = colour(config.style.grid_line);
    record.pointer = colour(config.style.pointer);
    record.selection_button = config.selection_button;
    record.aspect_policy = static_cast<std::uint64_t>(config.aspect_policy);
    record.window_width = config.window_size.width;
    record.window_height = config.window_size.height;
    record.monitor_idx = config.monitor_idx;
    record.swap_interval = config.swap_interval;
    record.fullscreen = config.fullscreen;
    record.resizable = config.resizable;
    record.visible = config.visible;
    return writer.report_presentation_config(record);
}

bool record_speech_presentation_config(execution::SpeechTraceWriter& writer,
                                       const SpeechCuePresenter& presenter)
{
    if (!presenter.prepared())
        return false;
    const auto& config = presenter.configuration();
    const auto& font = presenter.font_identity();
    if (validate(config) != SurfaceStatus::ok ||
        presenter.catalog_fingerprint() != writer.catalog_fingerprint() ||
        font.path != config.text.font_path || font.face_idx != config.text.face_idx)
    {
        return false;
    }
    execution::SpeechPresentationConfigRecord record{};
    record.font_path = font.path;
    record.font_sha256 = font.sha256;
    record.face_idx = font.face_idx;
    record.pixel_height = config.text.pixel_height;
    record.logical_left = config.logical_space.left;
    record.logical_bottom = config.logical_space.bottom;
    record.logical_width = config.logical_space.width;
    record.logical_height = config.logical_space.height;
    record.fixation_center_x = config.style.fixation_center.x;
    record.fixation_center_y = config.style.fixation_center.y;
    record.fixation_half_extent = config.style.fixation_half_extent;
    record.text_baseline_x = config.style.text_baseline.x;
    record.text_baseline_y = config.style.text_baseline.y;
    record.background = colour(config.style.background);
    record.fixation = colour(config.style.fixation);
    record.text = colour(config.style.text);
    record.aspect_policy = static_cast<std::uint64_t>(config.aspect_policy);
    record.window_width = config.window_size.width;
    record.window_height = config.window_size.height;
    record.monitor_idx = config.monitor_idx;
    record.swap_interval = config.swap_interval;
    record.fullscreen = config.fullscreen;
    record.resizable = config.resizable;
    record.visible = config.visible;
    return writer.report_presentation_config(std::move(record));
}

} // namespace neurale::experiment_presentation
