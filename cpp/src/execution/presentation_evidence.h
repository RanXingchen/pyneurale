/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/presentation.h>

#include "bounded_trace_queue.h"

namespace neurale::execution
{

using namespace neurale::experiments;

/// Drain every queued presentation failure, however many are waiting.
///
/// For shutdown and for callers that are not on a deadline. Never for a frame
/// callback: see ::kPresentationFailureDrainBudget.
inline constexpr std::size_t kUnboundedPresentationFailureDrain =
    (std::numeric_limits<std::size_t>::max)();

/// Presentation failures a task owner applies within one frame callback.
///
/// The handoff queue is bounded, so draining it fully was always finite work --
/// but finite is not the same as budgeted. The consumer runs on the acquisition
/// thread, where the cost that matters is per frame, and a full queue would put
/// its whole capacity worth of abnormal decisions and trace pushes into a
/// single callback. Eight per frame keeps the per-frame cost flat and still
/// clears any realistic backlog within a few frames, because a presenter
/// produces at most one failure per rendered frame and frames are far rarer
/// than acquisition frames. Deferred failures stay queued; none are dropped.
inline constexpr std::size_t kPresentationFailureDrainBudget = 8;

/// Presentation-only lifecycle evidence accepted by the recording bridge.
///
/// The renderer status is intentionally an opaque implementation code here:
/// this layer owns no renderer status vocabulary. The semantic success/failure
/// of a Speech request remains PresentationOutcome::status.
enum class PresentationLifecycleEvent : std::uint8_t
{
    opened = 0,
    prepared,
    presented,
    cancelled,
    closed,
    faulted,
};

struct PresentationSoftwareEvidence
{
    PresentationLifecycleEvent event{PresentationLifecycleEvent::opened};
    ExperimentTimeNs time_ns{};
    ExperimentTimeNs requested_ns{};
    ExperimentTimeNs intended_ns{};
    std::uint64_t submitted_renderer_ns{};
    ExperimentTimeNs submitted_ns{};
    std::uint64_t presented_renderer_ns{};
    ExperimentTimeNs presented_ns{};
    /// Renderer-internal frame identity (the presenter's own update ordinal).
    SequenceOrdinal update_ordinal{};
    /// The semantic-record identity this presentation frame displayed, supplied
    /// by the orchestrator so an offline reader can join a presentation report to
    /// the observation/pointer-update record it depicted. Distinct from
    /// update_ordinal, which is presenter-local and not guaranteed to align with
    /// any semantic trace ordinal. The caller must supply a real semantic
    /// identity here, not a default copied from the presenter's update counter.
    SequenceOrdinal source_ordinal{};
    TrialIdentity trial{};
    std::uint32_t implementation_status{};
    /// Whether source_ordinal names a semantic update. Ordinal zero is a
    /// valid first update, so absence cannot be encoded with a sentinel.
    bool has_source_ordinal{};
    /// Whether this evidence names an exact trial. Presented frames and active
    /// Speech runtime failures carry one; pre/post-run lifecycle evidence does
    /// not.
    bool has_trial{};
    /// The corresponding runtime failure belongs to the concrete task-owner
    /// handoff. The trace writer still persists this implementation evidence,
    /// but must not create a second presentation_failed event from it. Queue
    /// refusal is accounted as loss by that task-owner handoff.
    bool policy_delegated{};
    bool has_software_times{};
};

/// A renderer failure handed from the presentation producer to the concrete
/// task owner. Frame and runtime failures both carry the exact trial identity
/// known when the failure occurred. The owner rejects a stale/mismatched trial
/// rather than transferring the failure to whichever trial is active later.
struct PresentationFailureEvidence
{
    ExperimentTimeNs time_ns{};
    TrialIdentity trial{};
    std::uint32_t implementation_status{};
};

/// Identity and software timing of one presentation pointer or click input.
///
/// `kind` is the concrete WebGridPresentationInputKind numeric value. This
/// layer does not interpret it; it persists it so offline provenance can
/// distinguish a pointer callback from the button press that caused a
/// SelectionEvent.
struct WebGridPresentationInputEvidence
{
    std::uint64_t input_ordinal{};
    std::uint64_t renderer_time_ns{};
    ExperimentTimeNs experiment_time_ns{};
    std::uint8_t kind{};
    int button{};
    int modifiers{};
    bool inside_presentation{};
    bool available{};
};

/// One Speech outcome plus the full software timing observed by the presenter.
struct SpeechPresentationEvidence
{
    PresentationOutcome outcome{};
    PresentationSoftwareEvidence software{};
};

/// A presentation colour as plain components, kept here so the integration-layer
/// config records below do not depend on the presentation-layer Colour type.
/// The recording bridge converts the presenter's Colour into these values.
struct PresentationConfigColour
{
    double red{};
    double green{};
    double blue{};
    double alpha{};
};

/// Frozen presentation configuration for one Center-Out run, as the recording
/// keeps it. Plain, owned fields let an offline reader interpret the exact
/// visual geometry and style used by the run. Pure implementation sizing
/// (input_capacity, vertex budget) is deliberately not scientific provenance.
struct CenterOutPresentationConfigRecord
{
    std::uint64_t geometry_unit{};
    double logical_left{};
    double logical_bottom{};
    double logical_width{};
    double logical_height{};
    double target_radius{};
    double cursor_radius{};
    /// The hit geometry this run actually scored with, copied in beside the
    /// drawn geometry above.
    ///
    /// The drawn target is a circle of `target_radius`; the acceptance region
    /// is an axis-aligned rectangle of these half-extents, and the task header
    /// is explicit that the target radius "was a rendered sphere size" the hit
    /// test never reads. That independence is deliberate and the presenter must
    /// not reconcile it -- but a reader deciding whether a session showed
    /// subjects what it scored should not have to join two records and
    /// rederive the comparison. Both numbers, and the derived relation below,
    /// live here.
    double acceptance_half_extent_x{};
    double acceptance_half_extent_y{};
    /// The cursor half-extent the hit test reads, beside the drawn
    /// `cursor_radius`.
    double task_cursor_extent{};
    /// Whether the drawn target disc covers the whole acceptance rectangle
    /// (`target_radius >= hypot(half_extent_x, half_extent_y)`): every scored
    /// position was inside something the subject saw.
    bool target_circle_contains_acceptance{};
    /// Whether the acceptance rectangle covers the whole drawn target disc
    /// (`target_radius <= min(half_extent_x, half_extent_y)`): everything the
    /// subject saw as target was scored as target.
    bool acceptance_contains_target_circle{};
    /// Whether the drawn cursor radius equals the scored cursor half-extent.
    bool cursor_radius_matches_task_extent{};
    std::uint64_t circle_segments{};
    PresentationConfigColour background{};
    PresentationConfigColour center_target{};
    PresentationConfigColour outward_target{};
    PresentationConfigColour active_move{};
    PresentationConfigColour active_hold{};
    PresentationConfigColour success{};
    PresentationConfigColour failure{};
    PresentationConfigColour cursor{};
    std::uint64_t aspect_policy{};
    int window_width{};
    int window_height{};
    int monitor_idx{};
    int swap_interval{};
    bool fullscreen{};
    bool resizable{};
    bool visible{};
};

/// Frozen presentation configuration for one WebGrid run.
struct WebGridPresentationConfigRecord
{
    double logical_min_x{};
    double logical_max_x{};
    double logical_min_y{};
    double logical_max_y{};
    double pointer_radius{};
    std::uint64_t circle_segments{};
    PresentationConfigColour background{};
    PresentationConfigColour cell{};
    PresentationConfigColour active_target{};
    PresentationConfigColour correct_feedback{};
    PresentationConfigColour incorrect_feedback{};
    PresentationConfigColour grid_line{};
    PresentationConfigColour pointer{};
    int selection_button{};
    std::uint64_t aspect_policy{};
    int window_width{};
    int window_height{};
    int monitor_idx{};
    int swap_interval{};
    bool fullscreen{};
    bool resizable{};
    bool visible{};
};

/// Frozen presentation configuration for one Speech run, including the exact
/// font identity: the SHA-256 of the bytes FreeType consumed, the face index,
/// and the configured path. This is the record that closes the provenance gap --
/// the font identity the presenter computed is persisted here rather than dropped
/// at the recording boundary.
struct SpeechPresentationConfigRecord
{
    std::string font_path{};
    std::array<std::uint8_t, 32> font_sha256{};
    long face_idx{};
    unsigned int pixel_height{};
    double logical_left{};
    double logical_bottom{};
    double logical_width{};
    double logical_height{};
    double fixation_center_x{};
    double fixation_center_y{};
    double fixation_half_extent{};
    double text_baseline_x{};
    double text_baseline_y{};
    PresentationConfigColour background{};
    PresentationConfigColour fixation{};
    PresentationConfigColour text{};
    std::uint64_t aspect_policy{};
    int window_width{};
    int window_height{};
    int monitor_idx{};
    int swap_interval{};
    bool fullscreen{};
    bool resizable{};
    bool visible{};
};

/// Apply up to @p budget queued presentation failures, stopping at the first
/// one the owner refuses.
///
/// The loop is the same in all three paradigm controllers because what varies
/// is only what "apply" means: each owner decides whether a failure belongs to
/// the trial it is running. The budget, the stop-on-refusal, and the one place
/// a refused failure is charged as loss are properties of the handoff itself,
/// and three copies of them were three chances for one controller to keep
/// draining past a refusal, or to lose a failure without counting it.
///
/// @param queue     The producer-to-owner handoff.
/// @param budget    Most failures to apply; ::kUnboundedPresentationFailureDrain
///                  to drain it out.
/// @param apply     Called with each failure; returns StreamStatus::ok to
///                  continue. StreamStatus::invalid_frame means the failure did
///                  not belong to the running trial, and is charged as a drop.
/// @return StreamStatus::ok when the budget or the queue ran out, otherwise the
///         status that stopped the drain.
template <typename ApplyFailure>
[[nodiscard]] streaming::StreamStatus
drain_presentation_failures(BoundedTraceQueue<PresentationFailureEvidence>& queue,
                            std::size_t budget, ApplyFailure&& apply) noexcept
{
    PresentationFailureEvidence evidence{};
    std::size_t applied = 0;
    while (applied < budget && queue.try_pop(evidence) == streaming::StreamStatus::ok)
    {
        ++applied;
        const auto status = apply(evidence);
        if (status != streaming::StreamStatus::ok)
        {
            if (status == streaming::StreamStatus::invalid_frame)
                queue.note_dropped(1);
            return status;
        }
    }
    return streaming::StreamStatus::ok;
}

} // namespace neurale::execution
