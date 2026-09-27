/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/identity.h>

/**
 * @file
 * @brief Semantic presentation contract.
 *
 * This header describes *what* is to be presented. The presentation renderer
 * decides how it is drawn: fonts, layout, colours, monitors, surfaces, and
 * frame timing. Nothing here is a pixel, a coordinate in screen space, or a
 * window.
 *
 * Two times are kept apart throughout, because conflating them is how a task
 * ends up claiming display timing it never measured:
 *
 * - the semantic request time, which the paradigm supplies, and
 * - the presenter-reported software presentation time, which only a presenter
 *   can report and which is a documented software observation point, not
 *   physical display onset.
 *
 * The paradigm layer produces the first and never the second. A headless
 * paradigm test validates semantic timing only and carries no evidence about
 * a monitor, a vsync, or a photodiode.
 */
namespace neurale::experiments
{

/// Semantic cue kind.
///
/// This is the shared v1 set. It is append-only: existing values are never
/// renumbered, and adding a kind is a change to this contract rather than
/// something a paradigm does on its own.
enum class CueKind : std::uint8_t
{
    /// Nothing is presented.
    none = 0,
    /// A blank field.
    black,
    /// A fixation cross.
    fixation_cross,
    /// Textual stimulus content, identified by ::StimulusId.
    text_content,
    /// SSVEP target catalog; the paradigm phase determines cue, flicker, or feedback.
    ssvep_targets,
};

/// Whether @p cue is one of the declared v1 kinds.
[[nodiscard]] constexpr bool cue_kind_declared(CueKind cue) noexcept
{
    return cue == CueKind::none || cue == CueKind::black || cue == CueKind::fixation_cross ||
           cue == CueKind::text_content || cue == CueKind::ssvep_targets;
}

/// A paradigm's semantic request to present one cue.
///
/// It identifies phase, trial, and stimulus, which is the whole of what a
/// presenter needs to resolve content against a prepared catalog. It carries no
/// text: the hot path moves a ::StimulusId, and the content it stands for is
/// resolved outside the hot path.
struct PresentationRequest
{
    /// Experiment time the paradigm decided to present. Not the time anything
    /// appeared, and never used as one.
    ExperimentTimeNs requested_ns{};
    /// Experiment time from which the cue is intended to be shown.
    ExperimentTimeNs onset_ns{};
    /// Instant after which presenting this request is pointless, or
    /// ::kNoExpiryNs. A presenter that cannot meet it must skip rather than
    /// present late.
    ExperimentTimeNs valid_until_ns{kNoExpiryNs};
    /// Intended duration. Zero means "until a later request supersedes it".
    DurationNs duration_ns{};
    /// Emission ordinal within the session.
    SequenceOrdinal sequence{};
    /// Trial the request belongs to.
    TrialIdentity trial{};
    /// Paradigm whose phase enumeration `phase` belongs to.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Paradigm-owned phase enumerator.
    PhaseId phase{};
    /// What to present.
    CueKind cue{CueKind::none};
    /// Stimulus to resolve, or ::kUnsetStimulusId when the cue needs none.
    StimulusId stimulus_id{kUnsetStimulusId};
};

/// What the experiment believes is currently being presented.
///
/// A belief, not an observation: it advances when the paradigm issues a
/// request, not when a presenter reports one.
struct PresentationState
{
    /// Experiment time this state began, in the paradigm's own terms.
    ExperimentTimeNs since_ns{};
    /// Trial the state belongs to.
    TrialIdentity trial{};
    /// Paradigm whose phase enumeration `phase` belongs to.
    ParadigmId paradigm{kUnsetParadigmId};
    /// Paradigm-owned phase enumerator.
    PhaseId phase{};
    /// What is believed to be presented.
    CueKind cue{CueKind::none};
    /// Stimulus being presented, or ::kUnsetStimulusId.
    StimulusId stimulus_id{kUnsetStimulusId};
};

/// What a presenter did with one request.
enum class PresentationStatus : std::uint8_t
{
    /// No presenter reported anything. This is what a session with no
    /// attached presentation renderer carries.
    not_reported = 0,
    /// The presenter reports that presentation reached its documented software
    /// observation point. This is not physical display onset.
    presented,
    /// The presenter deliberately did not present it.
    skipped,
    /// The request passed `valid_until_ns` before it could be presented.
    expired,
};

/// Whether @p status is one of the declared PresentationStatus values.
[[nodiscard]] constexpr bool presentation_status_declared(PresentationStatus status) noexcept
{
    return static_cast<std::uint8_t>(status) <=
           static_cast<std::uint8_t>(PresentationStatus::expired);
}

/// A presenter's report about one request.
///
/// The paradigm layer never produces one: it defines the shape so that a
/// presentation renderer has somewhere to report a software presentation
/// instant without inventing a parallel vocabulary, and so that the
/// distinction between requested time and a presenter-reported software
/// observation point is visible in the contract rather than assumed.
struct PresentationOutcome
{
    /// `requested_ns` of the request this reports on.
    ///
    /// Corroboration, not identity. Two requests may carry the same instant --
    /// the time contract accepts equal instants -- so a time never identifies a
    /// request. `request_sequence` does.
    ExperimentTimeNs requested_ns{};
    /// Presenter-reported software presentation instant in experiment time. This
    /// is a documented software observation point (for example a swap completion
    /// the implementation names), not physical display onset. Meaningful only
    /// when `status` is PresentationStatus::presented.
    ExperimentTimeNs presented_ns{};
    /// Emission ordinal of *this report* within the session.
    SequenceOrdinal sequence{};
    /// `sequence` of the PresentationRequest this reports on.
    ///
    /// Separate from `sequence` on purpose: one field cannot mean both "which
    /// record is this" and "which record is this about", and a traceability
    /// pass that has to guess which one it meant is not traceability.
    SequenceOrdinal request_sequence{};
    /// Trial the request belonged to.
    TrialIdentity trial{};
    /// What the presenter did.
    PresentationStatus status{PresentationStatus::not_reported};
    /// Stimulus that was presented, or ::kUnsetStimulusId.
    StimulusId stimulus_id{kUnsetStimulusId};
};

/// Validate a presentation request.
[[nodiscard]] ContractStatus validate(const PresentationRequest& request) noexcept;

/// Validate a presentation state.
[[nodiscard]] ContractStatus validate(const PresentationState& state) noexcept;

/// Validate a presenter report.
[[nodiscard]] ContractStatus validate(const PresentationOutcome& outcome) noexcept;

} // namespace neurale::experiments
