/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The WebGrid side of the recording bridge.

#include <cstdint>
#include <string_view>

#include <neurale/experiments/webgrid.h>
#include <neurale/experiments/webgrid_replay.h>

#include "abnormal_reporter.h"
#include "presentation_evidence.h"
#include "session.h"
#include "webgrid_controller.h"

namespace neurale::execution
{

using namespace neurale::experiments;

/// Stable name of this paradigm in the session metadata.
inline constexpr std::string_view kWebGridExperimentType{"webgrid"};
inline constexpr std::uint32_t kWebGridExperimentVersion = webgrid::kWebGridRecordVersion;

/// Turns WebGridHeadlessController trace records into control records.
///
/// Pointer samples are recorded here rather than referenced, because a headless
/// WebGrid has no cursor stream to point at: the pointer observations *are* the
/// caller's, they arrive one per step, and a replay that cannot see them cannot
/// reproduce a selection. A caller whose pointer really does come from a
/// recorded stream names that stream through
/// ::WebGridTraceWriter::reference_cursor_stream. Each accepted update then
/// writes only its experiment-local ordinal and stream name, not a second copy
/// of the pointer values. That is deliberately only stream-level provenance:
/// until orchestration supplies a recorded external sample identity, offline
/// traceability reports the external pointer parent as partial.
class WebGridTraceWriter final : public ExperimentTraceSource
{
  public:
    /// Bind to the controller that will run, and to the paradigm identity the
    /// run is recorded under.
    ///
    /// The configuration is deliberately **not** a parameter. It is read from
    /// the controller, which is the only object that knows which configuration
    /// the run is actually executing; a writer given its own copy can record a
    /// valid configuration that the controller never saw, and no check inside
    /// either object would catch it.
    WebGridTraceWriter(WebGridHeadlessController& controller, ParadigmId paradigm)
        : controller_(controller), paradigm_(paradigm)
    {
        presentations_.prepare(64);
    }

    [[nodiscard]] bool report_presentation(const PresentationSoftwareEvidence& evidence) noexcept;
    /// Freeze one presentation configuration before the session writes its
    /// configuration records. Duplicate and post-start reports are rejected.
    [[nodiscard]] bool
    report_presentation_config(const WebGridPresentationConfigRecord& config) noexcept;
    /// The task configuration frozen by the controller this writer records.
    [[nodiscard]] const webgrid::WebGridConfig& task_configuration() const noexcept
    {
        return controller_.configuration();
    }
    void require_presentation_input_evidence(bool required) noexcept
    {
        require_presentation_input_evidence_ = required;
    }

    /// Name the recorded stream the pointer samples already live in. When set,
    /// the configuration record carries the name and each per-update record is
    /// a stream-level reference rather than a duplicate pointer payload. This
    /// API does not invent an external frame/sample identity.
    void reference_cursor_stream(std::string_view name) noexcept
    {
        cursor_stream_ = name;
    }

    [[nodiscard]] ContractStatus validate() const noexcept override;
    [[nodiscard]] streaming::StreamStatus
    start_execution(ExperimentTimeNs time_ns) noexcept override;
    void stop_execution(bool aborting) noexcept override;
    [[nodiscard]] std::uint64_t dropped_trace_count() const noexcept override;
    [[nodiscard]] SessionMetadata metadata() const noexcept override;
    void write_configuration(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept override;
    [[nodiscard]] std::size_t drain(ExperimentTraceSink& sink,
                                    std::size_t budget) noexcept override;
    void write_summary(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept override;
    [[nodiscard]] AbnormalSummary abnormal_summary() const noexcept override;
    void halt(ExperimentTimeNs time_ns, AbnormalCondition condition) noexcept override;

  private:
    void write_trace(ExperimentTraceSink& sink, const WebGridHeadlessTrace& trace) noexcept;
    void write_presentation(ExperimentTraceSink& sink,
                            const PresentationSoftwareEvidence& evidence) noexcept;
    void write_abnormal(ExperimentTraceSink& sink, const AbnormalEvent& event) noexcept;
    void write_presentation_config(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept;
    /// The configuration the controller froze at prepare(). The only one.
    [[nodiscard]] const webgrid::WebGridConfig& config() const noexcept
    {
        return controller_.configuration();
    }

    WebGridHeadlessController& controller_;
    ParadigmId paradigm_{kUnsetParadigmId};
    std::string_view cursor_stream_{};
    TargetId last_target_{kUnsetTargetId};
    ExperimentTimeNs last_onset_ns_{};
    SequenceOrdinal target_onset_ordinal_{};
    SequenceOrdinal active_target_onset_ordinal_{};
    bool has_target_{};
    /// Targets an abnormal condition made inadmissible. Reported in the
    /// summary beside the machine's own metric, because that metric counts them
    /// and this layer will not quietly recompute one the paradigm owns.
    std::uint64_t invalidated_targets_{};
    webgrid::WebGridMetrics metrics_{};
    BoundedTraceQueue<PresentationSoftwareEvidence> presentations_{};
    std::uint64_t presentation_reports_{};
    std::uint64_t presentation_failures_{};
    std::uint64_t missing_presentation_inputs_{};
    bool require_presentation_input_evidence_{};
    /// This writer's own reporter, separate from the controller's for the same
    /// thread-safety reason as the Speech and Center-Out writers.
    AbnormalReporter abnormal_{};
    WebGridPresentationConfigRecord presentation_config_{};
    bool has_presentation_config_{};
    bool configuration_written_{};
};

} // namespace neurale::execution
