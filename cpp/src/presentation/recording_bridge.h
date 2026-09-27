// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "center_out_presenter.h"
#include "speech_presenter.h"
#include "webgrid_presenter.h"

#include "../execution/center_out_trace_writer.h"
#include "../execution/speech_trace_writer.h"
#include "../execution/webgrid_trace_writer.h"

namespace neurale::experiment_presentation
{

/// Concrete presentation-to-recording seams. They only copy compact values into
/// the prepared integration queues; they perform no encoding or filesystem work.
[[nodiscard]] bool report_center_out_presentation(execution::CenterOutTraceWriter& writer,
                                                  execution::CenterOutController& controller,
                                                  const SoftwarePresentationTimes& times,
                                                  std::uint64_t update_ordinal,
                                                  std::uint64_t source_ordinal,
                                                  const experiments::TrialIdentity& trial) noexcept;

[[nodiscard]] bool report_center_out_lifecycle(execution::CenterOutTraceWriter& writer,
                                               execution::PresentationLifecycleEvent event,
                                               experiments::ExperimentTimeNs time_ns,
                                               SurfaceStatus status,
                                               std::uint64_t update_ordinal = 0) noexcept;

/// Report a presentation-runtime failure while Center-Out is running. Unlike a
/// pre/post-run lifecycle fault, this is handed to the task owner so the active
/// trial observes the configured presentation_failed policy.
[[nodiscard]] bool report_center_out_runtime_failure(execution::CenterOutTraceWriter& writer,
                                                     execution::CenterOutController& controller,
                                                     experiments::ExperimentTimeNs time_ns,
                                                     SurfaceStatus status,
                                                     const experiments::TrialIdentity& trial,
                                                     std::uint64_t update_ordinal = 0) noexcept;

[[nodiscard]] bool report_webgrid_presentation(execution::WebGridTraceWriter& writer,
                                               execution::WebGridHeadlessController& controller,
                                               const SoftwarePresentationTimes& times,
                                               std::uint64_t update_ordinal,
                                               std::uint64_t source_ordinal,
                                               const experiments::TrialIdentity& trial) noexcept;

[[nodiscard]] bool report_webgrid_lifecycle(execution::WebGridTraceWriter& writer,
                                            execution::PresentationLifecycleEvent event,
                                            experiments::ExperimentTimeNs time_ns,
                                            SurfaceStatus status,
                                            std::uint64_t update_ordinal = 0) noexcept;

[[nodiscard]] bool report_webgrid_runtime_failure(execution::WebGridTraceWriter& writer,
                                                  execution::WebGridHeadlessController& controller,
                                                  experiments::ExperimentTimeNs time_ns,
                                                  SurfaceStatus status,
                                                  const experiments::TrialIdentity& trial,
                                                  std::uint64_t update_ordinal = 0) noexcept;

[[nodiscard]] execution::WebGridPresentationInputEvidence
webgrid_input_evidence(const WebGridPresentationInput& input) noexcept;

[[nodiscard]] streaming::StreamStatus
process_webgrid_input(execution::WebGridHeadlessController& controller,
                      const WebGridPresentationInput& input) noexcept;

[[nodiscard]] streaming::StreamStatus
process_webgrid_selection(execution::WebGridHeadlessController& controller,
                          const WebGridPresentationInput& input,
                          const experiments::SelectionEvent& selection) noexcept;

[[nodiscard]] bool report_speech_presentation(execution::SpeechTraceWriter& writer,
                                              const SpeechPresentationEvidence& evidence) noexcept;

[[nodiscard]] bool report_speech_lifecycle(execution::SpeechTraceWriter& writer,
                                           execution::PresentationLifecycleEvent event,
                                           experiments::ExperimentTimeNs time_ns,
                                           SurfaceStatus status,
                                           std::uint64_t update_ordinal = 0) noexcept;

/// Report a runtime presentation failure against the exact active Speech
/// trial. It remains lifecycle evidence rather than fabricating an outcome for
/// a PresentationRequest that the failure did not answer. The bounded handoff
/// is applied by the Speech scheduler/task owner; recording drain order never
/// decides trial attribution.
[[nodiscard]] bool report_speech_runtime_failure(execution::SpeechTraceWriter& writer,
                                                 experiments::ExperimentTimeNs time_ns,
                                                 SurfaceStatus status,
                                                 const experiments::TrialIdentity& trial,
                                                 std::uint64_t update_ordinal = 0) noexcept;

/// Record the frozen presentation configuration for a run as provenance. Each
/// validates and copies the concrete presenter's explanatory fields into an
/// owned integration record, which the trace writer writes at configuration
/// time. The Speech seam also carries the font identity (path, face index, and
/// the SHA-256 of the bytes FreeType consumed) so the already-frozen font
/// provenance reaches the recording rather than being dropped at the boundary.
[[nodiscard]] bool
record_center_out_presentation_config(execution::CenterOutTraceWriter& writer,
                                      const CenterOut2DPresenter& presenter) noexcept;

[[nodiscard]] bool record_webgrid_presentation_config(execution::WebGridTraceWriter& writer,
                                                      const WebGridPresenter& presenter) noexcept;

[[nodiscard]] bool record_speech_presentation_config(execution::SpeechTraceWriter& writer,
                                                     const SpeechCuePresenter& presenter);

} // namespace neurale::experiment_presentation
