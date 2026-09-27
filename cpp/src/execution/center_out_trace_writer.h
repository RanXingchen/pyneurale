/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The Center-Out side of the recording bridge.

#include <array>
#include <cstdint>
#include <string_view>

#include <neurale/experiments/center_out.h>
#include <neurale/experiments/center_out_replay.h>

#include "abnormal_reporter.h"
#include "center_out_controller.h"
#include "presentation_evidence.h"
#include "session.h"

namespace neurale::execution
{

using namespace neurale::experiments;

inline constexpr std::string_view kCenterOutExperimentType{"center_out_2d"};
inline constexpr std::uint32_t kCenterOutExperimentVersion = center_out::kCenterOutRecordVersion;
inline constexpr std::size_t kDecoderPlanFingerprintBytes = 64;

/// One immutable decoder pipeline made eligible for trial-boundary activation.
struct DecoderPublicationEvidence
{
    std::uint64_t version{};
    ExperimentTimeNs time_ns{};
    std::uint64_t training_blocks{};
    std::uint64_t training_trials{};
    TrialOrdinal completed_trials{};
    std::array<char, kDecoderPlanFingerprintBytes> plan_fingerprint{};
};

/// Turns CenterOutController trace records into control records.
///
/// This is the paradigm with an actuator path, so it is the one the velocity
/// provenance rules are written for: every observation records which assistance
/// method and version were in force and with what parameter, what the decoded
/// command was, where the reference velocity came from, what the assisted
/// velocity was, and whether it was applied. A replay that has the decoded
/// command and the assistance record can recompute the applied velocity; one
/// that has only the applied velocity cannot say who produced it.
class CenterOutTraceWriter final : public ExperimentTraceSource
{
  public:
    /// Bind to the controller that will run, and to the host instant its
    /// experiment clock is anchored at.
    ///
    /// The configuration is read from the controller rather than handed in
    /// again, so the records describe the configuration the run is executing
    /// and cannot describe a second, equally valid one. The host epoch is a
    /// parameter because it is genuinely the caller's: it is the host instant
    /// the session's experiment time zero corresponds to, and no object in this
    /// layer knows it.
    CenterOutTraceWriter(CenterOutController& controller, streaming::HostTimeNs host_epoch_ns)
        : controller_(controller), host_epoch_ns_(host_epoch_ns)
    {
        presentations_.prepare(64);
        decoder_publications_.prepare(64);
    }

    /// Set the host instant corresponding to experiment time zero.
    ///
    /// Adaptive sessions are prepared before their Python owner is ready to
    /// start the runtime. Allow that owner to capture the epoch at the actual
    /// run boundary, but freeze it once configuration writing has begun.
    [[nodiscard]] bool set_host_epoch(streaming::HostTimeNs host_epoch_ns) noexcept
    {
        if (configuration_written_)
            return false;
        host_epoch_ns_ = host_epoch_ns;
        return true;
    }

    /// Accept one presentation lifecycle/software-timing report. Bounded and
    /// lock-free; false means the report became an ordinary experiment trace
    /// loss.
    [[nodiscard]] bool report_presentation(const PresentationSoftwareEvidence& evidence) noexcept;
    /// Accept one decoder publication. The fingerprint is the canonical
    /// lowercase SHA-256 hex digest of the complete immutable pipeline plan.
    [[nodiscard]] bool
    report_decoder_publication(std::uint64_t version, std::string_view plan_fingerprint,
                               std::uint64_t training_blocks, std::uint64_t training_trials,
                               TrialOrdinal completed_trials, ExperimentTimeNs time_ns) noexcept;
    /// Freeze one presentation configuration before the session writes its
    /// configuration records. Duplicate and post-start reports are rejected.
    [[nodiscard]] bool
    report_presentation_config(const CenterOutPresentationConfigRecord& config) noexcept;
    /// The task configuration frozen by the controller this writer records.
    [[nodiscard]] const center_out::CenterOut2DConfig& task_configuration() const noexcept
    {
        return controller_.configuration().task;
    }
    void require_presentation_evidence(bool required) noexcept
    {
        require_presentation_evidence_ = required;
    }

    /// Name the schema fingerprints the session ran between. Both default to
    /// zero, which means "the caller declared none" rather than "there were
    /// none".
    void declare_schemas(std::uint64_t input, std::uint64_t output) noexcept
    {
        input_schema_ = input;
        output_schema_ = output;
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
    void write_trace(ExperimentTraceSink& sink, const CenterOutControlTrace& trace) noexcept;
    void write_aborted_trial(ExperimentTraceSink& sink,
                             const CenterOutControlTrace& trace) noexcept;
    void write_step(ExperimentTraceSink& sink,
                    const center_out::CenterOutStepResult& step) noexcept;
    void write_velocity(ExperimentTraceSink& sink, const CenterOutControlTrace& trace) noexcept;
    void write_target(ExperimentTraceSink& sink, const center_out::CenterOutSnapshot& snapshot,
                      ExperimentTimeNs time_ns) noexcept;
    void write_presentation(ExperimentTraceSink& sink,
                            const PresentationSoftwareEvidence& evidence) noexcept;
    void write_decoder_publication(ExperimentTraceSink& sink,
                                   const DecoderPublicationEvidence& evidence) noexcept;
    void write_presentation_config(ExperimentTraceSink& sink, ExperimentTimeNs time_ns) noexcept;
    /// Write one abnormal-condition fault row (the abnormal-shape record whose
    /// code is the condition's contract name). Mirrors the Speech writer's helper.
    void write_abnormal(ExperimentTraceSink& sink, const AbnormalEvent& event) noexcept;
    void forget_target() noexcept;

    /// The full integration configuration the controller froze at prepare().
    [[nodiscard]] const CenterOutControllerConfig& config() const noexcept
    {
        return controller_.configuration();
    }

    CenterOutController& controller_;
    streaming::HostTimeNs host_epoch_ns_{};
    std::uint64_t input_schema_{};
    std::uint64_t output_schema_{};
    bool has_target_{};
    TargetId last_target_{kUnsetTargetId};
    SequenceOrdinal active_state_sequence_{};
    TrialOrdinal completed_{};
    std::uint64_t successes_{};
    BoundedTraceQueue<PresentationSoftwareEvidence> presentations_{};
    BoundedTraceQueue<DecoderPublicationEvidence> decoder_publications_{};
    std::uint64_t presentation_reports_{};
    std::uint64_t presentation_failures_{};
    std::uint64_t successful_presentations_{};
    bool require_presentation_evidence_{};
    /// This writer's own reporter, separate from the controller's. Presentation
    /// evidence is drained on the bridge thread while the controller meets its
    /// own conditions on the runtime thread; one shared non-atomic reporter would
    /// race, so -- like the Speech writer -- this one keeps its own and
    /// ::abnormal_summary merges the two.
    AbnormalReporter abnormal_{};
    CenterOutPresentationConfigRecord presentation_config_{};
    bool has_presentation_config_{};
    bool configuration_written_{};
};

} // namespace neurale::execution
