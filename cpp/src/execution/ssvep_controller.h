// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once

#include <atomic>
#include <memory>
#include <span>
#include <thread>
#include <vector>

#include <neurale/experiments/ssvep.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/frame_validation.h>

#include "bounded_trace_queue.h"
#include "interval_runner.h"
#include "session.h"

namespace neurale::execution
{

using namespace neurale::experiments;
struct SSVEPModelPublication
{
    std::string plan;
    std::string fingerprint;
    ExperimentTimeNs time_ns{};
};
struct SSVEPDisplayEvidence
{
    ssvep::SSVEPSnapshot snapshot;
    std::uint64_t requested{}, submitted{}, presented{};
    bool expired{};
};

class SSVEPController final : public ExperimentTraceSource
{
  public:
    SSVEPController(const streaming::StreamSchema& schema, ssvep::SSVEPConfig task,
                    std::size_t calibration_trials, IntervalRunner& processing);
    ~SSVEPController() override;
    void set_host_epoch(streaming::HostTimeNs epoch) noexcept
    {
        epoch_ = epoch;
    }
    void publish(SSVEPModelPublication publication);
    bool report_display(const SSVEPDisplayEvidence& value) noexcept
    {
        return displays_.try_push(value) == streaming::StreamStatus::ok;
    }
    std::size_t ready_trials() const noexcept
    {
        return processing_.ready();
    }
    bool training() const noexcept
    {
        return training_.load(std::memory_order_acquire);
    }
    bool complete() const noexcept
    {
        return complete_.load(std::memory_order_acquire);
    }
    std::span<const double> features(std::size_t trial) const;
    const ssvep::SSVEPTrial& trial(std::size_t index) const;
    std::size_t completed_trials() const noexcept
    {
        return completed_.load(std::memory_order_acquire);
    }
    streaming::StreamStatus pop_snapshot(ssvep::SSVEPSnapshot& snapshot) noexcept;
    void cancel() noexcept;
    streaming::StreamStatus status() const noexcept
    {
        const auto value = task_status_.load(std::memory_order_acquire);
        return value == streaming::StreamStatus::ok ? processing_.status() : value;
    }

    ContractStatus validate() const noexcept override;
    streaming::StreamStatus start_execution(ExperimentTimeNs time_ns) noexcept override;
    void stop_execution(bool aborting) noexcept override;
    std::uint64_t dropped_trace_count() const noexcept override;
    SessionMetadata metadata() const noexcept override;
    void write_configuration(ExperimentTraceSink&, ExperimentTimeNs) noexcept override;
    std::size_t drain(ExperimentTraceSink&, std::size_t) noexcept override;
    void write_summary(ExperimentTraceSink&, ExperimentTimeNs) noexcept override;
    AbnormalSummary abnormal_summary() const noexcept override
    {
        return {};
    }
    void halt(ExperimentTimeNs, AbnormalCondition) noexcept override
    {
        cancel();
    }

  private:
    streaming::StreamStatus accept(const ssvep::SSVEPStepResult& result) noexcept;
    streaming::StreamStatus advance(ExperimentTimeNs now) noexcept;
    streaming::StreamStatus apply_result(ExperimentTimeNs now, const IntervalResult&) noexcept;
    void task_main() noexcept;
    ssvep::SSVEPConfig task_;
    std::size_t calibration_, n_features_;
    IntervalRunner& processing_;
    std::vector<ssvep::SSVEPTrial> trials_;
    ssvep::SSVEPMachine machine_;
    BoundedTraceQueue<ssvep::SSVEPStepResult> traces_;
    BoundedTraceQueue<ssvep::SSVEPSnapshot> snapshots_;
    BoundedTraceQueue<SSVEPDisplayEvidence> displays_;
    std::vector<std::string> feature_names_;
    SSVEPModelPublication publication_;
    bool publication_written_{};
    std::atomic<bool> published_{};
    std::atomic<std::size_t> completed_{};
    std::atomic<bool> training_{}, complete_{}, cancelled_{};
    streaming::HostTimeNs epoch_{};
    std::uint64_t window_ns_{}, shift_ns_{}, scheduled_trial_{UINT64_MAX};
    ExperimentTimeNs last_time_{};
    bool started_{};
    std::atomic<streaming::StreamStatus> task_status_{streaming::StreamStatus::ok};
    std::thread task_thread_;
};
} // namespace neurale::execution
