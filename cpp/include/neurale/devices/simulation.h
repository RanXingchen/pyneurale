/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include <neurale/signal/neural_simulation.h>
#include <neurale/signal/simulation.h>
#include <neurale/streaming/clock.h>
#include <neurale/streaming/neural_intent.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/source.h>

namespace neurale::devices::simulation
{

/** Exact acquisition-side action applied before one data-frame ordinal. */
enum class AcquisitionEventKind : std::uint8_t
{
    sample_loss,
    would_block,
    stall,
    disconnect,
    source_fault,
    device_tick_jump,
    device_restart,
};

/**
 * @brief One fixed control-plane event in a deterministic acquisition schedule.
 *
 * Only the field named by ``kind`` may be nonzero. Events are ordered by
 * ``frame_ordinal``; events at the same ordinal retain input order.
 */
struct AcquisitionEvent
{
    std::uint64_t frame_ordinal{};
    AcquisitionEventKind kind{AcquisitionEventKind::would_block};
    std::uint64_t n_samples{};
    streaming::HostTimeNs duration_ns{};
    std::int64_t device_tick_delta{};
    streaming::DeviceTick restart_tick{};
};

/** Manually advanced host clock for deterministic acquisition and pacing tests. */
class ManualHostClock final : public streaming::NativeClock
{
  public:
    explicit ManualHostClock(streaming::HostTimeNs initial_time_ns = 0) noexcept;

    [[nodiscard]] streaming::HostTimeNs now_ns() noexcept override;
    void wait_until(streaming::HostTimeNs deadline_ns) noexcept override;
    void wake() noexcept override;

    [[nodiscard]] bool set(streaming::HostTimeNs time_ns) noexcept;
    void advance(streaming::HostTimeNs duration_ns) noexcept;

  private:
    std::atomic<streaming::HostTimeNs> now_ns_{};
    std::mutex mutex_;
    std::condition_variable condition_;
    std::uint64_t revision_{};
};

/** Immutable control-plane configuration for one simulated neural source. */
struct SimulatedNeuralSourceConfig
{
    /// Zero requests a source-owned nonzero run identifier at construction.
    streaming::SessionId session_id{};
    streaming::SchemaId schema_id{1};
    streaming::SignalId signal_id{1};
    streaming::ClockDomainId clock_domain{1};
    std::uint32_t nominal_samples_per_frame{};
    std::uint32_t max_samples_per_frame{};
    streaming::RationalRate fs{};
    /// Payload element type. ``float64`` writes generated values unchanged;
    /// ``int16`` quantizes them to saturating 16-bit counts. No other dtype is
    /// accepted, so an unsupported one is rejected instead of approximated.
    streaming::SignalDType sample_dtype{streaming::SignalDType::float64};
    streaming::PhysicalUnit physical_unit{streaming::PhysicalUnit::volts};
    streaming::ChannelSetId channel_set_id{1};
    std::vector<std::string> channel_names;
    std::vector<double> channel_impedances_ohm;
    streaming::CalibrationId calibration_id{};
    streaming::ReferenceId reference_id{};
    streaming::SampleIndex initial_sample_idx{};
    std::optional<std::uint64_t> total_sample_count;
    bool device_ticks{true};
    streaming::DeviceTick initial_device_tick{};
    std::int64_t clock_offset_ns{};
    std::int64_t clock_drift_ppm{};
    streaming::HostTimeNs clock_sync_uncertainty_ns{};
    bool paced{true};
    std::vector<AcquisitionEvent> events;
};

/**
 * @brief Prepared NSP-class simulator implementing NativeFrameSource.
 *
 * The source owns a deterministic signal generator and writes one sample-major
 * neural block of the configured dtype into caller-owned frame storage.
 * Construction freezes schema and capacity; read() performs no allocation --
 * the quantization scratch a non-float64 dtype needs is sized once, at
 * construction. A paced acquisition latches its host epoch on the first valid
 * read after construction or reset.
 *
 * read(), read_message(), and reset() share one exclusive gate, so a reset
 * concurrent with an in-flight read is refused with `invalid_state` rather
 * than racing the acquisition state it would rewrite. cancel() and close()
 * deliberately stay outside that gate: they must remain callable while a read
 * is blocked, which is what makes a blocked read return.
 */
class SimulatedNeuralSource : public streaming::NativeFrameSource
{
  public:
    SimulatedNeuralSource(signal::simulation::SignalGenerator generator,
                          SimulatedNeuralSourceConfig config);
    SimulatedNeuralSource(signal::simulation::SignalGenerator generator,
                          SimulatedNeuralSourceConfig config, streaming::NativeClock& clock);
    SimulatedNeuralSource(signal::simulation::SignalGenerator generator,
                          SimulatedNeuralSourceConfig config,
                          std::shared_ptr<ManualHostClock> clock);
    ~SimulatedNeuralSource() override = default;

    SimulatedNeuralSource(const SimulatedNeuralSource&) = delete;
    SimulatedNeuralSource& operator=(const SimulatedNeuralSource&) = delete;
    SimulatedNeuralSource(SimulatedNeuralSource&&) = delete;
    SimulatedNeuralSource& operator=(SimulatedNeuralSource&&) = delete;

    [[nodiscard]] streaming::StreamStatus read(streaming::MutableFrame& frame) noexcept override;
    [[nodiscard]] streaming::StreamStatus
    read_message(streaming::MutableFrame& frame,
                 streaming::DiscontinuityLease& discontinuity) noexcept override;
    [[nodiscard]] bool produces_discontinuities() const noexcept override;
    [[nodiscard]] streaming::SessionId session_id() const noexcept
    {
        return config_.session_id;
    }
    void cancel() noexcept override;
    void close() noexcept;
    [[nodiscard]] streaming::StreamStatus reset() noexcept override;

    [[nodiscard]] bool closed() const noexcept
    {
        return closed_.load(std::memory_order_acquire);
    }

    [[nodiscard]] const streaming::StreamSchema& schema() const noexcept
    {
        return schema_;
    }

    [[nodiscard]] std::uint64_t frames_emitted() const noexcept
    {
        return frames_emitted_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::uint64_t samples_emitted() const noexcept
    {
        return samples_emitted_.load(std::memory_order_acquire);
    }

  protected:
    [[nodiscard]] virtual streaming::StreamStatus
    generate_samples(streaming::SampleIndex sample_start, std::uint32_t n_samples,
                     std::span<double> output) noexcept;
    virtual void reset_generator() noexcept {}

  private:
    [[nodiscard]] streaming::HostTimeNs deadline_for(std::uint64_t sample_offset) const noexcept;
    [[nodiscard]] streaming::StreamStatus wait_for(streaming::HostTimeNs deadline) noexcept;
    [[nodiscard]] streaming::StreamStatus
    read_impl(streaming::MutableFrame& frame,
              streaming::DiscontinuityLease* discontinuity) noexcept;
    [[nodiscard]] streaming::StreamStatus
    apply_scheduled_events(streaming::DiscontinuityLease* discontinuity) noexcept;
    [[nodiscard]] streaming::StreamStatus
    publish_gap(streaming::DiscontinuityLease* discontinuity, streaming::GapReason reason,
                streaming::SampleIndex expected_sample, streaming::SampleIndex actual_sample,
                streaming::DeviceTick expected_tick, streaming::DeviceTick actual_tick,
                std::uint64_t missing_samples) noexcept;
    [[nodiscard]] bool sync_host_time(streaming::HostTimeNs deadline,
                                      streaming::HostTimeNs& result) const noexcept;

    signal::simulation::SignalGenerator generator_;
    SimulatedNeuralSourceConfig config_;
    streaming::StreamSchema schema_;
    /// Generation target for a dtype the generator cannot write directly.
    /// Empty for float64, where generation lands in the payload itself.
    std::vector<double> quantization_scratch_;
    streaming::RationalRate device_tick_rate_{};
    std::shared_ptr<ManualHostClock> owned_host_clock_;
    streaming::NativeClock& clock_;
    streaming::HostTimeNs epoch_host_ns_{};
    bool epoch_initialized_{};
    std::uint64_t acquisition_sample_offset_{};
    streaming::DeviceTick device_tick_{};
    std::uint32_t clock_sync_generation_{1};
    std::size_t next_event_{};
    bool has_discontinuity_events_{};
    std::atomic<std::uint64_t> frames_emitted_{};
    std::atomic<std::uint64_t> samples_emitted_{};
    /// Exclusive access to the acquisition state above, taken by every read
    /// and reset. Not taken by cancel() or close(), which touch only atomics.
    std::atomic<bool> operation_gate_{};
    std::atomic<bool> cancelled_{};
    std::atomic<bool> closed_{};
};

struct NeuralDriftSchedule
{
    std::uint64_t start_ordinal{};
    std::uint64_t end_ordinal{};
    double start_progress{};
    double end_progress{1.0};
};

struct AppliedNeuralControl
{
    std::uint64_t intent_sequence{};
    double intent_x{};
    double intent_y{};
    std::uint64_t context_ordinal{};
    bool intent_valid{};
    double drift_progress{};
    streaming::SampleIndex first_sample_index{};
    streaming::SampleIndex last_sample_index{};
    std::uint64_t generated_frame_sequence{};
};

class ManualNeuralControlState;

/** Opt-in feedback-driven neural simulator; ordinary simulated sources are unchanged. */
class IntentDrivenNeuralSource final : public SimulatedNeuralSource
{
  public:
    IntentDrivenNeuralSource(signal::simulation::NeuralSignalConfig neural_config,
                             SimulatedNeuralSourceConfig source_config,
                             std::optional<NeuralDriftSchedule> drift_schedule = std::nullopt,
                             std::size_t evidence_capacity = 4096);

    [[nodiscard]] streaming::StreamStatus
    bind_intent_source(std::shared_ptr<streaming::NeuralIntentSource> source) noexcept;
    [[nodiscard]] streaming::StreamStatus publish_control(double intent_x, double intent_y,
                                                          double drift_progress,
                                                          std::uint64_t context_ordinal) noexcept;
    [[nodiscard]] streaming::StreamStatus
    try_pop_applied_control(AppliedNeuralControl& control) noexcept;
    [[nodiscard]] std::uint64_t dropped_applied_control_count() const noexcept;
    [[nodiscard]] std::uint64_t drift_fingerprint() const noexcept;

  protected:
    [[nodiscard]] streaming::StreamStatus
    generate_samples(streaming::SampleIndex sample_start, std::uint32_t n_samples,
                     std::span<double> output) noexcept override;
    void reset_generator() noexcept override;

  private:
    enum class ControlMode : std::uint8_t
    {
        undecided,
        binding,
        manual,
        bound,
    };

    [[nodiscard]] double drift_progress(std::uint64_t context_ordinal) const noexcept;
    void push_applied_control(const AppliedNeuralControl& control) noexcept;

    signal::simulation::NeuralSignalGenerator generator_;
    std::shared_ptr<ManualNeuralControlState> manual_source_;
    std::shared_ptr<streaming::NeuralIntentSource> intent_source_;
    std::optional<NeuralDriftSchedule> drift_schedule_;
    std::vector<AppliedNeuralControl> evidence_;
    std::atomic<std::size_t> evidence_read_{};
    std::atomic<std::size_t> evidence_write_{};
    std::atomic<std::uint64_t> evidence_dropped_{};
    std::atomic<std::uint64_t> manual_sequence_{};
    std::uint64_t generated_frame_sequence_{};
    std::atomic<ControlMode> control_mode_{ControlMode::undecided};
};

} // namespace neurale::devices::simulation
