/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <span>
#include <thread>
#include <vector>

#include <neurale/streaming/actuator.h>
#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/consumer.h>
#include <neurale/streaming/continuity.h>
#include <neurale/streaming/critical_observer.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/fault.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/observer.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/realtime_config.h>
#include <neurale/streaming/safety.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/source.h>
#include <neurale/streaming/spsc_ring.h>
#include <neurale/streaming/stats.h>
#include <neurale/streaming/stream_message.h>

namespace neurale::streaming
{

class TerminalFrameEmitter;

#if defined(NEURALE_STREAMING_TEST_HOOKS)
struct RuntimeTestHooks
{
    std::atomic<std::size_t>* thread_starts_before_failure{};
    std::atomic<bool>* primary_fault_claimed{};
    std::atomic<bool>* primary_fault_reader_waiting{};
    std::atomic<bool>* release_primary_fault{};
};
#endif

enum class RuntimeState : std::uint8_t
{
    created,
    prepared,
    running,
    stopping,
    stopped,
    failed,
};

/// Native data plane with bounded ingress, critical, and observer edges.
class NativeStreamRunner
{
  public:
    NativeStreamRunner(StreamSchema schema, RealtimeConfig config, NativeFrameSource& source,
                       NativeFrameProcessor& processor, NativeFrameConsumer& consumer);
    NativeStreamRunner(StreamSchema schema, RealtimeConfig config, NativeFrameSource& source,
                       NativeFrameProcessor& processor, NativeFrameConsumer& consumer,
                       NativeClock& clock, SafetyController& safety_controller);
    NativeStreamRunner(StreamSchema schema, RealtimeConfig config, NativeFrameSource& source,
                       NativeFrameProcessor& processor, NativeActuator& actuator);
    NativeStreamRunner(StreamSchema schema, RealtimeConfig config, NativeFrameSource& source,
                       NativeFrameProcessor& processor, NativeActuator& actuator,
                       NativeClock& clock, SafetyController& safety_controller);
    NativeStreamRunner(StreamSchema schema, RealtimeConfig config, NativeFrameSource& source,
                       NativeFrameProcessor& processor, NativeActuator& actuator,
                       NativeClock& clock, SafetyController& safety_controller,
                       RealtimePlatform& realtime_platform);
    ~NativeStreamRunner() noexcept;

    NativeStreamRunner(const NativeStreamRunner&) = delete;
    NativeStreamRunner& operator=(const NativeStreamRunner&) = delete;
    NativeStreamRunner(NativeStreamRunner&&) = delete;
    NativeStreamRunner& operator=(NativeStreamRunner&&) = delete;

    /// Validate topology and allocate every bounded data-plane resource.
    [[nodiscard]] StreamStatus prepare();
    [[nodiscard]] StreamStatus arm() noexcept;
    [[nodiscard]] StreamStatus start();
    [[nodiscard]] StreamStatus join() noexcept;
    [[nodiscard]] StreamStatus run();
    [[nodiscard]] StreamStatus stop() noexcept;
    [[nodiscard]] StreamStatus abort() noexcept;
    [[nodiscard]] StreamStatus reset() noexcept;

    [[nodiscard]] RuntimeState state() const noexcept
    {
        return state_.load(std::memory_order_acquire);
    }
    [[nodiscard]] RuntimeStats stats() const noexcept
    {
        return counters_.snapshot();
    }
    [[nodiscard]] RuntimeHeartbeatSnapshot heartbeat() const noexcept;
    [[nodiscard]] RealtimeConfigurationStatus realtime_configuration_status() const noexcept;
    [[nodiscard]] std::optional<FaultRecord> primary_fault() const noexcept;
    [[nodiscard]] std::size_t copy_fault_history(std::span<FaultRecord> destination) const noexcept;
    [[nodiscard]] std::size_t fault_history_capacity() const noexcept
    {
        return config_.fault_history_capacity;
    }
    [[nodiscard]] const RealtimeConfig& resolved_config() const noexcept
    {
        return config_;
    }
    [[nodiscard]] std::size_t outstanding_frames() const noexcept;
    [[nodiscard]] std::size_t outstanding_discontinuities() const noexcept;
    [[nodiscard]] StreamStatus add_observer(NativeObserver& observer, ObserverEdgeConfig config);
    [[nodiscard]] StreamStatus add_critical_observer(NativeCriticalObserver& observer,
                                                     ObserverEdgeConfig config);
    /// Register a lossless observer for continuity-accepted source messages.
    /// Delivery is synchronous on the processing thread before the processor
    /// sees the message; implementations must therefore be bounded and
    /// non-blocking. This is intended for recording the acquisition stream
    /// when the processor changes the output schema.
    [[nodiscard]] StreamStatus add_input_critical_observer(NativeCriticalObserver& observer,
                                                           ObserverEdgeConfig config);
    [[nodiscard]] StreamStatus detach_observer(ObserverId id) noexcept;
    [[nodiscard]] std::optional<ObserverEdgeStats> observer_stats(ObserverId id) const noexcept;
    [[nodiscard]] std::size_t
    copy_observer_drop_ranges(ObserverId id,
                              std::span<ObserverDropRange> destination) const noexcept;

#if defined(NEURALE_STREAMING_TEST_HOOKS)
    void set_test_hooks(RuntimeTestHooks* hooks) noexcept
    {
        test_hooks_ = hooks;
    }
#endif

  private:
    struct FaultChannel;
    struct ObserverRegistration;
    struct ObserverEdge;
    struct CriticalObserverEdge;
    struct ThreadStartup;

    enum class EndReason : std::uint8_t
    {
        none,
        end_of_stream,
        stop,
        abort,
        fault,
    };

    void acquisition_loop() noexcept;
    void processing_loop() noexcept;
    void actuator_loop() noexcept;
    void observer_dispatch_loop() noexcept;
    void observer_loop(ObserverEdge& edge) noexcept;
    void watchdog_loop() noexcept;
    void configured_loop(RealtimeThreadRole role,
                         void (NativeStreamRunner::*loop)() noexcept) noexcept;
    void observer_startup_loop(ObserverEdge& edge) noexcept;
    [[nodiscard]] StreamStatus process_frame(FrameLease frame) noexcept;
    [[nodiscard]] StreamStatus handle_discontinuity(DiscontinuityLease discontinuity) noexcept;
    [[nodiscard]] StreamStatus finish_gracefully() noexcept;
    [[nodiscard]] StreamStatus finish_output(StreamStatus processor_status,
                                             FaultCode processor_code) noexcept;
    void record_fault(FaultCode code, StreamStatus status, FaultStage stage,
                      std::uint32_t detail = 0, const FrameHeader* frame = nullptr,
                      HostTimeNs detected_at_ns = 0) noexcept;
    void request_abort(bool fault) noexcept;
    void trigger_deadline_fault(FaultCode code, FaultStage stage, SafetyReason reason,
                                HostTimeNs detected_at_ns, std::uint32_t detail = 0) noexcept;
    [[nodiscard]] StreamStatus inhibit_safety(SafetyReason reason) noexcept;
    [[nodiscard]] StreamStatus wait_for_shutdown() noexcept;
    [[nodiscard]] std::optional<HostTimeNs> oldest_ingress_timestamp() const noexcept;
    void begin_shutdown() noexcept;
    void finish_producer(EndReason reason) noexcept;
    void notify_processing() noexcept;
    void notify_critical() noexcept;
    void notify_observer_dispatch() noexcept;
    [[nodiscard]] StreamStatus dispatch_frame_to_observer(ObserverEdge& edge,
                                                          FrameView frame) noexcept;
    [[nodiscard]] StreamStatus
    dispatch_discontinuity_to_observer(ObserverEdge& edge,
                                       const Discontinuity& discontinuity) noexcept;
    [[nodiscard]] StreamStatus
    dispatch_to_critical_observers(const StreamMessage& message) noexcept;
    [[nodiscard]] StreamStatus dispatch_to_input_critical_observers(FrameView frame) noexcept;
    [[nodiscard]] StreamStatus
    dispatch_to_input_critical_observers(const Discontinuity& discontinuity) noexcept;
    void note_rejected_before_acceptance(AcceptedMessageKind kind,
                                         std::uint64_t frame_sequence) noexcept;
    void report_critical_failure(CriticalObserverEdge& edge, StreamStatus status) noexcept;
    void finish_critical_observers() noexcept;
    void cancel_critical_observers() noexcept;
    [[nodiscard]] RuntimeTerminalReason terminal_reason() const noexcept;
    void drain_ingress() noexcept;
    void clear_resources() noexcept;
    [[nodiscard]] StreamStatus join_threads(bool lifecycle_locked = false) noexcept;
    [[nodiscard]] bool workers_done() const noexcept;
    [[nodiscard]] std::uint32_t pending_worker_mask() const noexcept;
    [[nodiscard]] std::optional<FaultRecord> primary_fault_unlocked() const noexcept;

    StreamSchema schema_;
    RealtimeConfig config_;
    NativeFrameSource& source_;
    NativeFrameProcessor& processor_;
    NativeActuator& actuator_;
    NativeFrameConsumer* terminal_consumer_{};
    NativeClock& clock_;
    SafetyController& safety_controller_;
    RealtimePlatform& realtime_platform_;
    RuntimeCounters counters_{};
    std::optional<PreparedProcessorContract> processor_contract_;
    bool processor_prepared_{};
    std::unique_ptr<FramePool> frame_pool_;
    std::unique_ptr<DiscontinuityPool> discontinuity_pool_;
    std::unique_ptr<ContinuityChecker> continuity_;
    std::unique_ptr<FrameValidator> output_validator_;
    std::unique_ptr<SpscRing<StreamMessage>> ingress_;
    std::unique_ptr<SpscRing<StreamMessage>> critical_edge_;
    std::unique_ptr<SpscRing<StreamMessage>> observer_dispatch_;
    std::unique_ptr<std::atomic<HostTimeNs>[]> ingress_timestamps_;
    std::unique_ptr<TerminalFrameEmitter> terminal_emitter_;
    std::unique_ptr<FaultChannel> faults_;
    std::unique_ptr<ThreadStartup> thread_startup_;
    std::vector<ObserverRegistration> observer_registrations_;
    std::vector<std::unique_ptr<ObserverEdge>> observer_edges_;
    std::vector<std::unique_ptr<CriticalObserverEdge>> critical_observer_edges_;
    std::thread acquisition_thread_;
    std::thread processing_thread_;
    std::thread actuator_thread_;
    std::thread observer_dispatch_thread_;
    std::thread watchdog_thread_;
    mutable std::shared_mutex lifecycle_mutex_;
    std::mutex join_mutex_;
    std::atomic<RuntimeState> state_{RuntimeState::created};
    std::atomic<EndReason> end_reason_{EndReason::none};
    std::atomic<bool> stop_requested_{};
    std::atomic<bool> abort_requested_{};
    std::atomic<bool> producer_done_{};
    std::atomic<bool> acquisition_done_{};
    std::atomic<bool> processing_done_{};
    std::atomic<bool> actuator_done_{};
    std::atomic<bool> observer_dispatch_done_{};
    std::atomic<bool> watchdog_stop_{};
    std::atomic<bool> shutdown_claimed_{};
    std::atomic<bool> shutdown_active_{};
    std::atomic<bool> shutdown_timeout_recorded_{};
    std::atomic<std::uint64_t> ingress_epoch_{};
    std::atomic<std::uint64_t> critical_epoch_{};
    std::atomic<std::uint64_t> observer_dispatch_epoch_{};
    std::atomic<std::uint64_t> next_command_id_{};
    std::atomic<std::uint64_t> next_data_message_ordinal_{};
    std::atomic<std::uint64_t> next_input_data_message_ordinal_{};
    std::atomic<HostTimeNs> runtime_started_ns_{};
#if defined(NEURALE_STREAMING_TEST_HOOKS)
    RuntimeTestHooks* test_hooks_{};
#endif
    std::atomic<HostTimeNs> shutdown_started_ns_{};
    std::atomic<HostTimeNs> last_source_heartbeat_ns_{};
    std::atomic<HostTimeNs> current_processor_start_ns_{};
    std::atomic<SessionId> current_processor_session_id_{};
    std::atomic<std::uint64_t> current_processor_frame_sequence_{};
    std::atomic<SchemaId> current_processor_schema_id_{};
    std::atomic<ClockDomainId> current_processor_clock_domain_{};
    std::atomic<HostTimeNs> last_processor_completion_ns_{};
    std::atomic<HostTimeNs> last_valid_output_ns_{};
    std::atomic<HostTimeNs> current_actuator_start_ns_{};
    std::atomic<HostTimeNs> last_inhibit_ns_{};
    std::atomic<std::uint64_t> runtime_generation_{};
    std::atomic<bool> processor_active_{};
    std::atomic<bool> actuator_active_{};
    std::atomic<bool> output_valid_{};
    std::atomic<bool> safety_inhibited_{true};
    FrameHeader current_frame_{};
    RealtimeConfigurationStatus realtime_status_{};
};

using NativeRuntime = NativeStreamRunner;

} // namespace neurale::streaming
