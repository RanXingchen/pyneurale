/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/observer_queue.h>
#include <neurale/streaming/runtime.h>

#include "terminal_frame_emitter.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <type_traits>
#include <utility>

namespace neurale::streaming
{
namespace
{

static_assert(std::is_trivially_copyable_v<FaultRecord>);
static_assert(std::is_trivially_copyable_v<RuntimeStats>);
static_assert(std::is_trivially_copyable_v<RuntimeHeartbeatSnapshot>);
static_assert(noexcept(std::declval<NativeFrameSource&>().read(std::declval<MutableFrame&>())));
static_assert(noexcept(std::declval<NativeFrameSource&>().cancel()));
static_assert(noexcept(std::declval<SafetyController&>().inhibit(std::declval<SafetyReason>())));

[[nodiscard]] std::size_t schema_payload_bytes(const StreamSchema& schema)
{
    std::size_t total = 0;
    for (const auto& signal : schema.signals())
    {
        if (signal.max_block_bytes > std::numeric_limits<std::size_t>::max() - total)
        {
            throw std::overflow_error("schema payload size overflows size_t");
        }
        total += signal.max_block_bytes;
    }
    return total;
}

[[nodiscard]] std::uint64_t minimum_frame_period_ns(const StreamSchema& schema)
{
    auto period = std::numeric_limits<std::uint64_t>::max();
    for (const auto& signal : schema.signals())
    {
        if (signal.fs.numerator == 0 || signal.nominal_block_samples == 0)
        {
            continue;
        }
        const auto value =
            std::ceil(static_cast<long double>(signal.nominal_block_samples) *
                      static_cast<long double>(signal.fs.denominator) * 1'000'000'000.0L /
                      static_cast<long double>(signal.fs.numerator));
        if (value > static_cast<long double>(std::numeric_limits<std::uint64_t>::max()))
        {
            throw std::overflow_error("schema frame period overflows uint64_t");
        }
        const auto bounded = static_cast<std::uint64_t>(std::max(1.0L, value));
        period = std::min(period, bounded);
    }
    return period == std::numeric_limits<std::uint64_t>::max() ? 1'000'000 : period;
}

[[nodiscard]] std::uint64_t output_warmup_ns(const StreamSchema& schema) noexcept
{
    std::uint64_t warmup = 0;
    for (const auto& feature_set : schema.feature_sets().descriptors())
    {
        warmup = std::max(warmup, feature_set.window_length_ns);
    }
    return warmup;
}

[[nodiscard]] std::size_t queue_capacity(RealtimeDuration duration, std::uint64_t period_ns)
{
    const auto duration_ns = static_cast<std::uint64_t>(duration.count());
    const auto frames = duration_ns / period_ns + (duration_ns % period_ns != 0 ? 1 : 0);
    if (frames > std::numeric_limits<std::size_t>::max() - 2)
    {
        throw std::overflow_error("derived queue capacity overflows size_t");
    }
    return std::max<std::size_t>(4, static_cast<std::size_t>(frames) + 2);
}

void resolve_automatic_resources(RealtimeConfig& config, const StreamSchema& input_schema,
                                 const PreparedProcessorContract& processor_contract,
                                 bool source_produces_discontinuities, bool has_observers)
{
    const auto& output_schema = processor_contract.output_schema;
    config.buffer_size =
        std::max(schema_payload_bytes(input_schema), schema_payload_bytes(output_schema));
    config.max_signal_blocks =
        std::max(input_schema.signals().size(), output_schema.signals().size());
    config.max_process_outputs = processor_contract.max_process_outputs_per_input;
    config.max_flush_outputs = processor_contract.max_flush_outputs;
    config.gaps_per_discontinuity = config.max_signal_blocks;

    const auto input_period = minimum_frame_period_ns(input_schema);
    const auto output_period = minimum_frame_period_ns(output_schema);
    const auto ingress_capacity = queue_capacity(config.max_ingress_dwell, input_period);
    const auto critical_capacity = queue_capacity(config.max_source_to_actuator_age, output_period);
    config.pool_capacity = PoolCapacityBudget{
        .source_owned = 1,
        .ingress_capacity = ingress_capacity,
        .processor_owned = processor_contract.required_resources.frame_pool_leases,
        .critical_edge_capacity = critical_capacity,
        .actuator_owned = 1,
        .observer_edge_capacity = has_observers ? critical_capacity : 0,
        .reserve = 2,
    };
    // Ordered gaps occupy the same bounded edges as frames. Two slots are not
    // enough for a source that can publish consecutive gaps before consumers run.
    config.discontinuity_capacity =
        source_produces_discontinuities ? config.pool_capacity.required_buffer_count() : 1;

    if (config.max_output_age.count() == 0)
    {
        const auto ingress_age = static_cast<std::uint64_t>(config.max_ingress_dwell.count());
        const auto limit = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
        const auto warmup = output_warmup_ns(output_schema);
        if (warmup > limit - ingress_age || output_period > (limit - ingress_age - warmup) / 2)
        {
            throw std::overflow_error("derived output age overflows realtime duration");
        }
        const auto output_age = ingress_age + warmup + 2 * output_period;
        config.max_output_age = RealtimeDuration{static_cast<std::int64_t>(output_age)};
    }
    if (config.watchdog_period.count() == 0)
    {
        const auto shortest =
            std::min({config.source_stall_timeout, config.max_ingress_dwell,
                      config.processor_execution_deadline, config.max_source_to_actuator_age,
                      config.max_output_age, config.actuator_deadline});
        config.watchdog_period = std::max(RealtimeDuration{1}, shortest / 10);
    }
}

void validate_runtime_config(const StreamSchema& schema, const RealtimeConfig& config)
{
    config.validate();
    if (config.max_signal_blocks < schema.signals().size())
    {
        throw std::invalid_argument("max_signal_blocks is smaller than the stream schema");
    }
    std::uint64_t required_payload = 0;
    for (const auto& signal : schema.signals())
    {
        if (signal.max_block_bytes > std::numeric_limits<std::uint64_t>::max() - required_payload)
        {
            throw std::overflow_error("schema payload size overflows uint64_t");
        }
        required_payload += signal.max_block_bytes;
    }
    if (required_payload > config.buffer_size)
    {
        throw std::invalid_argument("buffer_size cannot hold one maximum-size schema frame");
    }
}

[[nodiscard]] bool equal_stream_schema(const StreamSchema& left, const StreamSchema& right) noexcept
{
    return left.equivalent(right);
}

[[nodiscard]] bool equal_processor_contract(const PreparedProcessorContract& left,
                                            const PreparedProcessorContract& right) noexcept
{
    return equal_stream_schema(left.accepted_input_schema, right.accepted_input_schema) &&
           equal_stream_schema(left.output_schema, right.output_schema) &&
           left.max_process_outputs_per_input == right.max_process_outputs_per_input &&
           left.max_flush_outputs == right.max_flush_outputs &&
           left.can_forward_input == right.can_forward_input &&
           left.required_resources.workspace_bytes == right.required_resources.workspace_bytes &&
           left.required_resources.frame_pool_leases == right.required_resources.frame_pool_leases;
}

enum class ProcessorContractDetail : std::uint32_t
{
    prepare_failed = 1,
    input_schema_mismatch,
    output_schema_exceeds_storage,
    process_output_limit,
    flush_output_limit,
    frame_pool_bound,
    changed_after_reset,
    missing_prepared_contract,
};

[[nodiscard]] HostTimeNs duration_ns(RealtimeDuration duration) noexcept
{
    return static_cast<HostTimeNs>(duration.count());
}

[[nodiscard]] HostTimeNs saturating_add(HostTimeNs time, RealtimeDuration duration) noexcept
{
    const auto delta = duration_ns(duration);
    constexpr auto maximum =
        static_cast<HostTimeNs>(std::numeric_limits<RealtimeDuration::rep>::max());
    return time >= maximum || delta > maximum - time ? maximum : time + delta;
}

[[nodiscard]] bool expired(HostTimeNs now, HostTimeNs started, RealtimeDuration limit) noexcept
{
    return now >= started && now - started >= duration_ns(limit);
}

struct RealtimeStartupAbort
{
};

} // namespace

struct NativeStreamRunner::FaultChannel
{
    struct Slot
    {
        std::atomic<std::uint8_t> ready{};
        FaultRecord value{};
    };

#if defined(NEURALE_STREAMING_TEST_HOOKS)
    explicit FaultChannel(std::size_t capacity, RuntimeTestHooks* hooks)
        : capacity(capacity), history(std::make_unique_for_overwrite<Slot[]>(capacity)),
          test_hooks(hooks)
    {
    }
#else
    explicit FaultChannel(std::size_t capacity)
        : capacity(capacity), history(std::make_unique_for_overwrite<Slot[]>(capacity))
    {
    }
#endif

    enum class Result : std::uint8_t
    {
        primary,
        secondary,
        dropped
    };

    [[nodiscard]] Result record(const FaultRecord& fault) noexcept
    {
        std::uint8_t empty = 0;
        if (primary.ready.compare_exchange_strong(empty, 1, std::memory_order_acq_rel,
                                                  std::memory_order_relaxed))
        {
#if defined(NEURALE_STREAMING_TEST_HOOKS)
            if (test_hooks != nullptr && test_hooks->primary_fault_claimed != nullptr &&
                test_hooks->release_primary_fault != nullptr)
            {
                test_hooks->primary_fault_claimed->store(true, std::memory_order_release);
                test_hooks->primary_fault_claimed->notify_all();
                test_hooks->release_primary_fault->wait(false, std::memory_order_acquire);
            }
#endif
            primary.value = fault;
            primary.ready.store(2, std::memory_order_release);
            primary.ready.notify_all();
            return Result::primary;
        }

        const auto idx = claims.fetch_add(1, std::memory_order_relaxed);
        if (idx >= capacity)
        {
            return Result::dropped;
        }
        history[idx].value = fault;
        history[idx].ready.store(1, std::memory_order_release);
        return Result::secondary;
    }

    [[nodiscard]] std::optional<FaultRecord> first() const noexcept
    {
        auto ready = primary.ready.load(std::memory_order_acquire);
        while (ready == 1)
        {
#if defined(NEURALE_STREAMING_TEST_HOOKS)
            if (test_hooks != nullptr && test_hooks->primary_fault_reader_waiting != nullptr)
            {
                test_hooks->primary_fault_reader_waiting->store(true, std::memory_order_release);
                test_hooks->primary_fault_reader_waiting->notify_all();
            }
#endif
            primary.ready.wait(1, std::memory_order_acquire);
            ready = primary.ready.load(std::memory_order_acquire);
        }
        if (ready != 2)
        {
            return std::nullopt;
        }
        return primary.value;
    }

    [[nodiscard]] std::size_t copy(std::span<FaultRecord> destination) const noexcept
    {
        const auto count = std::min({
            claims.load(std::memory_order_acquire),
            capacity,
            destination.size(),
        });
        std::size_t copied = 0;
        for (std::size_t i = 0; i < count; ++i)
        {
            if (history[i].ready.load(std::memory_order_acquire) != 0)
            {
                destination[copied++] = history[i].value;
            }
        }
        return copied;
    }

    const std::size_t capacity;
    Slot primary{};
    std::unique_ptr<Slot[]> history;
    std::atomic<std::size_t> claims{};
#if defined(NEURALE_STREAMING_TEST_HOOKS)
    RuntimeTestHooks* test_hooks{};
#endif
};

struct NativeStreamRunner::ObserverRegistration
{
    NativeObserver* observer{};
    NativeCriticalObserver* critical_observer{};
    ObserverEdgeConfig config{};
    bool input_tap{};
};

struct NativeStreamRunner::ObserverEdge
{
    struct DropSlot
    {
        std::atomic<std::uint8_t> ready{};
        ObserverDropRange range{};
    };

    ObserverEdge(NativeObserver& endpoint, ObserverEdgeConfig edge_config,
                 const RealtimeConfig& runtime_config)
        : observer(endpoint), config(edge_config),
          frame_pool(edge_config.capacity, runtime_config.buffer_size,
                     runtime_config.max_signal_blocks),
          discontinuity_pool(edge_config.capacity, runtime_config.gaps_per_discontinuity),
          queue(edge_config.capacity), drop_history(std::make_unique_for_overwrite<DropSlot[]>(
                                           edge_config.drop_history_capacity))
    {
    }

    void record_drop(std::uint64_t sequence) noexcept
    {
        dropped.fetch_add(1, std::memory_order_relaxed);
        const auto idx = drop_claims.fetch_add(1, std::memory_order_relaxed);
        if (idx >= config.drop_history_capacity)
        {
            drop_history_dropped.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        drop_history[idx].range = {sequence, sequence};
        drop_history[idx].ready.store(1, std::memory_order_release);
    }

    [[nodiscard]] ObserverEdgeStats snapshot() const noexcept
    {
        return ObserverEdgeStats{
            .id = config.id,
            .enqueued = enqueued.load(std::memory_order_relaxed),
            .delivered = delivered.load(std::memory_order_relaxed),
            .discontinuities = discontinuities.load(std::memory_order_relaxed),
            .dropped = dropped.load(std::memory_order_relaxed),
            .drop_range_count =
                std::min(drop_claims.load(std::memory_order_acquire), config.drop_history_capacity),
            .drop_history_dropped = drop_history_dropped.load(std::memory_order_relaxed),
            .failures = failures.load(std::memory_order_relaxed),
            .high_water_mark = high_water_mark.load(std::memory_order_relaxed),
            .last_enqueued_sequence = last_enqueued_sequence.load(std::memory_order_relaxed),
            .last_delivered_sequence = last_delivered_sequence.load(std::memory_order_relaxed),
            .detached = !active.load(std::memory_order_acquire),
        };
    }

    void observe_size() noexcept
    {
        const auto size = queue.approximate_size();
        auto current = high_water_mark.load(std::memory_order_relaxed);
        while (current < size &&
               !high_water_mark.compare_exchange_weak(current, size, std::memory_order_relaxed,
                                                      std::memory_order_relaxed))
        {
        }
    }

    [[nodiscard]] bool discard_oldest() noexcept
    {
        StreamMessage dropped_message;
        if (queue.try_pop(dropped_message) != StreamStatus::ok)
        {
            return false;
        }
        const auto sequence = dropped_message.kind() == StreamMessageKind::frame
                                  ? dropped_message.frame().header.sequence
                                  : dropped_message.discontinuity().actual_frame_sequence;
        record_drop(sequence);
        return true;
    }

    void prefault() noexcept
    {
        frame_pool.prefault();
        discontinuity_pool.prefault();
        queue.prefault();
        for (std::size_t i = 0; i < config.drop_history_capacity; ++i)
        {
            drop_history[i].range = {};
            drop_history[i].ready.store(0, std::memory_order_relaxed);
        }
    }

    NativeObserver& observer;
    ObserverEdgeConfig config;
    FramePool frame_pool;
    DiscontinuityPool discontinuity_pool;
    ObserverQueue<StreamMessage> queue;
    std::unique_ptr<DropSlot[]> drop_history;
    std::thread thread;
    std::atomic<std::uint64_t> epoch{};
    std::atomic<std::uint64_t> enqueued{};
    std::atomic<std::uint64_t> delivered{};
    std::atomic<std::uint64_t> discontinuities{};
    std::atomic<std::uint64_t> dropped{};
    std::atomic<std::size_t> drop_claims{};
    std::atomic<std::uint64_t> drop_history_dropped{};
    std::atomic<std::uint64_t> failures{};
    std::atomic<std::uint64_t> high_water_mark{};
    std::atomic<std::uint64_t> last_enqueued_sequence{};
    std::atomic<std::uint64_t> last_delivered_sequence{};
    std::atomic<bool> active{true};
    std::atomic<bool> done{};
};

struct NativeStreamRunner::CriticalObserverEdge
{
    CriticalObserverEdge(NativeCriticalObserver& endpoint, ObserverEdgeConfig edge_config,
                         bool observes_input)
        : observer(endpoint), config(edge_config), input_tap(observes_input)
    {
    }

    [[nodiscard]] ObserverEdgeStats snapshot() const noexcept
    {
        return ObserverEdgeStats{
            .id = config.id,
            .enqueued = enqueued.load(std::memory_order_relaxed),
            .delivered = delivered.load(std::memory_order_relaxed),
            .discontinuities = discontinuities.load(std::memory_order_relaxed),
            .dropped = dropped.load(std::memory_order_relaxed),
            .failures = failures.load(std::memory_order_relaxed),
            .last_enqueued_sequence = last_enqueued_sequence.load(std::memory_order_relaxed),
            .last_delivered_sequence = last_delivered_sequence.load(std::memory_order_relaxed),
            .detached = !active.load(std::memory_order_acquire),
        };
    }

    NativeCriticalObserver& observer;
    ObserverEdgeConfig config;
    bool input_tap{};
    std::atomic<std::uint64_t> enqueued{};
    std::atomic<std::uint64_t> delivered{};
    std::atomic<std::uint64_t> discontinuities{};
    std::atomic<std::uint64_t> dropped{};
    std::atomic<std::uint64_t> failures{};
    std::atomic<std::uint64_t> last_enqueued_sequence{};
    std::atomic<std::uint64_t> last_delivered_sequence{};
    std::atomic<bool> active{true};
    std::atomic<bool> fault_reported{};
    std::atomic<bool> fault_delivered{};
    std::atomic<bool> done{};
};

struct NativeStreamRunner::ThreadStartup
{
    std::array<RealtimeApplyResult, static_cast<std::size_t>(RealtimeThreadRole::count)> results{};
    std::atomic<std::uint32_t> ready{};
    std::atomic<bool> release{};
    std::atomic<bool> abort{};
    std::uint32_t expected{};
};

NativeStreamRunner::NativeStreamRunner(StreamSchema schema, RealtimeConfig config,
                                       NativeFrameSource& source, NativeFrameProcessor& processor,
                                       NativeFrameConsumer& consumer)
    : NativeStreamRunner(std::move(schema), config, source, processor, consumer,
                         default_native_clock(), default_safety_controller())
{
}

NativeStreamRunner::NativeStreamRunner(StreamSchema schema, RealtimeConfig config,
                                       NativeFrameSource& source, NativeFrameProcessor& processor,
                                       NativeFrameConsumer& consumer, NativeClock& clock,
                                       SafetyController& safety_controller)
    : NativeStreamRunner(std::move(schema), config, source, processor,
                         static_cast<NativeActuator&>(consumer), clock, safety_controller,
                         default_realtime_platform())
{
    terminal_consumer_ = &consumer;
    consumer.bind_runtime_clock(clock);
}

NativeStreamRunner::NativeStreamRunner(StreamSchema schema, RealtimeConfig config,
                                       NativeFrameSource& source, NativeFrameProcessor& processor,
                                       NativeActuator& actuator)
    : NativeStreamRunner(std::move(schema), config, source, processor, actuator,
                         default_native_clock(), default_safety_controller())
{
}

NativeStreamRunner::NativeStreamRunner(StreamSchema schema, RealtimeConfig config,
                                       NativeFrameSource& source, NativeFrameProcessor& processor,
                                       NativeActuator& actuator, NativeClock& clock,
                                       SafetyController& safety_controller)
    : NativeStreamRunner(std::move(schema), config, source, processor, actuator, clock,
                         safety_controller, default_realtime_platform())
{
}

NativeStreamRunner::NativeStreamRunner(StreamSchema schema, RealtimeConfig config,
                                       NativeFrameSource& source, NativeFrameProcessor& processor,
                                       NativeActuator& actuator, NativeClock& clock,
                                       SafetyController& safety_controller,
                                       RealtimePlatform& realtime_platform)
    : schema_(std::move(schema)), config_(config), source_(source), processor_(processor),
      actuator_(actuator), clock_(clock), safety_controller_(safety_controller),
      realtime_platform_(realtime_platform)
{
}

NativeStreamRunner::~NativeStreamRunner() noexcept
{
    if (!safety_inhibited_.load(std::memory_order_acquire))
    {
        static_cast<void>(inhibit_safety(SafetyReason::explicit_stop));
    }
    if (acquisition_thread_.joinable() || processing_thread_.joinable() ||
        actuator_thread_.joinable() || observer_dispatch_thread_.joinable() ||
        watchdog_thread_.joinable())
    {
        static_cast<void>(abort());
        static_cast<void>(join_threads());
    }
    else if (state() == RuntimeState::prepared)
    {
        static_cast<void>(abort());
    }
}

StreamStatus NativeStreamRunner::add_observer(NativeObserver& observer, ObserverEdgeConfig config)
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() != RuntimeState::created)
    {
        return StreamStatus::invalid_state;
    }
    if (config.id == 0 || config.capacity == 0 || config.drop_history_capacity == 0)
    {
        throw std::invalid_argument(
            "observer id, capacity, and drop history capacity must be positive");
    }
    if (config.critical_recorder)
    {
        throw std::invalid_argument(
            "critical observers must be registered with add_critical_observer");
    }
    if (config.drop_policy == ObserverDropPolicy::fault)
    {
        throw std::invalid_argument("fault drop policy is reserved for critical observers");
    }
    const auto duplicate =
        std::find_if(observer_registrations_.begin(), observer_registrations_.end(),
                     [config](const ObserverRegistration& registration)
                     { return registration.config.id == config.id; });
    if (duplicate != observer_registrations_.end())
    {
        throw std::invalid_argument("observer id must be unique");
    }
    observer_registrations_.push_back({&observer, nullptr, config, false});
    return StreamStatus::ok;
}

StreamStatus NativeStreamRunner::add_critical_observer(NativeCriticalObserver& observer,
                                                       ObserverEdgeConfig config)
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() != RuntimeState::created)
    {
        return StreamStatus::invalid_state;
    }
    if (config.id == 0 || config.capacity == 0 || config.drop_history_capacity == 0)
    {
        throw std::invalid_argument(
            "observer id, capacity, and drop history capacity must be positive");
    }
    if (!config.critical_recorder || config.drop_policy != ObserverDropPolicy::fault)
    {
        throw std::invalid_argument(
            "critical observers require critical_recorder=true and drop_policy=fault");
    }
    const auto duplicate =
        std::find_if(observer_registrations_.begin(), observer_registrations_.end(),
                     [config](const ObserverRegistration& registration)
                     { return registration.config.id == config.id; });
    if (duplicate != observer_registrations_.end())
    {
        throw std::invalid_argument("observer id must be unique");
    }
    observer_registrations_.push_back({nullptr, &observer, config, false});
    return StreamStatus::ok;
}

StreamStatus NativeStreamRunner::add_input_critical_observer(NativeCriticalObserver& observer,
                                                             ObserverEdgeConfig config)
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() != RuntimeState::created)
    {
        return StreamStatus::invalid_state;
    }
    if (config.id == 0 || config.capacity == 0 || config.drop_history_capacity == 0)
    {
        throw std::invalid_argument(
            "observer id, capacity, and drop history capacity must be positive");
    }
    if (!config.critical_recorder || config.drop_policy != ObserverDropPolicy::fault)
    {
        throw std::invalid_argument(
            "input critical observers require critical_recorder=true and drop_policy=fault");
    }
    const auto duplicate =
        std::find_if(observer_registrations_.begin(), observer_registrations_.end(),
                     [config](const ObserverRegistration& registration)
                     { return registration.config.id == config.id; });
    if (duplicate != observer_registrations_.end())
    {
        throw std::invalid_argument("observer id must be unique");
    }
    observer_registrations_.push_back({nullptr, &observer, config, true});
    return StreamStatus::ok;
}

StreamStatus NativeStreamRunner::prepare()
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() != RuntimeState::created)
    {
        return StreamStatus::invalid_state;
    }
    const auto automatic_resources = config_.automatic_resources;
    if (automatic_resources)
    {
        config_.max_process_outputs = std::numeric_limits<std::size_t>::max();
        config_.max_flush_outputs = std::numeric_limits<std::size_t>::max();
        config_.pool_capacity.processor_owned = std::numeric_limits<std::size_t>::max();
    }
    else
    {
        validate_runtime_config(schema_, config_);
    }

    // A generic source holds one discontinuity slot for the duration of every
    // read, whether or not that read produces a discontinuity, and the
    // continuity checker needs one of its own for the break it may infer. One
    // slot between them would deadlock the two, so the requirement is stated
    // here rather than discovered as a pool exhaustion under load.
    if (!automatic_resources && source_.produces_discontinuities() &&
        config_.discontinuity_capacity < 2)
    {
        throw std::invalid_argument(
            "a source that produces discontinuities needs discontinuity_capacity >= 2");
    }

#if defined(NEURALE_STREAMING_TEST_HOOKS)
    auto faults = std::make_unique<FaultChannel>(config_.fault_history_capacity, test_hooks_);
#else
    auto faults = std::make_unique<FaultChannel>(config_.fault_history_capacity);
#endif
    faults_ = std::move(faults);

    std::optional<PreparedProcessorContract> declared_contract;
    try
    {
        declared_contract.emplace(processor_.prepare(ProcessorPrepareContext{
            .input_schema = schema_,
            .max_process_outputs = config_.max_process_outputs,
            .max_flush_outputs = config_.max_flush_outputs,
            .available_frame_pool_leases = config_.pool_capacity.processor_owned,
        }));
    }
    catch (...)
    {
        record_fault(FaultCode::processor_prepare, StreamStatus::processor_failure,
                     FaultStage::processor,
                     static_cast<std::uint32_t>(ProcessorContractDetail::prepare_failed));
        state_.store(RuntimeState::failed, std::memory_order_release);
        return StreamStatus::processor_failure;
    }

    auto reject_contract = [this](StreamStatus status, ProcessorContractDetail detail)
    {
        record_fault(FaultCode::processor_prepare, status, FaultStage::processor,
                     static_cast<std::uint32_t>(detail));
        state_.store(RuntimeState::failed, std::memory_order_release);
        return status;
    };
    if (!equal_stream_schema(declared_contract->accepted_input_schema, schema_))
    {
        return reject_contract(StreamStatus::invalid_frame,
                               ProcessorContractDetail::input_schema_mismatch);
    }
    if (automatic_resources)
    {
        resolve_automatic_resources(config_, schema_, *declared_contract,
                                    source_.produces_discontinuities(),
                                    !observer_registrations_.empty());
        validate_runtime_config(schema_, config_);
    }
    try
    {
        validate_runtime_config(declared_contract->output_schema, config_);
    }
    catch (...)
    {
        return reject_contract(StreamStatus::invalid_frame,
                               ProcessorContractDetail::output_schema_exceeds_storage);
    }
    if (declared_contract->max_process_outputs_per_input > config_.max_process_outputs)
    {
        return reject_contract(StreamStatus::output_limit,
                               ProcessorContractDetail::process_output_limit);
    }
    if (declared_contract->max_flush_outputs > config_.max_flush_outputs)
    {
        return reject_contract(StreamStatus::output_limit,
                               ProcessorContractDetail::flush_output_limit);
    }
    if (declared_contract->required_resources.frame_pool_leases == 0 ||
        declared_contract->required_resources.frame_pool_leases >
            config_.pool_capacity.processor_owned)
    {
        return reject_contract(StreamStatus::buffer_exhausted,
                               ProcessorContractDetail::frame_pool_bound);
    }
    if (processor_contract_.has_value() &&
        !equal_processor_contract(*processor_contract_, *declared_contract))
    {
        return reject_contract(StreamStatus::invalid_state,
                               ProcessorContractDetail::changed_after_reset);
    }
    if (!processor_contract_.has_value())
    {
        processor_contract_.emplace(std::move(*declared_contract));
    }

    auto frame_pool = std::make_unique<FramePool>(config_.required_buffer_count(),
                                                  config_.buffer_size, config_.max_signal_blocks);
    auto discontinuity_pool = std::make_unique<DiscontinuityPool>(config_.discontinuity_capacity,
                                                                  config_.gaps_per_discontinuity);
    auto continuity = std::make_unique<ContinuityChecker>(schema_);
    auto validator = std::make_unique<FrameValidator>(processor_contract_->output_schema);
    auto ingress =
        std::make_unique<SpscRing<StreamMessage>>(config_.pool_capacity.ingress_capacity);
    auto critical_edge =
        std::make_unique<SpscRing<StreamMessage>>(config_.pool_capacity.critical_edge_capacity);
    std::unique_ptr<SpscRing<StreamMessage>> observer_dispatch;
    std::vector<std::unique_ptr<ObserverEdge>> observer_edges;
    std::vector<std::unique_ptr<CriticalObserverEdge>> critical_observer_edges;
    const auto needs_output_dispatch =
        std::any_of(observer_registrations_.begin(), observer_registrations_.end(),
                    [](const ObserverRegistration& registration)
                    { return registration.observer != nullptr || !registration.input_tap; });
    if (!observer_registrations_.empty())
    {
        if (needs_output_dispatch && config_.pool_capacity.observer_edge_capacity == 0)
        {
            throw std::invalid_argument("observer dispatch capacity must be positive "
                                        "when observers are registered");
        }
        if (needs_output_dispatch)
        {
            observer_dispatch = std::make_unique<SpscRing<StreamMessage>>(
                config_.pool_capacity.observer_edge_capacity);
        }
        observer_edges.reserve(observer_registrations_.size());
        critical_observer_edges.reserve(observer_registrations_.size());
        for (const auto& registration : observer_registrations_)
        {
            if (registration.critical_observer != nullptr)
            {
                critical_observer_edges.push_back(std::make_unique<CriticalObserverEdge>(
                    *registration.critical_observer, registration.config, registration.input_tap));
            }
            else
            {
                observer_edges.push_back(std::make_unique<ObserverEdge>(
                    *registration.observer, registration.config, config_));
            }
        }
    }
    auto ingress_timestamps =
        std::make_unique<std::atomic<HostTimeNs>[]>(config_.pool_capacity.ingress_capacity);
    auto terminal_emitter = std::unique_ptr<TerminalFrameEmitter>(new TerminalFrameEmitter(
        *frame_pool, *validator, *critical_edge, counters_, clock_, critical_epoch_));
    auto thread_startup = std::make_unique<ThreadStartup>();

    frame_pool_ = std::move(frame_pool);
    discontinuity_pool_ = std::move(discontinuity_pool);
    continuity_ = std::move(continuity);
    output_validator_ = std::move(validator);
    ingress_ = std::move(ingress);
    critical_edge_ = std::move(critical_edge);
    observer_dispatch_ = std::move(observer_dispatch);
    ingress_timestamps_ = std::move(ingress_timestamps);
    terminal_emitter_ = std::move(terminal_emitter);
    thread_startup_ = std::move(thread_startup);
    observer_edges_ = std::move(observer_edges);
    critical_observer_edges_ = std::move(critical_observer_edges);
    if (config_.platform.mode != RealtimeConfigMode::disabled && config_.platform.prefault_pools)
    {
        frame_pool_->prefault();
        discontinuity_pool_->prefault();
        ingress_->prefault();
        critical_edge_->prefault();
        if (observer_dispatch_ != nullptr)
        {
            observer_dispatch_->prefault();
        }
        for (auto& edge : observer_edges_)
        {
            edge->prefault();
        }
        for (std::size_t i = 0; i < config_.pool_capacity.ingress_capacity; ++i)
        {
            ingress_timestamps_[i].store(0, std::memory_order_relaxed);
        }
    }
    counters_.reset();
    stop_requested_.store(false, std::memory_order_relaxed);
    abort_requested_.store(false, std::memory_order_relaxed);
    producer_done_.store(false, std::memory_order_relaxed);
    acquisition_done_.store(false, std::memory_order_relaxed);
    processing_done_.store(false, std::memory_order_relaxed);
    actuator_done_.store(false, std::memory_order_relaxed);
    observer_dispatch_done_.store(observer_dispatch_ == nullptr, std::memory_order_relaxed);
    watchdog_stop_.store(false, std::memory_order_relaxed);
    shutdown_claimed_.store(false, std::memory_order_relaxed);
    shutdown_active_.store(false, std::memory_order_relaxed);
    shutdown_timeout_recorded_.store(false, std::memory_order_relaxed);
    processor_active_.store(false, std::memory_order_relaxed);
    actuator_active_.store(false, std::memory_order_relaxed);
    output_valid_.store(false, std::memory_order_relaxed);
    safety_inhibited_.store(true, std::memory_order_relaxed);
    end_reason_.store(EndReason::none, std::memory_order_relaxed);
    ingress_epoch_.store(0, std::memory_order_relaxed);
    critical_epoch_.store(0, std::memory_order_relaxed);
    observer_dispatch_epoch_.store(0, std::memory_order_relaxed);
    next_command_id_.store(1, std::memory_order_relaxed);
    next_data_message_ordinal_.store(0, std::memory_order_relaxed);
    next_input_data_message_ordinal_.store(0, std::memory_order_relaxed);
    runtime_started_ns_.store(0, std::memory_order_relaxed);
    shutdown_started_ns_.store(0, std::memory_order_relaxed);
    last_source_heartbeat_ns_.store(0, std::memory_order_relaxed);
    current_processor_start_ns_.store(0, std::memory_order_relaxed);
    current_processor_session_id_.store(0, std::memory_order_relaxed);
    current_processor_frame_sequence_.store(0, std::memory_order_relaxed);
    current_processor_schema_id_.store(0, std::memory_order_relaxed);
    current_processor_clock_domain_.store(0, std::memory_order_relaxed);
    current_actuator_start_ns_.store(0, std::memory_order_relaxed);
    last_processor_completion_ns_.store(0, std::memory_order_relaxed);
    last_valid_output_ns_.store(0, std::memory_order_relaxed);
    current_frame_ = {};
    realtime_status_ = {};
    processor_prepared_ = true;
    runtime_generation_.fetch_add(1, std::memory_order_relaxed);

    const auto now = clock_.now_ns();
    const auto safety_status = safety_controller_.inhibit(SafetyReason::startup);
    counters_.safety_inhibition();
    last_inhibit_ns_.store(now, std::memory_order_release);
    if (safety_status != StreamStatus::ok)
    {
        record_fault(FaultCode::safety_controller_failure, safety_status, FaultStage::runtime);
        state_.store(RuntimeState::failed, std::memory_order_release);
        return safety_status;
    }

    state_.store(RuntimeState::prepared, std::memory_order_release);
    return StreamStatus::ok;
}

StreamStatus NativeStreamRunner::arm() noexcept
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() != RuntimeState::prepared || !safety_inhibited_.load(std::memory_order_acquire))
    {
        return StreamStatus::invalid_state;
    }
    for (auto& edge : critical_observer_edges_)
    {
        const auto readiness = edge->observer.ready_for_runtime();
        if (readiness != StreamStatus::ok || edge->observer.health() != StreamStatus::ok)
        {
            const auto status = readiness != StreamStatus::ok ? readiness : edge->observer.health();
            edge->failures.fetch_add(1, std::memory_order_relaxed);
            edge->fault_reported.store(true, std::memory_order_release);
            record_fault(FaultCode::critical_observer_failure, status, FaultStage::observer,
                         edge->config.id);
            cancel_critical_observers();
            state_.store(RuntimeState::failed, std::memory_order_release);
            return status;
        }
    }
    const auto status = safety_controller_.release();
    if (status != StreamStatus::ok)
    {
        record_fault(FaultCode::safety_controller_failure, status, FaultStage::runtime);
        cancel_critical_observers();
        state_.store(RuntimeState::failed, std::memory_order_release);
        return status;
    }
    safety_inhibited_.store(false, std::memory_order_release);
    return StreamStatus::ok;
}

void NativeStreamRunner::configured_loop(RealtimeThreadRole role,
                                         void (NativeStreamRunner::*loop)() noexcept) noexcept
{
    const auto idx = static_cast<std::size_t>(role);
    if (config_.platform.mode != RealtimeConfigMode::disabled)
    {
        thread_startup_->results[idx] =
            realtime_platform_.configure_current_thread(config_.platform.thread(role));
    }
    else
    {
        thread_startup_->results[idx] = {};
    }
    thread_startup_->ready.fetch_add(1, std::memory_order_release);
    thread_startup_->ready.notify_one();

    thread_startup_->release.wait(false, std::memory_order_acquire);
    if (!thread_startup_->abort.load(std::memory_order_acquire))
    {
        (this->*loop)();
    }
}

void NativeStreamRunner::observer_startup_loop(ObserverEdge& edge) noexcept
{
    thread_startup_->ready.fetch_add(1, std::memory_order_release);
    thread_startup_->ready.notify_one();

    thread_startup_->release.wait(false, std::memory_order_acquire);
    if (!thread_startup_->abort.load(std::memory_order_acquire))
    {
        observer_loop(edge);
    }
}

StreamStatus NativeStreamRunner::start()
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() != RuntimeState::prepared || safety_inhibited_.load(std::memory_order_acquire))
    {
        return StreamStatus::invalid_state;
    }

    realtime_status_ = {};
    if (config_.platform.mode != RealtimeConfigMode::disabled)
    {
        realtime_status_.memory = realtime_platform_.configure_process_memory(config_.platform);
        if (config_.platform.mode == RealtimeConfigMode::strict && !realtime_status_.memory.ok())
        {
            record_fault(FaultCode::realtime_configuration,
                         StreamStatus::realtime_configuration_failed, FaultStage::runtime,
                         realtime_status_.memory.unsupported | realtime_status_.memory.failed);
            static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
            cancel_critical_observers();
            state_.store(RuntimeState::failed, std::memory_order_release);
            return StreamStatus::realtime_configuration_failed;
        }
    }

    auto strict_failure = false;
    auto observer_start_failure = StreamStatus::ok;
    auto create_thread = [this](auto&&... arguments)
    {
#if defined(NEURALE_STREAMING_TEST_HOOKS)
        if (test_hooks_ != nullptr && test_hooks_->thread_starts_before_failure != nullptr)
        {
            auto& remaining = *test_hooks_->thread_starts_before_failure;
            const auto value = remaining.load(std::memory_order_acquire);
            if (value != std::numeric_limits<std::size_t>::max())
            {
                if (value == 0)
                {
                    throw std::system_error(
                        std::make_error_code(std::errc::resource_unavailable_try_again));
                }
                remaining.store(value - 1, std::memory_order_release);
            }
        }
#endif
        return std::thread(std::forward<decltype(arguments)>(arguments)...);
    };
    try
    {
        thread_startup_->ready.store(0, std::memory_order_relaxed);
        thread_startup_->release.store(false, std::memory_order_relaxed);
        thread_startup_->abort.store(false, std::memory_order_relaxed);
        thread_startup_->expected = 4U + static_cast<std::uint32_t>(observer_edges_.size()) +
                                    (observer_dispatch_ == nullptr ? 0U : 1U);

        watchdog_thread_ =
            create_thread(&NativeStreamRunner::configured_loop, this, RealtimeThreadRole::watchdog,
                          &NativeStreamRunner::watchdog_loop);
        for (auto& edge : observer_edges_)
        {
            edge->thread =
                create_thread(&NativeStreamRunner::observer_startup_loop, this, std::ref(*edge));
        }
        if (observer_dispatch_ != nullptr)
        {
            observer_dispatch_thread_ = create_thread(&NativeStreamRunner::configured_loop, this,
                                                      RealtimeThreadRole::observer_dispatch,
                                                      &NativeStreamRunner::observer_dispatch_loop);
        }
        actuator_thread_ =
            create_thread(&NativeStreamRunner::configured_loop, this, RealtimeThreadRole::actuator,
                          &NativeStreamRunner::actuator_loop);
        processing_thread_ =
            create_thread(&NativeStreamRunner::configured_loop, this,
                          RealtimeThreadRole::processing, &NativeStreamRunner::processing_loop);
        acquisition_thread_ =
            create_thread(&NativeStreamRunner::configured_loop, this,
                          RealtimeThreadRole::acquisition, &NativeStreamRunner::acquisition_loop);

        auto ready = thread_startup_->ready.load(std::memory_order_acquire);
        while (ready < thread_startup_->expected)
        {
            thread_startup_->ready.wait(ready, std::memory_order_acquire);
            ready = thread_startup_->ready.load(std::memory_order_acquire);
        }
        if (config_.platform.mode != RealtimeConfigMode::disabled)
        {
            realtime_status_.threads = thread_startup_->results;
            strict_failure =
                config_.platform.mode == RealtimeConfigMode::strict && !realtime_status_.ok();
            if (strict_failure)
            {
                record_fault(FaultCode::realtime_configuration,
                             StreamStatus::realtime_configuration_failed, FaultStage::runtime,
                             realtime_status_.warning_count());
                throw RealtimeStartupAbort{};
            }
        }

        for (auto& edge : critical_observer_edges_)
        {
            const auto recorder_status = edge->observer.start_observing();
            if (recorder_status != StreamStatus::ok)
            {
                edge->failures.fetch_add(1, std::memory_order_relaxed);
                edge->fault_reported.store(true, std::memory_order_release);
                record_fault(FaultCode::critical_observer_failure, recorder_status,
                             FaultStage::observer, edge->config.id);
                static_cast<void>(inhibit_safety(SafetyReason::critical_observer_failure));
                observer_start_failure = recorder_status;
                throw RealtimeStartupAbort{};
            }
        }

        const auto now = clock_.now_ns();
        runtime_started_ns_.store(now, std::memory_order_relaxed);
        last_source_heartbeat_ns_.store(now, std::memory_order_relaxed);
        last_processor_completion_ns_.store(now, std::memory_order_relaxed);
        state_.store(RuntimeState::running, std::memory_order_release);
        thread_startup_->release.store(true, std::memory_order_release);
        thread_startup_->release.notify_all();
    }
    catch (...)
    {
        thread_startup_->abort.store(true, std::memory_order_release);
        thread_startup_->release.store(true, std::memory_order_release);
        thread_startup_->release.notify_all();
        if (!strict_failure && observer_start_failure == StreamStatus::ok)
        {
            record_fault(FaultCode::runtime_start, StreamStatus::invalid_state,
                         FaultStage::runtime);
        }
        static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
        request_abort(true);
        if (!acquisition_thread_.joinable())
        {
            acquisition_done_.store(true, std::memory_order_release);
            finish_producer(EndReason::fault);
        }
        if (acquisition_thread_.joinable())
        {
            acquisition_thread_.join();
        }
        if (processing_thread_.joinable())
        {
            processing_thread_.join();
        }
        if (critical_edge_ != nullptr)
        {
            critical_edge_->close();
            notify_critical();
        }
        actuator_.cancel();
        if (actuator_thread_.joinable())
        {
            actuator_thread_.join();
        }
        if (observer_dispatch_ != nullptr)
        {
            observer_dispatch_->close();
            notify_observer_dispatch();
        }
        if (observer_dispatch_thread_.joinable())
        {
            observer_dispatch_thread_.join();
        }
        for (auto& edge : observer_edges_)
        {
            edge->active.store(false, std::memory_order_release);
            edge->queue.close();
            edge->observer.cancel();
            edge->epoch.fetch_add(1, std::memory_order_release);
            edge->epoch.notify_all();
            if (edge->thread.joinable())
            {
                edge->thread.join();
            }
        }
        cancel_critical_observers();
        watchdog_stop_.store(true, std::memory_order_release);
        clock_.wake();
        if (watchdog_thread_.joinable())
        {
            watchdog_thread_.join();
        }
        acquisition_done_.store(true, std::memory_order_release);
        processing_done_.store(true, std::memory_order_release);
        actuator_done_.store(true, std::memory_order_release);
        observer_dispatch_done_.store(true, std::memory_order_release);
        state_.store(RuntimeState::failed, std::memory_order_release);
        if (strict_failure)
        {
            return StreamStatus::realtime_configuration_failed;
        }
        if (observer_start_failure != StreamStatus::ok)
        {
            return observer_start_failure;
        }
        throw;
    }
    return StreamStatus::ok;
}

void NativeStreamRunner::record_fault(FaultCode code, StreamStatus status, FaultStage stage,
                                      std::uint32_t detail, const FrameHeader* frame,
                                      HostTimeNs detected_at_ns) noexcept
{
    const FrameHeader context = frame == nullptr ? FrameHeader{} : *frame;
    const FaultRecord fault{
        .code = code,
        .status = status,
        .stage = stage,
        .component_id = 0,
        .detail = detail,
        .session_id = context.session_id,
        .runtime_generation = runtime_generation_.load(std::memory_order_relaxed),
        .frame_sequence = context.sequence,
        .schema_id = context.schema_id,
        .clock_domain = context.source_clock_domain,
        .detected_at_ns = detected_at_ns == 0 ? clock_.now_ns() : detected_at_ns,
    };
    counters_.fault();
    const auto result = faults_->record(fault);
    if (result != FaultChannel::Result::primary)
    {
        counters_.secondary_fault();
    }
    if (result == FaultChannel::Result::dropped)
    {
        counters_.fault_history_dropped();
    }
}

StreamStatus NativeStreamRunner::inhibit_safety(SafetyReason reason) noexcept
{
    if (safety_inhibited_.exchange(true, std::memory_order_acq_rel))
    {
        return StreamStatus::ok;
    }
    last_inhibit_ns_.store(clock_.now_ns(), std::memory_order_release);
    counters_.safety_inhibition();
    const auto status = safety_controller_.inhibit(reason);
    if (status != StreamStatus::ok)
    {
        record_fault(FaultCode::safety_controller_failure, status, FaultStage::runtime);
    }
    return status;
}

void NativeStreamRunner::trigger_deadline_fault(FaultCode code, FaultStage stage,
                                                SafetyReason reason, HostTimeNs detected_at_ns,
                                                std::uint32_t detail) noexcept
{
    counters_.deadline_fault();
    if (code == FaultCode::actuator_deadline)
    {
        counters_.actuator_deadline_miss();
        counters_.actuator_failure();
    }
    FrameHeader context{};
    const FrameHeader* context_pointer = nullptr;
    if (stage == FaultStage::processor)
    {
        context.session_id = current_processor_session_id_.load(std::memory_order_acquire);
        context.sequence = current_processor_frame_sequence_.load(std::memory_order_acquire);
        context.schema_id = current_processor_schema_id_.load(std::memory_order_acquire);
        context.source_clock_domain =
            current_processor_clock_domain_.load(std::memory_order_acquire);
        context_pointer = context.session_id == 0 ? nullptr : &context;
    }
    record_fault(code, StreamStatus::deadline_exceeded, stage, detail, context_pointer,
                 detected_at_ns);
    static_cast<void>(inhibit_safety(reason));
    request_abort(true);
}

void NativeStreamRunner::notify_processing() noexcept
{
    ingress_epoch_.fetch_add(1, std::memory_order_release);
    ingress_epoch_.notify_one();
}

void NativeStreamRunner::notify_critical() noexcept
{
    critical_epoch_.fetch_add(1, std::memory_order_release);
    critical_epoch_.notify_one();
}

void NativeStreamRunner::notify_observer_dispatch() noexcept
{
    observer_dispatch_epoch_.fetch_add(1, std::memory_order_release);
    observer_dispatch_epoch_.notify_one();
}

void NativeStreamRunner::begin_shutdown() noexcept
{
    auto unclaimed = false;
    if (!shutdown_claimed_.compare_exchange_strong(unclaimed, true, std::memory_order_acq_rel,
                                                   std::memory_order_relaxed))
    {
        return;
    }
    // The stamp is published before the flag readers gate on, not after.
    // wait_for_shutdown() and the watchdog both measure the shutdown deadline
    // from shutdown_started_ns_, and electing on shutdown_active_ itself let a
    // reader observe the flag while the stamp still held the reset value 0.
    // Against a monotonic host clock that expires instantly and reports a
    // spurious shutdown_timeout fault on an otherwise healthy stop. The relaxed
    // stamp store is sequenced before the releasing flag store, so every
    // acquire load of shutdown_active_ sees the stamp.
    shutdown_started_ns_.store(clock_.now_ns(), std::memory_order_relaxed);
    shutdown_active_.store(true, std::memory_order_release);
}

void NativeStreamRunner::finish_producer(EndReason reason) noexcept
{
    auto expected = EndReason::none;
    static_cast<void>(end_reason_.compare_exchange_strong(
        expected, reason, std::memory_order_release, std::memory_order_relaxed));
    ingress_->close();
    producer_done_.store(true, std::memory_order_release);
    begin_shutdown();
    notify_processing();
}

void NativeStreamRunner::request_abort(bool fault) noexcept
{
    if (fault)
    {
        end_reason_.store(EndReason::fault, std::memory_order_release);
    }
    else
    {
        auto expected = EndReason::none;
        static_cast<void>(end_reason_.compare_exchange_strong(
            expected, EndReason::abort, std::memory_order_release, std::memory_order_relaxed));
    }
    begin_shutdown();
    state_.store(RuntimeState::stopping, std::memory_order_release);
    abort_requested_.store(true, std::memory_order_release);
    source_.cancel();
    actuator_.cancel();
    for (auto& edge : observer_edges_)
    {
        edge->active.store(false, std::memory_order_release);
        edge->queue.close();
        edge->observer.cancel();
        edge->epoch.fetch_add(1, std::memory_order_release);
        edge->epoch.notify_all();
    }
    notify_processing();
    notify_critical();
    notify_observer_dispatch();
    clock_.wake();
}

void NativeStreamRunner::acquisition_loop() noexcept
{
    // Asked once: the answer is a property of the source type, and a per-read
    // virtual call to learn it would be pure cost on the acquisition thread.
    const bool generic_source = source_.produces_discontinuities();
    EndReason reason = EndReason::stop;
    FrameHeader last_source_context{};
    while (!stop_requested_.load(std::memory_order_acquire) &&
           !abort_requested_.load(std::memory_order_acquire))
    {
        FrameLease frame;
        const auto acquire_status = frame_pool_->try_acquire(frame);
        if (acquire_status != StreamStatus::ok)
        {
            counters_.pool_exhaustion();
            record_fault(FaultCode::frame_pool_exhausted, acquire_status, FaultStage::source);
            static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
            request_abort(true);
            reason = EndReason::fault;
            break;
        }

        // A generic source may answer with a discontinuity instead of a frame,
        // so it is handed a slot for each. The lease is acquired per read and
        // released with the loop iteration; a frame-only source is never asked
        // for one and keeps the pool it always had.
        DiscontinuityLease discontinuity;
        if (generic_source)
        {
            const auto lease_status = discontinuity_pool_->try_acquire(discontinuity);
            if (lease_status != StreamStatus::ok)
            {
                static_cast<void>(frame.reset());
                record_fault(FaultCode::frame_pool_exhausted, lease_status, FaultStage::source);
                static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
                request_abort(true);
                reason = EndReason::fault;
                break;
            }
        }

        const auto source_status = generic_source
                                       ? source_.read_message(frame.frame(), discontinuity)
                                       : source_.read(frame.frame());
        const auto now = clock_.now_ns();
        last_source_heartbeat_ns_.store(now, std::memory_order_release);
        const auto source_context = frame.frame().header();
        if (source_status == StreamStatus::would_block)
        {
            static_cast<void>(frame.reset());
            continue;
        }
        if (source_status == StreamStatus::discontinuity && generic_source)
        {
            static_cast<void>(frame.reset());
            if (stop_requested_.load(std::memory_order_acquire) ||
                abort_requested_.load(std::memory_order_acquire))
            {
                reason = end_reason_.load(std::memory_order_acquire);
                break;
            }
            const auto sequence = ingress_->producer_sequence();
            ingress_timestamps_[sequence % ingress_->capacity()].store(now,
                                                                       std::memory_order_relaxed);
            auto message = StreamMessage::from_discontinuity(std::move(discontinuity), now);
            const auto push_status = ingress_->try_push(std::move(message));
            if (push_status != StreamStatus::ok)
            {
                if (push_status == StreamStatus::queue_overflow)
                {
                    counters_.queue_overrun();
                    record_fault(FaultCode::queue_overrun, push_status, FaultStage::runtime);
                }
                static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
                request_abort(true);
                reason = EndReason::fault;
                break;
            }
            counters_.observe_ingress_size(ingress_->approximate_size());
            notify_processing();
            continue;
        }
        if (source_status == StreamStatus::end_of_stream)
        {
            static_cast<void>(frame.reset());
            counters_.end_of_stream();
            reason = EndReason::end_of_stream;
            break;
        }
        if (source_status != StreamStatus::ok)
        {
            static_cast<void>(frame.reset());
            if ((stop_requested_.load(std::memory_order_acquire) ||
                 abort_requested_.load(std::memory_order_acquire)) &&
                source_status == StreamStatus::stopped)
            {
                reason = end_reason_.load(std::memory_order_acquire);
                break;
            }
            const auto& fault_context =
                source_context.session_id != 0 ? source_context : last_source_context;
            record_fault(FaultCode::source_read, source_status, FaultStage::source, 0,
                         &fault_context);
            static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
            request_abort(true);
            reason = EndReason::fault;
            break;
        }

        counters_.acquired();
        frame.frame().header().host_received_ns = now;
        frame.frame().header().flags = frame.frame().header().flags | FrameFlags::source_received;
        const auto context = frame.frame().header();
        last_source_context = context;
        if (stop_requested_.load(std::memory_order_acquire) ||
            abort_requested_.load(std::memory_order_acquire))
        {
            counters_.aborted_frame();
            static_cast<void>(frame.reset());
            reason = end_reason_.load(std::memory_order_acquire);
            break;
        }

        const auto sequence = ingress_->producer_sequence();
        ingress_timestamps_[sequence % ingress_->capacity()].store(now, std::memory_order_relaxed);
        auto message = StreamMessage::from_frame(std::move(frame), now);
        const auto push_status = ingress_->try_push(std::move(message));
        if (push_status != StreamStatus::ok)
        {
            counters_.aborted_frame();
            if (push_status == StreamStatus::queue_overflow)
            {
                counters_.queue_overrun();
                record_fault(FaultCode::queue_overrun, push_status, FaultStage::runtime, 0,
                             &context);
            }
            static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
            request_abort(true);
            reason = EndReason::fault;
            break;
        }
        counters_.observe_ingress_size(ingress_->approximate_size());
        notify_processing();
    }

    if (abort_requested_.load(std::memory_order_acquire))
    {
        reason = end_reason_.load(std::memory_order_acquire);
    }
    finish_producer(reason);
    acquisition_done_.store(true, std::memory_order_release);
    clock_.wake();
}

StreamStatus NativeStreamRunner::handle_discontinuity(DiscontinuityLease discontinuity) noexcept
{
    counters_.discontinuity();
    const auto view = discontinuity.view();
    const FrameHeader discontinuity_context{.session_id = view.session_id,
                                            .sequence = view.actual_frame_sequence,
                                            .schema_id = schema_.id()};
    current_processor_session_id_.store(discontinuity_context.session_id,
                                        std::memory_order_relaxed);
    current_processor_frame_sequence_.store(discontinuity_context.sequence,
                                            std::memory_order_relaxed);
    current_processor_schema_id_.store(discontinuity_context.schema_id, std::memory_order_relaxed);
    current_processor_clock_domain_.store(0, std::memory_order_relaxed);
    const auto input_observer_status = dispatch_to_input_critical_observers(view);
    if (input_observer_status != StreamStatus::ok)
    {
        return input_observer_status;
    }
    const auto started = clock_.now_ns();
    current_processor_start_ns_.store(started, std::memory_order_release);
    processor_active_.store(true, std::memory_order_release);
    const auto processor_status = processor_.handle_discontinuity(view);
    const auto completed = clock_.now_ns();
    const auto execution_ns = completed >= started ? completed - started : 0;
    counters_.observe_processor_execution(execution_ns);
    processor_active_.store(false, std::memory_order_release);
    last_processor_completion_ns_.store(completed, std::memory_order_release);
    if (processor_status != StreamStatus::ok)
    {
        record_fault(FaultCode::processor_discontinuity, processor_status, FaultStage::processor, 0,
                     &discontinuity_context);
    }
    if (processor_status != StreamStatus::ok ||
        (terminal_consumer_ == nullptr && observer_dispatch_ == nullptr))
    {
        return processor_status;
    }
    auto message = StreamMessage::from_discontinuity(std::move(discontinuity));
    const auto status = critical_edge_->try_push(std::move(message));
    if (status == StreamStatus::ok)
    {
        notify_critical();
        return StreamStatus::ok;
    }
    record_fault(FaultCode::actuator_queue_overrun, status, FaultStage::actuator, 0,
                 &discontinuity_context);
    counters_.actuator_failure();
    return status;
}

StreamStatus NativeStreamRunner::finish_output(StreamStatus processor_status,
                                               FaultCode processor_code) noexcept
{
    const auto output_status = terminal_emitter_->finish();
    if (output_status != StreamStatus::ok)
    {
        const auto stage = terminal_emitter_->failure_code_ == FaultCode::actuator_queue_overrun
                               ? FaultStage::actuator
                               : FaultStage::output;
        record_fault(terminal_emitter_->failure_code_, output_status, stage, 0, &current_frame_);
        if (terminal_emitter_->failure_code_ == FaultCode::actuator_queue_overrun)
        {
            counters_.actuator_failure();
        }
    }
    if (processor_status != StreamStatus::ok && processor_status != output_status)
    {
        record_fault(processor_code, processor_status, FaultStage::processor, 0, &current_frame_);
    }
    return output_status != StreamStatus::ok ? output_status : processor_status;
}

StreamStatus NativeStreamRunner::process_frame(FrameLease frame) noexcept
{
    current_frame_ = frame.frame().header();
    current_processor_session_id_.store(current_frame_.session_id, std::memory_order_relaxed);
    current_processor_frame_sequence_.store(current_frame_.sequence, std::memory_order_relaxed);
    current_processor_schema_id_.store(current_frame_.schema_id, std::memory_order_relaxed);
    current_processor_clock_domain_.store(current_frame_.source_clock_domain,
                                          std::memory_order_relaxed);
    if (!processor_prepared_ || !processor_contract_.has_value())
    {
        record_fault(FaultCode::processor_prepare, StreamStatus::invalid_state,
                     FaultStage::processor,
                     static_cast<std::uint32_t>(ProcessorContractDetail::missing_prepared_contract),
                     &current_frame_);
        return StreamStatus::invalid_state;
    }
    const auto input_observer_status = dispatch_to_input_critical_observers(frame.frame().view());
    if (input_observer_status != StreamStatus::ok)
    {
        return input_observer_status;
    }
    terminal_emitter_->begin(std::move(frame), processor_contract_->max_process_outputs_per_input,
                             processor_contract_->can_forward_input, false);
    const auto started = clock_.now_ns();
    current_processor_start_ns_.store(started, std::memory_order_release);
    processor_active_.store(true, std::memory_order_release);
    const auto status = processor_.process(terminal_emitter_->input(), *terminal_emitter_);
    const auto completed = clock_.now_ns();
    const auto execution_ns = completed >= started ? completed - started : 0;
    counters_.observe_processor_execution(execution_ns);
    processor_active_.store(false, std::memory_order_release);
    last_processor_completion_ns_.store(completed, std::memory_order_release);
    counters_.processed();
    const auto published = terminal_emitter_->published_count();
    const auto result = finish_output(status, FaultCode::processor_process);
    if (result == StreamStatus::ok && published == 0)
    {
        counters_.zero_output();
    }
    return result;
}

void NativeStreamRunner::drain_ingress() noexcept
{
    StreamMessage message;
    for (;;)
    {
        const auto status = ingress_->try_pop(message);
        if (status == StreamStatus::ok)
        {
            if (message.kind() == StreamMessageKind::frame)
            {
                counters_.aborted_frame();
            }
            message = StreamMessage{};
            continue;
        }
        if (status == StreamStatus::stopped || producer_done_.load(std::memory_order_acquire))
        {
            return;
        }
        const auto epoch = ingress_epoch_.load(std::memory_order_acquire);
        if (ingress_->empty() && !producer_done_.load(std::memory_order_acquire))
        {
            ingress_epoch_.wait(epoch, std::memory_order_acquire);
        }
    }
}

StreamStatus NativeStreamRunner::finish_gracefully() noexcept
{
    state_.store(RuntimeState::stopping, std::memory_order_release);
    if (!processor_prepared_ || !processor_contract_.has_value())
    {
        record_fault(
            FaultCode::processor_prepare, StreamStatus::invalid_state, FaultStage::processor,
            static_cast<std::uint32_t>(ProcessorContractDetail::missing_prepared_contract));
        return StreamStatus::invalid_state;
    }
    terminal_emitter_->begin(FrameLease{}, processor_contract_->max_flush_outputs, false, true);
    const auto started = clock_.now_ns();
    current_processor_start_ns_.store(started, std::memory_order_release);
    processor_active_.store(true, std::memory_order_release);
    const auto processor_status = processor_.flush(*terminal_emitter_);
    const auto completed = clock_.now_ns();
    processor_active_.store(false, std::memory_order_release);
    last_processor_completion_ns_.store(completed, std::memory_order_release);
    const auto output_status = finish_output(processor_status, FaultCode::processor_flush);
    return output_status;
}

StreamStatus NativeStreamRunner::dispatch_frame_to_observer(ObserverEdge& edge,
                                                            FrameView frame) noexcept
{
    if (!edge.active.load(std::memory_order_acquire))
    {
        return StreamStatus::stopped;
    }

    if (edge.config.drop_policy == ObserverDropPolicy::latest_value)
    {
        while (edge.discard_oldest())
        {
        }
    }

    FrameLease copy;
    auto acquire_status = edge.frame_pool.try_acquire(copy);
    if (acquire_status != StreamStatus::ok &&
        edge.config.drop_policy != ObserverDropPolicy::drop_newest && edge.discard_oldest())
    {
        acquire_status = edge.frame_pool.try_acquire(copy);
    }
    if (acquire_status != StreamStatus::ok)
    {
        edge.record_drop(frame.header.sequence);
        return StreamStatus::observer_overrun;
    }

    copy.frame().header() = frame.header;
    std::copy(frame.blocks.begin(), frame.blocks.end(), copy.frame().block_storage().begin());
    std::memcpy(copy.frame().payload_storage().data(), frame.payload.data(), frame.payload.size());
    const auto sizes = copy.frame().set_used_sizes(frame.blocks.size(), frame.payload.size());
    if (sizes != StreamStatus::ok)
    {
        static_cast<void>(copy.reset());
        return sizes;
    }

    auto message = StreamMessage::from_frame(std::move(copy));
    auto push_status = edge.queue.try_push(std::move(message));
    if (push_status == StreamStatus::queue_overflow &&
        edge.config.drop_policy != ObserverDropPolicy::drop_newest && edge.discard_oldest())
    {
        push_status = edge.queue.try_push(std::move(message));
    }
    if (push_status != StreamStatus::ok)
    {
        edge.record_drop(frame.header.sequence);
        return StreamStatus::observer_overrun;
    }
    edge.enqueued.fetch_add(1, std::memory_order_relaxed);
    edge.last_enqueued_sequence.store(frame.header.sequence, std::memory_order_relaxed);
    edge.observe_size();
    edge.epoch.fetch_add(1, std::memory_order_release);
    edge.epoch.notify_one();
    return StreamStatus::ok;
}

StreamStatus
NativeStreamRunner::dispatch_discontinuity_to_observer(ObserverEdge& edge,
                                                       const Discontinuity& discontinuity) noexcept
{
    if (!edge.active.load(std::memory_order_acquire))
    {
        return StreamStatus::stopped;
    }
    if (edge.config.drop_policy == ObserverDropPolicy::latest_value)
    {
        while (edge.discard_oldest())
        {
        }
    }
    DiscontinuityLease copy;
    auto acquire_status = edge.discontinuity_pool.try_acquire(copy);
    if (acquire_status != StreamStatus::ok &&
        edge.config.drop_policy != ObserverDropPolicy::drop_newest && edge.discard_oldest())
    {
        acquire_status = edge.discontinuity_pool.try_acquire(copy);
    }
    if (acquire_status != StreamStatus::ok)
    {
        edge.record_drop(discontinuity.actual_frame_sequence);
        return StreamStatus::observer_overrun;
    }
    const auto assign_status = copy.assign(
        discontinuity.session_id, discontinuity.previous_frame_sequence,
        discontinuity.actual_frame_sequence, discontinuity.reason, discontinuity.signal_gaps);
    if (assign_status != StreamStatus::ok)
    {
        static_cast<void>(copy.reset());
        return assign_status;
    }
    auto message = StreamMessage::from_discontinuity(std::move(copy));
    auto push_status = edge.queue.try_push(std::move(message));
    if (push_status == StreamStatus::queue_overflow &&
        edge.config.drop_policy != ObserverDropPolicy::drop_newest && edge.discard_oldest())
    {
        push_status = edge.queue.try_push(std::move(message));
    }
    if (push_status != StreamStatus::ok)
    {
        edge.record_drop(discontinuity.actual_frame_sequence);
        return StreamStatus::observer_overrun;
    }
    edge.enqueued.fetch_add(1, std::memory_order_relaxed);
    edge.last_enqueued_sequence.store(discontinuity.actual_frame_sequence,
                                      std::memory_order_relaxed);
    edge.observe_size();
    edge.epoch.fetch_add(1, std::memory_order_release);
    edge.epoch.notify_one();
    return StreamStatus::ok;
}

void NativeStreamRunner::note_rejected_before_acceptance(AcceptedMessageKind kind,
                                                         std::uint64_t frame_sequence) noexcept
{
    for (auto& edge : critical_observer_edges_)
    {
        if (edge->input_tap || !edge->active.load(std::memory_order_acquire))
        {
            continue;
        }
        edge->dropped.fetch_add(1, std::memory_order_relaxed);
        edge->observer.note_rejected_before_acceptance({kind, frame_sequence});
    }
}

void NativeStreamRunner::report_critical_failure(CriticalObserverEdge& edge,
                                                 StreamStatus status) noexcept
{
    edge.failures.fetch_add(1, std::memory_order_relaxed);
    if (edge.fault_reported.exchange(true, std::memory_order_acq_rel))
    {
        return;
    }
    record_fault(FaultCode::critical_observer_failure, status, FaultStage::observer,
                 edge.config.id);
    static_cast<void>(inhibit_safety(SafetyReason::critical_observer_failure));
    request_abort(true);
}

StreamStatus
NativeStreamRunner::dispatch_to_critical_observers(const StreamMessage& message) noexcept
{
    if (!message.has_runtime_acceptance())
    {
        return StreamStatus::invalid_state;
    }
    StreamStatus first_failure = StreamStatus::ok;
    const auto acceptance = message.runtime_acceptance();
    for (auto& edge : critical_observer_edges_)
    {
        if (edge->input_tap || !edge->active.load(std::memory_order_acquire))
        {
            continue;
        }
        StreamStatus status = StreamStatus::ok;
        std::uint64_t sequence{};
        if (message.kind() == StreamMessageKind::frame)
        {
            sequence = message.frame().header.sequence;
            status = edge->observer.observe_accepted(message.frame(), acceptance);
        }
        else
        {
            sequence = message.discontinuity().actual_frame_sequence;
            status =
                edge->observer.handle_accepted_discontinuity(message.discontinuity(), acceptance);
        }
        if (status == StreamStatus::ok)
        {
            edge->delivered.fetch_add(1, std::memory_order_relaxed);
            if (message.kind() == StreamMessageKind::discontinuity)
            {
                edge->discontinuities.fetch_add(1, std::memory_order_relaxed);
            }
            edge->last_delivered_sequence.store(sequence, std::memory_order_relaxed);
        }
        else
        {
            if (first_failure == StreamStatus::ok)
            {
                first_failure = status;
            }
            report_critical_failure(*edge, status);
        }
        const auto health = edge->observer.health();
        if (health != StreamStatus::ok)
        {
            if (first_failure == StreamStatus::ok)
            {
                first_failure = health;
            }
            report_critical_failure(*edge, health);
        }
    }
    return first_failure;
}

StreamStatus NativeStreamRunner::dispatch_to_input_critical_observers(FrameView frame) noexcept
{
    const auto ordinal = next_input_data_message_ordinal_.load(std::memory_order_relaxed);
    const RuntimeAcceptance acceptance{ordinal, clock_.now_ns()};
    StreamStatus first_failure = StreamStatus::ok;
    bool delivered = false;
    for (auto& edge : critical_observer_edges_)
    {
        if (!edge->input_tap || !edge->active.load(std::memory_order_acquire))
        {
            continue;
        }
        delivered = true;
        edge->enqueued.fetch_add(1, std::memory_order_relaxed);
        edge->last_enqueued_sequence.store(frame.header.sequence, std::memory_order_relaxed);
        const auto status = edge->observer.observe_accepted(frame, acceptance);
        const auto health = edge->observer.health();
        const auto failure = status != StreamStatus::ok ? status : health;
        if (failure == StreamStatus::ok)
        {
            edge->delivered.fetch_add(1, std::memory_order_relaxed);
            edge->last_delivered_sequence.store(frame.header.sequence, std::memory_order_relaxed);
        }
        else
        {
            if (first_failure == StreamStatus::ok)
            {
                first_failure = failure;
            }
            report_critical_failure(*edge, failure);
        }
    }
    if (delivered && first_failure == StreamStatus::ok)
    {
        next_input_data_message_ordinal_.store(ordinal + 1, std::memory_order_relaxed);
    }
    return first_failure;
}

StreamStatus NativeStreamRunner::dispatch_to_input_critical_observers(
    const Discontinuity& discontinuity) noexcept
{
    const auto ordinal = next_input_data_message_ordinal_.load(std::memory_order_relaxed);
    const RuntimeAcceptance acceptance{ordinal, clock_.now_ns()};
    StreamStatus first_failure = StreamStatus::ok;
    bool delivered = false;
    for (auto& edge : critical_observer_edges_)
    {
        if (!edge->input_tap || !edge->active.load(std::memory_order_acquire))
        {
            continue;
        }
        delivered = true;
        edge->enqueued.fetch_add(1, std::memory_order_relaxed);
        edge->discontinuities.fetch_add(1, std::memory_order_relaxed);
        edge->last_enqueued_sequence.store(discontinuity.actual_frame_sequence,
                                           std::memory_order_relaxed);
        const auto status = edge->observer.handle_accepted_discontinuity(discontinuity, acceptance);
        const auto health = edge->observer.health();
        const auto failure = status != StreamStatus::ok ? status : health;
        if (failure == StreamStatus::ok)
        {
            edge->delivered.fetch_add(1, std::memory_order_relaxed);
            edge->last_delivered_sequence.store(discontinuity.actual_frame_sequence,
                                                std::memory_order_relaxed);
        }
        else
        {
            if (first_failure == StreamStatus::ok)
            {
                first_failure = failure;
            }
            report_critical_failure(*edge, failure);
        }
    }
    if (delivered && first_failure == StreamStatus::ok)
    {
        next_input_data_message_ordinal_.store(ordinal + 1, std::memory_order_relaxed);
    }
    return first_failure;
}

void NativeStreamRunner::observer_loop(ObserverEdge& edge) noexcept
{
    StreamMessage message;
    while (edge.active.load(std::memory_order_acquire))
    {
        const auto status = edge.queue.try_pop(message);
        if (status == StreamStatus::ok)
        {
            StreamStatus observer_status = StreamStatus::ok;
            if (message.kind() == StreamMessageKind::frame)
            {
                observer_status = edge.observer.observe(message.frame());
                if (observer_status == StreamStatus::ok)
                {
                    edge.delivered.fetch_add(1, std::memory_order_relaxed);
                    edge.last_delivered_sequence.store(message.frame().header.sequence,
                                                       std::memory_order_relaxed);
                }
            }
            else if (message.kind() == StreamMessageKind::discontinuity)
            {
                observer_status = edge.observer.handle_discontinuity(message.discontinuity());
                if (observer_status == StreamStatus::ok)
                {
                    edge.discontinuities.fetch_add(1, std::memory_order_relaxed);
                    edge.last_delivered_sequence.store(
                        message.discontinuity().actual_frame_sequence, std::memory_order_relaxed);
                }
            }
            message = StreamMessage{};
            if (observer_status != StreamStatus::ok)
            {
                edge.failures.fetch_add(1, std::memory_order_relaxed);
                edge.active.store(false, std::memory_order_release);
            }
            continue;
        }
        if (status == StreamStatus::stopped)
        {
            break;
        }
        if (edge.queue.closed() && edge.queue.empty())
        {
            break;
        }
        const auto epoch = edge.epoch.load(std::memory_order_acquire);
        if (edge.queue.empty() && !edge.queue.closed() &&
            edge.active.load(std::memory_order_acquire))
        {
            edge.epoch.wait(epoch, std::memory_order_acquire);
        }
    }

    StreamMessage dropped;
    while (edge.queue.try_pop(dropped) == StreamStatus::ok)
    {
        const auto sequence = dropped.kind() == StreamMessageKind::frame
                                  ? dropped.frame().header.sequence
                                  : dropped.discontinuity().actual_frame_sequence;
        edge.record_drop(sequence);
        dropped = StreamMessage{};
    }
    if (!abort_requested_.load(std::memory_order_acquire))
    {
        const auto status = edge.observer.flush();
        if (status != StreamStatus::ok)
        {
            edge.failures.fetch_add(1, std::memory_order_relaxed);
        }
    }
    edge.done.store(true, std::memory_order_release);
    clock_.wake();
}

RuntimeTerminalReason NativeStreamRunner::terminal_reason() const noexcept
{
    switch (end_reason_.load(std::memory_order_acquire))
    {
    case EndReason::end_of_stream:
        return RuntimeTerminalReason::end_of_stream;
    case EndReason::abort:
        return RuntimeTerminalReason::abort;
    case EndReason::fault:
        return RuntimeTerminalReason::fault;
    case EndReason::none:
    case EndReason::stop:
        return RuntimeTerminalReason::stop;
    }
    return RuntimeTerminalReason::fault;
}

void NativeStreamRunner::cancel_critical_observers() noexcept
{
    const auto fault = faults_ == nullptr ? std::optional<FaultRecord>{} : faults_->first();
    for (auto& edge : critical_observer_edges_)
    {
        if (fault.has_value() && !edge->fault_delivered.exchange(true, std::memory_order_acq_rel))
        {
            edge->observer.publish_primary_fault(*fault);
        }
        edge->observer.cancel();
    }
}

void NativeStreamRunner::finish_critical_observers() noexcept
{
    // Worker termination must not re-enter lifecycle_mutex_: start() may be
    // cleaning up a partially-created worker set while holding that mutex.
    // faults_ remains alive until every worker has joined and reset clears the
    // prepared resources.
    const auto fault = faults_ == nullptr ? std::optional<FaultRecord>{} : faults_->first();
    for (auto& edge : critical_observer_edges_)
    {
        const RuntimeTerminalNotice terminal{
            .reason = terminal_reason(),
            .requested_at_ns = shutdown_started_ns_.load(std::memory_order_acquire),
            .accepted_message_count =
                edge->input_tap ? next_input_data_message_ordinal_.load(std::memory_order_acquire)
                                : next_data_message_ordinal_.load(std::memory_order_acquire),
        };
        if (fault.has_value() && !edge->fault_delivered.exchange(true, std::memory_order_acq_rel))
        {
            edge->observer.publish_primary_fault(*fault);
        }
        const auto status = edge->observer.drain(terminal);
        const auto health = edge->observer.health();
        const auto failure = status != StreamStatus::ok ? status : health;
        if (failure != StreamStatus::ok)
        {
            edge->failures.fetch_add(1, std::memory_order_relaxed);
            if (!edge->fault_reported.exchange(true, std::memory_order_acq_rel))
            {
                // Do not publish this fault back into the observer whose
                // terminal path just failed. That would recurse through the
                // same reserved fault path.
                record_fault(FaultCode::critical_observer_failure, failure, FaultStage::observer,
                             edge->config.id);
                static_cast<void>(inhibit_safety(SafetyReason::critical_observer_failure));
                end_reason_.store(EndReason::fault, std::memory_order_release);
                abort_requested_.store(true, std::memory_order_release);
            }
        }
        edge->active.store(false, std::memory_order_release);
        edge->done.store(true, std::memory_order_release);
    }
}

void NativeStreamRunner::observer_dispatch_loop() noexcept
{
    StreamMessage message;
    for (;;)
    {
        const auto status = observer_dispatch_->try_pop(message);
        if (status == StreamStatus::ok)
        {
            static_cast<void>(dispatch_to_critical_observers(message));
            if (!abort_requested_.load(std::memory_order_acquire))
            {
                for (auto& edge : observer_edges_)
                {
                    if (!edge->active.load(std::memory_order_acquire))
                    {
                        continue;
                    }
                    static_cast<void>(
                        message.kind() == StreamMessageKind::frame
                            ? dispatch_frame_to_observer(*edge, message.frame())
                            : dispatch_discontinuity_to_observer(*edge, message.discontinuity()));
                }
            }
            message = StreamMessage{};
            continue;
        }
        if (status == StreamStatus::stopped ||
            (observer_dispatch_->closed() && observer_dispatch_->empty()))
        {
            break;
        }
        const auto epoch = observer_dispatch_epoch_.load(std::memory_order_acquire);
        if (observer_dispatch_->empty() && !observer_dispatch_->closed())
        {
            observer_dispatch_epoch_.wait(epoch, std::memory_order_acquire);
        }
    }
    for (auto& edge : observer_edges_)
    {
        edge->queue.close();
        edge->epoch.fetch_add(1, std::memory_order_release);
        edge->epoch.notify_all();
    }
    finish_critical_observers();
    observer_dispatch_done_.store(true, std::memory_order_release);
    clock_.wake();
}

void NativeStreamRunner::actuator_loop() noexcept
{
    StreamMessage message;
    bool failed = false;
    for (;;)
    {
        const auto status = critical_edge_->try_pop(message);
        if (status == StreamStatus::ok)
        {
            if (abort_requested_.load(std::memory_order_acquire))
            {
                message = StreamMessage{};
                continue;
            }
            if (safety_inhibited_.load(std::memory_order_acquire))
            {
                message = StreamMessage{};
                continue;
            }
            if (message.kind() == StreamMessageKind::discontinuity)
            {
                const auto sequence = message.discontinuity().actual_frame_sequence;
                if (terminal_consumer_ != nullptr)
                {
                    const auto started = clock_.now_ns();
                    current_actuator_start_ns_.store(started, std::memory_order_release);
                    actuator_active_.store(true, std::memory_order_release);
                    const auto consumer_status =
                        terminal_consumer_->handle_discontinuity(message.discontinuity());
                    actuator_active_.store(false, std::memory_order_release);
                    if (consumer_status != StreamStatus::ok)
                    {
                        counters_.actuator_failure();
                        const auto& discontinuity = message.discontinuity();
                        const FrameHeader context{.session_id = discontinuity.session_id,
                                                  .sequence = discontinuity.actual_frame_sequence,
                                                  .schema_id = schema_.id()};
                        record_fault(FaultCode::consumer_discontinuity, consumer_status,
                                     FaultStage::consumer, 0, &context);
                        static_cast<void>(inhibit_safety(SafetyReason::actuator_failure));
                        request_abort(true);
                        message = StreamMessage{};
                        failed = true;
                        break;
                    }
                }
                StreamStatus dispatch_status = StreamStatus::ok;
                if (observer_dispatch_ != nullptr)
                {
                    const auto ordinal = next_data_message_ordinal_.load(std::memory_order_relaxed);
                    message.set_runtime_acceptance({ordinal, clock_.now_ns()});
                    dispatch_status = observer_dispatch_->try_push(std::move(message));
                    if (dispatch_status == StreamStatus::ok)
                    {
                        next_data_message_ordinal_.store(ordinal + 1, std::memory_order_relaxed);
                        for (auto& edge : critical_observer_edges_)
                        {
                            if (edge->input_tap)
                            {
                                continue;
                            }
                            edge->enqueued.fetch_add(1, std::memory_order_relaxed);
                            edge->last_enqueued_sequence.store(sequence, std::memory_order_relaxed);
                        }
                    }
                }
                if (dispatch_status == StreamStatus::ok)
                {
                    if (observer_dispatch_ != nullptr)
                    {
                        notify_observer_dispatch();
                    }
                }
                else
                {
                    counters_.observer_dispatch_drop();
                    note_rejected_before_acceptance(AcceptedMessageKind::discontinuity, sequence);
                    const bool critical = std::any_of(
                        critical_observer_edges_.begin(), critical_observer_edges_.end(),
                        [](const auto& edge) { return !edge->input_tap; });
                    for (auto& edge : observer_edges_)
                    {
                        if (edge->active.load(std::memory_order_acquire))
                        {
                            edge->record_drop(sequence);
                        }
                    }
                    if (critical)
                    {
                        record_fault(FaultCode::critical_observer_overrun,
                                     StreamStatus::observer_overrun, FaultStage::observer);
                        static_cast<void>(inhibit_safety(SafetyReason::critical_observer_failure));
                        request_abort(true);
                        failed = true;
                    }
                }
                message = StreamMessage{};
                if (failed)
                {
                    break;
                }
                continue;
            }
            auto frame = message.take_frame();
            const auto generated_at = message.enqueued_at_ns();
            const auto configured_until = saturating_add(generated_at, config_.actuator_deadline);
            const auto header = frame.frame().header();
            const auto has_source_time = has_flag(header.flags, FrameFlags::source_received);
            const auto source_until =
                has_source_time
                    ? saturating_add(header.host_received_ns, config_.max_source_to_actuator_age)
                    : std::numeric_limits<HostTimeNs>::max();
            const auto valid_until =
                has_flag(header.flags, FrameFlags::valid_until)
                    ? std::min({header.valid_until_ns, configured_until, source_until})
                    : std::min(configured_until, source_until);
            const ActuatorCommand command{
                .id = next_command_id_.fetch_add(1, std::memory_order_relaxed),
                .sequence = header.sequence,
                .generated_at_ns = generated_at,
                .valid_until_ns = valid_until,
                .session_id = header.session_id,
                .runtime_generation = runtime_generation_.load(std::memory_order_acquire),
                .payload = frame.view(),
            };
            const auto now = clock_.now_ns();
            if (has_source_time)
            {
                counters_.observe_source_to_actuator(
                    now >= header.host_received_ns ? now - header.host_received_ns : 0);
            }
            current_actuator_start_ns_.store(now, std::memory_order_release);
            actuator_active_.store(true, std::memory_order_release);
            const auto actuator_status = actuator_.submit(command, now);
            actuator_active_.store(false, std::memory_order_release);
            if (actuator_status == StreamStatus::ok &&
                (abort_requested_.load(std::memory_order_acquire) ||
                 safety_inhibited_.load(std::memory_order_acquire)))
            {
                message = StreamMessage{};
                continue;
            }
            if (actuator_status != StreamStatus::ok)
            {
                const auto deadline = actuator_status == StreamStatus::deadline_exceeded;
                const auto stale_input =
                    deadline && has_source_time &&
                    expired(now, header.host_received_ns, config_.max_source_to_actuator_age);
                counters_.actuator_failure();
                if (deadline)
                {
                    counters_.actuator_deadline_miss();
                    counters_.deadline_fault();
                }
                record_fault(stale_input ? FaultCode::actuator_input_stale
                             : deadline  ? FaultCode::actuator_deadline
                                         : FaultCode::actuator_write,
                             actuator_status, FaultStage::actuator, 0, &header, now);
                static_cast<void>(inhibit_safety(SafetyReason::actuator_failure));
                request_abort(true);
                failed = true;
                message = StreamMessage{};
                break;
            }
            counters_.actuator_applied();
            last_valid_output_ns_.store(clock_.now_ns(), std::memory_order_release);
            output_valid_.store(true, std::memory_order_release);

            if (observer_dispatch_ != nullptr)
            {
                auto dispatch = StreamMessage::from_frame(std::move(frame));
                const auto ordinal = next_data_message_ordinal_.load(std::memory_order_relaxed);
                dispatch.set_runtime_acceptance({ordinal, clock_.now_ns()});
                const auto dispatch_status = observer_dispatch_->try_push(std::move(dispatch));
                if (dispatch_status == StreamStatus::ok)
                {
                    next_data_message_ordinal_.store(ordinal + 1, std::memory_order_relaxed);
                    for (auto& edge : critical_observer_edges_)
                    {
                        if (edge->input_tap)
                        {
                            continue;
                        }
                        edge->enqueued.fetch_add(1, std::memory_order_relaxed);
                        edge->last_enqueued_sequence.store(header.sequence,
                                                           std::memory_order_relaxed);
                    }
                    notify_observer_dispatch();
                }
                else
                {
                    counters_.observer_dispatch_drop();
                    note_rejected_before_acceptance(AcceptedMessageKind::frame, header.sequence);
                    const bool critical = std::any_of(
                        critical_observer_edges_.begin(), critical_observer_edges_.end(),
                        [](const auto& edge) { return !edge->input_tap; });
                    for (auto& edge : observer_edges_)
                    {
                        if (edge->active.load(std::memory_order_acquire))
                        {
                            edge->record_drop(header.sequence);
                        }
                    }
                    if (critical)
                    {
                        record_fault(FaultCode::critical_observer_overrun,
                                     StreamStatus::observer_overrun, FaultStage::observer, 0,
                                     &header);
                        static_cast<void>(inhibit_safety(SafetyReason::critical_observer_failure));
                        request_abort(true);
                        failed = true;
                    }
                }
            }
            message = StreamMessage{};
            if (failed)
            {
                break;
            }
            continue;
        }
        if (status == StreamStatus::stopped)
        {
            break;
        }
        if (abort_requested_.load(std::memory_order_acquire) &&
            processing_done_.load(std::memory_order_acquire))
        {
            failed = end_reason_.load(std::memory_order_acquire) == EndReason::fault;
            break;
        }
        if (critical_edge_->closed() && critical_edge_->empty())
        {
            break;
        }
        const auto epoch = critical_epoch_.load(std::memory_order_acquire);
        if (critical_edge_->empty() && !critical_edge_->closed())
        {
            critical_epoch_.wait(epoch, std::memory_order_acquire);
        }
    }

    // Drain every remaining message, including any frame the processing thread
    // publishes after the actuator broke out of its main loop mid-failure. The
    // main loop's early break can race a still-in-flight publication, so draining
    // only what is currently available would strand that frame's pool lease in
    // the critical edge until runtime destruction and leak outstanding_frames().
    // Wait until the producer (processing_loop) closes the edge and the queue is
    // empty, mirroring the main loop's wait protocol.
    StreamMessage abandoned;
    for (;;)
    {
        if (critical_edge_->try_pop(abandoned) == StreamStatus::ok)
        {
            abandoned = StreamMessage{};
            continue;
        }
        if (critical_edge_->closed() && critical_edge_->empty())
        {
            break;
        }
        const auto epoch = critical_epoch_.load(std::memory_order_acquire);
        if (critical_edge_->empty() && !critical_edge_->closed())
        {
            critical_epoch_.wait(epoch, std::memory_order_acquire);
        }
    }
    if (!failed && !abort_requested_.load(std::memory_order_acquire))
    {
        const auto status = actuator_.flush();
        if (status != StreamStatus::ok)
        {
            counters_.actuator_failure();
            record_fault(FaultCode::actuator_flush, status, FaultStage::actuator);
            static_cast<void>(inhibit_safety(SafetyReason::actuator_failure));
            request_abort(true);
            failed = true;
        }
    }
    if (observer_dispatch_ != nullptr)
    {
        observer_dispatch_->close();
        notify_observer_dispatch();
    }
    else
    {
        // Input taps are delivered synchronously and do not require an output
        // dispatch thread, but they still receive the same ordered terminal.
        finish_critical_observers();
    }

    const auto end_reason = end_reason_.load(std::memory_order_acquire);
    const auto safety_reason = end_reason == EndReason::end_of_stream ? SafetyReason::end_of_stream
                               : end_reason == EndReason::fault       ? SafetyReason::runtime_fault
                                                                      : SafetyReason::explicit_stop;
    if (inhibit_safety(safety_reason) != StreamStatus::ok)
    {
        failed = true;
    }
    actuator_done_.store(true, std::memory_order_release);
    clock_.wake();
}

void NativeStreamRunner::processing_loop() noexcept
{
    StreamMessage message;
    bool failed = false;
    for (;;)
    {
        if (abort_requested_.load(std::memory_order_acquire))
        {
            drain_ingress();
            failed = end_reason_.load(std::memory_order_acquire) == EndReason::fault;
            break;
        }

        const auto pop_status = ingress_->try_pop(message);
        if (pop_status == StreamStatus::ok)
        {
            const auto now = clock_.now_ns();
            const auto enqueued_at = message.enqueued_at_ns();
            counters_.observe_ingress_dwell(now >= enqueued_at ? now - enqueued_at : 0);
            if (expired(now, message.enqueued_at_ns(), config_.max_ingress_dwell))
            {
                counters_.aborted_frame();
                message = StreamMessage{};
                trigger_deadline_fault(FaultCode::ingress_dwell_timeout, FaultStage::runtime,
                                       SafetyReason::ingress_dwell, now);
                continue;
            }

            if (message.kind() == StreamMessageKind::discontinuity)
            {
                // An explicit source discontinuity is forwarded as it stands.
                // The checker is told about it so the frame that follows does
                // not also produce an inferred break for the same gap.
                auto lease = message.take_discontinuity();
                message = StreamMessage{};
                continuity_->accept_discontinuity(lease.view());
                const auto explicit_status = handle_discontinuity(std::move(lease));
                if (explicit_status != StreamStatus::ok)
                {
                    static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
                    request_abort(true);
                }
                continue;
            }

            auto frame = message.take_frame();
            current_frame_ = frame.frame().header();
            auto checked = continuity_->check(std::move(frame), *discontinuity_pool_);
            if (checked.status == ContinuityStatus::fatal)
            {
                counters_.aborted_frame();
                record_fault(FaultCode::continuity_validation, StreamStatus::invalid_frame,
                             FaultStage::continuity, static_cast<std::uint32_t>(checked.error),
                             &current_frame_);
                static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
                request_abort(true);
                continue;
            }

            StreamStatus status = StreamStatus::ok;
            bool frame_processed = false;
            for (std::size_t i = 0; i < checked.n_messages && status == StreamStatus::ok; ++i)
            {
                auto& output = checked.messages[i];
                if (output.kind() == StreamMessageKind::discontinuity)
                {
                    status = handle_discontinuity(output.take_discontinuity());
                }
                else if (output.kind() == StreamMessageKind::frame)
                {
                    frame_processed = true;
                    status = process_frame(output.take_frame());
                }
            }
            message = StreamMessage{};
            if (status != StreamStatus::ok)
            {
                if (!frame_processed)
                {
                    counters_.aborted_frame();
                }
                static_cast<void>(inhibit_safety(SafetyReason::runtime_fault));
                request_abort(true);
            }
            continue;
        }

        // producer_done_ alone cannot end the loop: try_pop above may have read
        // an empty ring microseconds before the producer's final push became
        // visible, and the producer publishes producer_done_ after that push.
        // Breaking on the fresh flag while acting on the stale pop stranded the
        // last frame in the ingress ring, leaking its pool lease and losing the
        // frame. try_pop's own stopped result already means "closed and fully
        // drained", so the flag only has to be paired with an emptiness check
        // the consumer can trust for itself.
        if (pop_status == StreamStatus::stopped ||
            (producer_done_.load(std::memory_order_acquire) && ingress_->empty()))
        {
            const auto status = finish_gracefully();
            failed = status != StreamStatus::ok;
            break;
        }

        const auto epoch = ingress_epoch_.load(std::memory_order_acquire);
        if (ingress_->empty() && !producer_done_.load(std::memory_order_acquire) &&
            !abort_requested_.load(std::memory_order_acquire))
        {
            ingress_epoch_.wait(epoch, std::memory_order_acquire);
        }
    }

    if (failed)
    {
        end_reason_.store(EndReason::fault, std::memory_order_release);
    }
    critical_edge_->close();
    notify_critical();
    processing_done_.store(true, std::memory_order_release);
    clock_.wake();
}

std::optional<HostTimeNs> NativeStreamRunner::oldest_ingress_timestamp() const noexcept
{
    for (std::size_t attempt = 0; attempt < 2; ++attempt)
    {
        const auto consumed = ingress_->consumer_sequence();
        const auto produced = ingress_->producer_sequence();
        if (consumed == produced)
        {
            return std::nullopt;
        }
        const auto timestamp =
            ingress_timestamps_[consumed % ingress_->capacity()].load(std::memory_order_relaxed);
        if (ingress_->consumer_sequence() == consumed)
        {
            return timestamp;
        }
    }
    return std::nullopt;
}

void NativeStreamRunner::watchdog_loop() noexcept
{
    while (!watchdog_stop_.load(std::memory_order_acquire))
    {
        const auto now = clock_.now_ns();
        const auto current_state = state();
        const auto monitoring_stages =
            (current_state == RuntimeState::running || current_state == RuntimeState::stopping) &&
            !abort_requested_.load(std::memory_order_acquire);
        if (current_state == RuntimeState::running || current_state == RuntimeState::stopping)
        {
            for (auto& edge : critical_observer_edges_)
            {
                const auto health = edge->observer.health();
                if (health != StreamStatus::ok)
                {
                    report_critical_failure(*edge, health);
                }
            }
        }
        if (monitoring_stages && processor_active_.load(std::memory_order_acquire) &&
            expired(now, current_processor_start_ns_.load(std::memory_order_acquire),
                    config_.processor_execution_deadline))
        {
            trigger_deadline_fault(FaultCode::processor_deadline, FaultStage::processor,
                                   SafetyReason::processor_deadline, now);
        }
        else if (monitoring_stages && actuator_active_.load(std::memory_order_acquire) &&
                 expired(now, current_actuator_start_ns_.load(std::memory_order_acquire),
                         config_.actuator_deadline))
        {
            trigger_deadline_fault(FaultCode::actuator_deadline, FaultStage::actuator,
                                   SafetyReason::actuator_failure, now);
        }
        else if (current_state == RuntimeState::running)
        {
            if (!producer_done_.load(std::memory_order_acquire) &&
                expired(now, last_source_heartbeat_ns_.load(std::memory_order_acquire),
                        config_.source_stall_timeout))
            {
                trigger_deadline_fault(FaultCode::source_stall, FaultStage::source,
                                       SafetyReason::source_stall, now);
            }
            else if (const auto oldest = oldest_ingress_timestamp();
                     oldest.has_value() && expired(now, *oldest, config_.max_ingress_dwell))
            {
                trigger_deadline_fault(FaultCode::ingress_dwell_timeout, FaultStage::runtime,
                                       SafetyReason::ingress_dwell, now);
            }
            else
            {
                const auto output_reference =
                    output_valid_.load(std::memory_order_acquire)
                        ? last_valid_output_ns_.load(std::memory_order_acquire)
                        : runtime_started_ns_.load(std::memory_order_acquire);
                if (expired(now, output_reference, config_.max_output_age))
                {
                    trigger_deadline_fault(FaultCode::output_stale, FaultStage::output,
                                           SafetyReason::output_stale, now);
                }
            }
        }
        else if (current_state == RuntimeState::stopping &&
                 shutdown_active_.load(std::memory_order_acquire) && !workers_done() &&
                 expired(now, shutdown_started_ns_.load(std::memory_order_acquire),
                         config_.shutdown_deadline) &&
                 !shutdown_timeout_recorded_.exchange(true, std::memory_order_acq_rel))
        {
            trigger_deadline_fault(FaultCode::shutdown_timeout, FaultStage::runtime,
                                   SafetyReason::shutdown_timeout, now, pending_worker_mask());
            cancel_critical_observers();
        }

        clock_.wait_until(saturating_add(now, config_.watchdog_period));
    }
}

StreamStatus NativeStreamRunner::wait_for_shutdown() noexcept
{
    while (!workers_done())
    {
        const auto now = clock_.now_ns();
        if (expired(now, shutdown_started_ns_.load(std::memory_order_acquire),
                    config_.shutdown_deadline))
        {
            if (!shutdown_timeout_recorded_.exchange(true, std::memory_order_acq_rel))
            {
                trigger_deadline_fault(FaultCode::shutdown_timeout, FaultStage::runtime,
                                       SafetyReason::shutdown_timeout, now, pending_worker_mask());
            }
            cancel_critical_observers();
            return StreamStatus::deadline_exceeded;
        }
        const auto period_end = saturating_add(now, config_.watchdog_period);
        const auto shutdown_end = saturating_add(
            shutdown_started_ns_.load(std::memory_order_acquire), config_.shutdown_deadline);
        clock_.wait_until(std::min(period_end, shutdown_end));
    }
    return StreamStatus::ok;
}

bool NativeStreamRunner::workers_done() const noexcept
{
    if (!acquisition_done_.load(std::memory_order_acquire) ||
        !processing_done_.load(std::memory_order_acquire) ||
        !actuator_done_.load(std::memory_order_acquire) ||
        !observer_dispatch_done_.load(std::memory_order_acquire))
    {
        return false;
    }
    return std::all_of(observer_edges_.begin(), observer_edges_.end(),
                       [](const auto& edge) { return edge->done.load(std::memory_order_acquire); });
}

std::uint32_t NativeStreamRunner::pending_worker_mask() const noexcept
{
    std::uint32_t result{};
    result |= acquisition_done_.load(std::memory_order_acquire) ? 0U : 1U;
    result |= processing_done_.load(std::memory_order_acquire) ? 0U : 2U;
    result |= actuator_done_.load(std::memory_order_acquire) ? 0U : 4U;
    result |= observer_dispatch_done_.load(std::memory_order_acquire) ? 0U : 8U;
    const auto observer_pending =
        std::any_of(observer_edges_.begin(), observer_edges_.end(),
                    [](const auto& edge) { return !edge->done.load(std::memory_order_acquire); });
    return observer_pending ? result | 16U : result;
}

StreamStatus NativeStreamRunner::join_threads(bool lifecycle_locked) noexcept
{
    const std::lock_guard lock(join_mutex_);
    if (acquisition_thread_.joinable())
    {
        acquisition_thread_.join();
    }
    if (processing_thread_.joinable())
    {
        processing_thread_.join();
    }
    if (actuator_thread_.joinable())
    {
        actuator_thread_.join();
    }
    if (observer_dispatch_thread_.joinable())
    {
        observer_dispatch_thread_.join();
    }
    for (auto& edge : observer_edges_)
    {
        if (edge->thread.joinable())
        {
            edge->thread.join();
        }
    }
    watchdog_stop_.store(true, std::memory_order_release);
    clock_.wake();
    if (watchdog_thread_.joinable())
    {
        watchdog_thread_.join();
    }
    const auto fault = lifecycle_locked ? primary_fault_unlocked() : primary_fault();
    state_.store(fault.has_value() ? RuntimeState::failed : RuntimeState::stopped,
                 std::memory_order_release);
    return fault.has_value() ? fault->status : StreamStatus::ok;
}

StreamStatus NativeStreamRunner::join() noexcept
{
    {
        const std::lock_guard lock(lifecycle_mutex_);
        const auto current = state();
        if (current == RuntimeState::created || current == RuntimeState::prepared)
        {
            return StreamStatus::invalid_state;
        }
    }
    while (!shutdown_active_.load(std::memory_order_acquire) && !workers_done())
    {
        const auto now = clock_.now_ns();
        clock_.wait_until(saturating_add(now, config_.watchdog_period));
    }
    if (!workers_done() && wait_for_shutdown() != StreamStatus::ok)
    {
        return StreamStatus::deadline_exceeded;
    }
    return join_threads();
}

StreamStatus NativeStreamRunner::run()
{
    const auto status = start();
    return status == StreamStatus::ok ? join() : status;
}

StreamStatus NativeStreamRunner::stop() noexcept
{
    bool wait = false;
    {
        const std::lock_guard lock(lifecycle_mutex_);
        const auto current = state();
        if (current == RuntimeState::prepared)
        {
            const auto status = inhibit_safety(SafetyReason::explicit_stop);
            cancel_critical_observers();
            state_.store(status == StreamStatus::ok ? RuntimeState::stopped : RuntimeState::failed,
                         std::memory_order_release);
            return status;
        }
        if (current != RuntimeState::running && current != RuntimeState::stopping &&
            current != RuntimeState::stopped && current != RuntimeState::failed)
        {
            return StreamStatus::invalid_state;
        }
        if (current == RuntimeState::running || current == RuntimeState::stopping)
        {
            state_.store(RuntimeState::stopping, std::memory_order_release);
            stop_requested_.store(true, std::memory_order_release);
            auto expected = EndReason::none;
            static_cast<void>(end_reason_.compare_exchange_strong(
                expected, EndReason::stop, std::memory_order_release, std::memory_order_relaxed));
            begin_shutdown();
            static_cast<void>(inhibit_safety(SafetyReason::explicit_stop));
            source_.cancel();
            notify_processing();
            clock_.wake();
            wait = true;
        }
    }
    if (wait && wait_for_shutdown() != StreamStatus::ok)
    {
        return StreamStatus::deadline_exceeded;
    }
    return join_threads();
}

StreamStatus NativeStreamRunner::abort() noexcept
{
    bool wait = false;
    {
        const std::lock_guard lock(lifecycle_mutex_);
        const auto current = state();
        if (current == RuntimeState::prepared)
        {
            const auto status = inhibit_safety(SafetyReason::explicit_stop);
            cancel_critical_observers();
            state_.store(status == StreamStatus::ok ? RuntimeState::stopped : RuntimeState::failed,
                         std::memory_order_release);
            return status;
        }
        if (current != RuntimeState::running && current != RuntimeState::stopping &&
            current != RuntimeState::stopped && current != RuntimeState::failed)
        {
            return StreamStatus::invalid_state;
        }
        if (current == RuntimeState::running || current == RuntimeState::stopping)
        {
            static_cast<void>(inhibit_safety(SafetyReason::explicit_stop));
            request_abort(false);
            wait = true;
        }
    }
    if (wait && wait_for_shutdown() != StreamStatus::ok)
    {
        return StreamStatus::deadline_exceeded;
    }
    return join_threads();
}

void NativeStreamRunner::clear_resources() noexcept
{
    terminal_emitter_.reset();
    critical_observer_edges_.clear();
    observer_edges_.clear();
    observer_dispatch_.reset();
    critical_edge_.reset();
    ingress_.reset();
    ingress_timestamps_.reset();
    continuity_.reset();
    output_validator_.reset();
    discontinuity_pool_.reset();
    frame_pool_.reset();
    faults_.reset();
    thread_startup_.reset();
}

StreamStatus NativeStreamRunner::reset() noexcept
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() != RuntimeState::stopped && state() != RuntimeState::failed)
    {
        return StreamStatus::invalid_state;
    }
    if (acquisition_thread_.joinable() || processing_thread_.joinable() ||
        actuator_thread_.joinable() || observer_dispatch_thread_.joinable() ||
        watchdog_thread_.joinable())
    {
        if (!workers_done())
        {
            return StreamStatus::invalid_state;
        }
        static_cast<void>(join_threads(true));
    }

    const auto source_status = source_.reset();
    if (source_status != StreamStatus::ok)
    {
        record_fault(FaultCode::source_reset, source_status, FaultStage::source);
    }
    const auto processor_status = processor_.reset();
    processor_prepared_ = false;
    if (processor_status != StreamStatus::ok)
    {
        record_fault(FaultCode::processor_reset, processor_status, FaultStage::processor);
    }
    const auto actuator_status = actuator_.reset();
    if (actuator_status != StreamStatus::ok)
    {
        record_fault(FaultCode::actuator_reset, actuator_status, FaultStage::actuator);
    }
    StreamStatus observer_status = StreamStatus::ok;
    for (auto& edge : observer_edges_)
    {
        const auto status = edge->observer.reset();
        if (status != StreamStatus::ok && observer_status == StreamStatus::ok)
        {
            observer_status = status;
            record_fault(FaultCode::critical_observer_failure, status, FaultStage::observer,
                         edge->config.id);
        }
    }
    for (auto& edge : critical_observer_edges_)
    {
        const auto status = edge->observer.reset();
        if (status != StreamStatus::ok && observer_status == StreamStatus::ok)
        {
            observer_status = status;
            record_fault(FaultCode::critical_observer_failure, status, FaultStage::observer,
                         edge->config.id);
        }
    }
    if (source_status != StreamStatus::ok || processor_status != StreamStatus::ok ||
        actuator_status != StreamStatus::ok || observer_status != StreamStatus::ok)
    {
        state_.store(RuntimeState::failed, std::memory_order_release);
        if (source_status != StreamStatus::ok)
        {
            return source_status;
        }
        if (processor_status != StreamStatus::ok)
        {
            return processor_status;
        }
        return actuator_status != StreamStatus::ok ? actuator_status : observer_status;
    }

    clear_resources();
    counters_.reset();
    current_frame_ = {};
    state_.store(RuntimeState::created, std::memory_order_release);
    return StreamStatus::ok;
}

RuntimeHeartbeatSnapshot NativeStreamRunner::heartbeat() const noexcept
{
    return RuntimeHeartbeatSnapshot{
        .last_source_heartbeat = last_source_heartbeat_ns_.load(std::memory_order_acquire),
        .current_processor_start = current_processor_start_ns_.load(std::memory_order_acquire),
        .last_processor_completion = last_processor_completion_ns_.load(std::memory_order_acquire),
        .last_valid_output = last_valid_output_ns_.load(std::memory_order_acquire),
        .current_actuator_start = current_actuator_start_ns_.load(std::memory_order_acquire),
        .last_inhibit = last_inhibit_ns_.load(std::memory_order_acquire),
        .runtime_generation = runtime_generation_.load(std::memory_order_acquire),
        .processor_active = processor_active_.load(std::memory_order_acquire),
        .actuator_active = actuator_active_.load(std::memory_order_acquire),
        .output_valid = output_valid_.load(std::memory_order_acquire),
        .safety_inhibited = safety_inhibited_.load(std::memory_order_acquire),
    };
}

RealtimeConfigurationStatus NativeStreamRunner::realtime_configuration_status() const noexcept
{
    const std::shared_lock lock(lifecycle_mutex_);
    return realtime_status_;
}

std::optional<FaultRecord> NativeStreamRunner::primary_fault() const noexcept
{
    const std::shared_lock lock(lifecycle_mutex_);
    return primary_fault_unlocked();
}

std::optional<FaultRecord> NativeStreamRunner::primary_fault_unlocked() const noexcept
{
    return faults_ == nullptr ? std::nullopt : faults_->first();
}

std::size_t
NativeStreamRunner::copy_fault_history(std::span<FaultRecord> destination) const noexcept
{
    const std::shared_lock lock(lifecycle_mutex_);
    return faults_ == nullptr ? 0 : faults_->copy(destination);
}

std::size_t NativeStreamRunner::outstanding_frames() const noexcept
{
    const std::shared_lock lock(lifecycle_mutex_);
    return frame_pool_ == nullptr ? 0 : frame_pool_->outstanding();
}

std::size_t NativeStreamRunner::outstanding_discontinuities() const noexcept
{
    const std::shared_lock lock(lifecycle_mutex_);
    return discontinuity_pool_ == nullptr ? 0 : discontinuity_pool_->outstanding();
}

StreamStatus NativeStreamRunner::detach_observer(ObserverId id) noexcept
{
    const std::lock_guard lock(lifecycle_mutex_);
    if (state() == RuntimeState::created)
    {
        const auto iterator =
            std::find_if(observer_registrations_.begin(), observer_registrations_.end(),
                         [id](const ObserverRegistration& registration)
                         { return registration.config.id == id; });
        if (iterator == observer_registrations_.end())
        {
            return StreamStatus::invalid_state;
        }
        observer_registrations_.erase(iterator);
        return StreamStatus::ok;
    }
    const auto iterator = std::find_if(observer_edges_.begin(), observer_edges_.end(),
                                       [id](const auto& edge) { return edge->config.id == id; });
    if (iterator == observer_edges_.end())
    {
        const auto current = state();
        const auto critical =
            std::find_if(critical_observer_edges_.begin(), critical_observer_edges_.end(),
                         [id](const auto& edge) { return edge->config.id == id; });
        if (critical == critical_observer_edges_.end() ||
            (current != RuntimeState::stopped && current != RuntimeState::failed))
        {
            return StreamStatus::invalid_state;
        }
        const auto worker_joinable =
            acquisition_thread_.joinable() || processing_thread_.joinable() ||
            actuator_thread_.joinable() || observer_dispatch_thread_.joinable() ||
            watchdog_thread_.joinable() ||
            std::any_of(observer_edges_.begin(), observer_edges_.end(),
                        [](const auto& edge) { return edge->thread.joinable(); });
        if (worker_joinable)
        {
            return StreamStatus::invalid_state;
        }
        const auto registration =
            std::find_if(observer_registrations_.begin(), observer_registrations_.end(),
                         [id](const ObserverRegistration& value)
                         { return value.config.id == id && value.critical_observer != nullptr; });
        if (registration == observer_registrations_.end())
        {
            return StreamStatus::invalid_state;
        }
        (*critical)->active.store(false, std::memory_order_release);
        (*critical)->observer.cancel();
        critical_observer_edges_.erase(critical);
        observer_registrations_.erase(registration);
        return StreamStatus::ok;
    }
    auto& edge = **iterator;
    if (!edge.active.exchange(false, std::memory_order_acq_rel))
    {
        return StreamStatus::stopped;
    }
    edge.queue.close();
    edge.observer.cancel();
    edge.epoch.fetch_add(1, std::memory_order_release);
    edge.epoch.notify_all();
    return StreamStatus::ok;
}

std::optional<ObserverEdgeStats> NativeStreamRunner::observer_stats(ObserverId id) const noexcept
{
    const std::shared_lock lock(lifecycle_mutex_);
    const auto iterator = std::find_if(observer_edges_.begin(), observer_edges_.end(),
                                       [id](const auto& edge) { return edge->config.id == id; });
    if (iterator != observer_edges_.end())
    {
        return (*iterator)->snapshot();
    }
    const auto critical =
        std::find_if(critical_observer_edges_.begin(), critical_observer_edges_.end(),
                     [id](const auto& edge) { return edge->config.id == id; });
    return critical == critical_observer_edges_.end()
               ? std::nullopt
               : std::optional<ObserverEdgeStats>{(*critical)->snapshot()};
}

std::size_t NativeStreamRunner::copy_observer_drop_ranges(
    ObserverId id, std::span<ObserverDropRange> destination) const noexcept
{
    const std::shared_lock lock(lifecycle_mutex_);
    const auto iterator = std::find_if(observer_edges_.begin(), observer_edges_.end(),
                                       [id](const auto& edge) { return edge->config.id == id; });
    if (iterator == observer_edges_.end())
    {
        return 0;
    }
    const auto& edge = **iterator;
    const auto count = std::min({
        edge.drop_claims.load(std::memory_order_acquire),
        edge.config.drop_history_capacity,
        destination.size(),
    });
    std::size_t copied = 0;
    for (std::size_t i = 0; i < count; ++i)
    {
        if (edge.drop_history[i].ready.load(std::memory_order_acquire) != 0)
        {
            destination[copied++] = edge.drop_history[i].range;
        }
    }
    return copied;
}

} // namespace neurale::streaming
