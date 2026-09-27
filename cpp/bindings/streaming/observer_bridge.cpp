/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "observer_bridge.h"
#include "types.h"

#include <neurale/streaming/observer_queue.h>
#include <neurale/streaming/runtime.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using namespace neurale::streaming;

using PythonObserverSignalBlock = python::SignalBlock;
using PythonObserverFrame = python::Frame;
using PythonObserverSignalGap = python::SignalGap;
using PythonObserverDiscontinuity = python::Discontinuity;

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

struct PythonObserverBridgeStats
{
    std::uint64_t enqueued{};
    std::uint64_t delivered{};
    std::uint64_t frames_delivered{};
    std::uint64_t discontinuities_delivered{};
    std::uint64_t dropped{};
    std::uint64_t drop_range_count{};
    std::uint64_t drop_history_dropped{};
    std::uint64_t callback_errors{};
    std::uint64_t high_water_mark{};
    std::size_t queue_size{};
    std::size_t native_frames_outstanding{};
    std::size_t native_discontinuities_outstanding{};
    bool callback_active{};
    bool closed{};
    bool worker_done{};
};

class PythonObserverBridge;

// Forward declarations for the interpreter-shutdown registry (defined after the
// class); the constructor/destructor register/unregister themselves there.
void register_live_bridge(PythonObserverBridge* bridge);
void unregister_live_bridge(PythonObserverBridge* bridge) noexcept;

class PythonObserverBridge final : public NativeObserver
{
  public:
    PythonObserverBridge(py::object callback, const StreamSchema& schema, std::size_t capacity,
                         ObserverDropPolicy drop_policy, std::size_t drop_history_capacity)
        : callback_(std::move(callback)), capacity_(capacity),
          payload_bytes_(schema_payload_bytes(schema)), max_signal_blocks_(schema.signals().size()),
          gaps_per_discontinuity_(schema.signals().size()), drop_policy_(drop_policy),
          drop_history_capacity_(drop_history_capacity),
          drop_history_(std::make_unique_for_overwrite<DropSlot[]>(drop_history_capacity))
    {
        if (!PyCallable_Check(callback_.ptr()))
        {
            throw std::invalid_argument("callback must be callable");
        }
        if (capacity == 0 || payload_bytes_ == 0 || max_signal_blocks_ == 0 ||
            gaps_per_discontinuity_ == 0 || drop_history_capacity == 0)
        {
            throw std::invalid_argument(
                "bridge capacity, native storage limits, and drop history capacity "
                "must be positive");
        }
        initialize_drop_history();
        initialize_resources();
        start_worker();
        register_live_bridge(this);
    }

    ~PythonObserverBridge() noexcept override
    {
        unregister_live_bridge(this);
        request_close(false);
        join_worker();
        release_resources();
    }

    PythonObserverBridge(const PythonObserverBridge&) = delete;
    PythonObserverBridge& operator=(const PythonObserverBridge&) = delete;

    StreamStatus observe(FrameView frame) noexcept override
    {
        if (!accepting_.load(std::memory_order_acquire))
        {
            record_drop(frame.header.sequence);
            return StreamStatus::ok;
        }
        if (drop_policy_ == ObserverDropPolicy::latest_value)
        {
            while (discard_oldest())
            {
            }
        }

        FrameLease copy;
        auto status = frame_pool_->try_acquire(copy);
        if (status != StreamStatus::ok && drop_policy_ != ObserverDropPolicy::drop_newest &&
            discard_oldest())
        {
            status = frame_pool_->try_acquire(copy);
        }
        if (status != StreamStatus::ok ||
            frame.blocks.size() > copy.frame().block_storage().size() ||
            frame.payload.size() > copy.frame().payload_storage().size())
        {
            record_drop(frame.header.sequence);
            return StreamStatus::ok;
        }

        copy.frame().header() = frame.header;
        std::copy(frame.blocks.begin(), frame.blocks.end(), copy.frame().block_storage().begin());
        std::memcpy(copy.frame().payload_storage().data(), frame.payload.data(),
                    frame.payload.size());
        status = copy.frame().set_used_sizes(frame.blocks.size(), frame.payload.size());
        if (status != StreamStatus::ok)
        {
            static_cast<void>(copy.reset());
            record_drop(frame.header.sequence);
            return StreamStatus::ok;
        }

        auto message = StreamMessage::from_frame(std::move(copy));
        status = queue_->try_push(std::move(message));
        if (status == StreamStatus::queue_overflow &&
            drop_policy_ != ObserverDropPolicy::drop_newest && discard_oldest())
        {
            status = queue_->try_push(std::move(message));
        }
        if (status != StreamStatus::ok)
        {
            record_drop(frame.header.sequence);
            return StreamStatus::ok;
        }
        enqueued_.fetch_add(1, std::memory_order_relaxed);
        observe_queue_size();
        notify_worker();
        return StreamStatus::ok;
    }

    StreamStatus handle_discontinuity(const Discontinuity& discontinuity) noexcept override
    {
        if (!accepting_.load(std::memory_order_acquire))
        {
            record_drop(discontinuity.actual_frame_sequence);
            return StreamStatus::ok;
        }
        if (drop_policy_ == ObserverDropPolicy::latest_value)
        {
            while (discard_oldest())
            {
            }
        }

        DiscontinuityLease copy;
        auto status = discontinuity_pool_->try_acquire(copy);
        if (status != StreamStatus::ok && drop_policy_ != ObserverDropPolicy::drop_newest &&
            discard_oldest())
        {
            status = discontinuity_pool_->try_acquire(copy);
        }
        if (status != StreamStatus::ok)
        {
            record_drop(discontinuity.actual_frame_sequence);
            return StreamStatus::ok;
        }
        status = copy.assign(discontinuity.session_id, discontinuity.previous_frame_sequence,
                             discontinuity.actual_frame_sequence, discontinuity.reason,
                             discontinuity.signal_gaps);
        if (status != StreamStatus::ok)
        {
            static_cast<void>(copy.reset());
            record_drop(discontinuity.actual_frame_sequence);
            return StreamStatus::ok;
        }

        auto message = StreamMessage::from_discontinuity(std::move(copy));
        status = queue_->try_push(std::move(message));
        if (status == StreamStatus::queue_overflow &&
            drop_policy_ != ObserverDropPolicy::drop_newest && discard_oldest())
        {
            status = queue_->try_push(std::move(message));
        }
        if (status != StreamStatus::ok)
        {
            record_drop(discontinuity.actual_frame_sequence);
            return StreamStatus::ok;
        }
        enqueued_.fetch_add(1, std::memory_order_relaxed);
        observe_queue_size();
        notify_worker();
        return StreamStatus::ok;
    }

    StreamStatus flush() noexcept override
    {
        request_close(true);
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        // Graceful: drain frames already queued through the callback before
        // re-arming, mirroring close(). A forceful reset makes the worker bail at
        // the loop top and drop every queued-but-undelivered frame in drain_pending,
        // so a run() immediately followed by reset() loses that run's frames
        // before they are ever delivered. join_worker() releases the GIL while it
        // waits, so the worker can acquire it to drain without deadlocking.
        request_close(true);
        join_worker();
        release_resources();
        clear_stats();
        initialize_drop_history();
        try
        {
            initialize_resources();
            accepting_.store(true, std::memory_order_release);
            graceful_close_.store(false, std::memory_order_relaxed);
            worker_done_.store(false, std::memory_order_relaxed);
            start_worker();
        }
        catch (...)
        {
            accepting_.store(false, std::memory_order_release);
            worker_done_.store(true, std::memory_order_release);
            return StreamStatus::buffer_exhausted;
        }
        return StreamStatus::ok;
    }

    void cancel() noexcept override
    {
        request_close(false);
    }

    // Graceful: let the worker drain frames already queued through the callback
    // before stopping. A forceful close (request_close(false)) makes the worker
    // bail at the loop top and drop every queued-but-undelivered frame, which
    // breaks the contract callers rely on -- that after run() + close() the
    // enqueued frames have been delivered. Forceful teardown is still performed
    // by the destructor (and cancel()), which must not deliver into an
    // interpreter that may already be finalizing.
    void close() noexcept
    {
        request_close(true);
        join_worker();
    }

    [[nodiscard]] PythonObserverBridgeStats stats() const noexcept
    {
        return PythonObserverBridgeStats{
            .enqueued = enqueued_.load(std::memory_order_relaxed),
            .delivered = delivered_.load(std::memory_order_relaxed),
            .frames_delivered = frames_delivered_.load(std::memory_order_relaxed),
            .discontinuities_delivered = discontinuities_delivered_.load(std::memory_order_relaxed),
            .dropped = dropped_.load(std::memory_order_relaxed),
            .drop_range_count =
                std::min(drop_claims_.load(std::memory_order_acquire), drop_history_capacity_),
            .drop_history_dropped = drop_history_dropped_.load(std::memory_order_relaxed),
            .callback_errors = callback_errors_.load(std::memory_order_relaxed),
            .high_water_mark = high_water_mark_.load(std::memory_order_relaxed),
            .queue_size = queue_ == nullptr ? 0 : queue_->approximate_size(),
            .native_frames_outstanding = frame_pool_ == nullptr ? 0 : frame_pool_->outstanding(),
            .native_discontinuities_outstanding =
                discontinuity_pool_ == nullptr ? 0 : discontinuity_pool_->outstanding(),
            .callback_active = callback_active_.load(std::memory_order_acquire),
            .closed = !accepting_.load(std::memory_order_acquire),
            .worker_done = worker_done_.load(std::memory_order_acquire),
        };
    }

    [[nodiscard]] std::vector<ObserverDropRange> drop_ranges() const
    {
        const auto count =
            std::min(drop_claims_.load(std::memory_order_acquire), drop_history_capacity_);
        std::vector<ObserverDropRange> ranges;
        ranges.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            if (drop_history_[i].ready.load(std::memory_order_acquire) != 0)
            {
                ranges.push_back(drop_history_[i].range);
            }
        }
        return ranges;
    }

    [[nodiscard]] std::optional<std::string> callback_error() const
    {
        const auto size = callback_error_size_.load(std::memory_order_acquire);
        return size == 0 ? std::nullopt
                         : std::optional<std::string>{std::string{callback_error_.data(), size}};
    }

  private:
    struct DropSlot
    {
        std::atomic<std::uint8_t> ready{};
        ObserverDropRange range{};
    };

    void initialize_resources()
    {
        frame_pool_ = std::make_unique<FramePool>(capacity_, payload_bytes_, max_signal_blocks_);
        discontinuity_pool_ =
            std::make_unique<DiscontinuityPool>(capacity_, gaps_per_discontinuity_);
        queue_ = std::make_unique<ObserverQueue<StreamMessage>>(capacity_);
    }

    void release_resources() noexcept
    {
        queue_.reset();
        discontinuity_pool_.reset();
        frame_pool_.reset();
    }

    void initialize_drop_history() noexcept
    {
        for (std::size_t i = 0; i < drop_history_capacity_; ++i)
        {
            drop_history_[i].ready.store(0, std::memory_order_relaxed);
            drop_history_[i].range = {};
        }
    }

    void clear_stats() noexcept
    {
        enqueued_.store(0, std::memory_order_relaxed);
        delivered_.store(0, std::memory_order_relaxed);
        frames_delivered_.store(0, std::memory_order_relaxed);
        discontinuities_delivered_.store(0, std::memory_order_relaxed);
        dropped_.store(0, std::memory_order_relaxed);
        drop_claims_.store(0, std::memory_order_relaxed);
        drop_history_dropped_.store(0, std::memory_order_relaxed);
        callback_errors_.store(0, std::memory_order_relaxed);
        high_water_mark_.store(0, std::memory_order_relaxed);
        callback_active_.store(false, std::memory_order_relaxed);
        callback_error_size_.store(0, std::memory_order_relaxed);
    }

    void start_worker()
    {
        worker_ = std::thread(&PythonObserverBridge::worker_loop, this);
    }

    void join_worker() noexcept
    {
        if (!worker_.joinable())
        {
            return;
        }
        if (Py_IsInitialized() != 0 && PyGILState_Check() != 0)
        {
            py::gil_scoped_release release;
            worker_.join();
            return;
        }
        worker_.join();
    }

    void request_close(bool graceful) noexcept
    {
        accepting_.store(false, std::memory_order_release);
        if (queue_ == nullptr)
        {
            return;
        }
        if (!graceful)
        {
            graceful_close_.store(false, std::memory_order_release);
        }
        else if (!queue_->closed())
        {
            graceful_close_.store(true, std::memory_order_release);
        }
        queue_->close();
        notify_worker();
    }

    void notify_worker() noexcept
    {
        epoch_.fetch_add(1, std::memory_order_release);
        epoch_.notify_one();
    }

    [[nodiscard]] static std::uint64_t message_sequence(const StreamMessage& message) noexcept
    {
        return message.kind() == StreamMessageKind::frame
                   ? message.frame().header.sequence
                   : message.discontinuity().actual_frame_sequence;
    }

    void record_drop(std::uint64_t sequence) noexcept
    {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        const auto idx = drop_claims_.fetch_add(1, std::memory_order_relaxed);
        if (idx >= drop_history_capacity_)
        {
            drop_history_dropped_.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        drop_history_[idx].range = {sequence, sequence};
        drop_history_[idx].ready.store(1, std::memory_order_release);
    }

    [[nodiscard]] bool discard_oldest() noexcept
    {
        StreamMessage dropped;
        if (queue_->try_pop(dropped) != StreamStatus::ok)
        {
            return false;
        }
        record_drop(message_sequence(dropped));
        return true;
    }

    void observe_queue_size() noexcept
    {
        const auto size = queue_->approximate_size();
        auto current = high_water_mark_.load(std::memory_order_relaxed);
        while (current < size &&
               !high_water_mark_.compare_exchange_weak(current, size, std::memory_order_relaxed,
                                                       std::memory_order_relaxed))
        {
        }
    }

    [[nodiscard]] static PythonObserverFrame make_frame(FrameView frame)
    {
        return python::copy_frame(frame);
    }

    [[nodiscard]] static PythonObserverDiscontinuity make_discontinuity(const Discontinuity& value)
    {
        return python::copy_discontinuity(value);
    }

    void record_callback_error(std::string_view message) noexcept
    {
        callback_errors_.fetch_add(1, std::memory_order_relaxed);
        const auto size = std::min(message.size(), callback_error_.size());
        std::copy_n(message.data(), size, callback_error_.data());
        callback_error_size_.store(size, std::memory_order_release);
        request_close(false);
    }

    [[nodiscard]] bool deliver(StreamMessage& message) noexcept
    {
        try
        {
            py::gil_scoped_acquire acquire;
            py::object snapshot;
            const auto kind = message.kind();
            if (kind == StreamMessageKind::frame)
            {
                snapshot = py::cast(make_frame(message.frame()));
            }
            else
            {
                snapshot = py::cast(make_discontinuity(message.discontinuity()));
            }
            message = StreamMessage{};
            callback_active_.store(true, std::memory_order_release);
            callback_(std::move(snapshot));
            callback_active_.store(false, std::memory_order_release);
            delivered_.fetch_add(1, std::memory_order_relaxed);
            if (kind == StreamMessageKind::frame)
            {
                frames_delivered_.fetch_add(1, std::memory_order_relaxed);
            }
            else
            {
                discontinuities_delivered_.fetch_add(1, std::memory_order_relaxed);
            }
            return true;
        }
        catch (const py::error_already_set& error)
        {
            message = StreamMessage{};
            callback_active_.store(false, std::memory_order_release);
            record_callback_error(error.what());
        }
        catch (const std::exception& error)
        {
            message = StreamMessage{};
            callback_active_.store(false, std::memory_order_release);
            record_callback_error(error.what());
        }
        catch (...)
        {
            message = StreamMessage{};
            callback_active_.store(false, std::memory_order_release);
            record_callback_error("unknown Python observer callback failure");
        }
        return false;
    }

    void drain_pending() noexcept
    {
        StreamMessage message;
        while (queue_->try_pop(message) == StreamStatus::ok)
        {
            record_drop(message_sequence(message));
            message = StreamMessage{};
        }
    }

    void worker_loop() noexcept
    {
        StreamMessage message;
        for (;;)
        {
            if (!graceful_close_.load(std::memory_order_acquire) &&
                !accepting_.load(std::memory_order_acquire))
            {
                break;
            }
            const auto status = queue_->try_pop(message);
            if (status == StreamStatus::ok)
            {
                if (!deliver(message))
                {
                    break;
                }
                continue;
            }
            if (status == StreamStatus::stopped || (queue_->closed() && queue_->empty()))
            {
                break;
            }
            const auto epoch = epoch_.load(std::memory_order_acquire);
            if (queue_->empty() && !queue_->closed())
            {
                epoch_.wait(epoch, std::memory_order_acquire);
            }
        }
        drain_pending();
        worker_done_.store(true, std::memory_order_release);
        worker_done_.notify_all();
    }

    py::object callback_;
    const std::size_t capacity_;
    const std::size_t payload_bytes_;
    const std::size_t max_signal_blocks_;
    const std::size_t gaps_per_discontinuity_;
    const ObserverDropPolicy drop_policy_;
    const std::size_t drop_history_capacity_;
    std::unique_ptr<DropSlot[]> drop_history_;
    std::unique_ptr<FramePool> frame_pool_;
    std::unique_ptr<DiscontinuityPool> discontinuity_pool_;
    std::unique_ptr<ObserverQueue<StreamMessage>> queue_;
    std::thread worker_;
    std::array<char, 512> callback_error_{};
    std::atomic<std::size_t> callback_error_size_{};
    std::atomic<std::uint64_t> epoch_{};
    std::atomic<std::uint64_t> enqueued_{};
    std::atomic<std::uint64_t> delivered_{};
    std::atomic<std::uint64_t> frames_delivered_{};
    std::atomic<std::uint64_t> discontinuities_delivered_{};
    std::atomic<std::uint64_t> dropped_{};
    std::atomic<std::size_t> drop_claims_{};
    std::atomic<std::uint64_t> drop_history_dropped_{};
    std::atomic<std::uint64_t> callback_errors_{};
    std::atomic<std::uint64_t> high_water_mark_{};
    std::atomic<bool> accepting_{true};
    std::atomic<bool> graceful_close_{true};
    std::atomic<bool> callback_active_{};
    std::atomic<bool> worker_done_{};
};

// --- interpreter-shutdown safety -------------------------------------------
// Each PythonObserverBridge owns a worker std::thread that touches the CPython
// API (GIL acquire, refcounting, callback invocation) on every delivered frame.
// That worker must be joined BEFORE the interpreter begins tearing itself down
// in Py_FinalizeEx. If it outlives that boundary, it can refcount/allocate
// Python objects while the runtime is in its "finalizing" state with the GIL
// released -- which aborts the process (see the
// test_bridge_destruction_during_interpreter_shutdown regression, which used to
// flake with "terminate called without an active exception",
// "_PyMem_DebugFree ... without holding the GIL", and "FATAL: exception not
// rethrown").
//
// CPython runs atexit-module callbacks very early in Py_FinalizeEx, while the
// interpreter is still fully alive and the GIL is held -- before module clearing
// and the GIL/thread-state teardown. Registering an atexit hook that joins every
// still-live bridge worker therefore guarantees no worker is ever left running
// once the interpreter enters the unsafe finalizing phase.

struct LiveBridgeRegistry
{
    std::mutex mutex;
    std::unordered_set<PythonObserverBridge*> bridges;
};

LiveBridgeRegistry& live_bridges()
{
    static LiveBridgeRegistry registry;
    return registry;
}

void register_live_bridge(PythonObserverBridge* bridge)
{
    std::lock_guard<std::mutex> lock(live_bridges().mutex);
    live_bridges().bridges.insert(bridge);
}

void unregister_live_bridge(PythonObserverBridge* bridge) noexcept
{
    std::lock_guard<std::mutex> lock(live_bridges().mutex);
    live_bridges().bridges.erase(bridge);
}

void close_all_live_bridges()
{
    auto& reg = live_bridges();
    std::vector<PythonObserverBridge*> snapshot;
    {
        std::lock_guard<std::mutex> lock(reg.mutex);
        snapshot.assign(reg.bridges.begin(), reg.bridges.end());
    }
    // Join outside the registry lock: close() blocks on the worker, and the
    // worker's delivery path never touches the registry, so there is no
    // deadlock risk. Bridges stay alive throughout atexit (they are still
    // referenced from Python), so the raw pointers in the snapshot remain valid.
    for (auto* bridge : snapshot)
    {
        bridge->close();
    }
}

void register_bridge_shutdown_hook()
{
    static std::once_flag once;
    std::call_once(once,
                   []
                   {
                       try
                       {
                           py::module_::import("atexit").attr("register")(
                               py::cpp_function([] { close_all_live_bridges(); }));
                       }
                       catch (...)
                       {
                           // Best-effort: an embedded interpreter without the atexit module simply
                           // falls back to relying on explicit close()/destruction. Never fail
                           // module import over the shutdown hook.
                       }
                   });
}

} // namespace

void bind_python_observer_bridge(py::module_& module)
{
    register_bridge_shutdown_hook();
    py::enum_<GapReason>(module, "GapReason")
        .value("FRAME_SEQUENCE_GAP", GapReason::frame_sequence_gap)
        .value("SAMPLE_GAP", GapReason::sample_gap)
        .value("DEVICE_TICK_GAP", GapReason::device_tick_gap)
        .value("DEVICE_RESTART", GapReason::device_restart)
        .value("SOURCE_GAP", GapReason::source_gap)
        .value("BUFFER_EXHAUSTED", GapReason::buffer_exhausted)
        .value("QUEUE_OVERFLOW", GapReason::queue_overflow);

    py::class_<PythonObserverSignalBlock>(module, "PythonObserverSignalBlock", py::is_final())
        .def(py::init<SampleIndex, DeviceTick, std::uint64_t, std::uint64_t, SignalId,
                      std::uint32_t, ClockSyncSnapshot, HostTimeNs, SampleIndex>(),
             py::arg("sample_idx_start"), py::arg("device_tick_start"), py::arg("payload_offset"),
             py::arg("payload_byte_count"), py::arg("signal_id"), py::arg("n_samples"),
             py::arg("clock_sync") = ClockSyncSnapshot{}, py::arg("observation_time_start_ns") = 0,
             py::arg("last_sample_idx") = 0)
        .def_readonly("sample_idx_start", &PythonObserverSignalBlock::sample_idx_start)
        .def_readonly("last_sample_idx", &PythonObserverSignalBlock::last_sample_idx)
        .def_readonly("device_tick_start", &PythonObserverSignalBlock::device_tick_start)
        .def_readonly("observation_time_start_ns",
                      &PythonObserverSignalBlock::observation_time_start_ns)
        .def_readonly("payload_offset", &PythonObserverSignalBlock::payload_offset)
        .def_readonly("payload_byte_count", &PythonObserverSignalBlock::payload_byte_count)
        .def_readonly("signal_id", &PythonObserverSignalBlock::signal_id)
        .def_readonly("n_samples", &PythonObserverSignalBlock::n_samples)
        .def_readonly("clock_sync", &PythonObserverSignalBlock::clock_sync);

    py::class_<PythonObserverFrame>(module, "PythonObserverFrame", py::is_final())
        .def(py::init<SessionId, std::uint64_t, HostTimeNs, std::optional<DeviceTick>,
                      std::optional<HostTimeNs>, SchemaId, ClockDomainId,
                      std::vector<PythonObserverSignalBlock>, py::array_t<std::uint8_t>>(),
             py::arg("session_id"), py::arg("sequence"), py::arg("host_received_ns"),
             py::arg("source_tick") = py::none(), py::arg("valid_until_ns") = py::none(),
             py::arg("schema_id") = 0, py::arg("source_clock_domain") = 0, py::arg("blocks"),
             py::arg("payload"))
        .def_readonly("session_id", &PythonObserverFrame::session_id)
        .def_readonly("sequence", &PythonObserverFrame::sequence)
        .def_readonly("host_received_ns", &PythonObserverFrame::host_received_ns)
        .def_readonly("source_tick", &PythonObserverFrame::source_tick)
        .def_readonly("valid_until_ns", &PythonObserverFrame::valid_until_ns)
        .def_readonly("schema_id", &PythonObserverFrame::schema_id)
        .def_readonly("source_clock_domain", &PythonObserverFrame::source_clock_domain)
        .def_readonly("blocks", &PythonObserverFrame::blocks)
        .def_readonly("payload", &PythonObserverFrame::payload);

    py::class_<PythonObserverSignalGap>(module, "PythonObserverSignalGap", py::is_final())
        .def_readonly("expected_sample_idx", &PythonObserverSignalGap::expected_sample_idx)
        .def_readonly("actual_sample_idx", &PythonObserverSignalGap::actual_sample_idx)
        .def_readonly("missing_samples", &PythonObserverSignalGap::missing_samples)
        .def_readonly("expected_device_tick", &PythonObserverSignalGap::expected_device_tick)
        .def_readonly("actual_device_tick", &PythonObserverSignalGap::actual_device_tick)
        .def_readonly("signal_id", &PythonObserverSignalGap::signal_id)
        .def_readonly("reason", &PythonObserverSignalGap::reason);

    py::class_<PythonObserverDiscontinuity>(module, "PythonObserverDiscontinuity", py::is_final())
        .def_readonly("session_id", &PythonObserverDiscontinuity::session_id)
        .def_readonly("previous_frame_sequence",
                      &PythonObserverDiscontinuity::previous_frame_sequence)
        .def_readonly("actual_frame_sequence", &PythonObserverDiscontinuity::actual_frame_sequence)
        .def_readonly("reason", &PythonObserverDiscontinuity::reason)
        .def_readonly("signal_gaps", &PythonObserverDiscontinuity::signal_gaps);

    py::class_<PythonObserverBridgeStats>(module, "PythonObserverBridgeStats", py::is_final())
        .def_readonly("enqueued", &PythonObserverBridgeStats::enqueued)
        .def_readonly("delivered", &PythonObserverBridgeStats::delivered)
        .def_readonly("frames_delivered", &PythonObserverBridgeStats::frames_delivered)
        .def_readonly("discontinuities_delivered",
                      &PythonObserverBridgeStats::discontinuities_delivered)
        .def_readonly("dropped", &PythonObserverBridgeStats::dropped)
        .def_readonly("drop_range_count", &PythonObserverBridgeStats::drop_range_count)
        .def_readonly("drop_history_dropped", &PythonObserverBridgeStats::drop_history_dropped)
        .def_readonly("callback_errors", &PythonObserverBridgeStats::callback_errors)
        .def_readonly("high_water_mark", &PythonObserverBridgeStats::high_water_mark)
        .def_readonly("queue_size", &PythonObserverBridgeStats::queue_size)
        .def_readonly("native_frames_outstanding",
                      &PythonObserverBridgeStats::native_frames_outstanding)
        .def_readonly("native_discontinuities_outstanding",
                      &PythonObserverBridgeStats::native_discontinuities_outstanding)
        .def_readonly("callback_active", &PythonObserverBridgeStats::callback_active)
        .def_readonly("closed", &PythonObserverBridgeStats::closed)
        .def_readonly("worker_done", &PythonObserverBridgeStats::worker_done);

    py::class_<PythonObserverBridge, NativeObserver> bridge(module, "PythonObserverBridge",
                                                            py::is_final());
    bridge
        .def(py::init<py::object, const StreamSchema&, std::size_t, ObserverDropPolicy,
                      std::size_t>(),
             py::arg("callback"), py::arg("schema"), py::arg("capacity") = 8,
             py::arg("drop_policy") = ObserverDropPolicy::drop_oldest,
             py::arg("drop_history_capacity") = 32)
        .def("close", &PythonObserverBridge::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("stats", &PythonObserverBridge::stats)
        .def_property_readonly("drop_ranges", &PythonObserverBridge::drop_ranges)
        .def_property_readonly("callback_error", &PythonObserverBridge::callback_error);
}
