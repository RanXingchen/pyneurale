/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// `NativeReplaySource`: a native streaming source whose device is a committed
/// recording.
///
/// It consumes one validated replay image and nothing else. Everything the run
/// emits -- the items, their order, the replay-run frame sequence, the block
/// metadata, the payload bytes, the omission and fidelity reports -- was
/// decided when the image was built (contract section 8.1-8.3), so this class
/// decides only what belongs to a *run*: when each item is emitted (section
/// 8.7), whether the caller has granted a permit for it (section 8.8), whether
/// the run was cancelled or has ended (section 8.9), where it restarts (section
/// 8.10), which injected faults fire (section 8.11), and the `SessionId` the
/// emitted messages carry (section 8.12).
///
/// That split is what keeps the steady-state read honest. After `prepare()`
/// returns, a read does no allocation, no filesystem metadata work, no JSON or
/// Zarr parsing, no logging, and never enters Python: it indexes fixed-width
/// records in a mapped file, copies payload bytes into the caller's frame, and
/// returns. Resolution of fault identities, gap-reason strings, scratch sizing,
/// and every bound that depends on the image happen once, in `prepare()`.
///
/// The direction of the dependency is one-way and stays that way: this target
/// consumes `neurale::streaming`'s frame, discontinuity, schema, and clock
/// types, and `neurale::streaming` does not know it exists.

#include "replay/image.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

#include <neurale/streaming/clock.h>
#include <neurale/streaming/discontinuity.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/schema.h>
#include <neurale/streaming/source.h>

namespace neurale::recording
{

/// The three mutually exclusive pacing modes of contract section 8.7.
enum class ReplayPacing : std::uint8_t
{
    /// Emit as soon as the consumer takes the previous item; never wait.
    as_fast_as_possible,
    /// Emit at the item's position on the recorded timeline, scaled.
    recorded,
    /// Emit only when the caller has granted a permit (section 8.8).
    step,
};

/// The four declared injected effects of contract section 8.11.
enum class ReplayFaultEffect : std::uint8_t
{
    stall,
    read_failure,
    sequence_gap,
    abnormal_end,
};

/// Terminal state of a run. Terminal states are sticky and do not downgrade;
/// only reset clears one (contract section 8.9).
enum class ReplayTerminal : std::uint8_t
{
    /// The run has not ended.
    none,
    /// The items ran out. The natural end, reported exactly once.
    end_of_data,
    /// cancel() was called.
    cancelled,
    /// An incomplete session's committed prefix ended, or an `abnormal_end`
    /// fault fired. Distinct from end_of_data (sections 8.5 and 8.11).
    abnormal_end,
    /// A `read_failure` fault fired, or the image asked for something this run
    /// cannot deliver (a frame larger than the pool slot, say).
    faulted,
};

/// Why a run could not be prepared. Every value is a refusal, never a
/// degradation: a replay that quietly did something else would be worse than
/// one that would not start.
enum class ReplayConfigStatus : std::uint8_t
{
    ok,
    /// No image is open.
    image_not_open,
    /// Another read, reset, or prepare holds the source. Reported rather than
    /// raced: prepare rewrites the state a read is standing on.
    operation_in_flight,
    /// The run was asked to carry the recording's own `SessionId`, which is the
    /// one identity a replay must never present (contract section 8.12).
    replay_session_id_conflict,
    /// The image declares a schema value -- a dtype, a layout, a signal kind --
    /// that this build does not know. Distinct from `schema_mismatch`: the
    /// schemas are not known to differ, they are not comparable at all.
    unsupported_schema_value,
    /// Preparation could not allocate. Nothing was changed.
    out_of_memory,
    /// `speed_factor` was given for a mode that is not `recorded`.
    speed_factor_rejected,
    /// `speed_factor` is not finite and in (0, 1000].
    invalid_speed_factor,
    /// A `sequence_gap` names a discontinuity. Dropping a discontinuity makes
    /// no frame-sequence hole, so the effect would not happen.
    sequence_gap_targets_discontinuity,
    /// A `sequence_gap` names the run's first or last emitted frame, so there
    /// is no frame on one side of it and no observable jump.
    sequence_gap_at_run_boundary,
    /// Two faults name the same item.
    duplicate_fault_target,
    /// A `stall` declared no finite positive duration.
    stall_bound_missing,
    /// A gap or discontinuity names a reason this build does not know.
    unknown_gap_reason,
    /// The image's declared schema is not the one the runtime was built for.
    schema_mismatch,
};

[[nodiscard]] const char* replay_config_status_text(ReplayConfigStatus status) noexcept;
[[nodiscard]] const char* replay_terminal_text(ReplayTerminal terminal) noexcept;

/// A fault's position, as a recorded identity and never a run-local one
/// (contract section 8.11). `stream_id` is borrowed for the duration of
/// `prepare()` only.
struct ReplayFaultTarget
{
    ReplayItemKind kind{ReplayItemKind::frame};
    /// True for `exact_frames` and `recorded_projection`, where the identity is
    /// a data-message ordinal; false for `stream_frames`, where it is a stream
    /// plus a block or discontinuity-record ordinal.
    bool ledger_based{true};
    std::uint64_t data_message_ordinal{};
    std::string_view stream_id{};
    std::uint64_t ordinal{};
};

struct ReplayFaultSpec
{
    ReplayFaultTarget target{};
    ReplayFaultEffect effect{ReplayFaultEffect::stall};
    /// Required and positive for `stall`; ignored otherwise.
    std::uint64_t stall_ns{};
};

/// What one injected fault did, in the shape section 8.11 freezes. `emitted`
/// is meaningless -- the frozen schema's null -- when the fault did not fire.
struct ReplayFaultReport
{
    ReplayFaultEffect effect{ReplayFaultEffect::stall};
    bool fired{};
    bool emitted{};
    bool emitted_known{};
    /// Position in the image's item list, or the absent marker when the
    /// identity lies outside what this image carries.
    std::uint64_t item_idx{kReplayAbsentU64};
};

struct ReplaySourceConfig
{
    ReplayPacing pacing{ReplayPacing::as_fast_as_possible};
    /// Only legal with `recorded`; finite and in (0, 1000].
    double speed_factor{1.0};
    /// Zero asks the source to allocate one. An explicit value equal to the
    /// recording's own `SessionId` is refused rather than quietly replaced --
    /// the caller asked for something the contract forbids, and substituting a
    /// different identity behind its back would hide that.
    streaming::SessionId replay_run_session_id{};
    /// Whether a read waits for a deadline or a permit, or reports
    /// would_block. Neither form spins (contract section 8.8).
    bool blocking{true};
    /// The bound on how long a wait can ignore a cancellation, independent of
    /// the remaining pacing interval or the arrival of the next permit.
    std::uint64_t wake_latency_ns{1'000'000};
    /// Keeping a recorded deadline is a test behaviour and never the default:
    /// a recorded `valid_until_ns` is stale by construction (section 8.6).
    bool preserve_recorded_valid_until{};
    /// Null uses the process's default monotonic clock.
    streaming::NativeClock* clock{};
    /// Borrowed for the duration of prepare() only.
    std::span<const ReplayFaultSpec> faults{};
};

/// Run counters. Every counter is stored atomically, so a snapshot may be taken
/// while a read is in flight; the counters are read one at a time, so such a
/// snapshot is a set of individually-valid counters and not a transaction.
struct ReplayStats
{
    std::uint64_t items_emitted{};
    std::uint64_t frames_emitted{};
    std::uint64_t discontinuities_emitted{};
    /// Items whose deadline had already passed when they were emitted, and the
    /// worst such lateness. Reported rather than hidden (section 8.7).
    std::uint64_t late_item_count{};
    std::uint64_t max_lateness_ns{};
    /// Recorded deltas that ran backwards and were clamped to zero.
    std::uint64_t negative_delta_count{};
    /// The total delay injected by fired `stall` faults.
    std::uint64_t injected_delay_ns{};
    std::uint64_t permits_granted{};
    std::uint64_t permits_consumed{};
    /// advance() was called at UINT64_MAX permits and granted nothing.
    bool permit_saturated{};
};

class NativeReplaySource final : public streaming::NativeFrameSource
{
  public:
    NativeReplaySource() noexcept;
    ~NativeReplaySource() override;

    NativeReplaySource(const NativeReplaySource&) = delete;
    NativeReplaySource& operator=(const NativeReplaySource&) = delete;
    NativeReplaySource(NativeReplaySource&&) = delete;
    NativeReplaySource& operator=(NativeReplaySource&&) = delete;

    /// Open and validate a replay image. Control plane only, and -- unlike
    /// prepare(), reset(), and the reads -- *not* guarded: it replaces the
    /// mapping every other member reads from, so it belongs with construction
    /// and destruction and must not run concurrently with anything else.
    [[nodiscard]] ReplayImageStatus open(const char* path) noexcept;

    /// Resolve the run: validate the configuration, resolve every fault
    /// identity against the image, size the scratch a read needs, and allocate
    /// the run's SessionId. Everything a read could otherwise have to do
    /// happens here. Control plane only.
    ///
    /// Transactional: a call that returns anything but `ok` changes nothing at
    /// all, so a source that was runnable stays runnable under exactly its
    /// previous configuration, and one that was not stays unprepared. A
    /// half-applied configuration would be a run nobody asked for.
    [[nodiscard]] ReplayConfigStatus prepare(const ReplaySourceConfig& config) noexcept;

    /// Check the image's declared schema -- every signal field, every
    /// feature-set descriptor, every unit -- against the one the runtime was
    /// configured with, using `streaming::equivalent`, which is the same
    /// definition `StreamSchema::equivalent()` is built from. Separate from
    /// prepare() because a source driven directly, without a runtime, has no
    /// schema to check against. Control plane only: it allocates.
    [[nodiscard]] ReplayConfigStatus
    check_schema(const streaming::StreamSchema& schema) const noexcept;

    [[nodiscard]] streaming::StreamStatus read(streaming::MutableFrame& frame) noexcept override;

    [[nodiscard]] streaming::StreamStatus
    read_message(streaming::MutableFrame& frame,
                 streaming::DiscontinuityLease& discontinuity) noexcept override;

    [[nodiscard]] bool produces_discontinuities() const noexcept override
    {
        return true;
    }

    void cancel() noexcept override;

    /// Return to the start of the same run (contract section 8.10). Rejected
    /// with invalid_state while a read is in flight rather than raced: reset
    /// and the reads compete for one gate, so whichever arrives first wins and
    /// the other is refused. There is no window in which both are inside.
    [[nodiscard]] streaming::StreamStatus reset() noexcept override;

    /// Add *permits* to the running total, returning how many were granted.
    /// The counter saturates at UINT64_MAX and a call that grants nothing is
    /// reported rather than silently accepted (contract section 8.8).
    std::uint64_t advance(std::uint64_t permits) noexcept;

    [[nodiscard]] std::uint64_t permits() const noexcept
    {
        return permits_.load(std::memory_order_acquire);
    }

    [[nodiscard]] ReplayTerminal terminal() const noexcept
    {
        return terminal_.load(std::memory_order_acquire);
    }

    /// The run's identity, carried by every frame and every discontinuity it
    /// emits, and kept across reset (contract section 8.12). Never zero and
    /// never the recording's own.
    [[nodiscard]] streaming::SessionId session_id() const noexcept
    {
        return session_id_.load(std::memory_order_acquire);
    }

    /// Run counters, readable from any thread (see `ReplayStats`).
    [[nodiscard]] ReplayStats stats() const noexcept;

    /// How many faults this run was configured with, fired or not.
    [[nodiscard]] std::size_t fault_count() const noexcept
    {
        return resolved_.size();
    }

    /// One fault's report, by value. A report is composed from the immutable
    /// resolution and one atomic word the reading thread updates, so it can be
    /// asked for while a read is in flight; an out-of-range index yields a
    /// default report.
    [[nodiscard]] ReplayFaultReport fault_report(std::size_t idx) const noexcept;

    [[nodiscard]] const ReplayImageFile& image() const noexcept
    {
        return image_;
    }

    /// How many items the run emits, before injected effects.
    [[nodiscard]] std::size_t item_count() const noexcept
    {
        return image_.item_count();
    }

  private:
    /// One fault's resolution. Fixed by prepare() and never written again; what
    /// a run does to it lives in the parallel `fault_state_` word.
    struct ResolvedFault
    {
        std::uint64_t item_idx{kReplayAbsentU64};
        ReplayFaultEffect effect{ReplayFaultEffect::stall};
        std::uint64_t stall_ns{};
    };

    /// Bits of one fault's run state. One atomic byte rather than three plain
    /// bools, because the reading thread writes it while another thread may be
    /// asking for the report.
    static constexpr std::uint8_t kFaultFired = 1U << 0U;
    static constexpr std::uint8_t kFaultEmittedKnown = 1U << 1U;
    static constexpr std::uint8_t kFaultEmitted = 1U << 2U;

    /// Every counter of a run, in the storage a concurrent reader needs.
    struct AtomicStats
    {
        std::atomic<std::uint64_t> items_emitted{};
        std::atomic<std::uint64_t> frames_emitted{};
        std::atomic<std::uint64_t> discontinuities_emitted{};
        std::atomic<std::uint64_t> late_item_count{};
        std::atomic<std::uint64_t> max_lateness_ns{};
        std::atomic<std::uint64_t> negative_delta_count{};
        std::atomic<std::uint64_t> injected_delay_ns{};
        std::atomic<std::uint64_t> permits_consumed{};

        void clear() noexcept;
    };

    /// A read's own decision about the item it is looking at.
    enum class Gate : std::uint8_t
    {
        emit,
        skip,
        wait,
    };

    [[nodiscard]] streaming::StreamStatus terminal_status() const noexcept;
    [[nodiscard]] streaming::StreamStatus finish(ReplayTerminal terminal) noexcept;
    [[nodiscard]] bool wait_until(std::uint64_t deadline_ns) noexcept;
    [[nodiscard]] std::uint64_t deadline_for(const ReplayImageItem& item) noexcept;
    void advance_timeline(const ReplayImageItem& item) noexcept;
    [[nodiscard]] std::size_t fault_at(std::size_t idx) const noexcept;
    void report_fault(std::size_t fault, bool emitted) noexcept;
    [[nodiscard]] streaming::StreamStatus fill_frame(const ReplayImageItem& item,
                                                     streaming::MutableFrame& frame) noexcept;
    [[nodiscard]] streaming::StreamStatus
    fill_discontinuity(const ReplayImageItem& item,
                       streaming::DiscontinuityLease& discontinuity) noexcept;

    /// The body of a read, with the operation gate already held.
    [[nodiscard]] streaming::StreamStatus
    read_locked(streaming::MutableFrame& frame,
                streaming::DiscontinuityLease& discontinuity) noexcept;
    void reset_locked() noexcept;

    // Preparation, all of it writing into caller-owned temporaries so that a
    // refusal leaves the live configuration untouched.
    [[nodiscard]] ReplayConfigStatus resolve_faults(const ReplaySourceConfig& config,
                                                    std::vector<ResolvedFault>& resolved) const;
    [[nodiscard]] ReplayConfigStatus build_reason_table(std::vector<std::uint8_t>& table) const;
    [[nodiscard]] std::uint64_t resolve_target(const ReplayFaultTarget& target) const noexcept;
    [[nodiscard]] streaming::GapReason reason_of(std::uint32_t string_idx) const noexcept;

    ReplayImageFile image_;
    streaming::NativeClock* clock_{};
    ReplayPacing pacing_{ReplayPacing::as_fast_as_possible};
    double speed_factor_{1.0};
    bool blocking_{true};
    bool preserve_valid_until_{};
    std::uint64_t wake_latency_ns_{1'000'000};
    bool prepared_{};

    // Resolved once, indexed in the read loop.
    std::vector<streaming::SignalGap> gap_scratch_;
    std::vector<std::uint8_t> reason_table_;
    std::vector<ResolvedFault> resolved_;
    std::unique_ptr<std::atomic<std::uint8_t>[]> fault_state_;

    // Run state, owned by whichever thread holds the operation gate.
    std::size_t next_idx_{};
    std::uint64_t origin_ns_{};
    std::uint64_t scaled_offset_ns_{};
    std::uint64_t previous_timeline_ns_{};
    std::uint64_t pending_offset_ns_{};
    std::uint64_t deadline_ns_{};
    std::uint64_t stall_until_ns_{};
    bool anchored_{};
    bool has_previous_{};
    bool deadline_valid_{};
    bool stall_started_{};
    AtomicStats stats_{};

    /// Read by cancel() and advance(), which are the two entry points that must
    /// work *while* a read is in flight and so cannot take the gate.
    std::atomic<streaming::NativeClock*> wake_clock_{};
    std::atomic<streaming::SessionId> session_id_{};
    std::atomic<ReplayTerminal> terminal_{ReplayTerminal::none};
    std::atomic<std::uint64_t> permits_{};
    std::atomic<std::uint64_t> permits_granted_{};
    std::atomic<bool> permit_saturated_{};

    /// The one gate every read, reset, and prepare passes through. Held by
    /// exactly one thread at a time; a caller that cannot take it is refused,
    /// never queued and never let in alongside (contract section 8.10).
    std::atomic<bool> gate_{};
};

} // namespace neurale::recording
