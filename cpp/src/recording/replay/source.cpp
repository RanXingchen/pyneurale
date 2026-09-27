/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "replay/source.h"

#include "replay/schema_names.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace neurale::recording
{
namespace
{

using streaming::DiscontinuityLease;
using streaming::FrameFlags;
using streaming::GapReason;
using streaming::MutableFrame;
using streaming::SignalGap;
using streaming::SignalGapFlags;
using streaming::StreamStatus;

constexpr std::uint64_t kUint64Max = std::numeric_limits<std::uint64_t>::max();
constexpr std::uint8_t kUnknownReason = 0xFFU;
constexpr double kMaxSpeedFactor = 1000.0;
constexpr std::size_t kNoFault = static_cast<std::size_t>(-1);

/// `GapReason`, by value. The names are the native enum's own, lower-cased,
/// which is exactly what a recorder writes into a discontinuity record --
/// the two engines must not spell one reason two ways.
constexpr std::string_view kGapReasonNames[] = {
    "frame_sequence_gap", "sample_gap",       "device_tick_gap", "device_restart",
    "source_gap",         "buffer_exhausted", "queue_overflow",
};
constexpr std::uint8_t kGapReasonCount = 7;

/// Non-zero run identities, handed out in order. A replay must never emit the
/// recording's own SessionId (contract section 8.12): a replay that did would
/// be indistinguishable from the original recording on the wire.
std::atomic<std::uint64_t> g_next_session_id{1};

/// Exclusive access to the run state, taken by every read, reset, and prepare.
///
/// Observing a flag and then acting on it is not a mutual-exclusion protocol:
/// two operations that both looked first would both proceed. One compare and
/// exchange is the whole gate, so whichever operation wins the exchange is the
/// only one inside, and the other is refused with a status.
class GateGuard
{
  public:
    explicit GateGuard(std::atomic<bool>& gate) noexcept : gate_(&gate)
    {
        bool expected = false;
        held_ = gate.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                             std::memory_order_acquire);
    }

    ~GateGuard()
    {
        if (held_)
        {
            gate_->store(false, std::memory_order_release);
        }
    }

    GateGuard(const GateGuard&) = delete;
    GateGuard& operator=(const GateGuard&) = delete;
    GateGuard(GateGuard&&) = delete;
    GateGuard& operator=(GateGuard&&) = delete;

    [[nodiscard]] bool held() const noexcept
    {
        return held_;
    }

  private:
    std::atomic<bool>* gate_;
    bool held_{};
};

[[nodiscard]] std::uint64_t saturating_add(std::uint64_t left, std::uint64_t right) noexcept
{
    return left > kUint64Max - right ? kUint64Max : left + right;
}

/// Counters are written only by the thread holding the gate and read by any
/// thread, so an atomic load/store pair is enough and a read-modify-write is
/// not. Relaxed: these are counters, they order nothing.
void bump(std::atomic<std::uint64_t>& counter, std::uint64_t amount = 1) noexcept
{
    counter.store(saturating_add(counter.load(std::memory_order_relaxed), amount),
                  std::memory_order_relaxed);
}

void raise_to(std::atomic<std::uint64_t>& counter, std::uint64_t value) noexcept
{
    if (value > counter.load(std::memory_order_relaxed))
    {
        counter.store(value, std::memory_order_relaxed);
    }
}

// --- schema value names ------------------------------------------------------
//
// The table lives in `schema_names.h`, because the conformance harness
// has to read the same spellings to rebuild a recorded schema from an image;
// a copy here would let the two disagree while each stayed consistent.

using names::dtype_from;
using names::layout_from;
using names::observation_timing_from;
using names::physical_unit_from;
using names::signal_kind_from;
using names::tick_tracking_from;
using names::timestamp_reference_from;

} // namespace

const char* replay_config_status_text(ReplayConfigStatus status) noexcept
{
    switch (status)
    {
    case ReplayConfigStatus::ok:
        return "ok";
    case ReplayConfigStatus::image_not_open:
        return "image_not_open";
    case ReplayConfigStatus::operation_in_flight:
        return "operation_in_flight";
    case ReplayConfigStatus::replay_session_id_conflict:
        return "replay_session_id_conflict";
    case ReplayConfigStatus::unsupported_schema_value:
        return "unsupported_schema_value";
    case ReplayConfigStatus::out_of_memory:
        return "out_of_memory";
    case ReplayConfigStatus::speed_factor_rejected:
        return "speed_factor_rejected";
    case ReplayConfigStatus::invalid_speed_factor:
        return "invalid_speed_factor";
    case ReplayConfigStatus::sequence_gap_targets_discontinuity:
        return "sequence_gap_targets_discontinuity";
    case ReplayConfigStatus::sequence_gap_at_run_boundary:
        return "sequence_gap_at_run_boundary";
    case ReplayConfigStatus::duplicate_fault_target:
        return "duplicate_fault_target";
    case ReplayConfigStatus::stall_bound_missing:
        return "stall_bound_missing";
    case ReplayConfigStatus::unknown_gap_reason:
        return "unknown_gap_reason";
    case ReplayConfigStatus::schema_mismatch:
        return "schema_mismatch";
    }
    return "unknown";
}

const char* replay_terminal_text(ReplayTerminal terminal) noexcept
{
    switch (terminal)
    {
    case ReplayTerminal::none:
        return "none";
    case ReplayTerminal::end_of_data:
        return "end_of_data";
    case ReplayTerminal::cancelled:
        return "cancelled";
    case ReplayTerminal::abnormal_end:
        return "abnormal_end";
    case ReplayTerminal::faulted:
        return "faulted";
    }
    return "unknown";
}

void NativeReplaySource::AtomicStats::clear() noexcept
{
    items_emitted.store(0, std::memory_order_relaxed);
    frames_emitted.store(0, std::memory_order_relaxed);
    discontinuities_emitted.store(0, std::memory_order_relaxed);
    late_item_count.store(0, std::memory_order_relaxed);
    max_lateness_ns.store(0, std::memory_order_relaxed);
    negative_delta_count.store(0, std::memory_order_relaxed);
    injected_delay_ns.store(0, std::memory_order_relaxed);
    permits_consumed.store(0, std::memory_order_relaxed);
}

NativeReplaySource::NativeReplaySource() noexcept = default;
NativeReplaySource::~NativeReplaySource() = default;

ReplayImageStatus NativeReplaySource::open(const char* path) noexcept
{
    // Everything resolved against the previous image goes with it, so nothing
    // between an open and the next prepare can report the old run's faults.
    prepared_ = false;
    resolved_.clear();
    fault_state_.reset();
    reason_table_.clear();
    gap_scratch_.clear();
    return image_.open(path);
}

// --- preparation -------------------------------------------------------------

GapReason NativeReplaySource::reason_of(std::uint32_t string_idx) const noexcept
{
    if (string_idx >= reason_table_.size())
    {
        return GapReason::source_gap;
    }
    const auto value = reason_table_[string_idx];
    return value == kUnknownReason ? GapReason::source_gap : static_cast<GapReason>(value);
}

ReplayConfigStatus NativeReplaySource::build_reason_table(std::vector<std::uint8_t>& table) const
{
    table.assign(image_.string_count(), kUnknownReason);
    const auto resolve = [this, &table](std::uint32_t idx)
    {
        if (idx >= table.size())
        {
            return false;
        }
        if (table[idx] != kUnknownReason)
        {
            return true;
        }
        const auto text = image_.string(idx);
        for (std::uint8_t value = 0; value < kGapReasonCount; ++value)
        {
            if (text == kGapReasonNames[value])
            {
                table[idx] = value;
                return true;
            }
        }
        return false;
    };

    // Only the reasons the image actually references have to be known; the
    // string table also holds stream ids, dtypes, and mode names.
    for (std::size_t i = 0; i < image_.item_count(); ++i)
    {
        const auto item = image_.item(i);
        if (item.kind != ReplayItemKind::discontinuity)
        {
            continue;
        }
        if (item.reason != kReplayAbsentU32 && !resolve(item.reason))
        {
            return ReplayConfigStatus::unknown_gap_reason;
        }
        for (std::uint64_t child = 0; child < item.n_children; ++child)
        {
            const auto gap = image_.gap(static_cast<std::size_t>(item.child_first + child));
            if (gap.reason != kReplayAbsentU32 && !resolve(gap.reason))
            {
                return ReplayConfigStatus::unknown_gap_reason;
            }
        }
    }
    return ReplayConfigStatus::ok;
}

std::uint64_t NativeReplaySource::resolve_target(const ReplayFaultTarget& target) const noexcept
{
    for (std::size_t i = 0; i < image_.item_count(); ++i)
    {
        const auto item = image_.item(i);
        if (item.kind != target.kind)
        {
            continue;
        }
        if (target.ledger_based)
        {
            if (item.data_message_ordinal == target.data_message_ordinal)
            {
                return i;
            }
            continue;
        }
        if (target.kind == ReplayItemKind::frame)
        {
            for (std::uint64_t child = 0; child < item.n_children; ++child)
            {
                const auto block = image_.block(static_cast<std::size_t>(item.child_first + child));
                if (block.source_block_ordinal == target.ordinal &&
                    image_.string(block.stream) == target.stream_id)
                {
                    return i;
                }
            }
            continue;
        }
        for (std::uint64_t child = 0; child < item.n_children; ++child)
        {
            const auto gap = image_.gap(static_cast<std::size_t>(item.child_first + child));
            if (gap.signal_gap_ordinal != target.ordinal)
            {
                continue;
            }
            // A synthesized discontinuity names its signal, and the fidelity
            // section is what maps a signal back to the stream it stands for.
            for (std::size_t entry = 0; entry < image_.fidelity_count(); ++entry)
            {
                const auto fidelity = image_.fidelity(entry);
                if (fidelity.native_signal_id == gap.native_signal_id &&
                    image_.string(fidelity.stream) == target.stream_id)
                {
                    return i;
                }
            }
        }
    }
    return kReplayAbsentU64;
}

ReplayConfigStatus NativeReplaySource::resolve_faults(const ReplaySourceConfig& config,
                                                      std::vector<ResolvedFault>& resolved) const
{
    resolved.clear();
    resolved.reserve(config.faults.size());

    // Pass one: duplicate detection is by recorded target identity, not by
    // resolved item index, so a target outside the image (which resolves to
    // kReplayAbsentU64) is still deduplicated. The Python ReplayConfig checks
    // by recorded identity too, and the native API must agree (contract
    // section 8.11): position is by recorded identity, not run-local
    // sequence; a target outside the image is kept as unfired, not dropped;
    // and two faults on one recorded position are always a duplicate.
    for (std::size_t fi = 0; fi < config.faults.size(); ++fi)
    {
        const auto& spec = config.faults[fi];
        if (spec.effect == ReplayFaultEffect::stall && spec.stall_ns == 0)
        {
            return ReplayConfigStatus::stall_bound_missing;
        }
        if (spec.effect == ReplayFaultEffect::sequence_gap &&
            spec.target.kind == ReplayItemKind::discontinuity)
        {
            return ReplayConfigStatus::sequence_gap_targets_discontinuity;
        }
        for (std::size_t pi = 0; pi < fi; ++pi)
        {
            const auto& prev = config.faults[pi].target;
            if (spec.target.kind == prev.kind && spec.target.ledger_based == prev.ledger_based &&
                (spec.target.ledger_based
                     ? spec.target.data_message_ordinal == prev.data_message_ordinal
                     : spec.target.stream_id == prev.stream_id &&
                           spec.target.ordinal == prev.ordinal))
            {
                return ReplayConfigStatus::duplicate_fault_target;
            }
        }
        const auto idx = resolve_target(spec.target);
        resolved.push_back(ResolvedFault{
            .item_idx = idx,
            .effect = spec.effect,
            .stall_ns = spec.stall_ns,
        });
    }

    // Pass two: a sequence_gap promises the consumer *sees a jump*, so the
    // frame before the target and the frame after it must both be emitted. A
    // target at either end of the run has no frame on one side, so there would
    // be no observable jump at all -- that is a different effect and would
    // need its own name. (Another injected effect that ends the run earlier is
    // the caller's own arrangement and is not second-guessed here.)
    for (const auto& fault : resolved)
    {
        if (fault.effect != ReplayFaultEffect::sequence_gap || fault.item_idx == kReplayAbsentU64)
        {
            continue;
        }
        bool before = false;
        bool after = false;
        for (std::size_t i = 0; i < image_.item_count(); ++i)
        {
            if (image_.item(i).kind != ReplayItemKind::frame)
            {
                continue;
            }
            bool skipped = false;
            for (const auto& other : resolved)
            {
                if (other.effect == ReplayFaultEffect::sequence_gap && other.item_idx == i)
                {
                    skipped = true;
                    break;
                }
            }
            if (skipped)
            {
                continue;
            }
            before = before || i < fault.item_idx;
            after = after || i > fault.item_idx;
        }
        if (!before || !after)
        {
            return ReplayConfigStatus::sequence_gap_at_run_boundary;
        }
    }
    return ReplayConfigStatus::ok;
}

ReplayConfigStatus NativeReplaySource::prepare(const ReplaySourceConfig& config) noexcept
{
    GateGuard guard{gate_};
    if (!guard.held())
    {
        return ReplayConfigStatus::operation_in_flight;
    }
    if (!image_.is_open())
    {
        return ReplayConfigStatus::image_not_open;
    }
    if (config.pacing != ReplayPacing::recorded)
    {
        // Rejected, not ignored: a speed factor that silently did nothing
        // would make a paced run and an unpaced one look alike in the request.
        if (config.speed_factor != 1.0)
        {
            return ReplayConfigStatus::speed_factor_rejected;
        }
    }
    else if (!std::isfinite(config.speed_factor) || config.speed_factor <= 0.0 ||
             config.speed_factor > kMaxSpeedFactor)
    {
        return ReplayConfigStatus::invalid_speed_factor;
    }

    // The run's identity is the run's, and the one value it may never take is
    // the recording's own (contract section 8.12). An explicit request for it
    // is refused: silently handing back a different identity would leave the
    // caller believing something untrue about the frames it is about to see.
    const auto recorded = image_.summary().native_session_id;
    const bool recorded_known = recorded != kReplayAbsentU64;
    auto session = config.replay_run_session_id;
    if (session != 0)
    {
        if (recorded_known && session == recorded)
        {
            return ReplayConfigStatus::replay_session_id_conflict;
        }
    }
    else
    {
        do
        {
            session = g_next_session_id.fetch_add(1, std::memory_order_relaxed);
        } while (session == 0 || (recorded_known && session == recorded));
    }

    // Everything below is built into temporaries. A refusal from here on must
    // leave the previous run exactly as it was -- a source that reported a bad
    // configuration and then kept running under a mixture of the old one and
    // the rejected one would be worse than either.
    std::vector<std::uint8_t> reason_table;
    std::vector<ResolvedFault> resolved;
    std::vector<SignalGap> gap_scratch;
    std::unique_ptr<std::atomic<std::uint8_t>[]> fault_state;
    try
    {
        const auto reasons = build_reason_table(reason_table);
        if (reasons != ReplayConfigStatus::ok)
        {
            return reasons;
        }
        const auto faults = resolve_faults(config, resolved);
        if (faults != ReplayConfigStatus::ok)
        {
            return faults;
        }

        // The widest discontinuity in the image sizes the scratch a read copies
        // gaps through, so a read never sizes anything.
        std::size_t widest = 0;
        for (std::size_t i = 0; i < image_.item_count(); ++i)
        {
            const auto item = image_.item(i);
            if (item.kind == ReplayItemKind::discontinuity)
            {
                widest = std::max<std::size_t>(widest, item.n_children);
            }
        }
        gap_scratch.assign(widest, SignalGap{});
        fault_state = std::make_unique<std::atomic<std::uint8_t>[]>(resolved.size());
    }
    catch (...)
    {
        return ReplayConfigStatus::out_of_memory;
    }

    // Commit. Every step below is a move or a scalar store and cannot fail.
    reason_table_ = std::move(reason_table);
    resolved_ = std::move(resolved);
    gap_scratch_ = std::move(gap_scratch);
    fault_state_ = std::move(fault_state);
    clock_ = config.clock == nullptr ? &streaming::default_native_clock() : config.clock;
    wake_clock_.store(clock_, std::memory_order_release);
    pacing_ = config.pacing;
    speed_factor_ = config.pacing == ReplayPacing::recorded ? config.speed_factor : 1.0;
    blocking_ = config.blocking;
    preserve_valid_until_ = config.preserve_recorded_valid_until;
    wake_latency_ns_ = config.wake_latency_ns == 0 ? 1 : config.wake_latency_ns;
    session_id_.store(session, std::memory_order_release);
    prepared_ = true;
    reset_locked();
    return ReplayConfigStatus::ok;
}

ReplayConfigStatus
NativeReplaySource::check_schema(const streaming::StreamSchema& schema) const noexcept
{
    if (!image_.is_open())
    {
        return ReplayConfigStatus::image_not_open;
    }
    if (schema.id() != image_.schema_id())
    {
        return ReplayConfigStatus::schema_mismatch;
    }
    const auto signals = schema.signals();
    const auto features = schema.feature_sets().descriptors();
    const auto units = schema.units().units();
    if (signals.size() != image_.signal_count() || features.size() != image_.feature_set_count() ||
        units.size() != image_.unit_count())
    {
        return ReplayConfigStatus::schema_mismatch;
    }

    try
    {
        for (std::size_t i = 0; i < signals.size(); ++i)
        {
            const auto& declared = image_.signal(i);
            streaming::SignalSchema image_signal{};
            image_signal.id = declared.id;
            image_signal.clock_domain = declared.clock_domain;
            image_signal.n_channels = declared.n_channels;
            image_signal.nominal_block_samples = declared.nominal_block_samples;
            image_signal.max_block_samples = declared.max_block_samples;
            image_signal.fs.numerator = declared.rate_num;
            image_signal.fs.denominator = declared.rate_den;
            image_signal.channel_set_id = declared.channel_set_id;
            image_signal.calibration_id = declared.calibration_id;
            image_signal.reference_id = declared.reference_id;
            image_signal.feature_set_id = declared.feature_set_id;
            image_signal.fixed_block_bytes = declared.fixed_block_bytes;
            image_signal.max_block_bytes = declared.max_block_bytes;
            if (!dtype_from(image_.string(declared.dtype), image_signal.dtype) ||
                !layout_from(image_.string(declared.layout), image_signal.layout) ||
                !tick_tracking_from(image_.string(declared.device_tick_tracking),
                                    image_signal.device_tick_tracking) ||
                !signal_kind_from(image_.string(declared.kind), image_signal.kind) ||
                !physical_unit_from(image_.string(declared.physical_unit),
                                    image_signal.physical_unit) ||
                !observation_timing_from(image_.string(declared.observation_timing),
                                         image_signal.observation_timing))
            {
                return ReplayConfigStatus::unsupported_schema_value;
            }
            // Channel names and impedance measurements are descriptive device
            // metadata persisted outside this fixed replay-image schema record.
            // The streaming comparison therefore covers the payload and timing
            // contract represented here.
            if (!streaming::equivalent(image_signal, signals[i]))
            {
                return ReplayConfigStatus::schema_mismatch;
            }
        }

        for (std::size_t i = 0; i < features.size(); ++i)
        {
            const auto& declared = image_.feature_set(i);
            streaming::FeatureSetDescriptor image_feature{};
            image_feature.id = declared.id;
            image_feature.source_stream_id = declared.source_stream_id;
            image_feature.source_stream = std::string(image_.string(declared.source_stream));
            image_feature.algorithm_name = std::string(image_.string(declared.algorithm_name));
            image_feature.algorithm_version =
                std::string(image_.string(declared.algorithm_version));
            image_feature.window_length_ns = declared.window_length_ns;
            image_feature.shift_ns = declared.shift_ns;
            if (!timestamp_reference_from(image_.string(declared.timestamp_reference),
                                          image_feature.timestamp_reference))
            {
                return ReplayConfigStatus::unsupported_schema_value;
            }
            image_feature.feature_names.reserve(declared.feature_name_count);
            for (std::uint32_t entry = 0; entry < declared.feature_name_count; ++entry)
            {
                image_feature.feature_names.emplace_back(
                    image_.string(image_.list_at(declared.feature_name_first + entry)));
            }
            image_feature.unit_ids.reserve(declared.unit_id_count);
            for (std::uint32_t entry = 0; entry < declared.unit_id_count; ++entry)
            {
                image_feature.unit_ids.push_back(image_.list_at(declared.unit_id_first + entry));
            }
            if (!streaming::equivalent(image_feature, features[i]))
            {
                return ReplayConfigStatus::schema_mismatch;
            }
        }

        for (std::size_t i = 0; i < units.size(); ++i)
        {
            const auto& declared = image_.unit(i);
            streaming::UnitDescriptor image_unit{};
            image_unit.id = declared.id;
            image_unit.symbol = std::string(image_.string(declared.symbol));
            image_unit.description = std::string(image_.string(declared.description));
            if (!streaming::equivalent(image_unit, units[i]))
            {
                return ReplayConfigStatus::schema_mismatch;
            }
        }
    }
    catch (...)
    {
        return ReplayConfigStatus::out_of_memory;
    }
    return ReplayConfigStatus::ok;
}

// --- run state ---------------------------------------------------------------

StreamStatus NativeReplaySource::terminal_status() const noexcept
{
    switch (terminal_.load(std::memory_order_acquire))
    {
    case ReplayTerminal::none:
        return StreamStatus::ok;
    case ReplayTerminal::end_of_data:
        return StreamStatus::end_of_stream;
    case ReplayTerminal::cancelled:
        return StreamStatus::stopped;
    case ReplayTerminal::abnormal_end:
    case ReplayTerminal::faulted:
        return StreamStatus::source_failure;
    }
    return StreamStatus::invalid_state;
}

StreamStatus NativeReplaySource::finish(ReplayTerminal terminal) noexcept
{
    // Terminal states do not downgrade: the first one to be reached is the
    // one the run keeps until reset (contract section 8.9).
    auto expected = ReplayTerminal::none;
    terminal_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
    return terminal_status();
}

void NativeReplaySource::cancel() noexcept
{
    auto expected = ReplayTerminal::none;
    terminal_.compare_exchange_strong(expected, ReplayTerminal::cancelled,
                                      std::memory_order_acq_rel);
    // Deliberately outside the gate: cancelling is only useful while a read is
    // in flight, which is exactly when the gate is unavailable.
    if (auto* clock = wake_clock_.load(std::memory_order_acquire); clock != nullptr)
    {
        clock->wake();
    }
}

std::uint64_t NativeReplaySource::advance(std::uint64_t permits) noexcept
{
    if (permits == 0)
    {
        return 0;
    }
    auto current = permits_.load(std::memory_order_acquire);
    std::uint64_t granted = 0;
    for (;;)
    {
        granted = std::min(permits, kUint64Max - current);
        if (granted == 0)
        {
            // Saturation is observable, so it is reported rather than
            // silently accepted or wrapped (contract section 8.8).
            permit_saturated_.store(true, std::memory_order_release);
            return 0;
        }
        if (permits_.compare_exchange_weak(current, current + granted, std::memory_order_acq_rel))
        {
            break;
        }
    }
    permits_granted_.fetch_add(granted, std::memory_order_acq_rel);
    if (auto* clock = wake_clock_.load(std::memory_order_acquire); clock != nullptr)
    {
        clock->wake();
    }
    return granted;
}

void NativeReplaySource::reset_locked() noexcept
{
    terminal_.store(ReplayTerminal::none, std::memory_order_release);
    permits_.store(0, std::memory_order_release);
    permits_granted_.store(0, std::memory_order_release);
    permit_saturated_.store(false, std::memory_order_release);
    next_idx_ = 0;
    origin_ns_ = 0;
    scaled_offset_ns_ = 0;
    previous_timeline_ns_ = 0;
    pending_offset_ns_ = 0;
    deadline_ns_ = 0;
    stall_until_ns_ = 0;
    anchored_ = false;
    has_previous_ = false;
    deadline_valid_ = false;
    stall_started_ = false;
    stats_.clear();
    for (std::size_t i = 0; i < resolved_.size(); ++i)
    {
        fault_state_[i].store(0, std::memory_order_release);
    }
}

StreamStatus NativeReplaySource::reset() noexcept
{
    GateGuard guard{gate_};
    if (!guard.held())
    {
        // A reset concurrent with an in-flight read is rejected rather than
        // raced (contract section 8.10).
        return StreamStatus::invalid_state;
    }
    if (!prepared_)
    {
        return StreamStatus::invalid_state;
    }
    reset_locked();
    return StreamStatus::ok;
}

ReplayStats NativeReplaySource::stats() const noexcept
{
    ReplayStats snapshot{};
    snapshot.items_emitted = stats_.items_emitted.load(std::memory_order_relaxed);
    snapshot.frames_emitted = stats_.frames_emitted.load(std::memory_order_relaxed);
    snapshot.discontinuities_emitted =
        stats_.discontinuities_emitted.load(std::memory_order_relaxed);
    snapshot.late_item_count = stats_.late_item_count.load(std::memory_order_relaxed);
    snapshot.max_lateness_ns = stats_.max_lateness_ns.load(std::memory_order_relaxed);
    snapshot.negative_delta_count = stats_.negative_delta_count.load(std::memory_order_relaxed);
    snapshot.injected_delay_ns = stats_.injected_delay_ns.load(std::memory_order_relaxed);
    snapshot.permits_consumed = stats_.permits_consumed.load(std::memory_order_relaxed);
    snapshot.permits_granted = permits_granted_.load(std::memory_order_acquire);
    snapshot.permit_saturated = permit_saturated_.load(std::memory_order_acquire);
    return snapshot;
}

ReplayFaultReport NativeReplaySource::fault_report(std::size_t idx) const noexcept
{
    ReplayFaultReport report{};
    if (idx >= resolved_.size())
    {
        return report;
    }
    const auto state = fault_state_[idx].load(std::memory_order_acquire);
    report.effect = resolved_[idx].effect;
    report.item_idx = resolved_[idx].item_idx;
    report.fired = (state & kFaultFired) != 0U;
    report.emitted_known = (state & kFaultEmittedKnown) != 0U;
    report.emitted = (state & kFaultEmitted) != 0U;
    return report;
}

std::size_t NativeReplaySource::fault_at(std::size_t idx) const noexcept
{
    for (std::size_t entry = 0; entry < resolved_.size(); ++entry)
    {
        if (resolved_[entry].item_idx != idx)
        {
            continue;
        }
        if ((fault_state_[entry].load(std::memory_order_acquire) & kFaultFired) == 0U)
        {
            return entry;
        }
    }
    return kNoFault;
}

void NativeReplaySource::report_fault(std::size_t fault, bool emitted) noexcept
{
    const std::uint8_t state = static_cast<std::uint8_t>(kFaultFired | kFaultEmittedKnown |
                                                         (emitted ? kFaultEmitted : 0U));
    fault_state_[fault].store(state, std::memory_order_release);
}

// --- pacing ------------------------------------------------------------------

std::uint64_t NativeReplaySource::deadline_for(const ReplayImageItem& item) noexcept
{
    if (deadline_valid_)
    {
        return deadline_ns_;
    }
    std::uint64_t delta = 0;
    if (has_previous_)
    {
        if (item.timeline_ns >= previous_timeline_ns_)
        {
            delta = item.timeline_ns - previous_timeline_ns_;
        }
        else
        {
            // Recorded host time that runs backwards is clamped to zero -- the
            // item is emitted immediately, never waited for backwards -- and
            // the occurrence is counted (contract section 8.7).
            bump(stats_.negative_delta_count);
        }
    }
    std::uint64_t scaled = delta;
    if (pacing_ == ReplayPacing::recorded && speed_factor_ != 1.0)
    {
        const auto value = static_cast<double>(delta) / speed_factor_;
        scaled = value >= static_cast<double>(kUint64Max) ? kUint64Max
                                                          : static_cast<std::uint64_t>(value);
    }
    pending_offset_ns_ = saturating_add(scaled_offset_ns_, scaled);
    // Deadlines are absolute -- the run's start instant plus this item's
    // cumulative offset -- so a late wake-up never pushes the next one out.
    deadline_ns_ = (pacing_ == ReplayPacing::recorded && anchored_)
                       ? saturating_add(origin_ns_, pending_offset_ns_)
                       : 0;
    deadline_valid_ = true;
    return deadline_ns_;
}

void NativeReplaySource::advance_timeline(const ReplayImageItem& item) noexcept
{
    // An item the run declines to emit leaves its time in place: the run still
    // spans the original elapsed time (contract section 8.7).
    scaled_offset_ns_ = pending_offset_ns_;
    previous_timeline_ns_ = item.timeline_ns;
    has_previous_ = true;
    deadline_valid_ = false;
    stall_started_ = false;
    stall_until_ns_ = 0;
    ++next_idx_;
}

bool NativeReplaySource::wait_until(std::uint64_t deadline_ns) noexcept
{
    for (;;)
    {
        if (terminal_.load(std::memory_order_acquire) != ReplayTerminal::none)
        {
            return false;
        }
        const auto now = clock_->now_ns();
        if (now >= deadline_ns)
        {
            return true;
        }
        if (!blocking_)
        {
            return false;
        }
        // Sliced, so cancellation latency is bounded by the declared wake
        // latency and not by the remaining pacing interval (section 8.9).
        const auto slice = std::min(deadline_ns, saturating_add(now, wake_latency_ns_));
        clock_->wait_until(slice);
    }
}

// --- emission ----------------------------------------------------------------

StreamStatus NativeReplaySource::fill_frame(const ReplayImageItem& item,
                                            MutableFrame& frame) noexcept
{
    auto blocks = frame.block_storage();
    auto payload = frame.payload_storage();
    if (item.n_children > blocks.size() || item.payload_byte_count > payload.size())
    {
        return StreamStatus::invalid_frame;
    }
    const auto bytes = image_.payload(item.payload_offset, item.payload_byte_count);
    if (bytes.size() != item.payload_byte_count)
    {
        return StreamStatus::invalid_frame;
    }
    if (!bytes.empty())
    {
        std::memcpy(payload.data(), bytes.data(), bytes.size());
    }

    for (std::uint64_t child = 0; child < item.n_children; ++child)
    {
        const auto source = image_.block(static_cast<std::size_t>(item.child_first + child));
        if (source.payload_offset + source.payload_byte_count > item.payload_byte_count)
        {
            return StreamStatus::invalid_frame;
        }
        auto& target = blocks[static_cast<std::size_t>(child)];
        target = streaming::SignalBlockHeader{};
        target.sample_idx_start = source.sample_idx_start;
        target.device_tick_start = source.device_tick_start;
        target.observation_time_start_ns = source.observation_time_start_ns;
        target.payload_offset = source.payload_offset;
        target.payload_byte_count = source.payload_byte_count;
        target.signal_id = source.native_signal_id;
        target.n_samples = source.n_samples;
        target.last_sample_idx = source.last_sample_idx;
        if ((source.flags & kReplayBlockFlagClockSync) != 0U)
        {
            target.clock_sync.device_tick_reference = source.clock_sync_device_tick_reference;
            target.clock_sync.host_time_reference_ns = source.clock_sync_host_time_reference_ns;
            target.clock_sync.device_tick_rate.numerator = source.clock_sync_rate_numerator;
            target.clock_sync.device_tick_rate.denominator = source.clock_sync_rate_denominator;
            target.clock_sync.uncertainty_ns = source.clock_sync_uncertainty_ns;
            target.clock_sync.clock_domain = source.clock_sync_clock_domain;
            target.clock_sync.generation = source.clock_sync_generation;
            target.clock_sync.flags =
                static_cast<streaming::ClockSyncFlags>(source.clock_sync_flags);
        }
        // A block index of the synthesized shape carries no clock-sync
        // snapshot at all. The zeroed snapshot above is that absence, reported
        // by the image's fidelity entry, and is never an invented one.
    }

    const auto status = frame.set_used_sizes(static_cast<std::size_t>(item.n_children),
                                             static_cast<std::size_t>(item.payload_byte_count));
    if (status != StreamStatus::ok)
    {
        return status;
    }

    auto& header = frame.header();
    header.session_id = session_id_.load(std::memory_order_relaxed);
    header.sequence = item.replay_sequence;
    // Current-run time. The original arrival time stays in the image as
    // provenance and is never presented as this run's timing (section 8.6).
    header.host_received_ns = clock_->now_ns();
    header.schema_id = item.native_schema_id;
    header.source_clock_domain = item.source_clock_domain;
    header.flags = FrameFlags::none;
    if ((item.flags & kReplayItemFlagSourceTick) != 0U)
    {
        header.source_tick = item.source_tick;
        header.flags = header.flags | FrameFlags::source_tick;
    }
    // A recorded deadline is stale by construction and is cleared by default;
    // preserving it is an explicit test behaviour, never a default.
    if (preserve_valid_until_ && (item.flags & kReplayItemFlagValidUntil) != 0U)
    {
        header.valid_until_ns = item.valid_until_ns;
        header.flags = header.flags | FrameFlags::valid_until;
    }
    return StreamStatus::ok;
}

StreamStatus NativeReplaySource::fill_discontinuity(const ReplayImageItem& item,
                                                    DiscontinuityLease& lease) noexcept
{
    if (!lease)
    {
        return StreamStatus::invalid_state;
    }
    if (item.n_children > lease.gap_capacity() || item.n_children > gap_scratch_.size())
    {
        return StreamStatus::buffer_exhausted;
    }
    for (std::uint64_t child = 0; child < item.n_children; ++child)
    {
        const auto source = image_.gap(static_cast<std::size_t>(item.child_first + child));
        auto& target = gap_scratch_[static_cast<std::size_t>(child)];
        target = SignalGap{};
        target.expected_sample_idx = source.expected_sample_idx;
        target.actual_sample_idx = source.actual_sample_idx;
        target.signal_id = source.native_signal_id;
        target.reason = reason_of(source.reason);
        auto flags = static_cast<SignalGapFlags>(static_cast<std::uint8_t>(source.gap_flags));
        if (source.missing_samples != kReplayAbsentU64)
        {
            target.missing_samples = source.missing_samples;
            // A value that is present is by definition known. A ledger-based
            // mode sets the recorded bit already; a synthesized record has no
            // flags of its own, so presence is the only honest source.
            flags = flags | SignalGapFlags::missing_samples_known;
        }
        if (source.expected_device_tick != kReplayAbsentU64 &&
            source.actual_device_tick != kReplayAbsentU64)
        {
            target.expected_device_tick = source.expected_device_tick;
            target.actual_device_tick = source.actual_device_tick;
        }
        target.flags = flags;
    }
    const auto previous =
        item.previous_replay_sequence == kReplayAbsentU64 ? 0 : item.previous_replay_sequence;
    return lease.assign(
        session_id_.load(std::memory_order_relaxed), previous, item.replay_sequence,
        reason_of(item.reason),
        std::span<const SignalGap>(gap_scratch_.data(), static_cast<std::size_t>(item.n_children)));
}

// --- the read loop -----------------------------------------------------------

StreamStatus NativeReplaySource::read_message(MutableFrame& frame,
                                              DiscontinuityLease& discontinuity) noexcept
{
    GateGuard guard{gate_};
    if (!guard.held())
    {
        return StreamStatus::invalid_state;
    }
    if (!prepared_)
    {
        return StreamStatus::invalid_state;
    }
    return read_locked(frame, discontinuity);
}

StreamStatus NativeReplaySource::read_locked(MutableFrame& frame,
                                             DiscontinuityLease& discontinuity) noexcept
{
    for (;;)
    {
        if (terminal_.load(std::memory_order_acquire) != ReplayTerminal::none)
        {
            return terminal_status();
        }
        if (next_idx_ >= image_.item_count())
        {
            // An incomplete session replayed under allow-incomplete ends
            // abnormally, never as if the data ran out naturally (8.5).
            return finish(image_.summary().abnormal_end_required() ? ReplayTerminal::abnormal_end
                                                                   : ReplayTerminal::end_of_data);
        }

        const auto item = image_.item(next_idx_);
        auto fault = fault_at(next_idx_);
        const auto effect_of = [this](std::size_t idx) noexcept { return resolved_[idx].effect; };

        if (fault != kNoFault && effect_of(fault) == ReplayFaultEffect::sequence_gap)
        {
            // The frame is not emitted, so the consumer sees a genuine jump in
            // the replay-run frame sequence -- and it costs no permit, because
            // a permit buys an emitted item.
            report_fault(fault, false);
            static_cast<void>(deadline_for(item));
            advance_timeline(item);
            continue;
        }

        if (pacing_ == ReplayPacing::step)
        {
            for (;;)
            {
                if (terminal_.load(std::memory_order_acquire) != ReplayTerminal::none)
                {
                    return terminal_status();
                }
                if (permits_.load(std::memory_order_acquire) != 0)
                {
                    break;
                }
                if (!blocking_)
                {
                    return StreamStatus::would_block;
                }
                clock_->wait_until(saturating_add(clock_->now_ns(), wake_latency_ns_));
            }
        }

        const auto deadline = deadline_for(item);
        if (fault != kNoFault && effect_of(fault) == ReplayFaultEffect::stall && !stall_started_)
        {
            stall_started_ = true;
            const auto base = std::max(deadline, clock_->now_ns());
            stall_until_ns_ = saturating_add(base, resolved_[fault].stall_ns);
            bump(stats_.injected_delay_ns, resolved_[fault].stall_ns);
        }
        const auto effective = std::max(deadline, stall_until_ns_);
        if (!wait_until(effective))
        {
            return terminal_.load(std::memory_order_acquire) != ReplayTerminal::none
                       ? terminal_status()
                       : StreamStatus::would_block;
        }
        if (stall_started_ && fault != kNoFault && effect_of(fault) == ReplayFaultEffect::stall)
        {
            report_fault(fault, true);
            fault = kNoFault;
        }

        if (fault != kNoFault && effect_of(fault) == ReplayFaultEffect::read_failure)
        {
            report_fault(fault, false);
            return finish(ReplayTerminal::faulted);
        }
        if (fault != kNoFault && effect_of(fault) == ReplayFaultEffect::abnormal_end)
        {
            report_fault(fault, false);
            return finish(ReplayTerminal::abnormal_end);
        }

        if (pacing_ == ReplayPacing::recorded && anchored_)
        {
            const auto now = clock_->now_ns();
            if (now > deadline)
            {
                // A late deadline emits immediately and nothing is skipped or
                // compressed; the lateness is reported instead.
                bump(stats_.late_item_count);
                raise_to(stats_.max_lateness_ns, now - deadline);
            }
        }

        StreamStatus status = StreamStatus::ok;
        if (item.kind == ReplayItemKind::frame)
        {
            status = fill_frame(item, frame);
        }
        else
        {
            status = fill_discontinuity(item, discontinuity);
        }
        if (status != StreamStatus::ok)
        {
            static_cast<void>(finish(ReplayTerminal::faulted));
            return status;
        }

        if (!anchored_)
        {
            // The run's timeline is anchored on the first item it emits; no
            // time is waited before that item (contract section 8.7).
            anchored_ = true;
            origin_ns_ = clock_->now_ns();
            pending_offset_ns_ = 0;
        }
        if (pacing_ == ReplayPacing::step)
        {
            permits_.fetch_sub(1, std::memory_order_acq_rel);
            bump(stats_.permits_consumed);
        }
        bump(stats_.items_emitted);
        if (item.kind == ReplayItemKind::frame)
        {
            bump(stats_.frames_emitted);
        }
        else
        {
            bump(stats_.discontinuities_emitted);
        }
        advance_timeline(item);
        return item.kind == ReplayItemKind::frame ? StreamStatus::ok : StreamStatus::discontinuity;
    }
}

StreamStatus NativeReplaySource::read(MutableFrame& frame) noexcept
{
    GateGuard guard{gate_};
    if (!guard.held())
    {
        return StreamStatus::invalid_state;
    }
    if (!prepared_)
    {
        return StreamStatus::invalid_state;
    }
    // The frame-only interface has nowhere to put a discontinuity, and
    // inferring one away is exactly what section 8 forbids. An image that
    // carries one is therefore refused rather than quietly flattened.
    if (terminal_.load(std::memory_order_acquire) == ReplayTerminal::none &&
        next_idx_ < image_.item_count() &&
        image_.item(next_idx_).kind == ReplayItemKind::discontinuity)
    {
        static_cast<void>(finish(ReplayTerminal::faulted));
        return StreamStatus::invalid_state;
    }
    DiscontinuityLease unused;
    return read_locked(frame, unused);
}

} // namespace neurale::recording
