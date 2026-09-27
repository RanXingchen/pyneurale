/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// The native end of the record -> spool -> NRF -> replay lifecycle.
///
/// Every other native replay suite states its own image, because a test about
/// pacing needs an image it can spell out. That is the right trade for those
/// pacing suites and the wrong one here: this suite's question is whether a
/// session that was actually recorded comes back, and an image this file wrote
/// could only answer whether this file is self-consistent. So the images here
/// arrive from outside -- `tests/unit/recording/test_end_to_end.py` records
/// a real native session, finalizes it, builds a real image, and runs this
/// binary over the result -- and everything below is reporting, not authorship.
///
/// Seven entries:
///
///   `--replay <image>`  one run through `NativeReplaySource` directly, with
///                       every emitted message printed as JSON: the payload of
///                       every block, checksummed, is what makes a payload
///                       comparison against the recorded samples possible.
///   `--runner <image>`  the same image through a real `NativeStreamRunner`,
///                       which is the only way to ask what a *consumer* sees --
///                       in particular whether the continuity checker inferred
///                       a break the recording never had, and what bytes
///                       arrived at the far end of the runtime and processor.
///   `--schema <image>`  the schema a process holding only the image can
///                       recover, printed so it can be compared with the schema
///                       the recording declared rather than with the image.
///   `--allocate <image>` the steady-state allocation count of a run over a
///                       real image, under the same `--wrap` tracker the other
///                       gates use.
///   `--rerecord <image> <plan> <spool>` replay through a real runtime and
///                       critical recorder into a second native spool. The
///                       Python half finalizes and replays that spool.
///                       `--crash-after-committed N` switches that entry to the
///                       bounded mapped backend and parks after N committed
///                       messages so the parent can terminate the process.
///                       Linux requires tmpfs; Windows requires local storage.
///   `--self-test`       the CTest entry: the harness's own reporting, over an
///                       image built here, so a checkout with no Python tests
///                       still runs it.
///
/// Section numbers below are `docs/development/native_recording_replay.md`.

#include "allocation_tracker.h"
#include "check_returns.h"
#include "crc32c.h"
#include "memory_spool_file.h"
#include "recorder.h"
#include "recording_plan.h"
#include "replay/schema_names.h"
#include "replay/source.h"
#include "replay_test_support.h"
#include "sha256.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
// clang-format off
#include <windows.h>
#include <tlhelp32.h>
// clang-format on
#endif

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/discontinuity_pool.h>
#include <neurale/streaming/runtime.h>
#include <neurale/streaming/schema.h>

namespace
{

using neurale::recording::BoundedMappedSpoolFile;
using neurale::recording::kReplayAbsentU32;
using neurale::recording::kReplayAbsentU64;
using neurale::recording::MemorySpoolFile;
using neurale::recording::NativeRecorderCore;
using neurale::recording::NativeRecordingPlan;
using neurale::recording::NativeReplaySource;
using neurale::recording::PlannedSignalRecording;
using neurale::recording::ReplayConfigStatus;
using neurale::recording::ReplayFaultEffect;
using neurale::recording::ReplayFaultSpec;
using neurale::recording::ReplayFaultTarget;
using neurale::recording::ReplayImageFile;
using neurale::recording::ReplayImageStatus;
using neurale::recording::ReplayItemKind;
using neurale::recording::ReplayPacing;
using neurale::recording::ReplaySourceConfig;
using neurale::recording::ReplayTerminal;

using namespace neurale::streaming;

// --- rebuilding the recording's schema from the image ------------------------
//
// The runner needs a StreamSchema, and the only description of the recorded
// schema available to a process holding nothing but an image is the image's own
// schema section. The spellings come from `schema_names.h`, which is the
// table the production reader uses -- a copy here could accept what the real
// reader refuses, and the harness would be certifying itself.

using namespace neurale::recording::names;

/// Rebuild the declared schema, or report which spelling defeated it.
[[nodiscard]] std::optional<StreamSchema> schema_from_image(const ReplayImageFile& image,
                                                            std::string& refusal)
{
    std::vector<SignalSchema> signals;
    signals.reserve(image.signal_count());
    for (std::size_t i = 0; i < image.signal_count(); ++i)
    {
        const auto& declared = image.signal(i);
        SignalSchema signal{};
        signal.id = declared.id;
        signal.clock_domain = declared.clock_domain;
        signal.n_channels = declared.n_channels;
        signal.nominal_block_samples = declared.nominal_block_samples;
        signal.max_block_samples = declared.max_block_samples;
        signal.fs.numerator = declared.rate_num;
        signal.fs.denominator = declared.rate_den;
        signal.channel_set_id = declared.channel_set_id;
        signal.calibration_id = declared.calibration_id;
        signal.reference_id = declared.reference_id;
        signal.feature_set_id = declared.feature_set_id;
        signal.fixed_block_bytes = declared.fixed_block_bytes;
        signal.max_block_bytes = declared.max_block_bytes;
        if (!dtype_from(image.string(declared.dtype), signal.dtype) ||
            !layout_from(image.string(declared.layout), signal.layout) ||
            !tick_tracking_from(image.string(declared.device_tick_tracking),
                                signal.device_tick_tracking) ||
            !signal_kind_from(image.string(declared.kind), signal.kind) ||
            !physical_unit_from(image.string(declared.physical_unit), signal.physical_unit) ||
            !observation_timing_from(image.string(declared.observation_timing),
                                     signal.observation_timing))
        {
            refusal = "unsupported schema value on signal " + std::to_string(declared.id);
            return std::nullopt;
        }
        signals.push_back(signal);
    }

    std::vector<FeatureSetDescriptor> features;
    features.reserve(image.feature_set_count());
    for (std::size_t i = 0; i < image.feature_set_count(); ++i)
    {
        const auto& declared = image.feature_set(i);
        FeatureSetDescriptor feature{};
        feature.id = declared.id;
        feature.source_stream_id = declared.source_stream_id;
        feature.source_stream = std::string(image.string(declared.source_stream));
        feature.algorithm_name = std::string(image.string(declared.algorithm_name));
        feature.algorithm_version = std::string(image.string(declared.algorithm_version));
        feature.window_length_ns = declared.window_length_ns;
        feature.shift_ns = declared.shift_ns;
        if (!timestamp_reference_from(image.string(declared.timestamp_reference),
                                      feature.timestamp_reference))
        {
            refusal = "unsupported timestamp reference";
            return std::nullopt;
        }
        for (std::uint32_t entry = 0; entry < declared.feature_name_count; ++entry)
        {
            feature.feature_names.emplace_back(
                image.string(image.list_at(declared.feature_name_first + entry)));
        }
        for (std::uint32_t entry = 0; entry < declared.unit_id_count; ++entry)
        {
            feature.unit_ids.push_back(image.list_at(declared.unit_id_first + entry));
        }
        features.push_back(std::move(feature));
    }

    std::vector<UnitDescriptor> units;
    units.reserve(image.unit_count());
    for (std::size_t i = 0; i < image.unit_count(); ++i)
    {
        const auto& declared = image.unit(i);
        UnitDescriptor unit{};
        unit.id = declared.id;
        unit.symbol = std::string(image.string(declared.symbol));
        unit.description = std::string(image.string(declared.description));
        units.push_back(std::move(unit));
    }

    try
    {
        return StreamSchema{image.schema_id(), signals, features, units};
    }
    catch (const std::exception& error)
    {
        refusal = error.what();
        return std::nullopt;
    }
}

/// The same FNV-1a the synthetic source records in its manifest.
///
/// The source states a digest of the bytes it wrote *before* anything
/// downstream sees them; this recomputes it over the bytes that came back. The
/// two must agree block for block, and neither side borrows the other's code --
/// the source's copy lives in the streaming bindings and the third is three
/// lines of Python.
[[nodiscard]] std::uint64_t payload_digest(std::span<const std::byte> bytes) noexcept
{
    std::uint64_t hash = 14695981039346656037ULL;
    for (const auto value : bytes)
    {
        hash ^= static_cast<std::uint64_t>(value);
        hash *= 1099511628211ULL;
    }
    return hash;
}

// --- JSON, by hand ------------------------------------------------------------

void emit_string(std::string_view key, std::string_view value, bool first = false)
{
    std::cout << (first ? "" : ",") << '"' << key << "\":\"" << value << '"';
}

void emit_u64(std::string_view key, std::uint64_t value, bool first = false)
{
    std::cout << (first ? "" : ",") << '"' << key << "\":" << value;
}

void emit_bool(std::string_view key, bool value, bool first = false)
{
    std::cout << (first ? "" : ",") << '"' << key << "\":" << (value ? "true" : "false");
}

[[nodiscard]] const char* effect_text(ReplayFaultEffect effect) noexcept
{
    switch (effect)
    {
    case ReplayFaultEffect::stall:
        return "stall";
    case ReplayFaultEffect::read_failure:
        return "read_failure";
    case ReplayFaultEffect::sequence_gap:
        return "sequence_gap";
    case ReplayFaultEffect::abnormal_end:
        return "abnormal_end";
    }
    return "unknown";
}

[[nodiscard]] const char* reason_text(GapReason reason) noexcept
{
    switch (reason)
    {
    case GapReason::frame_sequence_gap:
        return "frame_sequence_gap";
    case GapReason::sample_gap:
        return "sample_gap";
    case GapReason::device_tick_gap:
        return "device_tick_gap";
    case GapReason::device_restart:
        return "device_restart";
    case GapReason::source_gap:
        return "source_gap";
    case GapReason::buffer_exhausted:
        return "buffer_exhausted";
    case GapReason::queue_overflow:
        return "queue_overflow";
    }
    return "unknown";
}

[[nodiscard]] const char* status_text(StreamStatus status) noexcept
{
    switch (status)
    {
    case StreamStatus::ok:
        return "ok";
    case StreamStatus::end_of_stream:
        return "end_of_stream";
    case StreamStatus::stopped:
        return "stopped";
    case StreamStatus::source_failure:
        return "source_failure";
    case StreamStatus::buffer_exhausted:
        return "buffer_exhausted";
    case StreamStatus::queue_overflow:
        return "queue_overflow";
    case StreamStatus::invalid_frame:
        return "invalid_frame";
    case StreamStatus::processor_failure:
        return "processor_failure";
    case StreamStatus::consumer_failure:
        return "consumer_failure";
    case StreamStatus::actuator_failure:
        return "actuator_failure";
    case StreamStatus::observer_overrun:
        return "observer_overrun";
    case StreamStatus::would_block:
        return "would_block";
    case StreamStatus::discontinuity:
        return "discontinuity";
    case StreamStatus::invalid_state:
        return "invalid_state";
    case StreamStatus::output_limit:
        return "output_limit";
    case StreamStatus::deadline_exceeded:
        return "deadline_exceeded";
    case StreamStatus::safety_failure:
        return "safety_failure";
    case StreamStatus::realtime_configuration_failed:
        return "realtime_configuration_failed";
    }
    return "unknown";
}

[[nodiscard]] const char* runtime_state_text(RuntimeState state) noexcept
{
    switch (state)
    {
    case RuntimeState::created:
        return "created";
    case RuntimeState::prepared:
        return "prepared";
    case RuntimeState::running:
        return "running";
    case RuntimeState::stopping:
        return "stopping";
    case RuntimeState::stopped:
        return "stopped";
    case RuntimeState::failed:
        return "failed";
    }
    return "unknown";
}

[[nodiscard]] const char* fault_stage_text(FaultStage stage) noexcept
{
    switch (stage)
    {
    case FaultStage::source:
        return "source";
    case FaultStage::continuity:
        return "continuity";
    case FaultStage::processor:
        return "processor";
    case FaultStage::output:
        return "output";
    case FaultStage::consumer:
        return "consumer";
    case FaultStage::actuator:
        return "actuator";
    case FaultStage::observer:
        return "observer";
    case FaultStage::runtime:
        return "runtime";
    }
    return "runtime";
}

// --- the shapes a run is reported in -----------------------------------------

/// One emitted message, in the form a payload comparison needs. Kept in memory
/// so a second run can be compared against the first without the reader having
/// to do it (contract section 8.10: a reset run is the same run again).
struct Observed
{
    bool is_frame{};
    std::uint64_t session_id{};
    std::uint64_t sequence{};
    std::uint64_t schema_id{};
    std::uint64_t clock_domain{};
    std::uint64_t source_tick{};
    std::uint64_t valid_until_ns{};
    std::uint64_t previous_sequence{};
    std::uint32_t reason{};
    std::uint64_t payload_bytes{};
    std::uint32_t payload_crc{};
    std::uint32_t flags{};
    struct Block
    {
        std::uint64_t signal_id{};
        std::uint64_t sample_idx_start{};
        std::uint64_t last_sample_idx{};
        std::uint64_t n_samples{};
        std::uint64_t device_tick_start{};
        std::uint64_t observation_time_start_ns{};
        std::uint64_t payload_byte_count{};
        std::uint32_t payload_crc{};
        std::uint64_t payload_digest{};
        // The clock-sync snapshot is provenance no other suite follows across
        // the whole chain: the recorder writes it, the finalizer stores it, the
        // image keeps it behind a flag, and the replay source rebuilds it. A
        // block that arrived with the right samples under someone else's clock
        // reference is still a wrong block.
        std::uint64_t clock_sync_device_tick_reference{};
        std::uint64_t clock_sync_host_time_reference_ns{};
        std::uint64_t clock_sync_rate_numerator{};
        std::uint64_t clock_sync_rate_denominator{};
        std::uint64_t clock_sync_uncertainty_ns{};
        std::uint32_t clock_sync_clock_domain{};
        std::uint32_t clock_sync_generation{};
        std::uint32_t clock_sync_flags{};

        [[nodiscard]] bool operator==(const Block&) const noexcept = default;
    };
    struct Gap
    {
        std::uint64_t signal_id{};
        std::uint32_t reason{};
        std::uint64_t expected_sample_idx{};
        std::uint64_t actual_sample_idx{};
        std::uint64_t missing_samples{};
        std::uint64_t expected_device_tick{};
        std::uint64_t actual_device_tick{};
        std::uint32_t flags{};

        [[nodiscard]] bool operator==(const Gap&) const noexcept = default;
    };
    std::vector<Block> blocks;
    std::vector<Gap> gaps;

    [[nodiscard]] bool operator==(const Observed& other) const noexcept = default;
};

void print(const Observed& message, std::size_t idx)
{
    std::cout << '{';
    emit_u64("index", idx, true);
    emit_string("kind", message.is_frame ? "frame" : "discontinuity");
    emit_u64("session_id", message.session_id);
    emit_u64("sequence", message.sequence);
    if (message.is_frame)
    {
        emit_u64("schema_id", message.schema_id);
        emit_u64("clock_domain", message.clock_domain);
        emit_u64("source_tick", message.source_tick);
        emit_u64("valid_until_ns", message.valid_until_ns);
        emit_u64("flags", message.flags);
        emit_u64("payload_byte_count", message.payload_bytes);
        emit_u64("payload_crc32c", message.payload_crc);
        std::cout << ",\"blocks\":[";
        for (std::size_t entry = 0; entry < message.blocks.size(); ++entry)
        {
            const auto& block = message.blocks[entry];
            std::cout << (entry == 0 ? "" : ",") << '{';
            emit_u64("signal_id", block.signal_id, true);
            emit_u64("sample_idx_start", block.sample_idx_start);
            emit_u64("last_sample_idx", block.last_sample_idx);
            emit_u64("n_samples", block.n_samples);
            emit_u64("device_tick_start", block.device_tick_start);
            emit_u64("observation_time_start_ns", block.observation_time_start_ns);
            emit_u64("payload_byte_count", block.payload_byte_count);
            emit_u64("payload_crc32c", block.payload_crc);
            emit_u64("payload_digest", block.payload_digest);
            emit_u64("clock_sync_device_tick_reference", block.clock_sync_device_tick_reference);
            emit_u64("clock_sync_host_time_reference_ns", block.clock_sync_host_time_reference_ns);
            emit_u64("clock_sync_rate_numerator", block.clock_sync_rate_numerator);
            emit_u64("clock_sync_rate_denominator", block.clock_sync_rate_denominator);
            emit_u64("clock_sync_uncertainty_ns", block.clock_sync_uncertainty_ns);
            emit_u64("clock_sync_clock_domain", block.clock_sync_clock_domain);
            emit_u64("clock_sync_generation", block.clock_sync_generation);
            emit_u64("clock_sync_flags", block.clock_sync_flags);
            std::cout << '}';
        }
        std::cout << ']';
    }
    else
    {
        emit_u64("previous_sequence", message.previous_sequence);
        emit_string("reason", reason_text(static_cast<GapReason>(message.reason)));
        std::cout << ",\"gaps\":[";
        for (std::size_t entry = 0; entry < message.gaps.size(); ++entry)
        {
            const auto& gap = message.gaps[entry];
            std::cout << (entry == 0 ? "" : ",") << '{';
            emit_u64("signal_id", gap.signal_id, true);
            emit_string("reason", reason_text(static_cast<GapReason>(gap.reason)));
            emit_u64("expected_sample_idx", gap.expected_sample_idx);
            emit_u64("actual_sample_idx", gap.actual_sample_idx);
            emit_u64("missing_samples", gap.missing_samples);
            emit_u64("expected_device_tick", gap.expected_device_tick);
            emit_u64("actual_device_tick", gap.actual_device_tick);
            emit_u64("flags", gap.flags);
            std::cout << '}';
        }
        std::cout << ']';
    }
    std::cout << "}\n";
}

/// Pools sized from the image, so a run is not refused for a reason of the
/// harness's own making and a frame that does not fit is still the source's
/// refusal to report rather than this file's.
struct Pools
{
    Pools(const ReplayImageFile& image)
        : frames(4, payload_bound(image), block_bound(image)), discontinuities(4, gap_bound(image))
    {
    }

    static std::size_t payload_bound(const ReplayImageFile& image)
    {
        std::uint64_t largest = 64;
        for (std::size_t i = 0; i < image.item_count(); ++i)
        {
            largest = std::max(largest, image.item(i).payload_byte_count);
        }
        return static_cast<std::size_t>(largest);
    }

    static std::size_t block_bound(const ReplayImageFile& image)
    {
        std::uint64_t largest = 1;
        for (std::size_t i = 0; i < image.item_count(); ++i)
        {
            const auto item = image.item(i);
            if (item.kind == ReplayItemKind::frame)
            {
                largest = std::max<std::uint64_t>(largest, item.n_children);
            }
        }
        return static_cast<std::size_t>(largest);
    }

    static std::size_t gap_bound(const ReplayImageFile& image)
    {
        std::uint64_t largest = 1;
        for (std::size_t i = 0; i < image.item_count(); ++i)
        {
            const auto item = image.item(i);
            if (item.kind == ReplayItemKind::discontinuity)
            {
                largest = std::max<std::uint64_t>(largest, item.n_children);
            }
        }
        return static_cast<std::size_t>(largest);
    }

    FramePool frames;
    DiscontinuityPool discontinuities;
};

/// Read one whole run, recording every emitted message. The terminal status is
/// returned; a run that ends any way other than end_of_stream is a result to
/// report, not a failure of the harness.
StreamStatus drain(NativeReplaySource& source, Pools& pools, std::vector<Observed>& observed,
                   std::size_t bound)
{
    observed.clear();
    for (std::size_t step = 0; step < bound; ++step)
    {
        FrameLease frame;
        DiscontinuityLease discontinuity;
        if (pools.frames.try_acquire(frame) != StreamStatus::ok ||
            pools.discontinuities.try_acquire(discontinuity) != StreamStatus::ok)
        {
            return StreamStatus::buffer_exhausted;
        }
        const auto status = source.read_message(frame.frame(), discontinuity);
        if (status == StreamStatus::ok)
        {
            const auto view = frame.frame().view();
            Observed message;
            message.is_frame = true;
            message.session_id = view.header.session_id;
            message.sequence = view.header.sequence;
            message.schema_id = view.header.schema_id;
            message.clock_domain = view.header.source_clock_domain;
            message.source_tick = view.header.source_tick;
            message.valid_until_ns = view.header.valid_until_ns;
            message.flags = static_cast<std::uint32_t>(view.header.flags);
            message.payload_bytes = view.payload.size();
            message.payload_crc = neurale::recording::crc32c(view.payload);
            for (const auto& block : view.blocks)
            {
                const auto bytes =
                    view.payload.subspan(static_cast<std::size_t>(block.payload_offset),
                                         static_cast<std::size_t>(block.payload_byte_count));
                message.blocks.push_back(Observed::Block{
                    .signal_id = block.signal_id,
                    .sample_idx_start = block.sample_idx_start,
                    .last_sample_idx = block.last_sample_idx,
                    .n_samples = block.n_samples,
                    .device_tick_start = block.device_tick_start,
                    .observation_time_start_ns = block.observation_time_start_ns,
                    .payload_byte_count = block.payload_byte_count,
                    .payload_crc = neurale::recording::crc32c(bytes),
                    .payload_digest = payload_digest(bytes),
                    .clock_sync_device_tick_reference = block.clock_sync.device_tick_reference,
                    .clock_sync_host_time_reference_ns = block.clock_sync.host_time_reference_ns,
                    .clock_sync_rate_numerator = block.clock_sync.device_tick_rate.numerator,
                    .clock_sync_rate_denominator = block.clock_sync.device_tick_rate.denominator,
                    .clock_sync_uncertainty_ns = block.clock_sync.uncertainty_ns,
                    .clock_sync_clock_domain = block.clock_sync.clock_domain,
                    .clock_sync_generation = block.clock_sync.generation,
                    .clock_sync_flags = static_cast<std::uint32_t>(block.clock_sync.flags),
                });
            }
            observed.push_back(std::move(message));
            continue;
        }
        if (status == StreamStatus::discontinuity)
        {
            const auto view = discontinuity.view();
            Observed message;
            message.is_frame = false;
            message.session_id = view.session_id;
            message.sequence = view.actual_frame_sequence;
            message.previous_sequence = view.previous_frame_sequence;
            message.reason = static_cast<std::uint32_t>(view.reason);
            for (const auto& gap : view.signal_gaps)
            {
                message.gaps.push_back(Observed::Gap{
                    .signal_id = gap.signal_id,
                    .reason = static_cast<std::uint32_t>(gap.reason),
                    .expected_sample_idx = gap.expected_sample_idx,
                    .actual_sample_idx = gap.actual_sample_idx,
                    .missing_samples = gap.missing_samples,
                    .expected_device_tick = gap.expected_device_tick,
                    .actual_device_tick = gap.actual_device_tick,
                    .flags = static_cast<std::uint32_t>(gap.flags),
                });
            }
            observed.push_back(std::move(message));
            continue;
        }
        return status;
    }
    return StreamStatus::would_block;
}

void print_tail(const NativeReplaySource& source, StreamStatus status, const char* extra = nullptr)
{
    const auto stats = source.stats();
    std::cout << '{';
    emit_string("final", status_text(status), true);
    emit_string("terminal", neurale::recording::replay_terminal_text(source.terminal()));
    emit_u64("items_emitted", stats.items_emitted);
    emit_u64("frames_emitted", stats.frames_emitted);
    emit_u64("discontinuities_emitted", stats.discontinuities_emitted);
    emit_u64("late_item_count", stats.late_item_count);
    emit_u64("injected_delay_ns", stats.injected_delay_ns);
    std::cout << ",\"faults\":[";
    for (std::size_t i = 0; i < source.fault_count(); ++i)
    {
        const auto report = source.fault_report(i);
        std::cout << (i == 0 ? "" : ",") << '{';
        emit_string("effect", effect_text(report.effect), true);
        emit_bool("fired", report.fired);
        emit_bool("emitted_known", report.emitted_known);
        emit_bool("emitted", report.emitted);
        emit_u64("item_idx", report.item_idx);
        std::cout << '}';
    }
    std::cout << ']';
    if (extra != nullptr)
    {
        std::cout << ',' << extra;
    }
    std::cout << "}\n";
}

// --- the fault command line ---------------------------------------------------

/// `ledger:<frame|discontinuity>:<ordinal>:<effect>[:<stall_ns>]` or
/// `stream:<frame|discontinuity>:<stream_id>:<ordinal>:<effect>[:<stall_ns>]`.
/// Targets are recorded identities, which is the only kind a fault may name
/// (section 8.11), so the command line spells them the same way.
[[nodiscard]] bool parse_fault(const std::string& text, ReplayFaultSpec& spec,
                               std::string& stream_storage)
{
    std::vector<std::string> parts;
    std::string current;
    for (const char character : text)
    {
        if (character == ':')
        {
            parts.push_back(current);
            current.clear();
            continue;
        }
        current.push_back(character);
    }
    parts.push_back(current);
    if (parts.size() < 4)
    {
        return false;
    }

    std::size_t cursor = 0;
    if (parts[cursor] == "ledger")
    {
        spec.target.ledger_based = true;
    }
    else if (parts[cursor] == "stream")
    {
        spec.target.ledger_based = false;
    }
    else
    {
        return false;
    }
    ++cursor;

    if (parts[cursor] == "frame")
    {
        spec.target.kind = ReplayItemKind::frame;
    }
    else if (parts[cursor] == "discontinuity")
    {
        spec.target.kind = ReplayItemKind::discontinuity;
    }
    else
    {
        return false;
    }
    ++cursor;

    try
    {
        if (spec.target.ledger_based)
        {
            spec.target.data_message_ordinal = std::stoull(parts[cursor++]);
        }
        else
        {
            stream_storage = parts[cursor++];
            spec.target.stream_id = stream_storage;
            if (cursor >= parts.size())
            {
                return false;
            }
            spec.target.ordinal = std::stoull(parts[cursor++]);
        }
        if (cursor >= parts.size())
        {
            return false;
        }
        const auto& effect = parts[cursor++];
        if (effect == "stall")
        {
            spec.effect = ReplayFaultEffect::stall;
        }
        else if (effect == "read_failure")
        {
            spec.effect = ReplayFaultEffect::read_failure;
        }
        else if (effect == "sequence_gap")
        {
            spec.effect = ReplayFaultEffect::sequence_gap;
        }
        else if (effect == "abnormal_end")
        {
            spec.effect = ReplayFaultEffect::abnormal_end;
        }
        else
        {
            return false;
        }
        spec.stall_ns = cursor < parts.size() ? std::stoull(parts[cursor]) : 0;
    }
    catch (const std::exception&)
    {
        return false;
    }
    return true;
}

struct Options
{
    std::string image;
    std::string plan_document;
    std::string spool_output;
    std::string stage_fault;
    std::string storage_fault;
    bool reset{};
    bool cancel_replay{};
    std::uint64_t operator_abort_at{};
    std::uint64_t crash_after_committed{};
    std::vector<std::string> faults;
    std::uint64_t session_id{};
};

// --- the entries --------------------------------------------------------------

/// One direct run, reported message by message; optionally a second run through
/// reset(), reported only as whether it was the same run again.
int replay(const Options& options)
{
    NativeReplaySource source;
    const auto opened = source.open(options.image.c_str());
    if (opened != ReplayImageStatus::ok)
    {
        std::cout << "{\"status\":\"" << neurale::recording::replay_image_status_text(opened)
                  << "\"}\n";
        return 2;
    }

    std::vector<ReplayFaultSpec> specs;
    // The parsed stream ids are borrowed by the targets, so they must outlive
    // prepare() -- a vector that reallocated would leave every earlier target
    // pointing at freed storage.
    std::vector<std::string> stream_ids(options.faults.size());
    specs.reserve(options.faults.size());
    for (std::size_t i = 0; i < options.faults.size(); ++i)
    {
        ReplayFaultSpec spec;
        if (!parse_fault(options.faults[i], spec, stream_ids[i]))
        {
            std::cout << "{\"status\":\"bad_fault\"}\n";
            return 2;
        }
        specs.push_back(spec);
    }

    ReplaySourceConfig config;
    config.pacing = ReplayPacing::as_fast_as_possible;
    config.replay_run_session_id = options.session_id;
    config.faults = specs;
    const auto prepared = source.prepare(config);
    if (prepared != ReplayConfigStatus::ok)
    {
        std::cout << "{\"status\":\"" << neurale::recording::replay_config_status_text(prepared)
                  << "\"}\n";
        return 3;
    }

    const auto& image = source.image();
    const auto& summary = image.summary();
    std::cout << '{';
    emit_string("status", "ok", true);
    emit_string("mode", image.string(summary.mode));
    emit_u64("n_items", summary.n_items);
    emit_u64("n_frames", summary.n_frames);
    emit_u64("n_discontinuities", summary.n_discontinuities);
    emit_u64("block_count", summary.n_blocks);
    emit_u64("gap_count", summary.n_gaps);
    emit_u64("run_session_id", source.session_id());
    emit_u64("recorded_session_id", summary.native_session_id);
    emit_bool("abnormal_end_required", summary.abnormal_end_required());
    emit_bool("ledger_based", summary.ledger_based());
    std::cout << "}\n";

    Pools pools{image};
    std::vector<Observed> first;
    const auto status = drain(source, pools, first, image.item_count() + 8);
    for (std::size_t i = 0; i < first.size(); ++i)
    {
        print(first[i], i);
    }

    if (!options.reset)
    {
        print_tail(source, status);
        return 0;
    }

    // A reset run is the same run again, down to the identity it carries
    // (section 8.10, 8.12). Comparing here rather than in the caller keeps the
    // second run's own output out of the stream the caller is parsing.
    const auto identity = source.session_id();
    const auto reset_status = source.reset();
    std::vector<Observed> second;
    const auto second_status = drain(source, pools, second, image.item_count() + 8);
    std::string extra = "\"reset\":\"";
    extra += status_text(reset_status);
    extra += "\",\"second_run_identical\":";
    extra += (first == second && status == second_status) ? "true" : "false";
    extra += ",\"session_id_kept\":";
    extra += identity == source.session_id() ? "true" : "false";
    print_tail(source, second_status, extra.c_str());
    return 0;
}

/// The same image, but through a real runtime: a consumer's view.
///
/// This is where "no unintended continuity reset" can be asked at all. The
/// source emits the recording's own discontinuities; the runtime's continuity
/// checker forms its own opinion from the frames. If a projection's renumbering
/// made the checker infer a break the recording never had, the count here would
/// exceed the image's own -- and no direct read of the source could ever notice.
class RecordingProcessor final : public NativeFrameProcessor
{
  public:
    explicit RecordingProcessor(std::string fault_stage = {}) noexcept
        : fault_stage_(std::move(fault_stage))
    {
    }

    PreparedProcessorContract prepare(const ProcessorPrepareContext& context) override
    {
        return {
            .accepted_input_schema = context.input_schema.clone(),
            .output_schema = context.input_schema.clone(),
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = true,
            .required_resources =
                ProcessorResourceBounds{.workspace_bytes = 0, .frame_pool_leases = 1},
        };
    }

    StreamStatus process(FrameBorrow&, FrameEmitter& output) noexcept override
    {
        const auto attempt = process_attempts_.fetch_add(1, std::memory_order_relaxed);
        if (fault_stage_ == "processor" && attempt == 2)
        {
            return StreamStatus::processor_failure;
        }
        if (fault_stage_ == "watchdog" && attempt == 2)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        }
        if (fault_stage_ == "pace")
        {
            // Test-only pacing opens a deterministic window for the harness
            // to submit control records or request an operator abort while the
            // runtime is still active.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return output.publish_input();
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        seen_.fetch_add(1, std::memory_order_acq_rel);
        return StreamStatus::ok;
    }

    StreamStatus flush(FrameEmitter&) noexcept override
    {
        if (fault_stage_ == "flush")
        {
            return StreamStatus::processor_failure;
        }
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        seen_.store(0, std::memory_order_release);
        process_attempts_.store(0, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t seen() const noexcept
    {
        return seen_.load(std::memory_order_acquire);
    }

  private:
    std::string fault_stage_{};
    std::atomic<std::size_t> seen_{};
    std::atomic<std::size_t> process_attempts_{};
};

/// What the consumer at the end of the chain actually received.
///
/// Counting frames answers whether messages arrived, not whether the *data*
/// did: a runtime or processor seam that corrupted a payload would leave every
/// count intact. A folded checksum would notice corruption but not reordering,
/// which is the other half of the question. So the consumer keeps an ordered,
/// per-block record -- position and payload digest, in arrival order -- and the
/// Python suite compares it with the source's manifest and with a direct read
/// of the same image.
struct ConsumedBlock
{
    std::uint64_t frame_ordinal{};
    std::uint64_t sequence{};
    std::uint64_t signal_id{};
    std::uint64_t sample_idx_start{};
    std::uint64_t n_samples{};
    std::uint64_t device_tick_start{};
    std::uint64_t payload_byte_count{};
    std::uint64_t payload_digest{};
};

class RecordingConsumer final : public NativeFrameConsumer
{
  public:
    explicit RecordingConsumer(std::string_view fault_stage = {}) noexcept
        : fault_stage_(fault_stage)
    {
    }

    /// Storage for the whole run is reserved before the runtime starts, so
    /// `consume()` -- which runs on a runtime thread and must not allocate --
    /// only ever writes into it.
    void reserve(std::size_t blocks)
    {
        recorded_.assign(blocks, ConsumedBlock{});
    }

    StreamStatus consume(FrameView frame) noexcept override
    {
        const auto ordinal = frames_.fetch_add(1, std::memory_order_acq_rel);
        if (fault_stage_ == "actuator" && ordinal == 2)
        {
            return StreamStatus::actuator_failure;
        }
        payload_.fetch_add(frame.payload.size(), std::memory_order_acq_rel);
        sessions_.store(frame.header.session_id, std::memory_order_release);
        for (const auto& block : frame.blocks)
        {
            const auto slot = recorded_count_.load(std::memory_order_relaxed);
            if (slot >= recorded_.size())
            {
                truncated_.store(true, std::memory_order_release);
                break;
            }
            const auto bytes =
                frame.payload.subspan(static_cast<std::size_t>(block.payload_offset),
                                      static_cast<std::size_t>(block.payload_byte_count));
            recorded_[slot] = ConsumedBlock{
                .frame_ordinal = ordinal,
                .sequence = frame.header.sequence,
                .signal_id = block.signal_id,
                .sample_idx_start = block.sample_idx_start,
                .n_samples = block.n_samples,
                .device_tick_start = block.device_tick_start,
                .payload_byte_count = block.payload_byte_count,
                .payload_digest = payload_digest(bytes),
            };
            recorded_count_.store(slot + 1, std::memory_order_release);
        }
        return StreamStatus::ok;
    }

    [[nodiscard]] std::span<const ConsumedBlock> recorded() const noexcept
    {
        return std::span<const ConsumedBlock>{recorded_}.first(
            recorded_count_.load(std::memory_order_acquire));
    }

    [[nodiscard]] bool truncated() const noexcept
    {
        return truncated_.load(std::memory_order_acquire);
    }

    StreamStatus handle_discontinuity(const Discontinuity&) noexcept override
    {
        if (fault_stage_ == "consumer")
        {
            return StreamStatus::consumer_failure;
        }
        discontinuities_.fetch_add(1, std::memory_order_acq_rel);
        return StreamStatus::ok;
    }

    StreamStatus flush() noexcept override
    {
        return StreamStatus::ok;
    }

    StreamStatus reset() noexcept override
    {
        frames_.store(0, std::memory_order_release);
        discontinuities_.store(0, std::memory_order_release);
        payload_.store(0, std::memory_order_release);
        recorded_count_.store(0, std::memory_order_release);
        truncated_.store(false, std::memory_order_release);
        return StreamStatus::ok;
    }

    [[nodiscard]] std::size_t frames() const noexcept
    {
        return frames_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t discontinuities() const noexcept
    {
        return discontinuities_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::size_t payload() const noexcept
    {
        return payload_.load(std::memory_order_acquire);
    }
    [[nodiscard]] std::uint64_t session() const noexcept
    {
        return sessions_.load(std::memory_order_acquire);
    }

  private:
    std::string_view fault_stage_{};
    std::atomic<std::size_t> frames_{};
    std::atomic<std::size_t> discontinuities_{};
    std::atomic<std::size_t> payload_{};
    std::atomic<std::uint64_t> sessions_{};
    std::vector<ConsumedBlock> recorded_;
    std::atomic<std::size_t> recorded_count_{};
    std::atomic<bool> truncated_{};
};

int runner(const Options& options)
{
    NativeReplaySource source;
    const auto opened = source.open(options.image.c_str());
    if (opened != ReplayImageStatus::ok)
    {
        std::cout << "{\"status\":\"" << neurale::recording::replay_image_status_text(opened)
                  << "\"}\n";
        return 2;
    }

    std::string refusal;
    auto schema = schema_from_image(source.image(), refusal);
    if (!schema.has_value())
    {
        std::cout << "{\"status\":\"schema_not_rebuilt\",\"detail\":\"" << refusal << "\"}\n";
        return 3;
    }
    // The image's schema and the one the runtime was built for are the same
    // object here, so this asks a narrower question than the schema-mismatch
    // tests elsewhere: whether a schema rebuilt from the image is one this build can
    // still recognise as the image's own.
    const auto checked = source.check_schema(*schema);
    if (checked != ReplayConfigStatus::ok)
    {
        std::cout << "{\"status\":\"" << neurale::recording::replay_config_status_text(checked)
                  << "\"}\n";
        return 3;
    }

    ReplaySourceConfig config;
    config.pacing = options.cancel_replay ? ReplayPacing::step : ReplayPacing::as_fast_as_possible;
    config.blocking = options.cancel_replay;
    if (source.prepare(config) != ReplayConfigStatus::ok)
    {
        std::cout << "{\"status\":\"not_prepared\"}\n";
        return 3;
    }

    const auto& image = source.image();
    RealtimeConfig realtime;
    realtime.pool_capacity.source_owned = 1;
    realtime.pool_capacity.ingress_capacity = 16;
    realtime.pool_capacity.processor_owned = 2;
    realtime.pool_capacity.critical_edge_capacity = 16;
    realtime.pool_capacity.actuator_owned = 1;
    // The runtime sizes its pool from the *schema*, not from what this image
    // happens to carry: it must hold one maximum-size frame of the declared
    // schema even if no recorded frame was ever that large. The image's own
    // bound still matters -- a projection can carry more per frame than a
    // single signal's maximum -- so the pool takes whichever is larger.
    std::uint64_t schema_payload = 0;
    for (const auto& signal : schema->signals())
    {
        schema_payload += signal.max_block_bytes;
    }
    realtime.buffer_size = std::max<std::size_t>(Pools::payload_bound(image),
                                                 static_cast<std::size_t>(schema_payload));
    realtime.max_signal_blocks =
        std::max<std::size_t>(Pools::block_bound(image), schema->signals().size());
    realtime.discontinuity_capacity = 8;
    realtime.gaps_per_discontinuity = std::max<std::size_t>(Pools::gap_bound(image), 2);
    realtime.max_process_outputs = 2;
    realtime.max_flush_outputs = 1;
    realtime.fault_history_capacity = 8;

    RecordingProcessor processor{options.stage_fault};
    RecordingConsumer consumer;
    consumer.reserve(static_cast<std::size_t>(image.summary().n_blocks) + 8);
    NativeStreamRunner runtime{std::move(*schema), realtime, source, processor, consumer};
    if (runtime.prepare() != StreamStatus::ok || runtime.arm() != StreamStatus::ok ||
        runtime.start() != StreamStatus::ok)
    {
        std::cout << "{\"status\":\"runtime_refused\"}\n";
        return 3;
    }
    bool cancel_requested = false;
    auto cancel_elapsed = std::chrono::steady_clock::duration::zero();
    if (options.cancel_replay)
    {
        static_cast<void>(source.advance(1));
        const auto observed = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (consumer.frames() == 0 && runtime.state() == RuntimeState::running &&
               std::chrono::steady_clock::now() < observed)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        const auto begin = std::chrono::steady_clock::now();
        cancel_requested = runtime.abort() == StreamStatus::ok;
        cancel_elapsed = std::chrono::steady_clock::now() - begin;
    }
    const auto joined = runtime.join();
    const auto stopped = runtime.stop();
    const auto runtime_state = runtime.state();
    const auto primary_fault = runtime.primary_fault();
    const auto nominal = joined == StreamStatus::ok && runtime_state == RuntimeState::stopped &&
                         !primary_fault.has_value();
    const auto cancelled = options.cancel_replay && cancel_requested && nominal &&
                           source.terminal() == ReplayTerminal::cancelled;

    std::cout << '{';
    emit_string("status", cancelled ? "cancelled" : (nominal ? "ok" : "faulted"), true);
    emit_string("mode", image.string(image.summary().mode));
    emit_string("join", status_text(joined));
    emit_string("stop", status_text(stopped));
    emit_string("runtime_state", runtime_state_text(runtime_state));
    emit_bool("primary_fault_present", primary_fault.has_value());
    emit_bool("cancel_requested", cancel_requested);
    emit_u64("cancel_elapsed_ms",
             static_cast<std::uint64_t>(
                 std::chrono::duration_cast<std::chrono::milliseconds>(cancel_elapsed).count()));
    emit_u64("image_frame_count", image.summary().n_frames);
    emit_u64("image_discontinuity_count", image.summary().n_discontinuities);
    emit_u64("consumed_frames", consumer.frames());
    emit_u64("consumed_discontinuities", consumer.discontinuities());
    emit_u64("processor_discontinuities", processor.seen());
    emit_u64("consumed_payload_bytes", consumer.payload());
    emit_u64("consumer_session_id", consumer.session());
    emit_u64("run_session_id", source.session_id());
    emit_bool("consumer_record_truncated", consumer.truncated());
    emit_string("terminal", neurale::recording::replay_terminal_text(source.terminal()));
    emit_u64("outstanding_frames", runtime.outstanding_frames());
    emit_u64("outstanding_discontinuities", runtime.outstanding_discontinuities());
    std::cout << ",\"consumed\":[";
    const auto consumed = consumer.recorded();
    for (std::size_t i = 0; i < consumed.size(); ++i)
    {
        const auto& block = consumed[i];
        std::cout << (i == 0 ? "" : ",") << '{';
        emit_u64("frame_ordinal", block.frame_ordinal, true);
        emit_u64("sequence", block.sequence);
        emit_u64("signal_id", block.signal_id);
        emit_u64("sample_idx_start", block.sample_idx_start);
        emit_u64("n_samples", block.n_samples);
        emit_u64("device_tick_start", block.device_tick_start);
        emit_u64("payload_byte_count", block.payload_byte_count);
        emit_u64("payload_digest", block.payload_digest);
        std::cout << '}';
    }
    std::cout << "]}\n";
    return nominal ? 0 : 5;
}

namespace
{

struct ResourceCounts
{
    bool supported{};
    std::uint64_t threads{};
    std::uint64_t handles{};
};

[[nodiscard]] ResourceCounts process_resources() noexcept
{
#ifdef __linux__
    const auto count = [](const char* path) noexcept -> std::optional<std::uint64_t>
    {
        std::error_code error;
        std::uint64_t value = 0;
        for (const auto& entry : std::filesystem::directory_iterator(path, error))
        {
            static_cast<void>(entry);
            ++value;
        }
        return error ? std::nullopt : std::optional<std::uint64_t>{value};
    };
    const auto threads = count("/proc/self/task");
    const auto handles = count("/proc/self/fd");
    return ResourceCounts{threads.has_value() && handles.has_value(), threads.value_or(0),
                          handles.value_or(0)};
#elif defined(_WIN32)
    DWORD n_handles = 0;
    if (!GetProcessHandleCount(GetCurrentProcess(), &n_handles))
    {
        return {};
    }
    const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE)
    {
        return {};
    }
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    std::uint64_t n_threads = 0;
    if (Thread32First(snapshot, &entry))
    {
        const auto process_id = GetCurrentProcessId();
        do
        {
            if (entry.th32OwnerProcessID == process_id)
            {
                ++n_threads;
            }
            entry.dwSize = sizeof(entry);
        } while (Thread32Next(snapshot, &entry));
    }
    CloseHandle(snapshot);
    return ResourceCounts{true, n_threads, n_handles};
#else
    return {};
#endif
}

class EndToEndClock final : public neurale::recording::UnixClock
{
  public:
    [[nodiscard]] std::uint64_t unix_nanos() noexcept override
    {
        return next_.fetch_add(1'000, std::memory_order_relaxed);
    }

  private:
    std::atomic<std::uint64_t> next_{1'786'384'800'000'000'000ULL};
};

class EndToEndSpool final : public neurale::recording::SpoolFile
{
  public:
    EndToEndSpool(std::size_t capacity, std::string_view fault) : inner_(capacity), fault_(fault) {}

    neurale::recording::SpoolIoResult append(std::span<const std::byte> bytes) noexcept override
    {
        const auto call = append_calls_.fetch_add(1, std::memory_order_relaxed) + 1;
        if (call == 4 && fault_ == "short_write")
        {
            return {neurale::recording::SpoolIoStatus::incomplete, 0, 0};
        }
        if (call == 4 && fault_ == "writer_stall")
        {
            return {neurale::recording::SpoolIoStatus::stalled, 0, 0};
        }
        if (call == 2 && (fault_ == "queue_saturation" || fault_ == "control_queue_saturation"))
        {
            injected_append_active_.store(true, std::memory_order_release);
            const auto waits = fault_ == "control_queue_saturation" ? 2'000U : 100U;
            for (std::size_t wait = 0;
                 wait < waits && !cancelled_.load(std::memory_order_acquire) &&
                 !release_injected_append_.load(std::memory_order_acquire);
                 ++wait)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            injected_append_active_.store(false, std::memory_order_release);
            if (cancelled_.load(std::memory_order_acquire))
            {
                return {neurale::recording::SpoolIoStatus::cancelled, 0, 0};
            }
        }
        return inner_.append(bytes);
    }

    neurale::recording::SpoolIoResult read_at(std::uint64_t offset,
                                              std::span<std::byte> out) noexcept override
    {
        return inner_.read_at(offset, out);
    }
    neurale::recording::SpoolIoResult sync() noexcept override
    {
        return inner_.sync();
    }
    neurale::recording::SpoolIoResult truncate(std::uint64_t bytes) noexcept override
    {
        return inner_.truncate(bytes);
    }
    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        return inner_.size();
    }
    void request_cancel() noexcept override
    {
        cancelled_.store(true, std::memory_order_release);
    }
    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return true;
    }
    [[nodiscard]] std::vector<std::byte> snapshot() const
    {
        return inner_.snapshot();
    }
    [[nodiscard]] bool injected_append_active() const noexcept
    {
        return injected_append_active_.load(std::memory_order_acquire);
    }
    void release_injected_append() noexcept
    {
        release_injected_append_.store(true, std::memory_order_release);
    }

  private:
    MemorySpoolFile inner_;
    std::string_view fault_{};
    std::atomic<std::size_t> append_calls_{};
    std::atomic<bool> cancelled_{};
    std::atomic<bool> injected_append_active_{};
    std::atomic<bool> release_injected_append_{};
};

[[nodiscard]] bool read_bytes(const std::string& path, std::vector<std::byte>& out)
{
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
    {
        return false;
    }
    const auto end = input.tellg();
    if (end < 0)
    {
        return false;
    }
    out.resize(static_cast<std::size_t>(end));
    input.seekg(0);
    return out.empty() || static_cast<bool>(input.read(reinterpret_cast<char*>(out.data()),
                                                       static_cast<std::streamsize>(out.size())));
}

[[nodiscard]] bool write_bytes(const std::string& path, std::span<const std::byte> bytes)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    return output && (bytes.empty() ||
                      static_cast<bool>(output.write(reinterpret_cast<const char*>(bytes.data()),
                                                     static_cast<std::streamsize>(bytes.size()))));
}

} // namespace

/// Replay an image through the generic runtime and a critical recorder.
///
/// This is deliberately a test executable seam. `neurale_recording_replay`
/// remains independent of the recorder target; only this integration target
/// links both sides. Python supplies the canonical second-session plan, then
/// finalizes and replays the emitted spool through the production paths.
int rerecord(const Options& options)
{
    std::vector<std::byte> plan_document;
    if (!read_bytes(options.plan_document, plan_document) || plan_document.empty())
    {
        std::cout << "{\"status\":\"plan_not_read\"}\n";
        return 2;
    }

    const auto before = process_resources();
    std::vector<std::byte> spool_bytes;
    std::uint64_t accepted = 0;
    std::uint64_t committed = 0;
    std::uint64_t spool_byte_count = 0;
    std::uint64_t outstanding_frames = 0;
    std::uint64_t outstanding_discontinuities = 0;
    std::uint64_t control_offered = 0;
    std::uint64_t control_accepted = 0;
    std::uint64_t control_committed = 0;
    std::uint64_t control_rejected = 0;
    StreamStatus joined = StreamStatus::invalid_state;
    RuntimeState final_runtime_state = RuntimeState::created;
    bool runtime_fault = false;
    bool operator_aborted = false;
    bool control_saturation_observed = false;
    FaultStage runtime_fault_stage = FaultStage::runtime;
    neurale::recording::RecorderFaultReason recorder_fault_reason =
        neurale::recording::RecorderFaultReason::none;

    {
        NativeReplaySource source;
        const auto opened = source.open(options.image.c_str());
        if (opened != ReplayImageStatus::ok)
        {
            std::cout << "{\"status\":\"image_not_opened\"}\n";
            return 2;
        }
        std::string refusal;
        auto schema = schema_from_image(source.image(), refusal);
        if (!schema.has_value())
        {
            std::cout << "{\"status\":\"schema_not_rebuilt\"}\n";
            return 3;
        }

        constexpr std::uint64_t kRerecordSession = 0x52455245434F5244ULL;
        ReplaySourceConfig replay_config;
        const auto step_paced =
            options.operator_abort_at != 0 || options.crash_after_committed != 0;
        replay_config.pacing = step_paced ? ReplayPacing::step : ReplayPacing::as_fast_as_possible;
        replay_config.blocking = step_paced;
        replay_config.replay_run_session_id = kRerecordSession;
        std::array<ReplayFaultSpec, 1> source_fault{};
        if (options.stage_fault == "source")
        {
            source_fault[0].target = ReplayFaultTarget{
                .kind = ReplayItemKind::frame, .ledger_based = true, .data_message_ordinal = 3};
            source_fault[0].effect = ReplayFaultEffect::read_failure;
            replay_config.faults = source_fault;
        }
        if (source.prepare(replay_config) != ReplayConfigStatus::ok)
        {
            std::cout << "{\"status\":\"source_not_prepared\"}\n";
            return 3;
        }

        std::vector<PlannedSignalRecording> signals;
        signals.reserve(schema->signals().size());
        std::uint64_t max_payload = 0;
        for (const auto& signal : schema->signals())
        {
            signals.push_back(PlannedSignalRecording{signal.id, signal.max_block_bytes,
                                                     signal.max_block_samples});
            max_payload += signal.max_block_bytes;
        }
        std::sort(signals.begin(), signals.end(), [](const auto& left, const auto& right)
                  { return left.signal_id < right.signal_id; });

        NativeRecordingPlan plan{};
        plan.session_id = "018f4f30-6f9d-7b36-8c61-3f96cf5c5a41";
        plan.session_uuid = {0x01, 0x8f, 0x4f, 0x30, 0x6f, 0x9d, 0x7b, 0x36,
                             0x8c, 0x61, 0x3f, 0x96, 0xcf, 0x5c, 0x5a, 0x41};
        plan.created_unix_nanos = 1'786'384'800'000'000'000ULL;
        plan.plan_document = plan_document;
        plan.plan_fingerprint = neurale::recording::sha256(plan_document);
        plan.native_session_id = kRerecordSession;
        plan.native_schema_id = schema->id();
        plan.recorded_signals = signals;
        plan.frame_queue_capacity = options.storage_fault == "queue_saturation" ? 2 : 64;
        plan.control_queue_capacity = options.storage_fault == "control_queue_saturation" ? 2 : 64;
        plan.max_blocks_per_frame = static_cast<std::uint32_t>(schema->signals().size());
        plan.max_frame_payload_bytes = max_payload;
        plan.max_signal_gaps_per_discontinuity =
            static_cast<std::uint32_t>(std::max<std::size_t>(1, Pools::gap_bound(source.image())));
        plan.max_control_payload_bytes = 4096;
        plan.max_records_per_transaction =
            options.storage_fault.empty()
                ? 256
                : std::max<std::size_t>(signals.size() + 1,
                                        plan.max_signal_gaps_per_discontinuity + 1);
        plan.max_transaction_bytes =
            std::max<std::size_t>(4U << 20U, static_cast<std::size_t>(max_payload));
        plan.checkpoint_interval_transactions = 0;
        plan.worker_idle_poll_nanos = 50'000;
        plan.drain_timeout_nanos = 5'000'000'000ULL;
        plan.durability_policy = neurale::recording::DurabilityPolicy::buffered;

        const auto image_bytes = std::filesystem::file_size(options.image);
        const auto capacity = static_cast<std::size_t>(std::max<std::uintmax_t>(
            options.crash_after_committed == 0 ? (16U << 20U) : (4U << 20U),
            image_bytes * 4U + (1U << 20U)));
        std::unique_ptr<EndToEndSpool> memory_spool;
        std::unique_ptr<BoundedMappedSpoolFile> mapped_spool;
        neurale::recording::SpoolFile* spool = nullptr;
        if (options.crash_after_committed == 0)
        {
            memory_spool = std::make_unique<EndToEndSpool>(capacity, options.storage_fault);
            spool = memory_spool.get();
        }
        else
        {
            mapped_spool = std::make_unique<BoundedMappedSpoolFile>();
            const auto created = mapped_spool->create(options.spool_output, capacity);
            if (created.status != neurale::recording::SpoolIoStatus::ok)
            {
                std::cout << "{\"status\":\"mapped_spool_not_created\",\"platform_error\":"
                          << created.platform_error << "}\n";
                return 3;
            }
            spool = mapped_spool.get();
        }
        EndToEndClock clock;
        NativeRecorderCore recorder;
        if (recorder.prepare(plan, *spool, clock) != neurale::recording::RecorderStatusCode::ok)
        {
            std::cout << "{\"status\":\"recorder_not_prepared\"}\n";
            return 3;
        }

        RealtimeConfig realtime;
        realtime.pool_capacity.source_owned = 1;
        realtime.pool_capacity.ingress_capacity = 16;
        realtime.pool_capacity.processor_owned = 2;
        const auto edge_capacity = 64U;
        realtime.pool_capacity.critical_edge_capacity = edge_capacity;
        realtime.pool_capacity.actuator_owned = 1;
        realtime.pool_capacity.observer_edge_capacity = edge_capacity;
        realtime.buffer_size = std::max<std::size_t>(Pools::payload_bound(source.image()),
                                                     static_cast<std::size_t>(max_payload));
        realtime.max_signal_blocks =
            std::max<std::size_t>(Pools::block_bound(source.image()), signals.size());
        realtime.discontinuity_capacity = 16;
        realtime.gaps_per_discontinuity =
            std::max<std::size_t>(Pools::gap_bound(source.image()), 16);
        realtime.max_process_outputs = 1;
        realtime.fault_history_capacity = 16;
        if (options.stage_fault == "watchdog")
        {
            realtime.watchdog_period = RealtimeDuration{500'000};
            realtime.processor_execution_deadline = RealtimeDuration{1'000'000};
        }

        std::string processor_mode = options.stage_fault;
        if (options.operator_abort_at != 0 || options.crash_after_committed != 0 ||
            options.storage_fault == "control_queue_saturation")
        {
            processor_mode = "pace";
        }
        RecordingProcessor processor{std::move(processor_mode)};
        RecordingConsumer consumer{options.stage_fault};
        consumer.reserve(static_cast<std::size_t>(source.image().summary().n_blocks) + 8);
        NativeStreamRunner runtime{std::move(*schema), realtime, source, processor, consumer};
        const ObserverEdgeConfig edge{.id = 60012,
                                      .capacity = edge_capacity,
                                      .drop_history_capacity = 16,
                                      .drop_policy = ObserverDropPolicy::fault,
                                      .critical_recorder = true};
        const auto attached = runtime.add_critical_observer(recorder, edge);
        const auto prepared = attached == StreamStatus::ok ? runtime.prepare() : attached;
        const auto armed = prepared == StreamStatus::ok ? runtime.arm() : prepared;
        const auto started = armed == StreamStatus::ok ? runtime.start() : armed;
        if (attached != StreamStatus::ok || prepared != StreamStatus::ok ||
            armed != StreamStatus::ok || started != StreamStatus::ok)
        {
            std::cout << "{\"status\":\"runtime_refused\"}\n";
            return 3;
        }
        if (options.crash_after_committed != 0)
        {
            // Permit exactly the prefix the parent intends to observe. Granting
            // one extra item made the status line a racing snapshot: that item
            // could commit while stdout was being flushed, so recovery could
            // legitimately expose more than the line had reported. With no next
            // permit the source remains live and blocked in step pacing while the
            // committed prefix is stable.
            const auto permits = source.advance(options.crash_after_committed);
            if (permits != options.crash_after_committed)
            {
                std::cout << "{\"status\":\"crash_target_not_reachable\"}\n";
                return 3;
            }
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (recorder.status().spool_committed < options.crash_after_committed &&
                   runtime.state() == RuntimeState::running &&
                   std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            const auto crash_status = recorder.status();
            if (crash_status.spool_committed != options.crash_after_committed ||
                crash_status.runtime_accepted != options.crash_after_committed ||
                runtime.state() != RuntimeState::running)
            {
                std::cout << "{\"status\":\"crash_target_not_committed\",\"spool_committed\":"
                          << crash_status.spool_committed << "}\n";
                return 3;
            }
            std::cout << "{\"status\":\"recording_active\",\"spool_committed\":"
                      << crash_status.spool_committed
                      << ",\"runtime_accepted\":" << crash_status.runtime_accepted << "}"
                      << std::endl;
            for (;;)
            {
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
        if (options.operator_abort_at != 0 &&
            source.advance(options.operator_abort_at) != options.operator_abort_at)
        {
            std::cout << "{\"status\":\"abort_target_not_reachable\"}\n";
            return 3;
        }
        if (options.storage_fault == "control_queue_saturation")
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!memory_spool->injected_append_active() &&
                   std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            constexpr std::array<std::byte, 2> body{std::byte{'{'}, std::byte{'}'}};
            while (memory_spool->injected_append_active())
            {
                const auto identity = control_offered + 1;
                ++control_offered;
                const neurale::recording::ControlSubmission submission{
                    .kind = neurale::recording::ProducerIdentityKind::events,
                    .identity = identity,
                    .clock_domain = 0,
                    .time_ns = identity,
                    .body = body,
                };
                if (!recorder.submit_control(submission))
                {
                    control_saturation_observed = true;
                    memory_spool->release_injected_append();
                    break;
                }
            }
        }

        if (options.operator_abort_at != 0)
        {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (recorder.status().runtime_accepted < options.operator_abort_at &&
                   runtime.state() == RuntimeState::running &&
                   std::chrono::steady_clock::now() < deadline)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            if (recorder.status().runtime_accepted >= options.operator_abort_at)
            {
                operator_aborted = runtime.abort() == StreamStatus::ok;
            }
        }

        joined = runtime.join();
        static_cast<void>(runtime.stop());
        if (options.operator_abort_at != 0 && operator_aborted)
        {
            operator_aborted =
                recorder.abort("operator-abort") == neurale::recording::RecorderStatusCode::ok;
        }
        outstanding_frames = runtime.outstanding_frames();
        outstanding_discontinuities = runtime.outstanding_discontinuities();
        const auto primary_fault = runtime.primary_fault();
        runtime_fault = primary_fault.has_value();
        if (primary_fault.has_value())
        {
            runtime_fault_stage = primary_fault->stage;
        }
        final_runtime_state = runtime.state();
        static_cast<void>(runtime.detach_observer(edge.id));
        static_cast<void>(recorder.close());
        const auto recorder_status = recorder.status();
        accepted = recorder_status.runtime_accepted;
        committed = recorder_status.spool_committed;
        control_offered = recorder_status.control_offered;
        control_accepted = recorder_status.control_accepted;
        control_committed = recorder_status.control_spool_committed;
        control_rejected = recorder_status.control_rejected;
        recorder_fault_reason = recorder_status.primary_fault.reason;
        spool_bytes = memory_spool->snapshot();
        spool_byte_count = spool_bytes.size();
    }

    const auto after = process_resources();
    if (!write_bytes(options.spool_output, spool_bytes))
    {
        std::cout << "{\"status\":\"spool_not_written\"}\n";
        return 4;
    }
    std::cout << '{';
    const auto nominal = joined == StreamStatus::ok && !runtime_fault &&
                         final_runtime_state == RuntimeState::stopped;
    const auto aborted =
        operator_aborted && !runtime_fault && final_runtime_state == RuntimeState::stopped;
    emit_string("status", aborted ? "aborted" : (nominal ? "ok" : "faulted"), true);
    emit_string("join", status_text(joined));
    emit_string("runtime_state", runtime_state_text(final_runtime_state));
    emit_u64("runtime_accepted", accepted);
    emit_u64("spool_committed", committed);
    emit_u64("control_offered", control_offered);
    emit_u64("control_accepted", control_accepted);
    emit_u64("control_spool_committed", control_committed);
    emit_u64("control_rejected", control_rejected);
    emit_u64("spool_bytes", spool_byte_count);
    emit_u64("outstanding_frames", outstanding_frames);
    emit_u64("outstanding_discontinuities", outstanding_discontinuities);
    emit_string("fault_stage", runtime_fault ? fault_stage_text(runtime_fault_stage) : "none");
    emit_string("recorder_fault_reason", neurale::recording::to_string(recorder_fault_reason));
    emit_bool("control_saturation_observed", control_saturation_observed);
    emit_bool("operator_aborted", operator_aborted);
    emit_bool("resource_counts_supported", before.supported && after.supported);
    emit_u64("threads_before", before.threads);
    emit_u64("threads_after", after.threads);
    emit_u64("handles_before", before.handles);
    emit_u64("handles_after", after.handles);
    std::cout << "}\n";
    const auto expected_fault = !options.stage_fault.empty() || !options.storage_fault.empty();
    const auto fault_matches =
        expected_fault && runtime_fault &&
        (options.storage_fault != "control_queue_saturation" || control_saturation_observed);
    return nominal || fault_matches || aborted ? 0 : 5;
}

/// The schema a process holding nothing but the image can recover, as JSON.
///
/// Its point is what it is compared *against*. Checking a rebuilt schema
/// against the image it came from establishes only that the image is
/// self-consistent, which the image-construction step already covers. Printing it lets the Python suite
/// compare it with the schema the *recording declared* -- signal geometry,
/// dtypes, layouts, rates, clock domains, feature-set descriptors with their
/// source linkage, and units -- so a value that the recorder, the finalizer or
/// the image builder rewrote consistently is still caught.
///
/// Enumerations are printed as their numeric values on purpose: their spellings
/// come from the image, so comparing spellings would compare the image with
/// itself, while the number can be compared with what Python declared.
int schema(const Options& options)
{
    ReplayImageFile image;
    const auto opened = image.open(options.image.c_str());
    if (opened != ReplayImageStatus::ok)
    {
        std::cout << "{\"status\":\"" << neurale::recording::replay_image_status_text(opened)
                  << "\"}\n";
        return 2;
    }
    std::string refusal;
    const auto rebuilt = schema_from_image(image, refusal);
    if (!rebuilt.has_value())
    {
        std::cout << "{\"status\":\"schema_not_rebuilt\",\"detail\":\"" << refusal << "\"}\n";
        return 3;
    }

    std::cout << '{';
    emit_string("status", "ok", true);
    emit_u64("schema_id", rebuilt->id());
    std::cout << ",\"signals\":[";
    bool first = true;
    for (const auto& signal : rebuilt->signals())
    {
        std::cout << (first ? "" : ",") << '{';
        first = false;
        emit_u64("id", signal.id, true);
        emit_u64("dtype", static_cast<std::uint64_t>(signal.dtype));
        emit_u64("layout", static_cast<std::uint64_t>(signal.layout));
        emit_u64("kind", static_cast<std::uint64_t>(signal.kind));
        emit_u64("n_channels", signal.n_channels);
        emit_u64("nominal_block_samples", signal.nominal_block_samples);
        emit_u64("max_block_samples", signal.max_block_samples);
        emit_u64("rate_numerator", signal.fs.numerator);
        emit_u64("rate_denominator", signal.fs.denominator);
        emit_u64("clock_domain", signal.clock_domain);
        emit_u64("device_tick_tracking", static_cast<std::uint64_t>(signal.device_tick_tracking));
        emit_u64("physical_unit", static_cast<std::uint64_t>(signal.physical_unit));
        emit_u64("observation_timing", static_cast<std::uint64_t>(signal.observation_timing));
        emit_u64("feature_set_id", signal.feature_set_id);
        emit_u64("channel_set_id", signal.channel_set_id);
        emit_u64("calibration_id", signal.calibration_id);
        emit_u64("reference_id", signal.reference_id);
        emit_u64("fixed_block_bytes", signal.fixed_block_bytes);
        emit_u64("max_block_bytes", signal.max_block_bytes);
        std::cout << '}';
    }
    std::cout << "],\"feature_sets\":[";
    first = true;
    for (const auto& feature : rebuilt->feature_sets().descriptors())
    {
        std::cout << (first ? "" : ",") << '{';
        first = false;
        emit_u64("id", feature.id, true);
        emit_u64("source_stream_id", feature.source_stream_id);
        emit_string("source_stream", feature.source_stream);
        emit_string("algorithm_name", feature.algorithm_name);
        emit_string("algorithm_version", feature.algorithm_version);
        emit_u64("window_length_ns", feature.window_length_ns);
        emit_u64("shift_ns", feature.shift_ns);
        emit_u64("timestamp_reference", static_cast<std::uint64_t>(feature.timestamp_reference));
        std::cout << ",\"feature_names\":[";
        for (std::size_t i = 0; i < feature.feature_names.size(); ++i)
        {
            std::cout << (i == 0 ? "" : ",") << '"' << feature.feature_names[i] << '"';
        }
        std::cout << "],\"unit_ids\":[";
        for (std::size_t i = 0; i < feature.unit_ids.size(); ++i)
        {
            std::cout << (i == 0 ? "" : ",") << feature.unit_ids[i];
        }
        std::cout << "]}";
    }
    std::cout << "],\"units\":[";
    first = true;
    for (const auto& unit : rebuilt->units().units())
    {
        std::cout << (first ? "" : ",") << '{';
        first = false;
        emit_u64("id", unit.id, true);
        emit_string("symbol", unit.symbol);
        emit_string("description", unit.description);
        std::cout << '}';
    }
    std::cout << "]}\n";
    return 0;
}

/// The steady-state allocation count of a run over a *real* image.
///
/// The steady-state allocation gate elsewhere runs the same measurement over an image this repository's C++
/// tests wrote. That answers the question for a shape someone chose; this one
/// answers it for the shape a recording actually produced -- multi-block
/// frames, three signals, whatever payload sizes the session had.
int allocate(const Options& options)
{
    NativeReplaySource source;
    if (source.open(options.image.c_str()) != ReplayImageStatus::ok)
    {
        std::cout << "{\"status\":\"unreadable\"}\n";
        return 2;
    }
    ReplaySourceConfig config;
    config.pacing = ReplayPacing::as_fast_as_possible;
    if (source.prepare(config) != ReplayConfigStatus::ok)
    {
        std::cout << "{\"status\":\"not_prepared\"}\n";
        return 3;
    }

    Pools pools{source.image()};
    std::vector<Observed> observed;
    observed.reserve(source.image().item_count() + 8);
    // Warm up outside the measured interval: the first read of a run touches
    // pages, and the vector above must have grown to its final size before a
    // single allocation is allowed to count.
    static_cast<void>(drain(source, pools, observed, source.image().item_count() + 8));
    if (source.reset() != StreamStatus::ok)
    {
        std::cout << "{\"status\":\"not_reset\"}\n";
        return 3;
    }

    std::size_t emitted = 0;
    neurale::benchmark::reset_allocation_count();
    neurale::benchmark::set_allocation_tracking(true);
    for (std::size_t step = 0; step < source.image().item_count(); ++step)
    {
        FrameLease frame;
        DiscontinuityLease discontinuity;
        if (pools.frames.try_acquire(frame) != StreamStatus::ok ||
            pools.discontinuities.try_acquire(discontinuity) != StreamStatus::ok)
        {
            break;
        }
        const auto status = source.read_message(frame.frame(), discontinuity);
        if (status != StreamStatus::ok && status != StreamStatus::discontinuity)
        {
            break;
        }
        ++emitted;
    }
    neurale::benchmark::set_allocation_tracking(false);
    const auto allocations = neurale::benchmark::allocation_count();

    std::cout << '{';
    emit_string("case", "end_to_end_steady_state", true);
    emit_u64("items", emitted);
    emit_u64("allocations", allocations);
    emit_string("backend", neurale::benchmark::allocation_tracking_backend());
    std::cout << "}\n";
    return allocations == 0 ? 0 : 2;
}

/// The CTest entry: the harness's own reporting, over an image built here.
///
/// It is deliberately small. Everything this file exists for is checked by the
/// Python suite that drives it, and that suite is skipped in a checkout with no
/// C++ build -- so what remains here is what would silently rot otherwise: the
/// fault command line, the schema rebuild, and the claim that a reset run is
/// reported as the same run.
int self_test()
{
    neurale::recording::test::Scratch scratch{"neurale-end-to-end"};

    {
        ReplayFaultSpec spec;
        std::string storage;
        CHECK(parse_fault("ledger:frame:7:read_failure", spec, storage));
        CHECK(spec.target.ledger_based);
        CHECK(spec.target.kind == ReplayItemKind::frame);
        CHECK(spec.target.data_message_ordinal == 7);
        CHECK(spec.effect == ReplayFaultEffect::read_failure);

        CHECK(parse_fault("stream:discontinuity:neural:2:stall:1500", spec, storage));
        CHECK(!spec.target.ledger_based);
        CHECK(spec.target.kind == ReplayItemKind::discontinuity);
        CHECK(spec.target.stream_id == "neural");
        CHECK(spec.target.ordinal == 2);
        CHECK(spec.effect == ReplayFaultEffect::stall);
        CHECK(spec.stall_ns == 1500);

        CHECK(!parse_fault("ledger:frame:7", spec, storage));
        CHECK(!parse_fault("nowhere:frame:7:stall:1", spec, storage));
        CHECK(!parse_fault("ledger:frame:7:combust", spec, storage));
    }

    {
        // A run and the same run again after reset are reported identical, and
        // a run whose payloads differ is not -- the comparison has to be able
        // to fail, or "identical" would mean nothing.
        neurale::recording::test::ReplayImageBuilder builder;
        for (std::uint64_t i = 0; i < 4; ++i)
        {
            neurale::recording::test::BuiltFrame frame;
            frame.data_message_ordinal = i;
            frame.replay_sequence = i;
            frame.original_sequence = i;
            frame.timeline_ns = i * 1'000;
            frame.payload = {static_cast<std::byte>(i), static_cast<std::byte>(i + 1)};
            neurale::recording::test::BuiltBlock block;
            block.payload_byte_count = frame.payload.size();
            block.n_samples = 2;
            block.sample_idx_start = i * 2;
            frame.blocks.push_back(block);
            builder.add_frame(std::move(frame));
        }
        const auto path = scratch.path("self.nrimg");
        CHECK(builder.write(path));

        NativeReplaySource source;
        CHECK(source.open(path.c_str()) == ReplayImageStatus::ok);
        ReplaySourceConfig config;
        CHECK(source.prepare(config) == ReplayConfigStatus::ok);
        Pools pools{source.image()};
        std::vector<Observed> first;
        std::vector<Observed> second;
        CHECK(drain(source, pools, first, 16) == StreamStatus::end_of_stream);
        CHECK(first.size() == 4);
        CHECK(source.reset() == StreamStatus::ok);
        CHECK(drain(source, pools, second, 16) == StreamStatus::end_of_stream);
        CHECK(first == second);

        second[2].blocks[0].payload_crc ^= 1U;
        CHECK(!(first == second));
    }

    {
        // The schema rebuild is the runner entry's precondition, and an image
        // whose schema section this build cannot spell must refuse rather than
        // hand the runtime a schema it invented.
        neurale::recording::test::ReplayImageBuilder builder;
        neurale::recording::test::BuiltSignal signal;
        signal.dtype = "quantum48";
        builder.signals.push_back(signal);
        builder.add_frame(neurale::recording::test::BuiltFrame{});
        const auto path = scratch.path("unspellable.nrimg");
        CHECK(builder.write(path));

        ReplayImageFile image;
        CHECK(image.open(path.c_str()) == ReplayImageStatus::ok);
        std::string refusal;
        CHECK(!schema_from_image(image, refusal).has_value());
        CHECK(!refusal.empty());
    }

    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    std::string mode;
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument = argv[i];
        if (argument == "--self-test")
        {
            mode = argument;
            continue;
        }
        if (argument == "--replay" || argument == "--runner" || argument == "--allocate" ||
            argument == "--schema")
        {
            if (i + 1 >= argc)
            {
                std::cerr << argument << " needs an image path\n";
                return 2;
            }
            mode = argument;
            options.image = argv[++i];
            continue;
        }
        if (argument == "--rerecord")
        {
            if (i + 3 >= argc)
            {
                std::cerr << "--rerecord needs image, plan-document, and spool-output paths\n";
                return 2;
            }
            mode = argument;
            options.image = argv[++i];
            options.plan_document = argv[++i];
            options.spool_output = argv[++i];
            continue;
        }
        if (argument == "--operator-abort-at")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--operator-abort-at needs a positive runtime acceptance ordinal\n";
                return 2;
            }
            options.operator_abort_at = std::stoull(argv[++i]);
            if (options.operator_abort_at == 0)
            {
                std::cerr << "--operator-abort-at must be positive\n";
                return 2;
            }
            continue;
        }
        if (argument == "--reset")
        {
            options.reset = true;
            continue;
        }
        if (argument == "--cancel")
        {
            options.cancel_replay = true;
            continue;
        }
        if (argument == "--crash-after-committed")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--crash-after-committed needs a positive message count\n";
                return 2;
            }
            options.crash_after_committed = std::stoull(argv[++i]);
            if (options.crash_after_committed == 0)
            {
                std::cerr << "--crash-after-committed must be positive\n";
                return 2;
            }
            continue;
        }
        if (argument == "--stage-fault")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--stage-fault needs source, processor, flush, consumer, actuator, or "
                             "watchdog\n";
                return 2;
            }
            options.stage_fault = argv[++i];
            if (options.stage_fault != "source" && options.stage_fault != "processor" &&
                options.stage_fault != "flush" && options.stage_fault != "consumer" &&
                options.stage_fault != "actuator" && options.stage_fault != "watchdog")
            {
                std::cerr << "unsupported --stage-fault value\n";
                return 2;
            }
            continue;
        }
        if (argument == "--storage-fault")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--storage-fault needs short_write, writer_stall, queue_saturation, "
                             "or control_queue_saturation\n";
                return 2;
            }
            options.storage_fault = argv[++i];
            if (options.storage_fault != "short_write" && options.storage_fault != "writer_stall" &&
                options.storage_fault != "queue_saturation" &&
                options.storage_fault != "control_queue_saturation")
            {
                std::cerr << "unsupported --storage-fault value\n";
                return 2;
            }
            continue;
        }
        if (argument == "--fault")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--fault needs a specification\n";
                return 2;
            }
            options.faults.emplace_back(argv[++i]);
            continue;
        }
        if (argument == "--session-id")
        {
            if (i + 1 >= argc)
            {
                std::cerr << "--session-id needs a value\n";
                return 2;
            }
            options.session_id = std::stoull(argv[++i]);
            continue;
        }
        std::cerr << "unknown end-to-end option " << argument << '\n';
        return 2;
    }

    if (mode == "--self-test")
    {
        const int status = self_test();
        if (status == 0)
        {
            std::cout << "recording end-to-end harness ok\n";
            return 0;
        }
        return 1;
    }
    if (mode == "--replay")
    {
        return replay(options);
    }
    if (mode == "--runner")
    {
        return runner(options);
    }
    if (mode == "--allocate")
    {
        return allocate(options);
    }
    if (mode == "--schema")
    {
        return schema(options);
    }
    if (mode == "--rerecord")
    {
        return rerecord(options);
    }
    std::cerr << "usage: neurale_recording_end_to_end_test "
                 "(--self-test | --replay <image> [--reset] [--fault <spec>] [--session-id <n>] "
                 "| --runner <image> [--cancel] | --allocate <image> | --schema <image> "
                 "| --rerecord <image> <plan-document> <spool-output> "
                 "[--stage-fault <stage>] [--storage-fault <fault>] "
                 "[--operator-abort-at <ordinal>] [--crash-after-committed <count>])\n";
    return 2;
}
