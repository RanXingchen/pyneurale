/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// \file
/// The private Python surface over the native spool reader.
///
/// It exists so the container has exactly one reader. `_spool_format.py` used
/// to hold a second one -- its own CRC-32C, its own scanner, and its own
/// payload decoders -- kept in step with `spool_scanner.cpp` only by both
/// happening to pass the same frozen vectors. Two readers of one private layout
/// is the arrangement where a layout change followed by one side and not the
/// other leaves both self-consistent and disagreeing, and the spool is the only
/// copy of a recording.
///
/// What crosses this boundary is *layout decoding only*, in the same sense as
/// `replay_image_bindings.cpp`: raw enumerator values stay raw and flag words
/// stay words. Naming a gap reason or a control kind is the NRF layer's
/// vocabulary, not the container's, and stays in Python.
///
/// The record walk is a cursor rather than a list. A finalization reads the
/// committed prefix twice -- once to count what the session will hold, once to
/// write it -- and two bounded passes over a file cost a fixed buffer, while
/// one materialized pass costs the whole spool resident in Python objects.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/record_payloads.h"
#include "crc32c.h"
#include "spool_file.h"
#include "spool_layout.h"
#include "spool_scanner.h"

namespace py = pybind11;

using namespace neurale::recording;

namespace
{

/// A read-only `SpoolFile` over bytes Python already holds.
///
/// The scanner takes a `SpoolFile`, and a caller that has the spool in memory
/// -- a test vector, a spool snapshot taken from the memory backend -- would
/// otherwise have to write it to disk to read it. Nothing is copied: the span
/// points into the owner's buffer, which `PySpoolScan` keeps alive for as long
/// as the file exists.
class BorrowedSpoolFile final : public SpoolFile
{
  public:
    explicit BorrowedSpoolFile(std::span<const std::byte> data) noexcept : data_{data} {}

    SpoolIoResult append(std::span<const std::byte>) noexcept override
    {
        return {SpoolIoStatus::io_error, 0};
    }

    SpoolIoResult read_at(std::uint64_t offset, std::span<std::byte> out) noexcept override
    {
        if (offset > data_.size())
        {
            return {SpoolIoStatus::incomplete, 0};
        }
        const auto available = data_.size() - static_cast<std::size_t>(offset);
        const auto take = std::min(available, out.size());
        std::memcpy(out.data(), data_.data() + offset, take);
        return {take == out.size() ? SpoolIoStatus::ok : SpoolIoStatus::incomplete, take};
    }

    SpoolIoResult sync() noexcept override
    {
        return {SpoolIoStatus::io_error, 0};
    }

    SpoolIoResult truncate(std::uint64_t) noexcept override
    {
        return {SpoolIoStatus::io_error, 0};
    }

    [[nodiscard]] std::uint64_t size() const noexcept override
    {
        return static_cast<std::uint64_t>(data_.size());
    }

    [[nodiscard]] bool supports_bounded_cancel() const noexcept override
    {
        return true;
    }

  private:
    std::span<const std::byte> data_;
};

/// One committed record's fixed payload, decoded, plus the variable-length
/// bytes that follow it where the kind has any.
struct PySignalBlockPayload : SignalBlockPayloadFields
{
    py::bytes samples;
};

struct PyControlPayload : ControlPayloadFields
{
    py::bytes body;
};

/// One committed record: its position and header, and the decoded payload for
/// the kinds this version defines a payload for.
///
/// `payload` is `None` in two cases, which `kind` and `payload_bytes` tell
/// apart. For a checkpoint, an accounting snapshot, and a session-end record it
/// is None because the scan already read all three -- their content is in the
/// report -- so decoding them again here would be a second reading of the same
/// bytes with nowhere to disagree usefully. For any kind whose payload is
/// shorter than that kind fixes it is None because there is nothing to decode:
/// the scan has already recorded that as a finding, and raising here instead
/// would make the one spool a caller most needs to walk the one it cannot.
struct PySpoolRecord
{
    std::uint64_t transaction_id{};
    std::uint64_t transaction_offset{};
    std::uint64_t offset{};
    std::uint64_t payload_offset{};
    std::uint16_t kind{};
    std::uint64_t logical_ordinal{};
    std::uint64_t record_unix_nanos{};
    std::uint32_t payload_bytes{};
    py::object payload{py::none()};
};

class PySpoolScan;

/// Walks one scanned spool's committed records. Independent of every other
/// walk of the same scan: a finalization makes one to count and another to
/// write, and neither may disturb the other's position.
class PySpoolRecords
{
  public:
    PySpoolRecords(py::object owner, SpoolFile& file, const SpoolScanReport& report)
        : owner_{std::move(owner)}, cursor_{file, report},
          // A scan that established no committed prefix has nothing to walk,
          // and that is an empty walk rather than an error: a spool whose
          // superblock did not validate holds no promotable record, which is a
          // verdict the scan already reported. The cursor refuses such a report
          // by starting failed, so the distinction has to be drawn here --
          // otherwise a genuine read failure part-way through a good prefix
          // would be indistinguishable from a spool with no prefix at all.
          walkable_{report.readable() && !report.read_failed() && !report.findings_truncated()}
    {
    }

    [[nodiscard]] PySpoolRecord next();

  private:
    py::object owner_;
    SpoolRecordCursor cursor_;
    bool walkable_{};
    std::vector<std::byte> buffer_;
};

/// One scanned spool, and the file it was scanned from.
///
/// The file is held for the object's whole life because the record cursor
/// reads through it: a scan whose file closed at the end of the scan would
/// report a committed prefix nobody could then read.
class PySpoolScan
{
  public:
    /// Scan bytes Python holds. *owner* is kept alive so the borrowed span
    /// stays valid; it is the buffer object itself, not a copy of it.
    static std::unique_ptr<PySpoolScan> from_bytes(const py::buffer& source)
    {
        const auto info = source.request(false);
        if (info.ndim != 1 || info.itemsize != 1)
        {
            throw std::invalid_argument("a spool must be scanned from a flat byte buffer");
        }
        auto scan = std::unique_ptr<PySpoolScan>{new PySpoolScan{}};
        scan->owner_ = py::reinterpret_borrow<py::object>(source);
        const auto span = std::span<const std::byte>{static_cast<const std::byte*>(info.ptr),
                                                     static_cast<std::size_t>(info.size)};
        scan->file_ = std::make_unique<BorrowedSpoolFile>(span);
        scan->run();
        return scan;
    }

    /// Scan a spool on disk. Opened read-only: a read of a spool never writes,
    /// never truncates and never repairs (contract section 4.5), and the mode
    /// is what makes that structural rather than a matter of discipline.
    static std::unique_ptr<PySpoolScan> from_path(const std::string& path)
    {
        auto scan = std::unique_ptr<PySpoolScan>{new PySpoolScan{}};
        auto file = std::make_unique<PlatformSpoolFile>();
        const auto opened = file->open_existing(path, SpoolOpenMode::read_only);
        if (opened.status != SpoolIoStatus::ok)
        {
            throw std::runtime_error("could not open the spool for reading: " + path);
        }
        scan->file_ = std::move(file);
        scan->run();
        return scan;
    }

    [[nodiscard]] const SpoolScanReport& report() const noexcept
    {
        return report_;
    }

    [[nodiscard]] SpoolFile& file() const noexcept
    {
        return *file_;
    }

    [[nodiscard]] bool close() noexcept
    {
        auto* file = dynamic_cast<PlatformSpoolFile*>(file_.get());
        return file == nullptr || file->close().status == SpoolIoStatus::ok;
    }

    /// The plan document, read from the range the superblock declares. Read on
    /// demand rather than at scan time: it is the one unbounded field, a caller
    /// that only wants the verdict never needs it, and the scan has already
    /// checked these bytes against the stored CRC and fingerprint.
    [[nodiscard]] py::bytes plan_document() const
    {
        const auto length = static_cast<std::size_t>(report_.plan_document_bytes());
        std::vector<std::byte> buffer(length);
        if (length != 0)
        {
            const auto read = file_->read_at(report_.plan_document_offset(),
                                             std::span<std::byte>{buffer.data(), length});
            if (read.status != SpoolIoStatus::ok)
            {
                throw std::runtime_error("the spool's plan document could not be read");
            }
        }
        return py::bytes(reinterpret_cast<const char*>(buffer.data()), length);
    }

  private:
    PySpoolScan() = default;

    void run()
    {
        // The scan is bounded but reads the whole committed prefix, so a large
        // spool spends real time in `read_at`. Nothing it touches is a Python
        // object: the borrowed buffer is an immutable `bytes`, held by
        // `owner_`, and the platform file is this object's own.
        py::gil_scoped_release released;
        report_ = scan_spool(*file_);
    }

    py::object owner_{py::none()};
    std::unique_ptr<SpoolFile> file_;
    SpoolScanReport report_{};
};

PySpoolRecord PySpoolRecords::next()
{
    SpoolRecordView view{};
    if (!walkable_)
    {
        throw py::stop_iteration();
    }
    if (!cursor_.next(view))
    {
        if (cursor_.failed())
        {
            throw std::runtime_error("the spool's committed prefix could not be read");
        }
        throw py::stop_iteration();
    }

    PySpoolRecord record{};
    record.transaction_id = view.transaction_id;
    record.transaction_offset = view.transaction_offset;
    record.offset = view.record_offset;
    record.payload_offset = view.payload_offset;
    record.kind = view.kind;
    record.logical_ordinal = view.logical_ordinal;
    record.record_unix_nanos = view.record_unix_nanos;
    record.payload_bytes = view.payload_bytes;

    buffer_.resize(view.payload_bytes);
    if (view.payload_bytes != 0 &&
        !cursor_.read_payload(view, std::span<std::byte>{buffer_.data(), buffer_.size()}))
    {
        throw std::runtime_error("a committed record's payload could not be read");
    }
    const auto payload = std::span<const std::byte>{buffer_.data(), buffer_.size()};

    switch (static_cast<RecordKind>(view.kind))
    {
    case RecordKind::frame:
    {
        FramePayloadFields fields{};
        if (decode_frame_payload(payload, fields))
        {
            record.payload = py::cast(fields);
        }
        break;
    }
    case RecordKind::signal_block:
    {
        PySignalBlockPayload fields{};
        if (decode_signal_block_payload(payload, fields))
        {
            const auto header = kSignalBlockHeaderPayloadBytes;
            fields.samples = py::bytes(reinterpret_cast<const char*>(payload.data() + header),
                                       payload.size() - header);
            record.payload = py::cast(std::move(fields));
        }
        break;
    }
    case RecordKind::discontinuity:
    {
        DiscontinuityPayloadFields fields{};
        if (decode_discontinuity_payload(payload, fields))
        {
            record.payload = py::cast(fields);
        }
        break;
    }
    case RecordKind::signal_gap:
    {
        SignalGapPayloadFields fields{};
        if (decode_signal_gap_payload(payload, fields))
        {
            record.payload = py::cast(fields);
        }
        break;
    }
    case RecordKind::control:
    {
        PyControlPayload fields{};
        if (decode_control_payload(payload, fields))
        {
            const auto header = kControlHeaderPayloadBytes;
            fields.body = py::bytes(reinterpret_cast<const char*>(payload.data() + header),
                                    payload.size() - header);
            record.payload = py::cast(std::move(fields));
        }
        break;
    }
    case RecordKind::fault:
    {
        FaultPayloadFields fields{};
        if (decode_fault_payload(payload, fields))
        {
            record.payload = py::cast(fields);
        }
        break;
    }
    default:
        // A checkpoint, an accounting snapshot, or a session-end record. The
        // scan has already read all three; see `PySpoolRecord`.
        break;
    }
    return record;
}

void bind_payloads(py::module_& recording)
{
    py::class_<FramePayloadFields>(recording, "SpoolFramePayload", py::is_final())
        .def_readonly("data_message_ordinal", &FramePayloadFields::data_message_ordinal)
        .def_readonly("native_session_id", &FramePayloadFields::native_session_id)
        .def_readonly("frame_sequence", &FramePayloadFields::frame_sequence)
        .def_readonly("frame_ordinal", &FramePayloadFields::frame_ordinal)
        .def_readonly("host_received_ns", &FramePayloadFields::host_received_ns)
        .def_readonly("source_tick", &FramePayloadFields::source_tick)
        .def_readonly("valid_until_ns", &FramePayloadFields::valid_until_ns)
        .def_readonly("total_payload_byte_count", &FramePayloadFields::total_payload_byte_count)
        .def_readonly("runtime_accepted_host_time_ns",
                      &FramePayloadFields::runtime_accepted_host_time_ns)
        .def_readonly("native_schema_id", &FramePayloadFields::native_schema_id)
        .def_readonly("source_clock_domain", &FramePayloadFields::source_clock_domain)
        .def_readonly("frame_flags", &FramePayloadFields::frame_flags)
        .def_readonly("signal_block_count", &FramePayloadFields::signal_block_count)
        .def_readonly("recorded_signal_block_count",
                      &FramePayloadFields::recorded_signal_block_count);

    py::class_<PySignalBlockPayload>(recording, "SpoolSignalBlockPayload", py::is_final())
        .def_readonly("signal_block_ordinal", &PySignalBlockPayload::signal_block_ordinal)
        .def_readonly("data_message_ordinal", &PySignalBlockPayload::data_message_ordinal)
        .def_readonly("frame_ordinal", &PySignalBlockPayload::frame_ordinal)
        .def_readonly("sample_idx_start", &PySignalBlockPayload::sample_idx_start)
        .def_readonly("last_sample_idx", &PySignalBlockPayload::last_sample_idx)
        .def_readonly("device_tick_start", &PySignalBlockPayload::device_tick_start)
        .def_readonly("observation_time_start_ns", &PySignalBlockPayload::observation_time_start_ns)
        .def_readonly("payload_offset", &PySignalBlockPayload::payload_offset)
        .def_readonly("payload_byte_count", &PySignalBlockPayload::payload_byte_count)
        .def_readonly("clock_sync_device_tick_reference",
                      &PySignalBlockPayload::clock_sync_device_tick_reference)
        .def_readonly("clock_sync_host_time_reference_ns",
                      &PySignalBlockPayload::clock_sync_host_time_reference_ns)
        .def_readonly("clock_sync_rate_numerator", &PySignalBlockPayload::clock_sync_rate_numerator)
        .def_readonly("clock_sync_rate_denominator",
                      &PySignalBlockPayload::clock_sync_rate_denominator)
        .def_readonly("clock_sync_uncertainty_ns", &PySignalBlockPayload::clock_sync_uncertainty_ns)
        .def_readonly("block_idx_in_frame", &PySignalBlockPayload::block_idx_in_frame)
        .def_readonly("native_signal_id", &PySignalBlockPayload::native_signal_id)
        .def_readonly("n_samples", &PySignalBlockPayload::n_samples)
        .def_readonly("clock_sync_clock_domain", &PySignalBlockPayload::clock_sync_clock_domain)
        .def_readonly("clock_sync_generation", &PySignalBlockPayload::clock_sync_generation)
        .def_readonly("clock_sync_flags", &PySignalBlockPayload::clock_sync_flags)
        /// The block's sample bytes, exactly as recorded. A copy rather than a
        /// view: the cursor's buffer is reused by the next record, so a view
        /// would age out from under a caller that kept it.
        .def_readonly("samples", &PySignalBlockPayload::samples);

    py::class_<DiscontinuityPayloadFields>(recording, "SpoolDiscontinuityPayload", py::is_final())
        .def_readonly("data_message_ordinal", &DiscontinuityPayloadFields::data_message_ordinal)
        .def_readonly("native_session_id", &DiscontinuityPayloadFields::native_session_id)
        .def_readonly("previous_frame_sequence",
                      &DiscontinuityPayloadFields::previous_frame_sequence)
        .def_readonly("actual_frame_sequence", &DiscontinuityPayloadFields::actual_frame_sequence)
        .def_readonly("runtime_accepted_host_time_ns",
                      &DiscontinuityPayloadFields::runtime_accepted_host_time_ns)
        .def_readonly("signal_gap_count", &DiscontinuityPayloadFields::signal_gap_count)
        .def_readonly("reason", &DiscontinuityPayloadFields::reason);

    py::class_<SignalGapPayloadFields>(recording, "SpoolSignalGapPayload", py::is_final())
        .def_readonly("signal_gap_ordinal", &SignalGapPayloadFields::signal_gap_ordinal)
        .def_readonly("data_message_ordinal", &SignalGapPayloadFields::data_message_ordinal)
        .def_readonly("expected_sample_idx", &SignalGapPayloadFields::expected_sample_idx)
        .def_readonly("actual_sample_idx", &SignalGapPayloadFields::actual_sample_idx)
        .def_readonly("missing_samples", &SignalGapPayloadFields::missing_samples)
        .def_readonly("expected_device_tick", &SignalGapPayloadFields::expected_device_tick)
        .def_readonly("actual_device_tick", &SignalGapPayloadFields::actual_device_tick)
        .def_readonly("gap_idx_in_message", &SignalGapPayloadFields::gap_idx_in_message)
        .def_readonly("native_signal_id", &SignalGapPayloadFields::native_signal_id)
        .def_readonly("reason", &SignalGapPayloadFields::reason)
        .def_readonly("gap_flags", &SignalGapPayloadFields::gap_flags);

    py::class_<PyControlPayload>(recording, "SpoolControlPayload", py::is_final())
        .def_readonly("submission_ordinal", &PyControlPayload::submission_ordinal)
        .def_readonly("identity", &PyControlPayload::identity)
        .def_readonly("time_ns", &PyControlPayload::time_ns)
        .def_readonly("control_kind", &PyControlPayload::control_kind)
        .def_readonly("clock_domain", &PyControlPayload::clock_domain)
        .def_readonly("body", &PyControlPayload::body);

    py::class_<FaultPayloadFields>(recording, "SpoolFaultPayload", py::is_final())
        .def_readonly("native_session_id", &FaultPayloadFields::native_session_id)
        .def_readonly("runtime_generation", &FaultPayloadFields::runtime_generation)
        .def_readonly("frame_sequence", &FaultPayloadFields::frame_sequence)
        .def_readonly("sample_idx", &FaultPayloadFields::sample_idx)
        .def_readonly("device_tick", &FaultPayloadFields::device_tick)
        .def_readonly("detected_at_ns", &FaultPayloadFields::detected_at_ns)
        .def_readonly("data_ordinal_at_fault", &FaultPayloadFields::data_ordinal_at_fault)
        .def_readonly("control_ordinal_at_fault", &FaultPayloadFields::control_ordinal_at_fault)
        .def_readonly("component_id", &FaultPayloadFields::component_id)
        .def_readonly("detail", &FaultPayloadFields::detail)
        .def_readonly("schema_id", &FaultPayloadFields::schema_id)
        .def_readonly("clock_domain", &FaultPayloadFields::clock_domain)
        .def_readonly("signal_id", &FaultPayloadFields::signal_id)
        .def_readonly("fault_code", &FaultPayloadFields::fault_code)
        .def_readonly("stream_status", &FaultPayloadFields::stream_status)
        .def_readonly("fault_stage", &FaultPayloadFields::fault_stage)
        // Raw enumerator values, not enum objects: the rule this whole binding
        // follows is that naming belongs to the layer that spells the name, and
        // these two are written into NRF fields by the finalizer.
        .def_property_readonly("origin", [](const FaultPayloadFields& self)
                               { return static_cast<std::uint8_t>(self.origin); })
        .def_property_readonly("recorder_reason", [](const FaultPayloadFields& self)
                               { return static_cast<std::uint8_t>(self.recorder_reason); });

    py::class_<PySpoolRecord>(recording, "SpoolRecord", py::is_final())
        /// Which committed transaction made this record visible, and where that
        /// transaction starts. A caller that wants to cut a spool at a
        /// transaction boundary -- the shape a crash leaves -- cuts here.
        .def_readonly("transaction_id", &PySpoolRecord::transaction_id)
        .def_readonly("transaction_offset", &PySpoolRecord::transaction_offset)
        .def_readonly("offset", &PySpoolRecord::offset)
        /// Where the payload begins in the spool. Given as a position rather
        /// than as bytes so the walk stays one copy per record: a caller that
        /// wants the raw bytes already holds the buffer they are in.
        .def_readonly("payload_offset", &PySpoolRecord::payload_offset)
        .def_readonly("kind", &PySpoolRecord::kind)
        .def_readonly("logical_ordinal", &PySpoolRecord::logical_ordinal)
        .def_readonly("record_unix_nanos", &PySpoolRecord::record_unix_nanos)
        .def_readonly("payload_bytes", &PySpoolRecord::payload_bytes)
        .def_readonly("payload", &PySpoolRecord::payload);
}

void bind_accounting(py::module_& recording)
{
    py::class_<ScannedAccountingPosition>(recording, "SpoolAccountingPosition", py::is_final())
        .def_readonly("tag", &ScannedAccountingPosition::tag)
        .def_readonly("identity_kind", &ScannedAccountingPosition::identity_kind)
        .def_readonly("ordinal", &ScannedAccountingPosition::ordinal)
        .def_readonly("identity_value", &ScannedAccountingPosition::identity_value)
        .def_property_readonly("present", &ScannedAccountingPosition::present);

    py::class_<SpoolAccountingSnapshot>(recording, "SpoolAccounting", py::is_final())
        .def_readonly("runtime_accepted", &SpoolAccountingSnapshot::runtime_accepted)
        .def_readonly("recorder_accepted", &SpoolAccountingSnapshot::recorder_accepted)
        .def_readonly("spool_committed", &SpoolAccountingSnapshot::spool_committed)
        .def_readonly("rejected_before_runtime_acceptance",
                      &SpoolAccountingSnapshot::rejected_before_runtime_acceptance)
        .def_readonly("failed_between_runtime_and_recorder",
                      &SpoolAccountingSnapshot::failed_between_runtime_and_recorder)
        .def_readonly("lost_between_recorder_and_spool",
                      &SpoolAccountingSnapshot::lost_between_recorder_and_spool)
        .def_readonly("control_offered", &SpoolAccountingSnapshot::control_offered)
        .def_readonly("control_accepted", &SpoolAccountingSnapshot::control_accepted)
        .def_readonly("control_spool_committed", &SpoolAccountingSnapshot::control_spool_committed)
        .def_readonly("control_rejected", &SpoolAccountingSnapshot::control_rejected)
        .def_readonly("lost_between_control_acceptance_and_spool",
                      &SpoolAccountingSnapshot::lost_between_control_acceptance_and_spool)
        .def_readonly("rejected_after_close_data",
                      &SpoolAccountingSnapshot::rejected_after_close_data)
        .def_readonly("rejected_after_close_control",
                      &SpoolAccountingSnapshot::rejected_after_close_control)
        .def_readonly("control_offered_present", &SpoolAccountingSnapshot::control_offered_present)
        .def_readonly("producer_acceptance_known",
                      &SpoolAccountingSnapshot::producer_acceptance_known)
        .def_readonly("accounting_origin", &SpoolAccountingSnapshot::accounting_origin)
        .def_readonly("data_first_loss", &SpoolAccountingSnapshot::data_first_loss)
        .def_readonly("data_first_rejection", &SpoolAccountingSnapshot::data_first_rejection)
        .def_readonly("control_first_loss", &SpoolAccountingSnapshot::control_first_loss)
        .def_readonly("control_first_rejection", &SpoolAccountingSnapshot::control_first_rejection);
}

void bind_scan(py::module_& recording)
{
    py::class_<PySpoolRecords>(recording, "SpoolRecords", py::is_final())
        .def("__iter__", [](py::object self) { return self; })
        .def("__next__", &PySpoolRecords::next);

    py::class_<PySpoolScan>(recording, "NativeSpoolScan", py::is_final())
        .def_property_readonly("status",
                               [](const PySpoolScan& self) -> std::string
                               {
                                   const auto text = scan_status_text(self.report().status());
                                   return std::string{text.data(), text.size()};
                               })
        .def_property_readonly("readable",
                               [](const PySpoolScan& self) { return self.report().readable(); })
        .def_property_readonly("finalizable",
                               [](const PySpoolScan& self) { return self.report().finalizable(); })
        .def_property_readonly("read_failed",
                               [](const PySpoolScan& self) { return self.report().read_failed(); })
        .def_property_readonly("findings_truncated", [](const PySpoolScan& self)
                               { return self.report().findings_truncated(); })
        .def_property_readonly("codes",
                               [](const PySpoolScan& self)
                               {
                                   // In the order they were made, and not deduplicated: two
                                   // accounting identities that both fail are two findings, and
                                   // collapsing them would report one broken identity and four as
                                   // the same spool.
                                   py::tuple codes(self.report().findings().size());
                                   std::size_t idx = 0;
                                   for (const auto& finding : self.report().findings())
                                   {
                                       const auto text = spool_code_text(finding.code);
                                       codes[idx++] =
                                           py::str(std::string{text.data(), text.size()});
                                   }
                                   return codes;
                               })
        .def_property_readonly("finding_offsets",
                               [](const PySpoolScan& self)
                               {
                                   py::tuple offsets(self.report().findings().size());
                                   std::size_t idx = 0;
                                   for (const auto& finding : self.report().findings())
                                   {
                                       offsets[idx++] = py::int_(finding.offset);
                                   }
                                   return offsets;
                               })
        .def_property_readonly("committed_prefix_end", [](const PySpoolScan& self)
                               { return self.report().committed_prefix_end(); })
        .def_property_readonly("committed_transactions", [](const PySpoolScan& self)
                               { return self.report().committed_transactions(); })
        .def_property_readonly("last_transaction_id", [](const PySpoolScan& self)
                               { return self.report().last_transaction_id(); })
        .def_property_readonly("data_items",
                               [](const PySpoolScan& self) { return self.report().data_items(); })
        .def_property_readonly("control_items", [](const PySpoolScan& self)
                               { return self.report().control_items(); })
        .def_property_readonly("durable_extent_bytes", [](const PySpoolScan& self)
                               { return self.report().durable_extent_bytes(); })
        .def_property_readonly("session_end_present", [](const PySpoolScan& self)
                               { return self.report().session_end_present(); })
        // --- the superblock, present only when it validated ------------------
        .def_property_readonly("superblock_valid", [](const PySpoolScan& self)
                               { return self.report().has_durability_policy(); })
        .def_property_readonly("version_minor", [](const PySpoolScan& self)
                               { return self.report().version_minor(); })
        .def_property_readonly(
            "durability_policy", [](const PySpoolScan& self)
            { return static_cast<std::uint8_t>(self.report().durability_policy()); })
        .def_property_readonly("session_id",
                               [](const PySpoolScan& self) -> std::string
                               {
                                   const auto id = self.report().session_id();
                                   return std::string{id.data(), id.size()};
                               })
        .def_property_readonly("session_uuid",
                               [](const PySpoolScan& self)
                               {
                                   const auto& uuid = self.report().session_uuid();
                                   return py::bytes(reinterpret_cast<const char*>(uuid.data()),
                                                    uuid.size());
                               })
        .def_property_readonly("plan_fingerprint",
                               [](const PySpoolScan& self)
                               {
                                   const auto& digest = self.report().plan_fingerprint();
                                   return py::bytes(reinterpret_cast<const char*>(digest.data()),
                                                    digest.size());
                               })
        .def_property_readonly("created_unix_nanos", [](const PySpoolScan& self)
                               { return self.report().created_unix_nanos(); })
        .def_property_readonly("first_transaction_offset", [](const PySpoolScan& self)
                               { return self.report().first_transaction_offset(); })
        .def_property_readonly("plan_document", &PySpoolScan::plan_document)
        // --- the accounting snapshot, present only when one was committed ----
        .def_property_readonly("has_accounting", [](const PySpoolScan& self)
                               { return self.report().has_accounting(); })
        .def_property_readonly(
            "accounting", [](const PySpoolScan& self) -> const SpoolAccountingSnapshot&
            { return self.report().accounting(); }, py::return_value_policy::reference_internal)
        // --- the session-end record, present only when one was decoded -------
        .def_property_readonly("session_end_decoded", [](const PySpoolScan& self)
                               { return self.report().has_terminal_reason(); })
        .def_property_readonly("has_capture_outcome", [](const PySpoolScan& self)
                               { return self.report().has_capture_outcome(); })
        .def_property_readonly(
            "capture_outcome", [](const PySpoolScan& self)
            { return static_cast<std::uint8_t>(self.report().capture_outcome()); })
        .def_property_readonly("has_requested_terminal_intent", [](const PySpoolScan& self)
                               { return self.report().has_requested_terminal_intent(); })
        .def_property_readonly(
            "requested_terminal_intent", [](const PySpoolScan& self)
            { return static_cast<std::uint8_t>(self.report().requested_terminal_intent()); })
        .def_property_readonly("has_primary_fault_committed", [](const PySpoolScan& self)
                               { return self.report().has_primary_fault_committed(); })
        .def_property_readonly("primary_fault_committed", [](const PySpoolScan& self)
                               { return self.report().primary_fault_committed(); })
        .def_property_readonly("terminal_reason",
                               [](const PySpoolScan& self) -> std::string
                               {
                                   const auto reason = self.report().terminal_reason();
                                   return std::string{reason.data(), reason.size()};
                               })
        .def_property_readonly("session_end_unix_nanos", [](const PySpoolScan& self)
                               { return self.report().session_end_unix_nanos(); })
        .def("close", &PySpoolScan::close)
        .def("records",
             [](py::object self)
             {
                 auto& scan = self.cast<PySpoolScan&>();
                 return std::make_unique<PySpoolRecords>(self, scan.file(), scan.report());
             });

    recording.def("scan_spool_bytes", &PySpoolScan::from_bytes, py::arg("data"));
    recording.def("scan_spool_path", &PySpoolScan::from_path, py::arg("path"));
    // Exposed with its seed so a caller that writes in chunks -- the replay
    // image writer, which chunks precisely so the image is never all in memory
    // -- folds them through the same kernel the container is checked with,
    // rather than keeping a second one in step by hand.
    recording.def(
        "spool_crc32c",
        [](const py::buffer& source, std::uint32_t seed)
        {
            const auto info = source.request(false);
            if (info.ndim != 1 || info.itemsize != 1)
            {
                throw std::invalid_argument("crc32c takes a flat byte buffer");
            }
            return crc32c(std::span<const std::byte>{static_cast<const std::byte*>(info.ptr),
                                                     static_cast<std::size_t>(info.size)},
                          seed);
        },
        py::arg("data"), py::arg("seed") = 0U);
}

} // namespace

void bind_spool_module(py::module_& recording)
{
    bind_payloads(recording);
    bind_accounting(recording);
    bind_scan(recording);
}
