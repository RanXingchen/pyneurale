/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// \file
/// The private Python surface over the native replay-image reader.
///
/// It exists so the container has exactly one reader. `_replay_image.py` writes
/// the format and, before this binding, also read it: two decoders of one
/// private layout, kept in step only by a parity test that skips when the C++
/// test binaries are absent. A change to the layout that one side followed and
/// the other did not left both self-consistent and disagreeing.
///
/// What crosses this boundary is *layout decoding only*. Absent markers stay
/// absent, string-table indices stay indices, and flag words stay words: the
/// semantic reading -- resolving a name, turning a marker into `None`, grouping
/// the clock-sync fields -- belongs to the Python value objects, which are the
/// public surface and are not mirrored here.
///
/// Nothing here is on a replay read path. The native replay source consumes
/// `ReplayImageFile` directly; this binding serves the cache-validation and
/// reporting paths, which are offline by construction.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "image.h"

namespace py = pybind11;

using namespace neurale::recording;

namespace
{

/// An opened image, owned by Python. Opening validates the container whole; a
/// failure raises rather than yielding a half-read object, so every accessor
/// below may assume the bounds established at open.
class PyReplayImage
{
  public:
    explicit PyReplayImage(const std::string& path)
    {
        const auto status = file_.open(path.c_str());
        if (status != ReplayImageStatus::ok)
        {
            // The Python layer turns this into `ReplayImageError`. The status
            // text is the native reader's own, so the two paths cannot describe
            // the same failure differently.
            throw std::runtime_error(replay_image_status_text(status));
        }
    }

    [[nodiscard]] const ReplayImageFile& file() const noexcept
    {
        return file_;
    }

    void close() noexcept
    {
        file_.close();
    }

    [[nodiscard]] py::bytes fingerprint() const
    {
        const auto span = file_.fingerprint();
        return {reinterpret_cast<const char*>(span.data()), span.size()};
    }

    /// The payload bytes of one record. An out-of-range span is empty, which
    /// the caller must treat as an error rather than as an empty payload.
    [[nodiscard]] py::bytes payload(std::uint64_t offset, std::uint64_t count) const
    {
        const auto span = file_.payload(offset, count);
        if (span.size() != count)
        {
            throw std::runtime_error("a replay image record names payload it does not carry");
        }
        return {reinterpret_cast<const char*>(span.data()), span.size()};
    }

    /// One interned string, or `None` for the absent marker. Returning `None`
    /// rather than `""` is what lets an optional field stay optional.
    [[nodiscard]] py::object string(std::uint32_t idx) const
    {
        if (idx == kReplayAbsentU32)
        {
            return py::none();
        }
        const auto view = file_.string(idx);
        return py::str(view.data(), view.size());
    }

  private:
    ReplayImageFile file_;
};

void bind_records(py::module_& image)
{
    py::enum_<ReplayItemKind>(image, "ReplayItemKind")
        .value("frame", ReplayItemKind::frame)
        .value("discontinuity", ReplayItemKind::discontinuity);

    py::enum_<ReplayIdNamespace>(image, "ReplayIdNamespace")
        .value("signals", ReplayIdNamespace::signals)
        .value("clocks", ReplayIdNamespace::clocks)
        .value("feature_sets", ReplayIdNamespace::feature_sets)
        .value("units", ReplayIdNamespace::units);

    py::class_<ReplayImageItem>(image, "NativeReplayItem", py::is_final())
        .def_readonly("kind", &ReplayImageItem::kind)
        .def_readonly("flags", &ReplayImageItem::flags)
        .def_readonly("n_children", &ReplayImageItem::n_children)
        .def_readonly("child_first", &ReplayImageItem::child_first)
        .def_readonly("data_message_ordinal", &ReplayImageItem::data_message_ordinal)
        .def_readonly("replay_sequence", &ReplayImageItem::replay_sequence)
        .def_readonly("original_sequence", &ReplayImageItem::original_sequence)
        .def_readonly("previous_replay_sequence", &ReplayImageItem::previous_replay_sequence)
        .def_readonly("original_previous_sequence", &ReplayImageItem::original_previous_sequence)
        .def_readonly("timeline_ns", &ReplayImageItem::timeline_ns)
        .def_readonly("original_host_received_ns", &ReplayImageItem::original_host_received_ns)
        .def_readonly("payload_offset", &ReplayImageItem::payload_offset)
        .def_readonly("payload_byte_count", &ReplayImageItem::payload_byte_count)
        .def_readonly("original_payload_byte_count", &ReplayImageItem::original_payload_byte_count)
        .def_readonly("source_tick", &ReplayImageItem::source_tick)
        .def_readonly("valid_until_ns", &ReplayImageItem::valid_until_ns)
        .def_readonly("native_schema_id", &ReplayImageItem::native_schema_id)
        .def_readonly("source_clock_domain", &ReplayImageItem::source_clock_domain)
        .def_readonly("frame_flags", &ReplayImageItem::frame_flags)
        .def_readonly("reason", &ReplayImageItem::reason);

    py::class_<ReplayImageBlock>(image, "NativeReplayBlock", py::is_final())
        .def_readonly("native_signal_id", &ReplayImageBlock::native_signal_id)
        .def_readonly("block_idx_in_frame", &ReplayImageBlock::block_idx_in_frame)
        .def_readonly("stream", &ReplayImageBlock::stream)
        .def_readonly("n_samples", &ReplayImageBlock::n_samples)
        .def_readonly("sample_idx_start", &ReplayImageBlock::sample_idx_start)
        .def_readonly("last_sample_idx", &ReplayImageBlock::last_sample_idx)
        .def_readonly("device_tick_start", &ReplayImageBlock::device_tick_start)
        .def_readonly("observation_time_start_ns", &ReplayImageBlock::observation_time_start_ns)
        .def_readonly("payload_offset", &ReplayImageBlock::payload_offset)
        .def_readonly("payload_byte_count", &ReplayImageBlock::payload_byte_count)
        .def_readonly("original_payload_offset", &ReplayImageBlock::original_payload_offset)
        .def_readonly("clock_sync_device_tick_reference",
                      &ReplayImageBlock::clock_sync_device_tick_reference)
        .def_readonly("clock_sync_host_time_reference_ns",
                      &ReplayImageBlock::clock_sync_host_time_reference_ns)
        .def_readonly("clock_sync_rate_numerator", &ReplayImageBlock::clock_sync_rate_numerator)
        .def_readonly("clock_sync_rate_denominator", &ReplayImageBlock::clock_sync_rate_denominator)
        .def_readonly("clock_sync_uncertainty_ns", &ReplayImageBlock::clock_sync_uncertainty_ns)
        .def_readonly("source_block_ordinal", &ReplayImageBlock::source_block_ordinal)
        .def_readonly("recorded_frame_sequence", &ReplayImageBlock::recorded_frame_sequence)
        .def_readonly("clock_sync_clock_domain", &ReplayImageBlock::clock_sync_clock_domain)
        .def_readonly("clock_sync_generation", &ReplayImageBlock::clock_sync_generation)
        .def_readonly("clock_sync_flags", &ReplayImageBlock::clock_sync_flags)
        .def_readonly("flags", &ReplayImageBlock::flags);

    py::class_<ReplayImageGap>(image, "NativeReplayGap", py::is_final())
        .def_readonly("native_signal_id", &ReplayImageGap::native_signal_id)
        .def_readonly("gap_idx_in_message", &ReplayImageGap::gap_idx_in_message)
        .def_readonly("reason", &ReplayImageGap::reason)
        .def_readonly("gap_flags", &ReplayImageGap::gap_flags)
        .def_readonly("expected_sample_idx", &ReplayImageGap::expected_sample_idx)
        .def_readonly("actual_sample_idx", &ReplayImageGap::actual_sample_idx)
        .def_readonly("missing_samples", &ReplayImageGap::missing_samples)
        .def_readonly("expected_device_tick", &ReplayImageGap::expected_device_tick)
        .def_readonly("actual_device_tick", &ReplayImageGap::actual_device_tick)
        .def_readonly("signal_gap_ordinal", &ReplayImageGap::signal_gap_ordinal);

    py::class_<ReplayImageFidelity>(image, "NativeReplayFidelity", py::is_final())
        .def_readonly("stream", &ReplayImageFidelity::stream)
        .def_readonly("flags", &ReplayImageFidelity::flags)
        .def_readonly("block_idx_column_first", &ReplayImageFidelity::block_idx_column_first)
        .def_readonly("block_idx_column_count", &ReplayImageFidelity::block_idx_column_count)
        .def_readonly("native_signal_id", &ReplayImageFidelity::native_signal_id)
        .def_readonly("frames_emitted", &ReplayImageFidelity::frames_emitted)
        .def_readonly("blocks_emitted", &ReplayImageFidelity::blocks_emitted)
        .def_readonly("discontinuities_emitted", &ReplayImageFidelity::discontinuities_emitted)
        .def_readonly("committed_blocks", &ReplayImageFidelity::committed_blocks);

    py::class_<ReplayImageOmission>(image, "NativeReplayOmission", py::is_final())
        .def_readonly("kind", &ReplayImageOmission::kind)
        .def_readonly("reason", &ReplayImageOmission::reason)
        .def_readonly("stream", &ReplayImageOmission::stream)
        .def_readonly("data_message_ordinal", &ReplayImageOmission::data_message_ordinal)
        .def_readonly("original_frame_sequence", &ReplayImageOmission::original_frame_sequence)
        .def_readonly("source_ordinal", &ReplayImageOmission::source_ordinal);

    py::class_<ReplayImageStreamRange>(image, "NativeReplayStreamRange", py::is_final())
        .def_readonly("stream", &ReplayImageStreamRange::stream)
        .def_readonly("unit", &ReplayImageStreamRange::unit)
        .def_readonly("range_start", &ReplayImageStreamRange::range_start)
        .def_readonly("range_stop", &ReplayImageStreamRange::range_stop)
        .def_readonly("committed", &ReplayImageStreamRange::committed);

    py::class_<ReplayImageIdEntry>(image, "NativeReplayIdEntry", py::is_final())
        .def_readonly("name", &ReplayImageIdEntry::name)
        .def_readonly("native_id", &ReplayImageIdEntry::native_id);

    py::class_<ReplayImageSummary>(image, "NativeReplaySummary", py::is_final())
        .def_readonly("mode", &ReplayImageSummary::mode)
        .def_readonly("plan_coverage", &ReplayImageSummary::plan_coverage)
        .def_readonly("completeness", &ReplayImageSummary::completeness)
        .def_readonly("source_session_id", &ReplayImageSummary::source_session_id)
        .def_readonly("plan_fingerprint", &ReplayImageSummary::plan_fingerprint)
        .def_readonly("source_fingerprint", &ReplayImageSummary::source_fingerprint)
        .def_readonly("flags", &ReplayImageSummary::flags)
        .def_readonly("frame_construction", &ReplayImageSummary::frame_construction)
        .def_readonly("ordering_key", &ReplayImageSummary::ordering_key)
        .def_readonly("selected_list", &ReplayImageSummary::selected_list)
        .def_readonly("n_selected", &ReplayImageSummary::n_selected)
        .def_readonly("planned_list", &ReplayImageSummary::planned_list)
        .def_readonly("n_planned", &ReplayImageSummary::n_planned)
        .def_readonly("recorded_list", &ReplayImageSummary::recorded_list)
        .def_readonly("n_recorded", &ReplayImageSummary::n_recorded)
        .def_readonly("range_start", &ReplayImageSummary::range_start)
        .def_readonly("range_stop", &ReplayImageSummary::range_stop)
        .def_readonly("n_items", &ReplayImageSummary::n_items)
        .def_readonly("n_frames", &ReplayImageSummary::n_frames)
        .def_readonly("n_discontinuities", &ReplayImageSummary::n_discontinuities)
        .def_readonly("n_blocks", &ReplayImageSummary::n_blocks)
        .def_readonly("n_gaps", &ReplayImageSummary::n_gaps)
        .def_readonly("n_omissions", &ReplayImageSummary::n_omissions)
        .def_readonly("payload_byte_count", &ReplayImageSummary::payload_byte_count)
        .def_readonly("native_session_id", &ReplayImageSummary::native_session_id)
        .def_readonly("source_message_count", &ReplayImageSummary::source_message_count)
        .def_readonly("first_timeline_ns", &ReplayImageSummary::first_timeline_ns)
        .def_readonly("last_timeline_ns", &ReplayImageSummary::last_timeline_ns);

    py::class_<ReplayImageSignal>(image, "NativeReplaySignal", py::is_final())
        .def_readonly("id", &ReplayImageSignal::id)
        .def_readonly("clock_domain", &ReplayImageSignal::clock_domain)
        .def_readonly("n_channels", &ReplayImageSignal::n_channels)
        .def_readonly("nominal_block_samples", &ReplayImageSignal::nominal_block_samples)
        .def_readonly("max_block_samples", &ReplayImageSignal::max_block_samples)
        .def_readonly("dtype", &ReplayImageSignal::dtype)
        .def_readonly("layout", &ReplayImageSignal::layout)
        .def_readonly("device_tick_tracking", &ReplayImageSignal::device_tick_tracking)
        .def_readonly("kind", &ReplayImageSignal::kind)
        .def_readonly("physical_unit", &ReplayImageSignal::physical_unit)
        .def_readonly("channel_set_id", &ReplayImageSignal::channel_set_id)
        .def_readonly("calibration_id", &ReplayImageSignal::calibration_id)
        .def_readonly("reference_id", &ReplayImageSignal::reference_id)
        .def_readonly("feature_set_id", &ReplayImageSignal::feature_set_id)
        .def_readonly("observation_timing", &ReplayImageSignal::observation_timing)
        .def_readonly("rate_num", &ReplayImageSignal::rate_num)
        .def_readonly("rate_den", &ReplayImageSignal::rate_den)
        .def_readonly("fixed_block_bytes", &ReplayImageSignal::fixed_block_bytes)
        .def_readonly("max_block_bytes", &ReplayImageSignal::max_block_bytes);

    py::class_<ReplayImageFeatureSet>(image, "NativeReplayFeatureSet", py::is_final())
        .def_readonly("id", &ReplayImageFeatureSet::id)
        .def_readonly("source_stream_id", &ReplayImageFeatureSet::source_stream_id)
        .def_readonly("source_stream", &ReplayImageFeatureSet::source_stream)
        .def_readonly("algorithm_name", &ReplayImageFeatureSet::algorithm_name)
        .def_readonly("algorithm_version", &ReplayImageFeatureSet::algorithm_version)
        .def_readonly("timestamp_reference", &ReplayImageFeatureSet::timestamp_reference)
        .def_readonly("feature_name_first", &ReplayImageFeatureSet::feature_name_first)
        .def_readonly("feature_name_count", &ReplayImageFeatureSet::feature_name_count)
        .def_readonly("unit_id_first", &ReplayImageFeatureSet::unit_id_first)
        .def_readonly("unit_id_count", &ReplayImageFeatureSet::unit_id_count)
        .def_readonly("window_length_ns", &ReplayImageFeatureSet::window_length_ns)
        .def_readonly("shift_ns", &ReplayImageFeatureSet::shift_ns);

    py::class_<ReplayImageUnit>(image, "NativeReplayUnit", py::is_final())
        .def_readonly("id", &ReplayImageUnit::id)
        .def_readonly("symbol", &ReplayImageUnit::symbol)
        .def_readonly("description", &ReplayImageUnit::description);
}

} // namespace

void bind_replay_image_module(py::module_& recording)
{
    auto image = recording.def_submodule(
        "replay_image", "Private native reader for the replay-image container, version 1.");

    bind_records(image);

    py::class_<PyReplayImage>(image, "NativeReplayImage", py::is_final())
        .def(
            py::init([](const std::string& path) { return std::make_unique<PyReplayImage>(path); }),
            py::arg("path"))
        .def("close", &PyReplayImage::close)
        .def_property_readonly("fingerprint", &PyReplayImage::fingerprint)
        .def_property_readonly("summary",
                               [](const PyReplayImage& self) { return self.file().summary(); })
        .def_property_readonly("schema_id",
                               [](const PyReplayImage& self) { return self.file().schema_id(); })
        .def_property_readonly("n_items",
                               [](const PyReplayImage& self) { return self.file().item_count(); })
        .def_property_readonly("n_blocks",
                               [](const PyReplayImage& self) { return self.file().block_count(); })
        .def_property_readonly("n_gaps",
                               [](const PyReplayImage& self) { return self.file().gap_count(); })
        .def_property_readonly("n_fidelities", [](const PyReplayImage& self)
                               { return self.file().fidelity_count(); })
        .def_property_readonly("n_omissions", [](const PyReplayImage& self)
                               { return self.file().omission_count(); })
        .def_property_readonly("stream_range_count", [](const PyReplayImage& self)
                               { return self.file().stream_range_count(); })
        .def_property_readonly("n_signals",
                               [](const PyReplayImage& self) { return self.file().signal_count(); })
        .def_property_readonly("feature_set_count", [](const PyReplayImage& self)
                               { return self.file().feature_set_count(); })
        .def_property_readonly("n_units",
                               [](const PyReplayImage& self) { return self.file().unit_count(); })
        .def_property_readonly("string_count",
                               [](const PyReplayImage& self) { return self.file().string_count(); })
        .def_property_readonly("list_count",
                               [](const PyReplayImage& self) { return self.file().list_count(); })
        .def(
            "item", [](const PyReplayImage& self, std::size_t i) { return self.file().item(i); },
            py::arg("idx"))
        .def(
            "block", [](const PyReplayImage& self, std::size_t i) { return self.file().block(i); },
            py::arg("idx"))
        .def(
            "gap", [](const PyReplayImage& self, std::size_t i) { return self.file().gap(i); },
            py::arg("idx"))
        .def(
            "fidelity", [](const PyReplayImage& self, std::size_t i)
            { return self.file().fidelity(i); }, py::arg("idx"))
        .def(
            "omission", [](const PyReplayImage& self, std::size_t i)
            { return self.file().omission(i); }, py::arg("idx"))
        .def(
            "stream_range", [](const PyReplayImage& self, std::size_t i)
            { return self.file().stream_range(i); }, py::arg("idx"))
        .def(
            "signal", [](const PyReplayImage& self, std::size_t i)
            { return self.file().signal(i); }, py::arg("idx"))
        .def(
            "feature_set", [](const PyReplayImage& self, std::size_t i)
            { return self.file().feature_set(i); }, py::arg("idx"))
        .def(
            "unit", [](const PyReplayImage& self, std::size_t i) { return self.file().unit(i); },
            py::arg("idx"))
        .def("string", &PyReplayImage::string, py::arg("idx"))
        .def(
            "list_at", [](const PyReplayImage& self, std::uint64_t i)
            { return self.file().list_at(i); }, py::arg("idx"))
        .def(
            "id_registry_count", [](const PyReplayImage& self, ReplayIdNamespace space)
            { return self.file().id_registry_count(space); }, py::arg("namespace_"))
        .def(
            "id_registry_entry",
            [](const PyReplayImage& self, ReplayIdNamespace space, std::size_t i)
            { return self.file().id_registry_entry(space, i); }, py::arg("namespace_"),
            py::arg("idx"))
        .def("payload", &PyReplayImage::payload, py::arg("offset"), py::arg("count"));
}
