/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// The native reader of the private replay-image container.
///
/// Everything here is about the container and nothing about replay semantics:
/// what a well-formed image decodes to, and which malformed ones are refused
/// whole. The contract's reason for a refusal being total is in section 8 --
/// the session is the source of truth and the image is a cache, so a bad image
/// is rebuilt, never salvaged.

#include "check_returns.h"
#include "crc32c.h"
#include "replay/image.h"
#include "replay_test_support.h"

#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

namespace
{

using neurale::recording::ReplayImageFile;
using neurale::recording::ReplayImageStatus;
using neurale::recording::ReplayItemKind;
using neurale::recording::test::BuiltBlock;
using neurale::recording::test::BuiltDiscontinuity;
using neurale::recording::test::BuiltFrame;
using neurale::recording::test::BuiltGap;
using neurale::recording::test::ReplayImageBuilder;

using neurale::recording::test::Scratch;

std::vector<std::byte> bytes_of(std::initializer_list<unsigned char> values)
{
    std::vector<std::byte> result;
    for (const auto value : values)
    {
        result.push_back(static_cast<std::byte>(value));
    }
    return result;
}

/// One frame, one block, four payload bytes; the smallest image worth reading.
ReplayImageBuilder minimal_builder()
{
    ReplayImageBuilder builder;
    BuiltFrame frame;
    frame.data_message_ordinal = 7;
    frame.replay_sequence = 0;
    frame.original_sequence = 41;
    frame.timeline_ns = 1'000;
    frame.original_host_received_ns = 1'000;
    frame.payload = bytes_of({1, 2, 3, 4});
    BuiltBlock block;
    block.payload_byte_count = 4;
    block.n_samples = 1;
    block.source_block_ordinal = 3;
    frame.blocks.push_back(block);
    builder.add_frame(std::move(frame));
    builder.fidelity.push_back({.stream = "stream", .native_signal_id = 1});
    return builder;
}

int run()
{
    const Scratch scratch;

    {
        auto builder = minimal_builder();
        const auto path = scratch.path("minimal.nrimg");
        CHECK(builder.write(path));

        ReplayImageFile image;
        CHECK(image.open(path.c_str()) == ReplayImageStatus::ok);
        CHECK(image.is_open());
        CHECK(image.item_count() == 1);
        CHECK(image.block_count() == 1);
        CHECK(image.gap_count() == 0);
        CHECK(image.fidelity_count() == 1);
        CHECK(image.string(image.summary().mode) == "exact_frames");
        CHECK(image.summary().ledger_based());
        CHECK(!image.summary().abnormal_end_required());
        CHECK(image.summary().payload_byte_count == 4);

        const auto item = image.item(0);
        CHECK(item.kind == ReplayItemKind::frame);
        CHECK(item.data_message_ordinal == 7);
        CHECK(item.original_sequence == 41);
        CHECK(item.timeline_ns == 1'000);
        CHECK(item.n_children == 1);
        CHECK(item.payload_byte_count == 4);

        const auto block = image.block(0);
        CHECK(block.native_signal_id == 1);
        CHECK(block.source_block_ordinal == 3);
        CHECK(image.string(block.stream) == "stream");

        const auto payload = image.payload(item.payload_offset, item.payload_byte_count);
        CHECK(payload.size() == 4);
        CHECK(payload[0] == std::byte{1});
        CHECK(payload[3] == std::byte{4});

        // Reading past the end is answered with a value-initialized record and
        // never with a read outside the mapping.
        CHECK(image.item(1).n_children == 0);
        CHECK(image.block(9).payload_byte_count == 0);
        CHECK(image.payload(0, 99).empty());
        CHECK(image.string(9999).empty());
    }

    {
        // Discontinuities, their gaps, and the fidelity entry that maps a
        // signal back to the stream a synthesized replay stands for.
        ReplayImageBuilder builder;
        builder.mode = "stream_frames";
        builder.flags = 0;
        BuiltFrame frame;
        frame.replay_sequence = 0;
        frame.timeline_ns = 10;
        frame.payload = bytes_of({9});
        BuiltBlock block;
        block.payload_byte_count = 1;
        block.source_block_ordinal = 0;
        frame.blocks.push_back(block);
        builder.add_frame(std::move(frame));

        BuiltDiscontinuity gap_record;
        gap_record.replay_sequence = 1;
        gap_record.previous_replay_sequence = 0;
        gap_record.timeline_ns = 20;
        gap_record.reason = "sample_gap";
        BuiltGap gap;
        gap.missing_samples = 12;
        gap.signal_gap_ordinal = 4;
        gap.actual_sample_idx = 40;
        gap_record.gaps.push_back(gap);
        builder.add_discontinuity(std::move(gap_record));
        builder.fidelity.push_back({.stream = "cursor", .native_signal_id = 1});

        const auto path = scratch.path("synthesized.nrimg");
        CHECK(builder.write(path));
        ReplayImageFile image;
        CHECK(image.open(path.c_str()) == ReplayImageStatus::ok);
        CHECK(image.item_count() == 2);
        const auto item = image.item(1);
        CHECK(item.kind == ReplayItemKind::discontinuity);
        CHECK(item.previous_replay_sequence == 0);
        CHECK(image.string(item.reason) == "sample_gap");
        CHECK(item.n_children == 1);
        const auto decoded = image.gap(static_cast<std::size_t>(item.child_first));
        CHECK(decoded.missing_samples == 12);
        CHECK(decoded.signal_gap_ordinal == 4);
        CHECK(decoded.actual_sample_idx == 40);
        CHECK(image.string(image.fidelity(0).stream) == "cursor");
        CHECK(!image.summary().ledger_based());
    }

    {
        // Every refusal, one builder knob at a time.
        struct Case
        {
            const char* name;
            void (*apply)(ReplayImageBuilder&);
            ReplayImageStatus expected;
        };
        const Case cases[] = {
            {"magic", [](ReplayImageBuilder& builder) { builder.break_magic = true; },
             ReplayImageStatus::not_an_image},
            {"version", [](ReplayImageBuilder& builder) { builder.version_override = 2; },
             ReplayImageStatus::unsupported_version},
            {"content", [](ReplayImageBuilder& builder) { builder.break_content_checksum = true; },
             ReplayImageStatus::checksum_mismatch},
            {"length", [](ReplayImageBuilder& builder) { builder.lie_about_total_bytes = true; },
             ReplayImageStatus::truncated},
            {"duplicate", [](ReplayImageBuilder& builder)
             { builder.duplicate_items_section = true; }, ReplayImageStatus::malformed},
            {"ragged", [](ReplayImageBuilder& builder) { builder.ragged_items_section = true; },
             ReplayImageStatus::malformed},
            {"summary", [](ReplayImageBuilder& builder) { builder.omit_summary_section = true; },
             ReplayImageStatus::malformed},
            {"counts", [](ReplayImageBuilder& builder) { builder.lie_about_item_count = true; },
             ReplayImageStatus::malformed},
        };
        for (const auto& entry : cases)
        {
            auto builder = minimal_builder();
            entry.apply(builder);
            const auto path = scratch.path(std::string("bad-") + entry.name + ".nrimg");
            CHECK(builder.write(path));
            ReplayImageFile image;
            const auto status = image.open(path.c_str());
            if (status != entry.expected)
            {
                std::cerr << "case " << entry.name << " gave "
                          << neurale::recording::replay_image_status_text(status) << '\n';
                return __LINE__;
            }
            // A refusal leaves nothing half-open behind it.
            CHECK(!image.is_open());
            CHECK(image.item_count() == 0);
        }
    }

    {
        // A byte flipped anywhere in the body is caught by the content
        // checksum, which is the whole reason the image carries one.
        auto builder = minimal_builder();
        const auto path = scratch.path("flipped.nrimg");
        CHECK(builder.write(path));
        {
            std::FILE* handle = std::fopen(path.c_str(), "r+b");
            CHECK(handle != nullptr);
            CHECK(std::fseek(handle, 200, SEEK_SET) == 0);
            int value = std::fgetc(handle);
            CHECK(value >= 0);
            CHECK(std::fseek(handle, 200, SEEK_SET) == 0);
            CHECK(std::fputc(value ^ 0x5A, handle) >= 0);
            std::fclose(handle);
        }
        ReplayImageFile image;
        CHECK(image.open(path.c_str()) == ReplayImageStatus::checksum_mismatch);
    }

    {
        ReplayImageFile image;
        CHECK(image.open((scratch.path("absent.nrimg")).c_str()) == ReplayImageStatus::unreadable);
        CHECK(image.open(nullptr) == ReplayImageStatus::unreadable);
    }

    return 0;
}

} // namespace

/// Print what this reader sees in *path*, one JSON object per line.
///
/// This is the differential mode: `tests/unit/recording/test_native_parity.py`
/// builds an image with the production Python writer, runs this, and compares
/// the two readings field by field. Without it, the test-only writer in
/// `replay_test_support.h` and `_replay_image.py` could drift apart
/// while each stayed perfectly self-consistent.
int verify(const char* path)
{
    ReplayImageFile image;
    const auto status = image.open(path);
    if (status != ReplayImageStatus::ok)
    {
        std::cout << "{\"status\":\"" << neurale::recording::replay_image_status_text(status)
                  << "\"}\n";
        return 2;
    }
    const auto& summary = image.summary();
    std::cout << "{\"status\":\"ok\",\"mode\":\"" << image.string(summary.mode)
              << "\",\"item_count\":" << summary.n_items << ",\"frame_count\":" << summary.n_frames
              << ",\"discontinuity_count\":" << summary.n_discontinuities
              << ",\"block_count\":" << summary.n_blocks << ",\"gap_count\":" << summary.n_gaps
              << ",\"payload_byte_count\":" << summary.payload_byte_count
              << ",\"schema_id\":" << image.schema_id()
              << ",\"signal_count\":" << image.signal_count() << "}\n";
    for (std::size_t i = 0; i < image.item_count(); ++i)
    {
        const auto item = image.item(i);
        std::uint32_t payload_crc = 0;
        if (item.kind == ReplayItemKind::frame)
        {
            payload_crc = neurale::recording::crc32c(
                image.payload(item.payload_offset, item.payload_byte_count));
        }
        std::cout << "{\"index\":" << i << ",\"kind\":\""
                  << (item.kind == ReplayItemKind::frame ? "frame" : "discontinuity")
                  << "\",\"replay_sequence\":" << item.replay_sequence
                  << ",\"timeline_ns\":" << item.timeline_ns
                  << ",\"child_count\":" << item.n_children
                  << ",\"payload_byte_count\":" << item.payload_byte_count
                  << ",\"payload_crc32c\":" << payload_crc << "}\n";
    }
    return 0;
}

int main(int argc, char** argv)
{
    if (argc == 3 && std::string(argv[1]) == "--verify")
    {
        return verify(argv[2]);
    }
    if (argc != 1)
    {
        std::cerr << "usage: neurale_recording_replay_image_test [--verify <image>]\n";
        return 2;
    }
    const int status = run();
    if (status == 0)
    {
        std::cout << "replay image reader ok\n";
    }
    return status == 0 ? 0 : 1;
}
