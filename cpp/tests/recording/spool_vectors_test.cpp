/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// The C++ reader against every normative vector of the recording-spool specification.
///
/// This is the differential test that matters for this task: the vectors and
/// `vectors/index.json` were produced by an independent Python reference, and
/// a conforming implementation "reproduces the verdict in index.json for every
/// vector" (specification section 12). Nothing here compares diagnostic
/// wording -- only the codes, which are the stable part.

#include "check_returns.h"
#include "sha256.h"
#include "spool_scanner.h"
#include "spool_test_support.h"
#include "test_json.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace
{

#define CHECK_VECTOR(condition, name)                                                              \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::cerr << "vector " << (name) << " failed at line " << __LINE__ << '\n';            \
            return __LINE__;                                                                       \
        }                                                                                          \
    } while (false)

using neurale::recording::capture_outcome_text;
using neurale::recording::CaptureOutcome;
using neurale::recording::durability_policy_text;
using neurale::recording::DurabilityPolicy;
using neurale::recording::requested_terminal_intent_text;
using neurale::recording::RequestedTerminalIntent;
using neurale::recording::scan_status_text;
using neurale::recording::ScanStatus;
using neurale::recording::sha256;
using neurale::recording::spool_code_text;
using neurale::recording::SpoolScanner;
using neurale::recording::SpoolScanReport;
using neurale::recording::test::JsonParser;
using neurale::recording::test::JsonValue;
using neurale::recording::test::MemorySpoolFile;
using neurale::recording::test::read_file;
using neurale::recording::test::read_text_file;

[[nodiscard]] std::string vector_dir()
{
    return std::string{NEURALE_NATIVE_SPOOL_VECTOR_DIR};
}

[[nodiscard]] std::string hex(const std::array<std::uint8_t, 32>& digest)
{
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    for (const auto value : digest)
    {
        out.push_back(kDigits[value >> 4]);
        out.push_back(kDigits[value & 0x0F]);
    }
    return out;
}

int test_every_vector_reproduces_index_verdict()
{
    bool ok = false;
    const auto text = read_text_file(vector_dir() + "/index.json", ok);
    CHECK(ok);

    JsonValue idx;
    JsonParser parser{text};
    CHECK(parser.parse(idx));

    const auto* format = idx.find("format");
    CHECK(format != nullptr && format->string == "neurale-native-spool");
    const auto* version_major = idx.find("version_major");
    CHECK(version_major != nullptr && version_major->integer == 1);
    const auto* checksum = idx.find("checksum");
    CHECK(checksum != nullptr && checksum->string == "crc32c");

    const auto* vectors = idx.find("vectors");
    CHECK(vectors != nullptr && vectors->type == JsonValue::Type::array);
    // The vector set only ever grows; a shrunken one means the index was
    // rewritten and this differential test would be checking nothing.
    CHECK(vectors->array.size() >= 62);

    std::vector<std::byte> scratch(4096, std::byte{0});
    for (const auto& entry : vectors->array)
    {
        const auto* name_value = entry.find("name");
        CHECK(name_value != nullptr);
        const auto& name = name_value->string;
        const auto* file_value = entry.find("file");
        CHECK_VECTOR(file_value != nullptr, name);

        MemorySpoolFile file;
        bool read_ok = false;
        file.data = read_file(vector_dir() + "/" + file_value->string, read_ok);
        CHECK_VECTOR(read_ok, name);

        const auto* bytes_value = entry.find("bytes");
        CHECK_VECTOR(bytes_value != nullptr, name);
        CHECK_VECTOR(static_cast<std::int64_t>(file.data.size()) == bytes_value->integer, name);
        const auto* sha_value = entry.find("sha256");
        CHECK_VECTOR(sha_value != nullptr, name);
        CHECK_VECTOR(hex(sha256(file.data)) == sha_value->string, name);

        const auto* expect = entry.find("expect");
        CHECK_VECTOR(expect != nullptr && expect->type == JsonValue::Type::object, name);

        const auto before = file.data;
        SpoolScanner scanner;
        const auto report = scanner.scan(file, scratch);

        // A read of a spool never writes, never truncates, and never repairs.
        CHECK_VECTOR(file.data == before, name);
        CHECK_VECTOR(!report.read_failed(), name);

        const auto* status = expect->find("status");
        CHECK_VECTOR(status != nullptr, name);
        CHECK_VECTOR(scan_status_text(report.status()) == status->string, name);

        const auto integer = [&](const char* key, std::int64_t& out)
        {
            const auto* value = expect->find(key);
            if (value == nullptr || value->type != JsonValue::Type::integer)
            {
                return false;
            }
            out = value->integer;
            return true;
        };
        std::int64_t expected = 0;
        CHECK_VECTOR(integer("committed_prefix_end", expected), name);
        CHECK_VECTOR(static_cast<std::int64_t>(report.committed_prefix_end()) == expected, name);
        CHECK_VECTOR(integer("committed_transactions", expected), name);
        CHECK_VECTOR(static_cast<std::int64_t>(report.committed_transactions()) == expected, name);
        CHECK_VECTOR(integer("last_transaction_id", expected), name);
        CHECK_VECTOR(static_cast<std::int64_t>(report.last_transaction_id()) == expected, name);
        CHECK_VECTOR(integer("data_items", expected), name);
        CHECK_VECTOR(static_cast<std::int64_t>(report.data_items()) == expected, name);
        CHECK_VECTOR(integer("control_items", expected), name);
        CHECK_VECTOR(static_cast<std::int64_t>(report.control_items()) == expected, name);
        CHECK_VECTOR(integer("durable_extent_bytes", expected), name);
        CHECK_VECTOR(static_cast<std::int64_t>(report.durable_extent_bytes()) == expected, name);

        const auto* session_end_present = expect->find("session_end_present");
        CHECK_VECTOR(session_end_present != nullptr, name);
        CHECK_VECTOR(report.session_end_present() == session_end_present->boolean, name);

        const auto* finalizable = expect->find("finalizable");
        CHECK_VECTOR(finalizable != nullptr, name);
        CHECK_VECTOR(report.finalizable() == finalizable->boolean, name);

        const auto* outcome = expect->find("capture_outcome");
        CHECK_VECTOR(outcome != nullptr, name);
        if (outcome->is_null())
        {
            CHECK_VECTOR(!report.has_capture_outcome(), name);
        }
        else
        {
            CHECK_VECTOR(report.has_capture_outcome(), name);
            CHECK_VECTOR(capture_outcome_text(report.capture_outcome()) == outcome->string, name);
        }

        const auto* intent = expect->find("requested_terminal_intent");
        CHECK_VECTOR(intent != nullptr, name);
        if (intent->is_null())
        {
            CHECK_VECTOR(!report.has_requested_terminal_intent(), name);
        }
        else
        {
            CHECK_VECTOR(report.has_requested_terminal_intent(), name);
            CHECK_VECTOR(requested_terminal_intent_text(report.requested_terminal_intent()) ==
                             intent->string,
                         name);
        }

        const auto* primary_fault = expect->find("primary_fault_committed");
        CHECK_VECTOR(primary_fault != nullptr, name);
        if (primary_fault->is_null())
        {
            CHECK_VECTOR(!report.has_primary_fault_committed(), name);
        }
        else
        {
            CHECK_VECTOR(report.has_primary_fault_committed(), name);
            CHECK_VECTOR(report.primary_fault_committed() == primary_fault->boolean, name);
        }

        const auto* reason = expect->find("terminal_reason");
        CHECK_VECTOR(reason != nullptr, name);
        if (reason->is_null())
        {
            CHECK_VECTOR(!report.has_terminal_reason(), name);
            CHECK_VECTOR(report.terminal_reason().empty(), name);
        }
        else
        {
            CHECK_VECTOR(report.has_terminal_reason(), name);
            CHECK_VECTOR(report.terminal_reason() == reason->string, name);
        }

        const auto* policy = expect->find("durability_policy");
        CHECK_VECTOR(policy != nullptr, name);
        if (policy->is_null())
        {
            CHECK_VECTOR(!report.has_durability_policy(), name);
        }
        else
        {
            CHECK_VECTOR(report.has_durability_policy(), name);
            CHECK_VECTOR(durability_policy_text(report.durability_policy()) == policy->string,
                         name);
        }

        const auto* codes = expect->find("codes");
        CHECK_VECTOR(codes != nullptr && codes->type == JsonValue::Type::array, name);
        CHECK_VECTOR(!report.findings_truncated(), name);
        CHECK_VECTOR(report.findings().size() == codes->array.size(), name);
        for (std::size_t i = 0; i < codes->array.size(); ++i)
        {
            CHECK_VECTOR(spool_code_text(report.findings()[i].code) == codes->array[i].string,
                         name);
        }
    }
    return 0;
}

int test_rejected_spool_reports_no_committed_prefix()
{
    // The one case where nothing at all is readable, checked as a property
    // rather than vector by vector.
    bool ok = false;
    const auto text = read_text_file(vector_dir() + "/index.json", ok);
    CHECK(ok);
    JsonValue idx;
    JsonParser parser{text};
    CHECK(parser.parse(idx));
    const auto* vectors = idx.find("vectors");
    CHECK(vectors != nullptr);

    std::vector<std::byte> scratch(4096, std::byte{0});
    std::size_t rejected = 0;
    for (const auto& entry : vectors->array)
    {
        MemorySpoolFile file;
        bool read_ok = false;
        file.data = read_file(vector_dir() + "/" + entry.find("file")->string, read_ok);
        CHECK(read_ok);
        SpoolScanner scanner;
        const auto report = scanner.scan(file, scratch);
        if (report.status() != ScanStatus::rejected)
        {
            // Whatever else it says, a readable spool never reports a durable
            // extent past its committed one.
            CHECK(report.durable_extent_bytes() <= report.committed_prefix_end());
            continue;
        }
        ++rejected;
        CHECK(!report.readable());
        CHECK(!report.finalizable());
        CHECK(report.committed_prefix_end() == 0);
        CHECK(report.committed_transactions() == 0);
        CHECK(report.data_items() == 0);
        CHECK(report.control_items() == 0);
        CHECK(!report.session_end_present());
    }
    CHECK(rejected >= 7);
    return 0;
}

int test_record_cursor_walks_committed_prefix()
{
    MemorySpoolFile file;
    bool ok = false;
    file.data = read_file(vector_dir() + "/valid-complete.spool", ok);
    CHECK(ok);

    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    const auto report = scanner.scan(file, scratch);
    CHECK(report.status() == ScanStatus::ok);

    neurale::recording::SpoolRecordCursor cursor{file, report};
    neurale::recording::SpoolRecordView view{};
    std::size_t records = 0;
    std::uint64_t data_items = 0;
    std::uint64_t control_items = 0;
    std::uint64_t last_transaction = 0;
    while (cursor.next(view))
    {
        ++records;
        if (neurale::recording::is_data_plane_item(view.kind))
        {
            ++data_items;
        }
        if (neurale::recording::is_control_plane_item(view.kind))
        {
            ++control_items;
        }
        CHECK(view.transaction_id >= last_transaction);
        last_transaction = view.transaction_id;
        CHECK(view.payload_offset + view.payload_bytes <= report.committed_prefix_end());

        std::vector<std::byte> payload(view.payload_bytes, std::byte{0});
        CHECK(cursor.read_payload(view, payload));
    }
    CHECK(!cursor.failed());
    // Six data-transaction records, one checkpoint, one accounting, one end.
    CHECK(records == 9);
    CHECK(data_items == report.data_items());
    CHECK(control_items == report.control_items());
    CHECK(last_transaction == report.last_transaction_id());
    return 0;
}

int test_cursor_stops_at_torn_tail()
{
    MemorySpoolFile file;
    bool ok = false;
    file.data = read_file(vector_dir() + "/torn-missing-trailer.spool", ok);
    CHECK(ok);

    std::vector<std::byte> scratch(4096, std::byte{0});
    SpoolScanner scanner;
    const auto report = scanner.scan(file, scratch);
    CHECK(report.status() == ScanStatus::torn_tail);

    neurale::recording::SpoolRecordCursor cursor{file, report};
    neurale::recording::SpoolRecordView view{};
    while (cursor.next(view))
    {
        // Nothing the cursor yields may lie in the tail: a record inside an
        // incomplete transaction must not be promoted by any code path.
        CHECK(view.record_offset < report.committed_prefix_end());
    }
    CHECK(!cursor.failed());
    return 0;
}

} // namespace

int main()
{
    for (const auto test : {
             test_every_vector_reproduces_index_verdict,
             test_rejected_spool_reports_no_committed_prefix,
             test_record_cursor_walks_committed_prefix,
             test_cursor_stops_at_torn_tail,
         })
    {
        if (const auto line = test(); line != 0)
        {
            return line;
        }
    }
    return 0;
}
