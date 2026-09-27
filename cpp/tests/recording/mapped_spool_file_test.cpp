/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "check_returns.h"
#include "sha256.h"
#include "spool_file.h"
#include "spool_layout.h"
#include "spool_writer.h"

#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#if defined(__linux__)
#include <sys/resource.h>
#include <sys/stat.h>
#elif defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

using namespace neurale::recording;

namespace
{

std::filesystem::path temporary_spool()
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
#if defined(__linux__)
    return std::filesystem::path{"/dev/shm"} /
#else
    return std::filesystem::temp_directory_path() /
#endif
           ("neurale-bounded-mapped-" + std::to_string(nonce) + ".spool");
}

#if defined(__linux__) || defined(_WIN32)
#if defined(__linux__)
bool physical_capacity_is_reserved(const std::filesystem::path& path, std::uint64_t capacity)
{
    struct stat information{};
    return ::stat(path.c_str(), &information) == 0 &&
           static_cast<std::uint64_t>(information.st_blocks) * 512U >= capacity;
}

std::uint64_t minor_faults()
{
    struct rusage usage{};
    return ::getrusage(RUSAGE_SELF, &usage) == 0 ? static_cast<std::uint64_t>(usage.ru_minflt)
                                                 : std::numeric_limits<std::uint64_t>::max();
}
#endif

int test_fixed_resident_mapping_is_bounded_buffered_storage()
{
    const auto path = temporary_spool();
    BoundedMappedSpoolFile file;
    CHECK(file.create(path.string(), 4U << 20U).status == SpoolIoStatus::ok);
    CHECK(file.supports_bounded_cancel());
    CHECK(file.supports_durability_policy(DurabilityPolicy::buffered));
    CHECK(!file.supports_durability_policy(DurabilityPolicy::checkpoint_sync));
    CHECK(!file.supports_durability_policy(DurabilityPolicy::transaction_sync));

    constexpr std::array<std::byte, 4> bytes{std::byte{0x10}, std::byte{0x20}, std::byte{0x30},
                                             std::byte{0x40}};
    CHECK(file.append(bytes).status == SpoolIoStatus::ok);
    std::array<std::byte, 4> read{};
    CHECK(file.read_at(0, read).status == SpoolIoStatus::ok);
    CHECK(read == bytes);

    CHECK(file.close().status == SpoolIoStatus::ok);
    CHECK(std::filesystem::file_size(path) == bytes.size());
    std::filesystem::remove(path);
    return 0;
}

int test_cancel_is_observable_without_entering_file_io()
{
    const auto path = temporary_spool();
    BoundedMappedSpoolFile file;
    CHECK(file.create(path.string(), 64U << 10U).status == SpoolIoStatus::ok);
    file.request_cancel();
    constexpr std::array<std::byte, 1> byte{std::byte{0x01}};
    CHECK(file.append(byte).status == SpoolIoStatus::cancelled);
    CHECK(file.size() == 0);
    CHECK(file.close().status == SpoolIoStatus::ok);
    std::filesystem::remove(path);
    return 0;
}

#if defined(__linux__)
int test_capacity_is_reserved_and_write_path_prefaulted()
{
    constexpr std::size_t capacity = 4U << 20U;
    const auto path = temporary_spool();
    BoundedMappedSpoolFile file;
    CHECK(file.create(path.string(), capacity).status == SpoolIoStatus::ok);
    CHECK(physical_capacity_is_reserved(path, capacity));

    const std::vector<std::byte> plan{};
    SpoolSessionIdentity identity{};
    identity.session_id = "mapped-readiness-test";
    identity.plan_document = plan;
    identity.plan_fingerprint = sha256(plan);
    SpoolWriter writer;
    const SpoolWriterLimits limits{
        .max_records_per_transaction = 2,
        .max_transaction_bytes = 4096,
    };
    CHECK(writer.prepare(file, identity, DurabilityPolicy::buffered, limits) ==
          SpoolWriterStatus::ok);
    CHECK(writer.committed_extent() > 0);
    CHECK(writer.durable_extent() == writer.committed_extent());

    const auto remaining = capacity - static_cast<std::size_t>(file.size());
    const auto first_size = remaining / 2;
    const std::vector<std::byte> first(first_size, std::byte{0x5A});
    const std::vector<std::byte> second(remaining - first_size, std::byte{0xA5});
    std::array<std::byte, 64> warm_source{};
    std::array<std::byte, 64> warm_destination{};
    std::memcpy(warm_destination.data(), warm_source.data(), warm_source.size());
    CHECK(warm_destination == warm_source);
    static_cast<void>(minor_faults());
    const auto before = minor_faults();
    CHECK(file.append(first).status == SpoolIoStatus::ok);
    const auto after = minor_faults();
    CHECK(before != std::numeric_limits<std::uint64_t>::max());
    CHECK(after == before);

    // A second real sync must not turn resident tmpfs pages back into
    // file-backed write faults. This is deliberately stricter than the running
    // buffered policy, which performs no sync after readiness.
    CHECK(file.sync().status == SpoolIoStatus::ok);
    const auto before_second = minor_faults();
    CHECK(file.append(second).status == SpoolIoStatus::ok);
    const auto after_second = minor_faults();
    CHECK(after_second == before_second);

    CHECK(file.close().status == SpoolIoStatus::ok);
    std::filesystem::remove(path);
    return 0;
}
#endif

int test_buffered_writer_claims_only_synced_superblock()
{
    const auto path = temporary_spool();
    BoundedMappedSpoolFile file;
    CHECK(file.create(path.string(), 64U << 10U).status == SpoolIoStatus::ok);

    const std::vector<std::byte> plan{};
    SpoolSessionIdentity identity{};
    identity.session_id = "mapped-sync-test";
    identity.plan_document = plan;
    identity.plan_fingerprint = sha256(plan);
    SpoolWriter writer;
    const SpoolWriterLimits limits{
        .max_records_per_transaction = 2,
        .max_transaction_bytes = 4096,
    };
    CHECK(writer.prepare(file, identity, DurabilityPolicy::buffered, limits) ==
          SpoolWriterStatus::ok);
    CHECK(writer.committed_extent() > 0);
    CHECK(writer.durable_extent() == writer.committed_extent());

    CHECK(file.close().status == SpoolIoStatus::ok);
    std::filesystem::remove(path);
    return 0;
}

int test_mappings_close_out_of_creation_order()
{
    const auto first_path = temporary_spool();
    auto second_path = first_path;
    second_path += ".second";
    BoundedMappedSpoolFile first;
    BoundedMappedSpoolFile second;
    CHECK(first.create(first_path.string(), 64U << 10U).status == SpoolIoStatus::ok);
    CHECK(second.create(second_path.string(), 64U << 10U).status == SpoolIoStatus::ok);
    CHECK(first.close().status == SpoolIoStatus::ok);

    constexpr std::array<std::byte, 1> byte{std::byte{0x01}};
    CHECK(second.append(byte).status == SpoolIoStatus::ok);
    CHECK(second.close().status == SpoolIoStatus::ok);
    std::filesystem::remove(first_path);
    std::filesystem::remove(second_path);
    return 0;
}

#if defined(_WIN32)
int test_default_windows_capacity_is_lockable()
{
    constexpr std::size_t capacity = 64U << 20U;
    const auto path = temporary_spool();
    BoundedMappedSpoolFile file;
    CHECK(file.create(path.string(), capacity).status == SpoolIoStatus::ok);
    CHECK(std::filesystem::file_size(path) == capacity);
    CHECK(file.supports_bounded_cancel());
    CHECK(file.close().status == SpoolIoStatus::ok);
    std::filesystem::remove(path);
    return 0;
}

int run_windows_crash_child(const std::filesystem::path& path)
{
    BoundedMappedSpoolFile file;
    if (file.create(path.string(), 64U << 10U).status != SpoolIoStatus::ok)
    {
        return 2;
    }
    constexpr std::array<std::byte, 4> bytes{std::byte{0xDE}, std::byte{0xAD}, std::byte{0xBE},
                                             std::byte{0xEF}};
    if (file.append(bytes).status != SpoolIoStatus::ok)
    {
        return 3;
    }
    TerminateProcess(GetCurrentProcess(), 99);
    return 4;
}

int test_windows_mapping_survives_process_termination()
{
    const auto path = temporary_spool();
    wchar_t executable[MAX_PATH]{};
    CHECK(GetModuleFileNameW(nullptr, executable, MAX_PATH) != 0);
    std::wstring command =
        L"\"" + std::wstring{executable} + L"\" --crash-child \"" + path.wstring() + L"\"";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION process{};
    CHECK(CreateProcessW(nullptr, mutable_command.data(), nullptr, nullptr, FALSE, CREATE_NO_WINDOW,
                         nullptr, nullptr, &startup, &process));
    CHECK(WaitForSingleObject(process.hProcess, 30000) == WAIT_OBJECT_0);
    DWORD exit_code = 0;
    CHECK(GetExitCodeProcess(process.hProcess, &exit_code));
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CHECK(exit_code == 99);

    PlatformSpoolFile persisted;
    CHECK(persisted.open_existing(path.string(), SpoolOpenMode::read_only).status ==
          SpoolIoStatus::ok);
    std::array<std::byte, 4> bytes{};
    CHECK(persisted.read_at(0, bytes).status == SpoolIoStatus::ok);
    constexpr std::array<std::byte, 4> expected{std::byte{0xDE}, std::byte{0xAD}, std::byte{0xBE},
                                                std::byte{0xEF}};
    CHECK(bytes == expected);
    CHECK(persisted.close().status == SpoolIoStatus::ok);
    std::filesystem::remove(path);
    return 0;
}
#endif

#if defined(__linux__)
int test_disk_backed_mapping_is_rejected()
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("neurale-unbounded-mapped-" + std::to_string(nonce) + ".spool");
    BoundedMappedSpoolFile file;
    const auto created = file.create(path.string(), 64U << 10U);
    CHECK(created.status == SpoolIoStatus::io_error);
    CHECK(created.platform_error != 0);
    CHECK(!file.supports_bounded_cancel());
    CHECK(!std::filesystem::exists(path));
    return 0;
}
#endif

#else

int test_unavailable_backend_creates_no_file()
{
    const auto path = temporary_spool();
    BoundedMappedSpoolFile file;
    const auto created = file.create(path.string(), 64U << 10U);
    CHECK(created.status == SpoolIoStatus::io_error);
    CHECK(created.platform_error != 0);
    CHECK(!file.supports_bounded_cancel());
    CHECK(!std::filesystem::exists(path));
    return 0;
}

#endif

} // namespace

int main(int argc, char** argv)
{
#if defined(_WIN32)
    if (argc == 3 && std::string_view{argv[1]} == "--crash-child")
    {
        return run_windows_crash_child(argv[2]);
    }
#else
    (void)argc;
    (void)argv;
#endif
#if defined(__linux__) || defined(_WIN32)
    if (const auto line = test_fixed_resident_mapping_is_bounded_buffered_storage(); line != 0)
        return line;
    if (const auto line = test_cancel_is_observable_without_entering_file_io(); line != 0)
        return line;
#if defined(__linux__)
    if (const auto line = test_capacity_is_reserved_and_write_path_prefaulted(); line != 0)
        return line;
#endif
    if (const auto line = test_buffered_writer_claims_only_synced_superblock(); line != 0)
        return line;
    if (const auto line = test_mappings_close_out_of_creation_order(); line != 0)
        return line;
#if defined(_WIN32)
    if (const auto line = test_default_windows_capacity_is_lockable(); line != 0)
        return line;
    return test_windows_mapping_survives_process_termination();
#elif defined(__linux__)
    return test_disk_backed_mapping_is_rejected();
#else
    return 0;
#endif
#else
    return test_unavailable_backend_creates_no_file();
#endif
}
