/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spool_file.h"

#include "spool_layout.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <limits>
#include <string>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#ifdef __linux__
#include <linux/magic.h>
#include <sys/vfs.h>
#endif
#endif

namespace neurale::recording
{
namespace
{

#if defined(__linux__) || defined(_WIN32)
void prefault_writable_pages(void* address, std::size_t capacity, std::size_t page_size) noexcept
{
    auto* bytes = static_cast<volatile std::byte*>(address);
    for (std::size_t offset = 0; offset < capacity; offset += page_size)
    {
        bytes[offset] = std::byte{0};
    }
    bytes[capacity - 1] = std::byte{0};
}
#endif

#if defined(_WIN32)
struct WindowsPageLockState
{
    SRWLOCK mutex = SRWLOCK_INIT;
    SIZE_T original_minimum{};
    SIZE_T original_maximum{};
    SIZE_T locked_bytes{};
    bool initialized{};
};

WindowsPageLockState& windows_page_lock_state() noexcept
{
    static WindowsPageLockState state{};
    return state;
}

constexpr SIZE_T windows_working_set_overhead = 1U << 20U;

[[nodiscard]] bool windows_lock_pages(void* address, SIZE_T bytes, DWORD& error) noexcept
{
    auto& state = windows_page_lock_state();
    AcquireSRWLockExclusive(&state.mutex);
    const HANDLE process = GetCurrentProcess();
    if (!state.initialized &&
        !GetProcessWorkingSetSize(process, &state.original_minimum, &state.original_maximum))
    {
        error = GetLastError();
        ReleaseSRWLockExclusive(&state.mutex);
        return false;
    }
    state.initialized = true;
    if (bytes > std::numeric_limits<SIZE_T>::max() - state.locked_bytes ||
        bytes + state.locked_bytes >
            std::numeric_limits<SIZE_T>::max() - windows_working_set_overhead)
    {
        error = ERROR_NOT_ENOUGH_MEMORY;
        ReleaseSRWLockExclusive(&state.mutex);
        return false;
    }
    const SIZE_T required = state.locked_bytes + bytes + windows_working_set_overhead;
    const SIZE_T minimum = std::max(state.original_minimum, required);
    const SIZE_T maximum = std::max(state.original_maximum, minimum);
    if (!SetProcessWorkingSetSize(process, minimum, maximum) || !VirtualLock(address, bytes))
    {
        error = GetLastError();
        if (state.locked_bytes == 0)
        {
            static_cast<void>(
                SetProcessWorkingSetSize(process, state.original_minimum, state.original_maximum));
            state.initialized = false;
        }
        else
        {
            const SIZE_T current_required = state.locked_bytes + windows_working_set_overhead;
            const SIZE_T current_minimum = std::max(state.original_minimum, current_required);
            static_cast<void>(SetProcessWorkingSetSize(
                process, current_minimum, std::max(state.original_maximum, current_minimum)));
        }
        ReleaseSRWLockExclusive(&state.mutex);
        return false;
    }
    state.locked_bytes += bytes;
    ReleaseSRWLockExclusive(&state.mutex);
    return true;
}

[[nodiscard]] DWORD windows_unlock_pages(void* address, SIZE_T bytes) noexcept
{
    auto& state = windows_page_lock_state();
    AcquireSRWLockExclusive(&state.mutex);
    DWORD error = 0;
    if (!VirtualUnlock(address, bytes))
    {
        error = GetLastError();
    }
    state.locked_bytes = bytes <= state.locked_bytes ? state.locked_bytes - bytes : 0;
    const HANDLE process = GetCurrentProcess();
    const SIZE_T required = state.locked_bytes + windows_working_set_overhead;
    const SIZE_T minimum = state.locked_bytes == 0 ? state.original_minimum
                                                   : std::max(state.original_minimum, required);
    const SIZE_T maximum = state.locked_bytes == 0 ? state.original_maximum
                                                   : std::max(state.original_maximum, minimum);
    if (!SetProcessWorkingSetSize(process, minimum, maximum) && error == 0)
    {
        error = GetLastError();
    }
    if (state.locked_bytes == 0)
    {
        state.initialized = false;
    }
    ReleaseSRWLockExclusive(&state.mutex);
    return error;
}

[[nodiscard]] bool windows_local_path(const std::wstring& path) noexcept
{
    wchar_t volume[MAX_PATH]{};
    if (!GetVolumePathNameW(path.c_str(), volume, MAX_PATH))
    {
        return false;
    }
    return GetDriveTypeW(volume) != DRIVE_REMOTE;
}
#endif

#ifndef _WIN32
[[nodiscard]] SpoolIoStatus classify(int error) noexcept
{
    switch (error)
    {
    case ENOSPC:
    case EDQUOT:
    case EFBIG:
        return SpoolIoStatus::out_of_space;
    case EINTR:
    case ECANCELED:
        return SpoolIoStatus::cancelled;
    default:
        return SpoolIoStatus::io_error;
    }
}
#else
[[nodiscard]] SpoolIoStatus classify(DWORD error) noexcept
{
    switch (error)
    {
    case ERROR_DISK_FULL:
    case ERROR_HANDLE_DISK_FULL:
    case ERROR_FILE_TOO_LARGE:
    case ERROR_NOT_ENOUGH_QUOTA:
        return SpoolIoStatus::out_of_space;
    case ERROR_OPERATION_ABORTED:
        return SpoolIoStatus::cancelled;
    default:
        return SpoolIoStatus::io_error;
    }
}

[[nodiscard]] int platform_error(DWORD error) noexcept
{
    return error > static_cast<DWORD>(std::numeric_limits<int>::max())
               ? std::numeric_limits<int>::max()
               : static_cast<int>(error);
}

[[nodiscard]] HANDLE handle(std::intptr_t descriptor) noexcept
{
    return reinterpret_cast<HANDLE>(descriptor);
}

[[nodiscard]] bool utf8_path(const std::string& path, std::wstring& out) noexcept
{
    if (path.empty() || path.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    {
        return false;
    }
    const auto input_bytes = static_cast<int>(path.size());
    const int length =
        MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), input_bytes, nullptr, 0);
    if (length <= 0)
    {
        return false;
    }
    try
    {
        out.resize(static_cast<std::size_t>(length));
    }
    catch (...)
    {
        return false;
    }
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path.data(), input_bytes, out.data(),
                               length) == length;
}

[[nodiscard]] SpoolIoResult transfer_result(BOOL completed, OVERLAPPED& operation,
                                            HANDLE file_handle, DWORD transferred,
                                            std::size_t requested) noexcept
{
    if (!completed)
    {
        auto error = GetLastError();
        if (error != ERROR_IO_PENDING ||
            !GetOverlappedResult(file_handle, &operation, &transferred, TRUE))
        {
            error = GetLastError();
            if (error == ERROR_HANDLE_EOF)
            {
                return SpoolIoResult{.status = SpoolIoStatus::incomplete,
                                     .transferred = transferred,
                                     .platform_error = 0};
            }
            return SpoolIoResult{.status = classify(error),
                                 .transferred = transferred,
                                 .platform_error = platform_error(error)};
        }
    }
    return SpoolIoResult{.status = transferred == requested ? SpoolIoStatus::ok
                                                            : SpoolIoStatus::incomplete,
                         .transferred = transferred,
                         .platform_error = 0};
}
#endif

} // namespace

PlatformSpoolFile::~PlatformSpoolFile()
{
    static_cast<void>(close());
}

SpoolIoResult PlatformSpoolFile::create(const std::string& path, std::uint64_t capacity) noexcept
{
#ifndef _WIN32
    if (descriptor_ >= 0)
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
    const int descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0)
    {
        return SpoolIoResult{.status = classify(errno), .transferred = 0, .platform_error = errno};
    }
    descriptor_ = descriptor;
    size_ = 0;
    capacity_ = capacity;
    writable_ = true;
    return SpoolIoResult{};
#else
    if (is_open())
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    std::wstring wide_path;
    if (!utf8_path(path, wide_path))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error,
                             .platform_error = platform_error(ERROR_INVALID_NAME)};
    }
    if (!windows_local_path(wide_path))
        return SpoolIoResult{.status = SpoolIoStatus::io_error,
                             .platform_error = platform_error(ERROR_BAD_NETPATH)};
    const auto file_handle =
        CreateFileW(wide_path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                    CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE)
    {
        const auto error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    descriptor_ = reinterpret_cast<std::intptr_t>(file_handle);
    size_ = 0;
    capacity_ = capacity;
    writable_ = true;
    return SpoolIoResult{};
#endif
}

SpoolIoResult PlatformSpoolFile::open_existing(const std::string& path, SpoolOpenMode mode) noexcept
{
#ifndef _WIN32
    if (descriptor_ >= 0)
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
    const int flags =
        mode == SpoolOpenMode::read_write ? (O_RDWR | O_CLOEXEC) : (O_RDONLY | O_CLOEXEC);
    const int descriptor = ::open(path.c_str(), flags);
    if (descriptor < 0)
    {
        return SpoolIoResult{.status = classify(errno), .transferred = 0, .platform_error = errno};
    }
    struct stat status{};
    if (::fstat(descriptor, &status) != 0)
    {
        const int error = errno;
        ::close(descriptor);
        return SpoolIoResult{.status = classify(error), .transferred = 0, .platform_error = error};
    }
    descriptor_ = descriptor;
    size_ = static_cast<std::uint64_t>(status.st_size);
    writable_ = mode == SpoolOpenMode::read_write;
    return SpoolIoResult{};
#else
    if (is_open())
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    std::wstring wide_path;
    if (!utf8_path(path, wide_path))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error,
                             .platform_error = platform_error(ERROR_INVALID_NAME)};
    }
    const DWORD access =
        mode == SpoolOpenMode::read_write ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ;
    const auto file_handle =
        CreateFileW(wide_path.c_str(), access, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED, nullptr);
    if (file_handle == INVALID_HANDLE_VALUE)
    {
        const auto error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    LARGE_INTEGER bytes{};
    if (!GetFileSizeEx(file_handle, &bytes) || bytes.QuadPart < 0)
    {
        const auto error = GetLastError();
        CloseHandle(file_handle);
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    descriptor_ = reinterpret_cast<std::intptr_t>(file_handle);
    size_ = static_cast<std::uint64_t>(bytes.QuadPart);
    writable_ = mode == SpoolOpenMode::read_write;
    return SpoolIoResult{};
#endif
}

SpoolIoResult PlatformSpoolFile::close() noexcept
{
#ifndef _WIN32
    if (descriptor_ < 0)
    {
        return SpoolIoResult{};
    }
    const int descriptor = descriptor_;
    descriptor_ = -1;
    writable_ = false;
    if (::close(descriptor) != 0)
    {
        return SpoolIoResult{.status = classify(errno), .transferred = 0, .platform_error = errno};
    }
    return SpoolIoResult{};
#else
    if (!is_open())
    {
        return SpoolIoResult{};
    }
    const auto file_handle = handle(descriptor_);
    descriptor_ = -1;
    writable_ = false;
    if (!CloseHandle(file_handle))
    {
        const auto error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    return SpoolIoResult{};
#endif
}

SpoolIoResult PlatformSpoolFile::append(std::span<const std::byte> data) noexcept
{
    const auto current = size_.load(std::memory_order_acquire);
    if (current > capacity_ || data.size() > capacity_ - current)
    {
        return SpoolIoResult{.status = SpoolIoStatus::out_of_space};
    }
    if (!writable_)
    {
        // A read-only handle is for diagnosis: it never appends. Refused
        // explicitly rather than silently no-op'd, so a mistaken caller learns.
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
#ifndef _WIN32
    if (descriptor_ < 0)
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
    const auto written = ::pwrite(descriptor_, data.data(), data.size(), static_cast<off_t>(size_));
    if (written < 0)
    {
        return SpoolIoResult{.status = classify(errno), .transferred = 0, .platform_error = errno};
    }
    const auto transferred = static_cast<std::size_t>(written);
    size_ += transferred;
    return SpoolIoResult{.status = transferred == data.size() ? SpoolIoStatus::ok
                                                              : SpoolIoStatus::incomplete,
                         .transferred = transferred,
                         .platform_error = 0};
#else
    if (!is_open() || size_ > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    const auto requested = static_cast<DWORD>(std::min<std::size_t>(
        data.size(), static_cast<std::size_t>(std::numeric_limits<DWORD>::max())));
    OVERLAPPED operation{};
    operation.Offset = static_cast<DWORD>(size_ & 0xFFFFFFFFULL);
    operation.OffsetHigh = static_cast<DWORD>(size_ >> 32U);
    DWORD transferred = 0;
    const auto completed =
        WriteFile(handle(descriptor_), data.data(), requested, &transferred, &operation);
    auto result =
        transfer_result(completed, operation, handle(descriptor_), transferred, requested);
    size_ += result.transferred;
    if (result.status == SpoolIoStatus::ok && result.transferred != data.size())
    {
        result.status = SpoolIoStatus::incomplete;
    }
    return result;
#endif
}

SpoolIoResult PlatformSpoolFile::read_at(std::uint64_t offset, std::span<std::byte> out) noexcept
{
#ifndef _WIN32
    if (descriptor_ < 0)
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
    std::size_t total = 0;
    while (total < out.size())
    {
        const auto read = ::pread(descriptor_, out.data() + total, out.size() - total,
                                  static_cast<off_t>(offset + total));
        if (read < 0)
        {
            return SpoolIoResult{
                .status = classify(errno), .transferred = total, .platform_error = errno};
        }
        if (read == 0)
        {
            break;
        }
        total += static_cast<std::size_t>(read);
    }
    return SpoolIoResult{.status =
                             total == out.size() ? SpoolIoStatus::ok : SpoolIoStatus::incomplete,
                         .transferred = total,
                         .platform_error = 0};
#else
    if (!is_open() || offset > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    std::size_t total = 0;
    while (total < out.size())
    {
        const auto current = offset + total;
        if (current < offset ||
            current > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        {
            return SpoolIoResult{.status = SpoolIoStatus::io_error, .transferred = total};
        }
        const auto requested = static_cast<DWORD>(std::min<std::size_t>(
            out.size() - total, static_cast<std::size_t>(std::numeric_limits<DWORD>::max())));
        OVERLAPPED operation{};
        operation.Offset = static_cast<DWORD>(current & 0xFFFFFFFFULL);
        operation.OffsetHigh = static_cast<DWORD>(current >> 32U);
        DWORD transferred = 0;
        const auto completed =
            ReadFile(handle(descriptor_), out.data() + total, requested, &transferred, &operation);
        const auto result =
            transfer_result(completed, operation, handle(descriptor_), transferred, requested);
        total += result.transferred;
        if (result.status != SpoolIoStatus::ok)
        {
            return SpoolIoResult{.status = result.status,
                                 .transferred = total,
                                 .platform_error = result.platform_error};
        }
    }
    return SpoolIoResult{.status = SpoolIoStatus::ok, .transferred = total};
#endif
}

SpoolIoResult PlatformSpoolFile::sync() noexcept
{
    if (!writable_)
    {
        // A read-only handle makes no durability claim and has nothing to flush.
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
#ifndef _WIN32
    if (descriptor_ < 0)
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
    if (::fsync(descriptor_) != 0)
    {
        return SpoolIoResult{.status = classify(errno), .transferred = 0, .platform_error = errno};
    }
    return SpoolIoResult{};
#else
    if (!is_open())
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    if (!FlushFileBuffers(handle(descriptor_)))
    {
        const auto error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    return SpoolIoResult{};
#endif
}

SpoolIoResult PlatformSpoolFile::truncate(std::uint64_t bytes) noexcept
{
    if (!writable_)
    {
        // Only an explicit repair truncates, and only through a read-write
        // handle; a read-only diagnosis handle cannot reach this.
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
#ifndef _WIN32
    if (descriptor_ < 0)
    {
        return SpoolIoResult{
            .status = SpoolIoStatus::io_error, .transferred = 0, .platform_error = 0};
    }
    if (::ftruncate(descriptor_, static_cast<off_t>(bytes)) != 0)
    {
        return SpoolIoResult{.status = classify(errno), .transferred = 0, .platform_error = errno};
    }
    size_ = bytes;
    return SpoolIoResult{};
#else
    if (!is_open() || bytes > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    LARGE_INTEGER pos{};
    pos.QuadPart = static_cast<LONGLONG>(bytes);
    if (!SetFilePointerEx(handle(descriptor_), pos, nullptr, FILE_BEGIN) ||
        !SetEndOfFile(handle(descriptor_)))
    {
        const auto error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    size_ = bytes;
    return SpoolIoResult{};
#endif
}

std::uint64_t PlatformSpoolFile::size() const noexcept
{
    return size_;
}

BoundedMappedSpoolFile::~BoundedMappedSpoolFile()
{
    static_cast<void>(close());
}

SpoolIoResult BoundedMappedSpoolFile::create(const std::string& path,
                                             std::size_t capacity_bytes) noexcept
{
    if (is_open() || capacity_bytes == 0 ||
        capacity_bytes > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }

#if defined(__linux__)
    const int descriptor = ::open(path.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
    if (descriptor < 0)
    {
        return SpoolIoResult{.status = classify(errno), .platform_error = errno};
    }
    struct statfs file_system{};
    if (::fstatfs(descriptor, &file_system) != 0)
    {
        const int error = errno;
        ::close(descriptor);
        ::unlink(path.c_str());
        return SpoolIoResult{.status = classify(error), .platform_error = error};
    }
    if (static_cast<unsigned long>(file_system.f_type) != static_cast<unsigned long>(TMPFS_MAGIC))
    {
        ::close(descriptor);
        ::unlink(path.c_str());
        return SpoolIoResult{.status = SpoolIoStatus::io_error, .platform_error = EOPNOTSUPP};
    }
    const int allocation_error =
        ::posix_fallocate(descriptor, 0, static_cast<off_t>(capacity_bytes));
    if (allocation_error != 0)
    {
        ::close(descriptor);
        ::unlink(path.c_str());
        return SpoolIoResult{.status = classify(allocation_error),
                             .platform_error = allocation_error};
    }
    void* address =
        ::mmap(nullptr, capacity_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0);
    if (address == MAP_FAILED)
    {
        const int error = errno;
        ::close(descriptor);
        ::unlink(path.c_str());
        return SpoolIoResult{.status = classify(error), .platform_error = error};
    }
    if (::mlock(address, capacity_bytes) != 0)
    {
        const int error = errno;
        ::munmap(address, capacity_bytes);
        ::close(descriptor);
        ::unlink(path.c_str());
        return SpoolIoResult{.status = classify(error), .platform_error = error};
    }
    const long page_size = ::sysconf(_SC_PAGESIZE);
    if (page_size <= 0)
    {
        const int error = errno == 0 ? EINVAL : errno;
        ::munlock(address, capacity_bytes);
        ::munmap(address, capacity_bytes);
        ::close(descriptor);
        ::unlink(path.c_str());
        return SpoolIoResult{.status = classify(error), .platform_error = error};
    }
    prefault_writable_pages(address, capacity_bytes, static_cast<std::size_t>(page_size));

    descriptor_ = descriptor;
    mapping_ = static_cast<std::byte*>(address);
    pages_locked_ = true;
#elif defined(_WIN32)
    std::wstring wide_path;
    if (!utf8_path(path, wide_path) || !windows_local_path(wide_path))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error,
                             .platform_error = platform_error(ERROR_BAD_NETPATH)};
    }
    const HANDLE file =
        CreateFileW(wide_path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, nullptr,
                    CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE)
    {
        const DWORD error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    const auto fail = [&](DWORD error, HANDLE mapping_handle = nullptr, void* address = nullptr,
                          bool locked = false) noexcept
    {
        if (locked)
        {
            static_cast<void>(windows_unlock_pages(address, capacity_bytes));
        }
        if (address != nullptr)
        {
            static_cast<void>(UnmapViewOfFile(address));
        }
        if (mapping_handle != nullptr)
        {
            static_cast<void>(CloseHandle(mapping_handle));
        }
        static_cast<void>(CloseHandle(file));
        static_cast<void>(DeleteFileW(wide_path.c_str()));
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    };

    FILE_ALLOCATION_INFO allocation{};
    allocation.AllocationSize.QuadPart = static_cast<LONGLONG>(capacity_bytes);
    if (!SetFileInformationByHandle(file, FileAllocationInfo, &allocation, sizeof(allocation)))
    {
        return fail(GetLastError());
    }
    LARGE_INTEGER extent{};
    extent.QuadPart = static_cast<LONGLONG>(capacity_bytes);
    if (!SetFilePointerEx(file, extent, nullptr, FILE_BEGIN) || !SetEndOfFile(file))
    {
        return fail(GetLastError());
    }
    const HANDLE mapping_handle = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
    if (mapping_handle == nullptr)
    {
        return fail(GetLastError());
    }
    void* address =
        MapViewOfFile(mapping_handle, FILE_MAP_READ | FILE_MAP_WRITE, 0, 0, capacity_bytes);
    if (address == nullptr)
    {
        return fail(GetLastError(), mapping_handle);
    }
    DWORD lock_error = 0;
    if (!windows_lock_pages(address, capacity_bytes, lock_error))
    {
        return fail(lock_error, mapping_handle, address);
    }
    SYSTEM_INFO system{};
    GetSystemInfo(&system);
    if (system.dwPageSize == 0)
    {
        return fail(ERROR_INVALID_PARAMETER, mapping_handle, address, true);
    }
    prefault_writable_pages(address, capacity_bytes, system.dwPageSize);

    descriptor_ = reinterpret_cast<std::intptr_t>(file);
    mapping_descriptor_ = reinterpret_cast<std::intptr_t>(mapping_handle);
    mapping_ = static_cast<std::byte*>(address);
    pages_locked_ = true;
#else
    constexpr int unsupported = EOPNOTSUPP;
    (void)path;
    return SpoolIoResult{.status = SpoolIoStatus::io_error, .platform_error = unsupported};
#endif

    capacity_ = capacity_bytes;
    size_.store(0, std::memory_order_release);
    cancelled_.store(false, std::memory_order_release);
    bounded_backend_ready_ = true;
    return SpoolIoResult{};
}

SpoolIoResult BoundedMappedSpoolFile::close() noexcept
{
    if (!is_open())
    {
        return SpoolIoResult{};
    }
#if defined(_WIN32)
    const auto logical_size = size_.load(std::memory_order_acquire);
    DWORD first_error = 0;
    auto remember = [&](DWORD error) noexcept
    {
        if (first_error == 0)
        {
            first_error = error;
        }
    };
    auto* address = mapping_;
    mapping_ = nullptr;
    const HANDLE file = handle(descriptor_);
    const HANDLE mapping_handle = handle(mapping_descriptor_);

    if (logical_size != 0 && !FlushViewOfFile(address, static_cast<SIZE_T>(logical_size)))
    {
        remember(GetLastError());
    }
    if (!FlushFileBuffers(file))
    {
        remember(GetLastError());
    }
    if (pages_locked_)
    {
        const DWORD error = windows_unlock_pages(address, capacity_);
        if (error != 0)
        {
            remember(error);
        }
    }
    pages_locked_ = false;
    if (!UnmapViewOfFile(address))
    {
        remember(GetLastError());
    }
    if (!CloseHandle(mapping_handle))
    {
        remember(GetLastError());
    }
    mapping_descriptor_ = -1;
    LARGE_INTEGER extent{};
    extent.QuadPart = static_cast<LONGLONG>(logical_size);
    if (!SetFilePointerEx(file, extent, nullptr, FILE_BEGIN) || !SetEndOfFile(file))
    {
        remember(GetLastError());
    }
    if (!FlushFileBuffers(file))
    {
        remember(GetLastError());
    }
    if (!CloseHandle(file))
    {
        remember(GetLastError());
    }
    descriptor_ = -1;
    capacity_ = 0;
    bounded_backend_ready_ = false;
    return first_error == 0 ? SpoolIoResult{}
                            : SpoolIoResult{.status = classify(first_error),
                                            .platform_error = platform_error(first_error)};
#elif !defined(__linux__)
    return SpoolIoResult{.status = SpoolIoStatus::io_error};
#else
    const auto logical_size = size_.load(std::memory_order_acquire);
    int first_error = 0;

    auto* address = mapping_;
    mapping_ = nullptr;
    if (pages_locked_ && ::munlock(address, capacity_) != 0)
    {
        first_error = errno;
    }
    pages_locked_ = false;
    if (::munmap(address, capacity_) != 0 && first_error == 0)
    {
        first_error = errno;
    }
    const int descriptor = descriptor_;
    descriptor_ = -1;
    if (::ftruncate(descriptor, static_cast<off_t>(logical_size)) != 0 && first_error == 0)
    {
        first_error = errno;
    }
    if (::close(descriptor) != 0 && first_error == 0)
    {
        first_error = errno;
    }

    capacity_ = 0;
    bounded_backend_ready_ = false;
    return first_error == 0
               ? SpoolIoResult{}
               : SpoolIoResult{.status = SpoolIoStatus::io_error, .platform_error = first_error};
#endif
}

SpoolIoResult BoundedMappedSpoolFile::append(std::span<const std::byte> data) noexcept
{
    if (!is_open())
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    if (cancelled_.load(std::memory_order_acquire))
    {
        return SpoolIoResult{.status = SpoolIoStatus::cancelled};
    }
    const auto current = size_.load(std::memory_order_relaxed);
    if (current > capacity_ || data.size() > capacity_ - static_cast<std::size_t>(current))
    {
        return SpoolIoResult{.status = SpoolIoStatus::out_of_space};
    }
    if (!data.empty())
    {
        std::memcpy(mapping_ + static_cast<std::size_t>(current), data.data(), data.size());
    }
    size_.store(current + data.size(), std::memory_order_release);
    return SpoolIoResult{.status = SpoolIoStatus::ok, .transferred = data.size()};
}

SpoolIoResult BoundedMappedSpoolFile::read_at(std::uint64_t offset,
                                              std::span<std::byte> out) noexcept
{
    if (!is_open())
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    const auto logical_size = size_.load(std::memory_order_acquire);
    const auto available = offset < logical_size ? logical_size - offset : 0;
    const auto transferred = static_cast<std::size_t>(
        std::min<std::uint64_t>(available, static_cast<std::uint64_t>(out.size())));
    if (transferred != 0)
    {
        std::memcpy(out.data(), mapping_ + static_cast<std::size_t>(offset), transferred);
    }
    return SpoolIoResult{.status = transferred == out.size() ? SpoolIoStatus::ok
                                                             : SpoolIoStatus::incomplete,
                         .transferred = transferred};
}

SpoolIoResult BoundedMappedSpoolFile::sync() noexcept
{
    if (!is_open())
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
#if defined(__linux__)
    const auto logical_size = size_.load(std::memory_order_acquire);
    if (logical_size != 0 &&
        ::msync(mapping_, static_cast<std::size_t>(logical_size), MS_SYNC) != 0)
    {
        return SpoolIoResult{.status = classify(errno), .platform_error = errno};
    }
    if (::fsync(descriptor_) != 0)
    {
        return SpoolIoResult{.status = classify(errno), .platform_error = errno};
    }
#elif defined(_WIN32)
    const auto logical_size = size_.load(std::memory_order_acquire);
    if (logical_size != 0 && !FlushViewOfFile(mapping_, static_cast<SIZE_T>(logical_size)))
    {
        const DWORD error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
    if (!FlushFileBuffers(handle(descriptor_)))
    {
        const DWORD error = GetLastError();
        return SpoolIoResult{.status = classify(error), .platform_error = platform_error(error)};
    }
#else
    return SpoolIoResult{.status = SpoolIoStatus::io_error};
#endif
    return SpoolIoResult{};
}

SpoolIoResult BoundedMappedSpoolFile::truncate(std::uint64_t bytes) noexcept
{
    if (!is_open() || bytes > size_.load(std::memory_order_acquire))
    {
        return SpoolIoResult{.status = SpoolIoStatus::io_error};
    }
    size_.store(bytes, std::memory_order_release);
    return close();
}

std::uint64_t BoundedMappedSpoolFile::size() const noexcept
{
    return size_.load(std::memory_order_acquire);
}

bool BoundedMappedSpoolFile::supports_durability_policy(DurabilityPolicy policy) const noexcept
{
    return policy == DurabilityPolicy::buffered;
}

} // namespace neurale::recording
