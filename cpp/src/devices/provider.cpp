/* SPDX-License-Identifier: MIT */
#include <algorithm>
#include <atomic>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <neurale/devices/provider.h>
#include <new>
#include <provider.h>
#include <stdexcept>
#include <thread>
#include <utility>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace neurale::devices
{
using namespace streaming;
namespace
{
constexpr uint64_t magic = 0x504e444556303032ULL;
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(std::atomic<uint32_t>::is_always_lock_free);
struct alignas(64) QueueHeader
{
    uint64_t signature{}, capacity{}, stride{}, bytes{};
    std::atomic<uint64_t> write{0}, read{0}, published{0};
    std::atomic<uint32_t> cancelled{0}, error{0}, finished{0};
};
struct alignas(64) Slot
{
    std::atomic<uint64_t> sequence{0};
    uint64_t count{};
    uint64_t bytes{};
};
uint64_t aligned(uint64_t n)
{
    return (n + 63) & ~uint64_t{63};
}
struct Mapping
{
    void* data{};
    std::size_t bytes{};
    std::filesystem::path path;
    bool owner{}, remove_directory{};
#ifdef _WIN32
    HANDLE file{INVALID_HANDLE_VALUE}, mapping{};
#else
    int file{-1};
#endif
    ~Mapping()
    {
#ifdef _WIN32
        if (data)
            UnmapViewOfFile(data);
        if (mapping)
            CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE)
            CloseHandle(file);
#else
        if (data)
            munmap(data, bytes);
        if (file >= 0)
            ::close(file);
#endif
        if (owner)
        {
            std::error_code ignored;
            std::filesystem::remove(path, ignored);
            if (remove_directory)
                std::filesystem::remove(path.parent_path(), ignored);
        }
    }
    void open(const std::string& filename, std::size_t length, bool create)
    {
        path = std::filesystem::u8path(filename);
        owner = create;
        bytes = length;
#ifdef _WIN32
        file = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                           FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                           create ? CREATE_ALWAYS : OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE)
            throw std::runtime_error("cannot open device shared memory");
        LARGE_INTEGER size{};
        size.QuadPart = static_cast<LONGLONG>(length);
        if (create)
        {
            if (!SetFilePointerEx(file, size, nullptr, FILE_BEGIN) || !SetEndOfFile(file))
                throw std::runtime_error("cannot size device shared memory");
        }
        else
        {
            LARGE_INTEGER actual{};
            if (!GetFileSizeEx(file, &actual) || actual.QuadPart != size.QuadPart)
                throw std::runtime_error("device shared memory size mismatch");
        }
        mapping = CreateFileMappingW(file, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (!mapping)
            throw std::runtime_error("cannot map device shared memory");
        data = MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, length);
        if (!data)
            throw std::runtime_error("cannot view device shared memory");
#else
        file = ::open(path.c_str(), O_RDWR | (create ? O_CREAT | O_TRUNC : 0), 0600);
        if (file < 0)
            throw std::runtime_error("cannot open device shared memory");
        struct stat info{};
        if ((create && ftruncate(file, static_cast<off_t>(length)) != 0) ||
            fstat(file, &info) != 0 || static_cast<uint64_t>(info.st_size) != length)
            throw std::runtime_error("device shared memory size mismatch");
        data = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, file, 0);
        if (data == MAP_FAILED)
        {
            data = nullptr;
            throw std::runtime_error("cannot map device shared memory");
        }
#endif
    }
};
struct Library
{
#ifdef _WIN32
    HMODULE handle{};
#else
    void* handle{};
#endif
    ~Library()
    {
#ifdef _WIN32
        if (handle)
            FreeLibrary(handle);
#else
        if (handle)
            dlclose(handle);
#endif
    }
    const pn_provider_v1* load(const std::string& path)
    {
#ifdef _WIN32
        handle =
            LoadLibraryExW(std::filesystem::u8path(path).c_str(), nullptr,
                           LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        auto entry =
            handle
                ? reinterpret_cast<pn_get_provider_fn>(GetProcAddress(handle, "pn_device_provider"))
                : nullptr;
#else
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        auto entry = handle
                         ? reinterpret_cast<pn_get_provider_fn>(dlsym(handle, "pn_device_provider"))
                         : nullptr;
#endif
        if (!entry)
            throw std::runtime_error("cannot load device plugin or pn_device_provider entry point");
        const auto* api = entry();
        if (!api || api->abi_version != PN_DEVICE_ABI ||
            api->struct_size < sizeof(pn_provider_v1) || !api->open || !api->describe ||
            !api->start || !api->cancel || !api->close)
            throw std::invalid_argument("unsupported device Provider ABI");
        return api;
    }
};
struct Gate
{
    std::atomic_flag& flag;
    bool held;
    explicit Gate(std::atomic_flag& value)
        : flag(value), held(!flag.test_and_set(std::memory_order_acquire))
    {
    }
    ~Gate()
    {
        if (held)
            flag.clear(std::memory_order_release);
    }
};
std::atomic<uint64_t> next_session{0x100000000ULL};
} // namespace

struct DeviceIngress::Impl
{
    std::vector<SignalSchema> signals;
    Mapping mapping;
    QueueHeader* header{};
    uint64_t payload_capacity{};
    Slot* slot(uint64_t position) const noexcept
    {
        return reinterpret_cast<Slot*>(static_cast<std::byte*>(mapping.data) + sizeof(QueueHeader) +
                                       (position % header->capacity) * header->stride);
    }
    static pn_block_v1* blocks(Slot* slot) noexcept
    {
        return reinterpret_cast<pn_block_v1*>(reinterpret_cast<std::byte*>(slot) + sizeof(Slot));
    }
    void* payload(Slot* slot) const noexcept
    {
        return blocks(slot) + signals.size();
    }
};
DeviceIngress::DeviceIngress(const std::string& path, std::vector<SignalSchema> signals,
                             std::size_t capacity, bool create)
    : impl_(std::make_unique<Impl>())
{
    if (capacity < 2 || signals.empty())
        throw std::invalid_argument("device queue needs signals and at least two slots");
    StreamSchema checked(1, signals);
    impl_->signals.assign(checked.signals().begin(), checked.signals().end());
    for (const auto& signal : impl_->signals)
    {
        if (signal.kind != SignalKind::sampled && signal.kind != SignalKind::event)
            throw std::invalid_argument("device signals must be sampled or event");
        if (signal.max_block_bytes > std::numeric_limits<uint64_t>::max() - impl_->payload_capacity)
            throw std::overflow_error("device payload capacity overflow");
        impl_->payload_capacity += signal.max_block_bytes;
    }
    const auto headers = sizeof(Slot) + signals.size() * sizeof(pn_block_v1);
    if (impl_->payload_capacity > std::numeric_limits<std::size_t>::max() - headers - 63)
        throw std::overflow_error("device payload capacity overflow");
    const auto stride = aligned(headers + impl_->payload_capacity);
    if (capacity > (std::numeric_limits<std::size_t>::max() - sizeof(QueueHeader)) / stride)
        throw std::overflow_error("device queue size overflow");
    const auto bytes = sizeof(QueueHeader) + capacity * stride;
    impl_->mapping.open(path, bytes, create);
    impl_->header = static_cast<QueueHeader*>(impl_->mapping.data);
    if (create)
    {
        std::memset(impl_->mapping.data, 0, bytes); // control-plane prefault
        auto* h = new (impl_->mapping.data) QueueHeader;
        h->signature = magic;
        h->capacity = capacity;
        h->stride = stride;
        h->bytes = bytes;
        for (uint64_t i = 0; i < capacity; ++i)
        {
            auto* slot = new (impl_->slot(i)) Slot;
            slot->sequence.store(i, std::memory_order_relaxed);
        }
    }
    else if (impl_->header->signature != magic || impl_->header->capacity != capacity ||
             impl_->header->stride != stride || impl_->header->bytes != bytes)
    {
        throw std::invalid_argument("incompatible device shared memory");
    }
}
DeviceIngress::~DeviceIngress() = default;
void DeviceIngress::remove_directory_on_destroy()
{
    impl_->mapping.remove_directory = true;
}
const std::vector<SignalSchema>& DeviceIngress::signals() const noexcept
{
    return impl_->signals;
}
bool DeviceIngress::cancelled() const noexcept
{
    return impl_->header->cancelled.load() != 0;
}
void DeviceIngress::cancel() noexcept
{
    impl_->header->cancelled.store(1);
}
void DeviceIngress::finish(pn_status status) noexcept
{
    if (status != PN_OK && !(status == PN_STOPPED && cancelled()))
        impl_->header->error.store(static_cast<uint32_t>(status));
    impl_->header->finished.store(1, std::memory_order_release);
}
uint64_t DeviceIngress::published() const noexcept
{
    return impl_->header->published.load();
}
uint64_t DeviceIngress::consumed() const noexcept
{
    return impl_->header->read.load();
}
uint32_t DeviceIngress::error() const noexcept
{
    return impl_->header->error.load();
}
void DeviceIngress::clear()
{
    auto& h = *impl_->header;
    for (uint64_t i = 0; i < h.capacity; ++i)
        impl_->slot(i)->sequence.store(i);
    h.write.store(0);
    h.read.store(0);
    h.published.store(0);
    h.error.store(0);
    h.finished.store(0);
    h.cancelled.store(0);
}
pn_status DeviceIngress::publish(const pn_block_v1& block, const void* data,
                                 uint64_t bytes) noexcept
{
    const Block item{&block, data, bytes};
    return publish_frame({&item, 1});
}
pn_status DeviceIngress::publish_frame(std::span<const Block> blocks) noexcept
{
    auto& h = *impl_->header;
    if (cancelled())
        return PN_STOPPED;
    if (h.error.load() || h.finished.load())
        return PN_FAILED;
    auto invalid = [&]()
    {
        h.error.store(PN_INVALID);
        return PN_INVALID;
    };
    if (blocks.empty() || blocks.size() > impl_->signals.size())
        return invalid();
    uint64_t total_bytes = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        const auto& item = blocks[i];
        if (!item.metadata)
            return invalid();
        const auto& block = *item.metadata;
        const auto bytes = item.bytes;
        const auto* data = item.data;
        if (blocks.size() > 1 && block.kind != PN_DATA)
            return invalid();
        for (std::size_t j = 0; j < i; ++j)
            if (blocks[j].metadata->signal_index == block.signal_index)
                return invalid();
        if (block.struct_size != sizeof(block) || block.signal_index >= impl_->signals.size() ||
            block.kind > PN_RESTART || (block.flags & ~uint32_t{15}) != 0)
            return invalid();
        const auto& signal = impl_->signals[block.signal_index];
        if (block.kind == PN_DATA)
        {
            const auto expected =
                uint64_t{block.samples} * signal.n_channels * signal_dtype_size(signal.dtype);
            if (!block.samples || block.samples > signal.max_block_samples || bytes != expected ||
                !data ||
                ((signal.device_tick_tracking == DeviceTickTracking::sample_counter) &&
                 !(block.flags & PN_HAS_TICK)))
                return invalid();
        }
        else if (bytes != 0 || block.samples != 0)
            return invalid();
        if ((block.flags & PN_HAS_CLOCK_SYNC) &&
            (!(block.flags & PN_HAS_TICK) || !block.tick_rate_numerator ||
             !block.tick_rate_denominator))
            return invalid();
        if (bytes > impl_->payload_capacity - total_bytes)
            return invalid();
        total_bytes += bytes;
    }
    auto position = h.write.load(std::memory_order_relaxed);
    Slot* slot;
    // Bounded retry: concurrent producers cannot impose an unbounded spin on a callback.
    bool reserved = false;
    for (unsigned attempt = 0; attempt < 64; ++attempt)
    {
        slot = impl_->slot(position);
        const auto sequence = slot->sequence.load(std::memory_order_acquire);
        if (sequence == position)
        {
            if (h.write.compare_exchange_weak(position, position + 1, std::memory_order_relaxed))
            {
                reserved = true;
                break;
            }
        }
        else
        {
            if (sequence < position)
                break;
            position = h.write.load(std::memory_order_relaxed);
        }
    }
    if (!reserved)
    {
        h.error.store(PN_FULL);
        return PN_FULL;
    }
    slot->count = blocks.size();
    slot->bytes = total_bytes;
    uint64_t offset = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        Impl::blocks(slot)[i] = *blocks[i].metadata;
        if (blocks[i].bytes)
            std::memcpy(static_cast<std::byte*>(impl_->payload(slot)) + offset, blocks[i].data,
                        static_cast<std::size_t>(blocks[i].bytes));
        offset += blocks[i].bytes;
    }
    h.published.fetch_add(1, std::memory_order_relaxed);
    slot->sequence.store(position + 1, std::memory_order_release);
    return PN_OK;
}

struct GenericDeviceSource::Impl
{
    Library library;
    const pn_provider_v1* api{};
    void* handle{};
    std::string config;
    std::unique_ptr<StreamSchema> schema;
    std::shared_ptr<DeviceIngress> ingress;
    pn_ingress_v1 host{};
    std::vector<uint64_t> sample_positions;
    uint64_t sequence{}, session{next_session.fetch_add(1)};
    std::atomic_flag read_gate = ATOMIC_FLAG_INIT;
    std::mutex control;
    std::atomic<bool> closed{false};
    bool started{};
    std::atomic<bool> stop_monitor{false};
    std::thread cancellation_monitor;
    std::vector<std::string> signal_names;
    std::function<void()> reset_action;
    ~Impl()
    {
        shutdown();
    }
    void shutdown()
    {
        if (ingress)
            ingress->cancel();
        stop_monitor.store(true);
        if (cancellation_monitor.joinable())
            cancellation_monitor.join();
        if (handle)
        {
            api->cancel(handle);
            api->close(handle);
            handle = nullptr;
        }
    }
    std::vector<SignalSchema> describe()
    {
        pn_descriptor_v1 descriptor{};
        descriptor.struct_size = sizeof(descriptor);
        if (api->describe(handle, &descriptor) != PN_OK ||
            descriptor.struct_size != sizeof(descriptor) || !descriptor.signals ||
            !descriptor.signal_count || descriptor.signal_count > 65536)
            throw std::runtime_error("invalid device descriptor");
        std::vector<SignalSchema> signals;
        std::vector<std::string> names_seen;
        for (uint32_t i = 0; i < descriptor.signal_count; ++i)
        {
            const auto& s = descriptor.signals[i];
            if (s.struct_size != sizeof(s) || !s.name || !s.name[0] || s.dtype > PN_FLOAT64 ||
                s.layout > PN_CHANNEL_MAJOR || s.kind > PN_EVENT ||
                s.unit > PN_UNIT_DIMENSIONLESS || s.sample_counter > 1 || !s.clock_domain)
                throw std::invalid_argument("invalid device signal descriptor");
            if (std::find(names_seen.begin(), names_seen.end(), s.name) != names_seen.end())
                throw std::invalid_argument("duplicate device signal name");
            names_seen.emplace_back(s.name);
            std::vector<std::string> names;
            if (s.channel_names)
                for (uint32_t c = 0; c < s.channels; ++c)
                {
                    if (!s.channel_names[c])
                        throw std::invalid_argument("null device channel name");
                    names.emplace_back(s.channel_names[c]);
                }
            signals.emplace_back(
                i + 1, static_cast<SignalDType>(s.dtype), s.channels, s.max_samples, s.max_samples,
                RationalRate{s.rate_numerator, s.rate_denominator}, s.clock_domain,
                static_cast<SignalLayout>(s.layout),
                s.sample_counter ? DeviceTickTracking::sample_counter
                                 : DeviceTickTracking::unavailable,
                static_cast<PhysicalUnit>(s.unit), i + 1, 0, 0, static_cast<SignalKind>(s.kind), 0,
                ObservationTiming::not_applicable, 0, std::move(names));
        }
        if (!signal_names.empty() && signal_names != names_seen)
            throw std::invalid_argument("device signal names changed during reset");
        signal_names = std::move(names_seen);
        return signals;
    }
    void prepare(std::vector<SignalSchema> signals, const std::string& path, std::size_t capacity)
    {
        schema = std::make_unique<StreamSchema>(1, signals);
        ingress = std::make_shared<DeviceIngress>(path, std::move(signals), capacity, true);
        sample_positions.resize(schema->signals().size());
        host = {PN_DEVICE_ABI,
                sizeof(pn_ingress_v1),
                ingress.get(),
                [](void* p, const pn_block_v1* b, const void* d, uint64_t n) -> pn_status
                {
                    if (!b)
                    {
                        static_cast<DeviceIngress*>(p)->finish(PN_INVALID);
                        return PN_INVALID;
                    }
                    return static_cast<DeviceIngress*>(p)->publish(*b, d, n);
                },
                [](void* p) -> uint32_t { return static_cast<DeviceIngress*>(p)->cancelled(); },
                [](void* p, pn_status s) { static_cast<DeviceIngress*>(p)->finish(s); }};
    }
};
GenericDeviceSource::GenericDeviceSource(const std::string& library, const std::string& config,
                                         const std::string& path, std::size_t capacity)
    : impl_(std::make_unique<Impl>())
{
    impl_->api = impl_->library.load(library);
    impl_->config = config;
    if (impl_->api->open(config.c_str(), &impl_->handle) != PN_OK || !impl_->handle)
        throw std::runtime_error("device open failed");
    impl_->prepare(impl_->describe(), path, capacity);
}
GenericDeviceSource::GenericDeviceSource(std::vector<SignalSchema> signals, const std::string& path,
                                         std::size_t capacity)
    : impl_(std::make_unique<Impl>())
{
    impl_->prepare(std::move(signals), path, capacity);
}
GenericDeviceSource::~GenericDeviceSource() = default;
const StreamSchema& GenericDeviceSource::schema() const noexcept
{
    return *impl_->schema;
}
SessionId GenericDeviceSource::session_id() const noexcept
{
    return impl_->session;
}
std::shared_ptr<DeviceIngress> GenericDeviceSource::ingress() const noexcept
{
    return impl_->ingress;
}
void GenericDeviceSource::set_reset_action(std::function<void()> action)
{
    impl_->reset_action = std::move(action);
}
void GenericDeviceSource::start()
{
    std::lock_guard lock(impl_->control);
    if (impl_->closed || impl_->started || impl_->ingress->cancelled())
        throw std::runtime_error("device is closed, cancelled, or already started");
    if (impl_->api && impl_->api->start(impl_->handle, &impl_->host) != PN_OK)
    {
        impl_->ingress->finish(PN_FAILED);
        throw std::runtime_error("device start failed");
    }
    if (impl_->api)
    {
        // Vendor cancellation never runs on the realtime consumer. The handle
        // remains owned until shutdown joins this control-plane monitor.
        impl_->stop_monitor.store(false);
        try
        {
            impl_->cancellation_monitor = std::thread(
                [state = impl_.get()]
                {
                    while (!state->stop_monitor.load())
                    {
                        if (state->ingress->cancelled() || state->ingress->error())
                        {
                            state->api->cancel(state->handle);
                            return;
                        }
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                });
        }
        catch (...)
        {
            impl_->ingress->cancel();
            impl_->api->cancel(impl_->handle);
            throw;
        }
    }
    impl_->started = true;
}
void GenericDeviceSource::cancel() noexcept
{
    impl_->ingress->cancel();
}
void GenericDeviceSource::close()
{
    std::lock_guard lock(impl_->control);
    impl_->closed.store(true);
    impl_->shutdown();
}
StreamStatus GenericDeviceSource::reset() noexcept
{
    Gate gate(impl_->read_gate);
    if (!gate.held)
        return StreamStatus::invalid_state;
    try
    {
        std::lock_guard lock(impl_->control);
        if (impl_->closed)
            return StreamStatus::invalid_state;
        impl_->ingress->cancel();
        if (impl_->api)
        {
            impl_->shutdown();
            if (impl_->api->open(impl_->config.c_str(), &impl_->handle) != PN_OK || !impl_->handle)
                return StreamStatus::source_failure;
            StreamSchema next(1, impl_->describe());
            if (!impl_->schema->equivalent(next))
                return StreamStatus::invalid_state;
            // Channel names also belong to the frozen device description.
            for (std::size_t i = 0; i < next.signals().size(); ++i)
                if (next.signals()[i].channel_names != impl_->schema->signals()[i].channel_names)
                    return StreamStatus::invalid_state;
        }
        else if (impl_->reset_action)
            impl_->reset_action();
        impl_->ingress->clear();
        impl_->sequence = 0;
        std::fill(impl_->sample_positions.begin(), impl_->sample_positions.end(), 0);
        impl_->started = false;
        return StreamStatus::ok;
    }
    catch (...)
    {
        return StreamStatus::source_failure;
    }
}
StreamStatus GenericDeviceSource::read(MutableFrame& frame) noexcept
{
    DiscontinuityLease empty;
    return read_message(frame, empty);
}
StreamStatus GenericDeviceSource::read_message(MutableFrame& frame,
                                               DiscontinuityLease& discontinuity) noexcept
{
    Gate gate(impl_->read_gate);
    if (!gate.held)
        return StreamStatus::invalid_state;
    if (impl_->closed || impl_->ingress->cancelled())
        return StreamStatus::stopped;
    auto& queue = *impl_->ingress->impl_;
    auto& h = *queue.header;
    if (h.error.load())
        return StreamStatus::source_failure;
    const auto position = h.read.load(std::memory_order_relaxed);
    auto* slot = queue.slot(position);
    if (slot->sequence.load(std::memory_order_acquire) != position + 1)
    {
        return h.finished.load(std::memory_order_acquire) && h.write.load() == position
                   ? StreamStatus::end_of_stream
                   : StreamStatus::would_block;
    }
    const auto& b = DeviceIngress::Impl::blocks(slot)[0];
    const auto& signal = impl_->schema->signals()[b.signal_index];
    auto& next = impl_->sample_positions[b.signal_index];
    const auto sample = (b.flags & PN_HAS_SAMPLE_INDEX) ? b.sample_index : next;
    StreamStatus status;
    if (b.kind != PN_DATA)
    {
        if ((b.flags & PN_KNOWN_LOSS) &&
            b.missing_samples > std::numeric_limits<uint64_t>::max() - next)
            return StreamStatus::invalid_frame;
        const auto actual = (b.flags & PN_HAS_SAMPLE_INDEX)
                                ? sample
                                : next + ((b.flags & PN_KNOWN_LOSS) ? b.missing_samples : 0);
        if (b.kind == PN_GAP && (b.flags & PN_KNOWN_LOSS) &&
            (actual < next || actual - next != b.missing_samples))
            return StreamStatus::invalid_frame;
        const auto reason =
            b.kind == PN_RESTART ? GapReason::device_restart : GapReason::source_gap;
        const SignalGap gap{.expected_sample_idx = next,
                            .actual_sample_idx = actual,
                            .missing_samples = b.missing_samples,
                            .signal_id = signal.id,
                            .reason = reason,
                            .flags = (b.flags & PN_KNOWN_LOSS)
                                         ? SignalGapFlags::missing_samples_known
                                         : SignalGapFlags::none};
        status = discontinuity.assign(impl_->session, impl_->sequence ? impl_->sequence - 1 : 0,
                                      impl_->sequence, reason, {&gap, 1});
        if (status == StreamStatus::ok)
        {
            next = actual;
            status = StreamStatus::discontinuity;
        }
    }
    else
    {
        if (frame.block_storage().size() < slot->count ||
            frame.payload_storage().size() < slot->bytes)
            return StreamStatus::invalid_frame;
        frame.header() = {.session_id = impl_->session,
                          .sequence = impl_->sequence,
                          .source_tick = b.device_tick,
                          .schema_id = impl_->schema->id(),
                          .source_clock_domain = signal.clock_domain,
                          .signal_block_count = static_cast<uint32_t>(slot->count),
                          .flags =
                              (b.flags & PN_HAS_TICK) ? FrameFlags::source_tick : FrameFlags::none};
        uint64_t offset = 0;
        for (std::size_t i = 0; i < slot->count; ++i)
        {
            const auto& item = DeviceIngress::Impl::blocks(slot)[i];
            const auto& descriptor = impl_->schema->signals()[item.signal_index];
            const auto start = (item.flags & PN_HAS_SAMPLE_INDEX)
                                   ? item.sample_index
                                   : impl_->sample_positions[item.signal_index];
            if (start > std::numeric_limits<uint64_t>::max() - item.samples)
                return StreamStatus::invalid_frame;
            const uint64_t bytes = uint64_t{item.samples} * descriptor.n_channels *
                                   signal_dtype_size(descriptor.dtype);
            auto& block = frame.block_storage()[i];
            block = {.sample_idx_start = start,
                     .device_tick_start = item.device_tick,
                     .payload_offset = offset,
                     .payload_byte_count = bytes,
                     .signal_id = descriptor.id,
                     .n_samples = item.samples};
            if (item.flags & PN_HAS_CLOCK_SYNC)
                block.clock_sync = {item.tick_reference,
                                    item.host_reference_ns,
                                    {item.tick_rate_numerator, item.tick_rate_denominator},
                                    item.uncertainty_ns,
                                    descriptor.clock_domain,
                                    item.clock_generation,
                                    item.synchronized ? ClockSyncFlags::synchronized
                                                      : ClockSyncFlags::none};
            offset += bytes;
        }
        std::memcpy(frame.payload_storage().data(), queue.payload(slot), slot->bytes);
        status = frame.set_used_sizes(slot->count, slot->bytes);
        if (status == StreamStatus::ok)
        {
            for (std::size_t i = 0; i < slot->count; ++i)
            {
                const auto& item = DeviceIngress::Impl::blocks(slot)[i];
                const auto& block = frame.block_storage()[i];
                impl_->sample_positions[item.signal_index] =
                    block.sample_idx_start + block.n_samples;
            }
            ++impl_->sequence;
        }
    }
    if (status == StreamStatus::ok || status == StreamStatus::discontinuity)
    {
        slot->sequence.store(position + h.capacity, std::memory_order_release);
        h.read.store(position + 1, std::memory_order_release);
    }
    return status;
}
} // namespace neurale::devices
