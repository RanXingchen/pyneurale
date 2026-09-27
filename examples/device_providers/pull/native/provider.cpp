#include <cstring>
#include <helpers.hpp>
#include <memory>
using namespace neurale_device;
static const pn_signal_v1 signals[] = {
    {sizeof(pn_signal_v1), "amplifier", 2, PN_FLOAT64, PN_SAMPLE_MAJOR, PN_SAMPLED, 4,
     PN_UNIT_UNSPECIFIED, 0, 1, 1000, 1, nullptr},
    {sizeof(pn_signal_v1), "auxiliary", 1, PN_FLOAT64, PN_SAMPLE_MAJOR, PN_SAMPLED, 2,
     PN_UNIT_UNSPECIFIED, 0, 1, 500, 1, nullptr},
    {sizeof(pn_signal_v1), "trigger", 1, PN_INT32, PN_SAMPLE_MAJOR, PN_EVENT, 1,
     PN_UNIT_DIMENSIONLESS, 0, 1, 0, 1, nullptr}};
static pn_status next(unsigned& step, void* data, uint64_t capacity, pn_block_v1& b,
                      uint64_t& bytes)
{
    if (capacity < 64)
        return PN_INVALID;
    b = {};
    b.struct_size = sizeof(b);
    bytes = 0;
    switch (step++)
    {
    case 0:
    {
        b.samples = 4;
        bytes = 64;
        const double values[] = {0, 1, 2, 3, 4, 5, 6, 7};
        std::memcpy(data, values, bytes);
        break;
    }
    case 1:
        b.kind = PN_GAP;
        b.flags = PN_KNOWN_LOSS;
        b.missing_samples = 2;
        break;
    case 2:
    {
        b.samples = 2;
        bytes = 32;
        const double values[] = {12, 13, 14, 15};
        std::memcpy(data, values, bytes);
        break;
    }
    case 3:
    {
        b.signal_index = 1;
        b.samples = 2;
        bytes = 16;
        const double values[] = {20, 21};
        std::memcpy(data, values, bytes);
        break;
    }
    case 4:
    {
        b.signal_index = 2;
        b.samples = 1;
        bytes = 4;
        const int32_t value = 7;
        std::memcpy(data, &value, bytes);
        break;
    }
    case 5:
        b.kind = PN_RESTART;
        break;
    default:
        return PN_END;
    }
    return PN_OK;
}

struct Device
{
    unsigned step{};
    PullWorker worker{64, [this](void* p, uint64_t n, pn_block_v1& b, uint64_t& bytes)
                      { return next(step, p, n, b, bytes); }};
};
static pn_status start(void* p, const pn_ingress_v1* h)
{
    try
    {
        static_cast<Device*>(p)->worker.start(*h);
        return PN_OK;
    }
    catch (...)
    {
        return PN_FAILED;
    }
}
static void cancel(void* p)
{
    static_cast<Device*>(p)->worker.cancel();
}

static pn_status open_device(const char*, void** out)
{
    try
    {
        *out = new Device;
        return PN_OK;
    }
    catch (...)
    {
        return PN_FAILED;
    }
}
static pn_status describe(void*, pn_descriptor_v1* d)
{
    *d = {sizeof(*d), 3, signals};
    return PN_OK;
}
static void close_device(void* p)
{
    cancel(p);
    auto* d = static_cast<Device*>(p);
    d->worker.join();
    delete d;
}
extern "C" PN_DEVICE_EXPORT const pn_provider_v1* PN_DEVICE_CALL pn_device_provider(void)
{
    static const pn_provider_v1 api = {
        PN_DEVICE_ABI, sizeof(pn_provider_v1), open_device, describe, start, cancel, close_device};
    return &api;
}
