/* SPDX-License-Identifier: MIT */
#include <atomic>
#include <cstring>
#include <provider.h>

namespace
{
std::atomic<unsigned> opens{0};
struct Handle
{
    pn_signal_v1 signal{sizeof(pn_signal_v1),
                        "test",
                        1,
                        PN_FLOAT64,
                        PN_SAMPLE_MAJOR,
                        PN_SAMPLED,
                        1,
                        PN_UNIT_UNSPECIFIED,
                        0,
                        1,
                        1000,
                        1,
                        nullptr};
    pn_ingress_v1 ingress{};
    bool fail_start{};
};
pn_status open_device(const char* config, void** output)
{
    if (std::strcmp(config, "fail-open") == 0)
        return PN_FAILED;
    try
    {
        auto* h = new Handle;
        h->fail_start = std::strcmp(config, "fail-start") == 0;
        if (std::strcmp(config, "change-schema") == 0)
            h->signal.channels += opens.fetch_add(1);
        *output = h;
        return PN_OK;
    }
    catch (...)
    {
        return PN_FAILED;
    }
}
pn_status describe(void* handle, pn_descriptor_v1* out)
{
    *out = {sizeof(*out), 1, &static_cast<Handle*>(handle)->signal};
    return PN_OK;
}
pn_status start(void* handle, const pn_ingress_v1* ingress)
{
    auto& h = *static_cast<Handle*>(handle);
    h.ingress = *ingress;
    return h.fail_start ? PN_FAILED : PN_OK;
}
void cancel(void* handle)
{
    auto& h = *static_cast<Handle*>(handle);
    // Evidence that the control-plane cancellation reached the vendor adapter.
    if (h.ingress.finish)
        h.ingress.finish(h.ingress.context, PN_FAILED);
}
void close_device(void* handle)
{
    delete static_cast<Handle*>(handle);
}
} // namespace
extern "C" PN_DEVICE_EXPORT const pn_provider_v1* PN_DEVICE_CALL pn_device_provider(void)
{
#ifdef PN_BAD_ABI
    static const pn_provider_v1 api{
        999, sizeof(pn_provider_v1), open_device, describe, start, cancel, close_device};
#else
    static const pn_provider_v1 api{
        PN_DEVICE_ABI, sizeof(pn_provider_v1), open_device, describe, start, cancel, close_device};
#endif
    return &api;
}
