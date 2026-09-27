/* SPDX-License-Identifier: MIT */
#ifndef NEURALE_DEVICE_PROVIDER_H
#define NEURALE_DEVICE_PROVIDER_H
#include <stdint.h>
#ifdef _WIN32
#define PN_DEVICE_EXPORT __declspec(dllexport)
#define PN_DEVICE_CALL __cdecl
#else
#define PN_DEVICE_EXPORT __attribute__((visibility("default")))
#define PN_DEVICE_CALL
#endif
#ifdef __cplusplus
extern "C"
{
#endif
/* ABI 1: native-endian, same architecture as the host. All pointers are borrowed.
 * describe storage lives until close. publish copies before returning. No exceptions
 * may cross this boundary. start is nonblocking; cancel is thread-safe/nonblocking;
 * close cancels and joins all callbacks/workers before freeing the handle.
 */
#define PN_DEVICE_ABI 1u
#define PN_OK 0
#define PN_STOPPED 1
#define PN_FULL 2
#define PN_INVALID 3
#define PN_FAILED 4
#define PN_WOULD_BLOCK 5
#define PN_END 6
#define PN_INT16 0u
#define PN_INT32 1u
#define PN_FLOAT32 2u
#define PN_FLOAT64 3u
#define PN_SAMPLED 0u
#define PN_EVENT 1u
#define PN_SAMPLE_MAJOR 0u
#define PN_CHANNEL_MAJOR 1u
#define PN_UNIT_UNSPECIFIED 0u
#define PN_UNIT_VOLTS 1u
#define PN_UNIT_AMPERES 2u
#define PN_UNIT_DIMENSIONLESS 3u
#define PN_DATA 0u
#define PN_GAP 1u
#define PN_RESTART 2u
#define PN_HAS_TICK 1u
#define PN_HAS_SAMPLE_INDEX 2u
#define PN_KNOWN_LOSS 4u
#define PN_HAS_CLOCK_SYNC 8u
    typedef int32_t pn_status;
    typedef struct pn_signal_v1
    {
        uint32_t struct_size;
        const char* name;
        uint32_t channels, dtype, layout, kind;
        uint32_t max_samples, unit, sample_counter, clock_domain;
        uint64_t rate_numerator, rate_denominator;
        const char* const* channel_names; /* NULL, or channels strings */
    } pn_signal_v1;
    typedef struct pn_descriptor_v1
    {
        uint32_t struct_size, signal_count;
        const pn_signal_v1* signals;
    } pn_descriptor_v1;
    typedef struct pn_block_v1
    {
        uint32_t struct_size, signal_index, kind, flags;
        uint32_t samples, reserved;
        uint64_t sample_index, device_tick, missing_samples;
        /* Optional measured mapping, distinct from host receive time. */
        uint64_t tick_reference, host_reference_ns, tick_rate_numerator, tick_rate_denominator;
        uint64_t uncertainty_ns;
        uint32_t clock_generation, synchronized;
    } pn_block_v1;
    typedef struct pn_ingress_v1
    {
        uint32_t abi_version, struct_size;
        void* context;
        pn_status(PN_DEVICE_CALL* publish)(void*, const pn_block_v1*, const void*, uint64_t);
        uint32_t(PN_DEVICE_CALL* cancelled)(void*);
        void(PN_DEVICE_CALL* finish)(void*, pn_status); /* after all publishers have returned */
    } pn_ingress_v1;
    typedef struct pn_provider_v1
    {
        uint32_t abi_version, struct_size;
        pn_status(PN_DEVICE_CALL* open)(const char* config_json, void** handle);
        pn_status(PN_DEVICE_CALL* describe)(void*, pn_descriptor_v1*);
        pn_status(PN_DEVICE_CALL* start)(void*, const pn_ingress_v1*);
        void(PN_DEVICE_CALL* cancel)(void*);
        void(PN_DEVICE_CALL* close)(void*);
    } pn_provider_v1;
    typedef const pn_provider_v1*(PN_DEVICE_CALL* pn_get_provider_fn)(void);
    /* Each plugin exports exactly this entry point. */
    PN_DEVICE_EXPORT const pn_provider_v1* PN_DEVICE_CALL pn_device_provider(void);
#ifdef __cplusplus
}
#endif
#endif
