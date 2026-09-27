/* SPDX-License-Identifier: MIT */
#include "allocation_counter.h"
#include "check_returns.h"
#include <array>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <neurale/devices/provider.h>
#include <neurale/streaming/buffer_pool.h>
#include <provider.h>
#include <thread>
using namespace neurale::devices;
using namespace neurale::streaming;
int main()
{
    const auto path =
        (std::filesystem::temp_directory_path() /
         ("neurale-provider-" +
          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())))
            .string();
    std::vector<SignalSchema> signals{SignalSchema{1, SignalDType::float64, 2, 4, 4, {1000, 1}, 1},
                                      SignalSchema{2,
                                                   SignalDType::int32,
                                                   1,
                                                   1,
                                                   1,
                                                   {0, 1},
                                                   1,
                                                   SignalLayout::sample_major,
                                                   DeviceTickTracking::unavailable,
                                                   PhysicalUnit::dimensionless,
                                                   0,
                                                   0,
                                                   0,
                                                   SignalKind::event}};
    {
        GenericDeviceSource source(signals, path, 1024);
        DeviceIngress producer(path, signals, 1024, false);
        FramePool pool(1, 64, 1);
        FrameLease frame;
        CHECK(pool.try_acquire(frame) == StreamStatus::ok);
        DiscontinuityPool gaps(1, 1);
        DiscontinuityLease gap;
        CHECK(gaps.try_acquire(gap) == StreamStatus::ok);
        pn_block_v1 b{};
        b.struct_size = sizeof(b);
        b.samples = 4;
        std::array<double, 8> data{0, 1, 2, 3, 4, 5, 6, 7};
        const auto before = allocations.load();
        CHECK(producer.publish(b, data.data(), sizeof(data)) == PN_OK);
        CHECK(source.read_message(frame.frame(), gap) == StreamStatus::ok);
        CHECK(allocations.load() == before);
        CHECK(frame.view().blocks[0].n_samples == 4);
        CHECK(std::memcmp(frame.view().payload.data(), data.data(), sizeof(data)) == 0);
        b.samples = 0;
        b.kind = PN_GAP;
        b.flags = PN_KNOWN_LOSS;
        b.missing_samples = 2;
        CHECK(producer.publish(b, nullptr, 0) == PN_OK);
        CHECK(source.read_message(frame.frame(), gap) == StreamStatus::discontinuity);
        CHECK(gap.view().signal_gaps[0].expected_sample_idx == 4);
        CHECK(gap.view().signal_gaps[0].actual_sample_idx == 6);
        CHECK(gap.view().signal_gaps[0].missing_samples == 2);
        b = {};
        b.struct_size = sizeof(b);
        b.samples = 2;
        CHECK(producer.publish(b, data.data(), 32) == PN_OK);
        CHECK(source.read_message(frame.frame(), gap) == StreamStatus::ok);
        CHECK(frame.view().blocks[0].sample_idx_start == 6);
        b.signal_index = 1;
        b.samples = 1;
        int32_t event = 7;
        CHECK(producer.publish(b, &event, sizeof(event)) == PN_OK);
        CHECK(source.read_message(frame.frame(), gap) == StreamStatus::ok);
        CHECK(frame.view().blocks[0].signal_id == 2);
        producer.finish(PN_OK);
        CHECK(source.read_message(frame.frame(), gap) == StreamStatus::end_of_stream);
        source.cancel();
        CHECK(source.read(frame.frame()) == StreamStatus::stopped);
        CHECK(producer.publish(b, &event, sizeof(event)) == PN_STOPPED);
        CHECK(source.reset() == StreamStatus::ok);
        CHECK(source.read(frame.frame()) == StreamStatus::would_block);
        b = {};
        b.struct_size = sizeof(b);
        b.samples = 4;
        // Multiple callback threads, each publishing complete owned blocks.
        std::atomic<int> failures{0};
        auto publish = [&]
        {
            for (int i = 0; i < 100; ++i)
                if (producer.publish(b, data.data(), 64) != PN_OK)
                    ++failures;
        };
        std::thread first(publish), second(publish);
        first.join();
        second.join();
        CHECK(failures == 0);
        for (int i = 0; i < 200; ++i)
            CHECK(source.read(frame.frame()) == StreamStatus::ok);
        CHECK(source.read(frame.frame()) == StreamStatus::would_block);
        CHECK(source.reset() == StreamStatus::ok);
        std::thread concurrent_first(publish), concurrent_second(publish);
        int received = 0;
        bool intact = true;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (received < 200 && std::chrono::steady_clock::now() < deadline)
        {
            const auto status = source.read(frame.frame());
            if (status == StreamStatus::would_block)
                continue;
            if (status != StreamStatus::ok)
            {
                intact = false;
                break;
            }
            intact = intact && std::memcmp(frame.view().payload.data(), data.data(), 64) == 0;
            ++received;
        }
        concurrent_first.join();
        concurrent_second.join();
        CHECK(failures == 0);
        CHECK(intact && received == 200);
        CHECK(source.reset() == StreamStatus::ok);
        for (int i = 0; i < 1024; ++i)
            CHECK(producer.publish(b, data.data(), 64) == PN_OK);
        CHECK(producer.publish(b, data.data(), 64) == PN_FULL);
        CHECK(source.read(frame.frame()) == StreamStatus::source_failure);
        CHECK(source.reset() == StreamStatus::ok);
        b.samples = 5;
        CHECK(producer.publish(b, data.data(), 64) == PN_INVALID);
        CHECK(source.read(frame.frame()) == StreamStatus::source_failure);
        CHECK(source.reset() == StreamStatus::ok);
        b.samples = 4;
        constexpr int iterations = 10000;
        const auto started = std::chrono::steady_clock::now();
        const auto allocation_start = allocations.load();
        for (int i = 0; i < iterations; ++i)
        {
            CHECK(producer.publish(b, data.data(), 64) == PN_OK);
            CHECK(source.read(frame.frame()) == StreamStatus::ok);
        }
        CHECK(allocations.load() == allocation_start);
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
        std::cout << "publish+read mean ns=" << elapsed / iterations << '\n';
        // An entire multi-signal frame occupies one slot and allocates no memory.
        CHECK(source.reset() == StreamStatus::ok);
        FramePool paired_pool(1, 68, 2);
        FrameLease paired;
        CHECK(paired_pool.try_acquire(paired) == StreamStatus::ok);
        pn_block_v1 event_block{};
        event_block.struct_size = sizeof(event_block);
        event_block.signal_index = 1;
        event_block.samples = 1;
        const std::array<DeviceIngress::Block, 2> pair{
            {{&b, data.data(), 64}, {&event_block, &event, 4}}};
        const auto pair_allocations = allocations.load();
        for (int i = 0; i < 100; ++i)
        {
            CHECK(producer.publish_frame(pair) == PN_OK);
            CHECK(source.read(paired.frame()) == StreamStatus::ok);
            CHECK(paired.view().blocks.size() == 2);
            CHECK(paired.view().blocks[0].sample_idx_start == uint64_t(i) * 4);
            CHECK(paired.view().blocks[1].sample_idx_start == uint64_t(i));
            CHECK(paired.view().blocks[1].payload_offset == 64);
            CHECK(std::memcmp(paired.view().payload.data(), data.data(), 64) == 0);
            CHECK(std::memcmp(paired.view().payload.data() + 64, &event, 4) == 0);
        }
        CHECK(allocations.load() == pair_allocations);
        CHECK(producer.published() == 100 && producer.consumed() == 100);
        CHECK(source.reset() == StreamStatus::ok);
        auto publish_pairs = [&]
        {
            for (int i = 0; i < 50; ++i)
                if (producer.publish_frame(pair) != PN_OK)
                    ++failures;
        };
        std::thread pairs_first(publish_pairs), pairs_second(publish_pairs);
        pairs_first.join();
        pairs_second.join();
        CHECK(failures == 0);
        for (int i = 0; i < 100; ++i)
        {
            CHECK(source.read(paired.frame()) == StreamStatus::ok);
            CHECK(paired.view().blocks.size() == 2);
            CHECK(paired.view().blocks[1].sample_idx_start == uint64_t(i));
            CHECK(std::memcmp(paired.view().payload.data(), data.data(), 64) == 0);
        }
        // Reject the whole frame before reserving a slot.
        event_block.samples = 2;
        CHECK(producer.publish_frame(pair) == PN_INVALID);
        CHECK(producer.published() == 100);
        CHECK(source.reset() == StreamStatus::ok);
        event_block.samples = 1;
        for (int i = 0; i < 1024; ++i)
            CHECK(producer.publish_frame(pair) == PN_OK);
        CHECK(producer.publish_frame(pair) == PN_FULL);
        source.close();
        source.close();
        CHECK(source.reset() == StreamStatus::invalid_state);
        CHECK(source.read(frame.frame()) == StreamStatus::stopped);
    }
    CHECK(!std::filesystem::exists(path));
    {
        bool rejected = false;
        try
        {
            GenericDeviceSource bad(PN_BAD_FIXTURE_PATH, "{}", path, 4);
        }
        catch (const std::invalid_argument&)
        {
            rejected = true;
        }
        CHECK(rejected);
        rejected = false;
        try
        {
            GenericDeviceSource bad(PN_FIXTURE_PATH, "fail-open", path, 4);
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        CHECK(rejected);
        GenericDeviceSource bad_start(PN_FIXTURE_PATH, "fail-start", path, 4);
        rejected = false;
        try
        {
            bad_start.start();
        }
        catch (const std::runtime_error&)
        {
            rejected = true;
        }
        CHECK(rejected);
        CHECK(bad_start.ingress()->error() != 0);
    }
    {
        GenericDeviceSource source(PN_FIXTURE_PATH, "{}", path, 4);
        source.start();
        const auto start = std::chrono::steady_clock::now();
        source.cancel();
        while (!source.ingress()->error() &&
               std::chrono::steady_clock::now() - start < std::chrono::seconds(2))
            std::this_thread::yield();
        CHECK(source.ingress()->error() == PN_FAILED);
        std::cout << "vendor cancel handoff ns="
                  << std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::steady_clock::now() - start)
                         .count()
                  << '\n';
        source.close();
    }
    {
        GenericDeviceSource source(PN_FIXTURE_PATH, "change-schema", path, 4);
        CHECK(source.schema().signals()[0].n_channels == 1);
        CHECK(source.reset() == StreamStatus::invalid_state);
        CHECK(source.schema().signals()[0].n_channels == 1);
    }
    CHECK(!std::filesystem::exists(path));
    return 0;
}
