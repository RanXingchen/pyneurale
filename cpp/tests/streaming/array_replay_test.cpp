/* SPDX-License-Identifier: MIT */
#include "allocation_counter.h"
#include "check_returns.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <neurale/streaming/array_replay.h>
#include <neurale/streaming/buffer_pool.h>
#include <thread>

using namespace neurale::streaming;

int main()
{
    const std::array signals{SignalSchema{1,
                                          SignalDType::float64,
                                          2,
                                          10,
                                          10,
                                          {1000, 1},
                                          1,
                                          SignalLayout::sample_major,
                                          DeviceTickTracking::sample_counter}};
    StreamSchema schema{1, signals};
    std::array<double, 40> data{};
    for (std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<double>(i);
    ArrayReplaySource source{schema, data, false};
    NativeResultSink sink{schema, 20};
    FramePool pool{1, 160, 1};
    FrameLease lease;
    CHECK(pool.try_acquire(lease) == StreamStatus::ok);
    const auto before = allocations.load();
    for (std::size_t i = 0; i < 2; ++i)
    {
        CHECK(source.read(lease.frame()) == StreamStatus::ok);
        CHECK(lease.view().blocks[0].sample_idx_start == i * 10);
        CHECK(sink.consume(lease.view()) == StreamStatus::ok);
    }
    CHECK(source.read(lease.frame()) == StreamStatus::end_of_stream);
    CHECK(sink.consume(lease.view()) == StreamStatus::output_limit);
    CHECK(allocations.load() == before);
    CHECK(sink.timings().size() == 20);
    CHECK(std::equal(data.begin(), data.end(), sink.values(20).begin()));
    CHECK(source.timings()[1].planned_ns - source.timings()[0].planned_ns == 10000000);
    CHECK(source.reset() == StreamStatus::ok);
    CHECK(sink.reset() == StreamStatus::ok);
    CHECK(source.timings().empty());
    CHECK(sink.timings().empty());
    source.cancel();
    CHECK(source.read(lease.frame()) == StreamStatus::stopped);
    CHECK(source.reset() == StreamStatus::ok);
    CHECK(source.read(lease.frame()) == StreamStatus::ok);
    lease.frame().header().schema_id = 999;
    CHECK(sink.consume(lease.view()) == StreamStatus::invalid_frame);

    auto slow_signal = signals;
    slow_signal[0].fs = {1, 1};
    StreamSchema slow_schema{2, slow_signal};
    ArrayReplaySource slow{slow_schema, data};
    StreamStatus result = StreamStatus::ok;
    std::thread worker([&] { result = slow.read(lease.frame()); });
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
    const auto start = std::chrono::steady_clock::now();
    slow.cancel();
    worker.join();
    CHECK(result == StreamStatus::stopped);
    CHECK(std::chrono::steady_clock::now() - start < std::chrono::seconds{1});
    return 0;
}
