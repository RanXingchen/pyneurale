/* SPDX-License-Identifier: MIT */
#pragma once
#include "provider.h"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <thread>
#include <vector>

namespace neurale_device
{
class CallbackIngress
{
  public:
    explicit CallbackIngress(const pn_ingress_v1& host) : host_(host) {}
    bool cancelled() const noexcept
    {
        return host_.cancelled(host_.context) != 0;
    }
    pn_status publish(const pn_block_v1& block, const void* data, uint64_t bytes) const noexcept
    {
        return host_.publish(host_.context, &block, data, bytes);
    }
    void finish(pn_status status = PN_OK) const noexcept
    {
        host_.finish(host_.context, status);
    }

  private:
    pn_ingress_v1 host_;
};
/* read writes into preallocated storage and reports the used byte count. A blocking
 * vendor read must be interrupted by the plugin's cancel operation. */
class PullWorker
{
  public:
    using Read = std::function<pn_status(void*, uint64_t, pn_block_v1&, uint64_t&)>;
    PullWorker(uint64_t capacity, Read read) : storage_(capacity), read_(std::move(read)) {}
    ~PullWorker()
    {
        cancel();
        join();
    }
    void start(const pn_ingress_v1& host)
    {
        cancelled_.store(false);
        worker_ = std::thread(
            [this, host]
            {
                CallbackIngress ingress(host);
                try
                {
                    while (!cancelled_.load() && !ingress.cancelled())
                    {
                        pn_block_v1 block{};
                        block.struct_size = sizeof(block);
                        uint64_t bytes = 0;
                        auto status = read_(storage_.data(), storage_.size(), block, bytes);
                        if (status == PN_WOULD_BLOCK)
                        {
                            std::this_thread::sleep_for(std::chrono::microseconds(100));
                            continue;
                        }
                        if (status == PN_END)
                        {
                            ingress.finish();
                            return;
                        }
                        if (status != PN_OK)
                        {
                            ingress.finish(status);
                            return;
                        }
                        if (bytes > storage_.size())
                        {
                            ingress.finish(PN_INVALID);
                            return;
                        }
                        if (ingress.publish(block, storage_.data(), bytes) != PN_OK)
                            return;
                    }
                }
                catch (...)
                {
                    ingress.finish(PN_FAILED);
                }
            });
    }
    void cancel() noexcept
    {
        cancelled_.store(true);
    }
    void join()
    {
        if (worker_.joinable())
            worker_.join();
    }

  private:
    std::vector<std::byte> storage_;
    Read read_;
    std::atomic<bool> cancelled_{false};
    std::thread worker_;
};
} // namespace neurale_device
