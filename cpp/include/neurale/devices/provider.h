/* SPDX-License-Identifier: MIT */
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <neurale/streaming/source.h>
#include <span>
#include <string>
#include <vector>

struct pn_block_v1;

namespace neurale::devices
{
class DeviceIngress
{
  public:
    DeviceIngress(const std::string& path, std::vector<streaming::SignalSchema> signals,
                  std::size_t capacity, bool create);
    ~DeviceIngress();
    DeviceIngress(const DeviceIngress&) = delete;
    DeviceIngress& operator=(const DeviceIngress&) = delete;
    struct Block
    {
        const pn_block_v1* metadata;
        const void* data;
        uint64_t bytes;
    };
    int32_t publish(const pn_block_v1&, const void*, uint64_t) noexcept;
    int32_t publish_frame(std::span<const Block>) noexcept;
    bool cancelled() const noexcept;
    void cancel() noexcept;
    void finish(int32_t) noexcept;
    uint64_t published() const noexcept;
    uint64_t consumed() const noexcept;
    uint32_t error() const noexcept;
    void clear(); // all producers and consumer must be quiescent
    void remove_directory_on_destroy();
    const std::vector<streaming::SignalSchema>& signals() const noexcept;

  private:
    friend class GenericDeviceSource;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
class GenericDeviceSource final : public streaming::NativeFrameSource
{
  public:
    GenericDeviceSource(const std::string& library, const std::string& config,
                        const std::string& path, std::size_t capacity);
    GenericDeviceSource(std::vector<streaming::SignalSchema> signals, const std::string& path,
                        std::size_t capacity);
    ~GenericDeviceSource() override;
    streaming::StreamStatus read(streaming::MutableFrame&) noexcept override;
    streaming::StreamStatus read_message(streaming::MutableFrame&,
                                         streaming::DiscontinuityLease&) noexcept override;
    bool produces_discontinuities() const noexcept override
    {
        return true;
    }
    void cancel() noexcept override;
    streaming::StreamStatus reset() noexcept override;
    void start();
    void close();
    const streaming::StreamSchema& schema() const noexcept;
    streaming::SessionId session_id() const noexcept;
    std::shared_ptr<DeviceIngress> ingress() const noexcept;
    void set_reset_action(std::function<void()> action);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace neurale::devices
