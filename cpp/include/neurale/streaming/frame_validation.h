/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <neurale/streaming/frame.h>
#include <neurale/streaming/schema.h>

namespace neurale::streaming
{

enum class FrameValidationError : std::uint8_t
{
    none,
    schema_changed,
    block_count_mismatch,
    unknown_signal,
    duplicate_signal,
    invalid_sample_count,
    invalid_feature_timing,
    sample_idx_overflow,
    invalid_clock_sync,
    payload_size_mismatch,
    feature_payload_shape_mismatch,
    payload_layout_invalid,
    invalid_sparse_idx_range,
};

/// Allocation-free frame/schema validation prepared before realtime startup.
class FrameValidator
{
  public:
    explicit FrameValidator(const StreamSchema& schema);
    ~FrameValidator();

    FrameValidator(const FrameValidator&) = delete;
    FrameValidator& operator=(const FrameValidator&) = delete;
    FrameValidator(FrameValidator&&) = delete;
    FrameValidator& operator=(FrameValidator&&) = delete;

    [[nodiscard]] FrameValidationError validate(FrameView frame) noexcept;

  private:
    struct SignalDescriptor;

    [[nodiscard]] SignalDescriptor* find_signal(SignalId id) noexcept;

    SchemaId schema_id_{};
    std::size_t n_signals_{};
    std::unique_ptr<SignalDescriptor[]> signals_;
    std::uint64_t validation_epoch_{};
};

} // namespace neurale::streaming
