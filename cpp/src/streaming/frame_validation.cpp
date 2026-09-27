/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/frame_validation.h>

#include <algorithm>
#include <limits>

namespace neurale::streaming
{
namespace
{

[[nodiscard]] bool checked_add(std::uint64_t left, std::uint64_t right,
                               std::uint64_t& result) noexcept
{
    if (right > std::numeric_limits<std::uint64_t>::max() - left)
    {
        return false;
    }
    result = left + right;
    return true;
}

[[nodiscard]] bool checked_multiply(std::uint64_t left, std::uint64_t right,
                                    std::uint64_t& result) noexcept
{
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    {
        return false;
    }
    result = left * right;
    return true;
}

} // namespace

struct FrameValidator::SignalDescriptor
{
    SignalId id{};
    ClockDomainId clock_domain{};
    std::uint32_t n_channels{};
    std::uint32_t max_block_samples{};
    std::size_t scalar_size{};
    SignalKind kind{SignalKind::sampled};
    std::uint64_t observation_shift_ns{};
    std::uint64_t fixed_block_bytes{};
    std::uint64_t validation_epoch{};
};

FrameValidator::FrameValidator(const StreamSchema& schema)
    : schema_id_(schema.id()), n_signals_(schema.signals().size()),
      signals_(std::make_unique<SignalDescriptor[]>(n_signals_))
{
    for (std::size_t i = 0; i < n_signals_; ++i)
    {
        const auto& signal = schema.signals()[i];
        const auto* feature_set = signal.kind == SignalKind::feature
                                      ? schema.feature_sets().find(signal.feature_set_id)
                                      : nullptr;
        signals_[i] = SignalDescriptor{
            .id = signal.id,
            .clock_domain = signal.clock_domain,
            .n_channels = signal.n_channels,
            .max_block_samples = signal.max_block_samples,
            .scalar_size = signal_dtype_size(signal.dtype),
            .kind = signal.kind,
            .observation_shift_ns = feature_set != nullptr ? feature_set->shift_ns : 0,
            .fixed_block_bytes = signal.fixed_block_bytes,
        };
    }
    std::sort(signals_.get(), signals_.get() + n_signals_,
              [](const SignalDescriptor& left, const SignalDescriptor& right)
              { return left.id < right.id; });
}

FrameValidator::~FrameValidator() = default;

FrameValidator::SignalDescriptor* FrameValidator::find_signal(SignalId id) noexcept
{
    std::size_t first = 0;
    std::size_t last = n_signals_;
    while (first < last)
    {
        const auto middle = first + (last - first) / 2;
        if (signals_[middle].id < id)
        {
            first = middle + 1;
        }
        else
        {
            last = middle;
        }
    }
    return first < n_signals_ && signals_[first].id == id ? &signals_[first] : nullptr;
}

FrameValidationError FrameValidator::validate(FrameView frame) noexcept
{
    if (frame.header.schema_id != schema_id_)
    {
        return FrameValidationError::schema_changed;
    }
    if (frame.header.signal_block_count != frame.blocks.size() || frame.blocks.empty() ||
        frame.blocks.size() > n_signals_)
    {
        return FrameValidationError::block_count_mismatch;
    }

    ++validation_epoch_;
    if (validation_epoch_ == 0)
    {
        for (std::size_t i = 0; i < n_signals_; ++i)
        {
            signals_[i].validation_epoch = 0;
        }
        validation_epoch_ = 1;
    }

    std::uint64_t payload_end = 0;
    for (const auto& block : frame.blocks)
    {
        auto* signal = find_signal(block.signal_id);
        if (signal == nullptr)
        {
            return FrameValidationError::unknown_signal;
        }
        if (signal->validation_epoch == validation_epoch_)
        {
            return FrameValidationError::duplicate_signal;
        }
        signal->validation_epoch = validation_epoch_;
        if ((block.n_samples == 0 && signal->kind != SignalKind::spike) ||
            block.n_samples > signal->max_block_samples)
        {
            return FrameValidationError::invalid_sample_count;
        }
        if (signal->kind == SignalKind::spike &&
            ((block.n_samples == 0 && block.last_sample_idx != 0) ||
             (block.n_samples != 0 && block.last_sample_idx < block.sample_idx_start)))
        {
            return FrameValidationError::invalid_sparse_idx_range;
        }
        if (signal->kind == SignalKind::feature)
        {
            std::uint64_t observation_span{};
            std::uint64_t observation_end{};
            if (!checked_multiply(block.n_samples - 1, signal->observation_shift_ns,
                                  observation_span) ||
                !checked_add(block.observation_time_start_ns, observation_span, observation_end))
            {
                return FrameValidationError::invalid_feature_timing;
            }
        }
        if (has_flag(block.clock_sync.flags, ClockSyncFlags::synchronized) &&
            (block.clock_sync.clock_domain != signal->clock_domain ||
             block.clock_sync.generation == 0 || block.clock_sync.device_tick_rate.numerator == 0 ||
             block.clock_sync.device_tick_rate.denominator == 0))
        {
            return FrameValidationError::invalid_clock_sync;
        }

        std::uint64_t sample_end{};
        if (signal->kind != SignalKind::spike &&
            !checked_add(block.sample_idx_start, block.n_samples, sample_end))
        {
            return FrameValidationError::sample_idx_overflow;
        }
        std::uint64_t n_scalars{};
        std::uint64_t expected_bytes{};
        if ((signal->kind == SignalKind::spike
                 ? (expected_bytes = signal->fixed_block_bytes, false)
                 : (!checked_multiply(signal->n_channels, block.n_samples, n_scalars) ||
                    !checked_multiply(n_scalars, signal->scalar_size, expected_bytes))) ||
            block.payload_byte_count != expected_bytes)
        {
            return signal->kind == SignalKind::feature
                       ? FrameValidationError::feature_payload_shape_mismatch
                       : FrameValidationError::payload_size_mismatch;
        }
        if (block.payload_offset != payload_end ||
            !checked_add(block.payload_offset, block.payload_byte_count, payload_end) ||
            payload_end > frame.payload.size())
        {
            return FrameValidationError::payload_layout_invalid;
        }
    }
    return payload_end == frame.payload.size() ? FrameValidationError::none
                                               : FrameValidationError::payload_layout_invalid;
}

} // namespace neurale::streaming
