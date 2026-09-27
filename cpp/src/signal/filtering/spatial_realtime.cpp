/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "spatial_realtime.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace neurale::signal::detail
{

CommonReferenceRealtimeProcessor::CommonReferenceRealtimeProcessor(
    std::size_t n_channels, std::span<const std::size_t> reference_channels,
    ReferenceStatistic statistic)
    : n_channels_(n_channels), statistic_(statistic)
{
    if (n_channels_ == 0)
    {
        throw std::invalid_argument("common reference requires at least one channel");
    }
    if (reference_channels.size() > n_channels_)
    {
        throw std::invalid_argument("common reference has more reference channels than channels");
    }
    for (std::size_t i = 0; i < reference_channels.size(); ++i)
    {
        const auto channel = reference_channels[i];
        if (channel >= n_channels_)
        {
            throw std::invalid_argument("common reference channel index is out of bounds");
        }
        if (std::find(reference_channels.begin(), reference_channels.begin() + i, channel) !=
            reference_channels.begin() + i)
        {
            throw std::invalid_argument("common reference channels must be unique");
        }
    }

    all_channels_ = reference_channels.empty() || reference_channels.size() == n_channels_;
    if (!all_channels_)
    {
        reference_.assign(reference_channels.begin(), reference_channels.end());
    }
    // n_reference_channels() rather than reference_.size(): the whole-set case
    // deliberately leaves reference_ empty, and the workspace still needs one
    // slot per contributing channel.
    const auto count = n_reference_channels();
    reference_count_ = static_cast<double>(count);
    middle_ = count / 2;
    even_ = count % 2 == 0;
    if (statistic_ == ReferenceStatistic::median)
    {
        workspace_.resize(count);
    }
}

void CommonReferenceRealtimeProcessor::process(std::span<double> frame,
                                               std::size_t n_samples) noexcept
{
    apply<false>(frame.data(), n_samples, nullptr);
}

void CommonReferenceRealtimeProcessor::process(std::span<double> frame, std::size_t n_samples,
                                               std::span<double> reference) noexcept
{
    apply<true>(frame.data(), n_samples, reference.data());
}

// EmitReference is a template parameter rather than a null check so the
// real-time instantiation does not merely predict the branch but does not
// contain it: apply<false> compiles to what this kernel compiled to before the
// batch caller existed. Only spatial.cpp asks for apply<true>.
template <bool EmitReference>
void CommonReferenceRealtimeProcessor::apply(double* data, std::size_t n_samples,
                                             double* reference) noexcept
{
    if (statistic_ == ReferenceStatistic::median)
    {
        apply_median<EmitReference>(data, n_samples, reference);
        return;
    }
    apply_mean<EmitReference>(data, n_samples, reference);
}

template <bool EmitReference>
void CommonReferenceRealtimeProcessor::apply_mean(double* data, std::size_t n_samples,
                                                  double* reference) noexcept
{
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        double* const row = data + sample * n_channels_;
        double total = 0.0;
        if (all_channels_)
        {
            for (std::size_t channel = 0; channel < n_channels_; ++channel)
            {
                total += row[channel];
            }
        }
        else
        {
            for (const auto channel : reference_)
            {
                total += row[channel];
            }
        }
        const auto value = total / reference_count_;
        if constexpr (EmitReference)
        {
            reference[sample] = value;
        }
        for (std::size_t channel = 0; channel < n_channels_; ++channel)
        {
            row[channel] -= value;
        }
    }
}

// Median cannot select in place: std::nth_element reorders what it is given,
// and the reference still has to be subtracted from the *original* row. Hence
// the gather into workspace_, which is why that buffer exists at all.
//
// This stays inside the real-time contract. std::nth_element is introselect --
// median-of-three quickselect under a 2*log2(n) depth limit, falling back to a
// heap select beyond it -- so it is in-place, allocation-free, and bounded by
// O(n log n) rather than quickselect's quadratic worst case. The even-count
// second pass is exactly middle_ - 1 comparisons, independent of the data.
//
// process() remains noexcept because comparing doubles cannot throw, so neither
// std::nth_element nor std::max_element can throw on this path.
//
// No data-oblivious sorting network here: at 256 channels a Batcher network
// costs roughly 4600 compare-exchanges against introselect's ~600 average, so
// it only wins on variance. Whether that trade is worth taking is a question
// for neurale_pipeline_spatial_reference_adapter_benchmark, not for a guess.
template <bool EmitReference>
void CommonReferenceRealtimeProcessor::apply_median(double* data, std::size_t n_samples,
                                                    double* reference) noexcept
{
    const auto middle = static_cast<std::ptrdiff_t>(middle_);
    for (std::size_t sample = 0; sample < n_samples; ++sample)
    {
        double* const row = data + sample * n_channels_;
        if (all_channels_)
        {
            std::memcpy(workspace_.data(), row, n_channels_ * sizeof(double));
        }
        else
        {
            for (std::size_t i = 0; i < reference_.size(); ++i)
            {
                workspace_[i] = row[reference_[i]];
            }
        }
        std::nth_element(workspace_.begin(), workspace_.begin() + middle, workspace_.end());
        auto value = workspace_[middle_];
        if (even_)
        {
            const auto lower = *std::max_element(workspace_.begin(), workspace_.begin() + middle);
            value = 0.5 * (lower + value);
        }
        if constexpr (EmitReference)
        {
            reference[sample] = value;
        }
        for (std::size_t channel = 0; channel < n_channels_; ++channel)
        {
            row[channel] -= value;
        }
    }
}

// Stateless: nothing carries across blocks, so a discontinuity costs nothing to
// absorb. Kept so the adapter's reset path reads like every other kernel's.
void CommonReferenceRealtimeProcessor::reset() noexcept {}

std::size_t CommonReferenceRealtimeProcessor::n_channels() const noexcept
{
    return n_channels_;
}

std::size_t CommonReferenceRealtimeProcessor::n_reference_channels() const noexcept
{
    return all_channels_ ? n_channels_ : reference_.size();
}

std::string_view CommonReferenceRealtimeProcessor::kernel_name() const noexcept
{
    return "scalar";
}

std::string_view CommonReferenceRealtimeProcessor::statistic_name() const noexcept
{
    return statistic_ == ReferenceStatistic::median ? "median" : "mean";
}

} // namespace neurale::signal::detail
