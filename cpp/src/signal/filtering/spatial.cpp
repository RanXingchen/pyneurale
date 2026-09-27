/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/spatial.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "spatial_realtime.h"

namespace neurale::signal
{

// The arithmetic lives once, in detail::CommonReferenceRealtimeProcessor. What
// stays here is what is genuinely batch and has no place on a real-time thread:
// argument checking that throws, the finiteness scan over every sample, and the
// copy that keeps `x` const for a caller that asked for a separate output.
//
// The copy is the price of sharing one loop with a kernel that writes in place.
// It is paid here rather than in the pybind11 binding on purpose: "this
// function does not modify x" is part of the published contract, so the
// obligation to honour it belongs to the function, not to each caller that
// might forget.
//
// Note the two places where this entry point is deliberately stricter or looser
// than the kernel it delegates to. An empty `reference_channels` is an error
// here, while the kernel reads it as "every channel" -- so the check below must
// run before the kernel is constructed. Duplicate positions were previously
// accepted here and silently double-weighted a channel; they are now refused,
// which tightens this path to match what the Python layer already enforced.
//
// Both the range and the uniqueness check are made here rather than left to the
// kernel, which would also reject them. This function already owns its argument
// vocabulary -- a caller sees "reference channel index ..." and not the kernel's
// "common reference channel ..." -- and a refusal that is part of the published
// contract should not be an emergent property of what it happens to delegate to.
void common_reference(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                      std::span<const std::size_t> reference_channels, std::string_view method,
                      std::span<double> output, std::span<double> reference)
{
    if (n_channels == 0 || reference_channels.empty())
    {
        throw std::invalid_argument("common reference requires at least one channel");
    }
    if (x.size() != n_samples * n_channels || output.size() != x.size() ||
        reference.size() != n_samples)
    {
        throw std::invalid_argument("invalid common-reference buffer shape");
    }
    if (method != "mean" && method != "median")
    {
        throw std::invalid_argument("common-reference method must be mean or median");
    }
    for (std::size_t i = 0; i < reference_channels.size(); ++i)
    {
        const auto channel = reference_channels[i];
        if (channel >= n_channels)
        {
            throw std::invalid_argument("reference channel index is out of bounds");
        }
        if (std::find(reference_channels.begin(), reference_channels.begin() + i, channel) !=
            reference_channels.begin() + i)
        {
            throw std::invalid_argument("reference channel indices must be unique");
        }
    }
    if (!std::all_of(x.begin(), x.end(), [](double value) { return std::isfinite(value); }))
    {
        throw std::invalid_argument("signal must contain finite values");
    }

    detail::CommonReferenceRealtimeProcessor processor{
        n_channels, reference_channels,
        method == "median" ? detail::ReferenceStatistic::median : detail::ReferenceStatistic::mean};
    std::copy(x.begin(), x.end(), output.begin());
    processor.process(output, n_samples, reference);
}

} // namespace neurale::signal
