// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "time_mapper.h"

#include <chrono>
#include <limits>

namespace neurale::experiment_presentation
{

void RendererTimeMapper::configure(RendererTimeNs renderer_origin_ns,
                                   experiments::ExperimentTimeNs experiment_origin_ns) noexcept
{
    renderer_origin_ns_ = renderer_origin_ns;
    last_renderer_ns_ = renderer_origin_ns;
    experiment_origin_ns_ = experiment_origin_ns;
    configured_ = true;
    mapped_ = false;
}

void RendererTimeMapper::reset() noexcept
{
    renderer_origin_ns_ = 0;
    last_renderer_ns_ = 0;
    experiment_origin_ns_ = 0;
    configured_ = false;
    mapped_ = false;
}

SurfaceStatus RendererTimeMapper::map(RendererTimeNs renderer_ns,
                                      experiments::ExperimentTimeNs& experiment_ns) noexcept
{
    if (!configured_)
    {
        return SurfaceStatus::invalid_state;
    }
    if (renderer_ns < renderer_origin_ns_ || (mapped_ && renderer_ns < last_renderer_ns_))
    {
        return SurfaceStatus::time_regressed;
    }
    const auto delta = renderer_ns - renderer_origin_ns_;
    if (delta > (std::numeric_limits<experiments::ExperimentTimeNs>::max)() - experiment_origin_ns_)
    {
        return SurfaceStatus::time_overflow;
    }
    experiment_ns = experiment_origin_ns_ + delta;
    last_renderer_ns_ = renderer_ns;
    mapped_ = true;
    return SurfaceStatus::ok;
}

RendererTimeNs renderer_monotonic_now_ns() noexcept
{
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<RendererTimeNs>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

} // namespace neurale::experiment_presentation
