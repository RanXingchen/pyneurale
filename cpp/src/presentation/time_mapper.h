// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <neurale/experiments/contract.h>

#include "surface_types.h"

namespace neurale::experiment_presentation
{

/// Renderer instant -> experiment instant, as a single fixed offset.
///
/// The mapping applies one offset for the whole run and never re-anchors or
/// rate-corrects. That is exact -- not an approximation -- but only under one
/// precondition, which is part of this contract rather than an implementation
/// accident:
///
/// **The renderer origin and the experiment origin must come from the same
/// monotonic clock.** In this repository both do: renderer_monotonic_now_ns()
/// reads std::chrono::steady_clock, and so does the streaming host clock that
/// experiment time is anchored to (cpp/src/streaming/clock.cpp). Because the
/// two are one clock, `experiment_ns = experiment_origin_ns + (renderer_ns -
/// renderer_origin_ns)` cannot drift.
///
/// Supplying a renderer origin sampled from a *different* epoch -- a wall
/// clock, a device clock, a raw performance counter -- yields well-formed but
/// wrong instants, offset by the difference between the two epochs, with
/// nothing in the recorded evidence to reveal it. PresentationSurface::open()
/// therefore rejects a renderer origin that lies in the future of
/// renderer_monotonic_now_ns(), which no origin sampled from that clock can.
class RendererTimeMapper
{
  public:
    /// @param renderer_origin_ns Instant from renderer_monotonic_now_ns(), or
    ///        from the same std::chrono::steady_clock epoch. See the class
    ///        precondition above; it is not re-derivable from a mapped value.
    /// @param experiment_origin_ns The experiment instant that names the same
    ///        moment as @p renderer_origin_ns.
    void configure(RendererTimeNs renderer_origin_ns,
                   experiments::ExperimentTimeNs experiment_origin_ns) noexcept;
    void reset() noexcept;
    [[nodiscard]] SurfaceStatus map(RendererTimeNs renderer_ns,
                                    experiments::ExperimentTimeNs& experiment_ns) noexcept;
    [[nodiscard]] bool configured() const noexcept
    {
        return configured_;
    }

  private:
    RendererTimeNs renderer_origin_ns_{};
    RendererTimeNs last_renderer_ns_{};
    experiments::ExperimentTimeNs experiment_origin_ns_{};
    bool configured_{};
    bool mapped_{};
};

[[nodiscard]] RendererTimeNs renderer_monotonic_now_ns() noexcept;

} // namespace neurale::experiment_presentation
