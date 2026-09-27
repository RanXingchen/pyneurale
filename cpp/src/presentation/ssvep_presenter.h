// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once
#include "surface.h"
#include <array>
#include <neurale/experiments/ssvep.h>

namespace neurale::experiment_presentation
{
struct SSVEPLayout
{
    std::array<Point2d, experiments::ssvep::kMaxSSVEPTargets> positions{};
    std::size_t count{};  // Zero selects the automatic grid.
    double target_size{}; // Zero selects the automatic square size.
};
struct SSVEPRenderTarget
{
    experiments::TargetId id{};
    Rect2d bounds{};
    Color fill{}, outline{};
    bool burst{};
    double burst_progress{};
};
struct SSVEPRenderPlan
{
    std::array<SSVEPRenderTarget, experiments::ssvep::kMaxSSVEPTargets> targets{};
    std::size_t count{};
};
// Pure geometry and luminance calculation, shared by tests and the presenter.
[[nodiscard]] SurfaceStatus
build_ssvep_render_plan(const experiments::ssvep::SSVEPConfig& task,
                        const experiments::ssvep::SSVEPSnapshot& snapshot,
                        experiments::ExperimentTimeNs time_ns, SSVEPRenderPlan& plan,
                        const SSVEPLayout& layout = {}) noexcept;

class SSVEPConcretePresenter
{
  public:
    [[nodiscard]] SurfaceStatus open(const experiments::ssvep::SSVEPConfig& task,
                                     const WindowConfig& config, RendererTimeNs renderer_origin,
                                     experiments::ExperimentTimeNs experiment_origin,
                                     const SSVEPLayout& layout = {});
    [[nodiscard]] SurfaceStatus pump_events() noexcept
    {
        return surface_.pump_events();
    }
    [[nodiscard]] SurfaceStatus poll_input(InputEvent& event, bool& available) noexcept
    {
        return surface_.poll_input(event, available);
    }
    [[nodiscard]] SoftwarePresentationTimes
    render(const experiments::ssvep::SSVEPSnapshot& snapshot,
           experiments::ExperimentTimeNs time_ns) noexcept;
    [[nodiscard]] PresentationEnvironment environment() const noexcept
    {
        return surface_.environment();
    }
    void close() noexcept;

  private:
    PresentationSurface surface_{};
    experiments::ssvep::SSVEPConfig task_{};
    SSVEPLayout layout_{};
    bool prepared_{};
    bool has_time_{};
    experiments::ExperimentTimeNs last_time_{};
};
} // namespace neurale::experiment_presentation
