// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#include "ssvep_presenter.h"
#include <algorithm>
#include <cmath>
#include <numbers>

namespace neurale::experiment_presentation
{
namespace ss = experiments::ssvep;
namespace
{
constexpr std::size_t kCircleSegments = 96;
constexpr std::size_t kBurstParticles = 8;
constexpr experiments::DurationNs kBurstDurationNs = 400'000'000;

SurfaceStatus resolve_layout(std::size_t count, const SSVEPLayout& layout,
                             std::array<Rect2d, ss::kMaxSSVEPTargets>& bounds) noexcept
{
    if (count < 2 || count > ss::kMaxSSVEPTargets || (layout.count != 0 && layout.count != count) ||
        !std::isfinite(layout.target_size) || layout.target_size < 0)
        return SurfaceStatus::invalid_configuration;
    const auto columns = static_cast<std::size_t>(std::ceil(std::sqrt(count)));
    const auto rows = (count + columns - 1) / columns;
    const double cell_w = 1.8 / columns, cell_h = 1.6 / rows;
    const double size =
        layout.target_size > 0 ? layout.target_size : std::min(cell_w, cell_h) * 0.55;
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto center = layout.count ? layout.positions[i]
                                         : Point2d{-0.9 + (i % columns + 0.5) * cell_w,
                                                   0.8 - (i / columns + 0.5) * cell_h};
        if (!std::isfinite(center.x) || !std::isfinite(center.y) ||
            std::abs(center.x) + size / 2 + 0.012 > 1 || std::abs(center.y) + size / 2 + 0.012 > 1)
            return SurfaceStatus::invalid_configuration;
        bounds[i] = {center.x - size / 2, center.y - size / 2, size, size};
        for (std::size_t j = 0; j < i; ++j)
            if (std::abs(bounds[i].left - bounds[j].left) < size + 0.024 &&
                std::abs(bounds[i].bottom - bounds[j].bottom) < size + 0.024)
                return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}
} // namespace
SurfaceStatus build_ssvep_render_plan(const ss::SSVEPConfig& task,
                                      const ss::SSVEPSnapshot& snapshot,
                                      experiments::ExperimentTimeNs time_ns, SSVEPRenderPlan& plan,
                                      const SSVEPLayout& layout) noexcept
{
    if (ss::validate(task) != experiments::ContractStatus::ok)
        return SurfaceStatus::invalid_configuration;
    if (time_ns < snapshot.time_ns)
        return SurfaceStatus::time_regressed;
    std::array<Rect2d, ss::kMaxSSVEPTargets> bounds{};
    if (resolve_layout(task.n_targets, layout, bounds) != SurfaceStatus::ok)
        return SurfaceStatus::invalid_configuration;
    SSVEPRenderPlan candidate{};
    switch (snapshot.state)
    {
    case ss::SSVEPState::idle:
    case ss::SSVEPState::complete:
    case ss::SSVEPState::await_decision:
    case ss::SSVEPState::inter_trial:
        plan = candidate;
        return SurfaceStatus::ok;
    case ss::SSVEPState::cue:
    case ss::SSVEPState::stimulation:
    case ss::SSVEPState::feedback:
        break;
    default:
        return SurfaceStatus::invalid_configuration;
    }
    if (!ss::contains_target(task, snapshot.trial.target_id))
        return SurfaceStatus::invalid_configuration;
    // Never draw a stale cue or continue flickering beyond the scheduled phase.
    if (time_ns < snapshot.active.start_ns || time_ns >= snapshot.active.end_ns)
        return SurfaceStatus::presentation_deadline_missed;
    for (std::size_t i = 0; i < task.n_targets; ++i)
    {
        auto& target = candidate.targets[candidate.count++];
        target.id = task.targets[i].id;
        target.bounds = bounds[i];
        target.fill = {0.07F, 0.11F, 0.14F, 1};
        target.outline = {0.22F, 0.34F, 0.37F, 1};
        if (snapshot.state == ss::SSVEPState::stimulation)
        {
            if (time_ns < snapshot.stimulation.start_ns)
                return SurfaceStatus::time_regressed;
            const double elapsed =
                static_cast<double>(time_ns - snapshot.stimulation.start_ns) / 1e9;
            const auto luminance = static_cast<float>(
                0.5 + 0.5 * std::cos(2 * std::numbers::pi *
                                     std::fmod(task.targets[i].frequency_hz * elapsed, 1.0)));
            target.fill = {luminance, luminance, luminance, 1};
            target.outline = target.fill;
        }
        else if (snapshot.state == ss::SSVEPState::cue && target.id == snapshot.trial.target_id)
        {
            target.outline = {1, 0.72F, 0.22F, 1};
        }
        else if (snapshot.state == ss::SSVEPState::feedback)
        {
            if (snapshot.has_selection && target.id == snapshot.selection.selected_id)
            {
                if (snapshot.outcome == experiments::TrialOutcome::success)
                {
                    target.fill = {0.12F, 0.83F, 0.67F, 1};
                    target.outline = {0.47F, 1, 0.79F, 1};
                    target.burst = true;
                    target.burst_progress =
                        std::min(1.0, static_cast<double>(time_ns - snapshot.active.start_ns) /
                                          static_cast<double>(kBurstDurationNs));
                }
                else
                {
                    target.fill = {0.62F, 0.08F, 0.12F, 1};
                    target.outline = {0.98F, 0.31F, 0.32F, 1};
                }
            }
            else if (target.id == snapshot.trial.target_id)
                target.outline = {1, 0.72F, 0.22F, 1};
        }
    }
    plan = candidate;
    return SurfaceStatus::ok;
}
SurfaceStatus SSVEPConcretePresenter::open(const ss::SSVEPConfig& task, const WindowConfig& config,
                                           RendererTimeNs renderer_origin,
                                           experiments::ExperimentTimeNs experiment_origin,
                                           const SSVEPLayout& layout)
{
    if (prepared_ || surface_.lifecycle() != SurfaceLifecycle::closed)
        return SurfaceStatus::invalid_state;
    if (ss::validate(task) != experiments::ContractStatus::ok || config.swap_interval != 1)
        return SurfaceStatus::invalid_configuration;
    std::array<Rect2d, ss::kMaxSSVEPTargets> bounds{};
    if (resolve_layout(task.n_targets, layout, bounds) != SurfaceStatus::ok)
        return SurfaceStatus::invalid_configuration;
    auto status = surface_.open(config, renderer_origin, experiment_origin);
    if (status != SurfaceStatus::ok)
        return status;
    const int refresh = surface_.environment().refresh_rate_hz;
    for (std::size_t i = 0; i < task.n_targets; ++i)
    {
        if (refresh <= 0 || task.targets[i].frequency_hz >= refresh * 0.5)
        {
            surface_.close();
            return SurfaceStatus::invalid_configuration;
        }
    }
    status = surface_.prepare(
        {.max_vertices = (task.n_targets * 2U + kBurstParticles + 2U) * kCircleSegments * 3U,
         .max_draw_batches = task.n_targets * 2U + kBurstParticles + 3U,
         .circle_segments = kCircleSegments},
        {});
    if (status != SurfaceStatus::ok)
    {
        surface_.close();
        return status;
    }
    task_ = task;
    layout_ = layout;
    prepared_ = true;
    has_time_ = false;
    return SurfaceStatus::ok;
}
SoftwarePresentationTimes
SSVEPConcretePresenter::render(const ss::SSVEPSnapshot& snapshot,
                               experiments::ExperimentTimeNs time_ns) noexcept
{
    auto failure = [&](SurfaceStatus status)
    {
        return SoftwarePresentationTimes{
            .status = status, .requested_ns = time_ns, .intended_ns = time_ns};
    };
    if (!prepared_)
        return failure(SurfaceStatus::invalid_state);
    if (has_time_ && time_ns < last_time_)
        return failure(SurfaceStatus::time_regressed);
    SSVEPRenderPlan plan{};
    auto status = build_ssvep_render_plan(task_, snapshot, time_ns, plan, layout_);
    if (status != SurfaceStatus::ok)
        return failure(status);
    status = surface_.begin_frame({0.025F, 0.045F, 0.055F, 1});
    if (status != SurfaceStatus::ok)
        return failure(status);
    for (std::size_t i = 0; i < plan.count; ++i)
    {
        const auto& target = plan.targets[i];
        const auto& b = target.bounds;
        const Point2d center{b.left + b.width / 2, b.bottom + b.height / 2};
        const double radius = b.width / 2;
        const double scale = target.burst ? std::max(0.0, 1.0 - target.burst_progress) : 1.0;
        if (scale > 0.01)
        {
            status = surface_.circle(center, radius * scale, target.outline, true);
            if (status == SurfaceStatus::ok)
                status = surface_.circle(center, radius * 0.87 * scale, target.fill, true);
        }
        if (status != SurfaceStatus::ok)
            return failure(status);
        if (target.burst && target.burst_progress < 1.0)
        {
            const double progress = target.burst_progress;
            const float alpha = static_cast<float>(std::sqrt(1.0 - progress));
            const double ring_radius = radius * (1.0 + 1.15 * progress);
            for (int ring = -1; status == SurfaceStatus::ok && ring <= 1; ++ring)
                status = surface_.circle(center, ring_radius + ring * 0.006,
                                         {0.47F, 1, 0.79F, alpha}, false);
            for (std::size_t particle = 0;
                 status == SurfaceStatus::ok && particle < kBurstParticles; ++particle)
            {
                const double angle = 2 * std::numbers::pi * particle / kBurstParticles;
                const double distance = radius * (0.75 + 1.35 * progress);
                const Point2d position{center.x + distance * std::cos(angle),
                                       center.y + distance * std::sin(angle)};
                status = surface_.circle(position, radius * (0.17 - 0.07 * progress),
                                         particle % 2 == 0 ? Color{0.9F, 1, 0.95F, alpha}
                                                           : Color{0.47F, 1, 0.79F, alpha},
                                         true);
            }
            if (status != SurfaceStatus::ok)
                return failure(status);
        }
    }
    auto times = surface_.present(time_ns, time_ns);
    if (times.status == SurfaceStatus::ok)
    {
        last_time_ = time_ns;
        has_time_ = true;
    }
    return times;
}
void SSVEPConcretePresenter::close() noexcept
{
    surface_.close();
    if (surface_.lifecycle() != SurfaceLifecycle::closed)
        return;
    task_ = {};
    layout_ = {};
    prepared_ = false;
    has_time_ = false;
    last_time_ = 0;
}
} // namespace neurale::experiment_presentation
