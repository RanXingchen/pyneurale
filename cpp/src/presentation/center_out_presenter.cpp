// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "center_out_presenter.h"

#include <cmath>
#include <limits>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace neurale::experiment_presentation
{
namespace
{

using namespace experiments;
namespace center_out = experiments::center_out;

[[nodiscard]] bool valid_color(Color color) noexcept
{
    return std::isfinite(color.red) && std::isfinite(color.green) && std::isfinite(color.blue) &&
           std::isfinite(color.alpha) && color.red >= 0.0F && color.red <= 1.0F &&
           color.green >= 0.0F && color.green <= 1.0F && color.blue >= 0.0F && color.blue <= 1.0F &&
           color.alpha >= 0.0F && color.alpha <= 1.0F;
}

[[nodiscard]] bool same_point(center_out::WorkspacePoint left,
                              center_out::WorkspacePoint right) noexcept
{
    return left.x == right.x && left.y == right.y;
}

[[nodiscard]] bool target_for_index(const center_out::CenterOut2DConfig& task,
                                    const center_out::CenterOutSnapshot& snapshot,
                                    const center_out::TargetPlacement*& target) noexcept
{
    if (snapshot.outward_idx >= task.layout.count)
    {
        return false;
    }
    target = &task.layout.surrounding[snapshot.outward_idx];
    return target->id == snapshot.outward_target;
}

struct Highlight
{
    TargetId target{kUnsetTargetId};
    CenterOutVisualState visual{CenterOutVisualState::inactive};
};

[[nodiscard]] bool resolve_highlight(const center_out::CenterOut2DConfig& task,
                                     const center_out::CenterOutSnapshot& snapshot,
                                     Highlight& highlight) noexcept
{
    if (!center_out::center_out_state_declared(snapshot.state))
    {
        return false;
    }
    const center_out::TargetPlacement* outward{};
    switch (snapshot.state)
    {
    case center_out::CenterOutState::idle:
    case center_out::CenterOutState::complete:
        return snapshot.active_target == kUnsetTargetId;
    case center_out::CenterOutState::move_to_center:
    case center_out::CenterOutState::hold_center:
        if (snapshot.phase != center_out::CenterOutPhase::to_center ||
            snapshot.active_target != task.layout.center.id ||
            !same_point(snapshot.active_position, task.layout.center.pos) ||
            !target_for_index(task, snapshot, outward))
        {
            return false;
        }
        highlight.target = task.layout.center.id;
        highlight.visual = snapshot.state == center_out::CenterOutState::hold_center
                               ? CenterOutVisualState::active_hold
                               : CenterOutVisualState::active_move;
        return true;
    case center_out::CenterOutState::center_success_dwell:
    case center_out::CenterOutState::center_failure_dwell:
        if (snapshot.phase != center_out::CenterOutPhase::to_center ||
            snapshot.active_target != kUnsetTargetId || !target_for_index(task, snapshot, outward))
        {
            return false;
        }
        highlight.target = task.layout.center.id;
        highlight.visual = snapshot.state == center_out::CenterOutState::center_success_dwell
                               ? CenterOutVisualState::success
                               : CenterOutVisualState::failure;
        return true;
    case center_out::CenterOutState::move_to_out:
    case center_out::CenterOutState::hold_out:
        if (snapshot.phase != center_out::CenterOutPhase::to_out ||
            !target_for_index(task, snapshot, outward) || snapshot.active_target != outward->id ||
            !same_point(snapshot.active_position, outward->pos))
        {
            return false;
        }
        highlight.target = outward->id;
        highlight.visual = snapshot.state == center_out::CenterOutState::hold_out
                               ? CenterOutVisualState::active_hold
                               : CenterOutVisualState::active_move;
        return true;
    case center_out::CenterOutState::out_success_dwell:
    case center_out::CenterOutState::out_failure_dwell:
        if (snapshot.phase != center_out::CenterOutPhase::to_out ||
            snapshot.active_target != kUnsetTargetId || !target_for_index(task, snapshot, outward))
        {
            return false;
        }
        highlight.target = outward->id;
        highlight.visual = snapshot.state == center_out::CenterOutState::out_success_dwell
                               ? CenterOutVisualState::success
                               : CenterOutVisualState::failure;
        return true;
    }
    return false;
}

[[nodiscard]] Color color_for_visual(const CenterOutPresentationStyle& style,
                                     CenterOutVisualState visual, CenterOutCircleRole role) noexcept
{
    switch (visual)
    {
    case CenterOutVisualState::active_move:
        return style.active_move;
    case CenterOutVisualState::active_hold:
        return style.active_hold;
    case CenterOutVisualState::success:
        return style.success;
    case CenterOutVisualState::failure:
        return style.failure;
    case CenterOutVisualState::cursor:
        return style.cursor;
    case CenterOutVisualState::inactive:
        return role == CenterOutCircleRole::center_target ? style.center_target
                                                          : style.outward_target;
    }
    return style.outward_target;
}

[[nodiscard]] SoftwarePresentationTimes failed_presentation(SurfaceStatus status,
                                                            ExperimentTimeNs requested_ns,
                                                            ExperimentTimeNs intended_ns) noexcept
{
    return SoftwarePresentationTimes{
        .status = status,
        .requested_ns = requested_ns,
        .intended_ns = intended_ns,
    };
}

} // namespace

SurfaceStatus validate(const CenterOutPresentationStyle& style) noexcept
{
    if (!valid_color(style.background) || !valid_color(style.center_target) ||
        !valid_color(style.outward_target) || !valid_color(style.active_move) ||
        !valid_color(style.active_hold) || !valid_color(style.success) ||
        !valid_color(style.failure) || !valid_color(style.cursor) ||
        !std::isfinite(style.target_radius) || style.target_radius <= 0.0 ||
        !std::isfinite(style.cursor_radius) || style.cursor_radius <= 0.0 ||
        style.circle_segments < 3 || style.circle_segments > kMaxCenterOutCircleSegments)
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus validate(const CenterOutPresentationConfig& config) noexcept
{
    using center_out::GeometryUnit;
    // The presentation geometry is one concrete unit, shared with the task. An
    // unspecified unit is rejected here so a default-constructed config cannot
    // reach the surface: the radii and logical space would be unitless and the
    // task match below would be meaningless.
    if (!center_out::geometry_unit_declared(config.geometry_unit) ||
        config.geometry_unit == GeometryUnit::unspecified)
    {
        return SurfaceStatus::invalid_configuration;
    }
    if (validate(config.style) != SurfaceStatus::ok)
    {
        return SurfaceStatus::invalid_configuration;
    }
    // Match CoordinateMapper::configure exactly: every logical-space field is
    // finite and the extents are positive, and the aspect policy is one of the
    // two declared values. Without the left/bottom check a NaN origin would pass
    // this preflight and only be rejected later when the surface configures the
    // mapper, so validate_center_out_presentation() would not be the complete
    // configuration validator it claims to be.
    if (!std::isfinite(config.logical_space.left) || !std::isfinite(config.logical_space.bottom) ||
        !std::isfinite(config.logical_space.width) || !std::isfinite(config.logical_space.height) ||
        config.logical_space.width <= 0.0 || config.logical_space.height <= 0.0 ||
        config.style.cursor_radius > config.logical_space.width * 0.5 ||
        config.style.cursor_radius > config.logical_space.height * 0.5 ||
        (config.aspect_policy != AspectPolicy::fit_letterbox &&
         config.aspect_policy != AspectPolicy::stretch))
    {
        return SurfaceStatus::invalid_configuration;
    }
    if (config.window_size.width <= 0 || config.window_size.height <= 0 ||
        config.input_capacity == 0 || (config.swap_interval != 0 && config.swap_interval != 1))
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus
validate_center_out_presentation(const center_out::CenterOut2DConfig& task,
                                 const CenterOutPresentationConfig& presentation) noexcept
{
    if (center_out::validate(task) != ContractStatus::ok ||
        validate(presentation) != SurfaceStatus::ok)
    {
        return SurfaceStatus::invalid_configuration;
    }
    // The frozen boundary: the task and the presentation share one geometry
    // unit. No units are converted here, so a mismatch is a configuration error
    // rather than a silent rescale that would draw a millimetre task in a
    // normalized workspace (or vice versa).
    if (task.geometry_unit != presentation.geometry_unit)
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus build_center_out_render_plan(const center_out::CenterOut2DConfig& task,
                                           const CenterOutPresentationStyle& style,
                                           const center_out::CenterOutSnapshot& snapshot,
                                           const center_out::WorkspacePoint& cursor,
                                           CenterOutRenderPlan& plan) noexcept
{
    if (center_out::validate(task) != ContractStatus::ok || validate(style) != SurfaceStatus::ok ||
        !std::isfinite(cursor.x) || !std::isfinite(cursor.y))
    {
        return SurfaceStatus::invalid_configuration;
    }

    Highlight highlight{};
    if (!resolve_highlight(task, snapshot, highlight))
    {
        return SurfaceStatus::invalid_configuration;
    }

    CenterOutRenderPlan candidate{};
    candidate.background = style.background;
    const auto append_target =
        [&](const center_out::TargetPlacement& target, CenterOutCircleRole role) noexcept
    {
        const bool active = target.id == highlight.target;
        const auto visual = active ? highlight.visual : CenterOutVisualState::inactive;
        candidate.circles[candidate.n_circles++] = CenterOutRenderCircle{
            .role = role,
            .visual = visual,
            .target_id = target.id,
            .center = target.pos,
            .radius = style.target_radius,
            .color = color_for_visual(style, visual, role),
            .filled = active,
        };
    };

    append_target(task.layout.center, CenterOutCircleRole::center_target);
    for (std::uint8_t i = 0; i < task.layout.count; ++i)
    {
        append_target(task.layout.surrounding[i], CenterOutCircleRole::outward_target);
    }
    candidate.circles[candidate.n_circles++] = CenterOutRenderCircle{
        .role = CenterOutCircleRole::cursor,
        .visual = CenterOutVisualState::cursor,
        .target_id = kUnsetTargetId,
        .center = cursor,
        .radius = style.cursor_radius,
        .color = style.cursor,
        .filled = true,
    };
    plan = candidate;
    return SurfaceStatus::ok;
}

bool center_out_control_event(const InputEvent& input,
                              CenterOutPresentationControlEvent& event) noexcept
{
    CenterOutPresentationControlKind kind{};
    if (input.kind == InputEventKind::window_close)
    {
        kind = CenterOutPresentationControlKind::window_close_requested;
    }
    else if (input.kind == InputEventKind::key && input.action == InputAction::press &&
             input.code == GLFW_KEY_ESCAPE)
    {
        kind = CenterOutPresentationControlKind::escape_requested;
    }
    else
    {
        return false;
    }
    event = CenterOutPresentationControlEvent{
        .kind = kind,
        .input_ordinal = input.ordinal,
        .renderer_time_ns = input.renderer_time_ns,
        .experiment_time_ns = input.experiment_time_ns,
    };
    return true;
}

SurfaceStatus CenterOut2DPresenter::open(const CenterOutPresentationConfig& config,
                                         RendererTimeNs renderer_origin_ns,
                                         ExperimentTimeNs experiment_origin_ns)
{
    if (validate(config) != SurfaceStatus::ok)
    {
        return SurfaceStatus::invalid_configuration;
    }
    const WindowConfig window{
        .title = config.title,
        .window_size = config.window_size,
        .logical_space = config.logical_space,
        .aspect_policy = config.aspect_policy,
        .monitor_idx = config.monitor_idx,
        .swap_interval = config.swap_interval,
        .input_capacity = config.input_capacity,
        .input_admission = InputAdmission::key | InputAdmission::window_close,
        .fullscreen = config.fullscreen,
        .resizable = config.resizable,
        .visible = config.visible,
    };
    const auto status = surface_.open(window, renderer_origin_ns, experiment_origin_ns);
    if (status != SurfaceStatus::ok)
    {
        return status;
    }
    // See title_'s declaration for why the bytes are owned here.
    title_.assign(config.title);
    presentation_ = config;
    presentation_.title = title_;
    return SurfaceStatus::ok;
}

SurfaceStatus CenterOut2DPresenter::prepare(const center_out::CenterOut2DConfig& task)
{
    if (prepared_)
    {
        return SurfaceStatus::invalid_state;
    }
    // The presentation geometry set at open and the task geometry must be the
    // same unit; no units are converted here, so a mismatch is reported, not drawn.
    if (validate_center_out_presentation(task, presentation_) != SurfaceStatus::ok)
    {
        return SurfaceStatus::invalid_configuration;
    }
    constexpr std::size_t max_draws = kMaxCenterOutRenderCircles;
    const auto max_vertices = max_draws * presentation_.style.circle_segments * 3;
    const auto status = surface_.prepare({.max_vertices = max_vertices,
                                          .max_draw_batches = max_draws,
                                          .circle_segments = presentation_.style.circle_segments},
                                         {});
    if (status != SurfaceStatus::ok)
    {
        return status;
    }
    task_ = task;
    style_ = presentation_.style;
    latest_plan_ = {};
    latest_update_ordinal_ = 0;
    rendered_update_ordinal_ = 0;
    has_update_ = false;
    prepared_ = true;
    return SurfaceStatus::ok;
}

SurfaceStatus CenterOut2DPresenter::update(const center_out::CenterOutSnapshot& snapshot,
                                           const center_out::WorkspacePoint& cursor) noexcept
{
    if (!prepared_)
    {
        return SurfaceStatus::invalid_state;
    }
    if (latest_update_ordinal_ == (std::numeric_limits<std::uint64_t>::max)())
    {
        return SurfaceStatus::invalid_state;
    }
    CenterOutRenderPlan candidate{};
    const auto status = build_center_out_render_plan(task_, style_, snapshot, cursor, candidate);
    if (status != SurfaceStatus::ok)
    {
        return status;
    }
    latest_plan_ = candidate;
    ++latest_update_ordinal_;
    has_update_ = true;
    return SurfaceStatus::ok;
}

SurfaceStatus CenterOut2DPresenter::pump_events() noexcept
{
    return surface_.pump_events();
}

SurfaceStatus CenterOut2DPresenter::poll_control(CenterOutPresentationControlEvent& event,
                                                 bool& available) noexcept
{
    available = false;
    InputEvent input{};
    bool input_available{};
    auto status = surface_.poll_input(input, input_available);
    while (status == SurfaceStatus::ok && input_available)
    {
        if (center_out_control_event(input, event))
        {
            available = true;
            return SurfaceStatus::ok;
        }
        status = surface_.poll_input(input, input_available);
    }
    return status;
}

SoftwarePresentationTimes CenterOut2DPresenter::render(ExperimentTimeNs requested_ns,
                                                       ExperimentTimeNs intended_ns) noexcept
{
    if (!prepared_ || !has_update_)
    {
        return failed_presentation(SurfaceStatus::invalid_state, requested_ns, intended_ns);
    }
    auto status = surface_.begin_frame(latest_plan_.background);
    if (status != SurfaceStatus::ok)
    {
        return failed_presentation(status, requested_ns, intended_ns);
    }
    for (std::uint8_t i = 0; i < latest_plan_.n_circles; ++i)
    {
        const auto& circle = latest_plan_.circles[i];
        const Point2d center{circle.center.x, circle.center.y};
        status = surface_.circle(center, circle.radius, circle.color, circle.filled);
        if (status != SurfaceStatus::ok)
        {
            return failed_presentation(status, requested_ns, intended_ns);
        }
    }
    auto result = surface_.present(requested_ns, intended_ns);
    if (result.status == SurfaceStatus::ok)
    {
        rendered_update_ordinal_ = latest_update_ordinal_;
    }
    return result;
}

SurfaceStatus CenterOut2DPresenter::request_window_size(Size2i size) noexcept
{
    return surface_.request_window_size(size);
}

void CenterOut2DPresenter::cancel() noexcept
{
    surface_.cancel();
}

void CenterOut2DPresenter::close() noexcept
{
    surface_.close();
    if (surface_.lifecycle() != SurfaceLifecycle::closed)
    {
        return;
    }
    task_ = {};
    presentation_ = {};
    title_.clear();
    style_ = {};
    latest_plan_ = {};
    latest_update_ordinal_ = 0;
    rendered_update_ordinal_ = 0;
    prepared_ = false;
    has_update_ = false;
}

} // namespace neurale::experiment_presentation
