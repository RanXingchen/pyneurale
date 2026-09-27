// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "webgrid_presenter.h"

#include <cmath>
#include <limits>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace neurale::experiment_presentation
{
namespace
{

namespace webgrid = experiments::webgrid;

[[nodiscard]] bool valid_color(Color color) noexcept
{
    return std::isfinite(color.red) && std::isfinite(color.green) && std::isfinite(color.blue) &&
           std::isfinite(color.alpha) && color.red >= 0.0F && color.red <= 1.0F &&
           color.green >= 0.0F && color.green <= 1.0F && color.blue >= 0.0F && color.blue <= 1.0F &&
           color.alpha >= 0.0F && color.alpha <= 1.0F;
}

[[nodiscard]] Rect2d task_rectangle(const webgrid::TaskBounds& bounds) noexcept
{
    return {bounds.min_x, bounds.min_y, bounds.max_x - bounds.min_x, bounds.max_y - bounds.min_y};
}

[[nodiscard]] Color cell_color(const WebGridPresentationStyle& style,
                               WebGridCellVisual visual) noexcept
{
    switch (visual)
    {
    case WebGridCellVisual::active:
        return style.active_target;
    case WebGridCellVisual::correct_feedback:
        return style.correct_feedback;
    case WebGridCellVisual::incorrect_feedback:
        return style.incorrect_feedback;
    case WebGridCellVisual::inactive:
        return style.cell;
    }
    return style.cell;
}

[[nodiscard]] SoftwarePresentationTimes
failed_presentation(SurfaceStatus status, experiments::ExperimentTimeNs requested_ns,
                    experiments::ExperimentTimeNs intended_ns) noexcept
{
    return {.status = status, .requested_ns = requested_ns, .intended_ns = intended_ns};
}

} // namespace

SurfaceStatus validate(const WebGridPresentationStyle& style) noexcept
{
    if (!valid_color(style.background) || !valid_color(style.cell) ||
        !valid_color(style.active_target) || !valid_color(style.correct_feedback) ||
        !valid_color(style.incorrect_feedback) || !valid_color(style.grid_line) ||
        !valid_color(style.pointer) || !std::isfinite(style.pointer_radius) ||
        style.pointer_radius <= 0.0 || style.circle_segments < 3 || style.circle_segments > 256)
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus validate(const WebGridPresentationConfig& config) noexcept
{
    if (validate(config.style) != SurfaceStatus::ok || config.title.empty() ||
        config.window_size.width <= 0 || config.window_size.height <= 0 ||
        (config.aspect_policy != AspectPolicy::fit_letterbox &&
         config.aspect_policy != AspectPolicy::stretch) ||
        config.monitor_idx < -1 || config.input_capacity == 0 ||
        config.selection_button < GLFW_MOUSE_BUTTON_1 ||
        config.selection_button > GLFW_MOUSE_BUTTON_LAST)
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus validate_webgrid_presentation(const webgrid::WebGridConfig& task,
                                            const WebGridPresentationConfig& presentation) noexcept
{
    if (webgrid::validate(task) != experiments::ContractStatus::ok ||
        validate(presentation) != SurfaceStatus::ok)
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus build_webgrid_render_plan(const webgrid::WebGridConfig& task,
                                        const WebGridPresentationStyle& style,
                                        const webgrid::WebGridSnapshot& snapshot,
                                        const webgrid::PointerPosition& pointer,
                                        const webgrid::WebGridSelectionRecord* last_selection,
                                        WebGridRenderPlan& plan) noexcept
{
    if (webgrid::validate(task) != experiments::ContractStatus::ok ||
        validate(style) != SurfaceStatus::ok ||
        webgrid::validate(pointer) != experiments::ContractStatus::ok ||
        !webgrid::webgrid_state_declared(snapshot.state))
    {
        return SurfaceStatus::invalid_configuration;
    }
    // Selection feedback is derived from the task's already-decided record,
    // never asserted here. A null last_selection means no feedback to show.
    bool has_feedback = false;
    bool feedback_correct = false;
    experiments::TargetId feedback_cell = experiments::kUnsetTargetId;
    if (last_selection != nullptr)
    {
        if (webgrid::validate(*last_selection) != experiments::ContractStatus::ok)
        {
            return SurfaceStatus::invalid_configuration;
        }
        has_feedback = true;
        feedback_correct = last_selection->event.correct;
        feedback_cell = last_selection->event.selected_id;
    }
    if ((snapshot.state == webgrid::WebGridState::active_target) !=
        (snapshot.active_target != experiments::kUnsetTargetId))
    {
        return SurfaceStatus::invalid_configuration;
    }
    if (snapshot.active_target != experiments::kUnsetTargetId)
    {
        webgrid::GridCell ignored{};
        if (webgrid::grid_cell(task, snapshot.active_target, ignored) !=
            experiments::ContractStatus::ok)
        {
            return SurfaceStatus::invalid_configuration;
        }
    }

    WebGridRenderPlan candidate{};
    candidate.background = style.background;
    candidate.task_bounds = task.bounds;
    candidate.pointer = pointer;
    candidate.pointer_radius = style.pointer_radius;
    candidate.pointer_color =
        (has_feedback && !feedback_correct) ? style.incorrect_feedback : style.pointer;
    const auto count = webgrid::cell_count(task);
    for (std::size_t offset = 0; offset < count; ++offset)
    {
        webgrid::GridCell cell{};
        if (webgrid::grid_cell(task, static_cast<experiments::TargetId>(offset + 1), cell) !=
            experiments::ContractStatus::ok)
        {
            return SurfaceStatus::invalid_configuration;
        }
        auto visual = cell.id == snapshot.active_target ? WebGridCellVisual::active
                                                        : WebGridCellVisual::inactive;
        if (has_feedback && cell.id == feedback_cell &&
            feedback_cell != experiments::kUnsetTargetId)
        {
            visual = feedback_correct ? WebGridCellVisual::correct_feedback
                                      : WebGridCellVisual::incorrect_feedback;
        }
        candidate.cells[offset] = {
            .id = cell.id,
            .bounds = cell.bounds,
            .visual = visual,
            .fill = cell_color(style, visual),
        };
    }
    candidate.n_cells = static_cast<std::uint16_t>(count);
    plan = candidate;
    return SurfaceStatus::ok;
}

SurfaceStatus WebGridPointerInputAdapter::prepare(int selection_button) noexcept
{
    if (selection_button < GLFW_MOUSE_BUTTON_1 || selection_button > GLFW_MOUSE_BUTTON_LAST)
    {
        return SurfaceStatus::invalid_configuration;
    }
    selection_button_ = selection_button;
    prepared_ = true;
    has_ordinal_ = false;
    last_ordinal_ = 0;
    return SurfaceStatus::ok;
}

SurfaceStatus WebGridPointerInputAdapter::adapt(const InputEvent& input,
                                                WebGridPresentationInput& output,
                                                bool& available) noexcept
{
    available = false;
    if (!prepared_)
    {
        return SurfaceStatus::invalid_state;
    }
    if (has_ordinal_ && input.ordinal <= last_ordinal_)
    {
        return SurfaceStatus::invalid_state;
    }
    last_ordinal_ = input.ordinal;
    has_ordinal_ = true;

    WebGridPresentationInputKind kind{};
    if (input.kind == InputEventKind::pointer_moved)
    {
        kind = WebGridPresentationInputKind::pointer_update;
    }
    else if (input.kind == InputEventKind::mouse_button && input.action == InputAction::press &&
             input.code == selection_button_)
    {
        kind = WebGridPresentationInputKind::selection_request;
    }
    else if (input.kind == InputEventKind::key && input.action == InputAction::press &&
             input.code == GLFW_KEY_ESCAPE)
    {
        kind = WebGridPresentationInputKind::escape_requested;
    }
    else if (input.kind == InputEventKind::window_close)
    {
        kind = WebGridPresentationInputKind::window_close_requested;
    }
    else
    {
        return SurfaceStatus::ok;
    }

    output = {
        .kind = kind,
        .pointer = {input.pointer.x, input.pointer.y},
        .inside_presentation = input.pointer_inside,
        .input_ordinal = input.ordinal,
        .renderer_time_ns = input.renderer_time_ns,
        .experiment_time_ns = input.experiment_time_ns,
        .button = input.kind == InputEventKind::mouse_button ? input.code : 0,
        .modifiers = input.modifiers,
    };
    available = true;
    return SurfaceStatus::ok;
}

void WebGridPointerInputAdapter::reset() noexcept
{
    has_ordinal_ = false;
    last_ordinal_ = 0;
}

SurfaceStatus WebGridPresenter::open(const webgrid::WebGridConfig& task,
                                     const WebGridPresentationConfig& presentation,
                                     RendererTimeNs renderer_origin_ns,
                                     experiments::ExperimentTimeNs experiment_origin_ns)
{
    if (opened_)
    {
        return SurfaceStatus::invalid_state;
    }
    if (validate_webgrid_presentation(task, presentation) != SurfaceStatus::ok)
    {
        return SurfaceStatus::invalid_configuration;
    }
    const auto status = surface_.open(
        {.title = presentation.title,
         .window_size = presentation.window_size,
         .logical_space = task_rectangle(task.bounds),
         .aspect_policy = presentation.aspect_policy,
         .monitor_idx = presentation.monitor_idx,
         .swap_interval = presentation.swap_interval,
         .input_capacity = presentation.input_capacity,
         .input_admission = InputAdmission::pointer_moved | InputAdmission::mouse_button |
                            InputAdmission::key | InputAdmission::window_close,
         .fullscreen = presentation.fullscreen,
         .resizable = presentation.resizable,
         .visible = presentation.visible},
        renderer_origin_ns, experiment_origin_ns);
    if (status != SurfaceStatus::ok)
    {
        return status;
    }
    task_ = task;
    // See title_'s declaration for why the bytes are owned here.
    title_.assign(presentation.title);
    presentation_ = presentation;
    presentation_.title = title_;
    opened_ = true;
    return input_.prepare(presentation.selection_button);
}

SurfaceStatus WebGridPresenter::prepare()
{
    if (!opened_ || prepared_)
    {
        return SurfaceStatus::invalid_state;
    }
    const auto cells = webgrid::cell_count(task_);
    const auto max_vertices = cells * 10 + 4 + presentation_.style.circle_segments * 3;
    const auto max_batches = cells * 2 + 2;
    const auto status = surface_.prepare({.max_vertices = max_vertices,
                                          .max_draw_batches = max_batches,
                                          .circle_segments = presentation_.style.circle_segments},
                                         {});
    if (status != SurfaceStatus::ok)
    {
        return status;
    }
    latest_plan_ = {};
    latest_update_ordinal_ = 0;
    rendered_update_ordinal_ = 0;
    has_update_ = false;
    prepared_ = true;
    return SurfaceStatus::ok;
}

SurfaceStatus
WebGridPresenter::update(const webgrid::WebGridSnapshot& snapshot,
                         const webgrid::PointerPosition& pointer,
                         const webgrid::WebGridSelectionRecord* last_selection) noexcept
{
    if (!prepared_ || latest_update_ordinal_ == (std::numeric_limits<std::uint64_t>::max)())
    {
        return SurfaceStatus::invalid_state;
    }
    WebGridRenderPlan candidate{};
    const auto status = build_webgrid_render_plan(task_, presentation_.style, snapshot, pointer,
                                                  last_selection, candidate);
    if (status != SurfaceStatus::ok)
    {
        return status;
    }
    latest_plan_ = candidate;
    ++latest_update_ordinal_;
    has_update_ = true;
    return SurfaceStatus::ok;
}

SurfaceStatus WebGridPresenter::pump_events() noexcept
{
    return surface_.pump_events();
}

SurfaceStatus WebGridPresenter::poll_input(WebGridPresentationInput& input,
                                           bool& available) noexcept
{
    available = false;
    InputEvent raw{};
    bool raw_available{};
    auto status = surface_.poll_input(raw, raw_available);
    while (status == SurfaceStatus::ok && raw_available)
    {
        status = input_.adapt(raw, input, available);
        if (status != SurfaceStatus::ok || available)
        {
            return status;
        }
        status = surface_.poll_input(raw, raw_available);
    }
    return status;
}

SoftwarePresentationTimes
WebGridPresenter::render(experiments::ExperimentTimeNs requested_ns,
                         experiments::ExperimentTimeNs intended_ns) noexcept
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
    for (std::uint16_t i = 0; i < latest_plan_.n_cells; ++i)
    {
        const auto& cell = latest_plan_.cells[i];
        const Rect2d rectangle{cell.bounds.min_x, cell.bounds.min_y,
                               cell.bounds.max_x - cell.bounds.min_x,
                               cell.bounds.max_y - cell.bounds.min_y};
        status = surface_.rectangle(rectangle, cell.fill, true);
        if (status == SurfaceStatus::ok)
        {
            status = surface_.rectangle(rectangle, presentation_.style.grid_line, false);
        }
        if (status != SurfaceStatus::ok)
        {
            return failed_presentation(status, requested_ns, intended_ns);
        }
    }
    status = surface_.rectangle(task_rectangle(latest_plan_.task_bounds),
                                presentation_.style.grid_line, false);
    if (status == SurfaceStatus::ok)
    {
        status = surface_.circle({latest_plan_.pointer.x, latest_plan_.pointer.y},
                                 latest_plan_.pointer_radius, latest_plan_.pointer_color, true);
    }
    if (status != SurfaceStatus::ok)
    {
        return failed_presentation(status, requested_ns, intended_ns);
    }
    auto result = surface_.present(requested_ns, intended_ns);
    if (result.status == SurfaceStatus::ok)
    {
        rendered_update_ordinal_ = latest_update_ordinal_;
    }
    return result;
}

SurfaceStatus WebGridPresenter::request_window_size(Size2i size) noexcept
{
    return surface_.request_window_size(size);
}

void WebGridPresenter::cancel() noexcept
{
    surface_.cancel();
}

void WebGridPresenter::close() noexcept
{
    surface_.close();
    if (surface_.lifecycle() != SurfaceLifecycle::closed)
    {
        return;
    }
    input_.reset();
    task_ = {};
    presentation_ = {};
    title_.clear();
    latest_plan_ = {};
    latest_update_ordinal_ = 0;
    rendered_update_ordinal_ = 0;
    opened_ = false;
    prepared_ = false;
    has_update_ = false;
}

} // namespace neurale::experiment_presentation
