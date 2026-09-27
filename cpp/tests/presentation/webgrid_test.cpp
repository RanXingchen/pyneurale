// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "check_returns.h"
#include "skip_policy.h"
#include "webgrid_presenter.h"

#include <cmath>
#include <cstdlib>
#include <string_view>

namespace
{

namespace ep = neurale::experiment_presentation;
namespace wg = neurale::experiments::webgrid;
using neurale::experiments::ContractStatus;
using neurale::experiments::kUnsetTargetId;
using neurale::experiments::SelectionEvent;
using neurale::experiments::SequenceOrdinal;
using neurale::experiments::TargetId;

constexpr int kGlfwKeyEscape = 256;

bool near(double left, double right) noexcept
{
    return std::abs(left - right) <= 1e-12;
}

wg::WebGridConfig make_config() noexcept
{
    wg::WebGridConfig config{};
    config.rows = 2;
    config.columns = 2;
    config.bounds = {0.0, 2.0, 0.0, 2.0};
    config.n_candidates = 4;
    config.candidates[0] = 1;
    config.candidates[1] = 2;
    config.candidates[2] = 3;
    config.candidates[3] = 4;
    config.schedule = wg::TargetScheduleKind::explicit_sequence;
    config.immediate_repetition = wg::ImmediateRepetitionPolicy::allow;
    config.correct_selection = wg::CorrectSelectionPolicy::advance_target;
    config.incorrect_selection = wg::IncorrectSelectionPolicy::keep_current_target;
    config.n_explicit = 3;
    config.explicit_targets[0] = 1;
    config.explicit_targets[1] = 2;
    config.explicit_targets[2] = 3;
    config.initial_target = 1;
    config.target_count_limit = 3;
    config.metric_version = wg::kMetricVersion1;
    return config;
}

ep::WebGridPresentationConfig make_presentation() noexcept
{
    return {
        .style = {.background = {0.01F, 0.02F, 0.03F, 1.0F},
                  .cell = {0.1F, 0.1F, 0.1F, 1.0F},
                  .active_target = {0.9F, 0.7F, 0.1F, 1.0F},
                  .correct_feedback = {0.1F, 0.8F, 0.2F, 1.0F},
                  .incorrect_feedback = {0.9F, 0.1F, 0.2F, 1.0F},
                  .grid_line = {0.6F, 0.6F, 0.6F, 1.0F},
                  .pointer = {1.0F, 1.0F, 1.0F, 1.0F},
                  .pointer_radius = 0.04,
                  .circle_segments = 24},
        .title = "PyNeurale WebGrid presentation test",
        .window_size = {320, 240},
        .aspect_policy = ep::AspectPolicy::fit_letterbox,
        .monitor_idx = 0,
        .swap_interval = 0,
        .input_capacity = 32,
        .selection_button = 0,
        .resizable = true,
        .visible = false,
    };
}

wg::WebGridSnapshot start_snapshot(const wg::WebGridConfig& config, wg::WebGridMachine& machine)
{
    wg::WebGridStepResult result{};
    if (machine.start(11, config, 0, result) != ContractStatus::ok)
    {
        std::abort();
    }
    return result.snapshot;
}

int test_render_plan_uses_webgrid_geometry()
{
    const auto config = make_config();
    const auto presentation = make_presentation();
    CHECK(wg::validate(config) == ContractStatus::ok);
    CHECK(ep::validate_webgrid_presentation(config, presentation) == ep::SurfaceStatus::ok);

    wg::WebGridMachine machine{};
    const auto snapshot = start_snapshot(config, machine);
    ep::WebGridRenderPlan plan{};
    CHECK(ep::build_webgrid_render_plan(config, presentation.style, snapshot, {1.0, 0.5}, nullptr,
                                        plan) == ep::SurfaceStatus::ok);
    CHECK(plan.n_cells == 4);
    CHECK(plan.task_bounds.min_x == config.bounds.min_x);
    CHECK(plan.task_bounds.max_y == config.bounds.max_y);
    for (std::uint16_t i = 0; i < plan.n_cells; ++i)
    {
        wg::GridCell expected{};
        CHECK(wg::grid_cell(config, static_cast<TargetId>(i + 1), expected) == ContractStatus::ok);
        CHECK(plan.cells[i].id == expected.id);
        CHECK(near(plan.cells[i].bounds.min_x, expected.bounds.min_x));
        CHECK(near(plan.cells[i].bounds.max_x, expected.bounds.max_x));
        CHECK(near(plan.cells[i].bounds.min_y, expected.bounds.min_y));
        CHECK(near(plan.cells[i].bounds.max_y, expected.bounds.max_y));
    }
    CHECK(plan.cells[0].visual == ep::WebGridCellVisual::active);
    CHECK(plan.cells[1].visual == ep::WebGridCellVisual::inactive);
    CHECK(near(plan.pointer.x, 1.0));
    CHECK(near(plan.pointer.y, 0.5));

    // Selection feedback is derived from the already-decided SelectionRecord,
    // not recomputed here. An incorrect record (selected 2, intended 1)
    // colors cell 2 red; a correct record (selected 1, intended 1) colors cell
    // 1 green. validate(SelectionEvent) forces correct == (selected == intended).
    auto make_record = [&](bool correct, TargetId selected, TargetId intended)
    {
        wg::WebGridSelectionRecord record{};
        record.target_onset_ns = 0;
        record.event.time_ns = 100;
        record.elapsed_since_target_onset_ns = 100;
        record.event.kind = neurale::experiments::SelectionKind::discrete;
        record.event.paradigm = machine.paradigm();
        record.event.correct = correct;
        record.event.selected_id = selected;
        record.event.intended_id = intended;
        record.event.trial.target_id = intended;
        record.event.trial.stimulus_id = neurale::experiments::kUnsetStimulusId;
        return record;
    };
    const auto incorrect_record = make_record(false, 2, 1);
    CHECK(ep::build_webgrid_render_plan(config, presentation.style, snapshot, {0.5, 0.5},
                                        &incorrect_record, plan) == ep::SurfaceStatus::ok);
    CHECK(plan.cells[0].visual == ep::WebGridCellVisual::active);
    CHECK(plan.cells[1].visual == ep::WebGridCellVisual::incorrect_feedback);

    const auto correct_record = make_record(true, 1, 1);
    CHECK(ep::build_webgrid_render_plan(config, presentation.style, snapshot, {0.5, 0.5},
                                        &correct_record, plan) == ep::SurfaceStatus::ok);
    CHECK(plan.cells[0].visual == ep::WebGridCellVisual::correct_feedback);

    const auto unchanged = plan;
    auto invalid_snapshot = snapshot;
    invalid_snapshot.active_target = 99;
    CHECK(ep::build_webgrid_render_plan(config, presentation.style, invalid_snapshot, {}, nullptr,
                                        plan) == ep::SurfaceStatus::invalid_configuration);
    CHECK(plan.n_cells == unchanged.n_cells);
    CHECK(plan.cells[0].visual == unchanged.cells[0].visual);
    return 0;
}

int test_pointer_mapping_edges_and_hidpi()
{
    const auto config = make_config();
    ep::CoordinateMapper mapper{};
    CHECK(mapper.configure({0.0, 0.0, 2.0, 2.0}, {400, 200}, {800, 400},
                           ep::AspectPolicy::fit_letterbox) == ep::SurfaceStatus::ok);
    const auto viewport = mapper.viewport();
    CHECK(viewport.left == 200);
    CHECK(viewport.top == 0);
    CHECK(viewport.width == 400);
    CHECK(viewport.height == 400);

    ep::Point2d logical{};
    bool inside{};
    CHECK(mapper.framebuffer_to_logical_unclipped({400.0, 200.0}, logical, inside));
    CHECK(inside);
    CHECK(near(logical.x, 1.0));
    CHECK(near(logical.y, 1.0));
    TargetId cell{};
    CHECK(wg::locate_cell(config, {logical.x, logical.y}, cell) == ContractStatus::ok);
    CHECK(cell == 4); // Both coordinates are exact half-open cell boundaries.

    CHECK(mapper.framebuffer_to_logical_unclipped({200.0, 400.0}, logical, inside));
    CHECK(
        inside); // bottom edge -> logical min_y; the coordinate mapper and grid-cell lookup agree it's included
    CHECK(near(logical.x, 0.0));
    CHECK(near(logical.y, 0.0));
    CHECK(wg::locate_cell(config, {logical.x, logical.y}, cell) == ContractStatus::ok);
    CHECK(cell ==
          1); // grid-cell lookup includes this logical corner; the coordinate mapper agrees.

    CHECK(mapper.framebuffer_to_logical_unclipped({600.0, 0.0}, logical, inside));
    CHECK(!inside);
    CHECK(near(logical.x, 2.0));
    CHECK(near(logical.y, 2.0));
    CHECK(wg::locate_cell(config, {logical.x, logical.y}, cell) == ContractStatus::ok);
    CHECK(cell == kUnsetTargetId); // grid-cell lookup excludes both maxima (half-open bounds).

    CHECK(mapper.framebuffer_to_logical_unclipped({100.0, 200.0}, logical, inside));
    CHECK(!inside);
    CHECK(near(logical.x, -0.5));
    CHECK(near(logical.y, 1.0));
    CHECK(wg::locate_cell(config, {logical.x, logical.y}, cell) == ContractStatus::ok);
    CHECK(cell == kUnsetTargetId);

    // Window coordinates are exactly half the framebuffer coordinates here.
    CHECK(mapper.window_pointer_to_logical_unclipped({200.0, 100.0}, logical, inside));
    CHECK(inside);
    CHECK(near(logical.x, 1.0));
    CHECK(near(logical.y, 1.0));
    const auto framebuffer = mapper.logical_to_framebuffer({0.25, 1.75});
    CHECK(mapper.framebuffer_to_logical_unclipped(framebuffer, logical, inside));
    CHECK(inside);
    CHECK(near(logical.x, 0.25));
    CHECK(near(logical.y, 1.75));

    // The four logical half-open boundaries, made explicit: min_x and min_y
    // are included, max_x and max_y are excluded, matching the grid contract.
    ep::Point2d boundary{};
    bool boundary_inside{};
    CHECK(mapper.framebuffer_to_logical_unclipped(mapper.logical_to_framebuffer({0.0, 1.0}),
                                                  boundary, boundary_inside));
    CHECK(boundary_inside); // logical min_x included
    CHECK(mapper.framebuffer_to_logical_unclipped(mapper.logical_to_framebuffer({2.0, 1.0}),
                                                  boundary, boundary_inside));
    CHECK(!boundary_inside); // logical max_x excluded
    CHECK(mapper.framebuffer_to_logical_unclipped(mapper.logical_to_framebuffer({1.0, 0.0}),
                                                  boundary, boundary_inside));
    CHECK(boundary_inside); // logical min_y included
    CHECK(mapper.framebuffer_to_logical_unclipped(mapper.logical_to_framebuffer({1.0, 2.0}),
                                                  boundary, boundary_inside));
    CHECK(!boundary_inside); // logical max_y excluded
    return 0;
}

SelectionEvent selection_from_request(const wg::WebGridConfig& config,
                                      const wg::WebGridMachine& machine,
                                      const ep::WebGridPresentationInput& request,
                                      SequenceOrdinal sequence)
{
    SelectionEvent selection{};
    const auto snapshot = machine.snapshot();
    if (wg::make_selection_event(config, request.pointer, snapshot.active_target, snapshot.trial,
                                 machine.paradigm(), request.experiment_time_ns, sequence,
                                 selection) != ContractStatus::ok)
    {
        std::abort();
    }
    return selection;
}

int test_pointer_adapter_matches_webgrid()
{
    ep::WebGridPointerInputAdapter adapter{};
    ep::WebGridPresentationInput adapted{};
    bool available{true};
    const ep::InputEvent pointer{.ordinal = 0,
                                 .renderer_time_ns = 100,
                                 .experiment_time_ns = 1'000,
                                 .kind = ep::InputEventKind::pointer_moved,
                                 .pointer = {1.0, 0.5},
                                 .pointer_inside = true};
    CHECK(adapter.adapt(pointer, adapted, available) == ep::SurfaceStatus::invalid_state);
    CHECK(adapter.prepare(0) == ep::SurfaceStatus::ok);
    CHECK(adapter.adapt(pointer, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available);
    CHECK(adapted.kind == ep::WebGridPresentationInputKind::pointer_update);
    CHECK(adapted.input_ordinal == 0);

    auto release = pointer;
    release.ordinal = 1;
    release.kind = ep::InputEventKind::mouse_button;
    release.action = ep::InputAction::release;
    release.code = 0;
    CHECK(adapter.adapt(release, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(!available);

    auto click = release;
    click.ordinal = 2;
    click.action = ep::InputAction::press;
    CHECK(adapter.adapt(click, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available);
    CHECK(adapted.kind == ep::WebGridPresentationInputKind::selection_request);
    CHECK(near(adapted.pointer.x, 1.0));
    CHECK(adapter.adapt(click, adapted, available) == ep::SurfaceStatus::invalid_state);
    CHECK(!available); // Reprocessing one physical click cannot emit twice.

    const auto config = make_config();
    wg::WebGridMachine machine{};
    start_snapshot(config, machine);
    auto selection = selection_from_request(config, machine, adapted, 1);
    CHECK(selection.selected_id == 2);
    CHECK(!selection.correct);
    wg::WebGridStepResult result{};
    CHECK(machine.step(adapted.experiment_time_ns, adapted.pointer, selection, result) ==
          ContractStatus::ok);
    CHECK(result.selection_processed);
    CHECK(!result.selection.event.correct);
    CHECK(result.snapshot.active_target == 1);

    auto correct_click = click;
    correct_click.ordinal = 3;
    correct_click.renderer_time_ns = 200;
    correct_click.experiment_time_ns = 2'000;
    correct_click.pointer = {0.5, 0.5};
    CHECK(adapter.adapt(correct_click, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available);
    selection = selection_from_request(config, machine, adapted, 2);
    CHECK(selection.selected_id == 1);
    CHECK(selection.correct);
    CHECK(machine.step(adapted.experiment_time_ns, adapted.pointer, selection, result) ==
          ContractStatus::ok);
    CHECK(result.trial_decided);
    CHECK(result.snapshot.active_target == 2);

    // Equal timestamps retain raw input ordinal: pointer first, click second.
    auto same_time_pointer = pointer;
    same_time_pointer.ordinal = 4;
    same_time_pointer.renderer_time_ns = 300;
    same_time_pointer.experiment_time_ns = 3'000;
    same_time_pointer.pointer = {1.5, 0.5};
    CHECK(adapter.adapt(same_time_pointer, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available && adapted.kind == ep::WebGridPresentationInputKind::pointer_update);
    CHECK(adapted.input_ordinal == 4);
    CHECK(machine.step(adapted.experiment_time_ns, adapted.pointer, result) == ContractStatus::ok);
    CHECK(!result.selection_processed);
    auto same_time_click = click;
    same_time_click.ordinal = 5;
    same_time_click.renderer_time_ns = 300;
    same_time_click.experiment_time_ns = 3'000;
    same_time_click.pointer = same_time_pointer.pointer;
    CHECK(adapter.adapt(same_time_click, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available && adapted.kind == ep::WebGridPresentationInputKind::selection_request);
    CHECK(adapted.input_ordinal == 5);
    selection = selection_from_request(config, machine, adapted, 3);
    CHECK(selection.selected_id == 2);
    CHECK(selection.correct);
    CHECK(machine.step(adapted.experiment_time_ns, adapted.pointer, selection, result) ==
          ContractStatus::ok);
    CHECK(result.trial_decided);
    CHECK(result.snapshot.active_target == 3);

    auto outside_click = same_time_click;
    outside_click.ordinal = 6;
    outside_click.experiment_time_ns = 4'000;
    outside_click.pointer = {-0.25, 0.5};
    outside_click.pointer_inside = false;
    CHECK(adapter.adapt(outside_click, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available && !adapted.inside_presentation);
    selection = selection_from_request(config, machine, adapted, 4);
    CHECK(selection.selected_id == kUnsetTargetId);
    CHECK(!selection.correct);

    auto close = outside_click;
    close.ordinal = 7;
    close.kind = ep::InputEventKind::window_close;
    CHECK(adapter.adapt(close, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available && adapted.kind == ep::WebGridPresentationInputKind::window_close_requested);
    auto escape = close;
    escape.ordinal = 8;
    escape.kind = ep::InputEventKind::key;
    escape.action = ep::InputAction::press;
    escape.code = kGlfwKeyEscape;
    CHECK(adapter.adapt(escape, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available && adapted.kind == ep::WebGridPresentationInputKind::escape_requested);

    adapter.reset();
    CHECK(adapter.adapt(pointer, adapted, available) == ep::SurfaceStatus::ok);
    CHECK(available);
    return 0;
}

int test_window()
{
    const auto config = make_config();
    auto presentation = make_presentation();
    wg::WebGridMachine machine{};
    const auto snapshot = start_snapshot(config, machine);
    ep::WebGridPresenter presenter{};
    auto status = presenter.open(config, presentation, ep::renderer_monotonic_now_ns(), 10'000);
    if (status == ep::SurfaceStatus::glfw_initialization_failed ||
        status == ep::SurfaceStatus::window_creation_failed ||
        status == ep::SurfaceStatus::monitor_unavailable ||
        status == ep::SurfaceStatus::opengl_function_missing)
    {
        return neurale::presentation_test::skip_or_fail(ep::surface_status_name(status));
    }
    CHECK(status == ep::SurfaceStatus::ok);
    CHECK(presenter.render(1, 2).status == ep::SurfaceStatus::invalid_state);
    CHECK(presenter.prepare() == ep::SurfaceStatus::ok);
    CHECK(presenter.update(snapshot, {0.5, 0.5}) == ep::SurfaceStatus::ok);
    for (int update = 1; update <= 64; ++update)
    {
        CHECK(presenter.update(snapshot, {0.01 * update, 0.02 * update}) == ep::SurfaceStatus::ok);
    }
    CHECK(presenter.latest_update_ordinal() == 65);
    CHECK(near(presenter.latest_render_plan().pointer.x, 0.64));
    CHECK(near(presenter.latest_render_plan().pointer.y, 1.28));
    const auto presented = presenter.render(20'000, 21'000);
    CHECK(presented.status == ep::SurfaceStatus::ok);
    CHECK(presenter.rendered_update_ordinal() == 65);
    const auto resources = presenter.resource_stats();
    CHECK(resources.vertex_capacity >=
          wg::cell_count(config) * 10 + 4 + presentation.style.circle_segments * 3);
    CHECK(resources.batch_capacity >= wg::cell_count(config) * 2 + 2);

    CHECK(presenter.request_window_size({400, 300}) == ep::SurfaceStatus::ok);
    CHECK(presenter.pump_events() == ep::SurfaceStatus::ok);
    CHECK(presenter.coordinates().window_size().width > 0);
    ep::WebGridPresentationInput input{};
    bool available{};
    CHECK(presenter.poll_input(input, available) == ep::SurfaceStatus::ok);

    presenter.cancel();
    CHECK(presenter.pump_events() == ep::SurfaceStatus::cancelled);
    presenter.cancel();
    presenter.close();
    presenter.close();
    CHECK(presenter.lifecycle() == ep::SurfaceLifecycle::closed);
    CHECK(!presenter.resource_stats().window_open);

    if (std::getenv("NEURALE_PRESENTATION_TEST_FULLSCREEN") != nullptr)
    {
        ep::WebGridPresenter fullscreen{};
        presentation.title = "PyNeurale WebGrid fullscreen test";
        presentation.fullscreen = true;
        presentation.resizable = false;
        presentation.input_capacity = 8;
        CHECK(fullscreen.open(config, presentation, ep::renderer_monotonic_now_ns(), 0) ==
              ep::SurfaceStatus::ok);
        CHECK(fullscreen.prepare() == ep::SurfaceStatus::ok);
        CHECK(fullscreen.update(snapshot, {0.5, 0.5}) == ep::SurfaceStatus::ok);
        CHECK(fullscreen.render(1, 1).status == ep::SurfaceStatus::ok);
        fullscreen.close();
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (const auto result = test_render_plan_uses_webgrid_geometry(); result != 0)
    {
        return result;
    }
    if (const auto result = test_pointer_mapping_edges_and_hidpi(); result != 0)
    {
        return result;
    }
    if (const auto result = test_pointer_adapter_matches_webgrid(); result != 0)
    {
        return result;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--window")
    {
        return test_window();
    }
    return 0;
}
