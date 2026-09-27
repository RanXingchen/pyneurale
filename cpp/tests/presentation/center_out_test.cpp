// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "allocation_counter.h"
#include "center_out_presenter.h"
#include "skip_policy.h"

#include "check_returns.h"

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <new>
#include <string_view>

// This suite drives code that allocates through the nothrow forms, which
// allocation_counter.h leaves alone; counting them here keeps the totals
// below honest without changing what every other suite measures.
void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

namespace
{
constexpr int kGlfwKeyEscape = 256;
} // namespace

namespace
{

namespace ep = neurale::experiment_presentation;
namespace co = neurale::experiments::center_out;
using neurale::experiments::ContractStatus;
using neurale::experiments::kUnsetTargetId;

bool near(double left, double right) noexcept
{
    return std::abs(left - right) <= 1e-12;
}

co::CenterOut2DConfig make_config()
{
    co::RadialLayoutRequest request{};
    request.radius = 0.75;
    request.count = 4;
    request.center_id = 1;
    for (std::uint8_t i = 0; i < request.count; ++i)
    {
        request.ids[i] = static_cast<neurale::experiments::TargetId>(i + 2);
        request.spokes[i] = static_cast<std::uint32_t>(i * 2);
    }

    co::CenterOut2DConfig config{};
    config.geometry_unit = co::GeometryUnit::normalized;
    if (co::build_radial_layout(request, config.layout) != ContractStatus::ok)
    {
        std::abort();
    }
    config.acceptance = {0.1, 0.1};
    config.cursor = {0.01};
    config.movement_timeout = {1'000, 1'000};
    config.hold_ns = 20;
    config.reward_dwell = {10, 10};
    config.punish_dwell = {10, 10};
    config.selection = co::TargetSelectionPolicy::repeat_until_success;
    config.seed = 0x1234;
    return config;
}

// A millimetre task -- the unit the experiment-contract tests use heavily. Radial targets sit
// 100 mm out, acceptance is 5 mm, the cursor extent is 2 mm.
co::CenterOut2DConfig make_mm_config()
{
    co::RadialLayoutRequest request{};
    request.radius = 100.0;
    request.count = 4;
    request.center_id = 1;
    for (std::uint8_t i = 0; i < request.count; ++i)
    {
        request.ids[i] = static_cast<neurale::experiments::TargetId>(i + 2);
        request.spokes[i] = static_cast<std::uint32_t>(i * 2);
    }
    co::CenterOut2DConfig config{};
    config.geometry_unit = co::GeometryUnit::millimetres;
    if (co::build_radial_layout(request, config.layout) != ContractStatus::ok)
    {
        std::abort();
    }
    config.acceptance = {5.0, 5.0};
    config.cursor = {2.0};
    config.movement_timeout = {1'000, 1'000};
    config.hold_ns = 20;
    config.reward_dwell = {10, 10};
    config.punish_dwell = {10, 10};
    config.selection = co::TargetSelectionPolicy::repeat_until_success;
    config.seed = 0x1234;
    return config;
}

ep::CenterOutPresentationStyle make_style() noexcept
{
    return {
        .background = {0.01F, 0.02F, 0.03F, 1.0F},
        .center_target = {0.1F, 0.1F, 0.1F, 1.0F},
        .outward_target = {0.2F, 0.2F, 0.2F, 1.0F},
        .active_move = {0.3F, 0.4F, 0.5F, 1.0F},
        .active_hold = {0.4F, 0.5F, 0.6F, 1.0F},
        .success = {0.1F, 0.8F, 0.2F, 1.0F},
        .failure = {0.9F, 0.1F, 0.2F, 1.0F},
        .cursor = {1.0F, 1.0F, 1.0F, 1.0F},
        .target_radius = 0.06,
        .cursor_radius = 0.025,
        .circle_segments = 24,
    };
}

// A normalized presentation contract matching make_config(). The unit, the
// logical space, and the style radii are all normalized, and the task's
// geometry_unit is normalized too, so the frozen boundary accepts it.
ep::CenterOutPresentationConfig make_presentation() noexcept
{
    return {
        .geometry_unit = co::GeometryUnit::normalized,
        .logical_space = {-1.0, -1.0, 2.0, 2.0},
        .style = make_style(),
        .title = "PyNeurale Center-Out presentation test",
        .window_size = {320, 240},
        .aspect_policy = ep::AspectPolicy::fit_letterbox,
        .monitor_idx = 0,
        .swap_interval = 0,
        .input_capacity = 32,
        .fullscreen = false,
        .resizable = true,
        .visible = false,
    };
}

co::CenterOutSnapshot start_snapshot(const co::CenterOut2DConfig& config,
                                     co::CenterOutMachine& machine)
{
    co::CenterOutStepResult result{};
    if (machine.start(7, config, 0, result) != ContractStatus::ok)
    {
        std::abort();
    }
    return result.snapshot;
}

const ep::CenterOutRenderCircle* find_target(const ep::CenterOutRenderPlan& plan,
                                             neurale::experiments::TargetId id) noexcept
{
    for (std::uint8_t i = 0; i < plan.n_circles; ++i)
    {
        if (plan.circles[i].target_id == id)
        {
            return &plan.circles[i];
        }
    }
    return nullptr;
}

int test_render_plan_geometry_and_styles()
{
    const auto config = make_config();
    const auto style = make_style();
    ep::CenterOutRenderPlan plan{};
    const co::CenterOutSnapshot idle{};
    CHECK(ep::build_center_out_render_plan(config, style, idle, {0.2, -0.3}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.n_circles == config.layout.count + 2);
    CHECK(find_target(plan, config.layout.center.id)->visual == ep::CenterOutVisualState::inactive);
    CHECK(!find_target(plan, config.layout.center.id)->filled);
    CHECK(plan.circles[plan.n_circles - 1].role == ep::CenterOutCircleRole::cursor);

    co::CenterOutMachine machine{};
    auto snapshot = start_snapshot(config, machine);
    CHECK(ep::build_center_out_render_plan(config, style, snapshot, {0.2, -0.3}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.n_circles == config.layout.count + 2);
    CHECK(plan.circles[0].role == ep::CenterOutCircleRole::center_target);
    CHECK(plan.circles[0].target_id == config.layout.center.id);
    CHECK(plan.circles[0].visual == ep::CenterOutVisualState::active_move);
    CHECK(plan.circles[0].filled);
    for (std::uint8_t i = 0; i < config.layout.count; ++i)
    {
        const auto* target = find_target(plan, config.layout.surrounding[i].id);
        CHECK(target != nullptr);
        CHECK(target->visual == ep::CenterOutVisualState::inactive);
        CHECK(!target->filled);
    }
    const auto& cursor = plan.circles[plan.n_circles - 1];
    CHECK(cursor.role == ep::CenterOutCircleRole::cursor);
    CHECK(cursor.filled);
    CHECK(near(cursor.center.x, 0.2));
    CHECK(near(cursor.center.y, -0.3));

    auto full_config = make_config();
    co::RadialLayoutRequest full_request{};
    full_request.radius = 0.75;
    full_request.count = static_cast<std::uint8_t>(co::kMaxSurroundingTargets);
    full_request.center_id = 1;
    for (std::uint8_t i = 0; i < full_request.count; ++i)
    {
        full_request.ids[i] = static_cast<neurale::experiments::TargetId>(i + 2);
        full_request.spokes[i] = i;
    }
    CHECK(co::build_radial_layout(full_request, full_config.layout) == ContractStatus::ok);
    co::CenterOutMachine full_machine{};
    const auto full_snapshot = start_snapshot(full_config, full_machine);
    ep::CenterOutRenderPlan full_plan{};
    CHECK(ep::build_center_out_render_plan(full_config, style, full_snapshot, {}, full_plan) ==
          ep::SurfaceStatus::ok);
    CHECK(full_plan.n_circles == ep::kMaxCenterOutRenderCircles);

    const auto allocation_baseline = allocations.load(std::memory_order_relaxed);
    for (int repetition = 0; repetition < 128; ++repetition)
    {
        CHECK(ep::build_center_out_render_plan(config, style, snapshot,
                                               {0.001 * repetition, -0.001 * repetition},
                                               plan) == ep::SurfaceStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == allocation_baseline);

    auto invalid_style = style;
    invalid_style.target_radius = 0.0;
    CHECK(ep::validate(invalid_style) == ep::SurfaceStatus::invalid_configuration);

    co::CenterOutStepResult result{};
    CHECK(machine.step(10, config.layout.center.pos, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == co::CenterOutState::hold_center);
    CHECK(ep::build_center_out_render_plan(config, style, result.snapshot, {}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(find_target(plan, config.layout.center.id)->visual ==
          ep::CenterOutVisualState::active_hold);

    CHECK(machine.step(30, config.layout.center.pos, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == co::CenterOutState::center_success_dwell);
    CHECK(ep::build_center_out_render_plan(config, style, result.snapshot, {}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(find_target(plan, config.layout.center.id)->visual == ep::CenterOutVisualState::success);

    const auto outward = config.layout.surrounding[result.snapshot.outward_idx];
    CHECK(machine.step(40, outward.pos, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == co::CenterOutState::hold_out);
    CHECK(ep::build_center_out_render_plan(config, style, result.snapshot, {}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.n_circles == config.layout.count + 2);
    CHECK(find_target(plan, config.layout.center.id)->visual == ep::CenterOutVisualState::inactive);
    CHECK(!find_target(plan, config.layout.center.id)->filled);
    CHECK(find_target(plan, outward.id)->visual == ep::CenterOutVisualState::active_hold);
    CHECK(find_target(plan, outward.id)->filled);

    CHECK(machine.step(60, outward.pos, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == co::CenterOutState::out_success_dwell);
    CHECK(ep::build_center_out_render_plan(config, style, result.snapshot, {}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.n_circles == config.layout.count + 2);
    CHECK(find_target(plan, config.layout.center.id)->visual == ep::CenterOutVisualState::inactive);
    CHECK(find_target(plan, outward.id)->visual == ep::CenterOutVisualState::success);

    co::CenterOutMachine timeout_machine{};
    start_snapshot(config, timeout_machine);
    CHECK(timeout_machine.step(1'000, {2.0, 2.0}, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == co::CenterOutState::center_failure_dwell);
    CHECK(ep::build_center_out_render_plan(config, style, result.snapshot, {}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(find_target(plan, config.layout.center.id)->visual == ep::CenterOutVisualState::failure);
    return 0;
}

int test_visual_geometry_cannot_change_semantics()
{
    const auto config = make_config();
    co::CenterOutMachine small_style_machine{};
    co::CenterOutMachine large_style_machine{};
    const auto snapshot = start_snapshot(config, small_style_machine);
    const auto matching_snapshot = start_snapshot(config, large_style_machine);
    CHECK(snapshot.state == matching_snapshot.state);
    CHECK(snapshot.outward_target == matching_snapshot.outward_target);
    auto small = make_style();
    auto large = small;
    small.target_radius = 0.01;
    small.cursor_radius = 0.005;
    large.target_radius = 0.4;
    large.cursor_radius = 0.3;

    ep::CenterOutRenderPlan small_plan{};
    ep::CenterOutRenderPlan large_plan{};
    const co::WorkspacePoint semantic_cursor{0.09, 0.0};
    CHECK(ep::build_center_out_render_plan(config, small, snapshot, semantic_cursor, small_plan) ==
          ep::SurfaceStatus::ok);
    CHECK(ep::build_center_out_render_plan(config, large, snapshot, semantic_cursor, large_plan) ==
          ep::SurfaceStatus::ok);
    CHECK(near(small_plan.circles[0].radius, 0.01));
    CHECK(near(large_plan.circles[0].radius, 0.4));
    CHECK(near(small_plan.circles[small_plan.n_circles - 1].radius, 0.005));
    CHECK(near(large_plan.circles[large_plan.n_circles - 1].radius, 0.3));

    bool inside{};
    CHECK(co::contains_cursor(config.acceptance, config.cursor, config.layout.center.pos,
                              semantic_cursor, inside) == ContractStatus::ok);
    CHECK(inside);

    co::CenterOutStepResult small_result{};
    co::CenterOutStepResult large_result{};
    CHECK(small_style_machine.step(10, semantic_cursor, small_result) == ContractStatus::ok);
    CHECK(large_style_machine.step(10, semantic_cursor, large_result) == ContractStatus::ok);
    CHECK(small_result.snapshot.state == large_result.snapshot.state);
    CHECK(small_result.snapshot.contained == large_result.snapshot.contained);
    CHECK(small_result.snapshot.active_target == large_result.snapshot.active_target);

    ep::CenterOutRenderPlan unchanged = small_plan;
    auto inconsistent = snapshot;
    inconsistent.active_target = 999;
    CHECK(ep::build_center_out_render_plan(config, small, inconsistent, semantic_cursor,
                                           unchanged) == ep::SurfaceStatus::invalid_configuration);
    CHECK(unchanged.n_circles == small_plan.n_circles);
    CHECK(unchanged.circles[0].target_id == small_plan.circles[0].target_id);
    return 0;
}

int test_boundaries_and_control_events()
{
    auto config = make_config();
    config.layout.surrounding[0].pos = {1.25, 1.0};
    co::CenterOutMachine machine{};
    const auto snapshot = start_snapshot(config, machine);
    ep::CenterOutRenderPlan plan{};
    CHECK(ep::build_center_out_render_plan(config, make_style(), snapshot, {-1.0, 1.0}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.n_circles == config.layout.count + 2);
    CHECK(near(find_target(plan, config.layout.surrounding[0].id)->center.x, 1.25));
    CHECK(near(plan.circles[plan.n_circles - 1].center.x, -1.0));

    ep::CoordinateMapper mapper{};
    CHECK(mapper.configure({-1.0, -1.0, 2.0, 2.0}, {320, 240}, {640, 480},
                           ep::AspectPolicy::fit_letterbox) == ep::SurfaceStatus::ok);
    const auto offscreen = mapper.logical_to_framebuffer({1.25, 1.0});
    const auto boundary = mapper.logical_to_framebuffer({1.0, 1.0});
    CHECK(offscreen.x > boundary.x);

    ep::CenterOutPresentationControlEvent control{};
    const ep::InputEvent escape{.ordinal = 4,
                                .renderer_time_ns = 50,
                                .experiment_time_ns = 70,
                                .kind = ep::InputEventKind::key,
                                .action = ep::InputAction::press,
                                .code = kGlfwKeyEscape};
    CHECK(ep::center_out_control_event(escape, control));
    CHECK(control.kind == ep::CenterOutPresentationControlKind::escape_requested);
    CHECK(control.input_ordinal == 4);
    CHECK(control.experiment_time_ns == 70);
    auto ignored = escape;
    ignored.action = ep::InputAction::repeat;
    CHECK(!ep::center_out_control_event(ignored, control));
    const ep::InputEvent close{.ordinal = 5,
                               .renderer_time_ns = 51,
                               .experiment_time_ns = 71,
                               .kind = ep::InputEventKind::window_close};
    CHECK(ep::center_out_control_event(close, control));
    CHECK(control.kind == ep::CenterOutPresentationControlKind::window_close_requested);
    return 0;
}

int test_geometry_unit_contract()
{
    // A millimetre task is a first-class configuration exercised by the
    // experiment-contract tests. The presentation contract must carry the same unit, and the render
    // plan's workspace positions must map inside the letterboxed viewport
    // rather than be silently clipped away -- the failure mode a normalized
    // default produced before this contract was frozen.
    const auto task = make_mm_config();
    CHECK(task.geometry_unit == co::GeometryUnit::millimetres);

    ep::CenterOutPresentationConfig presentation{
        .geometry_unit = co::GeometryUnit::millimetres,
        .logical_space = {-125.0, -125.0, 250.0, 250.0},
        .style = {.target_radius = 5.0, .cursor_radius = 2.0, .circle_segments = 24},
        .window_size = {320, 240},
        .aspect_policy = ep::AspectPolicy::fit_letterbox,
        .swap_interval = 0,
        .input_capacity = 8,
        .visible = false,
    };
    CHECK(ep::validate(presentation) == ep::SurfaceStatus::ok);
    CHECK(ep::validate_center_out_presentation(task, presentation) == ep::SurfaceStatus::ok);
    auto impossible_cursor = presentation;
    impossible_cursor.style.cursor_radius = 126.0;
    CHECK(ep::validate(impossible_cursor) == ep::SurfaceStatus::invalid_configuration);

    // The frozen boundary: a normalized presentation with a millimetre task is
    // an explicit configuration error, not a silent rescale. Both sides are
    // individually valid; only the unit match fails.
    ep::CenterOutPresentationConfig normalized{
        .geometry_unit = co::GeometryUnit::normalized,
        .logical_space = {-1.0, -1.0, 2.0, 2.0},
        .style = {.target_radius = 0.06, .cursor_radius = 0.025, .circle_segments = 24},
        .window_size = {320, 240},
        .swap_interval = 0,
        .input_capacity = 8,
        .visible = false,
    };
    CHECK(ep::validate(normalized) == ep::SurfaceStatus::ok);
    CHECK(ep::validate_center_out_presentation(task, normalized) ==
          ep::SurfaceStatus::invalid_configuration);
    // An unspecified presentation unit is rejected outright, so a
    // default-constructed contract cannot reach a surface.
    ep::CenterOutPresentationConfig unset = normalized;
    unset.geometry_unit = co::GeometryUnit::unspecified;
    CHECK(ep::validate(unset) == ep::SurfaceStatus::invalid_configuration);
    // Radii are no longer defaulted: a default-constructed style has no radii and
    // is rejected, so the caller must state them in the task's unit.
    CHECK(ep::validate(ep::CenterOutPresentationStyle{}) ==
          ep::SurfaceStatus::invalid_configuration);
    // The preflight validates the whole logical space and the aspect policy --
    // not only width/height -- so a NaN origin or an undeclared policy is caught
    // here rather than only later in CoordinateMapper::configure().
    ep::CenterOutPresentationConfig nan_origin = presentation;
    nan_origin.logical_space.left = std::nan("");
    CHECK(ep::validate(nan_origin) == ep::SurfaceStatus::invalid_configuration);
    ep::CenterOutPresentationConfig bad_aspect = presentation;
    bad_aspect.aspect_policy = static_cast<ep::AspectPolicy>(99);
    CHECK(ep::validate(bad_aspect) == ep::SurfaceStatus::invalid_configuration);

    // Map the millimetre render plan through a letterboxed framebuffer: the
    // centre is visible, every outward target lands inside the viewport, the
    // visual radii are the configured millimetre values, and an offscreen
    // coordinate is carried through the plan unclamped -- rendering clips the
    // framebuffer coordinate, never the workspace position the experiment owns.
    co::CenterOutMachine machine{};
    const auto snapshot = start_snapshot(task, machine);
    ep::CenterOutRenderPlan plan{};
    CHECK(ep::build_center_out_render_plan(task, presentation.style, snapshot, {0.0, 0.0}, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(near(plan.circles[0].center.x, task.layout.center.pos.x));
    CHECK(near(plan.circles[0].radius, 5.0));
    CHECK(near(plan.circles[plan.n_circles - 1].radius, 2.0));
    ep::CenterOutRenderPlan offscreen_plan{};
    CHECK(ep::build_center_out_render_plan(task, presentation.style, snapshot, {200.0, 0.0},
                                           offscreen_plan) == ep::SurfaceStatus::ok);
    CHECK(near(offscreen_plan.circles[offscreen_plan.n_circles - 1].center.x, 200.0));

    ep::CoordinateMapper mapper{};
    CHECK(mapper.configure(presentation.logical_space, {320, 240}, {640, 480},
                           ep::AspectPolicy::fit_letterbox) == ep::SurfaceStatus::ok);
    const auto viewport = mapper.viewport();
    const auto inside = [&](co::WorkspacePoint point)
    {
        // The experiment's WorkspacePoint and the presentation layer's Point2d
        // are distinct types with no implicit conversion; the explicit
        // construction marks the boundary between semantic geometry and the
        // rendering coordinate.
        const auto fb = mapper.logical_to_framebuffer({point.x, point.y});
        return fb.x >= viewport.left && fb.x < viewport.left + viewport.width &&
               fb.y >= viewport.top && fb.y < viewport.top + viewport.height;
    };
    CHECK(inside(task.layout.center.pos));
    for (std::uint8_t i = 0; i < task.layout.count; ++i)
    {
        CHECK(inside(task.layout.surrounding[i].pos));
    }
    const auto offscreen = mapper.logical_to_framebuffer({200.0, 0.0});
    CHECK(offscreen.x > viewport.left + viewport.width);
    return 0;
}

int test_window()
{
    const auto config = make_config();
    const auto style = make_style();
    co::CenterOutMachine machine{};
    const auto snapshot = start_snapshot(config, machine);
    ep::CenterOut2DPresenter presenter{};
    const auto origin = ep::renderer_monotonic_now_ns();
    auto presentation = make_presentation();
    auto status = presenter.open(presentation, origin, 10'000);
    if (status == ep::SurfaceStatus::glfw_initialization_failed ||
        status == ep::SurfaceStatus::window_creation_failed ||
        status == ep::SurfaceStatus::monitor_unavailable ||
        status == ep::SurfaceStatus::opengl_function_missing)
    {
        return neurale::presentation_test::skip_or_fail(ep::surface_status_name(status));
    }
    CHECK(status == ep::SurfaceStatus::ok);
    CHECK(presenter.render(1, 2).status == ep::SurfaceStatus::invalid_state);
    CHECK(presenter.prepare(config) == ep::SurfaceStatus::ok);

    CHECK(presenter.update(snapshot, {}) == ep::SurfaceStatus::ok);
    const auto baseline = allocations.load(std::memory_order_relaxed);
    for (int update = 1; update <= 128; ++update)
    {
        CHECK(presenter.update(snapshot, {0.001 * update, -0.002 * update}) ==
              ep::SurfaceStatus::ok);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == baseline);
    CHECK(presenter.latest_update_ordinal() == 129);
    const auto& latest = presenter.latest_render_plan();
    CHECK(near(latest.circles[latest.n_circles - 1].center.x, 0.128));
    CHECK(near(latest.circles[latest.n_circles - 1].center.y, -0.256));

    const auto presented = presenter.render(20'000, 21'000);
    CHECK(presented.status == ep::SurfaceStatus::ok);
    CHECK(presented.requested_ns == 20'000);
    CHECK(presented.intended_ns == 21'000);
    CHECK(presenter.rendered_update_ordinal() == presenter.latest_update_ordinal());
    const auto resources = presenter.resource_stats();
    CHECK(resources.vertex_capacity >= ep::kMaxCenterOutRenderCircles * style.circle_segments * 3);
    CHECK(resources.batch_capacity >= ep::kMaxCenterOutRenderCircles);

    CHECK(presenter.request_window_size({400, 300}) == ep::SurfaceStatus::ok);
    CHECK(presenter.pump_events() == ep::SurfaceStatus::ok);
    CHECK(presenter.coordinates().window_size().width > 0);
    presenter.cancel();
    CHECK(presenter.pump_events() == ep::SurfaceStatus::cancelled);
    presenter.cancel();
    presenter.close();
    presenter.close();
    CHECK(presenter.lifecycle() == ep::SurfaceLifecycle::closed);
    CHECK(!presenter.resource_stats().window_open);

    if (std::getenv("NEURALE_PRESENTATION_TEST_FULLSCREEN") != nullptr)
    {
        ep::CenterOut2DPresenter fullscreen{};
        auto fullscreen_presentation = make_presentation();
        fullscreen_presentation.title = "PyNeurale Center-Out fullscreen test";
        fullscreen_presentation.input_capacity = 8;
        fullscreen_presentation.fullscreen = true;
        fullscreen_presentation.resizable = false;
        CHECK(fullscreen.open(fullscreen_presentation, ep::renderer_monotonic_now_ns(), 0) ==
              ep::SurfaceStatus::ok);
        CHECK(fullscreen.prepare(config) == ep::SurfaceStatus::ok);
        CHECK(fullscreen.update(snapshot, {}) == ep::SurfaceStatus::ok);
        CHECK(fullscreen.render(1, 1).status == ep::SurfaceStatus::ok);
        fullscreen.close();
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (const auto result = test_render_plan_geometry_and_styles(); result != 0)
    {
        return result;
    }
    if (const auto result = test_visual_geometry_cannot_change_semantics(); result != 0)
    {
        return result;
    }
    if (const auto result = test_boundaries_and_control_events(); result != 0)
    {
        return result;
    }
    if (const auto result = test_geometry_unit_contract(); result != 0)
    {
        return result;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--window")
    {
        return test_window();
    }
    return 0;
}
