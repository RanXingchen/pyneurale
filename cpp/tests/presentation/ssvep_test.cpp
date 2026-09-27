// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#include "check_returns.h"
#include "skip_policy.h"
#include "ssvep_presenter.h"
#include <cmath>
#include <string_view>
namespace ep = neurale::experiment_presentation;
namespace ex = neurale::experiments;
namespace ss = ex::ssvep;
int main(int argc, char** argv)
{
    ss::SSVEPConfig task{};
    task.n_targets = 4;
    task.n_trials = 1;
    task.stimulus_id = 1;
    for (int i = 0; i < 4; ++i)
        task.targets[i] = {static_cast<ex::TargetId>(i + 1), 8.0 + 2 * i};
    task.cue_duration_ns = 1'000'000'000;
    task.stimulation_duration_ns = 2'500'000'000;
    task.decision_timeout_ns = 1'000'000'000;
    task.feedback_duration_ns = 500'000'000;
    task.inter_trial_ns = 500'000'000;
    ss::SSVEPMachine machine;
    ss::SSVEPStepResult result;
    CHECK(machine.start(1, task, 0, result) == ex::ContractStatus::ok);
    ep::SSVEPRenderPlan plan;
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 0, plan) == ep::SurfaceStatus::ok);
    CHECK(plan.count == 4);
    ep::SSVEPLayout layout{};
    layout.count = 4;
    layout.target_size = 0.3;
    layout.positions[0] = {-0.6, 0.6};
    layout.positions[1] = {0.6, 0.6};
    layout.positions[2] = {-0.6, -0.6};
    layout.positions[3] = {0.6, -0.6};
    ep::SSVEPRenderPlan custom;
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 0, custom, layout) ==
          ep::SurfaceStatus::ok);
    CHECK(std::abs(custom.targets[0].bounds.left + 0.75) < 1e-12);
    CHECK(custom.targets[0].bounds.width == 0.3);
    auto invalid = layout;
    invalid.count = 3;
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 0, custom, invalid) ==
          ep::SurfaceStatus::invalid_configuration);
    invalid = layout;
    invalid.positions[0] = invalid.positions[1];
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 0, custom, invalid) ==
          ep::SurfaceStatus::invalid_configuration);
    invalid = layout;
    invalid.target_size = 0.9;
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 0, custom, invalid) ==
          ep::SurfaceStatus::invalid_configuration);

    CHECK(plan.targets[0].bounds.left < plan.targets[1].bounds.left);
    CHECK(plan.targets[0].bounds.bottom > plan.targets[2].bounds.bottom);
    for (std::size_t i = 0; i < 4; ++i)
        CHECK((plan.targets[i].outline.red == 1) ==
              (plan.targets[i].id == result.snapshot.trial.target_id));
    CHECK(machine.step(1'000'000'000, result) == ex::ContractStatus::ok);
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 1'000'000'000, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.targets[0].fill.red == 1);
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 1'062'500'000, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(std::abs(plan.targets[0].fill.red) < 1e-6);
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 1'125'000'000, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(std::abs(plan.targets[0].fill.red - 1) < 1e-6);
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 3'500'000'000, plan) ==
          ep::SurfaceStatus::presentation_deadline_missed);
    CHECK(machine.step(3'500'000'000, result) == ex::ContractStatus::ok);
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 3'500'000'000, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.count == 0);
    CHECK(machine.step(4'500'000'000, result) == ex::ContractStatus::ok);
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 4'500'000'000, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.count == 4);
    CHECK(result.snapshot.outcome == ex::TrialOutcome::timeout);
    for (std::size_t i = 0; i < plan.count; ++i)
        CHECK(!plan.targets[i].burst);
    const auto feedback_snapshot = result.snapshot;
    CHECK(machine.step(5'000'000'000, result) == ex::ContractStatus::ok);
    CHECK(ep::build_ssvep_render_plan(task, result.snapshot, 5'000'000'000, plan) ==
          ep::SurfaceStatus::ok);
    CHECK(plan.count == 0);

    ss::SSVEPMachine selected_machine;
    ss::SSVEPStepResult selected_result;
    CHECK(selected_machine.start(1, task, 0, selected_result) == ex::ContractStatus::ok);
    CHECK(selected_machine.step(1'000'000'000, selected_result) == ex::ContractStatus::ok);
    ex::SelectionEvent selection{};
    selection.time_ns = 1'500'000'000;
    selection.trial = selected_result.snapshot.trial;
    selection.paradigm = 1;
    selection.selected_id = selection.trial.target_id;
    selection.intended_id = selection.trial.target_id;
    selection.correct = true;
    CHECK(selected_machine.step(selection.time_ns, selection, selected_result) ==
          ex::ContractStatus::ok);
    CHECK(selected_machine.step(3'500'000'000, selected_result) == ex::ContractStatus::ok);
    CHECK(selected_result.snapshot.outcome == ex::TrialOutcome::success);
    CHECK(ep::build_ssvep_render_plan(task, selected_result.snapshot, 3'500'000'000, plan) ==
          ep::SurfaceStatus::ok);
    bool found_burst = false;
    for (std::size_t i = 0; i < plan.count; ++i)
    {
        const auto& target = plan.targets[i];
        if (target.id == selection.selected_id)
        {
            CHECK(target.burst);
            CHECK(target.burst_progress == 0);
            found_burst = true;
        }
        else
            CHECK(!target.burst);
    }
    CHECK(found_burst);
    CHECK(ep::build_ssvep_render_plan(task, selected_result.snapshot, 3'700'000'000, plan) ==
          ep::SurfaceStatus::ok);
    for (std::size_t i = 0; i < plan.count; ++i)
        if (plan.targets[i].burst)
            CHECK(std::abs(plan.targets[i].burst_progress - 0.5) < 1e-12);
    CHECK(ep::build_ssvep_render_plan(task, selected_result.snapshot, 3'900'000'000, plan) ==
          ep::SurfaceStatus::ok);
    for (std::size_t i = 0; i < plan.count; ++i)
        if (plan.targets[i].burst)
            CHECK(plan.targets[i].burst_progress == 1);
    const auto successful_feedback_snapshot = selected_result.snapshot;
    selected_machine.reset();
    CHECK(selected_machine.start(1, task, 0, selected_result) == ex::ContractStatus::ok);
    CHECK(selected_machine.step(1'000'000'000, selected_result) == ex::ContractStatus::ok);
    selection.trial = selected_result.snapshot.trial;
    selection.selected_id = selection.trial.target_id % 4 + 1;
    selection.intended_id = selection.trial.target_id;
    selection.correct = false;
    CHECK(selected_machine.step(selection.time_ns, selection, selected_result) ==
          ex::ContractStatus::ok);
    CHECK(selected_machine.step(3'500'000'000, selected_result) == ex::ContractStatus::ok);
    CHECK(selected_result.snapshot.outcome == ex::TrialOutcome::failure);
    CHECK(ep::build_ssvep_render_plan(task, selected_result.snapshot, 3'700'000'000, plan) ==
          ep::SurfaceStatus::ok);
    for (std::size_t i = 0; i < plan.count; ++i)
    {
        CHECK(!plan.targets[i].burst);
        if (plan.targets[i].id == selection.selected_id)
            CHECK(plan.targets[i].fill.red > plan.targets[i].fill.green);
    }
    const auto incorrect_feedback_snapshot = selected_result.snapshot;
    if (argc > 1 && std::string_view(argv[1]) == "--window")
    {
        ep::PresentationSurface probe;
        const auto probe_status = probe.open({.visible = false}, 0, 0);
        if (probe_status != ep::SurfaceStatus::ok)
            return neurale::presentation_test::skip_or_fail(ep::surface_status_name(probe_status));
        const int refresh = probe.environment().refresh_rate_hz;
        probe.close();
        ep::SSVEPConcretePresenter presenter;
        auto status = presenter.open(task, {.visible = false}, 0, 0, layout);
        if (refresh <= 2 * task.targets[task.n_targets - 1].frequency_hz)
        {
            CHECK(status == ep::SurfaceStatus::invalid_configuration);
            return 0;
        }
        if (status != ep::SurfaceStatus::ok)
            return neurale::presentation_test::skip_or_fail(ep::surface_status_name(status));
        CHECK(presenter.render(successful_feedback_snapshot, 3'700'000'000).status ==
              ep::SurfaceStatus::ok);
        CHECK(presenter.render(incorrect_feedback_snapshot, 3'800'000'000).status ==
              ep::SurfaceStatus::ok);
        CHECK(presenter.render(feedback_snapshot, 4'500'000'000).status == ep::SurfaceStatus::ok);
        CHECK(presenter.render(feedback_snapshot, 4'499'999'999).status ==
              ep::SurfaceStatus::time_regressed);
        presenter.close();
        ep::PresentationSurface input_surface;
        CHECK(input_surface.open({.visible = false}, 0, 0) == ep::SurfaceStatus::ok);
        input_surface.inject_input({.renderer_time_ns = 1,
                                    .kind = ep::InputEventKind::key,
                                    .action = ep::InputAction::press,
                                    .code = 50});
        ep::InputEvent input{};
        bool available{};
        CHECK(input_surface.poll_input(input, available) == ep::SurfaceStatus::ok);
        CHECK(!available);
        input_surface.inject_input({.renderer_time_ns = 2,
                                    .kind = ep::InputEventKind::key,
                                    .action = ep::InputAction::repeat,
                                    .code = 50});
        CHECK(input_surface.poll_input(input, available) == ep::SurfaceStatus::ok);
        CHECK(!available);
        input_surface.inject_input({.renderer_time_ns = 3,
                                    .kind = ep::InputEventKind::key,
                                    .action = ep::InputAction::press,
                                    .code = 256});
        CHECK(input_surface.poll_input(input, available) == ep::SurfaceStatus::ok);
        CHECK(available && input.code == 256);
        input_surface.close();
        auto two_targets = task;
        two_targets.n_targets = 2;
        CHECK(presenter.open(two_targets, {.visible = false}, 0, 0) == ep::SurfaceStatus::ok);
        presenter.close();
        auto high_frequency = task;
        high_frequency.targets[0].frequency_hz = 1e6;
        CHECK(presenter.open(high_frequency, {.visible = false}, 0, 0) ==
              ep::SurfaceStatus::invalid_configuration);
        CHECK(presenter.render(feedback_snapshot, 4'500'000'000).status ==
              ep::SurfaceStatus::invalid_state);
    }
    return 0;
}
