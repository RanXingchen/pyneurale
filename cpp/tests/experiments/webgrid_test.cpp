/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/webgrid.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <type_traits>

#include "allocation_counter.h"
#include "check_counts.h"
#include <span>

namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::webgrid;

static_assert(std::is_trivially_copyable_v<WebGridConfig>);
static_assert(std::is_standard_layout_v<WebGridConfig>);

int failures = 0;

bool close(double left, double right) noexcept
{
    return std::fabs(left - right) <= 1e-12 * (1.0 + std::fabs(right));
}

WebGridConfig seeded_config() noexcept
{
    WebGridConfig config{};
    config.rows = 2;
    config.columns = 3;
    config.bounds = TaskBounds{-1.0, 1.0, 0.0, 2.0};
    config.n_candidates = 6;
    for (TargetId id = 1; id <= 6; ++id)
        config.candidates[id - 1] = id;
    config.schedule = TargetScheduleKind::seeded;
    config.immediate_repetition = ImmediateRepetitionPolicy::forbid;
    config.correct_selection = CorrectSelectionPolicy::advance_target;
    config.incorrect_selection = IncorrectSelectionPolicy::keep_current_target;
    config.seed = 918273;
    config.sampler_version = kCurrentSamplerVersion;
    config.session_duration_ns = 100;
    config.target_count_limit = 16;
    return config;
}

void test_geometry_and_boundaries()
{
    const WebGridConfig config = seeded_config();
    CHECK(validate(config) == ContractStatus::ok);

    GridCell cell{};
    CHECK(grid_cell(config, 0, 0, cell) == ContractStatus::ok);
    CHECK(cell.id == 1 && cell.row == 0 && cell.column == 0);
    CHECK(close(cell.bounds.min_x, -1.0));
    CHECK(close(cell.bounds.max_x, -1.0 / 3.0));
    CHECK(close(cell.bounds.min_y, 0.0));
    CHECK(close(cell.bounds.max_y, 1.0));

    CHECK(grid_cell(config, TargetId{6}, cell) == ContractStatus::ok);
    CHECK(cell.row == 1 && cell.column == 2);
    CHECK(close(cell.bounds.min_x, 1.0 / 3.0));
    CHECK(close(cell.bounds.max_x, 1.0));
    CHECK(close(cell.bounds.min_y, 1.0));
    CHECK(close(cell.bounds.max_y, 2.0));

    TargetId id = 99;
    CHECK(locate_cell(config, PointerPosition{-1.0, 0.0}, id) == ContractStatus::ok && id == 1);
    CHECK(locate_cell(config, PointerPosition{-1.0 / 3.0, 0.5}, id) == ContractStatus::ok &&
          id == 2);
    CHECK(locate_cell(config, PointerPosition{1.0 / 3.0, 1.0}, id) == ContractStatus::ok &&
          id == 6);
    CHECK(locate_cell(config, PointerPosition{1.0, 1.0}, id) == ContractStatus::ok &&
          id == kUnsetTargetId);
    CHECK(locate_cell(config, PointerPosition{0.0, 2.0}, id) == ContractStatus::ok &&
          id == kUnsetTargetId);
    CHECK(locate_cell(config, PointerPosition{-1.01, 0.5}, id) == ContractStatus::ok &&
          id == kUnsetTargetId);
    CHECK(locate_cell(config, PointerPosition{0.0, -0.01}, id) == ContractStatus::ok &&
          id == kUnsetTargetId);
}

void test_selectable_subset_and_selection_events()
{
    WebGridConfig config = seeded_config();
    config.n_candidates = 2;
    config.candidates = {};
    config.candidates[0] = 2;
    config.candidates[1] = 5;
    CHECK(validate(config) == ContractStatus::ok);

    TargetId id = 99;
    CHECK(locate_cell(config, PointerPosition{-0.8, 0.5}, id) == ContractStatus::ok && id == 1);
    CHECK(locate_selectable_cell(config, PointerPosition{-0.8, 0.5}, id) == ContractStatus::ok &&
          id == kUnsetTargetId);
    CHECK(locate_selectable_cell(config, PointerPosition{0.0, 0.5}, id) == ContractStatus::ok &&
          id == 2);

    const TrialIdentity trial{3, 19, 1, 2, kUnsetStimulusId};
    SelectionEvent event{};
    CHECK(make_selection_event(config, PointerPosition{0.0, 0.5}, 2, trial, 7, 400, 11, event) ==
          ContractStatus::ok);
    CHECK(event.correct && event.selected_id == 2 && event.intended_id == 2);
    CHECK(event.kind == SelectionKind::discrete && event.dwell_ns == 0);
    CHECK(event.time_ns == 400 && event.sequence == 11 && event.paradigm == 7);

    CHECK(make_selection_event(config, PointerPosition{0.0, 1.5}, 2, trial, 7, 401, 12, event) ==
          ContractStatus::ok);
    CHECK(!event.correct && event.selected_id == 5 && event.intended_id == 2);
    CHECK(config.incorrect_selection == IncorrectSelectionPolicy::keep_current_target);

    CHECK(make_selection_event(config, PointerPosition{-0.8, 0.5}, 2, trial, 7, 402, 13, event) ==
          ContractStatus::ok);
    CHECK(!event.correct && event.selected_id == kUnsetTargetId);
}

void test_target_schedules_and_limits()
{
    WebGridConfig first = seeded_config();
    first.initial_target = 4;
    WebGridConfig reset = first;
    std::array<TargetId, 16> sequence{};
    TargetId previous = kUnsetTargetId;
    for (TrialOrdinal ordinal = 0; ordinal < sequence.size(); ++ordinal)
    {
        CHECK(select_target(first, ordinal, previous, sequence[ordinal]) == ContractStatus::ok);
        CHECK(sequence[ordinal] != kUnsetTargetId);
        if (ordinal != 0)
            CHECK(sequence[ordinal] != sequence[ordinal - 1]);
        previous = sequence[ordinal];
    }
    previous = kUnsetTargetId;
    for (TrialOrdinal ordinal = 0; ordinal < sequence.size(); ++ordinal)
    {
        TargetId target{};
        CHECK(select_target(reset, ordinal, previous, target) == ContractStatus::ok);
        CHECK(target == sequence[ordinal]);
        previous = target;
    }
    TargetId terminal = 9;
    CHECK(select_target(first, 16, previous, terminal) == ContractStatus::ok &&
          terminal == kUnsetTargetId);

    WebGridConfig explicit_config = seeded_config();
    explicit_config.schedule = TargetScheduleKind::explicit_sequence;
    explicit_config.n_explicit = 3;
    explicit_config.explicit_targets[0] = 3;
    explicit_config.explicit_targets[1] = 6;
    explicit_config.explicit_targets[2] = 2;
    explicit_config.initial_target = 3;
    explicit_config.target_count_limit = 0;
    CHECK(validate(explicit_config) == ContractStatus::ok);
    const std::array<TargetId, 3> expected{3, 6, 2};
    for (TrialOrdinal ordinal = 0; ordinal < expected.size(); ++ordinal)
    {
        TargetId target{};
        CHECK(select_target(explicit_config, ordinal, kUnsetTargetId, target) ==
              ContractStatus::ok);
        CHECK(target == expected[ordinal]);
    }
    CHECK(select_target(explicit_config, 3, 2, terminal) == ContractStatus::ok &&
          terminal == kUnsetTargetId);
    CHECK(target_limit_reached(explicit_config, 3));

    bool reached = false;
    CHECK(session_duration_reached(first, 1'000, 1'099, reached) == ContractStatus::ok && !reached);
    CHECK(session_duration_reached(first, 1'000, 1'100, reached) == ContractStatus::ok && reached);
}

void test_sampler_replay_and_reset_contract()
{
    WebGridConfig config = seeded_config();
    config.target_count_limit = 0;

    TargetId without_previous = 91;
    TargetId stale_previous = 92;
    CHECK(select_target(config, 0, kUnsetTargetId, without_previous) == ContractStatus::ok);
    CHECK(select_target(config, 0, config.candidates[0], stale_previous) == ContractStatus::ok);
    CHECK(stale_previous == without_previous);

    TargetId unchanged = 77;
    CHECK(select_target(config, 1, kUnsetTargetId, unchanged) == ContractStatus::identity_missing);
    CHECK(unchanged == 77);

    config.sampler_version = 999;
    CHECK(validate(config) == ContractStatus::ok);
    GridCell cell{};
    CHECK(grid_cell(config, 0, 0, cell) == ContractStatus::ok);
    bool reached = true;
    CHECK(session_duration_reached(config, 0, 99, reached) == ContractStatus::ok && !reached);

    config.initial_target = 4;
    TargetId target = 88;
    CHECK(select_target(config, 0, 6, target) == ContractStatus::ok && target == 4);
    target = 88;
    CHECK(select_target(config, 1, 4, target) == ContractStatus::version_unsupported);
    CHECK(target == 88);

    WebGridConfig explicit_config = config;
    explicit_config.schedule = TargetScheduleKind::explicit_sequence;
    explicit_config.n_explicit = 2;
    explicit_config.explicit_targets[0] = 4;
    explicit_config.explicit_targets[1] = 2;
    CHECK(validate(explicit_config) == ContractStatus::ok);
    target = 88;
    CHECK(select_target(explicit_config, 1, 4, target) == ContractStatus::ok && target == 2);

    config = seeded_config();
    config.sampler_version = 0;
    CHECK(validate(config) == ContractStatus::identity_missing);
}

void test_fingerprint_covers_schedule_contract()
{
    const WebGridConfig config = seeded_config();
    const std::uint64_t digest = configuration_fingerprint(config);
    CHECK(digest == configuration_fingerprint(seeded_config()));
    const auto differs = [digest](const WebGridConfig& changed)
    { return configuration_fingerprint(changed) != digest; };

    WebGridConfig changed = config;
    changed.rows = 3;
    CHECK(differs(changed));
    changed = config;
    changed.columns = 4;
    CHECK(differs(changed));
    changed = config;
    changed.bounds.min_x -= 1.0;
    CHECK(differs(changed));
    changed = config;
    changed.bounds.max_x += 1.0;
    CHECK(differs(changed));
    changed = config;
    changed.bounds.min_y += 0.25;
    CHECK(differs(changed));
    changed = config;
    changed.bounds.max_y += 1.0;
    CHECK(differs(changed));
    changed = config;
    changed.n_candidates = 5;
    CHECK(differs(changed));
    changed = config;
    const TargetId held = changed.candidates[0];
    changed.candidates[0] = changed.candidates[1];
    changed.candidates[1] = held;
    CHECK(differs(changed));
    changed = config;
    changed.schedule = TargetScheduleKind::explicit_sequence;
    CHECK(differs(changed));
    changed = config;
    changed.immediate_repetition = ImmediateRepetitionPolicy::allow;
    CHECK(differs(changed));
    changed = config;
    changed.correct_selection = CorrectSelectionPolicy::unspecified;
    CHECK(differs(changed));
    changed = config;
    changed.incorrect_selection = IncorrectSelectionPolicy::unspecified;
    CHECK(differs(changed));
    changed = config;
    changed.seed += 1;
    CHECK(differs(changed));
    changed = config;
    changed.sampler_version += 1;
    CHECK(differs(changed));
    changed = config;
    changed.n_explicit = 1;
    CHECK(differs(changed));
    changed = config;
    changed.explicit_targets[0] = 3;
    CHECK(differs(changed));
    changed = config;
    changed.initial_target = 3;
    CHECK(differs(changed));
    changed = config;
    changed.session_duration_ns += 1;
    CHECK(differs(changed));
    changed = config;
    changed.target_count_limit += 1;
    CHECK(differs(changed));
    changed = config;
    changed.metric_version += 1;
    CHECK(differs(changed));

    WebGridConfig negative_zero = config;
    negative_zero.bounds.min_y = -0.0;
    WebGridConfig positive_zero = config;
    positive_zero.bounds.min_y = 0.0;
    CHECK(configuration_fingerprint(negative_zero) == configuration_fingerprint(positive_zero));
}

void test_maximum_capacity_geometry_and_schedules()
{
    WebGridConfig config = seeded_config();
    config.rows = 16;
    config.columns = 16;
    config.bounds = TaskBounds{0.0, 16.0, 0.0, 16.0};
    config.n_candidates = static_cast<std::uint16_t>(kMaxWebGridCells);
    config.candidates = {};
    config.target_count_limit = 0;
    for (std::size_t i = 0; i < kMaxWebGridCells; ++i)
        config.candidates[i] = static_cast<TargetId>(i + 1);
    CHECK(validate(config) == ContractStatus::ok);

    GridCell last{};
    CHECK(grid_cell(config, TargetId{256}, last) == ContractStatus::ok);
    CHECK(last.row == 15 && last.column == 15);
    CHECK(close(last.bounds.min_x, 15.0) && close(last.bounds.max_x, 16.0));
    TargetId id = kUnsetTargetId;
    CHECK(locate_cell(config, PointerPosition{15.999, 15.999}, id) == ContractStatus::ok &&
          id == 256);
    CHECK(locate_cell(config, PointerPosition{16.0, 15.999}, id) == ContractStatus::ok &&
          id == kUnsetTargetId);
    TargetId sampled = kUnsetTargetId;
    CHECK(select_target(config, 0, kUnsetTargetId, sampled) == ContractStatus::ok);
    CHECK(sampled >= 1 && sampled <= 256);

    config.schedule = TargetScheduleKind::explicit_sequence;
    config.n_explicit = static_cast<std::uint16_t>(kMaxWebGridCells);
    for (std::size_t i = 0; i < kMaxWebGridCells; ++i)
        config.explicit_targets[i] = static_cast<TargetId>(i + 1);
    CHECK(validate(config) == ContractStatus::ok);
    TargetId explicit_target = kUnsetTargetId;
    CHECK(select_target(config, 255, 255, explicit_target) == ContractStatus::ok &&
          explicit_target == 256);
    CHECK(select_target(config, 256, 256, explicit_target) == ContractStatus::ok &&
          explicit_target == kUnsetTargetId);
}

void test_invalid_configuration()
{
    WebGridConfig config = seeded_config();
    config.rows = 0;
    CHECK(validate(config) == ContractStatus::dimension_invalid);
    config = seeded_config();
    config.rows = 17;
    config.columns = 16;
    CHECK(validate(config) == ContractStatus::dimension_invalid);
    config = seeded_config();
    config.bounds.max_x = config.bounds.min_x;
    CHECK(validate(config) == ContractStatus::parameter_out_of_range);
    config = seeded_config();
    config.n_candidates = 0;
    CHECK(validate(config) == ContractStatus::target_set_invalid);
    config = seeded_config();
    config.candidates[1] = config.candidates[0];
    CHECK(validate(config) == ContractStatus::target_set_invalid);
    config = seeded_config();
    config.candidates[0] = 7;
    CHECK(validate(config) == ContractStatus::target_set_invalid);
    config = seeded_config();
    config.candidates[6] = 1;
    CHECK(validate(config) == ContractStatus::target_set_invalid);
    config = seeded_config();
    config.schedule = TargetScheduleKind::explicit_sequence;
    config.n_explicit = 2;
    config.explicit_targets[0] = 1;
    config.explicit_targets[1] = 1;
    CHECK(validate(config) == ContractStatus::target_set_invalid);
    config = seeded_config();
    config.n_candidates = 1;
    config.candidates = {};
    config.candidates[0] = 1;
    config.target_count_limit = 2;
    CHECK(validate(config) == ContractStatus::target_set_invalid);
    config = seeded_config();
    config.metric_version = 0;
    CHECK(validate(config) == ContractStatus::identity_missing);
}

void test_native_path_allocates_nothing()
{
    const WebGridConfig config = seeded_config();
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);
    GridCell cell{};
    TargetId id{};
    bool reached{};
    SelectionEvent event{};
    const TrialIdentity trial{0, 1, 0, 1, kUnsetStimulusId};
    CHECK(validate(config) == ContractStatus::ok);
    CHECK(grid_cell(config, 0, 0, cell) == ContractStatus::ok);
    CHECK(locate_selectable_cell(config, PointerPosition{-0.9, 0.1}, id) == ContractStatus::ok);
    CHECK(select_target(config, 0, kUnsetTargetId, id) == ContractStatus::ok);
    CHECK(session_duration_reached(config, 0, 99, reached) == ContractStatus::ok);
    CHECK(make_selection_event(config, PointerPosition{-0.9, 0.1}, 1, trial, 1, 1, 1, event) ==
          ContractStatus::ok);
    CHECK(configuration_fingerprint(config) != 0);
    CHECK(allocations.load(std::memory_order_relaxed) == baseline);
}

} // namespace

namespace machine_tests
{
namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::webgrid;

static_assert(std::is_trivially_copyable_v<WebGridMachine>);
static_assert(std::is_trivially_copyable_v<WebGridSelectionRecord>);
static_assert(std::is_trivially_copyable_v<WebGridTrial>);
static_assert(std::is_trivially_copyable_v<WebGridMetrics>);
static_assert(std::is_trivially_copyable_v<WebGridStepResult>);

constexpr ParadigmId kParadigm = 17;
constexpr ExperimentTimeNs kSecond = 1'000'000'000;
int failures = 0;

bool close(double left, double right) noexcept
{
    return std::fabs(left - right) <= 1e-12 * (1.0 + std::fabs(right));
}

WebGridConfig explicit_config(TrialOrdinal limit = 2) noexcept
{
    WebGridConfig config{};
    config.rows = 1;
    config.columns = 2;
    config.bounds = TaskBounds{0.0, 2.0, 0.0, 1.0};
    config.n_candidates = 2;
    config.candidates[0] = 1;
    config.candidates[1] = 2;
    config.schedule = TargetScheduleKind::explicit_sequence;
    config.immediate_repetition = ImmediateRepetitionPolicy::forbid;
    config.correct_selection = CorrectSelectionPolicy::advance_target;
    config.incorrect_selection = IncorrectSelectionPolicy::keep_current_target;
    config.n_explicit = 3;
    config.explicit_targets[0] = 1;
    config.explicit_targets[1] = 2;
    config.explicit_targets[2] = 1;
    config.initial_target = 1;
    config.target_count_limit = limit;
    config.metric_version = kMetricVersion1;
    return config;
}

SelectionEvent selection_for(const WebGridMachine& machine, PointerPosition pointer,
                             ExperimentTimeNs time_ns, SequenceOrdinal sequence) noexcept
{
    SelectionEvent event{};
    const WebGridSnapshot snapshot = machine.snapshot();
    const ContractStatus status =
        make_selection_event(machine.configuration(), pointer, snapshot.active_target,
                             snapshot.trial, machine.paradigm(), time_ns, sequence, event);
    CHECK(status == ContractStatus::ok);
    return event;
}

WebGridSelectionRecord raw_selection(TrialIdentity trial, ExperimentTimeNs target_onset_ns,
                                     ExperimentTimeNs time_ns, SequenceOrdinal sequence,
                                     bool correct, TargetId selected_id) noexcept
{
    SelectionEvent event{};
    event.time_ns = time_ns;
    event.sequence = sequence;
    event.trial = trial;
    event.paradigm = kParadigm;
    event.kind = SelectionKind::discrete;
    event.correct = correct;
    event.selected_id = selected_id;
    event.intended_id = trial.target_id;
    return WebGridSelectionRecord{event, target_onset_ns, time_ns - target_onset_ns};
}

void test_selection_transitions_and_metrics()
{
    const WebGridConfig config = explicit_config();
    WebGridMachine machine{};
    WebGridStepResult result{};
    CHECK(machine.state() == WebGridState::idle);
    CHECK(machine.step(0, PointerPosition{0.5, 0.5}, result) == ContractStatus::not_running);
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == WebGridState::active_target);
    CHECK(result.snapshot.active_target == 1);
    CHECK(result.snapshot.target_onset_ns == 0);
    CHECK(!result.snapshot.metrics.rates_defined);
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::already_running);

    // x == 1 is the half-open boundary and therefore belongs to cell 2.
    const PointerPosition cell_two{1.0, 0.5};
    const SelectionEvent wrong = selection_for(machine, cell_two, 10 * kSecond, 1);
    CHECK(!wrong.correct && wrong.selected_id == 2);
    CHECK(machine.step(10 * kSecond, cell_two, wrong, result) == ContractStatus::ok);
    CHECK(result.selection_processed && !result.selection.event.correct);
    CHECK(!result.trial_decided);
    CHECK(result.snapshot.active_target == 1);
    CHECK(result.snapshot.target_onset_ns == 0);
    CHECK(result.snapshot.metrics.incorrect_selections == 1);

    // A repeated event identity is rejected transactionally.
    WebGridStepResult unchanged{};
    unchanged.selection_processed = true;
    CHECK(machine.step(10 * kSecond, cell_two, wrong, unchanged) ==
          ContractStatus::outcome_invalid);
    CHECK(unchanged.selection_processed);
    CHECK(machine.snapshot().metrics.incorrect_selections == 1);

    std::array<WebGridSelectionRecord, 3> accepted{};
    accepted[0] = result.selection;
    const PointerPosition cell_one{0.5, 0.5};
    const SelectionEvent first = selection_for(machine, cell_one, 20 * kSecond, 2);
    CHECK(machine.step(20 * kSecond, cell_one, first, result) == ContractStatus::ok);
    accepted[1] = result.selection;
    CHECK(result.trial_decided && result.trial.selection.event.correct);
    CHECK(result.trial.acquisition_ns == 20 * kSecond);
    CHECK(result.trial.record.interval.start_ns == 0);
    CHECK(result.trial.record.interval.end_ns == 20 * kSecond);
    CHECK(result.snapshot.active_target == 2);
    CHECK(result.snapshot.target_onset_ns == 20 * kSecond);

    const SelectionEvent second = selection_for(machine, cell_two, 50 * kSecond, 3);
    CHECK(machine.step(50 * kSecond, cell_two, second, result) == ContractStatus::ok);
    accepted[2] = result.selection;
    CHECK(result.trial_decided && result.trial.acquisition_ns == 30 * kSecond);
    CHECK(result.snapshot.state == WebGridState::complete);
    CHECK(result.snapshot.active_target == kUnsetTargetId);
    CHECK(machine.complete());

    const WebGridMetrics& metrics = result.snapshot.metrics;
    CHECK(metrics.correct_selections == 2 && metrics.incorrect_selections == 1);
    CHECK(metrics.elapsed_active_ns == 50 * kSecond && metrics.rates_defined);
    CHECK(close(metrics.correct_targets_per_minute, 2.4));
    CHECK(close(metrics.net_correct_targets_per_minute, 1.2));
    CHECK(metrics.n_acquisitions == 2);
    CHECK(metrics.total_acquisition_ns == 50 * kSecond);
    CHECK(metrics.minimum_acquisition_ns == 20 * kSecond);
    CHECK(metrics.maximum_acquisition_ns == 30 * kSecond);
    CHECK(metrics.mean_acquisition_ns == 25 * kSecond);

    WebGridMetrics recomputed{};
    CHECK(summarize(std::span<const WebGridSelectionRecord>(accepted), 0, 50 * kSecond,
                    kMetricVersion1, recomputed) == ContractStatus::ok);
    CHECK(recomputed.correct_selections == metrics.correct_selections);
    CHECK(recomputed.incorrect_selections == metrics.incorrect_selections);
    CHECK(recomputed.total_acquisition_ns == metrics.total_acquisition_ns);
    CHECK(close(recomputed.correct_targets_per_minute, metrics.correct_targets_per_minute));
    CHECK(close(recomputed.net_correct_targets_per_minute, metrics.net_correct_targets_per_minute));

    std::array<WebGridSelectionRecord, 2> duplicate{accepted[0], accepted[0]};
    recomputed.correct_selections = 99;
    CHECK(summarize(std::span<const WebGridSelectionRecord>(duplicate), 0, 50 * kSecond,
                    kMetricVersion1, recomputed) == ContractStatus::outcome_invalid);
    CHECK(recomputed.correct_selections == 99);
}

void test_summary_lifecycle_and_version_contract()
{
    const TrialIdentity trial_zero{0, kUnsetTrialKey, 0, 1, kUnsetStimulusId};
    const TrialIdentity trial_one{1, kUnsetTrialKey, 0, 2, kUnsetStimulusId};
    WebGridMetrics metrics{};

    const std::array<WebGridSelectionRecord, 5> valid{
        raw_selection(trial_zero, 0, 5, 1, false, 2), raw_selection(trial_zero, 0, 7, 2, false, 2),
        raw_selection(trial_zero, 0, 10, 3, true, 1), raw_selection(trial_one, 10, 12, 4, false, 1),
        raw_selection(trial_one, 10, 20, 5, true, 2),
    };
    CHECK(summarize(std::span<const WebGridSelectionRecord>(valid), 0, 20, kMetricVersion1,
                    metrics) == ContractStatus::ok);
    CHECK(metrics.metric_version == kMetricVersion1);
    CHECK(metrics.correct_selections == 2 && metrics.incorrect_selections == 3);
    CHECK(metrics.total_acquisition_ns == 20 && metrics.mean_acquisition_ns == 10);

    const std::array<WebGridSelectionRecord, 2> two_correct_same_trial{
        raw_selection(trial_zero, 0, 10, 1, true, 1),
        raw_selection(trial_zero, 0, 20, 2, true, 1),
    };
    CHECK(summarize(std::span<const WebGridSelectionRecord>(two_correct_same_trial), 0, 20,
                    kMetricVersion1, metrics) == ContractStatus::outcome_invalid);

    const std::array<WebGridSelectionRecord, 2> changed_onset{
        raw_selection(trial_zero, 0, 5, 1, false, 2),
        raw_selection(trial_zero, 7, 10, 2, true, 1),
    };
    CHECK(summarize(std::span<const WebGridSelectionRecord>(changed_onset), 0, 10, kMetricVersion1,
                    metrics) == ContractStatus::outcome_invalid);

    std::array<WebGridSelectionRecord, 2> skipped_ordinal{
        raw_selection(trial_zero, 0, 10, 1, true, 1),
        raw_selection(TrialIdentity{2, kUnsetTrialKey, 0, 2, kUnsetStimulusId}, 10, 20, 2, true, 2),
    };
    CHECK(summarize(std::span<const WebGridSelectionRecord>(skipped_ordinal), 0, 20,
                    kMetricVersion1, metrics) == ContractStatus::outcome_invalid);

    metrics.correct_selections = 99;
    CHECK(summarize(std::span<const WebGridSelectionRecord>(valid), 0, 20, 999, metrics) ==
          ContractStatus::version_unsupported);
    CHECK(metrics.correct_selections == 99);
    CHECK(summarize(std::span<const WebGridSelectionRecord>(valid), 0, 20, 0, metrics) ==
          ContractStatus::identity_missing);
    CHECK(metrics.correct_selections == 99);
}

void test_duration_and_explicit_exhaustion()
{
    WebGridConfig config = explicit_config(0);
    config.session_duration_ns = 10;
    WebGridMachine machine{};
    WebGridStepResult result{};
    CHECK(machine.start(kParadigm, config, 100, result) == ContractStatus::ok);
    CHECK(machine.step(109, PointerPosition{0.5, 0.5}, result) == ContractStatus::ok);
    const SelectionEvent at_deadline = selection_for(machine, PointerPosition{0.5, 0.5}, 110, 1);
    CHECK(machine.step(110, PointerPosition{0.5, 0.5}, at_deadline, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == WebGridState::complete);
    CHECK(!result.selection_processed && !result.trial_decided);
    CHECK(result.snapshot.metrics.elapsed_active_ns == 10);

    config = explicit_config(0);
    config.n_explicit = 1;
    config.explicit_targets[1] = 0;
    config.explicit_targets[2] = 0;
    machine.reset();
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
    const SelectionEvent only = selection_for(machine, PointerPosition{0.5, 0.5}, 1, 1);
    CHECK(machine.step(1, PointerPosition{0.5, 0.5}, only, result) == ContractStatus::ok);
    CHECK(result.snapshot.state == WebGridState::complete);
}

void test_reset_and_chunk_independence()
{
    WebGridConfig config = explicit_config(4);
    config.schedule = TargetScheduleKind::seeded;
    config.n_explicit = 0;
    config.explicit_targets = {};
    config.initial_target = kUnsetTargetId;
    config.seed = 123456;

    WebGridMachine machine{};
    std::array<TargetId, 4> first{};
    std::array<TargetId, 4> second{};
    for (int run = 0; run < 2; ++run)
    {
        WebGridStepResult result{};
        CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
        for (std::size_t ordinal = 0; ordinal < first.size(); ++ordinal)
        {
            const WebGridSnapshot before = machine.snapshot();
            (run == 0 ? first : second)[ordinal] = before.active_target;
            GridCell cell{};
            CHECK(grid_cell(config, before.active_target, cell) == ContractStatus::ok);
            const PointerPosition pointer{(cell.bounds.min_x + cell.bounds.max_x) / 2.0,
                                          (cell.bounds.min_y + cell.bounds.max_y) / 2.0};
            const ExperimentTimeNs event_time = static_cast<ExperimentTimeNs>(ordinal + 1) * 10;
            if (run == 0)
                CHECK(machine.step(event_time - 1, pointer, result) == ContractStatus::ok);
            const SelectionEvent event = selection_for(machine, pointer, event_time, ordinal + 1);
            CHECK(machine.step(event_time, pointer, event, result) == ContractStatus::ok);
        }
        CHECK(machine.complete());
        machine.reset();
        CHECK(machine.state() == WebGridState::idle);
    }
    CHECK(first == second);
}

void test_versions_validation_and_allocation()
{
    WebGridConfig config = explicit_config();
    WebGridMachine machine{};
    WebGridStepResult result{};
    config.metric_version = 0;
    CHECK(validate(config) == ContractStatus::identity_missing);
    config.metric_version = 999;
    CHECK(validate(config) == ContractStatus::ok);
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::version_unsupported);

    config = explicit_config();
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);
    CHECK(machine.start(kParadigm, config, 0, result) == ContractStatus::ok);
    const PointerPosition pointer{0.5, 0.5};
    CHECK(machine.step(1, pointer, result) == ContractStatus::ok);
    const SelectionEvent event = selection_for(machine, pointer, 2, 1);
    CHECK(machine.step(2, pointer, event, result) == ContractStatus::ok);
    CHECK(allocations.load(std::memory_order_relaxed) == baseline);
}

} // namespace

int run()
{
    test_selection_transitions_and_metrics();
    test_summary_lifecycle_and_version_contract();
    test_duration_and_explicit_exhaustion();
    test_reset_and_chunk_independence();
    test_versions_validation_and_allocation();
    if (failures != 0)
        std::cerr << failures << " WebGrid machine assertion(s) failed\n";
    return failures == 0 ? 0 : 1;
}
} // namespace machine_tests

int main()
{
    const int machine_status = machine_tests::run();
    test_geometry_and_boundaries();
    test_selectable_subset_and_selection_events();
    test_target_schedules_and_limits();
    test_sampler_replay_and_reset_contract();
    test_fingerprint_covers_schedule_contract();
    test_maximum_capacity_geometry_and_schedules();
    test_invalid_configuration();
    test_native_path_allocates_nothing();
    if (failures != 0)
        std::cerr << failures << " WebGrid assertion(s) failed\n";
    return failures == 0 && machine_status == 0 ? 0 : 1;
}
