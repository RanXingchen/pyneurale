// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <neurale/experiments/webgrid.h>

namespace neurale::experiments::webgrid
{
namespace
{

[[nodiscard]] bool finite(double value) noexcept
{
    return std::isfinite(value);
}

[[nodiscard]] TargetId id_of(const WebGridConfig& config, std::uint16_t row,
                             std::uint16_t column) noexcept
{
    return static_cast<TargetId>(static_cast<std::uint32_t>(row) * config.columns + column + 1);
}

[[nodiscard]] bool padding_clear(const std::array<TargetId, kMaxWebGridCells>& values,
                                 std::size_t used) noexcept
{
    for (std::size_t i = used; i < values.size(); ++i)
        if (values[i] != kUnsetTargetId)
            return false;
    return true;
}

[[nodiscard]] bool unique_prefix(const std::array<TargetId, kMaxWebGridCells>& values,
                                 std::size_t used) noexcept
{
    for (std::size_t left = 0; left < used; ++left)
        for (std::size_t right = left + 1; right < used; ++right)
            if (values[left] == values[right])
                return false;
    return true;
}

[[nodiscard]] bool physical_cell_id(const WebGridConfig& config, TargetId id) noexcept
{
    return id != kUnsetTargetId && static_cast<std::size_t>(id) <= cell_count(config);
}

[[nodiscard]] GridCell make_cell_unchecked(const WebGridConfig& config, std::uint16_t row,
                                           std::uint16_t column) noexcept
{
    const double width = config.bounds.max_x - config.bounds.min_x;
    const double height = config.bounds.max_y - config.bounds.min_y;
    GridCell result{};
    result.id = id_of(config, row, column);
    result.row = row;
    result.column = column;
    result.bounds.min_x =
        config.bounds.min_x + width * static_cast<double>(column) / config.columns;
    result.bounds.max_x =
        column + 1 == config.columns
            ? config.bounds.max_x
            : config.bounds.min_x + width * static_cast<double>(column + 1) / config.columns;
    result.bounds.min_y = config.bounds.min_y + height * static_cast<double>(row) / config.rows;
    result.bounds.max_y =
        row + 1 == config.rows
            ? config.bounds.max_y
            : config.bounds.min_y + height * static_cast<double>(row + 1) / config.rows;
    return result;
}

void absorb_double(FingerprintAccumulator& accumulator, double value) noexcept
{
    // Geometry treats both zeros identically, so its persisted identity does too.
    if (value == 0.0)
        value = 0.0;
    accumulator.absorb(std::bit_cast<std::uint64_t>(value));
}

} // namespace

ContractStatus validate(const TaskBounds& bounds) noexcept
{
    if (!finite(bounds.min_x) || !finite(bounds.max_x) || !finite(bounds.min_y) ||
        !finite(bounds.max_y))
        return ContractStatus::value_not_finite;
    if (!(bounds.min_x < bounds.max_x) || !(bounds.min_y < bounds.max_y))
        return ContractStatus::parameter_out_of_range;
    return ContractStatus::ok;
}

ContractStatus validate(const PointerPosition& pointer) noexcept
{
    return finite(pointer.x) && finite(pointer.y) ? ContractStatus::ok
                                                  : ContractStatus::value_not_finite;
}

ContractStatus validate(const CellBounds& bounds) noexcept
{
    if (!finite(bounds.min_x) || !finite(bounds.max_x) || !finite(bounds.min_y) ||
        !finite(bounds.max_y))
        return ContractStatus::value_not_finite;
    return bounds.min_x < bounds.max_x && bounds.min_y < bounds.max_y
               ? ContractStatus::ok
               : ContractStatus::parameter_out_of_range;
}

ContractStatus validate(const GridCell& cell) noexcept
{
    if (cell.id == kUnsetTargetId)
        return ContractStatus::identity_missing;
    return validate(cell.bounds);
}

bool selectable(const WebGridConfig& config, TargetId id) noexcept
{
    const std::size_t count = config.n_candidates < config.candidates.size()
                                  ? config.n_candidates
                                  : config.candidates.size();
    for (std::size_t i = 0; i < count; ++i)
        if (config.candidates[i] == id)
            return true;
    return false;
}

ContractStatus validate(const WebGridConfig& config) noexcept
{
    if (config.rows == 0 || config.columns == 0 || cell_count(config) > kMaxWebGridCells)
        return ContractStatus::dimension_invalid;
    if (const ContractStatus status = validate(config.bounds); status != ContractStatus::ok)
        return status;
    if (!target_schedule_kind_declared(config.schedule) ||
        !immediate_repetition_policy_declared(config.immediate_repetition) ||
        !correct_selection_policy_declared(config.correct_selection) ||
        !incorrect_selection_policy_declared(config.incorrect_selection))
        return ContractStatus::enum_undeclared;
    if (config.schedule == TargetScheduleKind::unspecified ||
        config.immediate_repetition == ImmediateRepetitionPolicy::unspecified ||
        config.correct_selection != CorrectSelectionPolicy::advance_target ||
        config.incorrect_selection != IncorrectSelectionPolicy::keep_current_target)
        return ContractStatus::identity_missing;
    if (config.n_candidates == 0 || config.n_candidates > kMaxWebGridCells ||
        !padding_clear(config.candidates, config.n_candidates) ||
        !unique_prefix(config.candidates, config.n_candidates))
        return ContractStatus::target_set_invalid;
    for (std::size_t i = 0; i < config.n_candidates; ++i)
        if (!physical_cell_id(config, config.candidates[i]))
            return ContractStatus::target_set_invalid;
    if (config.initial_target != kUnsetTargetId && !selectable(config, config.initial_target))
        return ContractStatus::target_set_invalid;
    if (config.sampler_version == 0)
        return ContractStatus::identity_missing;
    if (config.metric_version == 0)
        return ContractStatus::identity_missing;

    if (config.schedule == TargetScheduleKind::explicit_sequence)
    {
        if (config.n_explicit == 0 || config.n_explicit > kMaxWebGridCells ||
            !padding_clear(config.explicit_targets, config.n_explicit))
            return ContractStatus::target_set_invalid;
        for (std::size_t i = 0; i < config.n_explicit; ++i)
        {
            if (!selectable(config, config.explicit_targets[i]))
                return ContractStatus::target_set_invalid;
            if (i != 0 && config.immediate_repetition == ImmediateRepetitionPolicy::forbid &&
                config.explicit_targets[i] == config.explicit_targets[i - 1])
                return ContractStatus::target_set_invalid;
        }
        if (config.initial_target != kUnsetTargetId &&
            config.initial_target != config.explicit_targets[0])
            return ContractStatus::target_set_invalid;
        if (config.target_count_limit > config.n_explicit)
            return ContractStatus::target_set_invalid;
    }
    else
    {
        if (config.n_explicit != 0 || !padding_clear(config.explicit_targets, 0))
            return ContractStatus::target_set_invalid;
        // An unknown nonzero version is structurally valid: replay may use the
        // recorded realized schedule without evaluating this sampler. Refuse
        // only when select_target() actually needs a seeded draw.
        const bool more_than_one_target =
            config.target_count_limit == 0 || config.target_count_limit > 1;
        if (config.immediate_repetition == ImmediateRepetitionPolicy::forbid &&
            config.n_candidates < 2 && more_than_one_target)
            return ContractStatus::target_set_invalid;
    }
    return ContractStatus::ok;
}

ContractStatus grid_cell(const WebGridConfig& config, std::uint16_t row, std::uint16_t column,
                         GridCell& cell) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (row >= config.rows || column >= config.columns)
        return ContractStatus::target_set_invalid;

    const GridCell result = make_cell_unchecked(config, row, column);
    if (const ContractStatus status = validate(result); status != ContractStatus::ok)
        return status;
    cell = result;
    return ContractStatus::ok;
}

ContractStatus grid_cell(const WebGridConfig& config, TargetId id, GridCell& cell) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (!physical_cell_id(config, id))
        return ContractStatus::target_set_invalid;
    const std::uint32_t offset = id - 1;
    cell = make_cell_unchecked(config, static_cast<std::uint16_t>(offset / config.columns),
                               static_cast<std::uint16_t>(offset % config.columns));
    return ContractStatus::ok;
}

ContractStatus locate_cell(const WebGridConfig& config, const PointerPosition& pointer,
                           TargetId& id) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(pointer); status != ContractStatus::ok)
        return status;

    TargetId result = kUnsetTargetId;
    if (pointer.x >= config.bounds.min_x && pointer.x < config.bounds.max_x &&
        pointer.y >= config.bounds.min_y && pointer.y < config.bounds.max_y)
    {
        for (std::uint16_t row = 0; row < config.rows && result == kUnsetTargetId; ++row)
            for (std::uint16_t column = 0; column < config.columns; ++column)
            {
                const GridCell cell = make_cell_unchecked(config, row, column);
                if (pointer.x >= cell.bounds.min_x && pointer.x < cell.bounds.max_x &&
                    pointer.y >= cell.bounds.min_y && pointer.y < cell.bounds.max_y)
                {
                    result = cell.id;
                    break;
                }
            }
    }
    id = result;
    return ContractStatus::ok;
}

ContractStatus locate_selectable_cell(const WebGridConfig& config, const PointerPosition& pointer,
                                      TargetId& id) noexcept
{
    TargetId result = kUnsetTargetId;
    if (const ContractStatus status = locate_cell(config, pointer, result);
        status != ContractStatus::ok)
        return status;
    id = selectable(config, result) ? result : kUnsetTargetId;
    return ContractStatus::ok;
}

bool target_limit_reached(const WebGridConfig& config, TrialOrdinal completed) noexcept
{
    if (config.target_count_limit != 0 && completed >= config.target_count_limit)
        return true;
    return config.schedule == TargetScheduleKind::explicit_sequence &&
           completed >= config.n_explicit;
}

ContractStatus select_target(const WebGridConfig& config, TrialOrdinal ordinal, TargetId previous,
                             TargetId& target) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (target_limit_reached(config, ordinal))
    {
        target = kUnsetTargetId;
        return ContractStatus::ok;
    }
    if (config.schedule == TargetScheduleKind::explicit_sequence)
    {
        target = config.explicit_targets[static_cast<std::size_t>(ordinal)];
        return ContractStatus::ok;
    }
    if (ordinal == 0 && config.initial_target != kUnsetTargetId)
    {
        target = config.initial_target;
        return ContractStatus::ok;
    }

    std::uint64_t sampled = 0;
    const DrawKey key{config.seed, kTargetSelectionStream, ordinal, 0};
    if (ordinal == 0 || config.immediate_repetition == ImmediateRepetitionPolicy::allow)
    {
        if (!sampler_version_supported(config.sampler_version))
            return ContractStatus::version_unsupported;
        if (const ContractStatus status = sample_index(key, config.n_candidates, sampled);
            status != ContractStatus::ok)
            return status;
        target = config.candidates[static_cast<std::size_t>(sampled)];
        return ContractStatus::ok;
    }

    if (previous == kUnsetTargetId)
        return ContractStatus::identity_missing;
    if (!sampler_version_supported(config.sampler_version))
        return ContractStatus::version_unsupported;

    std::size_t previous_idx = config.n_candidates;
    for (std::size_t i = 0; i < config.n_candidates; ++i)
        if (config.candidates[i] == previous)
        {
            previous_idx = i;
            break;
        }
    if (previous_idx == config.n_candidates)
        return ContractStatus::target_set_invalid;
    if (const ContractStatus status = sample_index(key, config.n_candidates - 1, sampled);
        status != ContractStatus::ok)
        return status;
    std::size_t selected = static_cast<std::size_t>(sampled);
    if (selected >= previous_idx)
        ++selected;
    target = config.candidates[selected];
    return ContractStatus::ok;
}

ContractStatus session_duration_reached(const WebGridConfig& config, ExperimentTimeNs start_ns,
                                        ExperimentTimeNs time_ns, bool& reached) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (time_ns < start_ns)
        return ContractStatus::time_regressed;
    if (config.session_duration_ns == 0)
    {
        reached = false;
        return ContractStatus::ok;
    }
    if (start_ns > (std::numeric_limits<ExperimentTimeNs>::max)() - config.session_duration_ns)
        return ContractStatus::duration_overflow;
    reached = time_ns >= start_ns + config.session_duration_ns;
    return ContractStatus::ok;
}

ContractStatus make_selection_event(const WebGridConfig& config, const PointerPosition& pointer,
                                    TargetId intended, TrialIdentity trial, ParadigmId paradigm,
                                    ExperimentTimeNs time_ns, SequenceOrdinal sequence,
                                    SelectionEvent& event) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (!selectable(config, intended))
        return ContractStatus::target_set_invalid;
    if (trial.target_id != kUnsetTargetId && trial.target_id != intended)
        return ContractStatus::outcome_invalid;

    TargetId selected = kUnsetTargetId;
    if (const ContractStatus status = locate_selectable_cell(config, pointer, selected);
        status != ContractStatus::ok)
        return status;
    SelectionEvent result{};
    result.time_ns = time_ns;
    result.sequence = sequence;
    result.trial = trial;
    result.paradigm = paradigm;
    result.kind = SelectionKind::discrete;
    result.correct = selected == intended;
    result.selected_id = selected;
    result.intended_id = intended;
    result.dwell_ns = 0;
    if (const ContractStatus status = neurale::experiments::validate(result);
        status != ContractStatus::ok)
        return status;
    event = result;
    return ContractStatus::ok;
}

std::uint64_t configuration_fingerprint(const WebGridConfig& config) noexcept
{
    FingerprintAccumulator accumulator;
    accumulator.absorb(config.rows);
    accumulator.absorb(config.columns);
    absorb_double(accumulator, config.bounds.min_x);
    absorb_double(accumulator, config.bounds.max_x);
    absorb_double(accumulator, config.bounds.min_y);
    absorb_double(accumulator, config.bounds.max_y);
    accumulator.absorb(config.n_candidates);
    for (const TargetId value : config.candidates)
        accumulator.absorb(value);
    accumulator.absorb(static_cast<std::uint8_t>(config.schedule));
    accumulator.absorb(static_cast<std::uint8_t>(config.immediate_repetition));
    accumulator.absorb(static_cast<std::uint8_t>(config.correct_selection));
    accumulator.absorb(static_cast<std::uint8_t>(config.incorrect_selection));
    accumulator.absorb(config.seed);
    accumulator.absorb(config.sampler_version);
    accumulator.absorb(config.n_explicit);
    for (const TargetId value : config.explicit_targets)
        accumulator.absorb(value);
    accumulator.absorb(config.initial_target);
    accumulator.absorb(config.session_duration_ns);
    accumulator.absorb(config.target_count_limit);
    accumulator.absorb(config.metric_version);
    return accumulator.value();
}

// State machine.
namespace
{

constexpr double kNanosecondsPerMinute = 60'000'000'000.0;

[[nodiscard]] bool same_identity(const TrialIdentity& left, const TrialIdentity& right) noexcept
{
    return left.ordinal == right.ordinal && left.key == right.key && left.block == right.block &&
           left.target_id == right.target_id && left.stimulus_id == right.stimulus_id;
}

[[nodiscard]] WebGridMetrics make_metrics(std::uint16_t metric_version, std::uint64_t correct,
                                          std::uint64_t incorrect, DurationNs elapsed,
                                          DurationNs total_acquisition,
                                          DurationNs minimum_acquisition,
                                          DurationNs maximum_acquisition) noexcept
{
    WebGridMetrics metrics{};
    metrics.metric_version = metric_version;
    metrics.correct_selections = correct;
    metrics.incorrect_selections = incorrect;
    metrics.elapsed_active_ns = elapsed;
    metrics.n_acquisitions = correct;
    metrics.total_acquisition_ns = total_acquisition;
    metrics.minimum_acquisition_ns = correct == 0 ? 0 : minimum_acquisition;
    metrics.maximum_acquisition_ns = correct == 0 ? 0 : maximum_acquisition;
    metrics.mean_acquisition_ns = correct == 0 ? 0 : total_acquisition / correct;
    if (elapsed != 0)
    {
        metrics.rates_defined = true;
        const double scale = kNanosecondsPerMinute / static_cast<double>(elapsed);
        metrics.correct_targets_per_minute = static_cast<double>(correct) * scale;
        metrics.net_correct_targets_per_minute =
            (static_cast<double>(correct) - static_cast<double>(incorrect)) * scale;
    }
    return metrics;
}

} // namespace

ContractStatus validate(const WebGridSelectionRecord& record) noexcept
{
    if (const ContractStatus status = neurale::experiments::validate(record.event);
        status != ContractStatus::ok)
        return status;
    if (record.event.kind != SelectionKind::discrete)
        return ContractStatus::outcome_invalid;
    if (record.event.trial.target_id != record.event.intended_id ||
        record.event.trial.stimulus_id != kUnsetStimulusId)
        return ContractStatus::outcome_invalid;
    if (record.event.time_ns < record.target_onset_ns)
        return ContractStatus::time_regressed;
    if (record.elapsed_since_target_onset_ns != record.event.time_ns - record.target_onset_ns)
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const WebGridTrial& trial) noexcept
{
    if (const ContractStatus status = neurale::experiments::validate(trial.record);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(trial.selection); status != ContractStatus::ok)
        return status;
    if (!trial.selection.event.correct || trial.record.outcome != TrialOutcome::success ||
        trial.record.reason != static_cast<std::uint32_t>(WebGridReason::target_selected))
        return ContractStatus::outcome_invalid;
    if (!same_identity(trial.record.trial, trial.selection.event.trial) ||
        trial.record.paradigm != trial.selection.event.paradigm ||
        trial.record.interval.start_ns != trial.selection.target_onset_ns ||
        trial.record.interval.end_ns != trial.selection.event.time_ns ||
        trial.acquisition_ns != trial.selection.elapsed_since_target_onset_ns)
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus summarize(std::span<const WebGridSelectionRecord> selections,
                         ExperimentTimeNs start_ns, ExperimentTimeNs end_ns,
                         std::uint16_t metric_version, WebGridMetrics& metrics) noexcept
{
    if (metric_version == 0)
        return ContractStatus::identity_missing;
    if (metric_version != kMetricVersion1)
        return ContractStatus::version_unsupported;
    if (end_ns < start_ns)
        return ContractStatus::interval_inverted;

    std::uint64_t correct = 0;
    std::uint64_t incorrect = 0;
    DurationNs total = 0;
    DurationNs minimum = 0;
    DurationNs maximum = 0;
    const WebGridSelectionRecord* previous = nullptr;
    for (const WebGridSelectionRecord& record : selections)
    {
        if (const ContractStatus status = validate(record); status != ContractStatus::ok)
            return status;
        if (record.target_onset_ns < start_ns || record.event.time_ns > end_ns)
            return ContractStatus::outcome_invalid;
        if (previous != nullptr)
        {
            if (record.event.time_ns < previous->event.time_ns ||
                record.event.sequence <= previous->event.sequence ||
                record.event.paradigm != previous->event.paradigm)
                return ContractStatus::outcome_invalid;
            if (!previous->event.correct)
            {
                if (!same_identity(record.event.trial, previous->event.trial) ||
                    record.target_onset_ns != previous->target_onset_ns)
                    return ContractStatus::outcome_invalid;
            }
            else
            {
                if (previous->event.trial.ordinal == (std::numeric_limits<TrialOrdinal>::max)() ||
                    record.event.trial.ordinal != previous->event.trial.ordinal + 1 ||
                    record.target_onset_ns != previous->event.time_ns)
                    return ContractStatus::outcome_invalid;
            }
        }
        previous = &record;
        if (!record.event.correct)
        {
            ++incorrect;
            continue;
        }
        if (total > (std::numeric_limits<DurationNs>::max)() - record.elapsed_since_target_onset_ns)
            return ContractStatus::duration_overflow;
        total += record.elapsed_since_target_onset_ns;
        if (correct == 0 || record.elapsed_since_target_onset_ns < minimum)
            minimum = record.elapsed_since_target_onset_ns;
        if (record.elapsed_since_target_onset_ns > maximum)
            maximum = record.elapsed_since_target_onset_ns;
        ++correct;
    }
    metrics = make_metrics(metric_version, correct, incorrect, end_ns - start_ns, total, minimum,
                           maximum);
    return ContractStatus::ok;
}

WebGridSnapshot WebGridMachine::build_snapshot(const Run& run,
                                               ExperimentTimeNs time_ns) const noexcept
{
    WebGridSnapshot snapshot{};
    snapshot.time_ns = time_ns;
    snapshot.state = run.state;
    snapshot.trial = run.trial;
    snapshot.active_target = run.active_target;
    snapshot.target_onset_ns = run.target_onset_ns;
    snapshot.completed = run.completed;
    const ExperimentTimeNs active_end =
        run.state == WebGridState::complete ? run.ended_ns : time_ns;
    const DurationNs elapsed =
        run.state == WebGridState::idle ? 0 : active_end - run.session_started_ns;
    snapshot.metrics = make_metrics(config_.metric_version, run.completed, run.incorrect, elapsed,
                                    run.total_acquisition_ns, run.minimum_acquisition_ns,
                                    run.maximum_acquisition_ns);
    return snapshot;
}

ContractStatus WebGridMachine::start(ParadigmId paradigm, const WebGridConfig& config,
                                     ExperimentTimeNs time_ns, WebGridStepResult& result) noexcept
{
    if (run_.state != WebGridState::idle)
        return ContractStatus::already_running;
    if (paradigm == kUnsetParadigmId)
        return ContractStatus::identity_missing;
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (config.metric_version != kCurrentMetricVersion)
        return ContractStatus::version_unsupported;
    if (config.session_duration_ns != 0 &&
        time_ns > (std::numeric_limits<ExperimentTimeNs>::max)() - config.session_duration_ns)
        return ContractStatus::duration_overflow;

    TargetId target = kUnsetTargetId;
    if (const ContractStatus status = select_target(config, 0, kUnsetTargetId, target);
        status != ContractStatus::ok)
        return status;

    WebGridMachine started{};
    started.paradigm_ = paradigm;
    started.config_ = config;
    Run& run = started.run_;
    if (const ContractStatus status = run.gate.accept(time_ns); status != ContractStatus::ok)
        return status;
    run.state = WebGridState::active_target;
    run.session_started_ns = time_ns;
    run.trial = TrialIdentity{0, kUnsetTrialKey, 0, target, kUnsetStimulusId};
    run.active_target = target;
    run.target_onset_ns = time_ns;

    WebGridStepResult local{};
    local.snapshot = started.build_snapshot(run, time_ns);
    *this = started;
    result = local;
    return ContractStatus::ok;
}

void WebGridMachine::reset() noexcept
{
    *this = WebGridMachine{};
}

ContractStatus WebGridMachine::step(ExperimentTimeNs time_ns, const PointerPosition& pointer,
                                    WebGridStepResult& result) noexcept
{
    return step_impl(time_ns, pointer, nullptr, result);
}

ContractStatus WebGridMachine::step(ExperimentTimeNs time_ns, const PointerPosition& pointer,
                                    const SelectionEvent& selection,
                                    WebGridStepResult& result) noexcept
{
    return step_impl(time_ns, pointer, &selection, result);
}

ContractStatus WebGridMachine::step_impl(ExperimentTimeNs time_ns, const PointerPosition& pointer,
                                         const SelectionEvent* selection,
                                         WebGridStepResult& result) noexcept
{
    if (run_.state != WebGridState::active_target)
        return ContractStatus::not_running;
    if (const ContractStatus status = validate(pointer); status != ContractStatus::ok)
        return status;

    Run run = run_;
    if (const ContractStatus status = run.gate.accept(time_ns); status != ContractStatus::ok)
        return status;
    WebGridStepResult local{};

    if (config_.session_duration_ns != 0)
    {
        const ExperimentTimeNs deadline = run.session_started_ns + config_.session_duration_ns;
        if (time_ns >= deadline)
        {
            run.state = WebGridState::complete;
            run.ended_ns = deadline;
            run.active_target = kUnsetTargetId;
            local.snapshot = build_snapshot(run, time_ns);
            run_ = run;
            result = local;
            return ContractStatus::ok;
        }
    }

    if (selection == nullptr)
    {
        local.snapshot = build_snapshot(run, time_ns);
        run_ = run;
        result = local;
        return ContractStatus::ok;
    }

    if (const ContractStatus status = neurale::experiments::validate(*selection);
        status != ContractStatus::ok)
        return status;
    if (selection->kind != SelectionKind::discrete || selection->time_ns != time_ns ||
        selection->paradigm != paradigm_ || !same_identity(selection->trial, run.trial) ||
        selection->intended_id != run.active_target)
        return ContractStatus::outcome_invalid;
    if (run.has_last_selection && selection->sequence <= run.last_selection_sequence)
        return ContractStatus::outcome_invalid;

    TargetId selected = kUnsetTargetId;
    if (const ContractStatus status = locate_selectable_cell(config_, pointer, selected);
        status != ContractStatus::ok)
        return status;
    if (selected != selection->selected_id)
        return ContractStatus::outcome_invalid;

    WebGridSelectionRecord accepted{*selection, run.target_onset_ns, time_ns - run.target_onset_ns};
    if (const ContractStatus status = validate(accepted); status != ContractStatus::ok)
        return status;

    run.has_last_selection = true;
    run.last_selection_sequence = selection->sequence;
    local.selection_processed = true;
    local.selection = accepted;
    if (!selection->correct)
    {
        if (run.incorrect == (std::numeric_limits<std::uint64_t>::max)())
            return ContractStatus::ordinal_exhausted;
        ++run.incorrect;
        local.snapshot = build_snapshot(run, time_ns);
        run_ = run;
        result = local;
        return ContractStatus::ok;
    }

    if (run.total_acquisition_ns >
        (std::numeric_limits<DurationNs>::max)() - accepted.elapsed_since_target_onset_ns)
        return ContractStatus::duration_overflow;
    if (run.completed == (std::numeric_limits<TrialOrdinal>::max)())
        return ContractStatus::ordinal_exhausted;

    WebGridTrial trial{};
    trial.record.trial = run.trial;
    trial.record.interval = TimeInterval{run.target_onset_ns, time_ns};
    trial.record.paradigm = paradigm_;
    trial.record.outcome = TrialOutcome::success;
    trial.record.reason = static_cast<std::uint32_t>(WebGridReason::target_selected);
    trial.selection = accepted;
    trial.acquisition_ns = accepted.elapsed_since_target_onset_ns;
    if (const ContractStatus status = validate(trial); status != ContractStatus::ok)
        return status;

    run.total_acquisition_ns += accepted.elapsed_since_target_onset_ns;
    if (run.completed == 0 || accepted.elapsed_since_target_onset_ns < run.minimum_acquisition_ns)
        run.minimum_acquisition_ns = accepted.elapsed_since_target_onset_ns;
    if (accepted.elapsed_since_target_onset_ns > run.maximum_acquisition_ns)
        run.maximum_acquisition_ns = accepted.elapsed_since_target_onset_ns;
    ++run.completed;

    if (target_limit_reached(config_, run.completed))
    {
        run.state = WebGridState::complete;
        run.ended_ns = time_ns;
        run.active_target = kUnsetTargetId;
    }
    else
    {
        TargetId next = kUnsetTargetId;
        if (const ContractStatus status =
                select_target(config_, run.completed, run.active_target, next);
            status != ContractStatus::ok)
            return status;
        if (next == kUnsetTargetId)
        {
            run.state = WebGridState::complete;
            run.ended_ns = time_ns;
            run.active_target = kUnsetTargetId;
        }
        else
        {
            run.active_target = next;
            run.target_onset_ns = time_ns;
            run.trial = TrialIdentity{run.completed, kUnsetTrialKey, 0, next, kUnsetStimulusId};
        }
    }

    local.trial_decided = true;
    local.trial = trial;
    local.snapshot = build_snapshot(run, time_ns);
    run_ = run;
    result = local;
    return ContractStatus::ok;
}

WebGridSnapshot WebGridMachine::snapshot() const noexcept
{
    const ExperimentTimeNs time_ns = run_.gate.started() ? run_.gate.last_ns() : 0;
    return build_snapshot(run_, time_ns);
}
} // namespace neurale::experiments::webgrid
