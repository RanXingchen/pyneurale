// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#include "allocation_counter.h"
#include "ssvep_test_support.h"
#include <neurale/experiments/center_out.h>
#include <type_traits>
#include <vector>

namespace machine_tests
{
namespace
{
void outcomes()
{
    for (const auto at : {10ULL, 29ULL, 30ULL, 31ULL, 39ULL, 40ULL, 41ULL})
        for (bool correct : {false, true})
        {
            SSVEPMachine m{};
            SSVEPStepResult out{};
            CHECK(m.start(7, config(), 0, out) == ContractStatus::ok);
            check_output(out);
            CHECK(m.step(at, choice(m, at, correct), out) == ContractStatus::ok);
            check_output(out);
            CHECK(out.selection_disposition == (at < 40 ? SSVEPSelectionDisposition::accepted
                                                        : SSVEPSelectionDisposition::expired));
            const auto feedback = at < 30 ? 30 : at < 40 ? at : 40;
            if (at < 30)
                CHECK(m.state() == SSVEPState::stimulation);
            CHECK(m.step(feedback + 5, out) == ContractStatus::ok);
            check_output(out);
            CHECK(out.trial_decided && out.snapshot.completed == 1);
            CHECK(out.trial.record.outcome == (at >= 40  ? TrialOutcome::timeout
                                               : correct ? TrialOutcome::success
                                                         : TrialOutcome::failure));
            CHECK(out.trial.decision_ns == feedback);
            CHECK(out.trial.record.interval.end_ns == feedback + 5);
            CHECK(out.trial.n_phases == (at <= 30 ? 3 : 4));
            CHECK(m.step(feedback + 10, out) == ContractStatus::ok);
            check_output(out);
            CHECK(m.complete() && !out.trial_decided);
            CHECK(out.requests[out.n_requests - 1].request.cue == CueKind::black);
        }
}

void rejections()
{
    SSVEPMachine m{};
    SSVEPStepResult out{};
    CHECK(m.step(0, out) == ContractStatus::not_running);
    out.n_events = 31;
    CHECK(m.start(0, config(), 0, out) == ContractStatus::identity_missing);
    CHECK(out.n_events == 31 && m.state() == SSVEPState::idle);
    CHECK(m.start(7, config(), std::numeric_limits<ExperimentTimeNs>::max() - 49, out) ==
          ContractStatus::duration_overflow);
    CHECK(out.n_events == 31 && m.state() == SSVEPState::idle);
    CHECK(m.start(7, config(), 0, out) == ContractStatus::ok);
    CHECK(m.start(7, config(), 0, out) == ContractStatus::already_running);
    CHECK(m.step(9, choice(m, 9), out) == ContractStatus::outcome_invalid);
    CHECK(m.snapshot().time_ns == 0);
    for (int bad = 0; bad < 7; ++bad)
    {
        auto e = choice(m, 10);
        switch (bad)
        {
        case 0:
            e.trial.ordinal++;
            break;
        case 1:
            e.trial.key++;
            break;
        case 2:
            e.paradigm++;
            break;
        case 3:
            e.selected_id = 99;
            e.correct = false;
            break;
        case 4:
            e.correct = !e.correct;
            break;
        case 5:
            e.time_ns++;
            break;
        case 6:
            e.kind = SelectionKind::dwell;
            e.dwell_ns = 1;
            break;
        }
        out.n_events = 31;
        CHECK(m.step(10, e, out) == ContractStatus::outcome_invalid);
        CHECK(out.n_events == 31 && m.snapshot().time_ns == 0);
    }
    auto e = choice(m, 10, true, 3);
    CHECK(m.step(10, e, out) == ContractStatus::ok);
    CHECK(m.step(10, e, out) == ContractStatus::outcome_invalid);
    CHECK(m.step(9, out) == ContractStatus::time_regressed);
    CHECK(m.step(10, out) == ContractStatus::ok);
    CHECK(out.settled && out.n_events == 0 && out.n_requests == 0 && out.n_transitions == 0);
    CHECK(m.step(30, choice(m, 30, false, 4), out) == ContractStatus::outcome_invalid);
    CHECK(m.step(100, out) == ContractStatus::ok);
    check_output(out);
    CHECK(out.trial_decided && !out.settled);
    CHECK(m.step(100, out) == ContractStatus::ok);
    check_output(out);
    CHECK(m.complete());
    CHECK(m.step(100, out) == ContractStatus::not_running);
    CHECK(m.stop(100, out) == ContractStatus::not_running);
    m.reset();
    CHECK(m.start(7, config(), 0, out) == ContractStatus::ok);
    CHECK(out.snapshot.trial.target_id == 4);
}

void stop_states()
{
    for (auto at : {0ULL, 5ULL, 10ULL, 20ULL, 30ULL, 35ULL, 40ULL, 42ULL, 45ULL, 47ULL})
    {
        SSVEPMachine m{};
        SSVEPStepResult out{};
        CHECK(m.start(7, config(), 0, out) == ContractStatus::ok);
        CHECK(m.step(at, out) == ContractStatus::ok);
        const bool ended = m.state() == SSVEPState::inter_trial;
        CHECK(m.stop(at, out) == ContractStatus::ok);
        check_output(out);
        CHECK(m.complete() && out.trial_decided == !ended);
        if (!ended)
            CHECK(out.trial.record.outcome == TrialOutcome::aborted);
    }
    SSVEPMachine m{};
    SSVEPStepResult out{};
    CHECK(m.start(7, config(2), 0, out) == ContractStatus::ok);
    CHECK(m.stop(100, out) == ContractStatus::ok);
    check_output(out);
    CHECK(out.trial.record.interval.end_ns == 100 && out.snapshot.completed == 1);
    CHECK(out.trial.n_phases == 1); // Stop does not synthesize unobserved trials.
}

std::vector<ExperimentEvent> simulate(bool dense)
{
    SSVEPMachine m{};
    SSVEPStepResult out{};
    std::vector<ExperimentEvent> events;
    CHECK(m.start(7, config(4), 0, out) == ContractStatus::ok);
    const auto collect = [&]
    {
        check_output(out);
        for (std::size_t i = 0; i < out.n_events; ++i)
            events.push_back(out.events[i]);
    };
    collect();
    for (ExperimentTimeNs t = dense ? 1 : 200; !m.complete(); t += dense ? 1 : 0)
    {
        CHECK(m.step(t, out) == ContractStatus::ok);
        collect();
    }
    return events;
}
void sparse_and_records()
{
    const auto a = simulate(true), b = simulate(false);
    CHECK(a.size() == b.size());
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i)
    {
        CHECK(a[i].time_ns == b[i].time_ns && a[i].sequence == b[i].sequence);
        CHECK(a[i].kind == b[i].kind && a[i].code == b[i].code && a[i].value == b[i].value);
        CHECK(same_trial(a[i].trial, b[i].trial));
    }
    SSVEPMachine m{};
    SSVEPStepResult out{};
    CHECK(m.start(7, config(2), 0, out) == ContractStatus::ok);
    auto stale = choice(m, 100);
    CHECK(m.step(100, stale, out) == ContractStatus::ok);
    check_output(out);
    CHECK(out.trial.record.outcome == TrialOutcome::timeout && out.trial.record.trial.ordinal == 0);
    CHECK(m.step(100, out) == ContractStatus::ok);
    check_output(out);
    CHECK(out.trial.record.trial.ordinal == 1 && !out.trial.has_selection);
    auto corrupted = out.trial;
    corrupted.record.reason = 999;
    CHECK(validate(corrupted) == ContractStatus::outcome_invalid);
    corrupted = out.trial;
    corrupted.phases[0].interval.end_ns++;
    CHECK(validate(corrupted) == ContractStatus::outcome_invalid);
}

void equal_time_and_allocation()
{
    SSVEPMachine m{};
    SSVEPStepResult out{};
    CHECK(m.start(7, config(2), 0, out) == ContractStatus::ok);
    CHECK(m.step(30, out) == ContractStatus::ok);
    CHECK(m.step(30, choice(m, 30, true, 7), out) == ContractStatus::ok);
    CHECK(m.step(35, out) == ContractStatus::ok);
    check_output(out);
    CHECK(out.trial.n_phases == 3);
    CHECK(m.step(40, out) == ContractStatus::ok);
    out.n_events = 31;
    CHECK(m.step(50, choice(m, 50, true, 7), out) == ContractStatus::outcome_invalid);
    CHECK(out.n_events == 31 && m.snapshot().time_ns == 40);
    CHECK(m.step(50, choice(m, 50, true, 8), out) == ContractStatus::ok);

    const auto before = allocations.load(std::memory_order_relaxed);
    bool ok = true;
    for (int i = 0; i < 100; ++i)
    {
        m.reset();
        ok &= m.start(7, config(2), 0, out) == ContractStatus::ok;
        ok &= m.step(20, choice(m, 20), out) == ContractStatus::ok;
        ok &= m.step(100, out) == ContractStatus::ok;
        ok &= m.step(100, out) == ContractStatus::ok;
        ok &= m.stop(100, out) == ContractStatus::ok;
    }
    const auto after = allocations.load(std::memory_order_relaxed);
    CHECK(ok && before == after);
}
} // namespace
int run()
{
    {
        SSVEPMachine machine;
        SSVEPStepResult result;
        CHECK(machine.start(7, config(2), 0, result) == ContractStatus::ok);
        CHECK(machine.step(30, choice(machine, 30), result) == ContractStatus::ok);
        CHECK(machine.step(35, result) == ContractStatus::ok);
        CHECK(machine.hold_inter_trial_until(100) == ContractStatus::ok);
        CHECK(machine.step(99, result) == ContractStatus::ok);
        CHECK(machine.state() == SSVEPState::inter_trial);
        CHECK(machine.step(100, result) == ContractStatus::ok);
        CHECK(machine.snapshot().trial.ordinal == 1);
    }
    outcomes();
    rejections();
    stop_states();
    sparse_and_records();
    equal_time_and_allocation();
    return failures ? 1 : 0;
}
} // namespace machine_tests

int main()
{
    const int machine_status = machine_tests::run();
    static_assert(std::is_trivially_copyable_v<SSVEPStepResult>);
    static_assert(std::is_trivially_copyable_v<SSVEPMachine>);
    auto c = config(13);
    CHECK(validate(c) == ContractStatus::ok);
    // Frozen before extracting Center-Out's shuffle into a shared helper.
    constexpr std::array<TargetId, 12> expected{4, 3, 1, 2, 4, 3, 1, 2, 1, 4, 3, 2};
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        SSVEPTrialSchedule s{};
        CHECK(prepare_trial(c, i, s) == ContractStatus::ok);
        CHECK(s.ordinal == i && s.target_id == expected[i]);
    }
    SSVEPTrialSchedule s{999, 99};
    CHECK(prepare_trial(c, 13, s) == ContractStatus::outcome_invalid);
    CHECK(s.ordinal == 999 && s.target_id == 99);
    c.sampler_version = 999;
    CHECK(validate(c) == ContractStatus::ok);
    CHECK(prepare_trial(c, 0, s) == ContractStatus::version_unsupported);
    CHECK(s.target_id == 99);
    c = config();
    c.targets[1].id = c.targets[0].id;
    CHECK(validate(c) == ContractStatus::target_set_invalid);
    c = config();
    c.targets[1].frequency_hz = c.targets[0].frequency_hz;
    CHECK(validate(c) == ContractStatus::target_set_invalid);
    c = config();
    c.targets[0].frequency_hz = std::numeric_limits<double>::quiet_NaN();
    CHECK(validate(c) == ContractStatus::value_not_finite);
    c = config();
    c.targets[0].frequency_hz = -1;
    CHECK(validate(c) == ContractStatus::parameter_out_of_range);
    c = config();
    c.n_targets = 65;
    CHECK(validate(c) == ContractStatus::target_set_invalid);
    CHECK(!contains_target(c, 1));
    c = config();
    c.decision_timeout_ns = 0;
    CHECK(validate(c) == ContractStatus::range_empty);
    c = config();
    c.stimulus_id = 0;
    CHECK(validate(c) == ContractStatus::identity_missing);
    c = config(65);
    c.n_targets = 64;
    for (std::size_t i = 0; i < 64; ++i)
        c.targets[i] = {static_cast<TargetId>(i + 1), double(i + 1)};
    std::array<bool, 64> seen{};
    for (std::size_t i = 0; i < 64; ++i)
    {
        CHECK(prepare_trial(c, i, s) == ContractStatus::ok);
        CHECK(!seen[s.target_id - 1]);
        seen[s.target_id - 1] = true;
    }
    CHECK(prepare_trial(c, 64, s) == ContractStatus::ok);
    return failures == 0 && machine_status == 0 ? 0 : 1;
}
