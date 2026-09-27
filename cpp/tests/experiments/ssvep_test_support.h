// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT
#pragma once
#include <iostream>
#include <limits>
#include <neurale/experiments/ssvep.h>

using namespace neurale::experiments;
using namespace neurale::experiments::ssvep;
inline int failures = 0;
#define CHECK(...)                                                                                 \
    do                                                                                             \
    {                                                                                              \
        if (!(__VA_ARGS__))                                                                        \
        {                                                                                          \
            ++failures;                                                                            \
            std::cerr << __LINE__ << ": " << #__VA_ARGS__ << '\n';                                 \
        }                                                                                          \
    } while (false)
inline SSVEPConfig config(TrialOrdinal count = 1)
{
    SSVEPConfig c{};
    c.n_targets = 4;
    c.targets[0] = {1, 8};
    c.targets[1] = {2, 10};
    c.targets[2] = {3, 12};
    c.targets[3] = {4, 15};
    c.stimulus_id = 5;
    c.n_trials = count;
    c.seed = 42;
    c.cue_duration_ns = 10;
    c.stimulation_duration_ns = 20;
    c.decision_timeout_ns = 10;
    c.feedback_duration_ns = 5;
    c.inter_trial_ns = 5;
    return c;
}
inline SelectionEvent choice(const SSVEPMachine& machine, ExperimentTimeNs time,
                             bool correct = true, SequenceOrdinal sequence = 0)
{
    SelectionEvent e{};
    e.time_ns = time;
    e.sequence = sequence;
    e.trial = machine.snapshot().trial;
    e.paradigm = machine.paradigm();
    e.intended_id = e.trial.target_id;
    e.selected_id = correct ? e.intended_id : e.intended_id % 4 + 1;
    e.correct = correct;
    return e;
}
inline void check_output(const SSVEPStepResult& out)
{
    CHECK(out.n_transitions <= kMaxStepTransitions);
    CHECK(out.n_events <= kMaxStepEvents);
    CHECK(out.n_requests <= kMaxStepRequests);
    for (std::size_t i = 0; i < out.n_events; ++i)
        CHECK(validate(out.events[i]) == ContractStatus::ok);
    for (std::size_t i = 0; i < out.n_transitions; ++i)
        CHECK(validate(out.transitions[i]) == ContractStatus::ok);
    for (std::size_t i = 0; i < out.n_requests; ++i)
        CHECK(validate(out.requests[i]) == ContractStatus::ok);
    if (out.trial_decided)
        CHECK(validate(out.trial) == ContractStatus::ok);
}
