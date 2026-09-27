/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/command.h>
#include <neurale/experiments/contract.h>
#include <neurale/experiments/events.h>
#include <neurale/experiments/identity.h>
#include <neurale/experiments/presentation.h>
#include <neurale/experiments/replay.h>
#include <neurale/experiments/schedule.h>
#include <neurale/experiments/time.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string_view>
#include <type_traits>

#include "allocation_counter.h"
#include "check_returns.h"

namespace
{

using namespace neurale::experiments;

// Every runtime record is a fixed-size aggregate that can be memcpy'd into a
// bounded trace buffer. A record that stopped being trivially copyable would be
// a record that had grown an owning member, which is how a trace acquires an
// allocation on the path that produces it.
static_assert(std::is_trivially_copyable_v<TimeInterval>);
static_assert(std::is_trivially_copyable_v<TrialIdentity>);
static_assert(std::is_trivially_copyable_v<ExperimentEvent>);
static_assert(std::is_trivially_copyable_v<StateTransition>);
static_assert(std::is_trivially_copyable_v<SelectionEvent>);
static_assert(std::is_trivially_copyable_v<TrialRecord>);
static_assert(std::is_trivially_copyable_v<PresentationRequest>);
static_assert(std::is_trivially_copyable_v<PresentationState>);
static_assert(std::is_trivially_copyable_v<PresentationOutcome>);
static_assert(std::is_trivially_copyable_v<CommandSpace>);
static_assert(std::is_trivially_copyable_v<CommandRequest>);
static_assert(std::is_trivially_copyable_v<CommandOutcome>);
static_assert(std::is_trivially_copyable_v<ScheduleIdentity>);
static_assert(std::is_trivially_copyable_v<ScheduleDraw>);
static_assert(std::is_trivially_copyable_v<ExperimentSnapshot>);
static_assert(std::is_trivially_copyable_v<DecisionSnapshot>);

// ContractStatus ordinals are a frozen part of the shared native/Python
// contract: a later task that adds a status must append it, not insert it, or it
// renumbers an already-public value. These pin every ordinal so a reordering
// fails to compile rather than silently shifting a frozen value.
static_assert(static_cast<std::uint8_t>(ContractStatus::ok) == 0);
static_assert(static_cast<std::uint8_t>(ContractStatus::time_regressed) == 1);
static_assert(static_cast<std::uint8_t>(ContractStatus::interval_inverted) == 2);
static_assert(static_cast<std::uint8_t>(ContractStatus::duration_overflow) == 3);
static_assert(static_cast<std::uint8_t>(ContractStatus::dimension_invalid) == 4);
static_assert(static_cast<std::uint8_t>(ContractStatus::value_not_finite) == 5);
static_assert(static_cast<std::uint8_t>(ContractStatus::expiry_before_generation) == 6);
static_assert(static_cast<std::uint8_t>(ContractStatus::range_empty) == 7);
static_assert(static_cast<std::uint8_t>(ContractStatus::sampling_exhausted) == 8);
static_assert(static_cast<std::uint8_t>(ContractStatus::ordinal_exhausted) == 9);
static_assert(static_cast<std::uint8_t>(ContractStatus::identity_missing) == 10);
static_assert(static_cast<std::uint8_t>(ContractStatus::enum_undeclared) == 11);
static_assert(static_cast<std::uint8_t>(ContractStatus::presentation_invalid) == 12);
static_assert(static_cast<std::uint8_t>(ContractStatus::outcome_invalid) == 13);
static_assert(static_cast<std::uint8_t>(ContractStatus::assistance_out_of_range) == 14);
static_assert(static_cast<std::uint8_t>(ContractStatus::domain_mask_invalid) == 15);
static_assert(static_cast<std::uint8_t>(ContractStatus::version_unsupported) == 16);
static_assert(static_cast<std::uint8_t>(ContractStatus::numerical_failure) == 17);
static_assert(static_cast<std::uint8_t>(ContractStatus::parameter_out_of_range) == 18);
static_assert(static_cast<std::uint8_t>(ContractStatus::target_set_invalid) == 19);
static_assert(static_cast<std::uint8_t>(ContractStatus::not_running) == 20);
static_assert(static_cast<std::uint8_t>(ContractStatus::already_running) == 21);

// Compact as well as fixed. The bound is deliberately generous; what it rules
// out is a record that has quietly become a payload. CommandRequest is the
// largest by construction, because it carries kMaxCommandDim doubles.
static_assert(sizeof(ExperimentEvent) <= 128);
static_assert(sizeof(StateTransition) <= 128);
static_assert(sizeof(SelectionEvent) <= 128);
static_assert(sizeof(TrialRecord) <= 128);
static_assert(sizeof(PresentationRequest) <= 128);
static_assert(sizeof(PresentationState) <= 128);
static_assert(sizeof(PresentationOutcome) <= 128);
static_assert(sizeof(CommandRequest) <= 128);
static_assert(sizeof(CommandOutcome) <= 128);
static_assert(sizeof(ExperimentSnapshot) <= 128);
static_assert(sizeof(DecisionSnapshot) <= 128);

// The counters are distinct types even though both count with a 64-bit integer,
// so a record ordinal cannot be handed to something expecting a trial ordinal.
static_assert(!std::is_same_v<TrialCounter, SequenceCounter>);

int check_time()
{
    constexpr TimeInterval phase{1'000, 2'000};
    CHECK(phase.well_formed());
    CHECK(!phase.empty());
    CHECK(phase.duration_ns() == 1'000);

    // Half-open: the start instant is inside, the end instant is not, and the
    // end instant therefore belongs to whatever phase follows.
    CHECK(phase.contains(1'000));
    CHECK(phase.contains(1'999));
    CHECK(!phase.contains(2'000));
    CHECK(!phase.contains(999));
    CHECK(!phase.elapsed_at(1'999));
    CHECK(phase.elapsed_at(2'000));
    CHECK(phase.elapsed_at(2'001));

    constexpr TimeInterval following{2'000, 3'000};
    CHECK(following.contains(2'000));

    constexpr TimeInterval pending{5'000, 5'000};
    CHECK(pending.well_formed());
    CHECK(pending.empty());
    CHECK(!pending.contains(5'000));
    CHECK(pending.duration_ns() == 0);

    constexpr TimeInterval inverted{5'000, 4'000};
    CHECK(!inverted.well_formed());
    CHECK(validate(inverted) == ContractStatus::interval_inverted);
    CHECK(inverted.duration_ns() == 0);

    TimeInterval built{};
    CHECK(interval_from_duration(10, 5, built) == ContractStatus::ok);
    CHECK(built.start_ns == 10 && built.end_ns == 15);
    const ExperimentTimeNs late = (std::numeric_limits<ExperimentTimeNs>::max)() - 3;
    CHECK(interval_from_duration(late, 3, built) == ContractStatus::ok);
    CHECK(interval_from_duration(late, 4, built) == ContractStatus::duration_overflow);
    CHECK(!time_fits(late, 4));

    // The gate guards a precondition; it never produces a time of its own.
    MonotonicTimeGate gate;
    CHECK(!gate.started());
    CHECK(gate.accept(100) == ContractStatus::ok);
    CHECK(gate.started());
    CHECK(gate.accept(100) == ContractStatus::ok);
    CHECK(gate.accept(101) == ContractStatus::ok);
    CHECK(gate.accept(100) == ContractStatus::time_regressed);
    CHECK(gate.last_ns() == 101);
    gate.reset();
    CHECK(!gate.started());
    CHECK(gate.accept(1) == ContractStatus::ok);
    gate.reset(500);
    CHECK(gate.started() && gate.last_ns() == 500);
    CHECK(gate.accept(499) == ContractStatus::time_regressed);
    CHECK(gate.accept(500) == ContractStatus::ok);

    const MonotonicTimeGate seeded{42};
    CHECK(seeded.started() && seeded.last_ns() == 42);
    return 0;
}

int check_identity()
{
    TrialCounter first;
    TrialCounter second;
    TrialOrdinal issued = 0;
    for (TrialOrdinal expected = 0; expected < 8; ++expected)
    {
        CHECK(first.peek() == expected);
        CHECK(first.issue(issued) == ContractStatus::ok && issued == expected);
        CHECK(second.issue(issued) == ContractStatus::ok && issued == expected);
    }
    CHECK(first.issued() == 8);
    CHECK(first.origin() == 0);

    // Reset is a session boundary: it restarts the whole ordinal sequence.
    first.reset();
    CHECK(first.issued() == 0);
    CHECK(first.issue(issued) == ContractStatus::ok && issued == 0);
    first.reset(100);
    CHECK(first.origin() == 100);
    CHECK(first.issue(issued) == ContractStatus::ok && issued == 100);
    CHECK(first.issue(issued) == ContractStatus::ok && issued == 101);
    CHECK(first.issued() == 2);

    SequenceOrdinal emitted = 0;
    SequenceCounter sequence{7};
    CHECK(sequence.issue(emitted) == ContractStatus::ok && emitted == 7);
    CHECK(sequence.issue(emitted) == ContractStatus::ok && emitted == 8);

    // An exhausted counter issues nothing at all. It neither wraps -- which
    // would re-issue ordinals already persisted -- nor saturates onto its last
    // value, which would hand the same identity out forever.
    TrialCounter high{(std::numeric_limits<TrialOrdinal>::max)()};
    CHECK(high.exhausted());
    TrialOrdinal refused = 0xFEED;
    CHECK(high.issue(refused) == ContractStatus::ordinal_exhausted);
    CHECK(refused == 0xFEED);
    CHECK(high.issue(refused) == ContractStatus::ordinal_exhausted);
    CHECK(refused == 0xFEED);
    CHECK(high.issued() == 0);

    // One below the boundary still issues exactly once, and exactly once more
    // is refused: the boundary is a state, not an approximation.
    TrialCounter last{(std::numeric_limits<TrialOrdinal>::max)() - 1};
    CHECK(!last.exhausted());
    CHECK(last.issue(issued) == ContractStatus::ok &&
          issued == (std::numeric_limits<TrialOrdinal>::max)() - 1);
    CHECK(last.exhausted());
    CHECK(last.issue(issued) == ContractStatus::ordinal_exhausted);

    TrialIdentity left{};
    left.ordinal = 3;
    left.block = 1;
    left.key = 900;
    TrialIdentity right{};
    right.ordinal = 3;
    right.block = 1;
    CHECK(same_trial(left, right));
    right.block = 2;
    CHECK(!same_trial(left, right));

    CHECK(!trial_ended(TrialOutcome::pending));
    CHECK(trial_ended(TrialOutcome::success));
    CHECK(trial_ended(TrialOutcome::failure));
    CHECK(trial_ended(TrialOutcome::timeout));
    CHECK(trial_ended(TrialOutcome::aborted));

    // An undeclared value is not a terminal outcome, and is reported as
    // undeclared rather than waved through.
    CHECK(trial_outcome_declared(TrialOutcome::pending));
    CHECK(trial_outcome_declared(TrialOutcome::aborted));
    CHECK(!trial_outcome_declared(static_cast<TrialOutcome>(255)));
    CHECK(!trial_ended(static_cast<TrialOutcome>(255)));
    TrialRecord corrupt{};
    corrupt.paradigm = 4;
    corrupt.outcome = static_cast<TrialOutcome>(255);
    CHECK(validate(corrupt) == ContractStatus::enum_undeclared);
    return 0;
}

int check_schedule()
{
    constexpr DrawKey key{0xC0FFEEULL, 3, 11, 0};

    // Stateless and addressed: the same coordinates give the same value however
    // many times, and in any order.
    CHECK(sample_bits(key) == sample_bits(key));
    CHECK(sample_bits(key) != sample_bits(DrawKey{key.seed, key.stream, key.trial_idx, 1}));
    CHECK(sample_bits(key) != sample_bits(DrawKey{key.seed, key.stream, 12, key.draw}));
    CHECK(sample_bits(key) != sample_bits(DrawKey{key.seed, 4, key.trial_idx, key.draw}));
    CHECK(sample_bits(key) !=
          sample_bits(DrawKey{key.seed + 1, key.stream, key.trial_idx, key.draw}));

    std::uint64_t value = 0;
    CHECK(sample_inclusive(key, 7, 7, value) == ContractStatus::ok && value == 7);
    CHECK(sample_inclusive(key, 8, 7, value) == ContractStatus::range_empty);

    // Inclusive: both endpoints are reachable and nothing leaves the range.
    std::array<int, 4> inclusive_hits{};
    for (std::uint32_t draw = 0; draw < 400; ++draw)
    {
        CHECK(sample_inclusive(DrawKey{99, 1, 0, draw}, 0, 3, value) == ContractStatus::ok);
        CHECK(value <= 3);
        ++inclusive_hits[static_cast<std::size_t>(value)];
    }
    for (const int hits : inclusive_hits)
        CHECK(hits > 0);

    // Exclusive: neither endpoint is ever realized, which is what makes a
    // Speech duration strictly positive and strictly below its bound.
    for (std::uint32_t draw = 0; draw < 400; ++draw)
    {
        CHECK(sample_exclusive(DrawKey{7, 2, 5, draw}, 0, 5, value) == ContractStatus::ok);
        CHECK(value > 0 && value < 5);
    }
    CHECK(sample_exclusive(key, 0, 2, value) == ContractStatus::ok && value == 1);
    CHECK(sample_exclusive(key, 0, 1, value) == ContractStatus::range_empty);
    CHECK(sample_exclusive(key, 0, 0, value) == ContractStatus::range_empty);
    CHECK(sample_exclusive(key, 5, 5, value) == ContractStatus::range_empty);
    CHECK(sample_exclusive(key, (std::numeric_limits<std::uint64_t>::max)(),
                           (std::numeric_limits<std::uint64_t>::max)(),
                           value) == ContractStatus::range_empty);

    CHECK(sample_index(key, 0, value) == ContractStatus::range_empty);
    CHECK(sample_index(key, 1, value) == ContractStatus::ok && value == 0);
    for (std::uint32_t draw = 0; draw < 200; ++draw)
    {
        CHECK(sample_index(DrawKey{1234, 8, 2, draw}, 6, value) == ContractStatus::ok);
        CHECK(value < 6);
    }

    // Masked rejection removes bias rather than folding it into one bucket.
    std::array<int, 3> buckets{};
    for (std::uint32_t draw = 0; draw < 3'000; ++draw)
    {
        CHECK(sample_index(DrawKey{555, 9, 0, draw}, 3, value) == ContractStatus::ok);
        ++buckets[static_cast<std::size_t>(value)];
    }
    for (const int hits : buckets)
        CHECK(hits > 800 && hits < 1'200);

    // Exhaustion is reported, never folded into range. Every attempt is accepted
    // with probability above one half, so hunting the key space for a DrawKey
    // that reaches this branch is not a test; the mapping is exercised through a
    // source that rejects deliberately instead.
    // Range [0, 4] masks to 7, so 5, 6 and 7 are the rejected residues.
    std::uint64_t exhausted_value = 0xD15EA5E;
    CHECK(detail::sample_inclusive_from([](std::uint32_t) { return std::uint64_t{7}; }, 0, 4,
                                        exhausted_value) == ContractStatus::sampling_exhausted);
    // A rejected draw leaves the caller's value untouched rather than writing a
    // biased one, so a caller that ignores the status cannot mistake it for a
    // sample.
    CHECK(exhausted_value == 0xD15EA5E);

    // The bound is exactly kMaxRejectionDraws attempts: the last one still counts.
    std::uint64_t late_value = 0;
    CHECK(detail::sample_inclusive_from(
              [](std::uint32_t slot)
              { return slot + 1 < kMaxRejectionDraws ? std::uint64_t{7} : std::uint64_t{3}; }, 10,
              14, late_value) == ContractStatus::ok);
    CHECK(late_value == 13);
    std::uint64_t never_value = 0;
    CHECK(detail::sample_inclusive_from(
              [](std::uint32_t slot)
              { return slot < kMaxRejectionDraws ? std::uint64_t{7} : std::uint64_t{3}; }, 10, 14,
              never_value) == ContractStatus::sampling_exhausted);

    // A full-width span has an all-ones mask, so it accepts on the first attempt
    // and cannot reach exhaustion.
    std::uint64_t full_span = 0;
    CHECK(sample_inclusive(key, 0, (std::numeric_limits<std::uint64_t>::max)(), full_span) ==
          ContractStatus::ok);
    CHECK(full_span == sample_bits(key));

    // A whole schedule regenerates bit-exactly from the determining tuple.
    std::uint64_t first_pass = 0;
    std::uint64_t second_pass = 0;
    for (TrialOrdinal trial = 0; trial < 64; ++trial)
    {
        std::uint64_t black = 0;
        std::uint64_t content = 0;
        CHECK(sample_exclusive(DrawKey{0xABCDEF, 1, trial, 0}, 0, 1'000'000, black) ==
              ContractStatus::ok);
        CHECK(sample_exclusive(DrawKey{0xABCDEF, 1, trial, 1}, 0, 2'000'000, content) ==
              ContractStatus::ok);
        first_pass = sampler_mix64(first_pass + black) ^ content;
    }
    for (TrialOrdinal trial = 0; trial < 64; ++trial)
    {
        std::uint64_t black = 0;
        std::uint64_t content = 0;
        CHECK(sample_exclusive(DrawKey{0xABCDEF, 1, trial, 0}, 0, 1'000'000, black) ==
              ContractStatus::ok);
        CHECK(sample_exclusive(DrawKey{0xABCDEF, 1, trial, 1}, 0, 2'000'000, content) ==
              ContractStatus::ok);
        second_pass = sampler_mix64(second_pass + black) ^ content;
    }
    CHECK(first_pass == second_pass);

    // Two cases, and no third: an unsupported sampler version is not resampled.
    CHECK(sampler_version_supported(kSamplerVersion1));
    CHECK(sampler_version_supported(kCurrentSamplerVersion));
    CHECK(!sampler_version_supported(0));
    CHECK(!sampler_version_supported(kSamplerVersion1 + 1));
    CHECK(replay_authority(kSamplerVersion1) == ReplayAuthority::regenerate);
    CHECK(replay_authority(kSamplerVersion1 + 1) == ReplayAuthority::recorded_schedule);
    CHECK(replay_authority(0) == ReplayAuthority::recorded_schedule);

    ScheduleIdentity identity{};
    identity.seed = 42;
    identity.configuration_fingerprint = 7;
    CHECK(validate(identity) == ContractStatus::ok);
    const std::uint64_t base = schedule_fingerprint(identity);
    CHECK(schedule_fingerprint(identity) == base);
    ScheduleIdentity other = identity;
    other.seed = 43;
    CHECK(schedule_fingerprint(other) != base);
    other = identity;
    other.sampler_version = kSamplerVersion1 + 1;
    CHECK(schedule_fingerprint(other) != base);
    other = identity;
    other.configuration_fingerprint = 8;
    CHECK(schedule_fingerprint(other) != base);
    other = identity;
    other.catalog_fingerprint = 1;
    CHECK(schedule_fingerprint(other) != base);

    // A version of zero names no sampler at all, and that is the only identity
    // the contract rejects: an unsupported one still has a replay path.
    ScheduleIdentity versionless = identity;
    versionless.sampler_version = 0;
    CHECK(validate(versionless) == ContractStatus::identity_missing);
    ScheduleIdentity unsupported = identity;
    unsupported.sampler_version = 4'000;
    CHECK(validate(unsupported) == ContractStatus::ok);

    FingerprintAccumulator forward;
    forward.absorb(1);
    forward.absorb(2);
    FingerprintAccumulator backward;
    backward.absorb(2);
    backward.absorb(1);
    CHECK(forward.value() != backward.value());
    FingerprintAccumulator repeat;
    repeat.absorb(1);
    repeat.absorb(2);
    CHECK(repeat.value() == forward.value());

    constexpr std::array<std::byte, 3> content{std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    constexpr std::array<std::byte, 3> altered{std::byte{'a'}, std::byte{'b'}, std::byte{'d'}};
    FingerprintAccumulator text;
    text.absorb_bytes(std::span<const std::byte>(content));
    FingerprintAccumulator changed;
    changed.absorb_bytes(std::span<const std::byte>(altered));
    CHECK(text.value() != changed.value());

    ScheduleDraw draw{};
    draw.trial.ordinal = 4;
    draw.stream = 1;
    draw.value = 99;
    CHECK(validate(draw) == ContractStatus::ok);
    draw.sampler_version = 0;
    CHECK(validate(draw) == ContractStatus::identity_missing);
    return 0;
}

int check_events()
{
    ExperimentEvent event{};
    CHECK(validate(event) == ContractStatus::identity_missing);
    event.kind = ExperimentEventKind::trial_start;
    CHECK(validate(event) == ContractStatus::identity_missing);
    event.paradigm = 5;
    CHECK(validate(event) == ContractStatus::ok);

    StateTransition transition{};
    CHECK(validate(transition) == ContractStatus::identity_missing);
    transition.paradigm = 5;
    transition.from_state = 1;
    transition.to_state = 2;
    CHECK(validate(transition) == ContractStatus::ok);
    transition.to_state = 1;
    CHECK(validate(transition) == ContractStatus::ok);

    SelectionEvent selection{};
    selection.paradigm = 6;
    selection.intended_id = 12;
    selection.selected_id = 12;
    selection.correct = true;
    CHECK(validate(selection) == ContractStatus::ok);

    // A misclick: a real selection of the wrong item.
    selection.selected_id = 13;
    selection.correct = false;
    CHECK(validate(selection) == ContractStatus::ok);
    selection.correct = true;
    CHECK(validate(selection) == ContractStatus::outcome_invalid);

    // Nothing selectable under the pointer is still a selection attempt.
    selection.selected_id = kUnsetTargetId;
    selection.correct = false;
    CHECK(validate(selection) == ContractStatus::ok);

    selection.intended_id = kUnsetTargetId;
    CHECK(validate(selection) == ContractStatus::identity_missing);
    selection.intended_id = 12;

    // Dwell and duration agree in both directions.
    selection.kind = SelectionKind::dwell;
    CHECK(validate(selection) == ContractStatus::outcome_invalid);
    selection.dwell_ns = 250'000;
    CHECK(validate(selection) == ContractStatus::ok);
    selection.kind = SelectionKind::discrete;
    CHECK(validate(selection) == ContractStatus::outcome_invalid);
    selection.dwell_ns = 0;
    CHECK(validate(selection) == ContractStatus::ok);

    TrialRecord record{};
    CHECK(validate(record) == ContractStatus::identity_missing);
    record.paradigm = 2;
    record.interval = TimeInterval{100, 100};
    CHECK(validate(record) == ContractStatus::ok);
    // A pending trial has no end yet, and says so with an empty interval.
    record.interval = TimeInterval{100, 200};
    CHECK(validate(record) == ContractStatus::outcome_invalid);
    record.outcome = TrialOutcome::success;
    CHECK(validate(record) == ContractStatus::ok);
    record.interval = TimeInterval{200, 100};
    CHECK(validate(record) == ContractStatus::interval_inverted);

    // Enumerated fields are checked against the declared set, not merely against
    // the one enumerator a rule happens to name. A number outside the set is
    // corrupt or comes from a newer build, and either way is unreadable here.
    CHECK(experiment_event_kind_declared(ExperimentEventKind::unspecified));
    CHECK(experiment_event_kind_declared(ExperimentEventKind::paradigm_marker));
    CHECK(!experiment_event_kind_declared(static_cast<ExperimentEventKind>(255)));
    event.kind = static_cast<ExperimentEventKind>(255);
    CHECK(validate(event) == ContractStatus::enum_undeclared);

    CHECK(selection_kind_declared(SelectionKind::discrete));
    CHECK(selection_kind_declared(SelectionKind::dwell));
    CHECK(!selection_kind_declared(static_cast<SelectionKind>(255)));
    // An undeclared kind used to be read as "not dwell", which quietly turned a
    // corrupt record into a well-formed discrete selection.
    selection.kind = static_cast<SelectionKind>(255);
    CHECK(validate(selection) == ContractStatus::enum_undeclared);
    return 0;
}

int check_presentation()
{
    CHECK(cue_kind_declared(CueKind::none));
    CHECK(cue_kind_declared(CueKind::black));
    CHECK(cue_kind_declared(CueKind::fixation_cross));
    CHECK(cue_kind_declared(CueKind::text_content));
    CHECK(!cue_kind_declared(static_cast<CueKind>(200)));

    PresentationRequest request{};
    CHECK(validate(request) == ContractStatus::identity_missing);
    request.paradigm = 3;
    request.requested_ns = 1'000;
    request.onset_ns = 1'000;
    request.duration_ns = 500'000;
    request.cue = CueKind::black;
    CHECK(validate(request) == ContractStatus::ok);

    // The semantic request time and the intended onset are separate values, and
    // an onset may not precede the decision that produced it.
    request.onset_ns = 999;
    CHECK(validate(request) == ContractStatus::time_regressed);
    request.onset_ns = 2'000;
    CHECK(validate(request) == ContractStatus::ok);

    request.valid_until_ns = 1'000;
    CHECK(validate(request) == ContractStatus::expiry_before_generation);
    request.valid_until_ns = 1'500;
    CHECK(validate(request) == ContractStatus::expiry_before_generation);
    request.valid_until_ns = 2'500;
    CHECK(validate(request) == ContractStatus::ok);
    request.valid_until_ns = kNoExpiryNs;
    CHECK(validate(request) == ContractStatus::ok);

    // A cue that stands for content needs an identifier; one that does not must
    // not carry one, so "which stimulus is showing" has exactly one answer.
    request.stimulus_id = 17;
    CHECK(validate(request) == ContractStatus::presentation_invalid);
    request.cue = CueKind::text_content;
    CHECK(validate(request) == ContractStatus::ok);
    request.stimulus_id = kUnsetStimulusId;
    CHECK(validate(request) == ContractStatus::identity_missing);
    request.stimulus_id = 17;

    request.cue = static_cast<CueKind>(200);
    CHECK(validate(request) == ContractStatus::enum_undeclared);
    request.cue = CueKind::text_content;

    request.onset_ns = (std::numeric_limits<ExperimentTimeNs>::max)() - 1;
    request.duration_ns = 4;
    CHECK(validate(request) == ContractStatus::duration_overflow);

    PresentationState state{};
    state.paradigm = 3;
    state.cue = CueKind::fixation_cross;
    CHECK(validate(state) == ContractStatus::ok);
    state.stimulus_id = 4;
    CHECK(validate(state) == ContractStatus::presentation_invalid);
    state.cue = CueKind::text_content;
    CHECK(validate(state) == ContractStatus::ok);

    // The controller layer never produces a presented outcome: only a presenter
    // can report one, and a session without a presenter carries not_reported
    // with no presentation time.
    PresentationOutcome outcome{};
    outcome.requested_ns = 5'000;
    CHECK(outcome.status == PresentationStatus::not_reported);
    CHECK(validate(outcome) == ContractStatus::ok);
    outcome.presented_ns = 5'100;
    CHECK(validate(outcome) == ContractStatus::presentation_invalid);
    outcome.status = PresentationStatus::presented;
    CHECK(validate(outcome) == ContractStatus::ok);
    outcome.presented_ns = 4'999;
    CHECK(validate(outcome) == ContractStatus::time_regressed);
    outcome.status = PresentationStatus::skipped;
    CHECK(validate(outcome) == ContractStatus::presentation_invalid);
    outcome.presented_ns = 0;
    CHECK(validate(outcome) == ContractStatus::ok);
    outcome.status = PresentationStatus::expired;
    CHECK(validate(outcome) == ContractStatus::ok);

    // A status read back as a number naming no enumerator is rejected, not
    // silently sorted into whichever branch the switch happens to fall through.
    CHECK(presentation_status_declared(PresentationStatus::not_reported));
    CHECK(presentation_status_declared(PresentationStatus::expired));
    CHECK(!presentation_status_declared(static_cast<PresentationStatus>(255)));
    outcome.status = static_cast<PresentationStatus>(255);
    CHECK(validate(outcome) == ContractStatus::enum_undeclared);

    // The request and the report it is about are named by separate fields, so a
    // trace can join them without guessing which ordinal a field meant.
    PresentationOutcome joined{};
    joined.sequence = 41;
    joined.request_sequence = 12;
    CHECK(joined.sequence != joined.request_sequence);
    return 0;
}

int check_command()
{
    CommandSpace space{};
    CHECK(validate(space) == ContractStatus::identity_missing);
    space.id = 1;
    CHECK(validate(space) == ContractStatus::dimension_invalid);
    space.dim = 2;
    CHECK(validate(space) == ContractStatus::identity_missing);
    space.frame = CommandFrame::workspace_2d;
    CHECK(validate(space) == ContractStatus::identity_missing);
    space.axes[0] = CommandAxis{CommandAxisName::x, CommandUnit::metres_per_second};
    space.axes[1] = CommandAxis{CommandAxisName::y, CommandUnit::metres_per_second};
    CHECK(validate(space) == ContractStatus::ok);

    // An axis described beyond the declared dimension is a contradiction, not a
    // harmless leftover: a transform reading it would read a unit nobody meant.
    space.axes[2] = CommandAxis{CommandAxisName::z, CommandUnit::metres_per_second};
    CHECK(validate(space) == ContractStatus::dimension_invalid);
    space.axes[2] = CommandAxis{};
    CHECK(validate(space) == ContractStatus::ok);

    space.dim = static_cast<std::uint8_t>(kMaxCommandDim + 1);
    CHECK(validate(space) == ContractStatus::dimension_invalid);
    space.dim = 2;

    CommandRequest request{};
    CHECK(validate(request) == ContractStatus::identity_missing);
    request.space = 1;
    CHECK(validate(request) == ContractStatus::dimension_invalid);
    request.dim = 2;
    request.values[0] = 0.25;
    request.values[1] = -0.5;
    CHECK(validate(request) == ContractStatus::ok);

    // A realtime command value must be finite.
    request.values[1] = std::numeric_limits<double>::quiet_NaN();
    CHECK(validate(request) == ContractStatus::value_not_finite);
    request.values[1] = std::numeric_limits<double>::infinity();
    CHECK(validate(request) == ContractStatus::value_not_finite);
    request.values[1] = -std::numeric_limits<double>::infinity();
    CHECK(validate(request) == ContractStatus::value_not_finite);
    request.values[1] = -0.5;
    CHECK(validate(request) == ContractStatus::ok);

    // Unused slots stay exactly zero, so equal commands have equal bytes.
    request.values[5] = 1.0;
    CHECK(validate(request) == ContractStatus::dimension_invalid);
    request.values[5] = 0.0;
    CHECK(validate(request) == ContractStatus::ok);

    request.generated_ns = 1'000;
    request.valid_until_ns = 1'000;
    CHECK(validate(request) == ContractStatus::expiry_before_generation);
    request.valid_until_ns = 1'001;
    CHECK(validate(request) == ContractStatus::ok);
    request.valid_until_ns = kNoExpiryNs;
    CHECK(validate(request) == ContractStatus::ok);

    CHECK(validate_against(request, space) == ContractStatus::ok);
    request.dim = 1;
    CHECK(validate_against(request, space) == ContractStatus::dimension_invalid);
    request.dim = 2;
    request.space = 2;
    CHECK(validate_against(request, space) == ContractStatus::identity_missing);
    request.space = 1;
    CHECK(validate_against(request, space) == ContractStatus::ok);

    // The outcome states what the runtime could prove and nothing more.
    CommandOutcome outcome{};
    outcome.generated_ns = 1'000;
    outcome.submitted_ns = 1'000;
    CHECK(outcome.application == CommandApplication::not_submitted);
    CHECK(validate(outcome) == ContractStatus::ok);
    outcome.status_code = 3;
    CHECK(validate(outcome) == ContractStatus::outcome_invalid);
    outcome.application = CommandApplication::rejected;
    CHECK(validate(outcome) == ContractStatus::ok);
    outcome.application = CommandApplication::accepted;
    outcome.submitted_ns = 1'200;
    CHECK(validate(outcome) == ContractStatus::ok);
    outcome.submitted_ns = 999;
    CHECK(validate(outcome) == ContractStatus::time_regressed);
    outcome.submitted_ns = 1'200;
    outcome.application = CommandApplication::expired;
    CHECK(validate(outcome) == ContractStatus::ok);

    // The report and the request it is about carry separate ordinals.
    outcome.sequence = 90;
    outcome.request_sequence = 12;
    CHECK(outcome.sequence != outcome.request_sequence);

    // Frame, axis name, unit, and application status are all checked against
    // their declared sets. "Not the unspecified one" is not the same question as
    // "one this build declares", and only the second one is answerable.
    CHECK(command_frame_declared(CommandFrame::unspecified));
    CHECK(command_frame_declared(CommandFrame::device_native));
    CHECK(!command_frame_declared(static_cast<CommandFrame>(255)));
    CHECK(command_axis_name_declared(CommandAxisName::grasp));
    CHECK(!command_axis_name_declared(static_cast<CommandAxisName>(255)));
    CHECK(command_unit_declared(CommandUnit::radians_per_second));
    CHECK(!command_unit_declared(static_cast<CommandUnit>(255)));
    CHECK(command_application_declared(CommandApplication::not_submitted));
    CHECK(command_application_declared(CommandApplication::expired));
    CHECK(!command_application_declared(static_cast<CommandApplication>(255)));

    space.frame = static_cast<CommandFrame>(255);
    CHECK(validate(space) == ContractStatus::enum_undeclared);
    space.frame = CommandFrame::workspace_2d;
    space.axes[0] = CommandAxis{static_cast<CommandAxisName>(255), CommandUnit::metres_per_second};
    CHECK(validate(space) == ContractStatus::enum_undeclared);
    space.axes[0] = CommandAxis{CommandAxisName::x, static_cast<CommandUnit>(255)};
    CHECK(validate(space) == ContractStatus::enum_undeclared);
    space.axes[0] = CommandAxis{CommandAxisName::x, CommandUnit::metres_per_second};
    CHECK(validate(space) == ContractStatus::ok);

    // Previously any value other than not_submitted returned ok immediately.
    outcome.application = static_cast<CommandApplication>(255);
    CHECK(validate(outcome) == ContractStatus::enum_undeclared);
    return 0;
}

int check_replay()
{
    ExperimentSnapshot session{};
    CHECK(validate(session) == ContractStatus::identity_missing);
    session.paradigm = 9;
    session.origin_ns = 1'000;
    session.schedule.seed = 77;
    session.first_trial_ordinal = 20;
    session.first_sequence = 300;
    CHECK(validate(session) == ContractStatus::ok);
    session.schedule.sampler_version = 0;
    CHECK(validate(session) == ContractStatus::identity_missing);

    DecisionSnapshot decision{};
    CHECK(validate(decision) == ContractStatus::identity_missing);
    decision.paradigm = 9;
    decision.schedule.seed = 77;
    decision.stream = 2;
    decision.trial.ordinal = 20;
    decision.draw_cursor = 4;
    CHECK(validate(decision) == ContractStatus::ok);

    // The cursor is what makes a decision re-derivable without replaying
    // everything before it.
    const DrawKey next = next_draw_key(decision);
    CHECK(next.seed == 77);
    CHECK(next.stream == 2);
    CHECK(next.trial_idx == 20);
    CHECK(next.draw == 4);
    std::uint64_t direct = 0;
    std::uint64_t reconstructed = 0;
    CHECK(sample_index(DrawKey{77, 2, 20, 4}, 8, direct) == ContractStatus::ok);
    CHECK(sample_index(next, 8, reconstructed) == ContractStatus::ok);
    CHECK(direct == reconstructed);
    return 0;
}

int check_abnormal()
{
    // Declaration is a range test over an append-only set, so the last
    // enumerator is the bound and one past it is not a condition.
    CHECK(abnormal_condition_declared(AbnormalCondition::presentation_report_unmatched));
    CHECK(!abnormal_condition_declared(static_cast<AbnormalCondition>(
        static_cast<std::uint16_t>(AbnormalCondition::presentation_report_unmatched) + 1)));
    CHECK(abnormal_policy_declared(AbnormalPolicy::abort_session));
    CHECK(!abnormal_policy_declared(static_cast<AbnormalPolicy>(3)));
    CHECK(abnormal_response_declared(AbnormalResponse::session_aborted));
    CHECK(!abnormal_response_declared(static_cast<AbnormalResponse>(5)));

    // The severity order the whole layer relies on.
    CHECK(escalate(AbnormalPolicy::record, AbnormalPolicy::abort_trial) ==
          AbnormalPolicy::abort_trial);
    CHECK(escalate(AbnormalPolicy::abort_session, AbnormalPolicy::abort_trial) ==
          AbnormalPolicy::abort_session);
    CHECK(escalate(AbnormalPolicy::record, AbnormalPolicy::record) == AbnormalPolicy::record);

    // Which responses leave a trial usable. This is the whole of the "no silent
    // success" rule, stated once so that no paradigm restates it differently.
    CHECK(abnormal_response_admits_trial(AbnormalResponse::recorded));
    CHECK(abnormal_response_admits_trial(AbnormalResponse::input_refused));
    CHECK(!abnormal_response_admits_trial(AbnormalResponse::trial_invalidated));
    CHECK(!abnormal_response_admits_trial(AbnormalResponse::trial_aborted));
    CHECK(!abnormal_response_admits_trial(AbnormalResponse::session_aborted));

    // The default set is the conservative one, and the defaults are part of the
    // contract rather than an implementation detail: a caller that configures
    // nothing gets a run that ends a trial whose continuity broke and ends a
    // run whose recorder failed.
    const AbnormalPolicySet defaults{};
    CHECK(policy_for(defaults, AbnormalCondition::decoded_command_invalid) ==
          AbnormalPolicy::abort_trial);
    CHECK(policy_for(defaults, AbnormalCondition::input_schema_mismatch) ==
          AbnormalPolicy::abort_trial);
    CHECK(policy_for(defaults, AbnormalCondition::source_discontinuity) ==
          AbnormalPolicy::abort_trial);
    CHECK(policy_for(defaults, AbnormalCondition::input_gap) == AbnormalPolicy::abort_trial);
    CHECK(policy_for(defaults, AbnormalCondition::input_stale) == AbnormalPolicy::abort_trial);
    CHECK(policy_for(defaults, AbnormalCondition::deadline_missed) == AbnormalPolicy::abort_trial);
    CHECK(policy_for(defaults, AbnormalCondition::presentation_failed) ==
          AbnormalPolicy::abort_trial);
    CHECK(policy_for(defaults, AbnormalCondition::presentation_evidence_missing) ==
          AbnormalPolicy::record);
    CHECK(policy_for(defaults, AbnormalCondition::recorder_fault) == AbnormalPolicy::abort_session);
    CHECK(policy_for(defaults, AbnormalCondition::actuator_fault) == AbnormalPolicy::abort_session);
    CHECK(policy_for(defaults, AbnormalCondition::runtime_fault) == AbnormalPolicy::abort_session);

    // Four conditions ignore the set entirely. A configuration that tried to
    // make a dropped monitoring frame end a run does not get to.
    AbnormalPolicySet severe{};
    severe.decoded_command_invalid = AbnormalPolicy::abort_session;
    severe.input_discontinuity = AbnormalPolicy::abort_session;
    severe.presentation_failed = AbnormalPolicy::abort_session;
    severe.presentation_evidence_missing = AbnormalPolicy::abort_session;
    severe.acquisition_fault = AbnormalPolicy::abort_session;
    CHECK(policy_for(severe, AbnormalCondition::observer_frame_drop) == AbnormalPolicy::record);
    CHECK(policy_for(severe, AbnormalCondition::input_after_terminal) == AbnormalPolicy::record);
    // A presenter's report the run cannot match to a request it made is never
    // allowed to end anything: it is not evidence about a trial at all.
    CHECK(policy_for(severe, AbnormalCondition::presentation_report_unmatched) ==
          AbnormalPolicy::record);
    CHECK(policy_for(severe, AbnormalCondition::emergency_stop) == AbnormalPolicy::abort_session);

    // A policy set is validated where a run freezes it, so an undeclared
    // severity cannot reach a record.
    CHECK(validate(AbnormalPolicySet{}) == ContractStatus::ok);
    CHECK(validate(severe) == ContractStatus::ok);
    AbnormalPolicySet undeclared{};
    undeclared.presentation_failed = static_cast<AbnormalPolicy>(9);
    CHECK(validate(undeclared) == ContractStatus::enum_undeclared);
    AbnormalPolicySet undeclared_acquisition{};
    undeclared_acquisition.acquisition_fault = static_cast<AbnormalPolicy>(200);
    CHECK(validate(undeclared_acquisition) == ContractStatus::enum_undeclared);

    // And a permissive configuration does not soften the two that are what they
    // are: an emergency stop ends the run whatever else is configured.
    AbnormalPolicySet permissive{};
    permissive.decoded_command_invalid = AbnormalPolicy::record;
    permissive.input_discontinuity = AbnormalPolicy::record;
    permissive.acquisition_fault = AbnormalPolicy::record;
    CHECK(policy_for(permissive, AbnormalCondition::emergency_stop) ==
          AbnormalPolicy::abort_session);
    CHECK(policy_for(permissive, AbnormalCondition::source_discontinuity) ==
          AbnormalPolicy::record);
    // An unset condition is the one case where not knowing is treated as the
    // severe answer.
    CHECK(policy_for(permissive, AbnormalCondition::unspecified) == AbnormalPolicy::abort_session);

    AbnormalEvent event{};
    CHECK(validate(event) == ContractStatus::identity_missing);
    event.paradigm = 7;
    // A record that does not say what it is about is not a record.
    CHECK(validate(event) == ContractStatus::identity_missing);
    event.condition = AbnormalCondition::source_discontinuity;
    CHECK(validate(event) == ContractStatus::ok);

    event.condition = static_cast<AbnormalCondition>(9999);
    CHECK(validate(event) == ContractStatus::enum_undeclared);
    event.condition = AbnormalCondition::source_discontinuity;
    event.response = static_cast<AbnormalResponse>(9);
    CHECK(validate(event) == ContractStatus::enum_undeclared);

    // A response that names a trial, on a record carrying none.
    event.response = AbnormalResponse::trial_aborted;
    event.policy = AbnormalPolicy::abort_trial;
    CHECK(validate(event) == ContractStatus::outcome_invalid);
    event.has_trial = true;
    event.trial.ordinal = 0;
    CHECK(validate(event) == ContractStatus::ok);

    // A response more severe than the policy recorded beside it. One of the two
    // is not what happened, and the record cannot say which.
    event.policy = AbnormalPolicy::record;
    CHECK(validate(event) == ContractStatus::outcome_invalid);
    event.policy = AbnormalPolicy::abort_trial;
    event.response = AbnormalResponse::session_aborted;
    CHECK(validate(event) == ContractStatus::outcome_invalid);
    event.policy = AbnormalPolicy::abort_session;
    CHECK(validate(event) == ContractStatus::ok);

    // Refusing an input is not on the severity scale: a value that cannot be
    // applied is not applied whatever the configuration says.
    event.policy = AbnormalPolicy::record;
    event.response = AbnormalResponse::input_refused;
    CHECK(validate(event) == ContractStatus::ok);

    // Every declared condition has a distinct, stable name, and an undeclared
    // one is named rather than read past the end of a table.
    constexpr std::array kConditions{
        AbnormalCondition::unspecified,
        AbnormalCondition::decoded_command_invalid,
        AbnormalCondition::input_schema_mismatch,
        AbnormalCondition::input_stale,
        AbnormalCondition::source_discontinuity,
        AbnormalCondition::input_gap,
        AbnormalCondition::deadline_missed,
        AbnormalCondition::observer_frame_drop,
        AbnormalCondition::recorder_fault,
        AbnormalCondition::actuator_fault,
        AbnormalCondition::runtime_fault,
        AbnormalCondition::presentation_failed,
        AbnormalCondition::presentation_evidence_missing,
        AbnormalCondition::input_after_terminal,
        AbnormalCondition::emergency_stop,
        AbnormalCondition::presentation_report_unmatched,
    };
    CHECK(kConditions.size() ==
          static_cast<std::size_t>(AbnormalCondition::presentation_report_unmatched) + 1);
    for (std::size_t left = 0; left < kConditions.size(); ++left)
    {
        const std::string_view name{abnormal_condition_name(kConditions[left])};
        CHECK(!name.empty());
        CHECK(name != "undeclared");
        for (std::size_t right = left + 1; right < kConditions.size(); ++right)
            CHECK(name != std::string_view{abnormal_condition_name(kConditions[right])});
    }
    CHECK(std::string_view{abnormal_condition_name(static_cast<AbnormalCondition>(9999))} ==
          "undeclared");
    return 0;
}

int check_status_messages()
{
    for (const ContractStatus status : {ContractStatus::ok,
                                        ContractStatus::time_regressed,
                                        ContractStatus::interval_inverted,
                                        ContractStatus::duration_overflow,
                                        ContractStatus::dimension_invalid,
                                        ContractStatus::value_not_finite,
                                        ContractStatus::expiry_before_generation,
                                        ContractStatus::range_empty,
                                        ContractStatus::sampling_exhausted,
                                        ContractStatus::ordinal_exhausted,
                                        ContractStatus::identity_missing,
                                        ContractStatus::enum_undeclared,
                                        ContractStatus::presentation_invalid,
                                        ContractStatus::outcome_invalid,
                                        ContractStatus::assistance_out_of_range,
                                        ContractStatus::domain_mask_invalid,
                                        ContractStatus::version_unsupported,
                                        ContractStatus::numerical_failure,
                                        ContractStatus::parameter_out_of_range,
                                        ContractStatus::target_set_invalid,
                                        ContractStatus::not_running,
                                        ContractStatus::already_running})
    {
        const char* message = contract_status_message(status);
        CHECK(message != nullptr);
        CHECK(message[0] != '\0');
    }
    CHECK(contract_status_message(static_cast<ContractStatus>(250)) != nullptr);
    return 0;
}

int run()
{
    // Everything below runs on values only. A single allocation here would mean
    // some record had grown owning storage, which is exactly what a bounded
    // trace on a realtime path cannot afford.
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);

    if (const int failure = check_time(); failure != 0)
        return failure;
    if (const int failure = check_identity(); failure != 0)
        return failure;
    if (const int failure = check_schedule(); failure != 0)
        return failure;
    if (const int failure = check_events(); failure != 0)
        return failure;
    if (const int failure = check_presentation(); failure != 0)
        return failure;
    if (const int failure = check_command(); failure != 0)
        return failure;
    if (const int failure = check_replay(); failure != 0)
        return failure;
    if (const int failure = check_abnormal(); failure != 0)
        return failure;
    if (const int failure = check_status_messages(); failure != 0)
        return failure;

    CHECK(allocations.load(std::memory_order_relaxed) == baseline);
    return 0;
}

} // namespace

int main()
{
    try
    {
        return run();
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
