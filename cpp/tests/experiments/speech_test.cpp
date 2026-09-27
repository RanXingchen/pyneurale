/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/speech.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>

#include "allocation_counter.h"
#include "check_counts.h"

namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::speech;

static_assert(std::is_trivially_copyable_v<SpeechCueConfig>);
static_assert(std::is_trivially_copyable_v<SpeechCatalog>);
static_assert(std::is_trivially_copyable_v<SpeechTrialSchedule>);
static_assert(std::is_trivially_copyable_v<SpeechTimeline>);

int failures = 0;

constexpr DurationNs kBlackBound = 500'000'000;
constexpr DurationNs kCrossBound = 300'000'000;
constexpr DurationNs kContentBound = 2'000'000'000;
constexpr ScheduleSeed kSeed = 0xC0FFEE1234567890ULL;

void set_text(SpeechStimulus& stimulus, const char* text) noexcept
{
    const std::size_t length = std::strlen(text);
    stimulus.text = {};
    for (std::size_t i = 0; i < length && i < kMaxSpeechTextBytes; ++i)
        stimulus.text[i] = text[i];
    stimulus.text_length = static_cast<std::uint8_t>(length);
}

SpeechCatalog make_catalog(std::uint16_t count)
{
    SpeechCatalog catalog{};
    catalog.count = count;
    for (std::uint16_t i = 0; i < count; ++i)
    {
        SpeechStimulus& stimulus = catalog.entries[i];
        stimulus.id = static_cast<StimulusId>(i + 1);
        stimulus.label = static_cast<std::uint32_t>(i % 4 + 1);
        stimulus.metadata = i;
        stimulus.content = SpeechContentKind::text;
        char text[16] = {};
        text[0] = 'w';
        text[1] = 'o';
        text[2] = 'r';
        text[3] = 'd';
        text[4] = static_cast<char>('a' + i % 26);
        set_text(stimulus, text);
    }
    return catalog;
}

SpeechCueConfig seeded_config(bool cross_enabled, StimulusOrderPolicy order, TrialOrdinal n_trials,
                              std::uint16_t n_stimuli)
{
    SpeechCueConfig config{};
    config.black_bound_ns = kBlackBound;
    config.cross_bound_ns = cross_enabled ? kCrossBound : 0;
    config.content_bound_ns = kContentBound;
    config.inter_trial_ns = 0;
    config.seed = kSeed;
    config.n_trials = n_trials;
    config.schedule = SpeechScheduleKind::seeded;
    config.stimulus_order = order;
    config.sampler_version = kCurrentSamplerVersion;
    config.cross_enabled = cross_enabled;
    config.n_stimuli = n_stimuli;
    for (std::uint16_t i = 0; i < n_stimuli; ++i)
        config.stimuli[i] = static_cast<StimulusId>(i + 1);
    return config;
}

// Freeze a seeded configuration into the explicit one that carries the same
// realized values, which is what a session recorded and replayed looks like.
SpeechCueConfig freeze(const SpeechCueConfig& seeded)
{
    SpeechCueConfig frozen{};
    frozen.black_bound_ns = seeded.black_bound_ns;
    frozen.cross_bound_ns = seeded.cross_bound_ns;
    frozen.content_bound_ns = seeded.content_bound_ns;
    frozen.inter_trial_ns = seeded.inter_trial_ns;
    frozen.seed = seeded.seed;
    frozen.n_trials = seeded.n_trials;
    frozen.schedule = SpeechScheduleKind::explicit_sequence;
    frozen.stimulus_order = StimulusOrderPolicy::unspecified;
    frozen.sampler_version = seeded.sampler_version;
    frozen.cross_enabled = seeded.cross_enabled;
    frozen.n_explicit = static_cast<std::uint16_t>(seeded.n_trials);
    for (TrialOrdinal ordinal = 0; ordinal < seeded.n_trials; ++ordinal)
    {
        SpeechTrialSchedule entry{};
        if (prepare_trial(seeded, ordinal, entry) != ContractStatus::ok)
            ++failures;
        frozen.explicit_schedule[static_cast<std::size_t>(ordinal)] = entry;
    }
    return frozen;
}

// What a plausibility check would have accepted: every duration strictly inside
// its bound, and the cross and sampler agreeing with the configuration. Used to
// show that the substitutes rejected below are rejected for *ownership*, not for
// being malformed.
bool inside_bounds(const SpeechTrialSchedule& schedule, const SpeechCueConfig& config) noexcept
{
    return schedule.cross_enabled == config.cross_enabled &&
           schedule.sampler_version == config.sampler_version && schedule.black_duration_ns != 0 &&
           schedule.black_duration_ns < config.black_bound_ns &&
           schedule.content_duration_ns != 0 &&
           schedule.content_duration_ns < config.content_bound_ns &&
           (!config.cross_enabled || (schedule.cross_duration_ns != 0 &&
                                      schedule.cross_duration_ns < config.cross_bound_ns));
}

bool identical(const SpeechTrialSchedule& left, const SpeechTrialSchedule& right) noexcept
{
    return left.ordinal == right.ordinal && left.black_duration_ns == right.black_duration_ns &&
           left.cross_duration_ns == right.cross_duration_ns &&
           left.content_duration_ns == right.content_duration_ns &&
           left.stimulus_id == right.stimulus_id && left.sampler_version == right.sampler_version &&
           left.cross_enabled == right.cross_enabled;
}

// ---------------------------------------------------------------------------

void test_sampled_durations_stay_strictly_inside_their_bounds()
{
    const SpeechCueConfig config =
        seeded_config(true, StimulusOrderPolicy::random_with_replacement, 512, 8);
    bool saw_low_black = false;
    bool saw_high_black = false;
    for (TrialOrdinal ordinal = 0; ordinal < config.n_trials; ++ordinal)
    {
        SpeechTrialSchedule schedule{};
        CHECK(prepare_trial(config, ordinal, schedule) == ContractStatus::ok);
        CHECK(schedule.ordinal == ordinal);
        CHECK(schedule.sampler_version == kCurrentSamplerVersion);
        // Open at both ends, on every single trial: never zero, never the bound.
        CHECK(schedule.black_duration_ns > 0);
        CHECK(schedule.black_duration_ns < config.black_bound_ns);
        CHECK(schedule.cross_duration_ns > 0);
        CHECK(schedule.cross_duration_ns < config.cross_bound_ns);
        CHECK(schedule.content_duration_ns > 0);
        CHECK(schedule.content_duration_ns < config.content_bound_ns);
        CHECK(validate_against(schedule, config) == ContractStatus::ok);
        if (schedule.black_duration_ns < config.black_bound_ns / 4)
            saw_low_black = true;
        if (schedule.black_duration_ns > config.black_bound_ns * 3 / 4)
            saw_high_black = true;
    }
    // Not a uniformity test -- the sampler's own suite owns that -- but a
    // constant sampler would pass every bound assertion above, so the range has
    // to be shown to be used rather than merely respected.
    CHECK(saw_low_black);
    CHECK(saw_high_black);

    // The narrowest admissible bound: (0, 2) holds exactly one integer, so the
    // realized duration is forced and the exclusive mapping is pinned rather
    // than merely constrained.
    SpeechCueConfig narrow = seeded_config(true, StimulusOrderPolicy::sequential, 8, 2);
    narrow.black_bound_ns = 2;
    narrow.cross_bound_ns = 2;
    narrow.content_bound_ns = 2;
    CHECK(validate(narrow) == ContractStatus::ok);
    for (TrialOrdinal ordinal = 0; ordinal < narrow.n_trials; ++ordinal)
    {
        SpeechTrialSchedule schedule{};
        CHECK(prepare_trial(narrow, ordinal, schedule) == ContractStatus::ok);
        CHECK(schedule.black_duration_ns == 1);
        CHECK(schedule.cross_duration_ns == 1);
        CHECK(schedule.content_duration_ns == 1);
    }
}

void test_cross_is_configuration_and_never_a_draw()
{
    const SpeechCueConfig enabled = seeded_config(true, StimulusOrderPolicy::sequential, 256, 4);
    SpeechCueConfig disabled = enabled;
    disabled.cross_enabled = false;
    disabled.cross_bound_ns = 0;
    CHECK(validate(enabled) == ContractStatus::ok);
    CHECK(validate(disabled) == ContractStatus::ok);

    for (TrialOrdinal ordinal = 0; ordinal < enabled.n_trials; ++ordinal)
    {
        SpeechTrialSchedule with{};
        SpeechTrialSchedule without{};
        CHECK(prepare_trial(enabled, ordinal, with) == ContractStatus::ok);
        CHECK(prepare_trial(disabled, ordinal, without) == ContractStatus::ok);

        CHECK(with.cross_enabled);
        CHECK(with.cross_duration_ns > 0);
        CHECK(!without.cross_enabled);
        // Exactly zero, and therefore unambiguous: the enabled case cannot
        // produce it.
        CHECK(without.cross_duration_ns == 0);

        // Fixed draw coordinates rather than a running counter: turning the
        // cross off must not re-roll anything else in the session.
        CHECK(with.black_duration_ns == without.black_duration_ns);
        CHECK(with.content_duration_ns == without.content_duration_ns);
        CHECK(with.stimulus_id == without.stimulus_id);

        SpeechTimeline shown{};
        SpeechTimeline hidden{};
        CHECK(build_timeline(enabled, with, 1'000, shown) == ContractStatus::ok);
        CHECK(build_timeline(disabled, without, 1'000, hidden) == ContractStatus::ok);
        CHECK(shown.count == 3);
        CHECK(shown.phases[1].phase == SpeechPhase::cross);
        CHECK(shown.phases[1].cue == CueKind::fixation_cross);
        // Absent, not present-and-empty.
        CHECK(hidden.count == 2);
        CHECK(hidden.phases[0].phase == SpeechPhase::black);
        CHECK(hidden.phases[1].phase == SpeechPhase::content);
        for (std::uint8_t i = 0; i < hidden.count; ++i)
            CHECK(hidden.phases[i].phase != SpeechPhase::cross);
    }
}

void test_timeline_is_contiguous_and_half_open()
{
    SpeechCueConfig config = seeded_config(true, StimulusOrderPolicy::sequential, 4, 3);
    config.inter_trial_ns = 250'000'000;
    SpeechTrialSchedule schedule{};
    CHECK(prepare_trial(config, 0, schedule) == ContractStatus::ok);

    constexpr ExperimentTimeNs start = 7'000'000'000;
    SpeechTimeline timeline{};
    CHECK(build_timeline(config, schedule, start, timeline) == ContractStatus::ok);
    CHECK(timeline.count == 4);
    CHECK(timeline.phases[0].phase == SpeechPhase::black);
    CHECK(timeline.phases[0].cue == CueKind::black);
    CHECK(timeline.phases[0].stimulus_id == kUnsetStimulusId);
    CHECK(timeline.phases[2].phase == SpeechPhase::content);
    CHECK(timeline.phases[2].cue == CueKind::text_content);
    CHECK(timeline.phases[2].stimulus_id == schedule.stimulus_id);
    CHECK(timeline.phases[3].phase == SpeechPhase::inter_trial);
    CHECK(timeline.phases[3].cue == CueKind::none);

    CHECK(timeline.phases[0].interval.start_ns == start);
    for (std::uint8_t i = 1; i < timeline.count; ++i)
    {
        // Contiguous and half-open: the exact boundary belongs to the phase
        // that starts there, and to nothing else.
        CHECK(timeline.phases[i].interval.start_ns == timeline.phases[i - 1].interval.end_ns);
        CHECK(!timeline.phases[i - 1].interval.contains(timeline.phases[i].interval.start_ns));
        CHECK(timeline.phases[i].interval.contains(timeline.phases[i].interval.start_ns));
        CHECK(timeline.phases[i - 1].interval.elapsed_at(timeline.phases[i].interval.start_ns));
    }

    DurationNs length = 0;
    CHECK(trial_duration(schedule, length) == ContractStatus::ok);
    CHECK(timeline.trial.start_ns == start);
    CHECK(timeline.trial.duration_ns() == length);
    // The trial is over at the end of its content; the gap comes after it.
    CHECK(timeline.trial.end_ns == timeline.phases[2].interval.end_ns);
    CHECK(timeline.next_trial_start_ns == timeline.trial.end_ns + config.inter_trial_ns);

    SpeechCueConfig back_to_back = config;
    back_to_back.inter_trial_ns = 0;
    SpeechTimeline chained{};
    CHECK(build_timeline(back_to_back, schedule, start, chained) == ContractStatus::ok);
    CHECK(chained.count == 3);
    CHECK(chained.next_trial_start_ns == chained.trial.end_ns);

    // Overflow is reported rather than wrapped into a timeline that ends before
    // it starts.
    SpeechTimeline overflowed{};
    CHECK(build_timeline(config, schedule, ~ExperimentTimeNs{0} - 10, overflowed) ==
          ContractStatus::duration_overflow);
}

void test_determinism_and_seed_divergence()
{
    const SpeechCueConfig config = seeded_config(true, StimulusOrderPolicy::shuffled_blocks, 64, 8);
    SpeechCueConfig other = config;
    other.seed = config.seed ^ 1;

    bool diverged = false;
    for (TrialOrdinal ordinal = 0; ordinal < config.n_trials; ++ordinal)
    {
        SpeechTrialSchedule first{};
        SpeechTrialSchedule again{};
        SpeechTrialSchedule elsewhere{};
        CHECK(prepare_trial(config, ordinal, first) == ContractStatus::ok);
        CHECK(prepare_trial(config, ordinal, again) == ContractStatus::ok);
        CHECK(prepare_trial(other, ordinal, elsewhere) == ContractStatus::ok);
        CHECK(identical(first, again));
        if (!identical(first, elsewhere))
            diverged = true;
    }
    CHECK(diverged);

    // Addressed, not advanced: trial 40 is the same whether or not the 40
    // before it were ever prepared, and preparing them backwards changes
    // nothing.
    SpeechTrialSchedule forwards{};
    SpeechTrialSchedule alone{};
    CHECK(prepare_trial(config, 40, forwards) == ContractStatus::ok);
    for (TrialOrdinal ordinal = config.n_trials; ordinal-- > 0;)
    {
        SpeechTrialSchedule ignored{};
        CHECK(prepare_trial(config, ordinal, ignored) == ContractStatus::ok);
    }
    CHECK(prepare_trial(config, 40, alone) == ContractStatus::ok);
    CHECK(identical(forwards, alone));

    // Past the end of the session is a refusal, not a wrapped ordinal.
    SpeechTrialSchedule beyond{};
    CHECK(prepare_trial(config, config.n_trials, beyond) == ContractStatus::range_empty);
    StimulusId unused = kUnsetStimulusId;
    CHECK(select_stimulus(config, config.n_trials, unused) == ContractStatus::range_empty);
}

void test_stimulus_order_policies()
{
    constexpr std::uint16_t kStimuli = 6;
    const SpeechCueConfig sequential =
        seeded_config(false, StimulusOrderPolicy::sequential, 4 * kStimuli, kStimuli);
    for (TrialOrdinal ordinal = 0; ordinal < sequential.n_trials; ++ordinal)
    {
        StimulusId id = kUnsetStimulusId;
        CHECK(select_stimulus(sequential, ordinal, id) == ContractStatus::ok);
        CHECK(id == static_cast<StimulusId>(ordinal % kStimuli + 1));
    }

    const SpeechCueConfig replaced =
        seeded_config(false, StimulusOrderPolicy::random_with_replacement, 600, kStimuli);
    std::array<std::size_t, kStimuli> seen{};
    bool immediate_repeat = false;
    StimulusId previous = kUnsetStimulusId;
    for (TrialOrdinal ordinal = 0; ordinal < replaced.n_trials; ++ordinal)
    {
        StimulusId id = kUnsetStimulusId;
        CHECK(select_stimulus(replaced, ordinal, id) == ContractStatus::ok);
        CHECK(id >= 1 && id <= kStimuli);
        ++seen[id - 1];
        if (id == previous)
            immediate_repeat = true;
        previous = id;
    }
    for (const std::size_t count : seen)
        CHECK(count > 0);
    // With replacement really does mean with replacement.
    CHECK(immediate_repeat);

    const SpeechCueConfig shuffled =
        seeded_config(false, StimulusOrderPolicy::shuffled_blocks, 5 * kStimuli, kStimuli);
    bool reordered = false;
    for (TrialOrdinal block = 0; block < 5; ++block)
    {
        std::array<std::size_t, kStimuli> within{};
        for (std::uint16_t i = 0; i < kStimuli; ++i)
        {
            StimulusId id = kUnsetStimulusId;
            CHECK(select_stimulus(shuffled, block * kStimuli + i, id) == ContractStatus::ok);
            CHECK(id >= 1 && id <= kStimuli);
            ++within[id - 1];
            if (id != static_cast<StimulusId>(i + 1))
                reordered = true;
        }
        // Balanced at every block boundary: each stimulus exactly once.
        for (const std::size_t count : within)
            CHECK(count == 1);
    }
    CHECK(reordered);
}

void test_explicit_schedule_is_own_authority()
{
    const SpeechCueConfig seeded = seeded_config(true, StimulusOrderPolicy::shuffled_blocks, 32, 8);
    const SpeechCueConfig frozen = freeze(seeded);
    CHECK(validate(frozen) == ContractStatus::ok);
    CHECK(replay_authority(seeded) == ReplayAuthority::regenerate);
    // Nothing to regenerate, whatever the sampler version says.
    CHECK(replay_authority(frozen) == ReplayAuthority::recorded_schedule);

    for (TrialOrdinal ordinal = 0; ordinal < seeded.n_trials; ++ordinal)
    {
        SpeechTrialSchedule regenerated{};
        SpeechTrialSchedule recorded{};
        CHECK(prepare_trial(seeded, ordinal, regenerated) == ContractStatus::ok);
        CHECK(prepare_trial(frozen, ordinal, recorded) == ContractStatus::ok);
        CHECK(identical(regenerated, recorded));
        StimulusId id = kUnsetStimulusId;
        CHECK(select_stimulus(frozen, ordinal, id) == ContractStatus::ok);
        CHECK(id == regenerated.stimulus_id);
    }

    // A frozen schedule replays under a sampler version this build cannot
    // regenerate, because it regenerates nothing. The whole session moves to
    // that version, entries included: the session identity takes its sampler
    // from the configuration, so a configuration saying 4242 over entries
    // saying 1 is not a recorded future session, it is two claims about one
    // session, and that is refused rather than replayed.
    SpeechCueConfig future = frozen;
    future.sampler_version = 4242;
    CHECK(validate(future) == ContractStatus::outcome_invalid);
    for (std::size_t i = 0; i < future.n_explicit; ++i)
        future.explicit_schedule[i].sampler_version = 4242;
    CHECK(validate(future) == ContractStatus::ok);
    CHECK(replay_authority(future) == ReplayAuthority::recorded_schedule);
    SpeechTrialSchedule replayed{};
    CHECK(prepare_trial(future, 0, replayed) == ContractStatus::ok);
    CHECK(replayed.sampler_version == 4242);
    CHECK(validate_against(replayed, future) == ContractStatus::ok);

    SpeechCueConfig unregenerable = seeded;
    unregenerable.sampler_version = 4242;
    SpeechTrialSchedule refused{};
    CHECK(prepare_trial(unregenerable, 0, refused) == ContractStatus::version_unsupported);
    // Stepping through a set in order needs no sampler, so it is not refused.
    SpeechCueConfig ordered = unregenerable;
    ordered.stimulus_order = StimulusOrderPolicy::sequential;
    StimulusId id = kUnsetStimulusId;
    CHECK(select_stimulus(ordered, 0, id) == ContractStatus::ok);
}

void test_supplied_schedules_are_checked_not_trusted()
{
    const SpeechCueConfig seeded = seeded_config(true, StimulusOrderPolicy::sequential, 4, 4);
    const SpeechCueConfig frozen = freeze(seeded);

    // A cross duration of zero under an enabled cross.
    SpeechCueConfig blanked = frozen;
    blanked.explicit_schedule[1].cross_duration_ns = 0;
    CHECK(validate(blanked) == ContractStatus::outcome_invalid);

    // A cross duration under a disabled cross.
    SpeechCueConfig off = freeze(seeded_config(false, StimulusOrderPolicy::sequential, 4, 4));
    CHECK(validate(off) == ContractStatus::ok);
    for (std::size_t i = 0; i < off.n_explicit; ++i)
        CHECK(off.explicit_schedule[i].cross_duration_ns == 0);
    SpeechCueConfig sneaked = off;
    sneaked.explicit_schedule[2].cross_duration_ns = 1;
    CHECK(validate(sneaked) == ContractStatus::outcome_invalid);

    // A duration that reaches its bound is outside the open range.
    SpeechCueConfig at_bound = frozen;
    at_bound.explicit_schedule[0].black_duration_ns = frozen.black_bound_ns;
    CHECK(validate(at_bound) == ContractStatus::parameter_out_of_range);
    SpeechCueConfig empty_black = frozen;
    empty_black.explicit_schedule[0].black_duration_ns = 0;
    CHECK(validate(empty_black) == ContractStatus::parameter_out_of_range);
    SpeechCueConfig long_content = frozen;
    long_content.explicit_schedule[3].content_duration_ns = frozen.content_bound_ns + 1;
    CHECK(validate(long_content) == ContractStatus::parameter_out_of_range);

    // Ordinals name their own position, so an entry cannot be silently moved.
    SpeechCueConfig shifted = frozen;
    shifted.explicit_schedule[2].ordinal = 9;
    CHECK(validate(shifted) == ContractStatus::outcome_invalid);
    SpeechCueConfig miscounted = frozen;
    miscounted.n_trials = 3;
    CHECK(validate(miscounted) == ContractStatus::outcome_invalid);
    SpeechCueConfig littered = frozen;
    littered.explicit_schedule[frozen.n_explicit].black_duration_ns = 1;
    CHECK(validate(littered) == ContractStatus::outcome_invalid);

    // Fields that decide a realized schedule are absent on the side that does
    // not decide it.
    SpeechCueConfig double_answer = frozen;
    double_answer.stimulus_order = StimulusOrderPolicy::sequential;
    CHECK(validate(double_answer) == ContractStatus::outcome_invalid);
    SpeechCueConfig double_set = frozen;
    double_set.n_stimuli = 1;
    double_set.stimuli[0] = 1;
    CHECK(validate(double_set) == ContractStatus::outcome_invalid);
    SpeechCueConfig strays = seeded;
    strays.n_explicit = 1;
    CHECK(validate(strays) == ContractStatus::outcome_invalid);
    SpeechCueConfig stray_entry = seeded;
    stray_entry.explicit_schedule[0].black_duration_ns = 1;
    CHECK(validate(stray_entry) == ContractStatus::outcome_invalid);
    // The seed is provenance and stays: a frozen schedule may record where it
    // came from without that deciding anything.
    CHECK(frozen.seed == seeded.seed);

    // validate_against() answers about one entry and its configuration.
    SpeechTrialSchedule good{};
    CHECK(prepare_trial(seeded, 2, good) == ContractStatus::ok);
    CHECK(validate_against(good, seeded) == ContractStatus::ok);
    SpeechTrialSchedule wrong_cross = good;
    wrong_cross.cross_enabled = false;
    wrong_cross.cross_duration_ns = 0;
    CHECK(validate_against(wrong_cross, seeded) == ContractStatus::outcome_invalid);
    SpeechTrialSchedule beyond = good;
    beyond.ordinal = seeded.n_trials;
    CHECK(validate_against(beyond, seeded) == ContractStatus::range_empty);
    SpeechTrialSchedule anonymous = good;
    anonymous.stimulus_id = kUnsetStimulusId;
    CHECK(validate(anonymous) == ContractStatus::identity_missing);
    SpeechTrialSchedule unversioned = good;
    unversioned.sampler_version = 0;
    CHECK(validate(unversioned) == ContractStatus::identity_missing);
}

void test_schedule_must_belong_to_configuration()
{
    const SpeechCueConfig seeded = seeded_config(true, StimulusOrderPolicy::sequential, 4, 2);
    SpeechTrialSchedule good{};
    CHECK(prepare_trial(seeded, 0, good) == ContractStatus::ok);
    CHECK(validate_against(good, seeded) == ContractStatus::ok);

    // Well-formed, inside every bound, and presenting a stimulus this session
    // does not have. Fitting the bounds is not evidence of belonging.
    SpeechTrialSchedule foreign = good;
    foreign.stimulus_id = 999;
    CHECK(validate(foreign) == ContractStatus::ok);
    CHECK(validate_against(foreign, seeded) == ContractStatus::target_set_invalid);
    SpeechTimeline timeline{};
    CHECK(build_timeline(seeded, foreign, 0, timeline) == ContractStatus::target_set_invalid);
    CHECK(timeline.count == 0);
    // Any member of the set is admitted: which one a given ordinal draws is a
    // question for replay verification, not for the gate every timeline passes.
    SpeechTrialSchedule other = good;
    other.stimulus_id = seeded.stimuli[1];
    CHECK(validate_against(other, seeded) == ContractStatus::ok);

    // One session, one sampler. The identity's version comes from the
    // configuration, so an entry naming another one is a contradiction rather
    // than extra provenance.
    SpeechTrialSchedule reversioned = good;
    reversioned.sampler_version = 4242;
    CHECK(validate(reversioned) == ContractStatus::ok);
    CHECK(validate_against(reversioned, seeded) == ContractStatus::outcome_invalid);

    // An explicit configuration holds the authoritative realized schedule for
    // every ordinal, so a substitute that merely fits the bounds is refused --
    // it is a second answer to a question the configuration already answered.
    const SpeechCueConfig frozen = freeze(seeded);
    SpeechTrialSchedule recorded{};
    CHECK(prepare_trial(frozen, 1, recorded) == ContractStatus::ok);
    CHECK(validate_against(recorded, frozen) == ContractStatus::ok);
    SpeechTrialSchedule shortened = recorded;
    shortened.black_duration_ns = 1;
    CHECK(validate(shortened) == ContractStatus::ok);
    CHECK(inside_bounds(shortened, frozen));
    CHECK(validate_against(shortened, frozen) == ContractStatus::outcome_invalid);
    SpeechTimeline substituted{};
    CHECK(build_timeline(frozen, shortened, 0, substituted) == ContractStatus::outcome_invalid);
    CHECK(substituted.count == 0);
    SpeechTrialSchedule restimulated = recorded;
    restimulated.stimulus_id = frozen.explicit_schedule[0].stimulus_id == recorded.stimulus_id
                                   ? recorded.stimulus_id + 1
                                   : frozen.explicit_schedule[0].stimulus_id;
    CHECK(validate_against(restimulated, frozen) == ContractStatus::outcome_invalid);
    // And the entry itself still passes, so the refusal is of the substitute.
    CHECK(validate_against(recorded, frozen) == ContractStatus::ok);
}

void test_invalid_timing_domains_and_configuration()
{
    const SpeechCueConfig base = seeded_config(true, StimulusOrderPolicy::sequential, 8, 4);
    CHECK(validate(base) == ContractStatus::ok);

    // (0, 0) and (0, 1) hold no integer, so they are empty domains rather than
    // bounds to be widened.
    for (const DurationNs bound : {DurationNs{0}, DurationNs{1}})
    {
        SpeechCueConfig black = base;
        black.black_bound_ns = bound;
        CHECK(validate(black) == ContractStatus::range_empty);
        SpeechCueConfig content = base;
        content.content_bound_ns = bound;
        CHECK(validate(content) == ContractStatus::range_empty);
        SpeechCueConfig cross = base;
        cross.cross_bound_ns = bound;
        CHECK(validate(cross) == ContractStatus::range_empty);
    }
    // Nothing is clamped: the bound that was rejected is still the bound.
    SpeechCueConfig rejected = base;
    rejected.black_bound_ns = 1;
    CHECK(validate(rejected) == ContractStatus::range_empty);
    CHECK(rejected.black_bound_ns == 1);

    // A bound nothing samples is not left lying about.
    SpeechCueConfig inert = base;
    inert.cross_enabled = false;
    CHECK(validate(inert) == ContractStatus::outcome_invalid);
    inert.cross_bound_ns = 0;
    CHECK(validate(inert) == ContractStatus::ok);

    SpeechCueConfig no_trials = base;
    no_trials.n_trials = 0;
    CHECK(validate(no_trials) == ContractStatus::range_empty);
    SpeechCueConfig no_kind = base;
    no_kind.schedule = SpeechScheduleKind::unspecified;
    CHECK(validate(no_kind) == ContractStatus::identity_missing);
    SpeechCueConfig no_order = base;
    no_order.stimulus_order = StimulusOrderPolicy::unspecified;
    CHECK(validate(no_order) == ContractStatus::identity_missing);
    SpeechCueConfig no_version = base;
    no_version.sampler_version = 0;
    CHECK(validate(no_version) == ContractStatus::identity_missing);
    SpeechCueConfig undeclared = base;
    undeclared.schedule = static_cast<SpeechScheduleKind>(9);
    CHECK(validate(undeclared) == ContractStatus::enum_undeclared);
    SpeechCueConfig undeclared_order = base;
    undeclared_order.stimulus_order = static_cast<StimulusOrderPolicy>(9);
    CHECK(validate(undeclared_order) == ContractStatus::enum_undeclared);

    SpeechCueConfig no_stimuli = base;
    no_stimuli.n_stimuli = 0;
    CHECK(validate(no_stimuli) == ContractStatus::target_set_invalid);
    SpeechCueConfig duplicated = base;
    duplicated.stimuli[1] = duplicated.stimuli[0];
    CHECK(validate(duplicated) == ContractStatus::target_set_invalid);
    SpeechCueConfig unset_member = base;
    unset_member.stimuli[2] = kUnsetStimulusId;
    CHECK(validate(unset_member) == ContractStatus::target_set_invalid);
    SpeechCueConfig padded = base;
    padded.stimuli[base.n_stimuli] = 99;
    CHECK(validate(padded) == ContractStatus::target_set_invalid);
    SpeechCueConfig oversized = base;
    oversized.n_stimuli = static_cast<std::uint16_t>(kMaxSpeechStimuli + 1);
    CHECK(validate(oversized) == ContractStatus::target_set_invalid);
}

void test_catalog_ownership_and_resolution()
{
    const SpeechCatalog catalog = make_catalog(12);
    CHECK(validate(catalog) == ContractStatus::ok);

    SpeechStimulus resolved{};
    CHECK(find_stimulus(catalog, 5, resolved) == ContractStatus::ok);
    CHECK(resolved.id == 5);
    CHECK(resolved.content == SpeechContentKind::text);
    CHECK(resolved.text_length == 5);
    CHECK(find_stimulus(catalog, 13, resolved) == ContractStatus::identity_missing);
    CHECK(find_stimulus(catalog, kUnsetStimulusId, resolved) == ContractStatus::identity_missing);

    SpeechCatalog empty{};
    CHECK(validate(empty) == ContractStatus::target_set_invalid);
    SpeechCatalog duplicated = catalog;
    duplicated.entries[3].id = duplicated.entries[2].id;
    CHECK(validate(duplicated) == ContractStatus::target_set_invalid);
    SpeechCatalog littered = catalog;
    littered.entries[catalog.count].id = 99;
    CHECK(validate(littered) == ContractStatus::target_set_invalid);
    SpeechCatalog silent = catalog;
    silent.entries[1].text_length = 0;
    CHECK(validate(silent) == ContractStatus::identity_missing);
    SpeechCatalog untyped = catalog;
    untyped.entries[1].content = SpeechContentKind::unspecified;
    CHECK(validate(untyped) == ContractStatus::identity_missing);
    SpeechCatalog undeclared = catalog;
    undeclared.entries[1].content = static_cast<SpeechContentKind>(7);
    CHECK(validate(undeclared) == ContractStatus::enum_undeclared);
    SpeechCatalog dirty = catalog;
    dirty.entries[1].text[kMaxSpeechTextBytes - 1] = 'x';
    CHECK(validate(dirty) == ContractStatus::outcome_invalid);

    // The catalog text is canonical UTF-8. A multi-script prompt -- "中文" is
    // E4 B8 AD E6 96 87 -- is one entry like any other, and validates as ok.
    SpeechCatalog chinese = catalog;
    {
        SpeechStimulus& entry = chinese.entries[1];
        entry.text = {};
        const unsigned char bytes[] = {0xE4, 0xB8, 0xAD, 0xE6, 0x96, 0x87};
        for (std::size_t i = 0; i < sizeof(bytes); ++i)
            entry.text[i] = static_cast<char>(bytes[i]);
        entry.text_length = static_cast<std::uint8_t>(sizeof(bytes));
    }
    CHECK(validate(chinese) == ContractStatus::ok);
    // A well-formed catalog still resolves and fingerprints normally.
    SpeechStimulus chinese_resolved{};
    CHECK(find_stimulus(chinese, chinese.entries[1].id, chinese_resolved) == ContractStatus::ok);
    CHECK(chinese_resolved.text_length == 6);

    // The boundary code points adjacent to the rejected ranges are themselves
    // valid, so a validator that over-applied the first-continuation window to
    // later bytes would fail here. U+D3FF is ED 9F BF (last before surrogates)
    // and U+10FFFF is F4 8F BF BF (the maximum).
    const auto expect_accepted = [&](const unsigned char* bytes, std::size_t length)
    {
        SpeechStimulus entry{};
        entry.id = 1;
        entry.content = SpeechContentKind::text;
        for (std::size_t i = 0; i < length; ++i)
            entry.text[i] = static_cast<char>(bytes[i]);
        entry.text_length = static_cast<std::uint8_t>(length);
        CHECK(validate(entry) == ContractStatus::ok);
    };
    {
        const unsigned char before_surrogates[] = {0xED, 0x9F, 0xBF};
        expect_accepted(before_surrogates, sizeof(before_surrogates));
    }
    {
        const unsigned char maximum[] = {0xF4, 0x8F, 0xBF, 0xBF};
        expect_accepted(maximum, sizeof(maximum));
    }

    // Malformed UTF-8 is rejected with the same status as a dirty padding
    // field, because both are byte-level invariants of the text the entry
    // must hold. Each case is one well-formedness rule.
    const auto expect_rejected = [&](const unsigned char* bytes, std::size_t length)
    {
        SpeechCatalog malformed = catalog;
        SpeechStimulus& entry = malformed.entries[1];
        entry.text = {};
        for (std::size_t i = 0; i < length; ++i)
            entry.text[i] = static_cast<char>(bytes[i]);
        entry.text_length = static_cast<std::uint8_t>(length);
        CHECK(validate(malformed) == ContractStatus::outcome_invalid);
    };
    {
        const unsigned char lone_continuation[] = {0x80};
        expect_rejected(lone_continuation, sizeof(lone_continuation));
    }
    {
        const unsigned char truncated[] = {0xC2};
        expect_rejected(truncated, sizeof(truncated));
    }
    {
        const unsigned char overlong[] = {0xC0, 0x80};
        expect_rejected(overlong, sizeof(overlong));
    }
    {
        const unsigned char surrogate[] = {0xED, 0xA0, 0x80};
        expect_rejected(surrogate, sizeof(surrogate));
    }
    {
        const unsigned char beyond_max[] = {0xF4, 0x90, 0x80, 0x80};
        expect_rejected(beyond_max, sizeof(beyond_max));
    }

    const SpeechCueConfig config = seeded_config(true, StimulusOrderPolicy::sequential, 8, 4);
    CHECK(validate_against(config, catalog) == ContractStatus::ok);
    SpeechCueConfig unknown = config;
    unknown.stimuli[2] = 900;
    CHECK(validate_against(unknown, catalog) == ContractStatus::identity_missing);
    const SpeechCueConfig frozen = freeze(config);
    CHECK(validate_against(frozen, catalog) == ContractStatus::ok);
    SpeechCueConfig frozen_unknown = frozen;
    frozen_unknown.explicit_schedule[1].stimulus_id = 900;
    CHECK(validate_against(frozen_unknown, catalog) == ContractStatus::identity_missing);
}

void test_identity_and_fingerprints()
{
    const SpeechCatalog catalog = make_catalog(8);
    const SpeechCueConfig config = seeded_config(true, StimulusOrderPolicy::sequential, 16, 4);
    const ScheduleIdentity identity = schedule_identity(config, catalog);
    CHECK(identity.seed == config.seed);
    CHECK(identity.sampler_version == config.sampler_version);
    CHECK(identity.configuration_fingerprint == configuration_fingerprint(config));
    CHECK(identity.catalog_fingerprint == catalog_fingerprint(catalog));
    CHECK(validate(identity) == ContractStatus::ok);

    // Stable: the same value fingerprints the same way, every time.
    CHECK(configuration_fingerprint(config) == configuration_fingerprint(config));
    CHECK(catalog_fingerprint(catalog) == catalog_fingerprint(catalog));
    CHECK(schedule_fingerprint(identity) ==
          schedule_fingerprint(schedule_identity(config, catalog)));

    const std::uint64_t baseline = schedule_fingerprint(identity);
    const auto moved = [&](const SpeechCueConfig& changed)
    { return schedule_fingerprint(schedule_identity(changed, catalog)) != baseline; };

    SpeechCueConfig seeded_elsewhere = config;
    seeded_elsewhere.seed ^= 1;
    CHECK(moved(seeded_elsewhere));
    SpeechCueConfig other_version = config;
    other_version.sampler_version = 2;
    CHECK(moved(other_version));
    // cross_enabled is in the fingerprint, which is the point: two sessions
    // differing only in whether they had a cross are not the same session.
    SpeechCueConfig no_cross = config;
    no_cross.cross_enabled = false;
    no_cross.cross_bound_ns = 0;
    CHECK(moved(no_cross));
    SpeechCueConfig longer = config;
    longer.content_bound_ns += 1;
    CHECK(moved(longer));
    SpeechCueConfig gapped = config;
    gapped.inter_trial_ns = 1;
    CHECK(moved(gapped));
    SpeechCueConfig reordered = config;
    reordered.stimulus_order = StimulusOrderPolicy::shuffled_blocks;
    CHECK(moved(reordered));
    SpeechCueConfig relabelled = config;
    relabelled.stimuli[0] = 9;
    CHECK(moved(relabelled));
    SpeechCueConfig shorter = config;
    shorter.n_trials = 15;
    CHECK(moved(shorter));
    CHECK(schedule_fingerprint(schedule_identity(config, make_catalog(9))) != baseline);
    // Content participates, not just identifiers.
    SpeechCatalog reworded = catalog;
    set_text(reworded.entries[0], "different");
    CHECK(catalog_fingerprint(reworded) != catalog_fingerprint(catalog));
    SpeechCatalog reclassed = catalog;
    reclassed.entries[0].label += 1;
    CHECK(catalog_fingerprint(reclassed) != catalog_fingerprint(catalog));
    SpeechCatalog retagged = catalog;
    retagged.entries[0].metadata += 1;
    CHECK(catalog_fingerprint(retagged) != catalog_fingerprint(catalog));

    // Unused capacity is absorbed along with the rest, so it has to be
    // canonical: a value hidden in a slot nothing reads must not be able to
    // separate two configurations that schedule identically. Every unused field
    // is default, sampler version included -- which is why that field defaults
    // to unset rather than to this build's current sampler, since a padding
    // default that tracked that constant would move the fingerprint of an
    // unchanged configuration the first time it is bumped.
    SpeechCueConfig littered = config;
    littered.explicit_schedule[7].sampler_version = 4242;
    CHECK(validate(littered) == ContractStatus::outcome_invalid);
    CHECK(moved(littered));
    SpeechCueConfig defaulted{};
    CHECK(defaulted.explicit_schedule[0].sampler_version == 0);
    const SpeechCueConfig frozen = freeze(config);
    CHECK(validate(frozen) == ContractStatus::ok);
    SpeechCueConfig beyond_count = frozen;
    beyond_count.explicit_schedule[frozen.n_explicit].sampler_version = kCurrentSamplerVersion;
    CHECK(validate(beyond_count) == ContractStatus::outcome_invalid);
    CHECK(configuration_fingerprint(beyond_count) != configuration_fingerprint(frozen));
}

void test_draw_provenance()
{
    const SpeechCueConfig enabled = seeded_config(true, StimulusOrderPolicy::sequential, 4, 4);
    SpeechCueConfig disabled = enabled;
    disabled.cross_enabled = false;
    disabled.cross_bound_ns = 0;
    const TrialIdentity trial{2, 7, 0, kUnsetTargetId, 3};

    SpeechTrialSchedule with{};
    CHECK(prepare_trial(enabled, 2, with) == ContractStatus::ok);
    SpeechTrialDraws drawn{};
    CHECK(make_schedule_draws(with, trial, 5'000, drawn) == ContractStatus::ok);
    CHECK(drawn.count == 3);
    CHECK(drawn.draws[0].draw == kBlackDraw);
    CHECK(drawn.draws[0].value == with.black_duration_ns);
    CHECK(drawn.draws[1].draw == kCrossDraw);
    CHECK(drawn.draws[1].value == with.cross_duration_ns);
    CHECK(drawn.draws[2].draw == kContentDraw);
    CHECK(drawn.draws[2].value == with.content_duration_ns);
    for (std::uint8_t i = 0; i < drawn.count; ++i)
    {
        CHECK(drawn.draws[i].stream == kTrialTimingStream);
        CHECK(drawn.draws[i].sampler_version == with.sampler_version);
        CHECK(same_trial(drawn.draws[i].trial, trial));
        CHECK(drawn.draws[i].time_ns == 5'000);
        CHECK(validate(drawn.draws[i]) == ContractStatus::ok);
        // Provenance a reader can check against a regeneration, without
        // knowing anything about this paradigm.
        std::uint64_t regenerated = 0;
        CHECK(sample_exclusive(
                  DrawKey{enabled.seed, drawn.draws[i].stream, with.ordinal, drawn.draws[i].draw},
                  0,
                  drawn.draws[i].draw == kBlackDraw   ? enabled.black_bound_ns
                  : drawn.draws[i].draw == kCrossDraw ? enabled.cross_bound_ns
                                                      : enabled.content_bound_ns,
                  regenerated) == ContractStatus::ok);
        CHECK(regenerated == drawn.draws[i].value);
    }

    SpeechTrialSchedule without{};
    CHECK(prepare_trial(disabled, 2, without) == ContractStatus::ok);
    SpeechTrialDraws fewer{};
    CHECK(make_schedule_draws(without, trial, 5'000, fewer) == ContractStatus::ok);
    // A disabled cross emits no draw at all, so a reader counting records sees
    // the phase was never sampled.
    CHECK(fewer.count == 2);
    CHECK(fewer.draws[0].draw == kBlackDraw);
    CHECK(fewer.draws[1].draw == kContentDraw);

    SpeechTrialSchedule broken = with;
    broken.cross_duration_ns = 0;
    SpeechTrialDraws refused{};
    CHECK(make_schedule_draws(broken, trial, 5'000, refused) == ContractStatus::outcome_invalid);
    CHECK(refused.count == 0);

    // The identity is the one field these records exist to carry. Values drawn
    // for one trial filed under another are not slightly wrong provenance, they
    // are provenance that says the wrong thing, so the identity is checked
    // rather than copied.
    TrialIdentity elsewhere = trial;
    elsewhere.ordinal = 9;
    SpeechTrialDraws misfiled{};
    CHECK(make_schedule_draws(with, elsewhere, 5'000, misfiled) == ContractStatus::outcome_invalid);
    CHECK(misfiled.count == 0);
    TrialIdentity relabelled = trial;
    relabelled.stimulus_id = 999;
    CHECK(make_schedule_draws(with, relabelled, 5'000, misfiled) ==
          ContractStatus::outcome_invalid);
    CHECK(misfiled.count == 0);
    TrialIdentity anonymous = trial;
    anonymous.stimulus_id = kUnsetStimulusId;
    CHECK(make_schedule_draws(with, anonymous, 5'000, misfiled) ==
          ContractStatus::identity_missing);
    CHECK(misfiled.count == 0);
    // Everything a draw record does not speak for stays free: the key, the
    // block, and a target this paradigm has none of.
    TrialIdentity blocked = trial;
    blocked.block = 4;
    blocked.key = 11;
    SpeechTrialDraws accepted{};
    CHECK(make_schedule_draws(with, blocked, 5'000, accepted) == ContractStatus::ok);
    CHECK(accepted.count == 3);
}

void test_full_capacity_and_long_sessions()
{
    // The whole catalog and the whole ordered set, at capacity.
    const SpeechCatalog catalog = make_catalog(static_cast<std::uint16_t>(kMaxSpeechStimuli));
    CHECK(validate(catalog) == ContractStatus::ok);
    const SpeechCueConfig dense =
        seeded_config(true, StimulusOrderPolicy::shuffled_blocks, 2 * kMaxSpeechStimuli,
                      static_cast<std::uint16_t>(kMaxSpeechStimuli));
    CHECK(validate(dense) == ContractStatus::ok);
    CHECK(validate_against(dense, catalog) == ContractStatus::ok);
    std::array<std::size_t, kMaxSpeechStimuli> seen{};
    for (TrialOrdinal ordinal = 0; ordinal < kMaxSpeechStimuli; ++ordinal)
    {
        SpeechTrialSchedule schedule{};
        CHECK(prepare_trial(dense, ordinal, schedule) == ContractStatus::ok);
        ++seen[schedule.stimulus_id - 1];
    }
    for (const std::size_t count : seen)
        CHECK(count == 1);

    const SpeechCueConfig full_explicit =
        freeze(seeded_config(true, StimulusOrderPolicy::sequential, kMaxSpeechExplicitTrials, 8));
    CHECK(validate(full_explicit) == ContractStatus::ok);
    SpeechCueConfig oversized = full_explicit;
    oversized.n_explicit = static_cast<std::uint16_t>(kMaxSpeechExplicitTrials + 1);
    CHECK(validate(oversized) == ContractStatus::parameter_out_of_range);

    // A long session costs nothing to schedule: the sampler is addressed, so
    // there is no prepared table anywhere and no per-trial state to keep.
    constexpr TrialOrdinal kLongSession = 200'000;
    const SpeechCueConfig marathon =
        seeded_config(true, StimulusOrderPolicy::random_with_replacement, kLongSession, 16);
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);
    ExperimentTimeNs cursor = 0;
    for (TrialOrdinal ordinal = 0; ordinal < kLongSession; ++ordinal)
    {
        SpeechTrialSchedule schedule{};
        if (prepare_trial(marathon, ordinal, schedule) != ContractStatus::ok)
        {
            ++failures;
            break;
        }
        if (schedule.black_duration_ns == 0 ||
            schedule.black_duration_ns >= marathon.black_bound_ns ||
            schedule.cross_duration_ns == 0 ||
            schedule.cross_duration_ns >= marathon.cross_bound_ns ||
            schedule.content_duration_ns == 0 ||
            schedule.content_duration_ns >= marathon.content_bound_ns)
        {
            ++failures;
            break;
        }
        SpeechTimeline timeline{};
        if (build_timeline(marathon, schedule, cursor, timeline) != ContractStatus::ok)
        {
            ++failures;
            break;
        }
        cursor = timeline.next_trial_start_ns;
    }
    CHECK(allocations.load(std::memory_order_relaxed) == baseline);
    CHECK(cursor > 0);
}

void test_native_path_allocates_nothing()
{
    const SpeechCatalog catalog = make_catalog(8);
    const SpeechCueConfig config = seeded_config(true, StimulusOrderPolicy::shuffled_blocks, 32, 8);
    const std::size_t baseline = allocations.load(std::memory_order_relaxed);
    SpeechTrialSchedule schedule{};
    SpeechTimeline timeline{};
    SpeechTrialDraws draws{};
    SpeechStimulus stimulus{};
    StimulusId id = kUnsetStimulusId;
    DurationNs length = 0;
    CHECK(validate(config) == ContractStatus::ok);
    CHECK(validate(catalog) == ContractStatus::ok);
    CHECK(validate_against(config, catalog) == ContractStatus::ok);
    CHECK(select_stimulus(config, 3, id) == ContractStatus::ok);
    CHECK(prepare_trial(config, 3, schedule) == ContractStatus::ok);
    CHECK(validate_against(schedule, config) == ContractStatus::ok);
    CHECK(trial_duration(schedule, length) == ContractStatus::ok);
    CHECK(build_timeline(config, schedule, 1'000, timeline) == ContractStatus::ok);
    CHECK(make_schedule_draws(schedule, TrialIdentity{3, 4, 0, kUnsetTargetId, id}, 1'000, draws) ==
          ContractStatus::ok);
    CHECK(find_stimulus(catalog, id, stimulus) == ContractStatus::ok);
    CHECK(configuration_fingerprint(config) != 0);
    CHECK(catalog_fingerprint(catalog) != 0);
    CHECK(allocations.load(std::memory_order_relaxed) == baseline);
}

} // namespace

namespace machine_tests
{
namespace
{

using namespace neurale::experiments;
using namespace neurale::experiments::speech;

static_assert(std::is_trivially_copyable_v<SpeechTrial>);
static_assert(std::is_trivially_copyable_v<SpeechSnapshot>);
static_assert(std::is_copy_assignable_v<SpeechMachine>);

int failures = 0;

constexpr DurationNs kBlackBound = 500'000'000;
constexpr DurationNs kCrossBound = 300'000'000;
constexpr DurationNs kContentBound = 2'000'000'000;
constexpr ScheduleSeed kSeed = 0xC0FFEE1234567890ULL;
constexpr ParadigmId kParadigm = 909;

SpeechCueConfig config_of(bool cross_enabled, TrialOrdinal n_trials)
{
    SpeechCueConfig config{};
    config.black_bound_ns = kBlackBound;
    config.cross_bound_ns = cross_enabled ? kCrossBound : 0;
    config.content_bound_ns = kContentBound;
    config.inter_trial_ns = 0;
    config.seed = kSeed;
    config.n_trials = n_trials;
    config.schedule = SpeechScheduleKind::seeded;
    config.stimulus_order = StimulusOrderPolicy::sequential;
    config.sampler_version = kCurrentSamplerVersion;
    config.cross_enabled = cross_enabled;
    config.n_stimuli = 4;
    for (std::uint16_t i = 0; i < 4; ++i)
        config.stimuli[i] = static_cast<StimulusId>(i + 1);
    return config;
}

SpeechTrialSchedule schedule_of(const SpeechCueConfig& config, TrialOrdinal ordinal)
{
    SpeechTrialSchedule schedule{};
    if (prepare_trial(config, ordinal, schedule) != ContractStatus::ok)
        ++failures;
    return schedule;
}

// Count events of one kind, or of one marker code, in a step's result.
std::size_t count_kind(const SpeechStepResult& result, ExperimentEventKind kind) noexcept
{
    std::size_t seen = 0;
    for (std::uint8_t i = 0; i < result.n_events; ++i)
        if (result.events[i].kind == kind)
            ++seen;
    return seen;
}

std::size_t count_marker(const SpeechStepResult& result, SpeechMarker marker) noexcept
{
    std::size_t seen = 0;
    for (std::uint8_t i = 0; i < result.n_events; ++i)
        if (result.events[i].kind == ExperimentEventKind::paradigm_marker &&
            result.events[i].code == static_cast<std::uint32_t>(marker))
            ++seen;
    return seen;
}

bool has_transition_to(const SpeechStepResult& result, SpeechState to) noexcept
{
    for (std::uint8_t i = 0; i < result.n_transitions; ++i)
        if (result.transitions[i].to_state == static_cast<StateId>(to))
            return true;
    return false;
}

// ---------------------------------------------------------------------------

void test_ordinary_black_cross_content_sequence()
{
    const SpeechCueConfig config = config_of(true, 3);
    const SpeechTrialSchedule first = schedule_of(config, 0);
    SpeechMachine machine{};
    SpeechStepResult result{};

    CHECK(machine.start(kParadigm, config, first, 1'000, result) == ContractStatus::ok);
    CHECK(machine.state() == SpeechState::black);
    CHECK(result.settled);
    CHECK(count_kind(result, ExperimentEventKind::session_start) == 1);
    CHECK(count_kind(result, ExperimentEventKind::trial_start) == 1);
    CHECK(count_marker(result, SpeechMarker::black_onset) == 1);
    CHECK(result.n_transitions == 1);
    CHECK(result.n_requests == 1);
    CHECK(result.requests[0].cue == CueKind::black);
    CHECK(result.requests[0].stimulus_id == kUnsetStimulusId);
    CHECK(result.requests[0].onset_ns == 1'000);
    CHECK(result.requests[0].duration_ns == first.black_duration_ns);
    CHECK(validate(result.requests[0]) == ContractStatus::ok);

    const ExperimentTimeNs black_end = 1'000 + first.black_duration_ns;
    const ExperimentTimeNs cross_end = black_end + first.cross_duration_ns;
    const ExperimentTimeNs content_end = cross_end + first.content_duration_ns;

    // Mid-BLACK: nothing happens, and nothing is emitted.
    CHECK(machine.step(black_end - 1, result) == ContractStatus::ok);
    CHECK(machine.state() == SpeechState::black);
    CHECK(result.n_transitions == 0 && result.n_events == 0 && result.n_requests == 0);

    // The exact BLACK end instant belongs to CROSS.
    CHECK(machine.step(black_end, result) == ContractStatus::ok);
    CHECK(machine.state() == SpeechState::fixation_cross);
    CHECK(count_marker(result, SpeechMarker::black_offset) == 1);
    CHECK(count_marker(result, SpeechMarker::cross_onset) == 1);
    CHECK(result.n_requests == 1);
    CHECK(result.requests[0].cue == CueKind::fixation_cross);
    CHECK(result.requests[0].onset_ns == black_end);
    CHECK(result.requests[0].duration_ns == first.cross_duration_ns);
    CHECK(result.snapshot.presentation.cue == CueKind::fixation_cross);

    CHECK(machine.step(cross_end, result) == ContractStatus::ok);
    CHECK(machine.state() == SpeechState::content);
    CHECK(count_marker(result, SpeechMarker::cross_offset) == 1);
    CHECK(count_marker(result, SpeechMarker::content_onset) == 1);
    CHECK(result.requests[0].cue == CueKind::text_content);
    CHECK(result.requests[0].stimulus_id == first.stimulus_id);
    CHECK(result.snapshot.trial.stimulus_id == first.stimulus_id);

    // The trial completes at the exact CONTENT end boundary, and the next trial
    // begins there.
    const SpeechTrialSchedule second = schedule_of(config, 1);
    CHECK(machine.step(content_end, second, result) == ContractStatus::ok);
    CHECK(result.trial_decided);
    CHECK(validate(result.trial) == ContractStatus::ok);
    CHECK(result.trial.record.interval.start_ns == 1'000);
    CHECK(result.trial.record.interval.end_ns == content_end);
    CHECK(result.trial.record.outcome == TrialOutcome::success);
    CHECK(result.trial.timeline.count == 3);
    CHECK(machine.state() == SpeechState::black);
    CHECK(machine.snapshot().trial.ordinal == 1);
    CHECK(machine.snapshot().active.start_ns == content_end);
}

void test_disabled_cross_has_no_state_event_or_interval()
{
    const SpeechCueConfig config = config_of(false, 1);
    const SpeechTrialSchedule only = schedule_of(config, 0);
    CHECK(only.cross_duration_ns == 0);
    CHECK(!only.cross_enabled);

    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, only, 0, result) == ContractStatus::ok);

    const ExperimentTimeNs black_end = only.black_duration_ns;
    CHECK(machine.step(black_end, result) == ContractStatus::ok);
    // CONTENT begins exactly at BLACK's end, with nothing in between.
    CHECK(machine.state() == SpeechState::content);
    CHECK(count_marker(result, SpeechMarker::black_offset) == 1);
    CHECK(count_marker(result, SpeechMarker::content_onset) == 1);
    CHECK(count_marker(result, SpeechMarker::cross_onset) == 0);
    CHECK(count_marker(result, SpeechMarker::cross_offset) == 0);
    CHECK(!has_transition_to(result, SpeechState::fixation_cross));
    CHECK(result.n_requests == 1);
    CHECK(result.requests[0].cue == CueKind::text_content);
    CHECK(result.requests[0].onset_ns == black_end);

    const ExperimentTimeNs content_end = black_end + only.content_duration_ns;
    CHECK(machine.step(content_end, result) == ContractStatus::ok);
    CHECK(result.trial_decided);
    // No zero-length cross is persisted anywhere: the phase is absent from the
    // recorded timeline, not present with no length.
    CHECK(result.trial.timeline.count == 2);
    for (std::uint8_t i = 0; i < result.trial.timeline.count; ++i)
        CHECK(result.trial.timeline.phases[i].phase != SpeechPhase::cross);
    CHECK(validate(result.trial) == ContractStatus::ok);
    CHECK(machine.state() == SpeechState::complete);
}

void test_boundaries_are_half_open()
{
    const SpeechCueConfig config = config_of(true, 1);
    const SpeechTrialSchedule only = schedule_of(config, 0);
    const ExperimentTimeNs start = 7'777;
    const ExperimentTimeNs black_end = start + only.black_duration_ns;
    const ExperimentTimeNs cross_end = black_end + only.cross_duration_ns;
    const ExperimentTimeNs content_end = cross_end + only.content_duration_ns;

    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, only, start, result) == ContractStatus::ok);
    for (const auto& probe : std::array<std::pair<ExperimentTimeNs, SpeechState>, 6>{
             {{start, SpeechState::black},
              {black_end - 1, SpeechState::black},
              {black_end, SpeechState::fixation_cross},
              {cross_end - 1, SpeechState::fixation_cross},
              {cross_end, SpeechState::content},
              {content_end - 1, SpeechState::content}}})
    {
        CHECK(machine.step(probe.first, result) == ContractStatus::ok);
        CHECK(machine.state() == probe.second);
    }
    // The end instant belongs to what follows, so the trial is over at it.
    CHECK(machine.step(content_end, result) == ContractStatus::ok);
    CHECK(result.trial_decided);
    CHECK(machine.state() == SpeechState::complete);
}

void test_large_jump_crosses_every_phase_once()
{
    const SpeechCueConfig config = config_of(true, 2);
    const SpeechTrialSchedule first = schedule_of(config, 0);
    const SpeechTrialSchedule second = schedule_of(config, 1);
    const ExperimentTimeNs black_end = first.black_duration_ns;
    const ExperimentTimeNs cross_end = black_end + first.cross_duration_ns;
    const ExperimentTimeNs content_end = cross_end + first.content_duration_ns;

    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, first, 0, result) == ContractStatus::ok);

    // Far past the end of the *second* trial as well, so the one-trial-per-step
    // bound is what stops the call rather than the clock running out.
    DurationNs second_total = 0;
    CHECK(trial_duration(second, second_total) == ContractStatus::ok);
    const ExperimentTimeNs far = content_end + second_total + 1'000'000;

    // One step, one whole trial. Every boundary is crossed at the instant it
    // actually occurred, not at the instant the caller happened to ask.
    CHECK(machine.step(far, second, result) == ContractStatus::ok);
    CHECK(count_marker(result, SpeechMarker::black_offset) == 1);
    CHECK(count_marker(result, SpeechMarker::cross_onset) == 1);
    CHECK(count_marker(result, SpeechMarker::cross_offset) == 1);
    CHECK(count_marker(result, SpeechMarker::content_onset) == 1);
    CHECK(count_marker(result, SpeechMarker::content_offset) == 1);
    CHECK(count_kind(result, ExperimentEventKind::trial_stop) == 1);
    CHECK(count_kind(result, ExperimentEventKind::trial_start) == 1);
    CHECK(result.trial_decided);
    for (std::uint8_t i = 0; i < result.n_events; ++i)
        CHECK(validate(result.events[i]) == ContractStatus::ok);
    // Boundaries carry their own instants.
    CHECK(result.trial.record.interval.end_ns == content_end);
    // A trial was decided while more time had already elapsed, so the step
    // stopped rather than inferring an unbounded run of trials.
    CHECK(!result.settled);
    CHECK(machine.state() == SpeechState::black);
    CHECK(machine.snapshot().trial.ordinal == 1);
    // Draining it takes one more call, and the second trial's boundaries are its
    // own rather than the instant the caller asked at.
    CHECK(machine.step(far, result) == ContractStatus::ok);
    CHECK(result.trial_decided);
    CHECK(result.trial.record.trial.ordinal == 1);
    CHECK(result.trial.record.interval.end_ns == content_end + second_total);
    CHECK(result.settled);
    CHECK(machine.complete());

    // Sequence ordinals are issued in order across every array a step fills.
    SequenceOrdinal previous = 0;
    bool first_record = true;
    for (std::uint8_t i = 0; i < result.n_events; ++i)
    {
        if (!first_record)
            CHECK(result.events[i].sequence > previous);
        previous = result.events[i].sequence;
        first_record = false;
    }
}

void test_repeated_timestamp_changes_nothing()
{
    const SpeechCueConfig config = config_of(true, 1);
    const SpeechTrialSchedule only = schedule_of(config, 0);
    const ExperimentTimeNs black_end = only.black_duration_ns;

    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, only, 0, result) == ContractStatus::ok);

    CHECK(machine.step(black_end, result) == ContractStatus::ok);
    CHECK(result.n_transitions == 1);
    const SpeechSnapshot after = machine.snapshot();
    for (int repeat = 0; repeat < 4; ++repeat)
    {
        CHECK(machine.step(black_end, result) == ContractStatus::ok);
        CHECK(result.n_transitions == 0);
        CHECK(result.n_events == 0);
        CHECK(result.n_requests == 0);
        CHECK(!result.trial_decided);
        CHECK(result.settled);
        CHECK(machine.snapshot().state == after.state);
        CHECK(machine.snapshot().active.start_ns == after.active.start_ns);
        CHECK(machine.snapshot().active.end_ns == after.active.end_ns);
    }
    // Time never moves backwards, and a refused step changes nothing.
    CHECK(machine.step(black_end - 1, result) == ContractStatus::time_regressed);
    CHECK(machine.snapshot().state == after.state);
}

void test_consecutive_trials_and_session_completion()
{
    constexpr TrialOrdinal kTrials = 5;
    const SpeechCueConfig config = config_of(true, kTrials);
    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, schedule_of(config, 0), 0, result) ==
          ContractStatus::ok);

    ExperimentTimeNs cursor = 0;
    std::size_t decided = 0;
    for (TrialOrdinal ordinal = 0; ordinal < kTrials; ++ordinal)
    {
        const SpeechTrialSchedule schedule = schedule_of(config, ordinal);
        DurationNs total = 0;
        CHECK(trial_duration(schedule, total) == ContractStatus::ok);
        cursor += total;
        // The next trial begins at the instant this one completes, so its BLACK
        // is the blank period between them and there is no fourth interval.
        if (ordinal + 1 < kTrials)
        {
            CHECK(machine.step(cursor, schedule_of(config, ordinal + 1), result) ==
                  ContractStatus::ok);
            CHECK(machine.snapshot().active.start_ns == cursor);
        }
        else
            CHECK(machine.step(cursor, result) == ContractStatus::ok);
        CHECK(result.trial_decided);
        CHECK(result.trial.record.trial.ordinal == ordinal);
        CHECK(result.trial.record.interval.end_ns == cursor);
        CHECK(validate(result.trial) == ContractStatus::ok);
        ++decided;
    }
    CHECK(decided == kTrials);
    CHECK(machine.complete());
    CHECK(machine.snapshot().completed == kTrials);
    CHECK(count_kind(result, ExperimentEventKind::session_stop) == 1);
    CHECK(has_transition_to(result, SpeechState::complete));
    // A finished session refuses to be stepped rather than sitting in a phase.
    CHECK(machine.step(cursor + 1, result) == ContractStatus::not_running);
    CHECK(machine.snapshot().presentation.cue == CueKind::none);
}

void test_trial_boundary_needs_next_schedule()
{
    const SpeechCueConfig config = config_of(true, 2);
    const SpeechTrialSchedule first = schedule_of(config, 0);
    DurationNs total = 0;
    CHECK(trial_duration(first, total) == ContractStatus::ok);

    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, first, 0, result) == ContractStatus::ok);

    // The boundary is not a place the machine can stop, so a missing schedule is
    // a missing input: nothing moves, and the caller can retry.
    const SpeechSnapshot before = machine.snapshot();
    CHECK(machine.step(total, result) == ContractStatus::identity_missing);
    CHECK(machine.snapshot().state == before.state);
    CHECK(machine.snapshot().completed == before.completed);
    CHECK(machine.snapshot().trial.ordinal == before.trial.ordinal);

    // A schedule for the wrong trial is refused, and so is one that does not
    // belong to the configuration at all.
    CHECK(machine.step(total, schedule_of(config, 0), result) == ContractStatus::outcome_invalid);
    SpeechTrialSchedule foreign = schedule_of(config, 1);
    foreign.stimulus_id = 999;
    CHECK(machine.step(total, foreign, result) == ContractStatus::target_set_invalid);
    CHECK(machine.snapshot().state == before.state);

    // The right one works, and the retry loses nothing.
    CHECK(machine.step(total, schedule_of(config, 1), result) == ContractStatus::ok);
    CHECK(result.trial_decided);
    CHECK(machine.snapshot().trial.ordinal == 1);
}

void test_start_rejects_unrunnable_configuration()
{
    const SpeechCueConfig config = config_of(true, 2);
    const SpeechTrialSchedule first = schedule_of(config, 0);
    SpeechStepResult result{};

    SpeechMachine unset{};
    CHECK(unset.start(kUnsetParadigmId, config, first, 0, result) ==
          ContractStatus::identity_missing);
    CHECK(unset.state() == SpeechState::idle);

    // A configured inter-trial gap is refused rather than ignored: this machine
    // has no state to spend it in, and running the session anyway would disagree
    // with the timeline the same configuration builds.
    SpeechCueConfig gapped = config;
    gapped.inter_trial_ns = 1'000;
    SpeechMachine gap_machine{};
    CHECK(gap_machine.start(kParadigm, gapped, first, 0, result) ==
          ContractStatus::outcome_invalid);
    CHECK(gap_machine.state() == SpeechState::idle);

    // The first trial is trial zero.
    SpeechMachine misnumbered{};
    CHECK(misnumbered.start(kParadigm, config, schedule_of(config, 1), 0, result) ==
          ContractStatus::outcome_invalid);

    SpeechMachine running{};
    CHECK(running.start(kParadigm, config, first, 0, result) == ContractStatus::ok);
    CHECK(running.start(kParadigm, config, first, 0, result) == ContractStatus::already_running);
    CHECK(running.step(0, result) == ContractStatus::ok);

    SpeechMachine idle{};
    CHECK(idle.step(0, result) == ContractStatus::not_running);
}

void test_reset_then_replay_reproduces_run()
{
    const SpeechCueConfig config = config_of(true, 3);
    const auto run_session = [&config](SpeechMachine& machine, std::array<SpeechTrial, 3>& trials)
    {
        SpeechStepResult result{};
        CHECK(machine.start(kParadigm, config, schedule_of(config, 0), 500, result) ==
              ContractStatus::ok);
        ExperimentTimeNs cursor = 500;
        for (TrialOrdinal ordinal = 0; ordinal < 3; ++ordinal)
        {
            DurationNs total = 0;
            CHECK(trial_duration(schedule_of(config, ordinal), total) == ContractStatus::ok);
            cursor += total;
            if (ordinal + 1 < 3)
                CHECK(machine.step(cursor, schedule_of(config, ordinal + 1), result) ==
                      ContractStatus::ok);
            else
                CHECK(machine.step(cursor, result) == ContractStatus::ok);
            trials[static_cast<std::size_t>(ordinal)] = result.trial;
        }
    };

    SpeechMachine machine{};
    std::array<SpeechTrial, 3> original{};
    run_session(machine, original);
    CHECK(machine.complete());

    // A reset session is a new session, not a rewound one, so it decides
    // everything again and reproduces the run exactly.
    machine.reset();
    CHECK(machine.state() == SpeechState::idle);
    CHECK(machine.paradigm() == kUnsetParadigmId);
    CHECK(machine.configuration().n_trials == 0);
    std::array<SpeechTrial, 3> replayed{};
    run_session(machine, replayed);
    for (std::size_t i = 0; i < original.size(); ++i)
    {
        CHECK(original[i].record.interval.start_ns == replayed[i].record.interval.start_ns);
        CHECK(original[i].record.interval.end_ns == replayed[i].record.interval.end_ns);
        CHECK(original[i].schedule.stimulus_id == replayed[i].schedule.stimulus_id);
        CHECK(original[i].timeline.count == replayed[i].timeline.count);
    }

    // A copy is a session, and a copy stepped alike stays identical.
    SpeechMachine copy = machine;
    CHECK(copy.state() == machine.state());
    CHECK(copy.snapshot().completed == machine.snapshot().completed);
}

void test_record_identity_timing_and_provenance()
{
    const SpeechCueConfig config = config_of(true, 1);
    const SpeechTrialSchedule only = schedule_of(config, 0);
    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, only, 4'000, result) == ContractStatus::ok);

    for (std::uint8_t i = 0; i < result.n_events; ++i)
    {
        CHECK(result.events[i].paradigm == kParadigm);
        CHECK(result.events[i].trial.ordinal == 0);
        CHECK(result.events[i].trial.stimulus_id == only.stimulus_id);
        CHECK(result.events[i].time_ns == 4'000);
    }
    CHECK(result.transitions[0].paradigm == kParadigm);
    CHECK(result.transitions[0].from_state == static_cast<StateId>(SpeechState::idle));
    CHECK(validate(result.transitions[0]) == ContractStatus::ok);

    // The identity the machine stamps is the identity make_schedule_draws()
    // requires, so a caller can file this trial's realized durations without
    // rebuilding one.
    SpeechTrialDraws draws{};
    CHECK(make_schedule_draws(only, result.snapshot.trial, 4'000, draws) == ContractStatus::ok);
    CHECK(draws.count == 3);

    DurationNs total = 0;
    CHECK(trial_duration(only, total) == ContractStatus::ok);
    CHECK(machine.step(4'000 + total, result) == ContractStatus::ok);
    CHECK(result.trial.record.paradigm == kParadigm);
    CHECK(result.trial.record.trial.stimulus_id == only.stimulus_id);
    CHECK(result.trial.timeline.trial.start_ns == 4'000);
    CHECK(result.trial.timeline.next_trial_start_ns == result.trial.timeline.trial.end_ns);

    // A hand-built trial whose artefacts disagree is refused.
    SpeechTrial tampered = result.trial;
    tampered.schedule.content_duration_ns += 1;
    CHECK(validate(tampered) == ContractStatus::outcome_invalid);
    SpeechTrial relabelled = result.trial;
    relabelled.record.trial.stimulus_id += 1;
    CHECK(validate(relabelled) == ContractStatus::outcome_invalid);
    SpeechTrial recrossed = result.trial;
    recrossed.schedule.cross_enabled = false;
    recrossed.schedule.cross_duration_ns = 0;
    CHECK(validate(recrossed) == ContractStatus::outcome_invalid);
}

void test_machine_allocates_nothing()
{
    const SpeechCueConfig config = config_of(true, 64);
    SpeechMachine machine{};
    SpeechStepResult result{};
    CHECK(machine.start(kParadigm, config, schedule_of(config, 0), 0, result) ==
          ContractStatus::ok);

    const std::size_t before = allocations.load(std::memory_order_relaxed);
    ExperimentTimeNs cursor = 0;
    for (TrialOrdinal ordinal = 0; ordinal < 64; ++ordinal)
    {
        SpeechTrialSchedule schedule{};
        if (prepare_trial(config, ordinal, schedule) != ContractStatus::ok)
            ++failures;
        DurationNs total = 0;
        if (trial_duration(schedule, total) != ContractStatus::ok)
            ++failures;
        cursor += total;
        if (ordinal + 1 < 64)
        {
            SpeechTrialSchedule next{};
            if (prepare_trial(config, ordinal + 1, next) != ContractStatus::ok)
                ++failures;
            if (machine.step(cursor, next, result) != ContractStatus::ok)
                ++failures;
        }
        else if (machine.step(cursor, result) != ContractStatus::ok)
            ++failures;
    }
    CHECK(allocations.load(std::memory_order_relaxed) == before);
    CHECK(machine.complete());
}

} // namespace

int run()
{
    test_ordinary_black_cross_content_sequence();
    test_disabled_cross_has_no_state_event_or_interval();
    test_boundaries_are_half_open();
    test_large_jump_crosses_every_phase_once();
    test_repeated_timestamp_changes_nothing();
    test_consecutive_trials_and_session_completion();
    test_trial_boundary_needs_next_schedule();
    test_start_rejects_unrunnable_configuration();
    test_reset_then_replay_reproduces_run();
    test_record_identity_timing_and_provenance();
    test_machine_allocates_nothing();
    if (failures != 0)
        std::cerr << failures << " Speech machine assertion(s) failed\n";
    return failures == 0 ? 0 : 1;
}
} // namespace machine_tests

int main()
{
    const int machine_status = machine_tests::run();
    test_sampled_durations_stay_strictly_inside_their_bounds();
    test_cross_is_configuration_and_never_a_draw();
    test_timeline_is_contiguous_and_half_open();
    test_determinism_and_seed_divergence();
    test_stimulus_order_policies();
    test_explicit_schedule_is_own_authority();
    test_supplied_schedules_are_checked_not_trusted();
    test_invalid_timing_domains_and_configuration();
    test_catalog_ownership_and_resolution();
    test_identity_and_fingerprints();
    test_schedule_must_belong_to_configuration();
    test_draw_provenance();
    test_full_capacity_and_long_sessions();
    test_native_path_allocates_nothing();
    if (failures != 0)
        std::cerr << failures << " Speech assertion(s) failed\n";
    return failures == 0 && machine_status == 0 ? 0 : 1;
}
