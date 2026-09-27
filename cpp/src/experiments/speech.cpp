// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "utf8.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <neurale/experiments/speech.h>

namespace neurale::experiments::speech
{
namespace
{

using neurale::experiments::detail::is_well_formed_utf8;

[[nodiscard]] bool text_padding_clear(const SpeechStimulus& stimulus) noexcept
{
    for (std::size_t i = stimulus.text_length; i < stimulus.text.size(); ++i)
        if (stimulus.text[i] != '\0')
            return false;
    return true;
}

// The stimulus text is canonical UTF-8, the one encoding the contract imposes,
// so an entry whose in-use bytes are not a complete, well-formed UTF-8
// sequence is rejected here rather than handed to a renderer that would then
// have to invent its own decoding policy. The shared decoder lives in utf8.h,
// so the native validator and the Python constructor check the same thing.
// This runs on the prepare path, never on a hot one.
[[nodiscard]] bool text_is_well_formed_utf8(const SpeechStimulus& stimulus) noexcept
{
    return is_well_formed_utf8(reinterpret_cast<const std::uint8_t*>(stimulus.text.data()),
                               stimulus.text_length);
}

// Every field, sampler version included. Unused slots are absorbed into
// configuration_fingerprint() along with the used ones, so "unused" has to mean
// one exact bit pattern: two configurations that schedule identically must not
// be told apart by a value nothing reads.
[[nodiscard]] bool default_entry(const SpeechTrialSchedule& entry) noexcept
{
    return entry.ordinal == 0 && entry.black_duration_ns == 0 && entry.cross_duration_ns == 0 &&
           entry.content_duration_ns == 0 && entry.stimulus_id == kUnsetStimulusId &&
           entry.sampler_version == 0 && !entry.cross_enabled;
}

[[nodiscard]] bool same_schedule(const SpeechTrialSchedule& left,
                                 const SpeechTrialSchedule& right) noexcept
{
    return left.ordinal == right.ordinal && left.black_duration_ns == right.black_duration_ns &&
           left.cross_duration_ns == right.cross_duration_ns &&
           left.content_duration_ns == right.content_duration_ns &&
           left.stimulus_id == right.stimulus_id && left.sampler_version == right.sampler_version &&
           left.cross_enabled == right.cross_enabled;
}

[[nodiscard]] bool configured_stimulus(const SpeechCueConfig& config, StimulusId id) noexcept
{
    for (std::size_t i = 0; i < config.n_stimuli; ++i)
        if (config.stimuli[i] == id)
            return true;
    return false;
}

[[nodiscard]] bool default_stimulus(const SpeechStimulus& stimulus) noexcept
{
    if (stimulus.id != kUnsetStimulusId || stimulus.label != kUnsetSpeechLabel ||
        stimulus.metadata != 0 || stimulus.content != SpeechContentKind::unspecified ||
        stimulus.text_length != 0)
        return false;
    for (const char byte : stimulus.text)
        if (byte != '\0')
            return false;
    return true;
}

// Resolution without validation, so that a caller checking a whole catalog
// against a whole configuration validates each of them once rather than once
// per lookup.
[[nodiscard]] bool lookup(const SpeechCatalog& catalog, StimulusId id,
                          SpeechStimulus& stimulus) noexcept
{
    for (std::size_t i = 0; i < catalog.count; ++i)
        if (catalog.entries[i].id == id)
        {
            stimulus = catalog.entries[i];
            return true;
        }
    return false;
}

// The open range (0, bound) holds an integer exactly when the bound is at least
// two: (0, 1) is empty, and (0, 2) holds only 1. This is the criterion
// sample_exclusive() applies, restated here so the configuration is rejected at
// validation rather than at the first trial that tries to draw against it.
[[nodiscard]] bool samplable_bound(DurationNs bound_ns) noexcept
{
    return bound_ns >= 2;
}

// The half of validate_against() that does not re-enter validate(), so that
// validate(const SpeechCueConfig&) can apply it to every explicit entry without
// recursing back into itself.
[[nodiscard]] ContractStatus check_entry_against(const SpeechTrialSchedule& schedule,
                                                 const SpeechCueConfig& config) noexcept
{
    if (schedule.cross_enabled != config.cross_enabled)
        return ContractStatus::outcome_invalid;
    // The session's ScheduleIdentity takes its sampler version from the
    // configuration. A schedule naming a different one is not a schedule with
    // extra provenance, it is a session claiming two samplers.
    if (schedule.sampler_version != config.sampler_version)
        return ContractStatus::outcome_invalid;
    if (schedule.black_duration_ns >= config.black_bound_ns)
        return ContractStatus::parameter_out_of_range;
    if (schedule.content_duration_ns >= config.content_bound_ns)
        return ContractStatus::parameter_out_of_range;
    if (config.cross_enabled && schedule.cross_duration_ns >= config.cross_bound_ns)
        return ContractStatus::parameter_out_of_range;
    return ContractStatus::ok;
}

// Stimulus resolution without re-validating the configuration, for the same
// reason as lookup() above: prepare_trial() has already validated it.
[[nodiscard]] ContractStatus select_stimulus_unchecked(const SpeechCueConfig& config,
                                                       TrialOrdinal ordinal,
                                                       StimulusId& stimulus_id) noexcept
{
    if (config.schedule == SpeechScheduleKind::explicit_sequence)
    {
        stimulus_id = config.explicit_schedule[static_cast<std::size_t>(ordinal)].stimulus_id;
        return ContractStatus::ok;
    }

    const std::uint64_t count = config.n_stimuli;
    if (config.stimulus_order == StimulusOrderPolicy::sequential)
    {
        // No draw is made, so no sampler version is required either: stepping
        // through a set in order is reproducible under any sampler, and
        // refusing it under an unsupported one would reject a session that has
        // nothing to regenerate.
        stimulus_id = config.stimuli[static_cast<std::size_t>(ordinal % count)];
        return ContractStatus::ok;
    }

    if (!sampler_version_supported(config.sampler_version))
        return ContractStatus::version_unsupported;

    if (config.stimulus_order == StimulusOrderPolicy::random_with_replacement)
    {
        std::uint64_t sampled = 0;
        const DrawKey key{config.seed, kStimulusOrderStream, ordinal, 0};
        if (const ContractStatus status = sample_index(key, count, sampled);
            status != ContractStatus::ok)
            return status;
        stimulus_id = config.stimuli[static_cast<std::size_t>(sampled)];
        return ContractStatus::ok;
    }

    // StimulusOrderPolicy::shuffled_blocks. One permutation per block of
    // `count` trials, addressed by the block index in the draw key's trial
    // field: every trial of a block therefore rebuilds the *same* permutation
    // and reads its own position out of it, which is what makes the policy
    // reconstructible from one ordinal without the ordinals before it.
    //
    // Durstenfeld's shuffle, written out rather than delegated: std::shuffle
    // takes a UniformRandomBitGenerator and its results are not specified
    // across standard libraries, so a session shuffled on one platform would
    // not replay on another.
    const std::uint64_t block = ordinal / count;
    const std::size_t pos = static_cast<std::size_t>(ordinal % count);
    std::array<StimulusId, kMaxSpeechStimuli> order{};
    for (std::size_t i = 0; i < count; ++i)
        order[i] = config.stimuli[i];
    std::uint32_t draw = 0;
    for (std::size_t i = static_cast<std::size_t>(count); i-- > 1; ++draw)
    {
        std::uint64_t chosen = 0;
        const DrawKey key{config.seed, kStimulusOrderStream, block, draw};
        if (const ContractStatus status = sample_inclusive(key, 0, i, chosen);
            status != ContractStatus::ok)
            return status;
        const StimulusId held = order[i];
        order[i] = order[static_cast<std::size_t>(chosen)];
        order[static_cast<std::size_t>(chosen)] = held;
    }
    stimulus_id = order[pos];
    return ContractStatus::ok;
}

void absorb_stimulus(FingerprintAccumulator& accumulator, const SpeechStimulus& stimulus) noexcept
{
    accumulator.absorb(stimulus.id);
    accumulator.absorb(stimulus.label);
    accumulator.absorb(stimulus.metadata);
    accumulator.absorb(static_cast<std::uint8_t>(stimulus.content));
    accumulator.absorb(stimulus.text_length);
    for (const char byte : stimulus.text)
        accumulator.absorb(static_cast<unsigned char>(byte));
}

void absorb_entry(FingerprintAccumulator& accumulator, const SpeechTrialSchedule& entry) noexcept
{
    accumulator.absorb(entry.ordinal);
    accumulator.absorb(entry.black_duration_ns);
    accumulator.absorb(entry.cross_duration_ns);
    accumulator.absorb(entry.content_duration_ns);
    accumulator.absorb(entry.stimulus_id);
    accumulator.absorb(entry.sampler_version);
    accumulator.absorb(entry.cross_enabled ? 1u : 0u);
}

} // namespace

ContractStatus validate(const SpeechStimulus& stimulus) noexcept
{
    if (!speech_content_kind_declared(stimulus.content))
        return ContractStatus::enum_undeclared;
    if (stimulus.id == kUnsetStimulusId)
        return ContractStatus::identity_missing;
    if (stimulus.content == SpeechContentKind::unspecified)
        return ContractStatus::identity_missing;
    if (static_cast<std::size_t>(stimulus.text_length) > kMaxSpeechTextBytes)
        return ContractStatus::parameter_out_of_range;
    // An entry with no content is not a narrower stimulus, it is one whose
    // meaning cannot be recovered from the catalog at all.
    if (stimulus.text_length == 0)
        return ContractStatus::identity_missing;
    if (!text_padding_clear(stimulus))
        return ContractStatus::outcome_invalid;
    if (!text_is_well_formed_utf8(stimulus))
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const SpeechCatalog& catalog) noexcept
{
    if (catalog.count == 0 || static_cast<std::size_t>(catalog.count) > kMaxSpeechStimuli)
        return ContractStatus::target_set_invalid;
    for (std::size_t i = 0; i < catalog.count; ++i)
        if (const ContractStatus status = validate(catalog.entries[i]);
            status != ContractStatus::ok)
            return status;
    for (std::size_t left = 0; left < catalog.count; ++left)
        for (std::size_t right = left + 1; right < catalog.count; ++right)
            if (catalog.entries[left].id == catalog.entries[right].id)
                return ContractStatus::target_set_invalid;
    for (std::size_t i = catalog.count; i < catalog.entries.size(); ++i)
        if (!default_stimulus(catalog.entries[i]))
            return ContractStatus::target_set_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const SpeechTrialSchedule& schedule) noexcept
{
    if (schedule.sampler_version == 0)
        return ContractStatus::identity_missing;
    if (schedule.stimulus_id == kUnsetStimulusId)
        return ContractStatus::identity_missing;
    if (schedule.black_duration_ns == 0 || schedule.content_duration_ns == 0)
        return ContractStatus::parameter_out_of_range;
    // The whole point of the open range. An enabled cross of zero and a
    // disabled cross of anything else are the two ways a schedule can claim a
    // phase it does not have.
    if (schedule.cross_enabled != (schedule.cross_duration_ns != 0))
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus validate(const SpeechCueConfig& config) noexcept
{
    if (!speech_schedule_kind_declared(config.schedule) ||
        !stimulus_order_policy_declared(config.stimulus_order))
        return ContractStatus::enum_undeclared;
    if (config.schedule == SpeechScheduleKind::unspecified)
        return ContractStatus::identity_missing;
    if (config.sampler_version == 0)
        return ContractStatus::identity_missing;
    if (config.n_trials == 0)
        return ContractStatus::range_empty;

    if (!samplable_bound(config.black_bound_ns) || !samplable_bound(config.content_bound_ns))
        return ContractStatus::range_empty;
    if (config.cross_enabled)
    {
        if (!samplable_bound(config.cross_bound_ns))
            return ContractStatus::range_empty;
    }
    else if (config.cross_bound_ns != 0)
    {
        // Inert and nonzero. Every other field of a disabled cross says the
        // phase does not exist; a bound left behind says a domain is being
        // sampled, and one of the two is going to be believed.
        return ContractStatus::outcome_invalid;
    }

    if (config.schedule == SpeechScheduleKind::seeded)
    {
        if (config.stimulus_order == StimulusOrderPolicy::unspecified)
            return ContractStatus::identity_missing;
        if (config.n_stimuli == 0 || static_cast<std::size_t>(config.n_stimuli) > kMaxSpeechStimuli)
            return ContractStatus::target_set_invalid;
        for (std::size_t i = 0; i < config.n_stimuli; ++i)
            if (config.stimuli[i] == kUnsetStimulusId)
                return ContractStatus::target_set_invalid;
        for (std::size_t left = 0; left < config.n_stimuli; ++left)
            for (std::size_t right = left + 1; right < config.n_stimuli; ++right)
                if (config.stimuli[left] == config.stimuli[right])
                    return ContractStatus::target_set_invalid;
        for (std::size_t i = config.n_stimuli; i < config.stimuli.size(); ++i)
            if (config.stimuli[i] != kUnsetStimulusId)
                return ContractStatus::target_set_invalid;
        if (config.n_explicit != 0)
            return ContractStatus::outcome_invalid;
        for (const SpeechTrialSchedule& entry : config.explicit_schedule)
            if (!default_entry(entry))
                return ContractStatus::outcome_invalid;
        return ContractStatus::ok;
    }

    // SpeechScheduleKind::explicit_sequence. Everything that would decide a
    // realized value has to be absent, because the schedule below decides all
    // of them and a second answer is not a default -- it is a disagreement
    // nobody would notice.
    if (config.stimulus_order != StimulusOrderPolicy::unspecified)
        return ContractStatus::outcome_invalid;
    if (config.n_stimuli != 0)
        return ContractStatus::outcome_invalid;
    for (const StimulusId id : config.stimuli)
        if (id != kUnsetStimulusId)
            return ContractStatus::outcome_invalid;
    if (static_cast<std::size_t>(config.n_explicit) > kMaxSpeechExplicitTrials)
        return ContractStatus::parameter_out_of_range;
    if (config.n_trials != config.n_explicit)
        return ContractStatus::outcome_invalid;
    for (std::size_t i = 0; i < config.n_explicit; ++i)
    {
        const SpeechTrialSchedule& entry = config.explicit_schedule[i];
        if (const ContractStatus status = validate(entry); status != ContractStatus::ok)
            return status;
        if (entry.ordinal != i)
            return ContractStatus::outcome_invalid;
        if (const ContractStatus status = check_entry_against(entry, config);
            status != ContractStatus::ok)
            return status;
    }
    for (std::size_t i = config.n_explicit; i < config.explicit_schedule.size(); ++i)
        if (!default_entry(config.explicit_schedule[i]))
            return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus validate_against(const SpeechTrialSchedule& schedule,
                                const SpeechCueConfig& config) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(schedule); status != ContractStatus::ok)
        return status;
    if (schedule.ordinal >= config.n_trials)
        return ContractStatus::range_empty;
    if (const ContractStatus status = check_entry_against(schedule, config);
        status != ContractStatus::ok)
        return status;

    if (config.schedule == SpeechScheduleKind::explicit_sequence)
    {
        // The configuration already holds the authoritative realized schedule
        // for this ordinal. Anything else is a different schedule wearing its
        // number, however well it fits the bounds.
        if (!same_schedule(schedule,
                           config.explicit_schedule[static_cast<std::size_t>(schedule.ordinal)]))
            return ContractStatus::outcome_invalid;
        return ContractStatus::ok;
    }

    // Seeded. The realized durations are not recomputed here -- see the header
    // -- but the stimulus is checked for membership, because a stimulus outside
    // the configured set is one the deterministic order could not have chosen
    // under any seed, and admitting it would let the public timeline present
    // content the schedule does not own.
    if (!configured_stimulus(config, schedule.stimulus_id))
        return ContractStatus::target_set_invalid;
    return ContractStatus::ok;
}

ContractStatus validate_against(const SpeechCueConfig& config,
                                const SpeechCatalog& catalog) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(catalog); status != ContractStatus::ok)
        return status;

    SpeechStimulus resolved{};
    if (config.schedule == SpeechScheduleKind::explicit_sequence)
    {
        for (std::size_t i = 0; i < config.n_explicit; ++i)
            if (!lookup(catalog, config.explicit_schedule[i].stimulus_id, resolved))
                return ContractStatus::identity_missing;
        return ContractStatus::ok;
    }
    for (std::size_t i = 0; i < config.n_stimuli; ++i)
        if (!lookup(catalog, config.stimuli[i], resolved))
            return ContractStatus::identity_missing;
    return ContractStatus::ok;
}

ContractStatus find_stimulus(const SpeechCatalog& catalog, StimulusId id,
                             SpeechStimulus& stimulus) noexcept
{
    if (const ContractStatus status = validate(catalog); status != ContractStatus::ok)
        return status;
    if (id == kUnsetStimulusId)
        return ContractStatus::identity_missing;
    SpeechStimulus resolved{};
    if (!lookup(catalog, id, resolved))
        return ContractStatus::identity_missing;
    stimulus = resolved;
    return ContractStatus::ok;
}

ContractStatus select_stimulus(const SpeechCueConfig& config, TrialOrdinal ordinal,
                               StimulusId& stimulus_id) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (ordinal >= config.n_trials)
        return ContractStatus::range_empty;
    StimulusId selected = kUnsetStimulusId;
    if (const ContractStatus status = select_stimulus_unchecked(config, ordinal, selected);
        status != ContractStatus::ok)
        return status;
    stimulus_id = selected;
    return ContractStatus::ok;
}

ContractStatus prepare_trial(const SpeechCueConfig& config, TrialOrdinal ordinal,
                             SpeechTrialSchedule& schedule) noexcept
{
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    if (ordinal >= config.n_trials)
        return ContractStatus::range_empty;

    if (config.schedule == SpeechScheduleKind::explicit_sequence)
    {
        schedule = config.explicit_schedule[static_cast<std::size_t>(ordinal)];
        return ContractStatus::ok;
    }

    if (!sampler_version_supported(config.sampler_version))
        return ContractStatus::version_unsupported;

    SpeechTrialSchedule produced{};
    produced.ordinal = ordinal;
    produced.sampler_version = config.sampler_version;
    produced.cross_enabled = config.cross_enabled;
    if (const ContractStatus status =
            select_stimulus_unchecked(config, ordinal, produced.stimulus_id);
        status != ContractStatus::ok)
        return status;

    // Open at both ends: sample_exclusive(key, 0, bound) is exactly
    // [1, bound - 1], so a realized duration is never zero and never reaches
    // the bound it was configured with.
    if (const ContractStatus status =
            sample_exclusive(DrawKey{config.seed, kTrialTimingStream, ordinal, kBlackDraw}, 0,
                             config.black_bound_ns, produced.black_duration_ns);
        status != ContractStatus::ok)
        return status;
    if (config.cross_enabled)
    {
        if (const ContractStatus status =
                sample_exclusive(DrawKey{config.seed, kTrialTimingStream, ordinal, kCrossDraw}, 0,
                                 config.cross_bound_ns, produced.cross_duration_ns);
            status != ContractStatus::ok)
            return status;
    }
    // kContentDraw whether or not the cross was drawn. The draw ordinals are
    // coordinates, not a running counter, so disabling the cross leaves every
    // black and content duration in the session exactly where it was.
    if (const ContractStatus status =
            sample_exclusive(DrawKey{config.seed, kTrialTimingStream, ordinal, kContentDraw}, 0,
                             config.content_bound_ns, produced.content_duration_ns);
        status != ContractStatus::ok)
        return status;

    schedule = produced;
    return ContractStatus::ok;
}

ContractStatus trial_duration(const SpeechTrialSchedule& schedule, DurationNs& duration_ns) noexcept
{
    if (const ContractStatus status = validate(schedule); status != ContractStatus::ok)
        return status;
    constexpr DurationNs limit = (std::numeric_limits<DurationNs>::max)();
    DurationNs total = schedule.black_duration_ns;
    if (schedule.cross_duration_ns > limit - total)
        return ContractStatus::duration_overflow;
    total += schedule.cross_duration_ns;
    if (schedule.content_duration_ns > limit - total)
        return ContractStatus::duration_overflow;
    total += schedule.content_duration_ns;
    duration_ns = total;
    return ContractStatus::ok;
}

ContractStatus build_timeline(const SpeechCueConfig& config, const SpeechTrialSchedule& schedule,
                              ExperimentTimeNs start_ns, SpeechTimeline& timeline) noexcept
{
    if (const ContractStatus status = validate_against(schedule, config);
        status != ContractStatus::ok)
        return status;

    SpeechTimeline produced{};
    ExperimentTimeNs cursor = start_ns;

    const auto append = [&produced, &cursor](SpeechPhase phase, CueKind cue, StimulusId stimulus_id,
                                             DurationNs duration_ns) noexcept
    {
        TimeInterval interval{};
        if (const ContractStatus status = interval_from_duration(cursor, duration_ns, interval);
            status != ContractStatus::ok)
            return status;
        produced.phases[produced.count] = SpeechPhaseInterval{interval, phase, cue, stimulus_id};
        ++produced.count;
        cursor = interval.end_ns;
        return ContractStatus::ok;
    };

    if (const ContractStatus status = append(SpeechPhase::black, CueKind::black, kUnsetStimulusId,
                                             schedule.black_duration_ns);
        status != ContractStatus::ok)
        return status;
    // Absent, not empty. A cross entry of zero length would be indistinguishable
    // from one that was configured away, and the whole disabled/enabled
    // distinction rests on those two never looking alike.
    if (schedule.cross_enabled)
        if (const ContractStatus status = append(SpeechPhase::cross, CueKind::fixation_cross,
                                                 kUnsetStimulusId, schedule.cross_duration_ns);
            status != ContractStatus::ok)
            return status;
    if (const ContractStatus status = append(SpeechPhase::content, CueKind::text_content,
                                             schedule.stimulus_id, schedule.content_duration_ns);
        status != ContractStatus::ok)
        return status;

    produced.trial = TimeInterval{start_ns, cursor};
    // The gap is not a phase of the trial: the trial is over at the end of its
    // content, and the gap is the space before the next one begins. It carries
    // no cue, and this paradigm makes no claim about what a display shows
    // during it.
    if (config.inter_trial_ns != 0)
        if (const ContractStatus status = append(SpeechPhase::inter_trial, CueKind::none,
                                                 kUnsetStimulusId, config.inter_trial_ns);
            status != ContractStatus::ok)
            return status;
    produced.next_trial_start_ns = cursor;

    timeline = produced;
    return ContractStatus::ok;
}

ContractStatus make_schedule_draws(const SpeechTrialSchedule& schedule, const TrialIdentity& trial,
                                   ExperimentTimeNs time_ns, SpeechTrialDraws& draws) noexcept
{
    if (const ContractStatus status = validate(schedule); status != ContractStatus::ok)
        return status;
    if (trial.stimulus_id == kUnsetStimulusId)
        return ContractStatus::identity_missing;
    if (trial.ordinal != schedule.ordinal || trial.stimulus_id != schedule.stimulus_id)
        return ContractStatus::outcome_invalid;

    SpeechTrialDraws produced{};
    const auto append =
        [&produced, &schedule, &trial, time_ns](std::uint32_t draw, std::uint64_t value) noexcept
    {
        ScheduleDraw record{};
        record.time_ns = time_ns;
        record.trial = trial;
        record.stream = kTrialTimingStream;
        record.draw = draw;
        record.value = value;
        record.sampler_version = schedule.sampler_version;
        produced.draws[produced.count] = record;
        ++produced.count;
    };

    append(kBlackDraw, schedule.black_duration_ns);
    if (schedule.cross_enabled)
        append(kCrossDraw, schedule.cross_duration_ns);
    append(kContentDraw, schedule.content_duration_ns);

    draws = produced;
    return ContractStatus::ok;
}

std::uint64_t configuration_fingerprint(const SpeechCueConfig& config) noexcept
{
    FingerprintAccumulator accumulator;
    accumulator.absorb(config.black_bound_ns);
    accumulator.absorb(config.cross_bound_ns);
    accumulator.absorb(config.content_bound_ns);
    accumulator.absorb(config.inter_trial_ns);
    accumulator.absorb(config.seed);
    accumulator.absorb(config.n_trials);
    accumulator.absorb(static_cast<std::uint8_t>(config.schedule));
    accumulator.absorb(static_cast<std::uint8_t>(config.stimulus_order));
    accumulator.absorb(config.sampler_version);
    accumulator.absorb(config.cross_enabled ? 1u : 0u);
    accumulator.absorb(config.n_stimuli);
    for (const StimulusId id : config.stimuli)
        accumulator.absorb(id);
    accumulator.absorb(config.n_explicit);
    for (const SpeechTrialSchedule& entry : config.explicit_schedule)
        absorb_entry(accumulator, entry);
    return accumulator.value();
}

std::uint64_t catalog_fingerprint(const SpeechCatalog& catalog) noexcept
{
    FingerprintAccumulator accumulator;
    accumulator.absorb(catalog.count);
    for (const SpeechStimulus& stimulus : catalog.entries)
        absorb_stimulus(accumulator, stimulus);
    return accumulator.value();
}

ScheduleIdentity schedule_identity(const SpeechCueConfig& config,
                                   const SpeechCatalog& catalog) noexcept
{
    ScheduleIdentity identity{};
    identity.seed = config.seed;
    identity.sampler_version = config.sampler_version;
    identity.configuration_fingerprint = configuration_fingerprint(config);
    identity.catalog_fingerprint = catalog_fingerprint(catalog);
    return identity;
}

// State machine.
namespace
{

// The trial the current records belong to. The stimulus rides along because a
// Speech record read back without it says which trial it came from and not what
// that trial presented, and because make_schedule_draws() requires exactly this
// identity for the same trial's realized durations.
[[nodiscard]] TrialIdentity identity_of(TrialOrdinal ordinal, StimulusId stimulus_id) noexcept
{
    TrialIdentity identity{};
    identity.ordinal = ordinal;
    identity.stimulus_id = stimulus_id;
    return identity;
}

[[nodiscard]] SpeechMarker onset_marker_of(SpeechPhase phase) noexcept
{
    return phase == SpeechPhase::black   ? SpeechMarker::black_onset
           : phase == SpeechPhase::cross ? SpeechMarker::cross_onset
                                         : SpeechMarker::content_onset;
}

[[nodiscard]] SpeechMarker offset_marker_of(SpeechPhase phase) noexcept
{
    return phase == SpeechPhase::black   ? SpeechMarker::black_offset
           : phase == SpeechPhase::cross ? SpeechMarker::cross_offset
                                         : SpeechMarker::content_offset;
}

[[nodiscard]] SpeechState state_of_phase(SpeechPhase phase) noexcept
{
    return phase == SpeechPhase::black   ? SpeechState::black
           : phase == SpeechPhase::cross ? SpeechState::fixation_cross
                                         : SpeechState::content;
}

// The timeline is the authority on where a phase runs, so the machine looks a
// boundary up rather than recomputing it from a duration. A phase the trial does
// not have -- the cross of a trial configured without one -- is simply not in
// there, which is how "absent" stays distinguishable from "zero length".
[[nodiscard]] bool interval_of(const SpeechTimeline& timeline, SpeechPhase phase,
                               SpeechPhaseInterval& found) noexcept
{
    for (std::size_t i = 0; i < timeline.count; ++i)
        if (timeline.phases[i].phase == phase)
        {
            found = timeline.phases[i];
            return true;
        }
    return false;
}

[[nodiscard]] bool default_phase(const SpeechPhaseInterval& entry) noexcept
{
    return entry.interval.start_ns == 0 && entry.interval.end_ns == 0 &&
           entry.phase == SpeechPhase::black && entry.cue == CueKind::none &&
           entry.stimulus_id == kUnsetStimulusId;
}

} // namespace

ContractStatus validate(const SpeechTrial& trial) noexcept
{
    if (const ContractStatus status = validate(trial.record); status != ContractStatus::ok)
        return status;
    if (const ContractStatus status = validate(trial.schedule); status != ContractStatus::ok)
        return status;
    // The one outcome this paradigm produces: it observes nothing, so a trial
    // that was presented in full is the only thing that can have happened, and a
    // record claiming anything else did not come from this machine.
    if (trial.record.outcome != TrialOutcome::success ||
        trial.record.reason != static_cast<std::uint32_t>(SpeechReason::content_elapsed))
        return ContractStatus::outcome_invalid;
    if (trial.record.trial.ordinal != trial.schedule.ordinal ||
        trial.record.trial.stimulus_id != trial.schedule.stimulus_id)
        return ContractStatus::outcome_invalid;

    // Exactly the phases the schedule calls for: two without a cross, three with
    // one. A fourth would be an inter-trial interval, which this machine does not
    // run, and a cross entry under a disabled cross is the zero-length phase the
    // whole enabled/disabled distinction exists to prevent.
    const std::uint8_t expected = trial.schedule.cross_enabled ? 3 : 2;
    if (trial.timeline.count != expected)
        return ContractStatus::outcome_invalid;
    for (std::size_t i = trial.timeline.count; i < trial.timeline.phases.size(); ++i)
        if (!default_phase(trial.timeline.phases[i]))
            return ContractStatus::outcome_invalid;

    const std::array<SpeechPhase, 3> order{SpeechPhase::black, SpeechPhase::cross,
                                           SpeechPhase::content};
    std::size_t idx = 0;
    ExperimentTimeNs cursor = trial.timeline.trial.start_ns;
    for (const SpeechPhase phase : order)
    {
        if (phase == SpeechPhase::cross && !trial.schedule.cross_enabled)
            continue;
        const SpeechPhaseInterval& entry = trial.timeline.phases[idx];
        const DurationNs duration = phase == SpeechPhase::black ? trial.schedule.black_duration_ns
                                    : phase == SpeechPhase::cross
                                        ? trial.schedule.cross_duration_ns
                                        : trial.schedule.content_duration_ns;
        const StimulusId stimulus =
            phase == SpeechPhase::content ? trial.schedule.stimulus_id : kUnsetStimulusId;
        if (entry.phase != phase || entry.cue != speech_cue_of(phase) ||
            entry.stimulus_id != stimulus)
            return ContractStatus::outcome_invalid;
        // Contiguous and half-open: each phase begins exactly where the last
        // ended, so no instant belongs to two of them and none belongs to
        // neither.
        if (entry.interval.start_ns != cursor || entry.interval.duration_ns() != duration ||
            entry.interval.end_ns != cursor + duration)
            return ContractStatus::outcome_invalid;
        cursor = entry.interval.end_ns;
        ++idx;
    }
    if (trial.timeline.trial.end_ns != cursor)
        return ContractStatus::outcome_invalid;
    // No gap: the next trial begins where this one ended, which is what makes the
    // next trial's BLACK the blank period between them.
    if (trial.timeline.next_trial_start_ns != cursor)
        return ContractStatus::outcome_invalid;
    if (trial.record.interval.start_ns != trial.timeline.trial.start_ns ||
        trial.record.interval.end_ns != trial.timeline.trial.end_ns)
        return ContractStatus::outcome_invalid;
    return ContractStatus::ok;
}

ContractStatus SpeechMachine::emit_transition(Run& run, SpeechStepResult& result,
                                              ExperimentTimeNs time_ns, SpeechState to,
                                              SpeechCause cause) const noexcept
{
    // Unreachable while the chain stops at one decided trial: the longest run is
    // three. Checked anyway, because a future state that lengthened the chain
    // must fail loudly rather than write past the array.
    if (result.n_transitions >= kMaxStepTransitions)
        return ContractStatus::numerical_failure;
    SequenceOrdinal sequence = 0;
    if (const ContractStatus status = run.sequence.issue(sequence); status != ContractStatus::ok)
        return status;

    StateTransition transition{};
    transition.time_ns = time_ns;
    transition.sequence = sequence;
    transition.trial = identity_of(run.ordinal, run.schedule.stimulus_id);
    transition.paradigm = paradigm_;
    transition.from_state = static_cast<StateId>(run.state);
    transition.to_state = static_cast<StateId>(to);
    transition.cause = static_cast<std::uint32_t>(cause);
    result.transitions[result.n_transitions++] = transition;
    run.state = to;
    return ContractStatus::ok;
}

ContractStatus SpeechMachine::emit_event(Run& run, SpeechStepResult& result,
                                         ExperimentTimeNs time_ns,
                                         ExperimentEventKind kind) const noexcept
{
    if (result.n_events >= kMaxStepEvents)
        return ContractStatus::numerical_failure;
    SequenceOrdinal sequence = 0;
    if (const ContractStatus status = run.sequence.issue(sequence); status != ContractStatus::ok)
        return status;

    ExperimentEvent event{};
    event.time_ns = time_ns;
    event.sequence = sequence;
    event.trial = identity_of(run.ordinal, run.schedule.stimulus_id);
    event.kind = kind;
    event.paradigm = paradigm_;
    result.events[result.n_events++] = event;
    return ContractStatus::ok;
}

ContractStatus SpeechMachine::emit_marker(Run& run, SpeechStepResult& result,
                                          ExperimentTimeNs time_ns,
                                          SpeechMarker marker) const noexcept
{
    if (result.n_events >= kMaxStepEvents)
        return ContractStatus::numerical_failure;
    SequenceOrdinal sequence = 0;
    if (const ContractStatus status = run.sequence.issue(sequence); status != ContractStatus::ok)
        return status;

    ExperimentEvent event{};
    event.time_ns = time_ns;
    event.sequence = sequence;
    event.trial = identity_of(run.ordinal, run.schedule.stimulus_id);
    event.kind = ExperimentEventKind::paradigm_marker;
    event.paradigm = paradigm_;
    event.code = static_cast<std::uint32_t>(marker);
    result.events[result.n_events++] = event;
    return ContractStatus::ok;
}

ContractStatus SpeechMachine::emit_request(Run& run, SpeechStepResult& result,
                                           SpeechPhase phase) const noexcept
{
    if (result.n_requests >= kMaxStepRequests)
        return ContractStatus::numerical_failure;
    SequenceOrdinal sequence = 0;
    if (const ContractStatus status = run.sequence.issue(sequence); status != ContractStatus::ok)
        return status;

    PresentationRequest request{};
    // Both are the instant the paradigm decided, and neither is the instant
    // anything appeared: the machine has no way to know that and never claims to.
    request.requested_ns = run.active.start_ns;
    request.onset_ns = run.active.start_ns;
    // Presenting a phase after it is over is pointless, so the phase's own end is
    // where the request expires. A presenter that cannot meet it must skip rather
    // than present a cue the trial has already moved past.
    request.valid_until_ns = run.active.end_ns;
    request.duration_ns = run.active.duration_ns();
    request.sequence = sequence;
    request.trial = identity_of(run.ordinal, run.schedule.stimulus_id);
    request.paradigm = paradigm_;
    request.phase = static_cast<PhaseId>(phase);
    request.cue = speech_cue_of(phase);
    request.stimulus_id =
        phase == SpeechPhase::content ? run.schedule.stimulus_id : kUnsetStimulusId;
    result.requests[result.n_requests++] = request;

    PresentationState presented{};
    presented.since_ns = run.active.start_ns;
    presented.trial = request.trial;
    presented.paradigm = paradigm_;
    presented.phase = request.phase;
    presented.cue = request.cue;
    presented.stimulus_id = request.stimulus_id;
    run.presentation = presented;
    return ContractStatus::ok;
}

ContractStatus SpeechMachine::enter_phase(Run& run, SpeechStepResult& result, SpeechPhase phase,
                                          SpeechCause cause) const noexcept
{
    SpeechPhaseInterval entry{};
    if (!interval_of(run.timeline, phase, entry))
        return ContractStatus::outcome_invalid;
    run.active = entry.interval;
    if (const ContractStatus status =
            emit_transition(run, result, entry.interval.start_ns, state_of_phase(phase), cause);
        status != ContractStatus::ok)
        return status;
    if (const ContractStatus status =
            emit_marker(run, result, entry.interval.start_ns, onset_marker_of(phase));
        status != ContractStatus::ok)
        return status;
    return emit_request(run, result, phase);
}

ContractStatus SpeechMachine::begin_trial(Run& run, SpeechStepResult& result,
                                          const SpeechTrialSchedule& schedule,
                                          ExperimentTimeNs time_ns, SpeechCause cause,
                                          bool first) const noexcept
{
    TrialOrdinal ordinal = 0;
    if (const ContractStatus status = run.trials.issue(ordinal); status != ContractStatus::ok)
        return status;
    // The schedule has to name the trial it is being run as. A schedule for some
    // other ordinal would run this trial on another trial's realized durations,
    // and every record would still look internally consistent.
    if (schedule.ordinal != ordinal)
        return ContractStatus::outcome_invalid;

    // build_timeline() runs validate_against(), so the ownership checks -- the
    // sampler version, the stimulus set, the explicit entry -- gate every
    // trial before it can produce a single transition.
    SpeechTimeline timeline{};
    if (const ContractStatus status = build_timeline(config_, schedule, time_ns, timeline);
        status != ContractStatus::ok)
        return status;

    run.ordinal = ordinal;
    run.schedule = schedule;
    run.timeline = timeline;
    if (const ContractStatus status = emit_event(run, result, time_ns,
                                                 first ? ExperimentEventKind::session_start
                                                       : ExperimentEventKind::trial_start);
        status != ContractStatus::ok)
        return status;
    if (first)
        if (const ContractStatus status =
                emit_event(run, result, time_ns, ExperimentEventKind::trial_start);
            status != ContractStatus::ok)
            return status;
    return enter_phase(run, result, SpeechPhase::black, cause);
}

ContractStatus SpeechMachine::complete_trial(Run& run, SpeechStepResult& result,
                                             ExperimentTimeNs time_ns) const noexcept
{
    SpeechTrial trial{};
    trial.record.trial = identity_of(run.ordinal, run.schedule.stimulus_id);
    trial.record.interval = run.timeline.trial;
    trial.record.paradigm = paradigm_;
    trial.record.outcome = TrialOutcome::success;
    trial.record.reason = static_cast<std::uint32_t>(SpeechReason::content_elapsed);
    trial.schedule = run.schedule;
    trial.timeline = run.timeline;

    ++run.completed;
    result.trial = trial;
    result.trial_decided = true;
    return emit_event(run, result, time_ns, ExperimentEventKind::trial_stop);
}

ContractStatus SpeechMachine::advance(Run& run, ExperimentTimeNs time_ns,
                                      const SpeechTrialSchedule* next,
                                      SpeechStepResult& result) const noexcept
{
    for (std::size_t guard = 0; guard <= kMaxStepTransitions; ++guard)
    {
        if (!speech_state_is_phase(run.state) || !run.active.elapsed_at(time_ns))
        {
            result.settled = true;
            return ContractStatus::ok;
        }

        // The phase ended at its own end instant, not at `time_ns`. A caller that
        // stopped polling for a whole trial crosses every boundary at the instant
        // it actually occurred, so the records it gets back are the records a
        // caller polling continuously would have got.
        const SpeechPhase phase = speech_phase_of(run.state);
        const ExperimentTimeNs boundary = run.active.end_ns;
        if (const ContractStatus status =
                emit_marker(run, result, boundary, offset_marker_of(phase));
            status != ContractStatus::ok)
            return status;

        if (phase == SpeechPhase::black)
        {
            // A trial without a cross goes straight to CONTENT at BLACK's end.
            // Nothing is emitted for the phase it does not have.
            const SpeechPhase entered =
                run.schedule.cross_enabled ? SpeechPhase::cross : SpeechPhase::content;
            if (const ContractStatus status =
                    enter_phase(run, result, entered, SpeechCause::black_elapsed);
                status != ContractStatus::ok)
                return status;
            continue;
        }
        if (phase == SpeechPhase::cross)
        {
            if (const ContractStatus status =
                    enter_phase(run, result, SpeechPhase::content, SpeechCause::cross_elapsed);
                status != ContractStatus::ok)
                return status;
            continue;
        }

        // CONTENT ended, so the trial is over at this instant.
        if (const ContractStatus status = complete_trial(run, result, boundary);
            status != ContractStatus::ok)
            return status;

        if (run.completed >= config_.n_trials)
        {
            if (const ContractStatus status =
                    emit_event(run, result, boundary, ExperimentEventKind::session_stop);
                status != ContractStatus::ok)
                return status;
            if (const ContractStatus status = emit_transition(
                    run, result, boundary, SpeechState::complete, SpeechCause::trial_limit_reached);
                status != ContractStatus::ok)
                return status;
            run.active = TimeInterval{};
            PresentationState ended{};
            ended.since_ns = boundary;
            ended.trial = identity_of(run.ordinal, run.schedule.stimulus_id);
            ended.paradigm = paradigm_;
            run.presentation = ended;
            result.settled = true;
            return ContractStatus::ok;
        }

        // There is no position between two trials: CONTENT's end *is* the next
        // BLACK's start. A caller with no schedule for what follows has not left
        // the machine somewhere, it has failed to supply an input, so nothing
        // moves and it can retry.
        if (next == nullptr)
            return ContractStatus::identity_missing;
        if (const ContractStatus status =
                begin_trial(run, result, *next, boundary, SpeechCause::content_elapsed, false);
            status != ContractStatus::ok)
            return status;
        // One trial per step. A caller that stopped polling for several trials
        // drains them one call at a time, supplying one schedule each, rather
        // than having a single call infer an unbounded run of them.
        result.settled = !run.active.elapsed_at(time_ns);
        return ContractStatus::ok;
    }
    return ContractStatus::numerical_failure;
}

SpeechSnapshot SpeechMachine::build_snapshot(const Run& run) const noexcept
{
    SpeechSnapshot snapshot{};
    snapshot.time_ns = run.gate.last_ns();
    snapshot.state = run.state;
    snapshot.phase = speech_phase_of(run.state);
    snapshot.trial = identity_of(run.ordinal, run.schedule.stimulus_id);
    snapshot.schedule = run.schedule;
    snapshot.timeline = run.timeline;
    snapshot.active = run.active;
    snapshot.presentation = run.presentation;
    snapshot.completed = run.completed;
    return snapshot;
}

ContractStatus SpeechMachine::start(ParadigmId paradigm, const SpeechCueConfig& config,
                                    const SpeechTrialSchedule& schedule, ExperimentTimeNs time_ns,
                                    SpeechStepResult& result) noexcept
{
    // A session already occupies this machine. Starting again would silently
    // replace it -- dropping the running trial, clearing the ordinals, and
    // emitting a fresh session_start -- so a new session needs reset() first.
    if (run_.state != SpeechState::idle)
        return ContractStatus::already_running;
    if (paradigm == kUnsetParadigmId)
        return ContractStatus::identity_missing;
    if (const ContractStatus status = validate(config); status != ContractStatus::ok)
        return status;
    // Refused rather than ignored. This machine has no state to occupy during a
    // separate gap, so a session that configures one is not a session it runs;
    // accepting the field and dropping it would make the machine and
    // build_timeline() disagree about when the next trial starts.
    if (config.inter_trial_ns != 0)
        return ContractStatus::outcome_invalid;

    // Built whole and committed only on success, so a refused start leaves a
    // machine that was already running exactly as it was.
    SpeechMachine started{};
    started.paradigm_ = paradigm;
    started.config_ = config;
    Run& run = started.run_;
    if (const ContractStatus status = run.gate.accept(time_ns); status != ContractStatus::ok)
        return status;

    SpeechStepResult local{};
    if (const ContractStatus status =
            started.begin_trial(run, local, schedule, time_ns, SpeechCause::session_started, true);
        status != ContractStatus::ok)
        return status;

    local.snapshot = started.build_snapshot(run);
    local.settled = true;
    *this = started;
    result = local;
    return ContractStatus::ok;
}

void SpeechMachine::reset() noexcept
{
    *this = SpeechMachine{};
}

ContractStatus SpeechMachine::step_impl(ExperimentTimeNs time_ns, const SpeechTrialSchedule* next,
                                        SpeechStepResult& result) noexcept
{
    if (!speech_state_is_phase(run_.state))
        return ContractStatus::not_running;

    Run run = run_;
    if (const ContractStatus status = run.gate.accept(time_ns); status != ContractStatus::ok)
        return status;

    SpeechStepResult local{};
    if (const ContractStatus status = advance(run, time_ns, next, local);
        status != ContractStatus::ok)
        return status;

    local.snapshot = build_snapshot(run);
    run_ = run;
    result = local;
    return ContractStatus::ok;
}

ContractStatus SpeechMachine::step(ExperimentTimeNs time_ns, SpeechStepResult& result) noexcept
{
    return step_impl(time_ns, nullptr, result);
}

ContractStatus SpeechMachine::step(ExperimentTimeNs time_ns, const SpeechTrialSchedule& next,
                                   SpeechStepResult& result) noexcept
{
    return step_impl(time_ns, &next, result);
}

SpeechSnapshot SpeechMachine::snapshot() const noexcept
{
    return build_snapshot(run_);
}
} // namespace neurale::experiments::speech
