// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "binding_helpers.h"
#include "utf8.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <neurale/experiments/speech.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using namespace neurale::experiments;
using neurale::bindings::experiments::prefix;
using namespace neurale::experiments::speech;

// Only the records in use. A caller reading past a count would be reading unset
// entries, which is exactly the padding the native side keeps out of everything
// that means anything.
} // namespace

static void bind_speech_machine(py::module_& speech)
{
    speech.attr("MAX_STEP_TRANSITIONS") = kMaxStepTransitions;
    speech.attr("MAX_STEP_EVENTS") = kMaxStepEvents;
    speech.attr("MAX_STEP_REQUESTS") = kMaxStepRequests;

    py::enum_<SpeechState>(speech, "SpeechState")
        .value("IDLE", SpeechState::idle)
        .value("BLACK", SpeechState::black)
        .value("FIXATION_CROSS", SpeechState::fixation_cross)
        .value("CONTENT", SpeechState::content)
        .value("COMPLETE", SpeechState::complete);
    py::enum_<SpeechCause>(speech, "SpeechCause")
        .value("UNSPECIFIED", SpeechCause::unspecified)
        .value("SESSION_STARTED", SpeechCause::session_started)
        .value("BLACK_ELAPSED", SpeechCause::black_elapsed)
        .value("CROSS_ELAPSED", SpeechCause::cross_elapsed)
        .value("CONTENT_ELAPSED", SpeechCause::content_elapsed)
        .value("TRIAL_LIMIT_REACHED", SpeechCause::trial_limit_reached);
    py::enum_<SpeechMarker>(speech, "SpeechMarker")
        .value("UNSPECIFIED", SpeechMarker::unspecified)
        .value("BLACK_ONSET", SpeechMarker::black_onset)
        .value("BLACK_OFFSET", SpeechMarker::black_offset)
        .value("CROSS_ONSET", SpeechMarker::cross_onset)
        .value("CROSS_OFFSET", SpeechMarker::cross_offset)
        .value("CONTENT_ONSET", SpeechMarker::content_onset)
        .value("CONTENT_OFFSET", SpeechMarker::content_offset);
    py::enum_<SpeechReason>(speech, "SpeechReason")
        .value("UNSPECIFIED", SpeechReason::unspecified)
        .value("CONTENT_ELAPSED", SpeechReason::content_elapsed);

    speech.def("speech_state_declared", &speech_state_declared, py::arg("state"));
    speech.def("speech_state_is_phase", &speech_state_is_phase, py::arg("state"));
    speech.def("speech_phase_of", &speech_phase_of, py::arg("state"));
    speech.def("speech_cue_of", &speech_cue_of, py::arg("phase"));

    py::class_<SpeechTrial>(speech, "SpeechTrial")
        .def(py::init([](const TrialRecord& record, const SpeechTrialSchedule& schedule,
                         const SpeechTimeline& timeline)
                      { return SpeechTrial{record, schedule, timeline}; }),
             py::arg("record") = TrialRecord{}, py::arg("schedule") = SpeechTrialSchedule{},
             py::arg("timeline") = SpeechTimeline{})
        .def_readonly("record", &SpeechTrial::record)
        .def_readonly("schedule", &SpeechTrial::schedule)
        .def_readonly("timeline", &SpeechTrial::timeline);

    // No public constructor: only the machine produces these.
    py::class_<SpeechSnapshot>(speech, "SpeechSnapshot")
        .def_readonly("time_ns", &SpeechSnapshot::time_ns)
        .def_readonly("state", &SpeechSnapshot::state)
        .def_readonly("phase", &SpeechSnapshot::phase)
        .def_readonly("trial", &SpeechSnapshot::trial)
        .def_readonly("schedule", &SpeechSnapshot::schedule)
        .def_readonly("timeline", &SpeechSnapshot::timeline)
        .def_readonly("active", &SpeechSnapshot::active)
        .def_readonly("presentation", &SpeechSnapshot::presentation)
        .def_readonly("completed", &SpeechSnapshot::completed);

    py::class_<SpeechStepResult>(speech, "SpeechStepResult")
        .def_readonly("snapshot", &SpeechStepResult::snapshot)
        .def_readonly("settled", &SpeechStepResult::settled)
        .def_readonly("n_transitions", &SpeechStepResult::n_transitions)
        .def_readonly("n_events", &SpeechStepResult::n_events)
        .def_readonly("n_requests", &SpeechStepResult::n_requests)
        .def_readonly("trial_decided", &SpeechStepResult::trial_decided)
        .def_readonly("trial", &SpeechStepResult::trial)
        .def_property_readonly("transitions", [](const SpeechStepResult& result)
                               { return prefix(result.transitions, result.n_transitions); })
        .def_property_readonly("events", [](const SpeechStepResult& result)
                               { return prefix(result.events, result.n_events); })
        .def_property_readonly("requests", [](const SpeechStepResult& result)
                               { return prefix(result.requests, result.n_requests); });

    speech.def(
        "validate", [](const SpeechTrial& trial) { return validate(trial); }, py::arg("trial"));

    py::class_<SpeechMachine>(speech, "SpeechMachine")
        .def(py::init<>())
        .def(
            "start",
            [](SpeechMachine& machine, ParadigmId paradigm, const SpeechCueConfig& config,
               const SpeechTrialSchedule& schedule, ExperimentTimeNs time_ns)
            {
                SpeechStepResult result{};
                const ContractStatus status =
                    machine.start(paradigm, config, schedule, time_ns, result);
                return std::pair{status, result};
            },
            py::arg("paradigm"), py::arg("config"), py::arg("schedule"), py::arg("time_ns"))
        .def("reset", &SpeechMachine::reset)
        .def(
            "step",
            [](SpeechMachine& machine, ExperimentTimeNs time_ns)
            {
                SpeechStepResult result{};
                const ContractStatus status = machine.step(time_ns, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"))
        .def(
            "step",
            [](SpeechMachine& machine, ExperimentTimeNs time_ns, const SpeechTrialSchedule& next)
            {
                SpeechStepResult result{};
                const ContractStatus status = machine.step(time_ns, next, result);
                return std::pair{status, result};
            },
            py::arg("time_ns"), py::arg("next"))
        .def("snapshot", &SpeechMachine::snapshot)
        .def("configuration", &SpeechMachine::configuration)
        .def_property_readonly("paradigm", &SpeechMachine::paradigm)
        .def_property_readonly("state", &SpeechMachine::state)
        .def_property_readonly("complete", &SpeechMachine::complete);
}

namespace
{

using namespace neurale::experiments;
using neurale::bindings::experiments::bind_validate;
using neurale::bindings::experiments::prefix;
using namespace neurale::experiments::speech;

template <typename Element, std::size_t Capacity>
std::array<Element, Capacity> to_fixed(const std::vector<Element>& items, const char* what)
{
    if (items.size() > Capacity)
        throw py::value_error(std::string(what) + " holds more than " + std::to_string(Capacity) +
                              " entries");
    std::array<Element, Capacity> fixed{};
    for (std::size_t i = 0; i < items.size(); ++i)
        fixed[i] = items[i];
    return fixed;
}

// The text of a catalog entry is a fixed byte array plus a length. Python sees
// the bytes in use and never the padding, which is what makes two entries with
// the same prompt compare equal there as well as fingerprint equal here.
//
// The contract carries canonical UTF-8, so a ``str`` is encoded as UTF-8 by
// pybind11 and a raw ``bytes`` value is admitted only when it is already a
// well-formed UTF-8 sequence. The native validator and this constructor share
// one decoder (utf8.h), so a stimulus that the contract rejects can never be
// built from Python in the first place.
SpeechStimulus make_stimulus(StimulusId id, SpeechContentKind content, const std::string& text,
                             std::uint32_t label, std::uint64_t metadata)
{
    if (text.size() > kMaxSpeechTextBytes)
        throw py::value_error("stimulus text is longer than " +
                              std::to_string(kMaxSpeechTextBytes) + " bytes");
    if (!neurale::experiments::detail::is_well_formed_utf8(
            reinterpret_cast<const std::uint8_t*>(text.data()), text.size()))
        throw py::value_error("stimulus text must be canonical UTF-8");
    SpeechStimulus stimulus{};
    stimulus.id = id;
    stimulus.label = label;
    stimulus.metadata = metadata;
    stimulus.content = content;
    stimulus.text_length = static_cast<std::uint8_t>(text.size());
    for (std::size_t i = 0; i < text.size(); ++i)
        stimulus.text[i] = text[i];
    return stimulus;
}

} // namespace

void bind_experiments_speech_module(py::module_& experiments)
{
    auto speech = experiments.def_submodule(
        "speech", "Speech cue configuration, immutable stimulus catalog, deterministic per-trial "
                  "timing schedule, and the intended phase timeline.");

    speech.attr("MAX_SPEECH_STIMULI") = kMaxSpeechStimuli;
    speech.attr("MAX_SPEECH_EXPLICIT_TRIALS") = kMaxSpeechExplicitTrials;
    speech.attr("MAX_SPEECH_TEXT_BYTES") = kMaxSpeechTextBytes;
    speech.attr("MAX_SPEECH_PHASES") = kMaxSpeechPhases;
    speech.attr("TRIAL_TIMING_STREAM") = kTrialTimingStream;
    speech.attr("STIMULUS_ORDER_STREAM") = kStimulusOrderStream;
    speech.attr("BLACK_DRAW") = kBlackDraw;
    speech.attr("CROSS_DRAW") = kCrossDraw;
    speech.attr("CONTENT_DRAW") = kContentDraw;
    speech.attr("UNSET_SPEECH_LABEL") = kUnsetSpeechLabel;

    py::enum_<SpeechContentKind>(speech, "SpeechContentKind")
        .value("UNSPECIFIED", SpeechContentKind::unspecified)
        .value("TEXT", SpeechContentKind::text);
    py::enum_<SpeechScheduleKind>(speech, "SpeechScheduleKind")
        .value("UNSPECIFIED", SpeechScheduleKind::unspecified)
        .value("SEEDED", SpeechScheduleKind::seeded)
        .value("EXPLICIT_SEQUENCE", SpeechScheduleKind::explicit_sequence);
    py::enum_<StimulusOrderPolicy>(speech, "StimulusOrderPolicy")
        .value("UNSPECIFIED", StimulusOrderPolicy::unspecified)
        .value("SEQUENTIAL", StimulusOrderPolicy::sequential)
        .value("RANDOM_WITH_REPLACEMENT", StimulusOrderPolicy::random_with_replacement)
        .value("SHUFFLED_BLOCKS", StimulusOrderPolicy::shuffled_blocks);
    py::enum_<SpeechPhase>(speech, "SpeechPhase")
        .value("BLACK", SpeechPhase::black)
        .value("CROSS", SpeechPhase::cross)
        .value("CONTENT", SpeechPhase::content)
        .value("INTER_TRIAL", SpeechPhase::inter_trial);

    speech.def("speech_content_kind_declared", &speech_content_kind_declared, py::arg("value"));
    speech.def("speech_schedule_kind_declared", &speech_schedule_kind_declared, py::arg("value"));
    speech.def("stimulus_order_policy_declared", &stimulus_order_policy_declared, py::arg("value"));
    speech.def("speech_phase_declared", &speech_phase_declared, py::arg("value"));

    py::class_<SpeechStimulus>(speech, "SpeechStimulus")
        .def(py::init(&make_stimulus), py::arg("id") = kUnsetStimulusId,
             py::arg("content") = SpeechContentKind::unspecified, py::arg("text") = std::string{},
             py::arg("label") = kUnsetSpeechLabel, py::arg("metadata") = 0)
        .def_readonly("id", &SpeechStimulus::id)
        .def_readonly("label", &SpeechStimulus::label)
        .def_readonly("metadata", &SpeechStimulus::metadata)
        .def_readonly("content", &SpeechStimulus::content)
        .def_readonly("text_length", &SpeechStimulus::text_length)
        .def_property_readonly("text", [](const SpeechStimulus& stimulus)
                               { return py::bytes(stimulus.text.data(), stimulus.text_length); });

    py::class_<SpeechCatalog>(speech, "SpeechCatalog")
        .def(py::init(
                 [](const std::vector<SpeechStimulus>& entries)
                 {
                     SpeechCatalog catalog{};
                     catalog.count = static_cast<std::uint16_t>(entries.size());
                     catalog.entries =
                         to_fixed<SpeechStimulus, kMaxSpeechStimuli>(entries, "catalog");
                     return catalog;
                 }),
             py::arg("entries") = std::vector<SpeechStimulus>{})
        .def_readonly("count", &SpeechCatalog::count)
        .def_property_readonly("entries", [](const SpeechCatalog& catalog)
                               { return prefix(catalog.entries, catalog.count); });

    py::class_<SpeechTrialSchedule>(speech, "SpeechTrialSchedule")
        .def(py::init(
                 [](TrialOrdinal ordinal, StimulusId stimulus_id, bool cross_enabled,
                    DurationNs black_duration_ns, DurationNs cross_duration_ns,
                    DurationNs content_duration_ns, SamplerVersion sampler_version)
                 {
                     return SpeechTrialSchedule{
                         ordinal,     black_duration_ns, cross_duration_ns, content_duration_ns,
                         stimulus_id, sampler_version,   cross_enabled};
                 }),
             py::arg("ordinal") = 0, py::arg("stimulus_id") = kUnsetStimulusId,
             py::arg("cross_enabled") = false, py::arg("black_duration_ns") = 0,
             py::arg("cross_duration_ns") = 0, py::arg("content_duration_ns") = 0,
             // The native default is zero -- unset -- because that is what an
             // unused explicit_schedule slot has to be for the configuration
             // fingerprint to stay canonical. Nothing constructed here is
             // padding, though: a caller building a realized schedule by hand
             // wants the sampler the configuration defaults to, and would have
             // to write it out on every entry otherwise.
             py::arg("sampler_version") = kCurrentSamplerVersion)
        .def_readonly("ordinal", &SpeechTrialSchedule::ordinal)
        .def_readonly("black_duration_ns", &SpeechTrialSchedule::black_duration_ns)
        .def_readonly("cross_duration_ns", &SpeechTrialSchedule::cross_duration_ns)
        .def_readonly("content_duration_ns", &SpeechTrialSchedule::content_duration_ns)
        .def_readonly("stimulus_id", &SpeechTrialSchedule::stimulus_id)
        .def_readonly("sampler_version", &SpeechTrialSchedule::sampler_version)
        .def_readonly("cross_enabled", &SpeechTrialSchedule::cross_enabled);

    // No public constructor: only build_timeline() produces these.
    py::class_<SpeechPhaseInterval>(speech, "SpeechPhaseInterval")
        .def_readonly("interval", &SpeechPhaseInterval::interval)
        .def_readonly("phase", &SpeechPhaseInterval::phase)
        .def_readonly("cue", &SpeechPhaseInterval::cue)
        .def_readonly("stimulus_id", &SpeechPhaseInterval::stimulus_id);

    py::class_<SpeechTimeline>(speech, "SpeechTimeline")
        .def_readonly("count", &SpeechTimeline::count)
        .def_readonly("trial", &SpeechTimeline::trial)
        .def_readonly("next_trial_start_ns", &SpeechTimeline::next_trial_start_ns)
        // Only the phases in use. A disabled cross and a zero gap are absent
        // here exactly as they are absent from the native record, so a caller
        // cannot read a phase the trial does not have.
        .def_property_readonly("phases", [](const SpeechTimeline& timeline)
                               { return prefix(timeline.phases, timeline.count); });

    py::class_<SpeechTrialDraws>(speech, "SpeechTrialDraws")
        .def_readonly("count", &SpeechTrialDraws::count)
        .def_property_readonly("draws", [](const SpeechTrialDraws& draws)
                               { return prefix(draws.draws, draws.count); });

    const SpeechCueConfig defaults{};
    py::class_<SpeechCueConfig>(speech, "SpeechCueConfig")
        .def(py::init(
                 [](DurationNs black_bound_ns, DurationNs cross_bound_ns,
                    DurationNs content_bound_ns, bool cross_enabled, DurationNs inter_trial_ns,
                    SpeechScheduleKind schedule, StimulusOrderPolicy stimulus_order,
                    ScheduleSeed seed, SamplerVersion sampler_version, TrialOrdinal n_trials,
                    const std::vector<StimulusId>& stimuli,
                    const std::vector<SpeechTrialSchedule>& explicit_schedule)
                 {
                     SpeechCueConfig config{};
                     config.black_bound_ns = black_bound_ns;
                     config.cross_bound_ns = cross_bound_ns;
                     config.content_bound_ns = content_bound_ns;
                     config.inter_trial_ns = inter_trial_ns;
                     config.seed = seed;
                     config.n_trials = n_trials;
                     config.schedule = schedule;
                     config.stimulus_order = stimulus_order;
                     config.sampler_version = sampler_version;
                     config.cross_enabled = cross_enabled;
                     config.n_stimuli = static_cast<std::uint16_t>(stimuli.size());
                     config.stimuli =
                         to_fixed<StimulusId, kMaxSpeechStimuli>(stimuli, "stimulus set");
                     config.n_explicit = static_cast<std::uint16_t>(explicit_schedule.size());
                     config.explicit_schedule =
                         to_fixed<SpeechTrialSchedule, kMaxSpeechExplicitTrials>(
                             explicit_schedule, "explicit schedule");
                     return config;
                 }),
             py::arg("black_bound_ns") = 0, py::arg("cross_bound_ns") = 0,
             py::arg("content_bound_ns") = 0, py::arg("cross_enabled") = false,
             py::arg("inter_trial_ns") = 0, py::arg("schedule") = SpeechScheduleKind::unspecified,
             py::arg("stimulus_order") = StimulusOrderPolicy::unspecified, py::arg("seed") = 0,
             py::arg("sampler_version") = defaults.sampler_version, py::arg("n_trials") = 0,
             py::arg("stimuli") = std::vector<StimulusId>{},
             py::arg("explicit_schedule") = std::vector<SpeechTrialSchedule>{})
        .def_readonly("black_bound_ns", &SpeechCueConfig::black_bound_ns)
        .def_readonly("cross_bound_ns", &SpeechCueConfig::cross_bound_ns)
        .def_readonly("content_bound_ns", &SpeechCueConfig::content_bound_ns)
        .def_readonly("inter_trial_ns", &SpeechCueConfig::inter_trial_ns)
        .def_readonly("seed", &SpeechCueConfig::seed)
        .def_readonly("n_trials", &SpeechCueConfig::n_trials)
        .def_readonly("schedule", &SpeechCueConfig::schedule)
        .def_readonly("stimulus_order", &SpeechCueConfig::stimulus_order)
        .def_readonly("sampler_version", &SpeechCueConfig::sampler_version)
        .def_readonly("cross_enabled", &SpeechCueConfig::cross_enabled)
        .def_readonly("n_stimuli", &SpeechCueConfig::n_stimuli)
        .def_property_readonly("stimuli", [](const SpeechCueConfig& config)
                               { return prefix(config.stimuli, config.n_stimuli); })
        .def_readonly("n_explicit", &SpeechCueConfig::n_explicit)
        .def_property_readonly("explicit_schedule", [](const SpeechCueConfig& config)
                               { return prefix(config.explicit_schedule, config.n_explicit); });

    bind_validate<SpeechStimulus>(speech);
    bind_validate<SpeechCatalog>(speech);
    bind_validate<SpeechTrialSchedule>(speech);
    bind_validate<SpeechCueConfig>(speech);

    speech.def(
        "validate_against", [](const SpeechTrialSchedule& schedule, const SpeechCueConfig& config)
        { return validate_against(schedule, config); }, py::arg("schedule"), py::arg("config"));
    speech.def(
        "validate_against", [](const SpeechCueConfig& config, const SpeechCatalog& catalog)
        { return validate_against(config, catalog); }, py::arg("config"), py::arg("catalog"));

    // Each of these reports a status rather than throwing, so the binding hands
    // Python both halves rather than inventing a sentinel result.
    speech.def(
        "find_stimulus",
        [](const SpeechCatalog& catalog, StimulusId id)
        {
            SpeechStimulus stimulus{};
            const ContractStatus status = find_stimulus(catalog, id, stimulus);
            return std::pair{status, stimulus};
        },
        py::arg("catalog"), py::arg("id"));
    speech.def(
        "select_stimulus",
        [](const SpeechCueConfig& config, TrialOrdinal ordinal)
        {
            StimulusId stimulus_id = kUnsetStimulusId;
            const ContractStatus status = select_stimulus(config, ordinal, stimulus_id);
            return std::pair{status, stimulus_id};
        },
        py::arg("config"), py::arg("ordinal"));
    speech.def(
        "prepare_trial",
        [](const SpeechCueConfig& config, TrialOrdinal ordinal)
        {
            SpeechTrialSchedule schedule{};
            const ContractStatus status = prepare_trial(config, ordinal, schedule);
            return std::pair{status, schedule};
        },
        py::arg("config"), py::arg("ordinal"));
    speech.def(
        "trial_duration",
        [](const SpeechTrialSchedule& schedule)
        {
            DurationNs duration_ns = 0;
            const ContractStatus status = trial_duration(schedule, duration_ns);
            return std::pair{status, duration_ns};
        },
        py::arg("schedule"));
    speech.def(
        "build_timeline",
        [](const SpeechCueConfig& config, const SpeechTrialSchedule& schedule,
           ExperimentTimeNs start_ns)
        {
            SpeechTimeline timeline{};
            const ContractStatus status = build_timeline(config, schedule, start_ns, timeline);
            return std::pair{status, timeline};
        },
        py::arg("config"), py::arg("schedule"), py::arg("start_ns"));
    speech.def(
        "make_schedule_draws",
        [](const SpeechTrialSchedule& schedule, const TrialIdentity& trial,
           ExperimentTimeNs time_ns)
        {
            SpeechTrialDraws draws{};
            const ContractStatus status = make_schedule_draws(schedule, trial, time_ns, draws);
            return std::pair{status, draws};
        },
        py::arg("schedule"), py::arg("trial"), py::arg("time_ns"));

    speech.def("configuration_fingerprint", &configuration_fingerprint, py::arg("config"));
    speech.def("catalog_fingerprint", &catalog_fingerprint, py::arg("catalog"));
    speech.def("schedule_identity", &schedule_identity, py::arg("config"), py::arg("catalog"));
    speech.def(
        "replay_authority", [](const SpeechCueConfig& config) { return replay_authority(config); },
        py::arg("config"));

    // The state machine lives in the same namespace as the configuration it
    // runs: a Speech state is meaningless against any other paradigm's.
    bind_speech_machine(speech);
}
