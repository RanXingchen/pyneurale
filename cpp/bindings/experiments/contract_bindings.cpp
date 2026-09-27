/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "binding_helpers.h"

#include <neurale/experiments/abnormal.h>
#include <neurale/experiments/command.h>
#include <neurale/experiments/contract.h>
#include <neurale/experiments/events.h>
#include <neurale/experiments/identity.h>
#include <neurale/experiments/presentation.h>
#include <neurale/experiments/replay.h>
#include <neurale/experiments/schedule.h>
#include <neurale/experiments/time.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace
{

using namespace neurale::experiments;
using neurale::bindings::experiments::bind_validate;

// Every value record below is bound with a complete constructor and readonly
// fields. A validated record that can still be edited afterwards carries no
// guarantee at all: the sequence "construct, validate, persist, mutate" would
// leave a fingerprinted ScheduleIdentity holding a seed nobody ever checked.
// Construction is therefore the only point at which a value is decided, and
// validate() answers about a value that can no longer change underneath it.
//
// The stateful helpers -- MonotonicTimeGate, the ordinal counters, and
// FingerprintAccumulator -- stay mutable on purpose. They are not records;
// advancing is what they are for.

// The native samplers report a range failure instead of throwing, so the
// binding hands Python both halves rather than inventing a sentinel value.
using SampleResult = std::pair<ContractStatus, std::uint64_t>;

SampleResult sample_inclusive_result(const DrawKey& key, std::uint64_t low, std::uint64_t high)
{
    std::uint64_t value = 0;
    const ContractStatus status = sample_inclusive(key, low, high, value);
    return {status, value};
}

SampleResult sample_exclusive_result(const DrawKey& key, std::uint64_t low, std::uint64_t high)
{
    std::uint64_t value = 0;
    const ContractStatus status = sample_exclusive(key, low, high, value);
    return {status, value};
}

SampleResult sample_index_result(const DrawKey& key, std::uint64_t count)
{
    std::uint64_t value = 0;
    const ContractStatus status = sample_index(key, count, value);
    return {status, value};
}

std::pair<ContractStatus, TimeInterval> interval_from_duration_result(ExperimentTimeNs start_ns,
                                                                      DurationNs duration_ns)
{
    TimeInterval interval{};
    const ContractStatus status = interval_from_duration(start_ns, duration_ns, interval);
    return {status, interval};
}

// The counters report exhaustion instead of reissuing an ordinal, so Python
// receives the status alongside the value rather than a value it cannot trust.
template <typename Counter> auto issue_result(Counter& counter)
{
    typename std::remove_reference_t<decltype(counter.peek())> value{};
    const ContractStatus status = counter.issue(value);
    return std::pair{status, value};
}

std::uint64_t fingerprint_of_bytes(const py::bytes& data)
{
    // Absorb the exact bytes Python handed over. std::string here is the
    // buffer, not a payload the contract carries: no native record holds text.
    const std::string buffer = data;
    FingerprintAccumulator accumulator;
    accumulator.absorb_bytes(std::as_bytes(std::span<const char>(buffer.data(), buffer.size())));
    return accumulator.value();
}

void check_command_index(std::size_t idx)
{
    if (idx >= kMaxCommandDim)
        throw py::index_error("command axis index out of range");
}

// A fixed-size record cannot take a variable-length sequence, so the sequence is
// padded out to the record's capacity. Padding is only ever added at or beyond
// `dimension`, where the contract already requires the slot to be unset: a slot
// the caller declared to be in use is never invented.
//
// That distinction is the whole point. A missing velocity component and a
// component that is deliberately zero are the same bytes once written, so
// filling in a declared axis would turn "I did not say" into "I said zero" with
// nothing downstream able to tell the difference. An over-long sequence is
// refused rather than truncated for the mirror-image reason.
template <typename Element, std::size_t Capacity>
std::array<Element, Capacity> to_fixed(const std::vector<Element>& items, std::size_t dim,
                                       const char* what)
{
    if (items.size() > Capacity)
        throw py::value_error(std::string(what) + " exceeds the fixed command capacity");
    if (items.size() < dim)
        throw py::value_error(std::string(what) + " supplies " + std::to_string(items.size()) +
                              " of the " + std::to_string(dim) +
                              " declared by dimension; a slot left out is not a zero");
    std::array<Element, Capacity> fixed{};
    for (std::size_t i = 0; i < items.size(); ++i)
        fixed[i] = items[i];
    return fixed;
}

} // namespace

void bind_experiments_contract(py::module_& experiments)
{
    py::enum_<ContractStatus>(experiments, "ContractStatus")
        .value("OK", ContractStatus::ok)
        .value("TIME_REGRESSED", ContractStatus::time_regressed)
        .value("INTERVAL_INVERTED", ContractStatus::interval_inverted)
        .value("DURATION_OVERFLOW", ContractStatus::duration_overflow)
        .value("DIMENSION_INVALID", ContractStatus::dimension_invalid)
        .value("VALUE_NOT_FINITE", ContractStatus::value_not_finite)
        .value("EXPIRY_BEFORE_GENERATION", ContractStatus::expiry_before_generation)
        .value("RANGE_EMPTY", ContractStatus::range_empty)
        .value("SAMPLING_EXHAUSTED", ContractStatus::sampling_exhausted)
        .value("ORDINAL_EXHAUSTED", ContractStatus::ordinal_exhausted)
        .value("IDENTITY_MISSING", ContractStatus::identity_missing)
        .value("ENUM_UNDECLARED", ContractStatus::enum_undeclared)
        .value("PRESENTATION_INVALID", ContractStatus::presentation_invalid)
        .value("OUTCOME_INVALID", ContractStatus::outcome_invalid)
        .value("ASSISTANCE_OUT_OF_RANGE", ContractStatus::assistance_out_of_range)
        .value("DOMAIN_MASK_INVALID", ContractStatus::domain_mask_invalid)
        .value("VERSION_UNSUPPORTED", ContractStatus::version_unsupported)
        .value("NUMERICAL_FAILURE", ContractStatus::numerical_failure)
        .value("PARAMETER_OUT_OF_RANGE", ContractStatus::parameter_out_of_range)
        .value("TARGET_SET_INVALID", ContractStatus::target_set_invalid)
        .value("NOT_RUNNING", ContractStatus::not_running)
        .value("ALREADY_RUNNING", ContractStatus::already_running);

    experiments.attr("NO_EXPIRY_NS") = kNoExpiryNs;
    experiments.attr("UNSET_TRIAL_KEY") = kUnsetTrialKey;
    experiments.attr("UNSET_TARGET_ID") = kUnsetTargetId;
    experiments.attr("UNSET_STIMULUS_ID") = kUnsetStimulusId;
    experiments.attr("UNSET_PARADIGM_ID") = kUnsetParadigmId;
    experiments.attr("UNSET_COMMAND_SPACE_ID") = kUnsetCommandSpaceId;
    experiments.attr("MAX_COMMAND_DIMENSION") = kMaxCommandDim;
    experiments.attr("SAMPLER_VERSION_1") = kSamplerVersion1;
    experiments.attr("CURRENT_SAMPLER_VERSION") = kCurrentSamplerVersion;
    experiments.attr("MAX_REJECTION_DRAWS") = kMaxRejectionDraws;
    experiments.attr("DRAW_SLOT_STRIDE") = kDrawSlotStride;

    experiments.def("contract_status_message", &contract_status_message, py::arg("status"));

    py::class_<TimeInterval>(experiments, "TimeInterval")
        .def(py::init([](ExperimentTimeNs start_ns, ExperimentTimeNs end_ns)
                      { return TimeInterval{start_ns, end_ns}; }),
             py::arg("start_ns") = 0, py::arg("end_ns") = 0)
        .def_readonly("start_ns", &TimeInterval::start_ns)
        .def_readonly("end_ns", &TimeInterval::end_ns)
        .def_property_readonly("well_formed", &TimeInterval::well_formed)
        .def_property_readonly("empty", &TimeInterval::empty)
        .def_property_readonly("duration_ns", &TimeInterval::duration_ns)
        .def("contains", &TimeInterval::contains, py::arg("time_ns"))
        .def("elapsed_at", &TimeInterval::elapsed_at, py::arg("time_ns"));

    experiments.def("time_fits", &time_fits, py::arg("start_ns"), py::arg("duration_ns"));
    experiments.def("interval_from_duration", &interval_from_duration_result, py::arg("start_ns"),
                    py::arg("duration_ns"));

    py::class_<MonotonicTimeGate>(experiments, "MonotonicTimeGate")
        .def(py::init<>())
        .def(py::init<ExperimentTimeNs>(), py::arg("origin_ns"))
        .def("accept", &MonotonicTimeGate::accept, py::arg("time_ns"))
        .def_property_readonly("last_ns", &MonotonicTimeGate::last_ns)
        .def_property_readonly("started", &MonotonicTimeGate::started)
        .def("reset", py::overload_cast<>(&MonotonicTimeGate::reset))
        .def("reset", py::overload_cast<ExperimentTimeNs>(&MonotonicTimeGate::reset),
             py::arg("origin_ns"));

    py::class_<TrialIdentity>(experiments, "TrialIdentity")
        .def(py::init([](TrialOrdinal ordinal, TrialKey key, BlockOrdinal block, TargetId target_id,
                         StimulusId stimulus_id)
                      { return TrialIdentity{ordinal, key, block, target_id, stimulus_id}; }),
             py::arg("ordinal") = 0, py::arg("key") = kUnsetTrialKey, py::arg("block") = 0,
             py::arg("target_id") = kUnsetTargetId, py::arg("stimulus_id") = kUnsetStimulusId)
        .def_readonly("ordinal", &TrialIdentity::ordinal)
        .def_readonly("key", &TrialIdentity::key)
        .def_readonly("block", &TrialIdentity::block)
        .def_readonly("target_id", &TrialIdentity::target_id)
        .def_readonly("stimulus_id", &TrialIdentity::stimulus_id);

    experiments.def("same_trial", &same_trial, py::arg("left"), py::arg("right"));

    py::enum_<TrialOutcome>(experiments, "TrialOutcome")
        .value("PENDING", TrialOutcome::pending)
        .value("SUCCESS", TrialOutcome::success)
        .value("FAILURE", TrialOutcome::failure)
        .value("TIMEOUT", TrialOutcome::timeout)
        .value("ABORTED", TrialOutcome::aborted);

    experiments.def("trial_outcome_declared", &trial_outcome_declared, py::arg("outcome"));
    experiments.def("trial_ended", &trial_ended, py::arg("outcome"));

    py::class_<TrialCounter>(experiments, "TrialCounter")
        .def(py::init<TrialOrdinal>(), py::arg("origin") = 0)
        .def_property_readonly("peek", &TrialCounter::peek)
        .def_property_readonly("issued", &TrialCounter::issued)
        .def_property_readonly("origin", &TrialCounter::origin)
        .def_property_readonly("exhausted", &TrialCounter::exhausted)
        .def("issue", &issue_result<TrialCounter>)
        .def("reset", py::overload_cast<>(&TrialCounter::reset))
        .def("reset", py::overload_cast<TrialOrdinal>(&TrialCounter::reset), py::arg("origin"));

    py::class_<SequenceCounter>(experiments, "SequenceCounter")
        .def(py::init<SequenceOrdinal>(), py::arg("origin") = 0)
        .def_property_readonly("peek", &SequenceCounter::peek)
        .def_property_readonly("issued", &SequenceCounter::issued)
        .def_property_readonly("origin", &SequenceCounter::origin)
        .def_property_readonly("exhausted", &SequenceCounter::exhausted)
        .def("issue", &issue_result<SequenceCounter>)
        .def("reset", py::overload_cast<>(&SequenceCounter::reset))
        .def("reset", py::overload_cast<SequenceOrdinal>(&SequenceCounter::reset),
             py::arg("origin"));

    py::class_<DrawKey>(experiments, "DrawKey")
        .def(py::init([](ScheduleSeed seed, DrawStream stream, TrialOrdinal trial_idx,
                         std::uint32_t draw) { return DrawKey{seed, stream, trial_idx, draw}; }),
             py::arg("seed") = 0, py::arg("stream") = 0, py::arg("trial_idx") = 0,
             py::arg("draw") = 0)
        .def_readonly("seed", &DrawKey::seed)
        .def_readonly("stream", &DrawKey::stream)
        .def_readonly("trial_idx", &DrawKey::trial_idx)
        .def_readonly("draw", &DrawKey::draw);

    experiments.def("sampler_mix64", &sampler_mix64, py::arg("value"));
    experiments.def("sample_bits", &sample_bits, py::arg("key"));
    experiments.def("sample_inclusive", &sample_inclusive_result, py::arg("key"), py::arg("low"),
                    py::arg("high"));
    experiments.def("sample_exclusive", &sample_exclusive_result, py::arg("key"), py::arg("low"),
                    py::arg("high"));
    experiments.def("sample_index", &sample_index_result, py::arg("key"), py::arg("count"));

    py::class_<FingerprintAccumulator>(experiments, "FingerprintAccumulator")
        .def(py::init<>())
        .def(py::init<std::uint64_t>(), py::arg("origin"))
        .def("absorb", &FingerprintAccumulator::absorb, py::arg("value"))
        .def_property_readonly("value", &FingerprintAccumulator::value);

    experiments.def("fingerprint_of_bytes", &fingerprint_of_bytes, py::arg("data"));

    py::class_<ScheduleIdentity>(experiments, "ScheduleIdentity")
        .def(py::init(
                 [](ScheduleSeed seed, SamplerVersion sampler_version,
                    std::uint64_t configuration_fingerprint, std::uint64_t catalog_fingerprint)
                 {
                     return ScheduleIdentity{seed, sampler_version, configuration_fingerprint,
                                             catalog_fingerprint};
                 }),
             py::arg("seed") = 0, py::arg("sampler_version") = kCurrentSamplerVersion,
             py::arg("configuration_fingerprint") = 0, py::arg("catalog_fingerprint") = 0)
        .def_readonly("seed", &ScheduleIdentity::seed)
        .def_readonly("sampler_version", &ScheduleIdentity::sampler_version)
        .def_readonly("configuration_fingerprint", &ScheduleIdentity::configuration_fingerprint)
        .def_readonly("catalog_fingerprint", &ScheduleIdentity::catalog_fingerprint);

    experiments.def("sampler_version_supported", &sampler_version_supported, py::arg("version"));
    experiments.def("schedule_fingerprint", &schedule_fingerprint, py::arg("identity"));

    py::enum_<ReplayAuthority>(experiments, "ReplayAuthority")
        .value("REGENERATE", ReplayAuthority::regenerate)
        .value("RECORDED_SCHEDULE", ReplayAuthority::recorded_schedule);

    experiments.def("replay_authority", &replay_authority, py::arg("version"));

    py::class_<ScheduleDraw>(experiments, "ScheduleDraw")
        .def(py::init(
                 [](ExperimentTimeNs time_ns, const TrialIdentity& trial, DrawStream stream,
                    std::uint32_t draw, std::uint64_t value, SamplerVersion sampler_version)
                 { return ScheduleDraw{time_ns, trial, stream, draw, value, sampler_version}; }),
             py::arg("time_ns") = 0, py::arg("trial") = TrialIdentity{}, py::arg("stream") = 0,
             py::arg("draw") = 0, py::arg("value") = 0,
             py::arg("sampler_version") = kCurrentSamplerVersion)
        .def_readonly("time_ns", &ScheduleDraw::time_ns)
        .def_readonly("trial", &ScheduleDraw::trial)
        .def_readonly("stream", &ScheduleDraw::stream)
        .def_readonly("draw", &ScheduleDraw::draw)
        .def_readonly("value", &ScheduleDraw::value)
        .def_readonly("sampler_version", &ScheduleDraw::sampler_version);

    py::enum_<ExperimentEventKind>(experiments, "ExperimentEventKind")
        .value("UNSPECIFIED", ExperimentEventKind::unspecified)
        .value("SESSION_START", ExperimentEventKind::session_start)
        .value("SESSION_STOP", ExperimentEventKind::session_stop)
        .value("BLOCK_START", ExperimentEventKind::block_start)
        .value("BLOCK_STOP", ExperimentEventKind::block_stop)
        .value("TRIAL_START", ExperimentEventKind::trial_start)
        .value("TRIAL_STOP", ExperimentEventKind::trial_stop)
        .value("STATE_TRANSITION", ExperimentEventKind::state_transition)
        .value("PRESENTATION_REQUEST", ExperimentEventKind::presentation_request)
        .value("PRESENTATION_STATE", ExperimentEventKind::presentation_state)
        .value("SELECTION", ExperimentEventKind::selection)
        .value("COMMAND_REQUEST", ExperimentEventKind::command_request)
        .value("COMMAND_OUTCOME", ExperimentEventKind::command_outcome)
        .value("SCHEDULE_DRAW", ExperimentEventKind::schedule_draw)
        .value("TRIAL_OUTCOME", ExperimentEventKind::trial_outcome)
        .value("PARADIGM_MARKER", ExperimentEventKind::paradigm_marker);

    experiments.def("experiment_event_kind_declared", &experiment_event_kind_declared,
                    py::arg("kind"));

    py::class_<ExperimentEvent>(experiments, "ExperimentEvent")
        .def(
            py::init(
                [](ExperimentTimeNs time_ns, SequenceOrdinal sequence, const TrialIdentity& trial,
                   ExperimentEventKind kind, ParadigmId paradigm, std::uint32_t code,
                   std::int64_t value)
                { return ExperimentEvent{time_ns, sequence, trial, kind, paradigm, code, value}; }),
            py::arg("time_ns") = 0, py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
            py::arg("kind") = ExperimentEventKind::unspecified,
            py::arg("paradigm") = kUnsetParadigmId, py::arg("code") = 0, py::arg("value") = 0)
        .def_readonly("time_ns", &ExperimentEvent::time_ns)
        .def_readonly("sequence", &ExperimentEvent::sequence)
        .def_readonly("trial", &ExperimentEvent::trial)
        .def_readonly("kind", &ExperimentEvent::kind)
        .def_readonly("paradigm", &ExperimentEvent::paradigm)
        .def_readonly("code", &ExperimentEvent::code)
        .def_readonly("value", &ExperimentEvent::value);

    py::class_<StateTransition>(experiments, "StateTransition")
        .def(py::init(
                 [](ExperimentTimeNs time_ns, SequenceOrdinal sequence, const TrialIdentity& trial,
                    ParadigmId paradigm, StateId from_state, StateId to_state, std::uint32_t cause)
                 {
                     return StateTransition{time_ns,    sequence, trial, paradigm,
                                            from_state, to_state, cause};
                 }),
             py::arg("time_ns") = 0, py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("paradigm") = kUnsetParadigmId, py::arg("from_state") = 0,
             py::arg("to_state") = 0, py::arg("cause") = 0)
        .def_readonly("time_ns", &StateTransition::time_ns)
        .def_readonly("sequence", &StateTransition::sequence)
        .def_readonly("trial", &StateTransition::trial)
        .def_readonly("paradigm", &StateTransition::paradigm)
        .def_readonly("from_state", &StateTransition::from_state)
        .def_readonly("to_state", &StateTransition::to_state)
        .def_readonly("cause", &StateTransition::cause);

    py::enum_<SelectionKind>(experiments, "SelectionKind")
        .value("DISCRETE", SelectionKind::discrete)
        .value("DWELL", SelectionKind::dwell);

    experiments.def("selection_kind_declared", &selection_kind_declared, py::arg("kind"));

    py::class_<SelectionEvent>(experiments, "SelectionEvent")
        .def(py::init(
                 [](ExperimentTimeNs time_ns, SequenceOrdinal sequence, const TrialIdentity& trial,
                    ParadigmId paradigm, SelectionKind kind, bool correct, TargetId selected_id,
                    TargetId intended_id, DurationNs dwell_ns)
                 {
                     return SelectionEvent{time_ns, sequence,    trial,       paradigm, kind,
                                           correct, selected_id, intended_id, dwell_ns};
                 }),
             py::arg("time_ns") = 0, py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("paradigm") = kUnsetParadigmId, py::arg("kind") = SelectionKind::discrete,
             py::arg("correct") = false, py::arg("selected_id") = kUnsetTargetId,
             py::arg("intended_id") = kUnsetTargetId, py::arg("dwell_ns") = 0)
        .def_readonly("time_ns", &SelectionEvent::time_ns)
        .def_readonly("sequence", &SelectionEvent::sequence)
        .def_readonly("trial", &SelectionEvent::trial)
        .def_readonly("paradigm", &SelectionEvent::paradigm)
        .def_readonly("kind", &SelectionEvent::kind)
        .def_readonly("correct", &SelectionEvent::correct)
        .def_readonly("selected_id", &SelectionEvent::selected_id)
        .def_readonly("intended_id", &SelectionEvent::intended_id)
        .def_readonly("dwell_ns", &SelectionEvent::dwell_ns);

    py::class_<TrialRecord>(experiments, "TrialRecord")
        .def(py::init([](const TrialIdentity& trial, const TimeInterval& interval,
                         ParadigmId paradigm, TrialOutcome outcome, std::uint32_t reason)
                      { return TrialRecord{trial, interval, paradigm, outcome, reason}; }),
             py::arg("trial") = TrialIdentity{}, py::arg("interval") = TimeInterval{},
             py::arg("paradigm") = kUnsetParadigmId, py::arg("outcome") = TrialOutcome::pending,
             py::arg("reason") = 0)
        .def_readonly("trial", &TrialRecord::trial)
        .def_readonly("interval", &TrialRecord::interval)
        .def_readonly("paradigm", &TrialRecord::paradigm)
        .def_readonly("outcome", &TrialRecord::outcome)
        .def_readonly("reason", &TrialRecord::reason);

    py::enum_<AbnormalCondition>(experiments, "AbnormalCondition")
        .value("UNSPECIFIED", AbnormalCondition::unspecified)
        .value("DECODED_COMMAND_INVALID", AbnormalCondition::decoded_command_invalid)
        .value("INPUT_SCHEMA_MISMATCH", AbnormalCondition::input_schema_mismatch)
        .value("INPUT_STALE", AbnormalCondition::input_stale)
        .value("SOURCE_DISCONTINUITY", AbnormalCondition::source_discontinuity)
        .value("INPUT_GAP", AbnormalCondition::input_gap)
        .value("DEADLINE_MISSED", AbnormalCondition::deadline_missed)
        .value("OBSERVER_FRAME_DROP", AbnormalCondition::observer_frame_drop)
        .value("RECORDER_FAULT", AbnormalCondition::recorder_fault)
        .value("ACTUATOR_FAULT", AbnormalCondition::actuator_fault)
        .value("RUNTIME_FAULT", AbnormalCondition::runtime_fault)
        .value("PRESENTATION_FAILED", AbnormalCondition::presentation_failed)
        .value("PRESENTATION_EVIDENCE_MISSING", AbnormalCondition::presentation_evidence_missing)
        .value("INPUT_AFTER_TERMINAL", AbnormalCondition::input_after_terminal)
        .value("EMERGENCY_STOP", AbnormalCondition::emergency_stop)
        .value("PRESENTATION_REPORT_UNMATCHED", AbnormalCondition::presentation_report_unmatched);

    py::enum_<AbnormalPolicy>(experiments, "AbnormalPolicy")
        .value("RECORD", AbnormalPolicy::record)
        .value("ABORT_TRIAL", AbnormalPolicy::abort_trial)
        .value("ABORT_SESSION", AbnormalPolicy::abort_session);

    py::enum_<AbnormalResponse>(experiments, "AbnormalResponse")
        .value("RECORDED", AbnormalResponse::recorded)
        .value("INPUT_REFUSED", AbnormalResponse::input_refused)
        .value("TRIAL_INVALIDATED", AbnormalResponse::trial_invalidated)
        .value("TRIAL_ABORTED", AbnormalResponse::trial_aborted)
        .value("SESSION_ABORTED", AbnormalResponse::session_aborted);

    experiments.def("abnormal_condition_declared", &abnormal_condition_declared,
                    py::arg("condition"));
    experiments.def("abnormal_policy_declared", &abnormal_policy_declared, py::arg("policy"));
    experiments.def("abnormal_response_declared", &abnormal_response_declared, py::arg("response"));
    experiments.def("abnormal_response_admits_trial", &abnormal_response_admits_trial,
                    py::arg("response"));
    experiments.def("escalate", &escalate, py::arg("left"), py::arg("right"));
    experiments.def("abnormal_condition_name", &abnormal_condition_name, py::arg("condition"));

    py::class_<AbnormalPolicySet>(experiments, "AbnormalPolicySet")
        .def(py::init(
                 [](AbnormalPolicy decoded_command_invalid, AbnormalPolicy input_discontinuity,
                    AbnormalPolicy presentation_failed,
                    AbnormalPolicy presentation_evidence_missing, AbnormalPolicy acquisition_fault)
                 {
                     return AbnormalPolicySet{decoded_command_invalid, input_discontinuity,
                                              presentation_failed, presentation_evidence_missing,
                                              acquisition_fault};
                 }),
             py::arg("decoded_command_invalid") = AbnormalPolicy::abort_trial,
             py::arg("input_discontinuity") = AbnormalPolicy::abort_trial,
             py::arg("presentation_failed") = AbnormalPolicy::abort_trial,
             py::arg("presentation_evidence_missing") = AbnormalPolicy::record,
             py::arg("acquisition_fault") = AbnormalPolicy::abort_session)
        .def_readonly("decoded_command_invalid", &AbnormalPolicySet::decoded_command_invalid)
        .def_readonly("input_discontinuity", &AbnormalPolicySet::input_discontinuity)
        .def_readonly("presentation_failed", &AbnormalPolicySet::presentation_failed)
        .def_readonly("presentation_evidence_missing",
                      &AbnormalPolicySet::presentation_evidence_missing)
        .def_readonly("acquisition_fault", &AbnormalPolicySet::acquisition_fault);

    experiments.def("policy_for", &policy_for, py::arg("policies"), py::arg("condition"));

    py::class_<AbnormalEvent>(experiments, "AbnormalEvent")
        .def(py::init(
                 [](ExperimentTimeNs time_ns, SequenceOrdinal sequence, const TrialIdentity& trial,
                    ParadigmId paradigm, AbnormalCondition condition, AbnormalPolicy policy,
                    AbnormalResponse response, bool has_trial, std::uint32_t detail)
                 {
                     return AbnormalEvent{time_ns, sequence, trial,     paradigm, condition,
                                          policy,  response, has_trial, detail};
                 }),
             py::arg("time_ns") = 0, py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("paradigm") = kUnsetParadigmId,
             py::arg("condition") = AbnormalCondition::unspecified,
             py::arg("policy") = AbnormalPolicy::record,
             py::arg("response") = AbnormalResponse::recorded, py::arg("has_trial") = false,
             py::arg("detail") = 0)
        .def_readonly("time_ns", &AbnormalEvent::time_ns)
        .def_readonly("sequence", &AbnormalEvent::sequence)
        .def_readonly("trial", &AbnormalEvent::trial)
        .def_readonly("paradigm", &AbnormalEvent::paradigm)
        .def_readonly("condition", &AbnormalEvent::condition)
        .def_readonly("policy", &AbnormalEvent::policy)
        .def_readonly("response", &AbnormalEvent::response)
        .def_readonly("has_trial", &AbnormalEvent::has_trial)
        .def_readonly("detail", &AbnormalEvent::detail);

    py::enum_<CueKind>(experiments, "CueKind")
        .value("NONE", CueKind::none)
        .value("BLACK", CueKind::black)
        .value("FIXATION_CROSS", CueKind::fixation_cross)
        .value("TEXT_CONTENT", CueKind::text_content)
        .value("SSVEP_TARGETS", CueKind::ssvep_targets);

    experiments.def("cue_kind_declared", &cue_kind_declared, py::arg("cue"));

    py::class_<PresentationRequest>(experiments, "PresentationRequest")
        .def(py::init(
                 [](ExperimentTimeNs requested_ns, ExperimentTimeNs onset_ns,
                    ExperimentTimeNs valid_until_ns, DurationNs duration_ns,
                    SequenceOrdinal sequence, const TrialIdentity& trial, ParadigmId paradigm,
                    PhaseId phase, CueKind cue, StimulusId stimulus_id)
                 {
                     return PresentationRequest{
                         requested_ns, onset_ns, valid_until_ns, duration_ns, sequence,
                         trial,        paradigm, phase,          cue,         stimulus_id};
                 }),
             py::arg("requested_ns") = 0, py::arg("onset_ns") = 0,
             py::arg("valid_until_ns") = kNoExpiryNs, py::arg("duration_ns") = 0,
             py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("paradigm") = kUnsetParadigmId, py::arg("phase") = 0,
             py::arg("cue") = CueKind::none, py::arg("stimulus_id") = kUnsetStimulusId)
        .def_readonly("requested_ns", &PresentationRequest::requested_ns)
        .def_readonly("onset_ns", &PresentationRequest::onset_ns)
        .def_readonly("valid_until_ns", &PresentationRequest::valid_until_ns)
        .def_readonly("duration_ns", &PresentationRequest::duration_ns)
        .def_readonly("sequence", &PresentationRequest::sequence)
        .def_readonly("trial", &PresentationRequest::trial)
        .def_readonly("paradigm", &PresentationRequest::paradigm)
        .def_readonly("phase", &PresentationRequest::phase)
        .def_readonly("cue", &PresentationRequest::cue)
        .def_readonly("stimulus_id", &PresentationRequest::stimulus_id);

    py::class_<PresentationState>(experiments, "PresentationState")
        .def(py::init(
                 [](ExperimentTimeNs since_ns, const TrialIdentity& trial, ParadigmId paradigm,
                    PhaseId phase, CueKind cue, StimulusId stimulus_id)
                 { return PresentationState{since_ns, trial, paradigm, phase, cue, stimulus_id}; }),
             py::arg("since_ns") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("paradigm") = kUnsetParadigmId, py::arg("phase") = 0,
             py::arg("cue") = CueKind::none, py::arg("stimulus_id") = kUnsetStimulusId)
        .def_readonly("since_ns", &PresentationState::since_ns)
        .def_readonly("trial", &PresentationState::trial)
        .def_readonly("paradigm", &PresentationState::paradigm)
        .def_readonly("phase", &PresentationState::phase)
        .def_readonly("cue", &PresentationState::cue)
        .def_readonly("stimulus_id", &PresentationState::stimulus_id);

    py::enum_<PresentationStatus>(experiments, "PresentationStatus")
        .value("NOT_REPORTED", PresentationStatus::not_reported)
        .value("PRESENTED", PresentationStatus::presented)
        .value("SKIPPED", PresentationStatus::skipped)
        .value("EXPIRED", PresentationStatus::expired);

    experiments.def("presentation_status_declared", &presentation_status_declared,
                    py::arg("status"));

    py::class_<PresentationOutcome>(experiments, "PresentationOutcome")
        .def(py::init(
                 [](ExperimentTimeNs requested_ns, ExperimentTimeNs presented_ns,
                    SequenceOrdinal sequence, SequenceOrdinal request_sequence,
                    const TrialIdentity& trial, PresentationStatus status, StimulusId stimulus_id)
                 {
                     return PresentationOutcome{requested_ns,     presented_ns, sequence,
                                                request_sequence, trial,        status,
                                                stimulus_id};
                 }),
             py::arg("requested_ns") = 0, py::arg("presented_ns") = 0, py::arg("sequence") = 0,
             py::arg("request_sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("status") = PresentationStatus::not_reported,
             py::arg("stimulus_id") = kUnsetStimulusId)
        .def_readonly("requested_ns", &PresentationOutcome::requested_ns)
        .def_readonly("presented_ns", &PresentationOutcome::presented_ns)
        .def_readonly("sequence", &PresentationOutcome::sequence)
        .def_readonly("request_sequence", &PresentationOutcome::request_sequence)
        .def_readonly("trial", &PresentationOutcome::trial)
        .def_readonly("status", &PresentationOutcome::status)
        .def_readonly("stimulus_id", &PresentationOutcome::stimulus_id);

    py::enum_<CommandFrame>(experiments, "CommandFrame")
        .value("UNSPECIFIED", CommandFrame::unspecified)
        .value("WORKSPACE_2D", CommandFrame::workspace_2d)
        .value("WORKSPACE_3D", CommandFrame::workspace_3d)
        .value("DEVICE_NATIVE", CommandFrame::device_native);

    py::enum_<CommandAxisName>(experiments, "CommandAxisName")
        .value("UNSPECIFIED", CommandAxisName::unspecified)
        .value("X", CommandAxisName::x)
        .value("Y", CommandAxisName::y)
        .value("Z", CommandAxisName::z)
        .value("ROLL", CommandAxisName::roll)
        .value("PITCH", CommandAxisName::pitch)
        .value("YAW", CommandAxisName::yaw)
        .value("GRASP", CommandAxisName::grasp);

    py::enum_<CommandUnit>(experiments, "CommandUnit")
        .value("UNSPECIFIED", CommandUnit::unspecified)
        .value("DIMENSIONLESS", CommandUnit::dimensionless)
        .value("NORMALIZED", CommandUnit::normalized)
        .value("METRES", CommandUnit::metres)
        .value("METRES_PER_SECOND", CommandUnit::metres_per_second)
        .value("RADIANS", CommandUnit::radians)
        .value("RADIANS_PER_SECOND", CommandUnit::radians_per_second);

    experiments.def("command_frame_declared", &command_frame_declared, py::arg("frame"));
    experiments.def("command_axis_name_declared", &command_axis_name_declared, py::arg("name"));
    experiments.def("command_unit_declared", &command_unit_declared, py::arg("unit"));

    py::class_<CommandAxis>(experiments, "CommandAxis")
        .def(py::init([](CommandAxisName name, CommandUnit unit)
                      { return CommandAxis{name, unit}; }),
             py::arg("name") = CommandAxisName::unspecified,
             py::arg("unit") = CommandUnit::unspecified)
        .def_readonly("name", &CommandAxis::name)
        .def_readonly("unit", &CommandAxis::unit);

    py::class_<CommandSpace>(experiments, "CommandSpace")
        // A space is prepared once, before the session, and the axes are part of
        // preparing it. There is no set_axis(): a space whose meaning can be
        // edited after it has been referenced by an identifier is not a space
        // that identifier can stand for.
        .def(py::init(
                 [](CommandSpaceId id, std::uint8_t dim, CommandFrame frame,
                    const std::vector<CommandAxis>& axes)
                 {
                     return CommandSpace{id, dim, frame,
                                         to_fixed<CommandAxis, kMaxCommandDim>(axes, dim, "axes")};
                 }),
             py::arg("id") = kUnsetCommandSpaceId, py::arg("dim") = 0,
             py::arg("frame") = CommandFrame::unspecified,
             py::arg("axes") = std::vector<CommandAxis>{})
        .def_readonly("id", &CommandSpace::id)
        .def_readonly("dim", &CommandSpace::dim)
        .def_readonly("frame", &CommandSpace::frame)
        .def_readonly("axes", &CommandSpace::axes)
        .def(
            "axis",
            [](const CommandSpace& space, std::size_t idx)
            {
                check_command_index(idx);
                return space.axes[idx];
            },
            py::arg("idx"));

    py::class_<CommandRequest>(experiments, "CommandRequest")
        .def(py::init(
                 [](ExperimentTimeNs generated_ns, ExperimentTimeNs valid_until_ns,
                    SequenceOrdinal sequence, const TrialIdentity& trial, CommandSpaceId space,
                    std::uint8_t dim, const std::vector<double>& values)
                 {
                     return CommandRequest{generated_ns,
                                           valid_until_ns,
                                           sequence,
                                           trial,
                                           space,
                                           dim,
                                           to_fixed<double, kMaxCommandDim>(values, dim, "values")};
                 }),
             py::arg("generated_ns") = 0, py::arg("valid_until_ns") = kNoExpiryNs,
             py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("space") = kUnsetCommandSpaceId, py::arg("dim") = 0,
             py::arg("values") = std::vector<double>{})
        .def_readonly("generated_ns", &CommandRequest::generated_ns)
        .def_readonly("valid_until_ns", &CommandRequest::valid_until_ns)
        .def_readonly("sequence", &CommandRequest::sequence)
        .def_readonly("trial", &CommandRequest::trial)
        .def_readonly("space", &CommandRequest::space)
        .def_readonly("dim", &CommandRequest::dim)
        .def_readonly("values", &CommandRequest::values)
        .def(
            "value",
            [](const CommandRequest& request, std::size_t idx)
            {
                check_command_index(idx);
                return request.values[idx];
            },
            py::arg("idx"));

    py::enum_<CommandApplication>(experiments, "CommandApplication")
        .value("NOT_SUBMITTED", CommandApplication::not_submitted)
        .value("ACCEPTED", CommandApplication::accepted)
        .value("REJECTED", CommandApplication::rejected)
        .value("EXPIRED", CommandApplication::expired);

    experiments.def("command_application_declared", &command_application_declared,
                    py::arg("application"));

    py::class_<CommandOutcome>(experiments, "CommandOutcome")
        .def(py::init(
                 [](ExperimentTimeNs generated_ns, ExperimentTimeNs submitted_ns,
                    SequenceOrdinal sequence, SequenceOrdinal request_sequence,
                    const TrialIdentity& trial, CommandApplication application,
                    std::uint16_t status_code)
                 {
                     return CommandOutcome{generated_ns, submitted_ns, sequence,   request_sequence,
                                           trial,        application,  status_code};
                 }),
             py::arg("generated_ns") = 0, py::arg("submitted_ns") = 0, py::arg("sequence") = 0,
             py::arg("request_sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("application") = CommandApplication::not_submitted, py::arg("status_code") = 0)
        .def_readonly("generated_ns", &CommandOutcome::generated_ns)
        .def_readonly("submitted_ns", &CommandOutcome::submitted_ns)
        .def_readonly("sequence", &CommandOutcome::sequence)
        .def_readonly("request_sequence", &CommandOutcome::request_sequence)
        .def_readonly("trial", &CommandOutcome::trial)
        .def_readonly("application", &CommandOutcome::application)
        .def_readonly("status_code", &CommandOutcome::status_code);

    py::class_<ExperimentSnapshot>(experiments, "ExperimentSnapshot")
        .def(py::init(
                 [](ExperimentTimeNs origin_ns, const ScheduleIdentity& schedule,
                    TrialOrdinal first_trial_ordinal, SequenceOrdinal first_sequence,
                    BlockOrdinal first_block, ParadigmId paradigm)
                 {
                     return ExperimentSnapshot{origin_ns,      schedule,    first_trial_ordinal,
                                               first_sequence, first_block, paradigm};
                 }),
             py::arg("origin_ns") = 0, py::arg("schedule") = ScheduleIdentity{},
             py::arg("first_trial_ordinal") = 0, py::arg("first_sequence") = 0,
             py::arg("first_block") = 0, py::arg("paradigm") = kUnsetParadigmId)
        .def_readonly("origin_ns", &ExperimentSnapshot::origin_ns)
        .def_readonly("schedule", &ExperimentSnapshot::schedule)
        .def_readonly("first_trial_ordinal", &ExperimentSnapshot::first_trial_ordinal)
        .def_readonly("first_sequence", &ExperimentSnapshot::first_sequence)
        .def_readonly("first_block", &ExperimentSnapshot::first_block)
        .def_readonly("paradigm", &ExperimentSnapshot::paradigm);

    py::class_<DecisionSnapshot>(experiments, "DecisionSnapshot")
        .def(py::init(
                 [](ExperimentTimeNs time_ns, SequenceOrdinal sequence, const TrialIdentity& trial,
                    const ScheduleIdentity& schedule, DrawStream stream, std::uint32_t draw_cursor,
                    ParadigmId paradigm, StateId state)
                 {
                     return DecisionSnapshot{time_ns, sequence,    trial,    schedule,
                                             stream,  draw_cursor, paradigm, state};
                 }),
             py::arg("time_ns") = 0, py::arg("sequence") = 0, py::arg("trial") = TrialIdentity{},
             py::arg("schedule") = ScheduleIdentity{}, py::arg("stream") = 0,
             py::arg("draw_cursor") = 0, py::arg("paradigm") = kUnsetParadigmId,
             py::arg("state") = 0)
        .def_readonly("time_ns", &DecisionSnapshot::time_ns)
        .def_readonly("sequence", &DecisionSnapshot::sequence)
        .def_readonly("trial", &DecisionSnapshot::trial)
        .def_readonly("schedule", &DecisionSnapshot::schedule)
        .def_readonly("stream", &DecisionSnapshot::stream)
        .def_readonly("draw_cursor", &DecisionSnapshot::draw_cursor)
        .def_readonly("paradigm", &DecisionSnapshot::paradigm)
        .def_readonly("state", &DecisionSnapshot::state);

    experiments.def("next_draw_key", &next_draw_key, py::arg("snapshot"));

    // The replay verdict vocabulary. The engines themselves are native -- a
    // replay steps a paradigm's own machine over a recorded timeline and
    // compares as it goes -- but the answer they produce is read by whoever
    // decides whether a recording is trustworthy, and that reader is usually in
    // Python.
    py::enum_<ReplayVerdict>(experiments, "ReplayVerdict")
        .value("MATCH", ReplayVerdict::match)
        .value("MISMATCH", ReplayVerdict::mismatch)
        .value("INCOMPLETE", ReplayVerdict::incomplete)
        .value("REJECTED", ReplayVerdict::rejected);

    py::enum_<ReplayRejection>(experiments, "ReplayRejection")
        .value("NONE", ReplayRejection::none)
        .value("PARADIGM_MISMATCH", ReplayRejection::paradigm_mismatch)
        .value("EXPERIMENT_VERSION_MISMATCH", ReplayRejection::experiment_version_mismatch)
        .value("CONFIGURATION_FINGERPRINT_MISMATCH",
               ReplayRejection::configuration_fingerprint_mismatch)
        .value("SCHEDULE_FINGERPRINT_MISMATCH", ReplayRejection::schedule_fingerprint_mismatch)
        .value("SEED_MISMATCH", ReplayRejection::seed_mismatch)
        .value("SAMPLER_VERSION_MISMATCH", ReplayRejection::sampler_version_mismatch)
        .value("SAMPLER_VERSION_UNSUPPORTED", ReplayRejection::sampler_version_unsupported)
        .value("REALIZED_SCHEDULE_MISMATCH", ReplayRejection::realized_schedule_mismatch)
        .value("METRIC_VERSION_MISMATCH", ReplayRejection::metric_version_mismatch)
        .value("POLICY_VERSION_MISMATCH", ReplayRejection::policy_version_mismatch)
        .value("PROVENANCE_INCOMPLETE", ReplayRejection::provenance_incomplete)
        .value("CONFIGURATION_INVALID", ReplayRejection::configuration_invalid)
        .value("EVIDENCE_INVALID", ReplayRejection::evidence_invalid)
        .value("METRIC_VERSION_UNSUPPORTED", ReplayRejection::metric_version_unsupported);

    py::enum_<ReplayItem>(experiments, "ReplayItem")
        .value("NONE", ReplayItem::none)
        .value("TRANSITION", ReplayItem::transition)
        .value("EVENT", ReplayItem::event)
        .value("TRIAL", ReplayItem::trial)
        .value("TARGET", ReplayItem::target)
        .value("CURSOR", ReplayItem::cursor)
        .value("ASSISTED_VELOCITY", ReplayItem::assisted_velocity)
        .value("GUIDANCE_SAMPLE", ReplayItem::guidance_sample)
        .value("SELECTION", ReplayItem::selection)
        .value("METRICS", ReplayItem::metrics)
        .value("PHASE", ReplayItem::phase)
        .value("PRESENTATION_REQUEST", ReplayItem::presentation_request)
        .value("SCHEDULE", ReplayItem::schedule)
        .value("STREAM_LENGTH", ReplayItem::stream_length);

    py::enum_<ReplayCompleteness>(experiments, "ReplayCompleteness")
        .value("COMPLETE", ReplayCompleteness::complete)
        .value("STREAM_ABSENT", ReplayCompleteness::stream_absent)
        .value("TRACE_LOSS_RECORDED", ReplayCompleteness::trace_loss_recorded)
        .value("RUN_END_MISSING", ReplayCompleteness::run_end_missing);

    py::enum_<ReplayIncompletePolicy>(experiments, "ReplayIncompletePolicy")
        .value("VERIFY_AVAILABLE", ReplayIncompletePolicy::verify_available)
        .value("REFUSE", ReplayIncompletePolicy::refuse);

    experiments.def("replay_verdict_declared", &replay_verdict_declared, py::arg("verdict"));
    experiments.def("replay_rejection_declared", &replay_rejection_declared, py::arg("rejection"));
    experiments.def("replay_item_declared", &replay_item_declared, py::arg("item"));
    experiments.def("replay_completeness_declared", &replay_completeness_declared,
                    py::arg("completeness"));
    experiments.def("replay_incomplete_policy_declared", &replay_incomplete_policy_declared,
                    py::arg("policy"));
    experiments.def("replay_verdict_name", &replay_verdict_name, py::arg("verdict"));
    experiments.def("replay_rejection_name", &replay_rejection_name, py::arg("rejection"));
    experiments.def("replay_item_name", &replay_item_name, py::arg("item"));
    experiments.def("replay_completeness_name", &replay_completeness_name, py::arg("completeness"));
    experiments.def("replay_reproduced", &replay_reproduced, py::arg("report"));
    experiments.def("check_provenance", &check_provenance, py::arg("recorded"),
                    py::arg("candidate"));

    py::class_<ReplayProvenance>(experiments, "ReplayProvenance")
        .def(py::init(
                 [](ParadigmId paradigm, std::uint32_t experiment_version,
                    std::uint64_t configuration_fingerprint, std::uint64_t schedule_fingerprint,
                    std::uint64_t realized_schedule_fingerprint, ScheduleSeed seed,
                    SamplerVersion sampler_version, std::uint32_t metric_version,
                    std::uint32_t policy_version)
                 {
                     return ReplayProvenance{paradigm,
                                             experiment_version,
                                             configuration_fingerprint,
                                             schedule_fingerprint,
                                             realized_schedule_fingerprint,
                                             seed,
                                             sampler_version,
                                             metric_version,
                                             policy_version};
                 }),
             py::arg("paradigm") = kUnsetParadigmId, py::arg("experiment_version") = 0,
             py::arg("configuration_fingerprint") = 0, py::arg("schedule_fingerprint") = 0,
             py::arg("realized_schedule_fingerprint") = 0, py::arg("seed") = 0,
             py::arg("sampler_version") = 0, py::arg("metric_version") = 0,
             py::arg("policy_version") = 0)
        .def_readonly("paradigm", &ReplayProvenance::paradigm)
        .def_readonly("experiment_version", &ReplayProvenance::experiment_version)
        .def_readonly("configuration_fingerprint", &ReplayProvenance::configuration_fingerprint)
        .def_readonly("schedule_fingerprint", &ReplayProvenance::schedule_fingerprint)
        .def_readonly("realized_schedule_fingerprint",
                      &ReplayProvenance::realized_schedule_fingerprint)
        .def_readonly("seed", &ReplayProvenance::seed)
        .def_readonly("sampler_version", &ReplayProvenance::sampler_version)
        .def_readonly("metric_version", &ReplayProvenance::metric_version)
        .def_readonly("policy_version", &ReplayProvenance::policy_version);

    py::class_<ReplayMismatch>(experiments, "ReplayMismatch")
        .def(py::init<>())
        .def_readonly("item", &ReplayMismatch::item)
        .def_readonly("idx", &ReplayMismatch::idx)
        .def_readonly("time_ns", &ReplayMismatch::time_ns)
        .def_readonly("trial", &ReplayMismatch::trial)
        .def_property_readonly("field",
                               [](const ReplayMismatch& value) { return std::string{value.field}; })
        .def_readonly("recorded", &ReplayMismatch::recorded)
        .def_readonly("regenerated", &ReplayMismatch::regenerated)
        .def_readonly("recorded_real", &ReplayMismatch::recorded_real)
        .def_readonly("regenerated_real", &ReplayMismatch::regenerated_real)
        .def_readonly("real_valued", &ReplayMismatch::real_valued);

    py::class_<ReplayReport>(experiments, "ReplayReport")
        .def(py::init<>())
        .def_readonly("verdict", &ReplayReport::verdict)
        .def_readonly("rejection", &ReplayReport::rejection)
        .def_readonly("completeness", &ReplayReport::completeness)
        .def_readonly("first_mismatch", &ReplayReport::first_mismatch)
        .def_readonly("inputs_replayed", &ReplayReport::inputs_replayed)
        .def_readonly("items_compared", &ReplayReport::items_compared)
        .def_readonly("fields_compared", &ReplayReport::fields_compared);

    // One overloaded entry point, dispatched by record type. Validation returns
    // a reason rather than raising, so a realtime caller never pays for an
    // exception and a Python caller still learns exactly what was wrong.
    bind_validate<AbnormalEvent>(experiments);
    bind_validate<AbnormalPolicySet>(experiments);
    bind_validate<TimeInterval>(experiments);
    bind_validate<ScheduleIdentity>(experiments);
    bind_validate<ScheduleDraw>(experiments);
    bind_validate<ExperimentEvent>(experiments);
    bind_validate<StateTransition>(experiments);
    bind_validate<SelectionEvent>(experiments);
    bind_validate<TrialRecord>(experiments);
    bind_validate<PresentationRequest>(experiments);
    bind_validate<PresentationState>(experiments);
    bind_validate<PresentationOutcome>(experiments);
    bind_validate<CommandSpace>(experiments);
    bind_validate<CommandRequest>(experiments);
    bind_validate<CommandOutcome>(experiments);
    bind_validate<ExperimentSnapshot>(experiments);
    bind_validate<DecisionSnapshot>(experiments);
    experiments.def("validate_against", &validate_against, py::arg("request"), py::arg("space"));
}
