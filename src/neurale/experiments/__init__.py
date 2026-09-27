#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared experiment value contract for the native paradigms.

This package exposes the smallest set of values genuinely shared by Center-Out
2D, WebGrid, and the Speech cue paradigm, including semantic-replay reports and
identity. It does not expose private native controllers or the recording bridge.
It is a value contract: there is no generic
experiment base class, task registry, factory, or scheduler. Each paradigm owns
its own configuration, state enumeration, transition logic, geometry or
stimulus semantics, and metrics.

Every value is implemented natively and reached through pybind11. Experiment
time is an integer nanosecond count throughout; there are no floating-point
seconds anywhere in the contract, and no state machine samples a clock.

The value records are immutable: each is built by a complete constructor and
exposes readonly fields. A record that could still be edited after it had been
validated, fingerprinted, or persisted would carry no guarantee at all, so
construction is the only point at which a value is decided. The stateful helpers
-- ``MonotonicTimeGate``, ``TrialCounter``, ``SequenceCounter``, and
``FingerprintAccumulator`` -- are the deliberate exception; advancing is what
they exist for.

The fixed-capacity sequences -- ``CommandSpace.axes`` and
``CommandRequest.values`` -- are padded out to the record's capacity only at or
beyond ``dimension``, where the contract requires the slot to be unset anyway. A
sequence shorter than the ``dimension`` it is given raises :exc:`ValueError`
rather than being filled in: a velocity component nobody supplied and one
deliberately set to zero are the same bytes once written, and construction is
the last point at which they can still be told apart.

Two things live beside the contract rather than in it.
:mod:`neurale.experiments.assistance` owns generic velocity shared control --
the linear blend and the orthogonal-impedance projection -- for any paradigm
whose movement command is a velocity vector. It is a submodule and not a
top-level ``neurale.control`` domain, and its transforms are blind: they see
vectors and parameters, never targets, trials, phases, or devices.
:mod:`neurale.experiments.center_out` owns one paradigm's task, protocol,
session, target geometry, containment predicate, and target schedule. It is nested for
the opposite reason: it is *not* shared, and a paradigm's configuration
reachable as though it were part of the contract would invite a second paradigm
to build on it.
:mod:`neurale.experiments.webgrid` likewise owns renderer-independent grid
geometry, a deterministic target schedule, explicit discrete-selection
semantics, its pure task state machine, and raw benchmark metrics; it contains
no browser or UI implementation. :mod:`neurale.experiments.speech` owns the
Speech cue configuration, the immutable semantic stimulus catalog, the
deterministic per-trial timing schedule whose phase durations are drawn from
open ranges, and the intended phase timeline; whether there is a fixation cross
is configuration there rather than an outcome, and a disabled cross is absent
from the timeline rather than present with a zero duration. It presents nothing,
captures no audio, and decodes no speech.
:mod:`neurale.experiments.traceability` sits beside those native semantics as
the offline, dependency-light provenance query layer. It keeps multi-range
feature sources and missing recording evidence explicit, and imports no
streaming, recording, NRF, or native machinery.

The contract also carries the vocabulary for what a run does when it stops
being normal: ``AbnormalCondition`` says what was observed, ``AbnormalPolicy``
and ``AbnormalPolicySet`` say how severely a run treats that class of
observation, and ``AbnormalResponse`` says what actually happened to the trial
in flight. The three are separate because a configured severity and its effect
are not the same fact -- a paradigm whose state machine owns trial termination
can only mark a trial inadmissible where another can end it -- and a record that
conflated them could not be read back. Nothing here inhibits or releases an
actuator: that belongs to the streaming runtime and its safety controller, and
this vocabulary has no second opinion about it.

This package deliberately defines no event or trial *data model*.
:class:`neurale.data.Event`, :class:`neurale.data.EventSeries`,
:class:`neurale.data.Trial`, and :class:`neurale.data.TrialTable` remain the
offline analysis containers, in floating-point seconds. ``ExperimentEvent`` and
``TrialRecord`` here are the runtime records: they carry a pending outcome, a
paradigm-owned reason code, and integer nanoseconds, none of which the offline
types can represent. Converting one into the other is an export boundary, and
it is the only place where nanoseconds become seconds.

Importing this package is lightweight: no native extension, no GUI, no vendor
SDK, no recording format, and no runtime is loaded until a contract name is
actually used.
"""

from __future__ import annotations

_NATIVE_EXPORTS = {
    "AbnormalCondition",
    "AbnormalEvent",
    "AbnormalPolicy",
    "AbnormalPolicySet",
    "AbnormalResponse",
    "CURRENT_SAMPLER_VERSION",
    "CommandApplication",
    "CommandAxis",
    "CommandAxisName",
    "CommandFrame",
    "CommandOutcome",
    "CommandRequest",
    "CommandSpace",
    "CommandUnit",
    "ContractStatus",
    "CueKind",
    "DRAW_SLOT_STRIDE",
    "DecisionSnapshot",
    "DrawKey",
    "ExperimentEvent",
    "ExperimentEventKind",
    "ExperimentSnapshot",
    "FingerprintAccumulator",
    "MAX_COMMAND_DIMENSION",
    "MAX_REJECTION_DRAWS",
    "MonotonicTimeGate",
    "NO_EXPIRY_NS",
    "PresentationOutcome",
    "PresentationRequest",
    "PresentationState",
    "PresentationStatus",
    "ReplayAuthority",
    "ReplayCompleteness",
    "ReplayIncompletePolicy",
    "ReplayItem",
    "ReplayMismatch",
    "ReplayProvenance",
    "ReplayRejection",
    "ReplayReport",
    "ReplayVerdict",
    "SAMPLER_VERSION_1",
    "ScheduleDraw",
    "ScheduleIdentity",
    "SelectionEvent",
    "SelectionKind",
    "SequenceCounter",
    "StateTransition",
    "TimeInterval",
    "TrialCounter",
    "TrialIdentity",
    "TrialOutcome",
    "TrialRecord",
    "UNSET_COMMAND_SPACE_ID",
    "UNSET_PARADIGM_ID",
    "UNSET_STIMULUS_ID",
    "UNSET_TARGET_ID",
    "UNSET_TRIAL_KEY",
    "abnormal_condition_declared",
    "abnormal_condition_name",
    "abnormal_policy_declared",
    "abnormal_response_admits_trial",
    "abnormal_response_declared",
    "check_provenance",
    "command_application_declared",
    "command_axis_name_declared",
    "command_frame_declared",
    "command_unit_declared",
    "contract_status_message",
    "cue_kind_declared",
    "escalate",
    "experiment_event_kind_declared",
    "fingerprint_of_bytes",
    "interval_from_duration",
    "next_draw_key",
    "policy_for",
    "presentation_status_declared",
    "replay_authority",
    "replay_completeness_declared",
    "replay_completeness_name",
    "replay_incomplete_policy_declared",
    "replay_item_declared",
    "replay_item_name",
    "replay_rejection_declared",
    "replay_rejection_name",
    "replay_reproduced",
    "replay_verdict_declared",
    "replay_verdict_name",
    "same_trial",
    "sample_bits",
    "sample_exclusive",
    "sample_inclusive",
    "sample_index",
    "sampler_mix64",
    "sampler_version_supported",
    "schedule_fingerprint",
    "selection_kind_declared",
    "time_fits",
    "trial_ended",
    "trial_outcome_declared",
    "validate",
    "validate_against",
}


# Submodules reachable as attributes without an explicit import of their own.
# Resolving one imports a Python module and nothing else; the native extension
# still waits until a name inside it is used.
_SUBMODULES = frozenset(
    {
        "assistance",
        "center_out",
        "presentation",
        "speech",
        "ssvep",
        "traceability",
        "webgrid",
    }
)


def __getattr__(name: str) -> object:
    if name in _SUBMODULES:
        import importlib

        value = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = value
        return value
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from neurale._native_loader import load_native_namespace

    value = getattr(load_native_namespace("experiments"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS | _SUBMODULES)


__all__ = sorted(_NATIVE_EXPORTS | _SUBMODULES)
