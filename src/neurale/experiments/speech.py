#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Speech cue configuration, stimulus catalog, and deterministic timing schedule.

One trial presents a blank field, optionally a fixation cross, and then textual
content::

    BLACK -> CROSS (optional, configuration-controlled) -> CONTENT

Whether there is a cross is ``cross_enabled``: configuration, never a draw. When
it is set, the cross duration is sampled from an open range and is therefore
strictly positive; when it is clear, no cross draw is made, the realized
``cross_duration_ns`` is exactly zero, and the timeline has no cross phase in it
-- not a cross phase of no length. That is what makes a recorded zero
unambiguous, and it is why ``build_timeline`` returns two phases rather than
three.

Every sampled duration lies strictly inside its configured bound: never zero,
and never equal to the bound. In integer nanoseconds ``black_duration_ns`` is
drawn from ``[1, t1 - 1]``, ``content_duration_ns`` from ``[1, t3 - 1]``, and
``cross_duration_ns`` from ``[1, t2 - 1]`` when the cross is enabled. A bound
admitting no such integer is rejected as ``RANGE_EMPTY`` rather than clamped.

Sampling is addressed rather than advanced, through the shared stateless
sampler: a trial's values are a pure function of the seed, the stream, the trial
ordinal, the draw ordinal, and the sampler version. There is no global RNG, no
``random_device``, no ``<random>`` distribution, and no NumPy state anywhere.
``prepare_trial(config, 900)`` therefore costs the same as ``prepare_trial(config,
0)`` and needs none of the trials before it, so a session of any length is
scheduled without a prepared table. The draw ordinals are fixed per phase
(``BLACK_DRAW``, ``CROSS_DRAW``, ``CONTENT_DRAW``) rather than allocated in the
order draws happen, so toggling ``cross_enabled`` leaves every black and content
duration in the session exactly where it was.

A configuration either regenerates its schedule from a seed or carries it
verbatim, and ``replay_authority`` says which. Fields that would decide a
realized value must be absent on the side that does not decide it: an explicit
schedule carries no stimulus set, no ordering policy, and no cross bound when
the cross is off. The seed is the exception, and stays as provenance.

A supplied schedule is validated, not trusted, and ``validate_against`` asks
whether it *belongs* to the configuration rather than whether it is plausible
under one: being well-formed and inside every bound is not evidence of
provenance. Its sampler version must be the configuration's, because a session
identity names one sampler and cannot name two. Under ``SEEDED`` its stimulus
must be one the configuration can present, so a schedule cannot put content into
a timeline that the deterministic order would never have selected. Under
``EXPLICIT_SEQUENCE`` it must equal ``config.explicit_schedule[ordinal]`` field
for field, because that entry *is* the replay authority for the trial and a
substitute that merely fits the bounds is a second answer to a question already
answered. What this does not do is re-run the sampler to confirm a seeded
schedule's values: admitting a schedule and verifying that a session reproduces
are different questions.

``make_schedule_draws`` files the realized values under a ``TrialIdentity``, and
that identity has to be the schedule's own -- its ordinal and stimulus are
checked rather than copied, because provenance attributing one trial's draw to
another is wrong in the one field it exists to carry.

The stimulus catalog is owned here and not by a renderer: a recorded session
carrying identifiers whose meaning lived only in the presentation layer would
not be reproducible. ``SpeechStimulus.text`` is canonical UTF-8 bytes: the
contract imposes exactly one encoding and validates it, so a ``str`` passed to
the constructor is encoded as UTF-8 and raw ``bytes`` are admitted only when they
are already a complete, well-formed UTF-8 sequence. Content is deliberately
narrow -- a text prompt, a stable identifier, an optional class label, and one
opaque integer of metadata. It is not a general stimulus framework: images,
audio, and phoneme-specific presentation need a new ``SpeechContentKind``
enumerator and its own fields, not an arbitrary payload.

``SpeechMachine`` runs one trial of that schedule as a pure deterministic timed
state machine: ``IDLE -> BLACK -> [FIXATION_CROSS] -> CONTENT -> ...``, ending in
``COMPLETE``. Its only inputs are an explicit ``time_ns``, the captured
configuration, and one prepared schedule per trial -- it samples nothing, and it
reads no clock, no cursor, no microphone, no decoder output, and no renderer. A
phase advances because time reached the instant its half-open window ends, so a
caller polling at 10 Hz and one polling at 1 kHz cross the same boundaries, a
caller that stopped polling for a whole trial crosses every boundary of it in one
step, and repeating a timestamp emits nothing at all.

The machine runs the timeline ``build_timeline`` produced, so the boundaries a
caller can inspect and the boundaries the machine takes are the same values. A
trial with no cross has no ``FIXATION_CROSS`` state, no cross marker, no cross
request, and no cross entry in the recorded timeline; ``CONTENT`` begins exactly
at ``BLACK``'s end. There is no inter-trial state: the next trial begins at the
instant the previous one completes, so its ``BLACK`` is the blank period between
them, and a configuration carrying a nonzero ``inter_trial_ns`` is refused rather
than silently ignored.

Each phase onset issues a ``PresentationRequest`` carrying the instant the
paradigm *decided* to present, never the instant anything appeared; the machine
produces no ``PresentationOutcome`` and cannot, because only a presenter can
report a software presentation observation point, and that point is not physical
display onset. A headless run is evidence about semantic timing and carries none
about a monitor, a vsync, or a photodiode.

There is no clock, no renderer, no microphone, no audio capture, and no decoder
in this module. Nothing here presents anything and nothing here decodes speech.
Importing it does not load the native extension until a name is used.
"""

from __future__ import annotations

_NATIVE_EXPORTS = {
    "BLACK_DRAW",
    "CONTENT_DRAW",
    "CROSS_DRAW",
    "MAX_SPEECH_EXPLICIT_TRIALS",
    "MAX_SPEECH_PHASES",
    "MAX_SPEECH_STIMULI",
    "MAX_SPEECH_TEXT_BYTES",
    "MAX_STEP_EVENTS",
    "MAX_STEP_REQUESTS",
    "MAX_STEP_TRANSITIONS",
    "STIMULUS_ORDER_STREAM",
    "SpeechCatalog",
    "SpeechCause",
    "SpeechContentKind",
    "SpeechCueConfig",
    "SpeechMachine",
    "SpeechMarker",
    "SpeechPhase",
    "SpeechPhaseInterval",
    "SpeechReason",
    "SpeechScheduleKind",
    "SpeechSnapshot",
    "SpeechState",
    "SpeechStepResult",
    "SpeechStimulus",
    "SpeechTimeline",
    "SpeechTrial",
    "SpeechTrialDraws",
    "SpeechTrialSchedule",
    "StimulusOrderPolicy",
    "TRIAL_TIMING_STREAM",
    "UNSET_SPEECH_LABEL",
    "build_timeline",
    "catalog_fingerprint",
    "configuration_fingerprint",
    "find_stimulus",
    "make_schedule_draws",
    "prepare_trial",
    "replay_authority",
    "schedule_identity",
    "select_stimulus",
    "speech_content_kind_declared",
    "speech_cue_of",
    "speech_phase_declared",
    "speech_phase_of",
    "speech_schedule_kind_declared",
    "speech_state_declared",
    "speech_state_is_phase",
    "stimulus_order_policy_declared",
    "trial_duration",
    "validate",
    "validate_against",
}


def __getattr__(name: str) -> object:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from neurale._native_loader import load_native_namespace

    value = getattr(load_native_namespace("experiments.speech"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS)


__all__ = sorted(_NATIVE_EXPORTS)
