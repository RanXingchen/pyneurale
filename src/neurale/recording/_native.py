#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Private bindings for the native critical recorder.

Importing this module loads the native extension, which is why nothing in
:mod:`neurale.recording` imports it at module scope: ``import neurale.recording``
must keep working on a build with no native extension, because the Python
recorder does not need one. :mod:`neurale.recording._native_recorder` imports
this lazily, inside the one call that first needs a native object.
"""

from neurale._native_loader import load_native_namespace

_bindings = load_native_namespace("recording")

CaptureOutcome = _bindings.CaptureOutcome
CompletenessVerdict = _bindings.CompletenessVerdict
DurabilityPolicy = _bindings.DurabilityPolicy
EffectiveSessionOutcome = _bindings.EffectiveSessionOutcome
FaultOrigin = _bindings.FaultOrigin
FinalizationStatus = _bindings.FinalizationStatus
NativeRecorderStatus = _bindings.NativeRecorderStatus
NativeRecordingPlan = _bindings.NativeRecordingPlan
PlannedSignalRecording = _bindings.PlannedSignalRecording
PositionTag = _bindings.PositionTag
ProducerIdentityKind = _bindings.ProducerIdentityKind
RecorderFaultReason = _bindings.RecorderFaultReason
RecorderFinalizationAttempt = _bindings.RecorderFinalizationAttempt
RecorderFirstPosition = _bindings.RecorderFirstPosition
RecorderLifecycleState = _bindings.RecorderLifecycleState
RecorderPrimaryFault = _bindings.RecorderPrimaryFault
RecorderStatusCode = _bindings.RecorderStatusCode
RecordingPlanStatus = _bindings.RecordingPlanStatus
RecoverabilityAnswer = _bindings.RecoverabilityAnswer
RequestedTerminalIntent = _bindings.RequestedTerminalIntent
SpoolBackend = _bindings.SpoolBackend
_NativeRecorder = _bindings._NativeRecorder
