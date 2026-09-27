#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Private bindings for the native streaming data plane."""

from neurale._native_loader import load_native_namespace

_bindings = load_native_namespace("streaming")

ArrayReplaySource = _bindings.ArrayReplaySource
NativeResultSink = _bindings.NativeResultSink

CountingNativeConsumer = _bindings.CountingNativeConsumer
CountingNativeObserver = _bindings.CountingNativeObserver
ClockSyncFlags = _bindings.ClockSyncFlags
ClockSyncSnapshot = _bindings.ClockSyncSnapshot
DeviceTickTracking = _bindings.DeviceTickTracking
FeatureSetDescriptor = _bindings.FeatureSetDescriptor
FeatureSetDescriptorRegistry = _bindings.FeatureSetDescriptorRegistry
FeatureTimestampReference = _bindings.FeatureTimestampReference
FaultCode = _bindings.FaultCode
FaultRecord = _bindings.FaultRecord
FaultStage = _bindings.FaultStage
GapReason = _bindings.GapReason
IdentityNativeProcessor = _bindings.IdentityNativeProcessor
ObserverDropPolicy = _bindings.ObserverDropPolicy
ObservationTiming = _bindings.ObservationTiming
ObserverDropRange = _bindings.ObserverDropRange
ObserverEdgeConfig = _bindings.ObserverEdgeConfig
ObserverEdgeStats = _bindings.ObserverEdgeStats
PoolCapacityBudget = _bindings.PoolCapacityBudget
PhysicalUnit = _bindings.PhysicalUnit
PythonObserverBridge = _bindings.PythonObserverBridge
PythonObserverBridgeStats = _bindings.PythonObserverBridgeStats
PythonObserverDiscontinuity = _bindings.PythonObserverDiscontinuity
PythonObserverFrame = _bindings.PythonObserverFrame
PythonObserverSignalBlock = _bindings.PythonObserverSignalBlock
PythonObserverSignalGap = _bindings.PythonObserverSignalGap
PythonProcessorAdapter = _bindings.PythonProcessorAdapter
PythonSinkAdapter = _bindings.PythonSinkAdapter
PythonSourceAdapter = _bindings.PythonSourceAdapter
RationalRate = _bindings.RationalRate
RealtimeApplyResult = _bindings.RealtimeApplyResult
RealtimeConfig = _bindings.RealtimeConfig
RealtimeConfigMode = _bindings.RealtimeConfigMode
RealtimeConfigurationStatus = _bindings.RealtimeConfigurationStatus
RealtimeFeature = _bindings.RealtimeFeature
RealtimePlatformCapabilities = _bindings.RealtimePlatformCapabilities
RealtimePlatformConfig = _bindings.RealtimePlatformConfig
RealtimeSchedulingPolicy = _bindings.RealtimeSchedulingPolicy
RealtimeThreadConfig = _bindings.RealtimeThreadConfig
RecordingNativeSafetyController = _bindings.RecordingNativeSafetyController
RuntimeHeartbeat = _bindings.RuntimeHeartbeat
RuntimeState = _bindings.RuntimeState
RuntimeStats = _bindings.RuntimeStats
SafetyReason = _bindings.SafetyReason
SignalDType = _bindings.SignalDType
SignalLayout = _bindings.SignalLayout
SignalKind = _bindings.SignalKind
SignalSchema = _bindings.SignalSchema
StreamSchema = _bindings.StreamSchema
StreamStatus = _bindings.StreamStatus
SyntheticNativeSource = _bindings.SyntheticNativeSource
UnitDescriptor = _bindings.UnitDescriptor
UnitRegistry = _bindings.UnitRegistry
_NativeStreamRunner = _bindings._NativeStreamRunner
realtime_platform_capabilities = _bindings.realtime_platform_capabilities
