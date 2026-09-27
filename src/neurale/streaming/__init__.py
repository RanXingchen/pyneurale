#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Python control plane for the native streaming data plane."""

from ._config import (
    RealtimeConfig,
    RealtimePlatformConfig,
    RealtimeSchedulingPolicy,
    RealtimeThreadConfig,
)
from .runner import ExecutionProfile, StreamRunner

_NATIVE_EXPORTS = {
    "ArrayReplaySource",
    "NativeResultSink",
    "CountingNativeConsumer",
    "CountingNativeObserver",
    "ClockSyncFlags",
    "ClockSyncSnapshot",
    "DeviceTickTracking",
    "FeatureSetDescriptor",
    "FeatureSetDescriptorRegistry",
    "FeatureTimestampReference",
    "FaultCode",
    "FaultRecord",
    "FaultStage",
    "GapReason",
    "IdentityNativeProcessor",
    "ObserverDropPolicy",
    "ObservationTiming",
    "ObserverDropRange",
    "ObserverEdgeConfig",
    "ObserverEdgeStats",
    "PhysicalUnit",
    "PythonObserverBridge",
    "PythonObserverBridgeStats",
    "PythonObserverDiscontinuity",
    "PythonObserverFrame",
    "PythonObserverSignalBlock",
    "PythonObserverSignalGap",
    "PythonProcessorAdapter",
    "PythonSinkAdapter",
    "PythonSourceAdapter",
    "RationalRate",
    "RealtimeApplyResult",
    "RealtimeConfigurationStatus",
    "RealtimeFeature",
    "RealtimePlatformCapabilities",
    "RecordingNativeSafetyController",
    "RuntimeHeartbeat",
    "RuntimeState",
    "RuntimeStats",
    "SafetyReason",
    "SignalDType",
    "SignalLayout",
    "SignalKind",
    "SignalSchema",
    "StreamSchema",
    "StreamStatus",
    "SyntheticNativeSource",
    "UnitDescriptor",
    "UnitRegistry",
    "realtime_platform_capabilities",
}

_CONVERSION_EXPORTS = {
    "feature_matrix_from_frame",
    "feature_matrix_to_frame",
}


def __getattr__(name: str) -> object:
    if name in _CONVERSION_EXPORTS:
        from . import _feature_matrix

        value = getattr(_feature_matrix, name)
        globals()[name] = value
        return value
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import _native

    value = getattr(_native, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS | _CONVERSION_EXPORTS)


__all__ = [
    "ArrayReplaySource",
    "ClockSyncFlags",
    "ClockSyncSnapshot",
    "CountingNativeConsumer",
    "CountingNativeObserver",
    "DeviceTickTracking",
    "ExecutionProfile",
    "FaultCode",
    "FaultRecord",
    "FaultStage",
    "FeatureSetDescriptor",
    "FeatureSetDescriptorRegistry",
    "FeatureTimestampReference",
    "GapReason",
    "IdentityNativeProcessor",
    "NativeResultSink",
    "ObservationTiming",
    "ObserverDropPolicy",
    "ObserverDropRange",
    "ObserverEdgeConfig",
    "ObserverEdgeStats",
    "PhysicalUnit",
    "PythonObserverBridge",
    "PythonObserverBridgeStats",
    "PythonObserverDiscontinuity",
    "PythonObserverFrame",
    "PythonObserverSignalBlock",
    "PythonObserverSignalGap",
    "PythonProcessorAdapter",
    "PythonSinkAdapter",
    "PythonSourceAdapter",
    "RationalRate",
    "RealtimeApplyResult",
    "RealtimeConfig",
    "RealtimeConfigurationStatus",
    "RealtimeFeature",
    "RealtimePlatformCapabilities",
    "RealtimePlatformConfig",
    "RealtimeSchedulingPolicy",
    "RealtimeThreadConfig",
    "RecordingNativeSafetyController",
    "RuntimeHeartbeat",
    "RuntimeState",
    "RuntimeStats",
    "SafetyReason",
    "SignalDType",
    "SignalKind",
    "SignalLayout",
    "SignalSchema",
    "StreamRunner",
    "StreamSchema",
    "StreamStatus",
    "SyntheticNativeSource",
    "UnitDescriptor",
    "UnitRegistry",
    "feature_matrix_from_frame",
    "feature_matrix_to_frame",
    "realtime_platform_capabilities",
]
