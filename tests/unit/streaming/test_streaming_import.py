#!/usr/bin/env python3

from _subprocess_probe import probe_json


def test_streaming_import_has_no_unrelated_side_effects() -> None:
    code = """
import importlib
import json
import sys

streaming = importlib.import_module("neurale.streaming")
forbidden = (
    "neurale._native",
    "neurale._native_cuda",
    "neurale.io",
    "neurale.devices",
    "neurale.visualization",
    "neurale.pipeline",
    "neurale.signal",
    "neurale.features",
    "neurale.models",
    "neurale.decoding",
    "neurale.data",
    "numpy",
    "torch",
    "PySide6",
    "PyQt6",
)
print(json.dumps({
    "exports": sorted(streaming.__all__),
    "loaded_forbidden": [name for name in forbidden if name in sys.modules],
}))
"""
    result = probe_json(code)

    assert result["exports"] == [
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
    assert result["loaded_forbidden"] == []


def test_top_level_import_does_not_reexport_streaming() -> None:
    code = """
import json
import neurale

print(json.dumps({
    "exports": neurale.__all__,
    "has_streaming": "streaming" in vars(neurale),
}))
"""
    result = probe_json(code)

    assert result == {"exports": ["__version__"], "has_streaming": False}


def test_pipeline_is_a_lazy_public_python_package() -> None:
    code = """
import importlib.util
import json
import neurale

print(json.dumps({
    "module_spec": importlib.util.find_spec("neurale.pipeline") is not None,
    "top_level_export": hasattr(neurale, "pipeline"),
}))
"""
    assert probe_json(code) == {
        "module_spec": True,
        "top_level_export": False,
    }
