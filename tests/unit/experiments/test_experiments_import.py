#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest
from _subprocess_probe import probe_json

# Names whose presence would mean the paradigm layer had grown the generic
# runtime it exists in order not to have: a base class, a registry, a factory,
# or a shared configuration type every paradigm would have to satisfy.
_FORBIDDEN_FRAMEWORK_NAMES = (
    "Experiment",
    "ExperimentBase",
    "ExperimentConfig",
    "Paradigm",
    "ParadigmBase",
    "ParadigmType",
    "ParadigmCreate",
    "ParadigmDestroy",
    "ParadigmDevice",
    "Task",
    "TaskBase",
    "TaskConfig",
    "TaskRegistry",
    "create_experiment",
    "create_paradigm",
    "create_task",
    "register_paradigm",
    "register_task",
)

# Names whose presence would mean shared assistance had grown the controller
# framework rule 7 exists in order not to have: a base class, a registry, a
# factory, or a string-keyed lookup standing in for one.
_FORBIDDEN_CONTROLLER_NAMES = (
    "Controller",
    "ControllerBase",
    "ControllerConfig",
    "ControllerRegistry",
    "AssistanceBase",
    "AssistanceRegistry",
    "SharedControl",
    "SharedController",
    "create_assistance",
    "create_controller",
    "make_assistance",
    "register_assistance",
    "register_controller",
)

# The offline analysis containers. The paradigm layer reuses them; it does not
# restate them.
_OFFLINE_DATA_MODEL_NAMES = ("Event", "EventSeries", "Trial", "TrialTable")

# Concrete native orchestration exists, but it stays in the uninstalled
# ``neurale_execution`` target.  Exposing one of these names from
# Python would turn a benchmark/test integration detail into a public runtime.
_PRIVATE_INTEGRATION_NAMES = (
    "CenterOutController",
    "CenterOutControllerConfig",
    "CenterOutControlTrace",
    "WebGridHeadlessController",
    "WebGridControllerConfig",
    "WebGridHeadlessTrace",
    "SpeechHeadlessScheduler",
    "SpeechHeadlessTrace",
    "PreparedPresentation",
    "ExperimentSession",
    "CenterOutTraceWriter",
    "WebGridTraceWriter",
    "SpeechTraceWriter",
)


def test_experiments_import_is_lightweight() -> None:
    script = """
import json
import sys
loaded_before = set(sys.modules)
import neurale.experiments as experiments
newly_loaded = set(sys.modules) - loaded_before
print(json.dumps({
    "native": "neurale._native" in sys.modules,
    "streaming": any(name.startswith("neurale.streaming") for name in sys.modules),
    "recording": any(name.startswith("neurale.recording") for name in sys.modules),
    "devices": any(name.startswith("neurale.devices") for name in sys.modules),
    "decoding": any(name.startswith("neurale.decoding") for name in sys.modules),
    "data": any(name.startswith("neurale.data") for name in sys.modules),
    "heavy": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "OpenGL", "PIL", "glfw", "matplotlib", "numcodecs", "numpy", "pygame",
        "serial", "tkinter", "zarr",
    }),
    "exports_present": all(
        hasattr(experiments, name) for name in ("TrialIdentity", "CueKind", "CommandRequest")
    ),
}))
"""
    assert probe_json(script) == {
        "native": False,
        "streaming": False,
        "recording": False,
        "devices": False,
        "decoding": False,
        "data": False,
        "heavy": [],
        "exports_present": True,
    }


def test_traceability_import_reaches_no_runtime_or_io() -> None:
    script = """
import json
import sys
loaded_before = set(sys.modules)
import neurale.experiments.traceability as traceability
newly_loaded = set(sys.modules) - loaded_before
import neurale.experiments as experiments
print(json.dumps({
    "native": "neurale._native" in sys.modules,
    "streaming": any(name.startswith("neurale.streaming") for name in sys.modules),
    "recording": any(name.startswith("neurale.recording") for name in sys.modules),
    "io": any(name.startswith("neurale.io") for name in sys.modules),
    "heavy": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "numpy", "scipy", "zarr",
    }),
    "attribute": experiments.traceability is traceability,
    "exports_present": all(
        hasattr(traceability, name)
        for name in (
            "CenterOutProvenanceIndex", "WebGridProvenanceIndex", "SpeechProvenanceIndex"
        )
    ),
}))
"""
    assert probe_json(script) == {
        "native": False,
        "streaming": False,
        "recording": False,
        "io": False,
        "heavy": [],
        "attribute": True,
        "exports_present": True,
    }


def test_experiments_defines_no_generic_paradigm_runtime() -> None:
    import neurale.experiments as experiments

    exported = set(experiments.__all__)
    for name in _FORBIDDEN_FRAMEWORK_NAMES:
        assert name not in exported
        assert not hasattr(experiments, name)


def test_private_native_integrations_are_not_python_api() -> None:
    import neurale.experiments as experiments
    from neurale.experiments import center_out, speech, webgrid

    for module in (experiments, center_out, webgrid, speech):
        exported = set(module.__all__)
        for name in _PRIVATE_INTEGRATION_NAMES:
            assert name not in exported
            assert not hasattr(module, name)


def test_experiments_does_not_duplicate_offline_model() -> None:
    import neurale.data as data
    import neurale.experiments as experiments

    exported = set(experiments.__all__)
    for name in _OFFLINE_DATA_MODEL_NAMES:
        assert hasattr(data, name)
        assert name not in exported
        assert not hasattr(experiments, name)

    # The runtime records are named apart from the offline ones on purpose, so
    # that no import site can pick up one while meaning the other.
    assert "TrialRecord" in exported
    assert "ExperimentEvent" in exported
    assert not hasattr(data, "TrialRecord")
    assert not hasattr(data, "ExperimentEvent")


def test_assistance_import_is_lightweight_and_reaches_no_runtime() -> None:
    script = """
import json
import sys
loaded_before = set(sys.modules)
import neurale.experiments.assistance as assistance
newly_loaded = set(sys.modules) - loaded_before
import neurale.experiments as experiments
print(json.dumps({
    "native": "neurale._native" in sys.modules,
    "streaming": any(name.startswith("neurale.streaming") for name in sys.modules),
    "decoding": any(name.startswith("neurale.decoding") for name in sys.modules),
    "devices": any(name.startswith("neurale.devices") for name in sys.modules),
    "heavy": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "OpenGL", "PIL", "glfw", "matplotlib", "numpy", "pygame", "serial", "tkinter",
    }),
    # Reachable as an attribute of the package as well as by its own import.
    "attribute": experiments.assistance is assistance,
    "exports_present": all(
        hasattr(assistance, name)
        for name in ("blend_velocity", "apply_ortho_impedance", "VelocityVector")
    ),
}))
"""
    assert probe_json(script) == {
        "native": False,
        "streaming": False,
        "decoding": False,
        "devices": False,
        "heavy": [],
        "attribute": True,
        "exports_present": True,
    }


def test_assistance_defines_no_controller_framework() -> None:
    import importlib

    import neurale.experiments.assistance as assistance

    exported = set(assistance.__all__)
    for name in _FORBIDDEN_CONTROLLER_NAMES:
        assert name not in exported
        assert not hasattr(assistance, name)

    # Rule 7: the shared capability is a small amount of velocity shared-control
    # mathematics, which is not enough to form a standalone domain.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("neurale.control")


def test_center_out_import_is_lightweight_and_reaches_no_runtime() -> None:
    script = """
import json
import sys
loaded_before = set(sys.modules)
import neurale.experiments.center_out as center_out
newly_loaded = set(sys.modules) - loaded_before
import neurale.experiments as experiments
print(json.dumps({
    "native": "neurale._native" in sys.modules,
    "streaming": any(name.startswith("neurale.streaming") for name in sys.modules),
    "recording": any(name.startswith("neurale.recording") for name in sys.modules),
    "decoding": any(name.startswith("neurale.decoding") for name in sys.modules),
    "devices": any(name.startswith("neurale.devices") for name in sys.modules),
    "heavy": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "OpenGL", "PIL", "glfw", "matplotlib", "numpy", "pygame", "serial", "tkinter",
    }),
    "attribute": experiments.center_out is center_out,
    "exports_present": all(
        hasattr(center_out, name)
        for name in ("CenterOutTask", "contains_cursor", "select_outward_target")
    ),
}))
"""
    assert probe_json(script) == {
        "native": False,
        "streaming": False,
        "recording": False,
        "decoding": False,
        "devices": False,
        "heavy": [],
        "attribute": True,
        "exports_present": True,
    }


def test_webgrid_import_reaches_no_runtime_or_ui() -> None:
    script = """
import json
import sys
loaded_before = set(sys.modules)
import neurale.experiments.webgrid as webgrid
newly_loaded = set(sys.modules) - loaded_before
import neurale.experiments as experiments
print(json.dumps({
    "native": "neurale._native" in sys.modules,
    "streaming": any(name.startswith("neurale.streaming") for name in sys.modules),
    "recording": any(name.startswith("neurale.recording") for name in sys.modules),
    "devices": any(name.startswith("neurale.devices") for name in sys.modules),
    "ui": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "OpenGL", "PIL", "glfw", "matplotlib", "pygame", "selenium", "tkinter",
    }),
    "attribute": experiments.webgrid is webgrid,
    "exports_present": all(
        hasattr(webgrid, name)
        for name in (
            "WebGridConfig", "WebGridMachine", "WebGridMetrics", "TaskBounds",
            "locate_cell", "select_target", "summarize",
        )
    ),
}))
"""
    assert probe_json(script) == {
        "native": False,
        "streaming": False,
        "recording": False,
        "devices": False,
        "ui": [],
        "attribute": True,
        "exports_present": True,
    }


def test_webgrid_metric_version_is_v1() -> None:
    import neurale.experiments.webgrid as webgrid

    assert webgrid.METRIC_VERSION_1 == 1
    assert webgrid.CURRENT_METRIC_VERSION == webgrid.METRIC_VERSION_1


def test_speech_import_reaches_no_renderer_or_audio() -> None:
    script = """
import json
import sys
loaded_before = set(sys.modules)
import neurale.experiments.speech as speech
newly_loaded = set(sys.modules) - loaded_before
import neurale.experiments as experiments
print(json.dumps({
    "native": "neurale._native" in sys.modules,
    "streaming": any(name.startswith("neurale.streaming") for name in sys.modules),
    "recording": any(name.startswith("neurale.recording") for name in sys.modules),
    "devices": any(name.startswith("neurale.devices") for name in sys.modules),
    "media": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "OpenGL", "PIL", "glfw", "matplotlib", "pyaudio", "pygame", "sounddevice",
        "soundfile", "tkinter", "wave",
    }),
    "attribute": experiments.speech is speech,
    "exports_present": all(
        hasattr(speech, name)
        for name in (
            "SpeechCueConfig", "SpeechCatalog", "SpeechTrialSchedule",
            "prepare_trial", "build_timeline", "select_stimulus",
            "SpeechMachine", "SpeechState", "SpeechStepResult", "SpeechTrial",
        )
    ),
}))
"""
    assert probe_json(script) == {
        "native": False,
        "streaming": False,
        "recording": False,
        "devices": False,
        "media": [],
        "attribute": True,
        "exports_present": True,
    }


def test_speech_content_kind_is_text_only() -> None:
    import neurale.experiments.speech as speech

    assert set(speech.SpeechContentKind.__members__) == {"UNSPECIFIED", "TEXT"}


def test_speech_machine_reads_no_clock_or_renderer() -> None:
    script = """
import json
import sys
loaded_before = set(sys.modules)
from neurale.experiments.speech import SpeechMachine, SpeechState
machine = SpeechMachine()
newly_loaded = set(sys.modules) - loaded_before
print(json.dumps({
    "idle": machine.state == SpeechState.IDLE,
    "paradigm": machine.paradigm,
    "streaming": any(name.startswith("neurale.streaming") for name in sys.modules),
    "recording": any(name.startswith("neurale.recording") for name in sys.modules),
    "devices": any(name.startswith("neurale.devices") for name in sys.modules),
    "media": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "OpenGL", "PIL", "asyncio", "glfw", "matplotlib", "pyaudio", "pygame",
        "sched", "sounddevice", "soundfile", "threading", "tkinter", "wave",
    }),
}))
"""
    # Constructing and holding a machine pulls in no clock, no timer, no thread,
    # no renderer, and no capture backend, because it reads none of them.
    assert probe_json(script) == {
        "idle": True,
        "paradigm": 0,
        "streaming": False,
        "recording": False,
        "devices": False,
        "media": [],
    }


def test_center_out_is_not_shared_contract_name() -> None:
    import neurale.experiments as experiments

    # One paradigm's configuration and machine must not look like part of the
    # vocabulary every paradigm shares. They are reachable as the submodule and
    # by nothing else.
    for name in (
        "CenterOutTask",
        "CenterOutProtocol",
        "CenterOutSession",
        "CenterOut2DLayout",
        "TargetSelectionPolicy",
        "CenterOutMachine",
        "CenterOutState",
        "CenterOutGuidance",
        "CenterOutGuidanceConfig",
        "SpeechCueConfig",
        "SpeechCatalog",
        "SpeechStimulus",
        "StimulusOrderPolicy",
        "SpeechMachine",
        "SpeechState",
        "SpeechMarker",
        "SpeechCause",
    ):
        assert name not in set(experiments.__all__)
        assert not hasattr(experiments, name)


def test_unknown_experiment_attribute_raises_attribute_error() -> None:
    import neurale.experiments as experiments

    try:
        experiments.NotAContractValue  # noqa: B018
    except AttributeError as error:
        assert "NotAContractValue" in str(error)
    else:  # pragma: no cover - the attribute must not resolve
        raise AssertionError("an unknown contract name must not resolve")
