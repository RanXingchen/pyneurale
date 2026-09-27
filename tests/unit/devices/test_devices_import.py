#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from _subprocess_probe import probe_json


def test_devices_import_is_lightweight() -> None:
    script = """
import importlib.util
import json
import sys
loaded_before = set(sys.modules)
import neurale.devices as devices
newly_loaded = set(sys.modules) - loaded_before
top_level = {
    "public": devices.__all__,
    "native": "neurale._native" in sys.modules,
    "runtime": any(name.startswith("neurale.runtime") for name in sys.modules),
    "signal": any(name.startswith("neurale.signal") for name in sys.modules),
    "simulation_loaded": "neurale.devices.simulation" in sys.modules,
    "top_level_simulation": importlib.util.find_spec("neurale.simulation") is not None,
    "vendor": sorted(name for name in newly_loaded if name.split(".")[0] in {
        "bleak", "bluetooth", "serial", "usb"
    }),
    "generic_framework": sorted(name for name in (
        "Device", "DeviceCapabilities", "DeviceEndpoint", "DeviceHealth",
        "DeviceIdentity", "DeviceState"
    ) if hasattr(devices, name)),
    "simulator_controls": sorted(name for name in (
        "AcquisitionEvent", "BoundedStall", "DeviceClockRestart",
        "DeviceTickJump", "Disconnect", "KnownSampleLoss", "ManualHostClock",
        "SimulatedNeuralDevice", "SimulationFaultPlan", "SimulationTimingConfig",
        "SourceFault", "TransientWouldBlock"
    ) if hasattr(devices, name)),
}
import neurale.devices.simulation as simulation
print(json.dumps({
    "top_level": top_level,
    "simulation": {
        "device": simulation.SimulatedNeuralDevice.__name__,
        "host_clock": simulation.ManualHostClock.__name__,
        "old_clock": hasattr(simulation, "ManualDeviceClock"),
        "native": "neurale._native" in sys.modules,
        "runtime": any(name.startswith("neurale.runtime") for name in sys.modules),
        "signal": any(name.startswith("neurale.signal") for name in sys.modules),
        "vendor": sorted(name for name in set(sys.modules) - loaded_before
                         if name.split(".")[0] in {"bleak", "bluetooth", "serial", "usb"}),
    },
}))
"""
    loaded = probe_json(script)
    assert loaded == {
        "top_level": {
            "public": ["Device", "NativeProvider", "PythonProvider", "Signal", "list", "open"],
            "native": False,
            "runtime": False,
            "signal": False,
            "simulation_loaded": False,
            "top_level_simulation": False,
            "vendor": [],
            "generic_framework": ["Device"],
            "simulator_controls": [],
        },
        "simulation": {
            "device": "SimulatedNeuralDevice",
            "host_clock": "ManualHostClock",
            "old_clock": False,
            "native": False,
            "runtime": False,
            "signal": False,
            "vendor": [],
        },
    }


def test_simulator_api_is_owned_by_simulation_submodule() -> None:
    import neurale.devices as devices
    from neurale.devices import simulation

    expected = {
        "AcquisitionEvent",
        "AppliedNeuralControl",
        "BoundedStall",
        "DeviceClockRestart",
        "DeviceTickJump",
        "Disconnect",
        "KnownSampleLoss",
        "IntentDrivenNeuralDevice",
        "ManualHostClock",
        "NeuralDriftSchedule",
        "SimulationFaultPlan",
        "SimulatedNeuralDevice",
        "SimulationTimingConfig",
        "SourceFault",
        "TransientWouldBlock",
    }
    assert set(simulation.__all__) == expected
    assert all(hasattr(simulation, name) for name in expected)
    assert not hasattr(simulation, "ManualDeviceClock")
    assert all(not hasattr(devices, name) for name in expected)
