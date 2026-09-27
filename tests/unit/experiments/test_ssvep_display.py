# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
from types import SimpleNamespace

import pytest
from _subprocess_probe import probe_json

from neurale.experiments.presentation import SSVEPDisplay, SSVEPDisplayConfig


def test_display_import_does_not_load_native():
    result = probe_json("""
import json, sys
from neurale.experiments.presentation import SSVEPDisplay, SSVEPDisplayConfig
SSVEPDisplayConfig()
print(json.dumps({'loaded': 'neurale._native' in sys.modules}))
""")
    assert result == {"loaded": False}


@pytest.mark.parametrize(
    "kwargs",
    [dict(window_size=(0, 800)), dict(monitor=-1), dict(fullscreen=1), dict(target_size=0)],
)
def test_config_rejects_invalid_window_values(kwargs):
    with pytest.raises((ValueError, TypeError)):
        SSVEPDisplayConfig(**kwargs)


def fake_display(events, *, pump_status=0):
    display = object.__new__(SSVEPDisplay)
    display._config = SSVEPDisplayConfig()
    display._should_close = False
    display._native = SimpleNamespace(
        PresentationRuntimeStatus=SimpleNamespace(OK=0, WINDOW_CLOSED=1)
    )
    events = iter([*events, (0, False, False)])
    display._impl = SimpleNamespace(
        pump_events=lambda: pump_status, poll_input=lambda: next(events)
    )
    return display


def test_poll_handles_escape():
    display = fake_display([(0, True, True)])
    assert display.poll() is None
    assert display.should_close
    assert display.poll() is None


def test_poll_handles_window_close():
    display = fake_display([], pump_status=1)
    assert display.poll() is None and display.should_close


def test_pump_failure_is_not_ignored():
    with pytest.raises(RuntimeError, match="presentation failed"):
        fake_display([], pump_status=2).poll()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(target_size=float("nan")),
        dict(target_size=True),
        dict(positions=[]),
        dict(positions=[(0,)]),
        dict(positions=[(2, 0)]),
        dict(positions=[(float("nan"), 0)]),
    ],
)
def test_invalid_geometry(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SSVEPDisplayConfig(**kwargs)


def test_geometry_is_immutable_and_display_has_no_clock_parameters():
    import inspect

    points = [[-0.5, 0.5], [0.5, -0.5]]
    config = SSVEPDisplayConfig(positions=points, target_size=0.3)
    points[0][0] = 0.1
    assert config.positions == ((-0.5, 0.5), (0.5, -0.5))
    assert "renderer_origin_ns" not in inspect.signature(SSVEPDisplay).parameters
    assert "experiment_origin_ns" not in inspect.signature(SSVEPDisplay).parameters
    with pytest.raises(TypeError):
        SSVEPDisplayConfig(keyboard_selection=True)
    display = fake_display([])
    assert display.poll() is None
