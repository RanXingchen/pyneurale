#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _subprocess_probe import probe_json

from neurale.exceptions import DependencyError, NativeExtensionError, NativeUnavailableError


def _require_presentation() -> None:
    from neurale.experiments.presentation._native import load_presentation_extension

    try:
        load_presentation_extension()
    except DependencyError as exc:
        if isinstance(exc.__cause__, NativeUnavailableError):
            pytest.skip("this installation was built with experiment presentation disabled")
        raise


def test_core_and_experiment_imports_do_not_load_presentation() -> None:
    script = """
import json
import sys
import neurale
import neurale.experiments
print(json.dumps({
    "presentation_native": "neurale._native" in sys.modules,
    "graphics_python": sorted(name for name in sys.modules if name.split(".")[0] in {
        "OpenGL", "freetype", "glfw", "uharfbuzz"
    }),
}))
"""
    assert probe_json(script) == {
        "presentation_native": False,
        "graphics_python": [],
    }


def test_presentation_python_namespace_is_lazy() -> None:
    script = """
import json
import sys
import neurale.experiments.presentation as presentation
print(json.dumps({
    "presentation_native": "neurale._native" in sys.modules,
    "has_entry_point": callable(presentation.dependency_versions),
}))
"""
    assert probe_json(script) == {
        "presentation_native": False,
        "has_entry_point": True,
    }


def test_native_exports_are_discoverable_before_first_access() -> None:
    _require_presentation()
    script = """
import inspect
import json
import sys
import neurale.experiments.presentation as presentation
names = dir(presentation)
native_before = "neurale._native" in sys.modules
members = dict(inspect.getmembers(presentation))
print(json.dumps({
    "native_before": native_before,
    "native_after": "neurale._native" in sys.modules,
    "listed": "AspectPolicy" in names,
    "resolved": "AspectPolicy" in members,
}))
"""
    assert probe_json(script) == {
        "native_before": False,
        "native_after": True,
        "listed": True,
        "resolved": True,
    }


def test_headless_presentation_introspection_does_not_load_native() -> None:
    script = """
import inspect
import json
import sys
import neurale.experiments.presentation as presentation
presentation._native_available = lambda: False
members = dict(inspect.getmembers(presentation))
print(json.dumps({
    "presentation_native": "neurale._native" in sys.modules,
    "dependency_versions": "dependency_versions" in members,
    "center_out_config": "CenterOutPresentationConfig" in members,
    "center_out_presenter": "CenterOutPresenter" in members,
}))
"""
    assert probe_json(script) == {
        "presentation_native": False,
        "dependency_versions": True,
        "center_out_config": True,
        "center_out_presenter": True,
    }


def test_namespace_lists_only_configuration_and_presenters() -> None:
    import neurale.experiments.presentation as presentation

    assert {
        "CenterOutPresenter",
        "CenterOutPresentationTheme",
        "WebGridPresenter",
        "SpeechPresenter",
        "CenterOutPresentationControlEvent",
        "SpeechPresentationControlEvent",
        "SpeechPresentationStyle",
        "PresentationRuntimeStatus",
    } <= set(presentation.__all__)
    assert "PresentationStatus" not in presentation.__all__
    assert "PresentationSurface" not in presentation.__all__
    assert "Renderer" not in presentation.__all__
    assert "Presenter" not in presentation.__all__
    assert "CenterOutPresentationStyle" not in presentation.__all__


def test_enabled_public_configuration_is_read_only() -> None:
    from dataclasses import FrozenInstanceError
    from inspect import signature

    from neurale.experiments.presentation import CenterOutPresentationConfig

    config = CenterOutPresentationConfig()
    assert tuple(signature(CenterOutPresentationConfig).parameters) == (
        "title",
        "window_size",
        "fullscreen",
        "monitor",
        "vsync",
        "theme",
    )
    assert config.window_size == (800, 600)
    assert config.monitor is None
    assert config.vsync is True
    with pytest.raises(FrozenInstanceError):
        config.fullscreen = True


@pytest.mark.parametrize(
    ("values", "error", "message"),
    [
        ({"window_size": (0, 600)}, ValueError, "window_size values must be positive"),
        ({"window_size": [800, 600]}, TypeError, "window_size must be"),
        ({"monitor": -1}, ValueError, "monitor must be non-negative"),
        ({"vsync": 1}, TypeError, "vsync must be a bool"),
        ({"theme": object()}, TypeError, "theme must be"),
    ],
)
def test_center_out_presentation_config_rejects_invalid_values(
    values: dict[str, object], error: type[Exception], message: str
) -> None:
    from neurale.experiments.presentation import CenterOutPresentationConfig

    with pytest.raises(error, match=message):
        CenterOutPresentationConfig(**values)


def test_center_out_presentation_theme_validates_rgba() -> None:
    from neurale.experiments.presentation import CenterOutPresentationTheme

    theme = CenterOutPresentationTheme(cursor=(0, 0.5, 1, 1))
    assert theme.cursor == (0.0, 0.5, 1.0, 1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        CenterOutPresentationTheme(cursor=(0.0, 0.0, 2.0, 1.0))


def test_center_out_presentation_resolves_task_geometry_and_private_runtime_fields() -> None:
    _require_presentation()

    from neurale.experiments import center_out as co
    from neurale.experiments.presentation import (
        CenterOutPresentationConfig,
        CenterOutPresentationTheme,
    )
    from neurale.experiments.presentation._center_out import (
        _resolve_center_out_presentation,
        _with_visibility,
    )

    task = co.CenterOutTask(
        geometry_unit=co.GeometryUnit.NORMALIZED,
        layout=co.build_radial_layout(co.RadialLayoutRequest(radius=0.5)),
        acceptance=co.AcceptanceRegion(0.08, 0.1),
        cursor_extent=0.025,
        movement_timeout_seconds=1.0,
        selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS,
    )
    config = _with_visibility(
        CenterOutPresentationConfig(
            title="resolved",
            window_size=(1024, 768),
            monitor=2,
            vsync=False,
            theme=CenterOutPresentationTheme(cursor=(0.2, 0.4, 0.6, 1.0)),
        ),
        visible=False,
    )
    resolved = _resolve_center_out_presentation(task, config)

    assert resolved.geometry_unit == task.geometry_unit
    assert resolved.logical_space.width == pytest.approx(resolved.logical_space.height)
    assert resolved.logical_space.left < -0.5
    assert resolved.logical_space.left + resolved.logical_space.width > 0.5
    assert resolved.style.target_radius == pytest.approx(0.08)
    assert resolved.style.cursor_radius == pytest.approx(0.025)
    assert resolved.style.circle_segments == 96
    assert resolved.style.cursor.red == pytest.approx(0.2)
    assert (resolved.window_size.width, resolved.window_size.height) == (1024, 768)
    assert resolved.monitor_idx == 2
    assert resolved.swap_interval == 0
    assert resolved.input_capacity == 256
    assert resolved.resizable is False
    assert resolved.visible is False


def test_webgrid_presenter_accepts_misselection_feedback() -> None:
    _require_presentation()

    from neurale.experiments import ContractStatus, PresentationStatus, webgrid
    from neurale.experiments.presentation import (
        PresentationRuntimeStatus,
        WebGridPresentationConfig,
        WebGridPresentationStyle,
        WebGridPresenter,
        renderer_monotonic_now_ns,
    )

    assert PresentationRuntimeStatus.__name__ == "PresentationRuntimeStatus"
    assert PresentationStatus.__name__ == "PresentationStatus"
    assert PresentationRuntimeStatus is not PresentationStatus

    task = webgrid.WebGridConfig(
        rows=1,
        columns=2,
        bounds=webgrid.TaskBounds(0.0, 2.0, 0.0, 1.0),
        candidates=[1, 2],
        schedule=webgrid.TargetScheduleKind.EXPLICIT_SEQUENCE,
        immediate_repetition=webgrid.ImmediateRepetitionPolicy.FORBID,
        correct_selection=webgrid.CorrectSelectionPolicy.ADVANCE_TARGET,
        incorrect_selection=webgrid.IncorrectSelectionPolicy.KEEP_CURRENT_TARGET,
        explicit_targets=[1, 2],
        initial_target=1,
        target_count_limit=2,
        metric_version=webgrid.METRIC_VERSION_1,
    )
    machine = webgrid.WebGridMachine()
    status, started = machine.start(17, task, 0)
    assert status == ContractStatus.OK
    pointer = webgrid.PointerPosition(1.5, 0.5)
    status, selection = webgrid.make_selection_event(
        task,
        pointer,
        started.snapshot.active_target,
        started.snapshot.trial,
        17,
        100,
        1,
    )
    assert status == ContractStatus.OK
    assert not selection.correct
    status, stepped = machine.step(100, pointer, selection)
    assert status == ContractStatus.OK
    assert stepped.selection_processed
    assert not stepped.selection.event.correct

    presenter = WebGridPresenter()
    runtime_status = presenter.open(
        task,
        WebGridPresentationConfig(WebGridPresentationStyle(0.03), swap_interval=0, visible=False),
        renderer_monotonic_now_ns(),
        0,
    )
    unavailable = {
        PresentationRuntimeStatus.GLFW_INITIALIZATION_FAILED,
        PresentationRuntimeStatus.MONITOR_UNAVAILABLE,
        PresentationRuntimeStatus.WINDOW_CREATION_FAILED,
        PresentationRuntimeStatus.OPENGL_FUNCTION_MISSING,
    }
    if runtime_status in unavailable:
        presenter.close()
        pytest.skip(f"presentation environment unavailable: {runtime_status}")
    assert runtime_status == PresentationRuntimeStatus.OK
    try:
        assert presenter.update(stepped.snapshot, pointer) == PresentationRuntimeStatus.OK
        assert (
            presenter.update(stepped.snapshot, pointer, stepped.selection)
            == PresentationRuntimeStatus.OK
        )
        assert presenter.render(100, 100).status == PresentationRuntimeStatus.OK
    finally:
        presenter.close()


def test_missing_optional_native_module_uses_public_dependency_error(monkeypatch) -> None:
    from neurale.experiments.presentation import _native

    _native.load_presentation_extension.cache_clear()

    def missing(_name: str):
        raise NativeUnavailableError("Native namespace 'experiments.presentation' is unavailable.")

    monkeypatch.setattr(_native, "load_native_namespace", missing)
    with pytest.raises(DependencyError, match="NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON"):
        _native.load_presentation_extension()
    _native.load_presentation_extension.cache_clear()


def test_broken_native_dependency_uses_public_error(monkeypatch) -> None:
    from neurale.experiments.presentation import _native

    _native.load_presentation_extension.cache_clear()

    def missing(_name: str):
        raise NativeExtensionError("The native extension has a missing dependency.")

    monkeypatch.setattr(_native, "load_native_namespace", missing)
    with pytest.raises(NativeExtensionError, match="missing dependency"):
        _native.load_presentation_extension()
    _native.load_presentation_extension.cache_clear()


def test_public_dependency_versions_normalizes_native_result(monkeypatch) -> None:
    import neurale.experiments.presentation as presentation
    from neurale.experiments.presentation import _native

    value = {
        "glfw": (3, 4, 0),
        "opengl_minimum": (3, 3),
        "freetype": (2, 13, 3),
        "harfbuzz": (10, 4, 0),
    }
    monkeypatch.setattr(
        _native,
        "load_presentation_extension",
        lambda: SimpleNamespace(dependency_versions=lambda: value),
    )
    assert presentation.dependency_versions() == value


def test_enabled_install_reports_all_native_dependency_versions() -> None:
    _require_presentation()

    import neurale.experiments.presentation as presentation

    versions = presentation.dependency_versions()
    assert set(versions) == {"glfw", "opengl_minimum", "freetype", "harfbuzz"}
    assert versions["glfw"][0] >= 3
    assert versions["opengl_minimum"] == (3, 3)
    assert versions["freetype"][0] >= 2
    assert versions["harfbuzz"][0] >= 1
