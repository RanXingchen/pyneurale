#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Researcher-facing Center-Out presentation configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .._center_out_geometry import _center_out_visual_geometry

_RGBA = tuple[float, float, float, float]


def _color(value: _RGBA, name: str) -> _RGBA:
    if not isinstance(value, tuple) or len(value) != 4:
        raise TypeError(f"{name} must be a four-item RGBA tuple")
    result = tuple(float(component) for component in value)
    if any(not math.isfinite(component) or not 0.0 <= component <= 1.0 for component in result):
        raise ValueError(f"{name} components must be finite values between 0 and 1")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class CenterOutPresentationTheme:
    """Optional Center-Out colours; task geometry remains authoritative."""

    background: _RGBA = (0.06, 0.09, 0.10, 1.0)
    center_target: _RGBA = (0.32, 0.43, 0.44, 1.0)
    outward_target: _RGBA = (0.27, 0.37, 0.38, 1.0)
    active_move: _RGBA = (1.0, 0.79, 0.34, 1.0)
    active_hold: _RGBA = (0.34, 0.78, 0.91, 1.0)
    success: _RGBA = (0.34, 0.82, 0.59, 1.0)
    failure: _RGBA = (0.95, 0.39, 0.38, 1.0)
    cursor: _RGBA = (0.97, 0.99, 1.0, 1.0)

    def __post_init__(self) -> None:
        for name in (
            "background",
            "center_target",
            "outward_target",
            "active_move",
            "active_hold",
            "success",
            "failure",
            "cursor",
        ):
            object.__setattr__(self, name, _color(getattr(self, name), name))


@dataclass(frozen=True, slots=True)
class CenterOutPresentationConfig:
    """Window and colour intent for a native Center-Out presentation.

    ``window_size`` applies only in windowed mode. Fullscreen presentation
    preserves the selected monitor's current resolution and refresh rate.
    """

    title: str = "PyNeurale Center-Out experiment"
    window_size: tuple[int, int] = (800, 600)
    fullscreen: bool = False
    monitor: int | None = None
    vsync: bool = True
    theme: CenterOutPresentationTheme | None = None
    _visible: bool = field(default=True, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.title, str):
            raise TypeError("title must be a string")
        if not isinstance(self.window_size, tuple) or len(self.window_size) != 2:
            raise TypeError("window_size must be a (width, height) tuple")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.window_size):
            raise TypeError("window_size values must be integers")
        if any(value <= 0 for value in self.window_size):
            raise ValueError("window_size values must be positive")
        if self.monitor is not None:
            if isinstance(self.monitor, bool) or not isinstance(self.monitor, int):
                raise TypeError("monitor must be an integer or None")
            if self.monitor < 0:
                raise ValueError("monitor must be non-negative")
        if not isinstance(self.fullscreen, bool):
            raise TypeError("fullscreen must be a bool")
        if not isinstance(self.vsync, bool):
            raise TypeError("vsync must be a bool")
        if self.theme is not None and not isinstance(self.theme, CenterOutPresentationTheme):
            raise TypeError("theme must be a CenterOutPresentationTheme or None")


def _with_visibility(
    config: CenterOutPresentationConfig, *, visible: bool
) -> CenterOutPresentationConfig:
    """Set validation-only window visibility without expanding the public API."""

    if not isinstance(config, CenterOutPresentationConfig):
        raise TypeError("config must be a CenterOutPresentationConfig")
    if not isinstance(visible, bool):
        raise TypeError("visible must be a bool")
    object.__setattr__(config, "_visible", visible)
    return config


def _resolve_center_out_presentation(task: Any, config: CenterOutPresentationConfig) -> Any:
    from ._native import load_presentation_extension

    if not isinstance(config, CenterOutPresentationConfig):
        raise TypeError("config must be a CenterOutPresentationConfig")

    target_radius, cursor_radius, logical_space = _center_out_visual_geometry(task)

    native = load_presentation_extension()
    theme = config.theme or CenterOutPresentationTheme()
    colors = {
        name: native.Color(*getattr(theme, name))
        for name in (
            "background",
            "center_target",
            "outward_target",
            "active_move",
            "active_hold",
            "success",
            "failure",
            "cursor",
        )
    }
    style = native.CenterOutPresentationStyle(
        target_radius=target_radius,
        cursor_radius=cursor_radius,
        **colors,
    )
    return native.CenterOutPresentationConfig(
        task.geometry_unit,
        native.LogicalRect(*logical_space),
        style,
        title=config.title,
        window_size=native.WindowSize(*config.window_size),
        aspect_policy=native.AspectPolicy.FIT_LETTERBOX,
        monitor_idx=-1 if config.monitor is None else config.monitor,
        swap_interval=1 if config.vsync else 0,
        input_capacity=256,
        fullscreen=config.fullscreen,
        resizable=False,
        visible=config._visible,
    )


class CenterOutPresenter:
    """Draw Center-Out snapshots without advancing experiment state."""

    __slots__ = ("_config", "_impl")

    def __init__(self) -> None:
        from ._native import load_presentation_extension

        self._config: Any | None = None
        self._impl = load_presentation_extension().CenterOutPresenter()

    def open(
        self,
        task: Any,
        config: CenterOutPresentationConfig,
        renderer_origin_ns: int,
        experiment_origin_ns: int,
    ) -> Any:
        """Open the window and prepare the task; return the runtime status."""
        self._config = _resolve_center_out_presentation(task, config)
        return self._impl.open(
            task,
            self._config,
            renderer_origin_ns,
            experiment_origin_ns,
        )

    def update(self, snapshot: Any, cursor: Any) -> Any:
        """Submit the current experiment snapshot and cursor for rendering."""
        return self._impl.update(snapshot, cursor)

    def pump_events(self) -> Any:
        return self._impl.pump_events()

    def poll_control(self) -> Any:
        return self._impl.poll_control()

    def render(self, requested_ns: int, intended_ns: int) -> Any:
        """Render one frame and return software presentation timing evidence."""
        return self._impl.render(requested_ns, intended_ns)

    def cancel(self) -> None:
        self._impl.cancel()

    def close(self) -> None:
        self._impl.close()
        self._config = None

    @property
    def resource_stats(self) -> Any:
        return self._impl.resource_stats

    @property
    def environment(self) -> Any:
        return self._impl.environment

    @property
    def latest_update_ordinal(self) -> int:
        """Ordinal of the latest submitted snapshot."""
        return int(self._impl.latest_update_ordinal)

    @property
    def rendered_update_ordinal(self) -> int:
        """Ordinal of the snapshot used by the latest rendered frame."""
        return int(self._impl.rendered_update_ordinal)


class CenterOutSessionPresentationBridge:
    """Record presentation configuration and outcomes for a prepared session."""

    __slots__ = ("_impl",)

    def __init__(self, session: Any) -> None:
        from ._native import load_presentation_extension

        self._impl = load_presentation_extension().CenterOutSessionPresentationBridge(session)

    def record_config(self, presenter: CenterOutPresenter) -> bool:
        """Attach the presenter's resolved configuration to the session."""
        return bool(self._impl.record_config(presenter._impl))

    def report(
        self,
        presenter: CenterOutPresenter,
        times: Any,
        source_ordinal: int,
        trial: Any,
    ) -> bool:
        """Record one frame's software timing evidence for the given trial."""
        return bool(self._impl.report(presenter._impl, times, source_ordinal, trial))

    def report_failure(
        self,
        presenter: CenterOutPresenter,
        time_ns: int,
        status: Any,
        trial: Any,
    ) -> bool:
        """Record a failed presentation attempt for the given trial."""
        return bool(self._impl.report_failure(presenter._impl, time_ns, status, trial))
