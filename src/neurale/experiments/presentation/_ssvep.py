# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT
"""Native SSVEP grid presentation. Luminance is calculated in C++."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..ssvep import SSVEPSnapshot, SSVEPTask
    from ._types import PresentationEnvironment, SoftwarePresentationTimes


@dataclass(frozen=True, slots=True, kw_only=True)
class SSVEPDisplayConfig:
    """Window settings. Fullscreen uses the monitor's current display mode.

    Vsync is always enabled. Targets follow task order, left to right and top
    to bottom. ``positions`` optionally supplies target centers in task order.
    Coordinates run from -1 to 1 on both axes; x points right and y points up.
    ``target_size`` is the circle diameter in the same units. None uses the automatic
    grid/size. The square workspace is letterboxed, not stretched into the window.
    These are screen-relative units, not degrees of visual angle.
    """

    title: str = "PyNeurale SSVEP"
    window_size: tuple[int, int] = (1000, 800)
    fullscreen: bool = False
    monitor: int | None = None
    positions: tuple[tuple[float, float], ...] | None = None
    target_size: float | None = None

    def __post_init__(self) -> None:
        # The existing window configuration owns the common validation rules.
        from ._center_out import CenterOutPresentationConfig

        CenterOutPresentationConfig(
            title=self.title,
            window_size=self.window_size,
            fullscreen=self.fullscreen,
            monitor=self.monitor,
        )
        if self.target_size is not None:
            if isinstance(self.target_size, bool) or not isinstance(self.target_size, int | float):
                raise TypeError("target_size must be a number or None")
            if not math.isfinite(self.target_size) or not 0 < self.target_size < 2:
                raise ValueError("target_size must be finite and between 0 and 2")
        if self.positions is not None:
            positions = tuple(tuple(point) for point in self.positions)
            for point in positions:
                if len(point) != 2:
                    raise ValueError("positions must contain (x, y) pairs")
                if any(isinstance(v, bool) or not isinstance(v, int | float) for v in point):
                    raise TypeError("positions coordinates must be numbers")
                if any(not math.isfinite(v) or not -1 < v < 1 for v in point):
                    raise ValueError("positions coordinates must be finite and between -1 and 1")
            if not positions:
                raise ValueError("positions must not be empty; use None for the automatic grid")
            object.__setattr__(self, "positions", positions)


class SSVEPDisplay:
    """Native window for an SSVEP task, usable as a context manager.

    Call ``poll()`` each frame, advance the task, then ``render(snapshot, time_ns)``.
    Time uses renderer_monotonic_now_ns(), shared with the task driver.
    All calls belong on the thread that opened the window. Render returns
    software submission/swap times, not physical display onset.
    """

    def __init__(self, task: SSVEPTask, config: SSVEPDisplayConfig | None = None) -> None:
        from ._native import load_presentation_extension

        self._config = SSVEPDisplayConfig() if config is None else config
        if not isinstance(self._config, SSVEPDisplayConfig):
            raise TypeError("config must be an SSVEPDisplayConfig")
        if self._config.positions is not None and len(self._config.positions) != len(task.targets):
            raise ValueError("positions must contain one center per task target")
        self._native = load_presentation_extension()
        self._impl = self._native._SSVEPConcretePresenter()
        self._should_close = False
        status = self._impl.open(
            task,
            self._config.title,
            self._native.WindowSize(*self._config.window_size),
            -1 if self._config.monitor is None else self._config.monitor,
            self._config.fullscreen,
            True,
            0,
            0,
            self._config.positions or (),
            self._config.target_size or 0.0,
        )
        if status != self._native.PresentationRuntimeStatus.OK:
            self._impl.close()
            if status == self._native.PresentationRuntimeStatus.INVALID_CONFIGURATION:
                raise ValueError(
                    "invalid SSVEP display configuration: check positions, target_size "
                    "and target frequencies against the monitor refresh rate"
                )
            self._check(status)

    def _check(self, status: object) -> None:
        if status != self._native.PresentationRuntimeStatus.OK:
            raise RuntimeError(f"SSVEP presentation failed: {status}")

    @property
    def should_close(self) -> bool:
        return self._should_close

    @property
    def environment(self) -> PresentationEnvironment:
        """Reported monitor mode; its refresh rate is not a timing measurement."""
        return self._impl.environment

    def poll(self) -> None:
        """Process window events; Escape/close sets ``should_close``."""
        if self._should_close:
            return
        status = self._impl.pump_events()
        if status == self._native.PresentationRuntimeStatus.WINDOW_CLOSED:
            self._should_close = True
            return
        self._check(status)
        while True:
            status, available, stop = self._impl.poll_input()
            self._check(status)
            if not available:
                return
            self._should_close |= stop

    def render(self, snapshot: SSVEPSnapshot, time_ns: int) -> SoftwarePresentationTimes:
        """Draw one frame; phase is relative to scheduled stimulation onset."""
        times = self._impl.render(snapshot, time_ns)
        self._check(times.status)
        return times

    def close(self) -> None:
        self._impl.close()
        self._should_close = True

    def _render_session(self, snapshot: SSVEPSnapshot, time_ns: int, epoch: int):
        """Black out an expired asynchronous snapshot and report the miss."""
        times = self._impl._render_relative(snapshot, time_ns, epoch)
        expired = (
            times.status == self._native.PresentationRuntimeStatus.PRESENTATION_DEADLINE_MISSED
        )
        if expired:
            from ..ssvep import SSVEPMachine

            times = self._impl._render_relative(SSVEPMachine().snapshot(), time_ns, epoch)
        self._check(times.status)
        return times, expired

    def __enter__(self) -> SSVEPDisplay:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
