#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Python control-plane facade for the native streaming runtime."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from ._config import RealtimeConfig, _native_realtime_config


class ExecutionProfile(StrEnum):
    """Execution guarantees applied when preparing a stream."""

    REALTIME = "realtime"
    RESEARCH = "research"
    OFFLINE = "offline"


class StreamRunner:
    """Control one C++ native streaming runtime.

    Python components must be wrapped in an explicit adapter and are accepted
    only by the ``research`` and ``offline`` profiles. The ``realtime`` profile
    rejects them during :meth:`prepare`.
    """

    __slots__ = (
        "_components",
        "_config",
        "_has_safety_controller",
        "_native",
        "_native_config",
        "_profile",
        "_runner",
    )

    def __init__(
        self,
        schema: Any,
        config: RealtimeConfig,
        source: Any,
        processor: Any,
        actuator: Any,
        *,
        profile: ExecutionProfile | str = ExecutionProfile.REALTIME,
        safety_controller: Any | None = None,
    ) -> None:
        from . import _native

        try:
            selected_profile = ExecutionProfile(profile)
        except ValueError as error:
            choices = ", ".join(item.value for item in ExecutionProfile)
            raise ValueError(f"profile must be one of: {choices}") from error

        native_config = _native_realtime_config(config, selected_profile)

        if safety_controller is None:
            runner = _native._NativeStreamRunner(schema, native_config, source, processor, actuator)
        else:
            runner = _native._NativeStreamRunner(
                schema,
                native_config,
                source,
                processor,
                actuator,
                safety_controller,
            )
        self._native = _native
        self._profile = selected_profile
        self._runner = runner
        self._components = (source, processor, actuator)
        self._config = config
        self._native_config = native_config
        self._has_safety_controller = safety_controller is not None

    @property
    def profile(self) -> ExecutionProfile:
        return self._profile

    @property
    def state(self) -> Any:
        return self._runner.state

    @property
    def stats(self) -> Any:
        return self._runner.stats

    @property
    def primary_fault(self) -> Any:
        return self._runner.primary_fault

    @property
    def fault_history(self) -> Any:
        return self._runner.fault_history

    @property
    def heartbeat(self) -> Any:
        return self._runner.heartbeat

    @property
    def realtime_configuration_status(self) -> Any:
        return self._runner.realtime_configuration_status

    @property
    def outstanding_frames(self) -> int:
        return self._runner.outstanding_frames

    @property
    def outstanding_discontinuities(self) -> int:
        return self._runner.outstanding_discontinuities

    def prepare(self) -> Any:
        self._validate_profile_contract()
        return self._runner.prepare()

    def _validate_profile_contract(self) -> None:
        """Validate Python-owned profile constraints without preparing native state."""

        if self._profile == ExecutionProfile.REALTIME:
            if not self._has_safety_controller:
                raise ValueError("realtime profile requires an explicit native safety controller")
            adapter_types = (
                self._native.PythonSourceAdapter,
                self._native.PythonProcessorAdapter,
                self._native.PythonSinkAdapter,
            )
            if any(isinstance(component, adapter_types) for component in self._components):
                raise ValueError("realtime profile requires native source, processor, and actuator")

    def arm(self) -> Any:
        return self._runner.arm()

    def start(self) -> Any:
        return self._runner.start()

    def run(self) -> Any:
        return self._runner.run()

    def stop(self) -> Any:
        return self._runner.stop()

    def abort(self) -> Any:
        return self._runner.abort()

    def join(self) -> Any:
        return self._runner.join()

    def reset(self) -> Any:
        return self._runner.reset()

    def close(self) -> Any:
        return self.stop()

    def add_observer(self, observer: Any, config: Any) -> Any:
        if isinstance(observer, self._native.PythonObserverBridge) and config.critical_recorder:
            raise ValueError("Python observer bridges cannot be critical recorders")
        return self._runner.add_observer(observer, config)

    def detach_observer(self, observer_id: int) -> Any:
        return self._runner.detach_observer(observer_id)

    def observer_stats(self, observer_id: int) -> Any:
        return self._runner.observer_stats(observer_id)

    def observer_drop_ranges(self, observer_id: int) -> Any:
        return self._runner.observer_drop_ranges(observer_id)

    def __enter__(self) -> StreamRunner:
        status = self.start()
        if status != self._native.StreamStatus.OK:
            raise RuntimeError(f"failed to start native StreamRunner: {status}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        if exc_type is None:
            status = self.stop()
            if status != self._native.StreamStatus.OK:
                raise RuntimeError(f"failed to stop native StreamRunner: {status}")
            return False

        status = self.abort()
        if status != self._native.StreamStatus.OK:
            note = f"failed to abort native StreamRunner: {status}"
            add_note = getattr(exc, "add_note", None)
            if add_note is not None:
                add_note(note)
            else:
                exc.__notes__ = [*getattr(exc, "__notes__", ()), note]
        return False
