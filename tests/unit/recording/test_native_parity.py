#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Convergence checks for the former parity boundary.

The pre-convergence Python engine no longer exists.  These checks retain the
useful parity evidence at the boundary that remains: the stable facade and the
native core consume the same compiled plan and expose the same typed control
operations.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from neurale.recording import SessionRecorder
from neurale.recording._native_recorder import NativeRecorderOptions, NativeSessionRecorder

from .conftest import native_schema, recorder_config, stream_specs


def test_public_facade_passes_compiled_plan_to_native(tmp_path: Path, monkeypatch) -> None:
    import neurale.recording._session_recorder as facade

    monkeypatch.setattr(
        facade,
        "_stable_native_options",
        lambda plan, _output, **_kwargs: NativeRecorderOptions(
            spool="memory",
            spool_capacity_bytes=plan.resource_bounds.spool_capacity_bytes,
            edge_capacity=plan.resource_bounds.frame_queue_capacity,
        ),
    )
    captured = []
    real_compile = facade.compile_recording_plan

    def capture_plan(config, source):
        plan = real_compile(config, source)
        captured.append(plan)
        return plan

    monkeypatch.setattr(facade, "compile_recording_plan", capture_plan)
    config = recorder_config(tmp_path / "session.nrf", streams=stream_specs())
    recorder = SessionRecorder.create(config, native_schema())
    try:
        assert recorder.plan is captured[0]
        assert recorder._impl.plan is recorder.plan
    finally:
        recorder.close()


def test_control_signatures_match_native_boundary() -> None:
    methods = (
        "record_event",
        "record_state",
        "record_command",
        "record_task_variable",
        "record_label",
        "record_target",
        "record_assistance",
        "record_trial",
        "record_fault",
        "checkpoint",
    )

    for name in methods:
        assert inspect.signature(getattr(SessionRecorder, name)) == inspect.signature(
            getattr(NativeSessionRecorder, name)
        )
