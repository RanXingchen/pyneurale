# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Pre-start queue sizing for continuous, nominal-cadence recording."""

from __future__ import annotations

import math
from dataclasses import replace

from ._errors import RecorderConfigError
from ._plan import RecordingPlan

# Five seconds of nominal input plus 25% burst margin. Not a throughput
# guarantee: overflow still faults instead of growing capture RAM.
_BUFFER_SECONDS = 5.0 * 1.25


def budget_source(plan: RecordingPlan) -> RecordingPlan:
    frames = 0
    for signal in plan.native_schema.signals:
        numerator, denominator = signal.fs
        if numerator <= 0 or denominator <= 0 or signal.nominal_block_samples <= 0:
            raise RecorderConfigError(
                "automatic recording sizing requires a nominal frame cadence; "
                "provide RecorderConfig(limits=RecorderLimits(...)) for this source"
            )
        frames += math.ceil(
            _BUFFER_SECONDS * numerator / denominator / signal.nominal_block_samples
        )
    return replace(
        plan,
        resource_bounds=replace(
            plan.resource_bounds,
            frame_queue_capacity=max(plan.resource_bounds.frame_queue_capacity, frames),
        ),
    )


def budget_recording(
    plan: RecordingPlan, *, duration_seconds: float, control_records: int
) -> RecordingPlan:
    """Budget a fixed control backlog, not the whole session's payloads."""
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise RecorderConfigError("recording duration must be finite and positive")
    if control_records < 0:
        raise RecorderConfigError("control record budget must be non-negative")
    plan = budget_source(plan)
    return replace(
        plan,
        resource_bounds=replace(
            plan.resource_bounds,
            control_queue_capacity=max(
                plan.resource_bounds.control_queue_capacity,
                math.ceil(_BUFFER_SECONDS * control_records / duration_seconds),
            ),
        ),
    )
