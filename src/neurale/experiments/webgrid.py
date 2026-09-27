#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Renderer-independent WebGrid configuration, deterministic machine, and metrics.

Coordinates are logical task coordinates. Cells use one-based row-major IDs,
with row zero at ``bounds.min_y`` and column zero at ``bounds.min_x``. Each
cell is half-open on its maximum edges, and the grid's outer maximum edges are
outside. Consequently an interior point belongs to at most one cell.

Pointer movement and selection are separate inputs. ``locate_cell`` performs
only hit testing; ``make_selection_event`` is the explicit discrete selection
operation. WebGrid v1 advances after a correct selection and keeps the current
target after an incorrect selection. ``WebGridMachine`` applies those policies
from explicit integer-nanosecond time and zero or one ``SelectionEvent`` per
step; pointer movement alone never selects.

Seeded schedules are reconstructed from ``seed`` and target ordinal and may
allow or forbid immediate repetition. Explicit schedules are finite and never
wrap. At ordinal zero ``previous`` is ignored; returning to ordinal zero
reproduces the documented sequence. An unknown nonzero sampler version remains
a valid recorded configuration, but a call that actually needs to execute its
seeded draw returns ``VERSION_UNSUPPORTED`` instead of using the current
sampler.

Raw metric version 1 reports correct and incorrect counts, elapsed active time,
correct targets/minute, net correct targets/minute, and acquisition-time
summaries over successful targets. The two rates divide by elapsed active time;
they are explicitly undefined when that duration is zero. Every accepted
selection retains its target onset and elapsed duration, so ``summarize`` can
recompute all published metrics offline. ``summarize`` requires the recorded
metric version explicitly and validates that incorrect selections retain their
trial/onset while each correct selection closes that trial. No
BPS/achieved-bitrate formula is defined.

There is no browser, renderer, monitor, DPI, input-event, timer, device, or
recording dependency in this module. Importing it does not load the native
extension until a name is used.
"""

from __future__ import annotations

_NATIVE_EXPORTS = {
    "CellBounds",
    "CURRENT_METRIC_VERSION",
    "CorrectSelectionPolicy",
    "GridCell",
    "ImmediateRepetitionPolicy",
    "IncorrectSelectionPolicy",
    "MAX_WEBGRID_CELLS",
    "METRIC_VERSION_1",
    "PointerPosition",
    "TARGET_SELECTION_STREAM",
    "TargetScheduleKind",
    "TaskBounds",
    "WebGridMachine",
    "WebGridMetrics",
    "WebGridReason",
    "WebGridSelectionRecord",
    "WebGridSnapshot",
    "WebGridState",
    "WebGridStepResult",
    "WebGridTrial",
    "WebGridConfig",
    "configuration_fingerprint",
    "correct_selection_policy_declared",
    "grid_cell",
    "grid_cell_by_id",
    "immediate_repetition_policy_declared",
    "incorrect_selection_policy_declared",
    "locate_cell",
    "locate_selectable_cell",
    "make_selection_event",
    "select_target",
    "selectable",
    "session_duration_reached",
    "summarize",
    "target_limit_reached",
    "target_schedule_kind_declared",
    "validate",
    "webgrid_state_declared",
}


def __getattr__(name: str) -> object:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from neurale._native_loader import load_native_namespace

    value = getattr(load_native_namespace("experiments.webgrid"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS)


__all__ = sorted(_NATIVE_EXPORTS)
