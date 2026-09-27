#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared task-derived Center-Out display and cursor workspace geometry."""

from __future__ import annotations

from typing import Any

_POINT_CURSOR_RADIUS_RATIO = 0.25
_VIEWPORT_EXPANSION = 2.0


def _center_out_visual_geometry(
    task: Any,
) -> tuple[float, float, tuple[float, float, float, float]]:
    acceptance_x = float(task.acceptance.half_extent_x)
    acceptance_y = float(task.acceptance.half_extent_y)
    target_radius = min(acceptance_x, acceptance_y)
    task_cursor_extent = float(task.cursor.extent)
    cursor_radius = (
        task_cursor_extent
        if task_cursor_extent > 0.0
        else target_radius * _POINT_CURSOR_RADIUS_RATIO
    )

    placements = (task.layout.center, *task.layout.surrounding)
    xs = tuple(float(placement.pos.x) for placement in placements)
    ys = tuple(float(placement.pos.y) for placement in placements)
    draw_extent = max(target_radius, cursor_radius)
    left = min(xs) - draw_extent
    right = max(xs) + draw_extent
    bottom = min(ys) - draw_extent
    top = max(ys) + draw_extent
    side = max(right - left, top - bottom) * _VIEWPORT_EXPANSION
    center_x = (left + right) * 0.5
    center_y = (bottom + top) * 0.5
    logical_space = (center_x - side * 0.5, center_y - side * 0.5, side, side)
    return target_radius, cursor_radius, logical_space


def _center_out_cursor_bounds(task: Any) -> tuple[float, float, float, float]:
    _, cursor_radius, (left, bottom, width, height) = _center_out_visual_geometry(task)
    return (
        left + cursor_radius,
        bottom + cursor_radius,
        left + width - cursor_radius,
        bottom + height - cursor_radius,
    )
