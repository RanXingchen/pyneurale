#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from itertools import pairwise

import pytest

from neurale.experiments import UNSET_TARGET_ID, ContractStatus, TrialIdentity, webgrid


def _config(**overrides: object) -> webgrid.WebGridConfig:
    values: dict[str, object] = {
        "rows": 2,
        "columns": 3,
        "bounds": webgrid.TaskBounds(-1.0, 1.0, 0.0, 2.0),
        "candidates": [1, 2, 3, 4, 5, 6],
        "schedule": webgrid.TargetScheduleKind.SEEDED,
        "immediate_repetition": webgrid.ImmediateRepetitionPolicy.FORBID,
        "correct_selection": webgrid.CorrectSelectionPolicy.ADVANCE_TARGET,
        "incorrect_selection": webgrid.IncorrectSelectionPolicy.KEEP_CURRENT_TARGET,
        "seed": 918273,
        "target_count_limit": 8,
        "session_duration_ns": 100,
        "metric_version": webgrid.METRIC_VERSION_1,
    }
    values.update(overrides)
    return webgrid.WebGridConfig(**values)


def test_geometry_uses_one_based_row_major_half_open_cells() -> None:
    config = _config()
    assert webgrid.validate(config) == ContractStatus.OK

    status, cell = webgrid.grid_cell(config, 0, 0)
    assert status == ContractStatus.OK
    assert (cell.id, cell.row, cell.column) == (1, 0, 0)
    assert (cell.bounds.min_x, cell.bounds.max_x) == pytest.approx((-1.0, -1.0 / 3.0))
    assert (cell.bounds.min_y, cell.bounds.max_y) == pytest.approx((0.0, 1.0))

    status, cell = webgrid.grid_cell_by_id(config, 6)
    assert status == ContractStatus.OK
    assert (cell.row, cell.column) == (1, 2)
    assert (cell.bounds.min_x, cell.bounds.max_x) == pytest.approx((1.0 / 3.0, 1.0))


@pytest.mark.parametrize(
    ("rows", "columns", "cell_id", "expected_row", "expected_column"),
    [(1, 1, 1, 0, 0), (3, 1, 3, 2, 0), (1, 4, 4, 0, 3)],
)
def test_geometry_is_derived_for_representative_grid_shapes(
    rows: int, columns: int, cell_id: int, expected_row: int, expected_column: int
) -> None:
    count = rows * columns
    config = _config(
        rows=rows,
        columns=columns,
        candidates=list(range(1, count + 1)),
        immediate_repetition=webgrid.ImmediateRepetitionPolicy.ALLOW,
    )
    status, cell = webgrid.grid_cell_by_id(config, cell_id)
    assert status == ContractStatus.OK
    assert (cell.row, cell.column) == (expected_row, expected_column)


@pytest.mark.parametrize(
    ("pointer", "expected"),
    [
        (webgrid.PointerPosition(-1.0, 0.0), 1),
        (webgrid.PointerPosition(-1.0 / 3.0, 0.5), 2),
        (webgrid.PointerPosition(1.0 / 3.0, 1.0), 6),
        (webgrid.PointerPosition(1.0, 1.0), UNSET_TARGET_ID),
        (webgrid.PointerPosition(0.0, 2.0), UNSET_TARGET_ID),
        (webgrid.PointerPosition(-1.01, 0.5), UNSET_TARGET_ID),
    ],
)
def test_cell_mapping_uses_exact_boundary_semantics(
    pointer: webgrid.PointerPosition, expected: int
) -> None:
    assert webgrid.locate_cell(_config(), pointer) == (ContractStatus.OK, expected)


def test_selectable_subset_is_distinct_from_physical_grid() -> None:
    config = _config(candidates=[2, 5])
    pointer = webgrid.PointerPosition(-0.8, 0.5)
    assert webgrid.locate_cell(config, pointer) == (ContractStatus.OK, 1)
    assert webgrid.locate_selectable_cell(config, pointer) == (
        ContractStatus.OK,
        UNSET_TARGET_ID,
    )
    assert webgrid.locate_selectable_cell(config, webgrid.PointerPosition(0.0, 0.5)) == (
        ContractStatus.OK,
        2,
    )


def test_discrete_selection_records_all_hit_kinds() -> None:
    config = _config(candidates=[2, 5])
    trial = TrialIdentity(ordinal=3, key=19, block=1, target_id=2)

    status, correct = webgrid.make_selection_event(
        config, webgrid.PointerPosition(0.0, 0.5), 2, trial, 7, 400, 11
    )
    assert status == ContractStatus.OK
    assert correct.correct
    assert (correct.selected_id, correct.intended_id, correct.dwell_ns) == (2, 2, 0)

    status, incorrect = webgrid.make_selection_event(
        config, webgrid.PointerPosition(0.0, 1.5), 2, trial, 7, 401, 12
    )
    assert status == ContractStatus.OK
    assert not incorrect.correct
    assert (incorrect.selected_id, incorrect.intended_id) == (5, 2)
    assert config.incorrect_selection == webgrid.IncorrectSelectionPolicy.KEEP_CURRENT_TARGET

    status, empty = webgrid.make_selection_event(
        config, webgrid.PointerPosition(-0.8, 0.5), 2, trial, 7, 402, 13
    )
    assert status == ContractStatus.OK
    assert not empty.correct
    assert empty.selected_id == UNSET_TARGET_ID


def test_seeded_schedule_is_deterministic_without_repetition() -> None:
    config = _config(initial_target=4)
    first: list[int] = []
    previous = UNSET_TARGET_ID
    for ordinal in range(8):
        status, target = webgrid.select_target(config, ordinal, previous)
        assert status == ContractStatus.OK
        first.append(target)
        previous = target

    assert first[0] == 4
    assert all(left != right for left, right in pairwise(first))

    second: list[int] = []
    previous = UNSET_TARGET_ID
    for ordinal in range(8):
        status, target = webgrid.select_target(config, ordinal, previous)
        assert status == ContractStatus.OK
        second.append(target)
        previous = target
    assert second == first
    assert webgrid.select_target(config, 8, previous) == (ContractStatus.OK, UNSET_TARGET_ID)


def test_ordinal_zero_ignores_stale_previous() -> None:
    config = _config(target_count_limit=0)
    assert webgrid.select_target(config, 0, UNSET_TARGET_ID) == webgrid.select_target(config, 0, 1)
    assert webgrid.select_target(config, 1, UNSET_TARGET_ID)[0] == ContractStatus.IDENTITY_MISSING


def test_future_sampler_version_is_rejected_on_draw() -> None:
    seeded = _config(sampler_version=999, initial_target=4, target_count_limit=0)
    assert webgrid.validate(seeded) == ContractStatus.OK
    assert webgrid.grid_cell(seeded, 0, 0)[0] == ContractStatus.OK
    assert webgrid.session_duration_reached(seeded, 0, 99) == (ContractStatus.OK, False)
    assert webgrid.select_target(seeded, 0, 6) == (ContractStatus.OK, 4)
    assert webgrid.select_target(seeded, 1, 4)[0] == ContractStatus.VERSION_UNSUPPORTED

    explicit = _config(
        schedule=webgrid.TargetScheduleKind.EXPLICIT_SEQUENCE,
        explicit_targets=[4, 2],
        initial_target=4,
        target_count_limit=0,
        sampler_version=999,
    )
    assert webgrid.validate(explicit) == ContractStatus.OK
    assert webgrid.select_target(explicit, 1, 4) == (ContractStatus.OK, 2)

    assert webgrid.validate(_config(sampler_version=0)) == ContractStatus.IDENTITY_MISSING


def test_explicit_schedule_is_finite_and_does_not_wrap() -> None:
    config = _config(
        schedule=webgrid.TargetScheduleKind.EXPLICIT_SEQUENCE,
        explicit_targets=[3, 6, 2],
        initial_target=3,
        target_count_limit=0,
    )
    assert webgrid.validate(config) == ContractStatus.OK
    assert [webgrid.select_target(config, ordinal)[1] for ordinal in range(3)] == [3, 6, 2]
    assert webgrid.select_target(config, 3) == (ContractStatus.OK, UNSET_TARGET_ID)
    assert webgrid.target_limit_reached(config, 3)


def test_seeded_schedule_may_allow_immediate_repetition() -> None:
    config = _config(
        candidates=[5],
        immediate_repetition=webgrid.ImmediateRepetitionPolicy.ALLOW,
        target_count_limit=3,
    )
    assert webgrid.validate(config) == ContractStatus.OK
    assert [webgrid.select_target(config, ordinal)[1] for ordinal in range(3)] == [5, 5, 5]


@pytest.mark.parametrize(
    "overrides",
    [
        {"rows": 3},
        {"columns": 4},
        {"bounds": webgrid.TaskBounds(-2.0, 1.0, 0.0, 2.0)},
        {"bounds": webgrid.TaskBounds(-1.0, 2.0, 0.0, 2.0)},
        {"bounds": webgrid.TaskBounds(-1.0, 1.0, 0.25, 2.0)},
        {"bounds": webgrid.TaskBounds(-1.0, 1.0, 0.0, 3.0)},
        {"candidates": [1, 2, 3, 4, 5]},
        {"candidates": [2, 1, 3, 4, 5, 6]},
        {"schedule": webgrid.TargetScheduleKind.EXPLICIT_SEQUENCE},
        {"immediate_repetition": webgrid.ImmediateRepetitionPolicy.ALLOW},
        {"correct_selection": webgrid.CorrectSelectionPolicy.UNSPECIFIED},
        {"incorrect_selection": webgrid.IncorrectSelectionPolicy.UNSPECIFIED},
        {"seed": 918274},
        {"sampler_version": 2},
        {"explicit_targets": [3]},
        {"initial_target": 3},
        {"session_duration_ns": 101},
        {"target_count_limit": 9},
        {"metric_version": 2},
    ],
)
def test_configuration_fields_participate_in_fingerprint(
    overrides: dict[str, object],
) -> None:
    assert webgrid.configuration_fingerprint(_config(**overrides)) != (
        webgrid.configuration_fingerprint(_config())
    )


def test_configuration_fingerprint_canonicalizes_zero() -> None:
    assert webgrid.configuration_fingerprint(_config()) == webgrid.configuration_fingerprint(
        _config()
    )
    negative = _config(bounds=webgrid.TaskBounds(-1.0, 1.0, -0.0, 2.0))
    positive = _config(bounds=webgrid.TaskBounds(-1.0, 1.0, 0.0, 2.0))
    assert webgrid.configuration_fingerprint(negative) == webgrid.configuration_fingerprint(
        positive
    )


def test_maximum_capacity_geometry_and_schedules() -> None:
    candidates = list(range(1, webgrid.MAX_WEBGRID_CELLS + 1))
    seeded = _config(
        rows=16,
        columns=16,
        bounds=webgrid.TaskBounds(0.0, 16.0, 0.0, 16.0),
        candidates=candidates,
        target_count_limit=0,
    )
    assert webgrid.validate(seeded) == ContractStatus.OK
    status, last = webgrid.grid_cell_by_id(seeded, 256)
    assert status == ContractStatus.OK
    assert (last.row, last.column) == (15, 15)
    assert webgrid.locate_cell(seeded, webgrid.PointerPosition(15.999, 15.999)) == (
        ContractStatus.OK,
        256,
    )
    assert webgrid.locate_cell(seeded, webgrid.PointerPosition(16.0, 15.999)) == (
        ContractStatus.OK,
        UNSET_TARGET_ID,
    )
    assert 1 <= webgrid.select_target(seeded, 0)[1] <= 256

    explicit = _config(
        rows=16,
        columns=16,
        bounds=webgrid.TaskBounds(0.0, 16.0, 0.0, 16.0),
        candidates=candidates,
        schedule=webgrid.TargetScheduleKind.EXPLICIT_SEQUENCE,
        explicit_targets=candidates,
        target_count_limit=0,
    )
    assert webgrid.validate(explicit) == ContractStatus.OK
    assert webgrid.select_target(explicit, 255, 255) == (ContractStatus.OK, 256)
    assert webgrid.select_target(explicit, 256, 256) == (ContractStatus.OK, UNSET_TARGET_ID)


def test_duration_endpoint_is_terminal() -> None:
    config = _config()
    assert webgrid.session_duration_reached(config, 1_000, 1_099) == (
        ContractStatus.OK,
        False,
    )
    assert webgrid.session_duration_reached(config, 1_000, 1_100) == (
        ContractStatus.OK,
        True,
    )


@pytest.mark.parametrize(
    "config",
    [
        _config(rows=0),
        _config(columns=0),
        _config(rows=17, columns=16),
        _config(candidates=[]),
        _config(candidates=[1, 1]),
        _config(candidates=[7]),
        _config(
            schedule=webgrid.TargetScheduleKind.EXPLICIT_SEQUENCE,
            explicit_targets=[1, 1],
            target_count_limit=0,
        ),
        _config(candidates=[1], target_count_limit=2),
        _config(metric_version=0),
    ],
)
def test_invalid_dimensions_and_schedules_are_rejected(
    config: webgrid.WebGridConfig,
) -> None:
    assert webgrid.validate(config) != ContractStatus.OK


def test_public_records_are_immutable() -> None:
    config = _config()
    with pytest.raises(AttributeError):
        config.rows = 3
    status, cell = webgrid.grid_cell(config, 0, 0)
    assert status == ContractStatus.OK
    with pytest.raises(AttributeError):
        cell.id = 9
