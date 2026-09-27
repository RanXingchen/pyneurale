#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Center-Out 2D configuration, target geometry, and target schedule."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurale.experiments import ContractStatus
from neurale.experiments import center_out as co

# ---------------------------------------------------------------------------
# Reference calculations


def _reference_target_positions(target_map: list[int], distance: float) -> np.ndarray:
    ordered = sorted(target_map)
    if ordered[0] != 0:
        raise ValueError("target map must include the centre target with index 0")
    count = len(ordered)
    if count == 1:
        raise ValueError("center out task at least has one surrounding target")
    targets = np.zeros((count, 2), dtype=float)
    angle = 2.0 * math.pi / max(count - 1.0, 8.0)
    for i in range(1, count):
        theta = angle * (ordered[i] - 1)
        targets[i, 0] = distance * math.cos(theta)
        targets[i, 1] = distance * math.sin(theta)
    return targets


def _reference_contains_cursor(
    player: tuple[float, float],
    player_radius: float,
    target: tuple[float, float],
    homecage: tuple[float, float],
) -> bool:
    left = player[0] - player_radius
    right = player[0] + player_radius
    top = player[1] - player_radius
    bottom = player[1] + player_radius
    return bool(
        left >= target[0] - homecage[0]
        and right <= target[0] + homecage[0]
        and top >= target[1] - homecage[1]
        and bottom <= target[1] + homecage[1]
    )


# ---------------------------------------------------------------------------
# Builders


def _radial_request(radius: float = 100.0, count: int = 8) -> object:
    return co.RadialLayoutRequest(
        radius=radius,
        center_id=1,
        ids=[idx + 2 for idx in range(count)],
        spokes=list(range(count)),
    )


def _ring(radius: float = 100.0, count: int = 8) -> object:
    return co.build_radial_layout(_radial_request(radius, count))


def _config(**overrides: object) -> object:
    fields: dict[str, object] = {
        "geometry_unit": co.GeometryUnit.MILLIMETRES,
        "layout": _ring(),
        "acceptance": co.AcceptanceRegion(half_extent_x=5.0, half_extent_y=5.0),
        "cursor": co.CursorGeometry(extent=2.0),
        "movement_timeout": co.PhaseDurations(to_center=1_000_000_000, to_out=1_000_000_000),
        "hold_ns": 100_000_000,
        "reward_dwell": co.PhaseDurations(to_center=100_000_000, to_out=100_000_000),
        "punish_dwell": co.PhaseDurations(to_center=200_000_000, to_out=200_000_000),
        "selection": co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL,
        "seed": 0xC0FFEE,
        "trial_limit": 0,
    }
    fields.update(overrides)
    return co._raw_config(**fields)


def _positions(layout: object) -> np.ndarray:
    return np.array([[t.pos.x, t.pos.y] for t in layout.surrounding], dtype=float)


# ---------------------------------------------------------------------------


class TestRadialGeometry:
    def test_radius_only_builds_classic_eight_direction_layout(self) -> None:
        request = co.RadialLayoutRequest(radius=0.5)
        assert request.center_id == 1
        assert request.count == 8
        assert list(request.ids) == list(range(2, 10))
        assert list(request.spokes) == list(range(8))

        layout = co.build_radial_layout(request)
        angles = np.arange(8, dtype=float) * (2.0 * np.pi / 8.0)
        expected = 0.5 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        np.testing.assert_allclose(_positions(layout), expected, atol=1e-12)

    def test_radius_is_required(self) -> None:
        with pytest.raises(TypeError):
            co.RadialLayoutRequest()

    def test_eight_target_ring_matches_numpy_coordinates(self) -> None:
        layout = _ring(radius=100.0, count=8)
        angles = np.arange(8, dtype=float) * (2.0 * np.pi / 8.0)
        expected = 100.0 * np.stack([np.cos(angles), np.sin(angles)], axis=1)
        np.testing.assert_allclose(_positions(layout), expected, atol=1e-12)
        assert (layout.center.pos.x, layout.center.pos.y) == (0.0, 0.0)
        assert layout.center.id == 1

    def test_fewer_targets_do_not_fill_circle(self) -> None:
        # Four targets occupy 0, 45, 90, and 135 degrees.
        layout = _ring(radius=1.0, count=4)
        angles = np.arange(4, dtype=float) * (2.0 * np.pi / 8.0)
        expected = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        np.testing.assert_allclose(_positions(layout), expected, atol=1e-12)
        assert layout.surrounding[3].pos.x < 0.0
        assert layout.surrounding[3].pos.y > 0.0

    @pytest.mark.parametrize("count", [1, 2, 3, 7, 8, 9, 12, 16])
    def test_step_is_two_pi_over_floored_count(self, count: int) -> None:
        assert co.radial_spoke_step(count) == pytest.approx(
            2.0 * math.pi / max(count, co.MIN_RADIAL_SPOKES)
        )

    def test_single_surrounding_target_lands_on_first_axis(self) -> None:
        layout = _ring(radius=7.5, count=1)
        assert layout.count == 1
        assert layout.surrounding[0].pos.x == pytest.approx(7.5)
        assert layout.surrounding[0].pos.y == pytest.approx(0.0, abs=1e-15)

    def test_layout_capacity_is_enforced(self) -> None:
        layout = _ring(radius=1.0, count=co.MAX_SURROUNDING_TARGETS)
        assert layout.count == co.MAX_SURROUNDING_TARGETS
        with pytest.raises(ValueError, match="more than"):
            _radial_request(1.0, co.MAX_SURROUNDING_TARGETS + 1)

    def test_spoke_beyond_count_wraps(self) -> None:
        request = co.RadialLayoutRequest(radius=1.0, center_id=1, ids=[2, 3], spokes=[0, 10])
        layout = co.build_radial_layout(request)
        theta = co.radial_spoke_step(2) * 10.0
        assert layout.surrounding[1].pos.x == pytest.approx(math.cos(theta))
        assert layout.surrounding[1].pos.y == pytest.approx(math.sin(theta))

    @pytest.mark.parametrize(
        "target_map",
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 8],
            [0, 1, 2, 3, 4],
            [0, 1],
            [0, 1, 3, 5, 7],
            [0, 2, 4, 6, 8, 10, 12, 14, 16],
            list(range(13)),
        ],
    )
    def test_layout_matches_reference_positions(self, target_map: list[int]) -> None:
        distance = 137.5
        reference = _reference_target_positions(target_map, distance)
        spokes = [value - 1 for value in sorted(target_map)[1:]]
        request = co.RadialLayoutRequest(
            radius=distance,
            center_id=1,
            ids=[idx + 2 for idx in range(len(spokes))],
            spokes=spokes,
        )
        layout = co.build_radial_layout(request)
        np.testing.assert_allclose(_positions(layout), reference[1:], atol=1e-12)
        np.testing.assert_allclose(
            [layout.center.pos.x, layout.center.pos.y], reference[0], atol=0.0
        )


class TestCallerConfigurationIsNotModified:
    def test_request_survives_being_built_from(self) -> None:
        request = co.RadialLayoutRequest(
            radius=3.0, center_id=1, ids=[10, 9, 8, 7, 6], spokes=[0, 1, 2, 3, 4]
        )
        before_ids = list(request.ids)
        before_spokes = list(request.spokes)
        co.build_radial_layout(request)
        assert list(request.ids) == before_ids
        assert list(request.spokes) == before_spokes

    def test_layout_keeps_caller_order(self) -> None:
        request = co.RadialLayoutRequest(
            radius=3.0, center_id=1, ids=[10, 9, 8, 7, 6], spokes=[0, 1, 2, 3, 4]
        )
        layout = co.build_radial_layout(request)
        assert [target.id for target in layout.surrounding] == [10, 9, 8, 7, 6]
        # The angle follows the spoke, not the identifier: descending names on
        # ascending spokes still walk the ring counter-clockwise.
        step = co.radial_spoke_step(5)
        expected = 3.0 * np.cos(np.arange(5, dtype=float) * step)
        np.testing.assert_allclose(_positions(layout)[:, 0], expected, atol=1e-12)

    def test_rejected_build_raises_specific_value_error(self) -> None:
        request = co.RadialLayoutRequest(radius=-1.0, center_id=1, ids=[2], spokes=[0])
        with pytest.raises(ValueError, match="radius must be greater than zero"):
            co.build_radial_layout(request)

    def test_records_are_immutable(self) -> None:
        layout = _ring()
        with pytest.raises(AttributeError):
            layout.count = 3  # type: ignore[misc]
        with pytest.raises(AttributeError):
            layout.center.id = 9  # type: ignore[misc]
        config = _config()
        with pytest.raises(AttributeError):
            config.seed = 1  # type: ignore[misc]

    def test_layout_round_trips_through_constructor(self) -> None:
        layout = _ring(count=5)
        rebuilt = co.CenterOut2DLayout(center=layout.center, surrounding=layout.surrounding)
        assert rebuilt.count == layout.count
        assert co.layout_fingerprint(rebuilt) == co.layout_fingerprint(layout)


class TestTargetIdentifiers:
    def test_valid_layout_is_accepted(self) -> None:
        assert co.validate(_ring()) == ContractStatus.OK

    def test_empty_surrounding_set_is_rejected(self) -> None:
        layout = co.CenterOut2DLayout(
            center=co.TargetPlacement(id=1, pos=co.WorkspacePoint(0.0, 0.0)),
            surrounding=[],
        )
        assert co.validate(layout) == ContractStatus.TARGET_SET_INVALID

    def test_unnamed_centre_is_rejected(self) -> None:
        # "Missing centre semantics": the centre is a field so it cannot be left
        # out, but it can be left unnamed.
        layout = co.CenterOut2DLayout(
            center=co.TargetPlacement(id=0, pos=co.WorkspacePoint(0.0, 0.0)),
            surrounding=[co.TargetPlacement(id=2, pos=co.WorkspacePoint(1.0, 0.0))],
        )
        assert co.validate(layout) == ContractStatus.IDENTITY_MISSING

    def test_unnamed_surrounding_target_is_rejected(self) -> None:
        layout = co.CenterOut2DLayout(
            center=co.TargetPlacement(id=1, pos=co.WorkspacePoint(0.0, 0.0)),
            surrounding=[
                co.TargetPlacement(id=2, pos=co.WorkspacePoint(1.0, 0.0)),
                co.TargetPlacement(id=0, pos=co.WorkspacePoint(0.0, 1.0)),
            ],
        )
        assert co.validate(layout) == ContractStatus.IDENTITY_MISSING

    def test_duplicate_identifier_is_rejected(self) -> None:
        layout = co.CenterOut2DLayout(
            center=co.TargetPlacement(id=1, pos=co.WorkspacePoint(0.0, 0.0)),
            surrounding=[
                co.TargetPlacement(id=2, pos=co.WorkspacePoint(1.0, 0.0)),
                co.TargetPlacement(id=2, pos=co.WorkspacePoint(0.0, 1.0)),
            ],
        )
        assert co.validate(layout) == ContractStatus.TARGET_SET_INVALID

    def test_surrounding_target_cannot_reuse_centre_id(self) -> None:
        layout = co.CenterOut2DLayout(
            center=co.TargetPlacement(id=1, pos=co.WorkspacePoint(0.0, 0.0)),
            surrounding=[co.TargetPlacement(id=1, pos=co.WorkspacePoint(1.0, 0.0))],
        )
        assert co.validate(layout) == ContractStatus.TARGET_SET_INVALID

    def test_non_finite_position_is_rejected(self) -> None:
        layout = co.CenterOut2DLayout(
            center=co.TargetPlacement(id=1, pos=co.WorkspacePoint(0.0, 0.0)),
            surrounding=[co.TargetPlacement(id=2, pos=co.WorkspacePoint(float("nan"), 0.0))],
        )
        assert co.validate(layout) == ContractStatus.VALUE_NOT_FINITE

    def test_two_identifiers_may_share_position(self) -> None:
        layout = co.CenterOut2DLayout(
            center=co.TargetPlacement(id=1, pos=co.WorkspacePoint(0.0, 0.0)),
            surrounding=[
                co.TargetPlacement(id=2, pos=co.WorkspacePoint(1.0, 0.0)),
                co.TargetPlacement(id=3, pos=co.WorkspacePoint(1.0, 0.0)),
            ],
        )
        assert co.validate(layout) == ContractStatus.OK

    @pytest.mark.parametrize(
        ("mutation", "expected", "message"),
        [
            ({"radius": 0.0}, ContractStatus.PARAMETER_OUT_OF_RANGE, "greater than zero"),
            ({"radius": -1.0}, ContractStatus.PARAMETER_OUT_OF_RANGE, "greater than zero"),
            ({"radius": float("inf")}, ContractStatus.VALUE_NOT_FINITE, "must be finite"),
            (
                {"ids": [], "spokes": []},
                ContractStatus.TARGET_SET_INVALID,
                "at least one surrounding target",
            ),
            ({"center_id": 0}, ContractStatus.IDENTITY_MISSING, "center_id must not be 0"),
            (
                {"ids": [2, 0], "spokes": [0, 1]},
                ContractStatus.IDENTITY_MISSING,
                r"ids\[1\] must not be 0",
            ),
            (
                {"ids": [2, 2], "spokes": [0, 1]},
                ContractStatus.TARGET_SET_INVALID,
                "duplicate target id 2",
            ),
            (
                {"ids": [1, 2], "spokes": [0, 1]},
                ContractStatus.TARGET_SET_INVALID,
                "reuses center_id 1",
            ),
        ],
    )
    def test_radial_request_reports_failure(
        self, mutation: dict[str, object], expected: ContractStatus, message: str
    ) -> None:
        fields: dict[str, object] = {
            "radius": 1.0,
            "center_id": 1,
            "ids": [2, 3],
            "spokes": [0, 1],
        }
        fields.update(mutation)
        request = co.RadialLayoutRequest(**fields)
        assert co.validate(request) == expected
        with pytest.raises(ValueError, match=message):
            co.build_radial_layout(request)

    @pytest.mark.parametrize(
        "fields",
        [
            {"ids": [2]},
            {"spokes": [0]},
        ],
    )
    def test_ids_and_spokes_must_be_provided_together(self, fields: dict[str, object]) -> None:
        with pytest.raises(ValueError, match="must be provided together"):
            co.RadialLayoutRequest(radius=1.0, **fields)

    def test_custom_center_requires_an_advanced_layout(self) -> None:
        with pytest.raises(ValueError, match="center_id may only be changed"):
            co.RadialLayoutRequest(radius=1.0, center_id=10)

    def test_target_without_spoke_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="has no position"):
            co.RadialLayoutRequest(radius=1.0, center_id=1, ids=[2, 3], spokes=[0])

    def test_indexed_accessor_stops_at_count(self) -> None:
        layout = _ring(count=3)
        assert layout.target(2).id == 4
        with pytest.raises(IndexError):
            layout.target(3)


class TestContainment:
    REGION = (5.0, 3.0)
    TARGET = (10.0, -4.0)

    def _region(self) -> object:
        return co.AcceptanceRegion(half_extent_x=self.REGION[0], half_extent_y=self.REGION[1])

    def _inside(self, cursor: tuple[float, float], extent: float = 1.0) -> bool:
        status, inside = co.contains_cursor(
            self._region(),
            co.CursorGeometry(extent=extent),
            co.WorkspacePoint(*self.TARGET),
            co.WorkspacePoint(*cursor),
        )
        assert status == ContractStatus.OK
        return inside

    def test_centre_is_inside(self) -> None:
        assert self._inside(self.TARGET)

    @pytest.mark.parametrize(
        "cursor", [(14.0, -4.0), (6.0, -4.0), (10.0, -2.0), (10.0, -6.0), (6.0, -2.0)]
    )
    def test_boundary_is_inside(self, cursor: tuple[float, float]) -> None:
        assert self._inside(cursor)

    @pytest.mark.parametrize(
        "cursor",
        [
            (math.nextafter(14.0, 20.0), -4.0),
            (math.nextafter(6.0, 0.0), -4.0),
            (10.0, math.nextafter(-2.0, 0.0)),
            (10.0, math.nextafter(-6.0, -20.0)),
        ],
    )
    def test_one_ulp_past_boundary_is_outside(self, cursor: tuple[float, float]) -> None:
        assert not self._inside(cursor)

    @pytest.mark.parametrize("cursor", [(100.0, -4.0), (10.0, 100.0), (0.0, 0.0)])
    def test_well_outside_is_outside(self, cursor: tuple[float, float]) -> None:
        assert not self._inside(cursor)

    def test_hit_test_uses_containment_not_overlap(self) -> None:
        # A cursor whose edge merely touches the region is outside. A distance
        # test or an overlap test would accept this one.
        assert not self._inside((15.5, -4.0))

    def test_point_containment_is_zero_extent_case(self) -> None:
        region = self._region()
        target = co.WorkspacePoint(*self.TARGET)
        for cursor in [(15.0, -1.0), (15.000001, -1.0), (10.0, -4.0)]:
            point = co.contains_point(region, target, co.WorkspacePoint(*cursor))
            body = co.contains_cursor(
                region, co.CursorGeometry(extent=0.0), target, co.WorkspacePoint(*cursor)
            )
            assert point == body

    def test_cursor_wider_than_region_is_outside(self) -> None:
        # The predicate stays total; the configuration is what gets refused.
        assert not self._inside(self.TARGET, extent=9.0)

    @pytest.mark.parametrize(
        ("region", "cursor_extent", "target", "cursor", "expected"),
        [
            ((0.0, 3.0), 1.0, (0.0, 0.0), (0.0, 0.0), ContractStatus.PARAMETER_OUT_OF_RANGE),
            ((-1.0, 3.0), 1.0, (0.0, 0.0), (0.0, 0.0), ContractStatus.PARAMETER_OUT_OF_RANGE),
            ((5.0, 3.0), -0.5, (0.0, 0.0), (0.0, 0.0), ContractStatus.PARAMETER_OUT_OF_RANGE),
            (
                (5.0, 3.0),
                1.0,
                (float("nan"), 0.0),
                (0.0, 0.0),
                ContractStatus.VALUE_NOT_FINITE,
            ),
            (
                (5.0, 3.0),
                1.0,
                (0.0, 0.0),
                (0.0, float("inf")),
                ContractStatus.VALUE_NOT_FINITE,
            ),
            (
                (float("nan"), 3.0),
                1.0,
                (0.0, 0.0),
                (0.0, 0.0),
                ContractStatus.VALUE_NOT_FINITE,
            ),
        ],
    )
    def test_invalid_geometry_is_reported(
        self,
        region: tuple[float, float],
        cursor_extent: float,
        target: tuple[float, float],
        cursor: tuple[float, float],
        expected: ContractStatus,
    ) -> None:
        status, _ = co.contains_cursor(
            co.AcceptanceRegion(half_extent_x=region[0], half_extent_y=region[1]),
            co.CursorGeometry(extent=cursor_extent),
            co.WorkspacePoint(*target),
            co.WorkspacePoint(*cursor),
        )
        assert status == expected

    def test_hit_test_matches_reference_containment(self) -> None:
        rng = np.random.default_rng(20260817)
        region = (5.0, 3.0)
        radii = [0.0, 0.5, 1.0, 2.5]
        targets = rng.uniform(-20.0, 20.0, size=(50, 2))
        cursors = rng.uniform(-25.0, 25.0, size=(50, 2))
        # Include positions on the boundary, where rounding matters.
        edges = np.array(
            [
                [t[0] + sx * (region[0] - r), t[1] + sy * (region[1] - r)]
                for t in targets[:10]
                for r in radii
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
            ]
        )
        checked = 0
        for radius in radii:
            for target in targets:
                for cursor in np.vstack([cursors, edges]):
                    expected = _reference_contains_cursor(
                        (float(cursor[0]), float(cursor[1])),
                        radius,
                        (float(target[0]), float(target[1])),
                        region,
                    )
                    status, inside = co.contains_cursor(
                        co.AcceptanceRegion(half_extent_x=region[0], half_extent_y=region[1]),
                        co.CursorGeometry(extent=radius),
                        co.WorkspacePoint(float(target[0]), float(target[1])),
                        co.WorkspacePoint(float(cursor[0]), float(cursor[1])),
                    )
                    assert status == ContractStatus.OK
                    assert inside == expected
                    checked += 1
        assert checked > 10_000


class TestConfiguration:
    def test_public_constructor_uses_human_units_and_scalar_shortcuts(self) -> None:
        config = co.CenterOutTask(
            geometry_unit=co.GeometryUnit.MILLIMETRES,
            layout=_ring(),
            acceptance=5.0,
            movement_timeout_seconds=(1.0, 2.0),
            selection=co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL,
            cursor_extent=2.0,
            hold_seconds=0.1,
            reward_dwell_seconds=(0.1, 0.2),
            punish_dwell_seconds=0.3,
            seed=0xC0FFEE,
        )

        assert co.validate(config) == ContractStatus.OK
        assert config.acceptance.half_extent_x == 5.0
        assert config.acceptance.half_extent_y == 5.0
        assert config.cursor.extent == 2.0
        assert tuple(config.movement_timeout_seconds) == (1.0, 2.0)
        assert config.hold_seconds == 0.1
        assert tuple(config.reward_dwell_seconds) == (0.1, 0.2)
        assert tuple(config.punish_dwell_seconds) == (0.3, 0.3)
        assert not hasattr(config, "trial_limit")

    def test_public_constructor_hides_native_schedule_fields(self) -> None:
        fields = {
            "geometry_unit": co.GeometryUnit.MILLIMETRES,
            "layout": _ring(),
            "acceptance": 5.0,
            "movement_timeout_seconds": 1.0,
            "selection": co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL,
        }
        with pytest.raises(TypeError):
            co.CenterOutTask(**fields, sampler_version=2)
        with pytest.raises(TypeError):
            co.CenterOutTask(**fields, trial_limit=10)

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"geometry_unit": co.GeometryUnit.UNSPECIFIED}, "geometry_unit"),
            ({"selection": co.TargetSelectionPolicy.UNSPECIFIED}, "selection"),
            ({"acceptance": 0.0}, "acceptance"),
            ({"movement_timeout_seconds": 0.0}, "movement_timeout_seconds"),
            ({"cursor_extent": 6.0}, "cursor_extent"),
        ],
    )
    def test_public_constructor_rejects_invalid_values_immediately(
        self, overrides: dict[str, object], message: str
    ) -> None:
        fields: dict[str, object] = {
            "geometry_unit": co.GeometryUnit.MILLIMETRES,
            "layout": _ring(),
            "acceptance": 5.0,
            "movement_timeout_seconds": 1.0,
            "selection": co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL,
        }
        fields.update(overrides)
        with pytest.raises(ValueError, match=message):
            co.CenterOutTask(**fields)

    def test_complete_configuration_is_accepted(self) -> None:
        assert co.validate(_config()) == ContractStatus.OK

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"geometry_unit": co.GeometryUnit.UNSPECIFIED}, ContractStatus.IDENTITY_MISSING),
            (
                {"selection": co.TargetSelectionPolicy.UNSPECIFIED},
                ContractStatus.IDENTITY_MISSING,
            ),
            ({"sampler_version": 0}, ContractStatus.IDENTITY_MISSING),
            (
                {"movement_timeout": co.PhaseDurations(to_center=0, to_out=1)},
                ContractStatus.PARAMETER_OUT_OF_RANGE,
            ),
            (
                {"movement_timeout": co.PhaseDurations(to_center=1, to_out=0)},
                ContractStatus.PARAMETER_OUT_OF_RANGE,
            ),
            (
                {"cursor": co.CursorGeometry(extent=5.5)},
                ContractStatus.PARAMETER_OUT_OF_RANGE,
            ),
            (
                {"acceptance": co.AcceptanceRegion(half_extent_x=0.0, half_extent_y=5.0)},
                ContractStatus.PARAMETER_OUT_OF_RANGE,
            ),
            (
                {"cursor": co.CursorGeometry(extent=float("nan"))},
                ContractStatus.VALUE_NOT_FINITE,
            ),
            (
                {"layout": co.CenterOut2DLayout(center=co.TargetPlacement(id=1), surrounding=[])},
                ContractStatus.TARGET_SET_INVALID,
            ),
        ],
    )
    def test_configuration_reports_failure(
        self, overrides: dict[str, object], expected: ContractStatus
    ) -> None:
        assert co.validate(_config(**overrides)) == expected

    @pytest.mark.parametrize(
        "overrides",
        [
            {"hold_ns": 0},
            {
                "reward_dwell": co.PhaseDurations(),
                "punish_dwell": co.PhaseDurations(),
            },
            {"cursor": co.CursorGeometry(extent=5.0)},
            {"cursor": co.CursorGeometry(extent=0.0)},
            {"trial_limit": 40},
            {"sampler_version": 999},
        ],
    )
    def test_admissible_configurations_are_accepted(self, overrides: dict[str, object]) -> None:
        # Zero hold, exact-width cursor, and recorded sampler versions are valid.
        assert co.validate(_config(**overrides)) == ContractStatus.OK

    def test_phase_accessor_selects_correct_leg(self) -> None:
        durations = co.PhaseDurations(to_center=11, to_out=22)
        assert durations.of(co.CenterOutPhase.TO_CENTER) == 11
        assert durations.of(co.CenterOutPhase.TO_OUT) == 22
        assert co.phase_duration(durations, co.CenterOutPhase.TO_OUT) == 22

    def test_trial_limit_is_optional(self) -> None:
        unlimited = _config(trial_limit=0)
        assert not co.trial_limit_reached(unlimited, 10**9)
        limited = _config(trial_limit=3)
        assert not co.trial_limit_reached(limited, 2)
        assert co.trial_limit_reached(limited, 3)
        assert co.trial_limit_reached(limited, 4)

    def test_declared_enumerator_predicates(self) -> None:
        assert co.geometry_unit_declared(co.GeometryUnit.METRES)
        assert co.center_out_phase_declared(co.CenterOutPhase.TO_OUT)
        assert co.target_selection_policy_declared(co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL)


class TestTargetSchedule:
    def test_repeat_until_success_advances_only_on_success(self) -> None:
        config = _config(selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS)
        status, first = co.select_outward_target(config, trial=7, successes=4)
        assert status == ContractStatus.OK
        # A failure leaves the success count where it was, so the next trial
        # presents the same target.
        assert co.select_outward_target(config, trial=8, successes=4)[1] == first
        assert co.select_outward_target(config, trial=9, successes=5)[1] != first

    def test_sample_each_trial_changes_after_failure(self) -> None:
        config = _config(selection=co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL)
        # The success count is not part of the key: the target moves with the
        # trial whether the previous one was a success or a failure.
        holding = [co.select_outward_target(config, trial=t, successes=4)[1] for t in range(40)]
        advancing = [co.select_outward_target(config, trial=t, successes=t)[1] for t in range(40)]
        assert holding == advancing
        assert len(set(holding)) > 1

    def test_balanced_shuffle_contains_each_direction_once_per_cycle(self) -> None:
        config = _config(selection=co.TargetSelectionPolicy.BALANCED_SHUFFLED_CYCLES, seed=29)
        count = config.layout.count
        sequence = [
            co.select_outward_target(config, trial=trial, successes=0)[1]
            for trial in range(count * 6)
        ]
        for start in range(0, len(sequence), count):
            assert sorted(sequence[start : start + count]) == list(range(count))

    def test_balanced_shuffle_prefix_counts_differ_by_at_most_one(self) -> None:
        config = _config(selection=co.TargetSelectionPolicy.BALANCED_SHUFFLED_CYCLES, seed=31)
        counts = [0] * config.layout.count
        for trial in range(53):
            status, idx = co.select_outward_target(config, trial=trial, successes=0)
            assert status == ContractStatus.OK
            counts[idx] += 1
            assert max(counts) - min(counts) <= 1

    def test_balanced_shuffle_failure_does_not_repeat_target(self) -> None:
        config = _config(selection=co.TargetSelectionPolicy.BALANCED_SHUFFLED_CYCLES)
        holding = [co.select_outward_target(config, trial=t, successes=0)[1] for t in range(8)]
        advancing = [co.select_outward_target(config, trial=t, successes=t)[1] for t in range(8)]
        assert holding == advancing

    def test_cycle_matches_reference_progression(self) -> None:
        config = _config(selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS)
        reference = _reference_target_positions(list(range(9)), 100.0)
        count = config.layout.count
        for successes in range(25):
            status, idx = co.select_outward_target(config, trial=successes, successes=successes)
            assert status == ContractStatus.OK
            reference_row = reference[successes % count + 1]
            target = config.layout.target(idx)
            np.testing.assert_allclose([target.pos.x, target.pos.y], reference_row, atol=1e-12)

    def test_same_seed_yields_same_sequence(self) -> None:
        config = _config(seed=99)
        first = [co.select_outward_target(config, trial=t, successes=0)[1] for t in range(64)]
        again = [
            co.select_outward_target(_config(seed=99), trial=t, successes=0)[1] for t in range(64)
        ]
        assert first == again

    def test_reset_replays_schedule(self) -> None:
        # Resetting is returning the ordinals to where they started; the sampler
        # holds nothing to reset, so the same trials yield the same targets.
        config = _config(seed=7)
        run = [co.select_outward_target(config, trial=t, successes=0)[1] for t in range(32)]
        _ = [co.select_outward_target(config, trial=t, successes=0)[1] for t in range(1000, 1032)]
        replay = [co.select_outward_target(config, trial=t, successes=0)[1] for t in range(32)]
        assert replay == run

    def test_different_seed_yields_different_sequence(self) -> None:
        left = [
            co.select_outward_target(_config(seed=1), trial=t, successes=0)[1] for t in range(64)
        ]
        right = [
            co.select_outward_target(_config(seed=2), trial=t, successes=0)[1] for t in range(64)
        ]
        assert left != right

    @pytest.mark.parametrize(
        "policy",
        [
            co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS,
            co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL,
            co.TargetSelectionPolicy.BALANCED_SHUFFLED_CYCLES,
        ],
    )
    def test_centre_is_never_selected(self, policy: object) -> None:
        config = _config(selection=policy)
        centre = config.layout.center.id
        for trial in range(400):
            status, idx = co.select_outward_target(config, trial=trial, successes=trial // 3)
            assert status == ContractStatus.OK
            assert 0 <= idx < config.layout.count
            assert config.layout.target(idx).id != centre

    def test_surrounding_targets_are_reachable(self) -> None:
        config = _config()
        seen = {co.select_outward_target(config, trial=t, successes=0)[1] for t in range(512)}
        assert seen == set(range(config.layout.count))

    def test_single_target_layout_answers_only_slot(self) -> None:
        for policy in (
            co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS,
            co.TargetSelectionPolicy.SAMPLE_EACH_TRIAL,
            co.TargetSelectionPolicy.BALANCED_SHUFFLED_CYCLES,
        ):
            config = _config(layout=_ring(count=1), selection=policy)
            for trial in range(20):
                assert co.select_outward_target(config, trial=trial, successes=trial)[1] == 0

    def test_invalid_configuration_selects_nothing(self) -> None:
        config = _config(selection=co.TargetSelectionPolicy.UNSPECIFIED)
        assert co.select_outward_target(config, trial=0, successes=0)[0] == (
            ContractStatus.IDENTITY_MISSING
        )

    def test_unsupported_sampler_is_rejected_only_on_draw(self) -> None:
        drawing = _config(sampler_version=999)
        assert co.select_outward_target(drawing, trial=0, successes=0)[0] == (
            ContractStatus.VERSION_UNSUPPORTED
        )
        cycling = _config(
            sampler_version=999, selection=co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS
        )
        status, idx = co.select_outward_target(cycling, trial=0, successes=3)
        assert status == ContractStatus.OK
        assert idx == 3


class TestFingerprints:
    def test_same_configuration_digests_same(self) -> None:
        assert co.configuration_fingerprint(_config()) == co.configuration_fingerprint(_config())
        assert co.layout_fingerprint(_ring()) == co.layout_fingerprint(_ring())

    @pytest.mark.parametrize(
        "overrides",
        [
            {"geometry_unit": co.GeometryUnit.METRES},
            {"acceptance": co.AcceptanceRegion(half_extent_x=5.0, half_extent_y=5.5)},
            {"cursor": co.CursorGeometry(extent=2.5)},
            {"movement_timeout": co.PhaseDurations(to_center=2, to_out=1_000_000_000)},
            {"hold_ns": 100_000_001},
            {"reward_dwell": co.PhaseDurations(to_center=100_000_000, to_out=1)},
            {"punish_dwell": co.PhaseDurations(to_center=1, to_out=200_000_000)},
            {"selection": co.TargetSelectionPolicy.REPEAT_UNTIL_SUCCESS},
            {"seed": 0xC0FFEF},
            {"sampler_version": 2},
            {"trial_limit": 40},
            {"layout": _ring(radius=101.0)},
            {"layout": _ring(count=7)},
        ],
    )
    def test_every_field_participates_in_digest(self, overrides: dict[str, object]) -> None:
        assert co.configuration_fingerprint(_config(**overrides)) != co.configuration_fingerprint(
            _config()
        )

    def test_order_is_part_of_layout(self) -> None:
        layout = _ring(count=4)
        targets = list(layout.surrounding)
        swapped = co.CenterOut2DLayout(
            center=layout.center, surrounding=[targets[1], targets[0], targets[2], targets[3]]
        )
        assert co.layout_fingerprint(swapped) != co.layout_fingerprint(layout)

    def test_negative_zero_equals_zero(self) -> None:
        centre = co.TargetPlacement(id=1, pos=co.WorkspacePoint(0.0, 0.0))
        negative = co.CenterOut2DLayout(
            center=centre,
            surrounding=[co.TargetPlacement(id=2, pos=co.WorkspacePoint(-0.0, 1.0))],
        )
        positive = co.CenterOut2DLayout(
            center=centre,
            surrounding=[co.TargetPlacement(id=2, pos=co.WorkspacePoint(0.0, 1.0))],
        )
        assert co.layout_fingerprint(negative) == co.layout_fingerprint(positive)


class TestStatusVocabulary:
    def test_new_status_has_own_message(self) -> None:
        from neurale.experiments import contract_status_message

        message = contract_status_message(ContractStatus.TARGET_SET_INVALID)
        assert message
        assert message != contract_status_message(ContractStatus.DIMENSION_INVALID)
        assert message != contract_status_message(ContractStatus.OUTCOME_INVALID)
