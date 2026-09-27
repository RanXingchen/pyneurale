#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared velocity assistance and numerical reference checks."""

from __future__ import annotations

import math

import numpy as np
import pytest

from neurale.experiments import (
    MAX_COMMAND_DIMENSION,
    CommandAxis,
    CommandAxisName,
    CommandFrame,
    CommandSpace,
    CommandUnit,
    ContractStatus,
    TrialIdentity,
    contract_status_message,
)
from neurale.experiments.assistance import (
    LINEAR_BLEND_VERSION_1,
    MAX_ASSISTANCE,
    MAX_DESIRED_VELOCITIES,
    MIN_ASSISTANCE,
    ORTHO_IMPEDANCE_VERSION_1,
    AssistanceMethod,
    DesiredVelocitySet,
    LinearAssistance,
    LinearAssistanceRecord,
    OrthoImpedanceParameters,
    OrthoImpedanceRecord,
    VelocityVector,
    apply_ortho_impedance,
    assistance_in_range,
    assistance_method_declared,
    blend_velocity,
    manifold_fingerprint,
    validate,
    validate_against,
)

_AXIS_NAMES = (
    CommandAxisName.X,
    CommandAxisName.Y,
    CommandAxisName.Z,
    CommandAxisName.ROLL,
    CommandAxisName.PITCH,
    CommandAxisName.YAW,
    CommandAxisName.GRASP,
    CommandAxisName.GRASP,
)


def _space(
    dim: int = 2,
    *,
    identifier: int = 1,
    unit: CommandUnit = CommandUnit.METRES_PER_SECOND,
    frame: CommandFrame = CommandFrame.WORKSPACE_2D,
) -> CommandSpace:
    return CommandSpace(
        id=identifier,
        dim=dim,
        frame=frame,
        axes=[CommandAxis(name, unit) for name in _AXIS_NAMES[:dim]],
    )


def _velocity(values: list[float], *, identifier: int = 1) -> VelocityVector:
    return VelocityVector(space=identifier, dim=len(values), values=values)


def _manifold(rows: list[list[float]], *, identifier: int = 1) -> DesiredVelocitySet:
    return DesiredVelocitySet(
        space=identifier,
        dim=len(rows[0]),
        count=len(rows),
        vectors=rows,
    )


def _parameters(
    domain: list[bool], impedance: list[float] | None = None
) -> OrthoImpedanceParameters:
    return OrthoImpedanceParameters(
        dim=len(domain),
        domain=domain,
        impedance=[0.0] * len(domain) if impedance is None else impedance,
    )


def _reference_positive_span(desired: np.ndarray, command: np.ndarray, ndims: int) -> np.ndarray:
    valid = desired[np.any(np.abs(desired) > np.finfo(desired.dtype).eps, axis=1)]
    if not valid.size:
        return np.zeros((ndims, ndims))

    directions = valid / np.linalg.norm(valid, axis=1, keepdims=True)
    active = np.ones(len(directions), dtype=bool)
    residual = command.copy()
    selected: list[np.ndarray] = []
    for _ in range(len(directions)):
        scores = np.where(active, directions @ residual, -np.inf)
        choice = int(np.argmax(scores))
        if scores[choice] <= 0.0:
            break
        direction = directions[choice]
        selected.append(direction)
        residual -= (residual @ direction / (direction @ direction)) * direction
        active[choice] = False
        active &= (directions @ residual) > 0.0

    if not selected:
        return np.zeros((ndims, ndims))
    basis = np.stack(selected)
    eigenvalues, eigenvectors = np.linalg.eigh(basis @ basis.T)
    cutoff = max(1e-8, 5e-6 * eigenvalues[-1])
    inverse = np.zeros_like(eigenvalues)
    kept = eigenvalues >= cutoff
    inverse[kept] = 1.0 / eigenvalues[kept]
    return basis.T @ (eigenvectors * inverse) @ eigenvectors.T @ basis


def _reference_apply(desired: np.ndarray, command: np.ndarray, impedance: np.ndarray) -> np.ndarray:
    projector = _reference_positive_span(desired, command, command.size)
    parallel = projector @ command
    return parallel + (1.0 - impedance) * (command - parallel)


def _reference_ortho(
    desired: np.ndarray, command: np.ndarray, impedance: np.ndarray, domain: list[bool]
) -> np.ndarray:
    """Evaluate the masked sub-problem and scatter results into place."""
    axes = [axis for axis, inside in enumerate(domain) if inside]
    masked = _reference_apply(desired[:, axes], command[axes], impedance[axes])
    result = command.copy()
    for i, axis in enumerate(axes):
        result[axis] = masked[i]
    return result


class TestVelocityValues:
    def test_velocity_belongs_to_one_prepared_space(self) -> None:
        space = _space(2)
        vel = _velocity([0.25, -0.5])

        assert validate(vel) == ContractStatus.OK
        assert validate_against(vel, space) == ContractStatus.OK
        assert vel.value(0) == 0.25
        assert vel.value(1) == -0.5

    def test_missing_component_is_rejected(self) -> None:
        # The same rule the command records state: a component nobody supplied
        # and one deliberately zero are the same bytes once written.
        with pytest.raises(ValueError):
            VelocityVector(space=1, dim=2, values=[0.25])
        with pytest.raises(ValueError):
            VelocityVector(space=1, dim=2, values=[0.0] * (MAX_COMMAND_DIMENSION + 1))
        assert VelocityVector(space=1, dim=2, values=[0.25, 0.0]).value(1) == 0.0

    def test_unset_and_non_finite_velocities_are_rejected(self) -> None:
        assert validate(VelocityVector()) == ContractStatus.IDENTITY_MISSING
        for bad in (math.nan, math.inf, -math.inf):
            assert validate(_velocity([0.25, bad])) == ContractStatus.VALUE_NOT_FINITE

    def test_velocity_axis_index_is_bounded(self) -> None:
        with pytest.raises(IndexError):
            _velocity([0.25, -0.5]).value(MAX_COMMAND_DIMENSION)


class TestSpaceAgreement:
    def test_dimension_must_match_space(self) -> None:
        space = _space(3)
        assert validate_against(_velocity([0.25, -0.5]), space) == ContractStatus.DIMENSION_INVALID

    def test_velocity_from_another_space_is_rejected(self) -> None:
        # Units and frame are properties of the space, so a velocity that names
        # a different space is exactly a unit or frame mismatch. Prepare two
        # spaces that differ only in units, then only in frame.
        metres = _space(2, identifier=1, unit=CommandUnit.METRES_PER_SECOND)
        normalized = _space(2, identifier=2, unit=CommandUnit.NORMALIZED)
        assert metres.axis(0).unit != normalized.axis(0).unit

        external = _velocity([0.25, -0.5], identifier=1)
        guidance = _velocity([1.0, 0.0], identifier=2)
        status, _ = blend_velocity(metres, external, guidance, LinearAssistance(0.5))
        assert status == ContractStatus.IDENTITY_MISSING

        device_frame = _space(2, identifier=3, frame=CommandFrame.DEVICE_NATIVE)
        assert device_frame.frame != metres.frame
        status, _ = blend_velocity(
            metres, external, _velocity([1.0, 0.0], identifier=3), LinearAssistance(0.5)
        )
        assert status == ContractStatus.IDENTITY_MISSING


class TestLinearBlend:
    @pytest.mark.parametrize("assistance", [0.0, 0.125, 0.5, 0.75, 1.0])
    def test_blend_matches_declared_arithmetic(self, assistance: float) -> None:
        space = _space(3)
        external = [0.25, -0.5, 2.0]
        guidance = [1.0, 0.0, -1.0]

        status, assisted = blend_velocity(
            space, _velocity(external), _velocity(guidance), LinearAssistance(assistance)
        )
        assert status == ContractStatus.OK

        expected = (1 - assistance) * np.array(external) + assistance * np.array(guidance)
        assert np.allclose(assisted.values[:3], expected, rtol=0, atol=1e-15)
        assert assisted.space == space.id
        assert assisted.dim == space.dim

    def test_endpoints_are_identity_on_stored_values(self) -> None:
        space = _space(2)
        external = _velocity([0.25, -0.5])
        guidance = _velocity([1.0, 0.0])

        status, none_at_all = blend_velocity(space, external, guidance, LinearAssistance(0.0))
        assert status == ContractStatus.OK
        assert none_at_all.values == external.values

        status, all_of_it = blend_velocity(space, external, guidance, LinearAssistance(1.0))
        assert status == ContractStatus.OK
        assert all_of_it.values == guidance.values

    def test_values_are_not_clipped_or_normalized(self) -> None:
        # A guidance velocity larger than any bound comes through at its own
        # size. The transform has no idea what a plausible velocity is, and a
        # transform that silently rescaled one would be deciding that.
        space = _space(2)
        status, assisted = blend_velocity(
            space, _velocity([0.0, 0.0]), _velocity([1e6, -1e6]), LinearAssistance(1.0)
        )
        assert status == ContractStatus.OK
        assert assisted.value(0) == 1e6
        assert assisted.value(1) == -1e6

    @pytest.mark.parametrize("assistance", [-1e-9, -1.0, 1.0 + 1e-9, 2.0, math.nan, math.inf])
    def test_assistance_outside_interval_is_rejected(self, assistance: float) -> None:
        assert not assistance_in_range(assistance)
        assert validate(LinearAssistance(assistance)) == ContractStatus.ASSISTANCE_OUT_OF_RANGE

        space = _space(2)
        status, assisted = blend_velocity(
            space, _velocity([0.25, -0.5]), _velocity([1.0, 0.0]), LinearAssistance(assistance)
        )
        assert status == ContractStatus.ASSISTANCE_OUT_OF_RANGE
        # A refused call returns no velocity to mistake for one.
        assert assisted.space == 0

    def test_declared_interval_is_unit_interval(self) -> None:
        assert (MIN_ASSISTANCE, MAX_ASSISTANCE) == (0.0, 1.0)
        assert assistance_in_range(MIN_ASSISTANCE)
        assert assistance_in_range(MAX_ASSISTANCE)

    def test_non_finite_command_is_rejected(self) -> None:
        space = _space(2)
        for bad in (math.nan, math.inf):
            status, _ = blend_velocity(
                space, _velocity([0.25, bad]), _velocity([1.0, 0.0]), LinearAssistance(0.5)
            )
            assert status == ContractStatus.VALUE_NOT_FINITE

    def test_blend_is_stateless(self) -> None:
        space = _space(2)
        external = _velocity([0.25, -0.5])
        guidance = _velocity([1.0, 0.0])
        before = (list(external.values), list(guidance.values))

        results = [
            blend_velocity(space, external, guidance, LinearAssistance(0.4)) for _ in range(5)
        ]
        assert all(status == ContractStatus.OK for status, _ in results)
        assert all(assisted.values == results[0][1].values for _, assisted in results)
        assert (list(external.values), list(guidance.values)) == before


class TestOrthoImpedance:
    def test_zero_impedance_is_identity(self) -> None:
        space = _space(2)
        external = _velocity([0.6, 0.8])
        status, assisted = apply_ortho_impedance(
            space, _manifold([[1.0, 0.0]]), external, _parameters([True, True])
        )
        assert status == ContractStatus.OK
        assert np.allclose(assisted.values[:2], [0.6, 0.8])

    def test_full_impedance_leaves_manifold_component(self) -> None:
        space = _space(2)
        status, assisted = apply_ortho_impedance(
            space,
            _manifold([[1.0, 0.0]]),
            _velocity([0.6, 0.8]),
            _parameters([True, True], [1.0, 1.0]),
        )
        assert status == ContractStatus.OK
        assert np.allclose(assisted.values[:2], [0.6, 0.0])

    def test_partial_impedance_scales_orthogonal_remainder(self) -> None:
        space = _space(2)
        status, assisted = apply_ortho_impedance(
            space,
            _manifold([[1.0, 0.0]]),
            _velocity([0.6, 0.8]),
            _parameters([True, True], [0.25, 0.25]),
        )
        assert status == ContractStatus.OK
        assert np.allclose(assisted.values[:2], [0.6, 0.6])

    @pytest.mark.parametrize(
        ("desired", "external"),
        [
            # Nothing to project onto: every desired direction is zero.
            ([[0.0, 0.0]], [0.6, 0.8]),
            # The command opposes the only direction there is.
            ([[1.0, 0.0]], [-0.6, 0.8]),
        ],
    )
    def test_without_span_result_is_impedance_alone(
        self, desired: list[list[float]], external: list[float]
    ) -> None:
        space = _space(2)
        status, assisted = apply_ortho_impedance(
            space,
            _manifold(desired),
            _velocity(external),
            _parameters([True, True], [0.25, 0.5]),
        )
        assert status == ContractStatus.OK
        assert np.allclose(assisted.values[:2], np.array([0.75, 0.5]) * np.array(external))

    def test_zero_command_stays_zero(self) -> None:
        space = _space(2)
        status, assisted = apply_ortho_impedance(
            space,
            _manifold([[1.0, 0.0], [0.0, 1.0]]),
            _velocity([0.0, 0.0]),
            _parameters([True, True], [0.5, 0.5]),
        )
        assert status == ContractStatus.OK
        assert assisted.values[:2] == [0.0, 0.0]

    def test_rank_deficient_manifold_projects_onto_its_span(self) -> None:
        space = _space(2)
        repeated = _manifold([[1.0, 0.0], [3.0, 0.0], [0.5, 0.0]])
        single = _manifold([[1.0, 0.0]])
        parameters = _parameters([True, True], [1.0, 1.0])
        external = _velocity([0.6, 0.8])

        status, from_repeated = apply_ortho_impedance(space, repeated, external, parameters)
        assert status == ContractStatus.OK
        status, from_single = apply_ortho_impedance(space, single, external, parameters)
        assert status == ContractStatus.OK
        assert np.allclose(from_repeated.values[:2], from_single.values[:2])
        assert np.allclose(from_repeated.values[:2], [0.6, 0.0])

    def test_full_span_leaves_command_alone(self) -> None:
        # Two orthogonal directions span the plane, so nothing is orthogonal to
        # the manifold and the impedance has nothing to act on.
        space = _space(2)
        status, assisted = apply_ortho_impedance(
            space,
            _manifold([[1.0, 0.0], [0.0, 1.0]]),
            _velocity([0.6, 0.8]),
            _parameters([True, True], [1.0, 1.0]),
        )
        assert status == ContractStatus.OK
        assert np.allclose(assisted.values[:2], [0.6, 0.8])

    def test_result_matches_reference_algorithm(self) -> None:
        generator = np.random.default_rng(20260817)
        worst = 0.0
        for case in range(300):
            dim = int(generator.integers(2, MAX_COMMAND_DIMENSION + 1))
            count = int(generator.integers(1, MAX_DESIRED_VELOCITIES + 1))
            desired = generator.normal(size=(count, dim))
            if case % 7 == 0:
                desired[0] = 0.0
            if case % 11 == 0 and count > 1:
                desired[1] = desired[0] * 2.0
            # Near-collinear rows: a small perturbation makes the Gram matrix
            # ill-conditioned so the conditioning cut actually fires, which the
            # well-conditioned random cases above never do.
            if case % 9 == 0 and count > 1:
                desired[1] = desired[0] + 1e-3 * generator.normal(size=dim)
            command = generator.normal(size=dim)
            if case % 13 == 0:
                command[:] = 0.0
            impedance = generator.random(dim)

            status, assisted = apply_ortho_impedance(
                _space(dim),
                _manifold([list(row) for row in desired]),
                _velocity(list(command)),
                _parameters([True] * dim, list(impedance)),
            )
            assert status == ContractStatus.OK

            expected = _reference_apply(desired, command, impedance)
            got = np.array(assisted.values[:dim])
            worst = max(worst, float(np.max(np.abs(got - expected))))
            assert np.allclose(got, expected, rtol=1e-9, atol=1e-12)
        # Two different factorizations of the same problem, so the agreement is
        # numerical rather than exact; this pins how far apart they are allowed
        # to drift before the assertion above stops meaning anything.
        assert worst < 1e-9

    def test_masked_domain_keeps_coordinate_order(self) -> None:
        space = _space(3)
        domain = [True, False, True]
        status, assisted = apply_ortho_impedance(
            space,
            _manifold([[1.0, 0.0, 0.0]]),
            _velocity([0.6, 5.0, 0.8]),
            _parameters(domain, [1.0, 0.0, 1.0]),
        )
        assert status == ContractStatus.OK
        assert assisted.value(1) == 5.0
        assert np.allclose([assisted.value(0), assisted.value(2)], [0.6, 0.0])

    def test_masked_domain_matches_sub_problem(self) -> None:
        generator = np.random.default_rng(555)
        for _ in range(200):
            dim = int(generator.integers(2, 5))
            domain = [bool(flag) for flag in generator.integers(0, 2, size=dim)]
            if not any(domain):
                domain[0] = True
            count = int(generator.integers(1, 4))
            desired = generator.normal(size=(count, dim))
            command = generator.normal(size=dim)
            impedance = generator.random(dim)

            status, assisted = apply_ortho_impedance(
                _space(dim),
                _manifold([list(row) for row in desired]),
                _velocity(list(command)),
                _parameters(domain, list(impedance)),
            )
            assert status == ContractStatus.OK
            expected = _reference_ortho(desired, command, impedance, domain)
            assert np.allclose(assisted.values[:dim], expected, rtol=1e-9, atol=1e-12)

    def test_manifold_outside_domain_takes_no_part(self) -> None:
        space = _space(3)
        parameters = _parameters([True, False, True], [1.0, 0.0, 1.0])
        external = _velocity([0.6, 5.0, 0.8])

        status, flat = apply_ortho_impedance(
            space, _manifold([[1.0, 0.0, 0.0]]), external, parameters
        )
        assert status == ContractStatus.OK
        status, leaning = apply_ortho_impedance(
            space, _manifold([[1.0, 100.0, 0.0]]), external, parameters
        )
        assert status == ContractStatus.OK
        assert flat.values == leaning.values

    def test_projection_is_stateless(self) -> None:
        space = _space(2)
        desired = _manifold([[1.0, 0.0], [0.0, 1.0]])
        external = _velocity([0.25, -0.5])
        parameters = _parameters([True, True], [0.75, 0.25])
        before = (
            [list(row) for row in desired.vectors],
            list(external.values),
            list(parameters.impedance),
            list(parameters.domain),
        )

        results = [apply_ortho_impedance(space, desired, external, parameters) for _ in range(5)]
        assert all(status == ContractStatus.OK for status, _ in results)
        assert all(assisted.values == results[0][1].values for _, assisted in results)
        assert (
            [list(row) for row in desired.vectors],
            list(external.values),
            list(parameters.impedance),
            list(parameters.domain),
        ) == before

    def test_near_collinear_manifold_keeps_dominant_direction(self) -> None:
        # Two nearly collinear desired directions make the span's Gram matrix
        # ill-conditioned: one eigenvalue is near zero. The conditioning cut must
        # drop that small eigenvalue and keep the dominant direction. With full
        # impedance the command, which lies in the span, comes back essentially
        # unchanged; the wrong-direction cut annihilated it instead.
        space = _space(2)
        status, assisted = apply_ortho_impedance(
            space,
            _manifold([[1.0, 0.0], [1.0, 1e-3]]),
            _velocity([2.0, 1e-3]),
            _parameters([True, True], [1.0, 1.0]),
        )
        assert status == ContractStatus.OK
        assert np.allclose(assisted.values[:2], [2.0, 1e-3], atol=1e-6)

    def test_rank_tol_zero_stays_finite(self) -> None:
        # A zero rank_tol disables the absolute floor and a zero
        # conditioning_tol disables the relative cut, so the near-collinear
        # manifold's tiny eigenvalue is inverted with a large gain. The result
        # must stay finite: a zero eigenvalue is never divided by. The span is
        # still full-rank 2D, so the projector is the identity.
        space = _space(2)
        parameters = _parameters([True, True], [1.0, 1.0])
        parameters = OrthoImpedanceParameters(
            dim=2,
            domain=[True, True],
            impedance=[1.0, 1.0],
            rank_tol=0.0,
            conditioning_tol=0.0,
        )
        status, assisted = apply_ortho_impedance(
            space, _manifold([[1.0, 0.0], [1.0, 1e-3]]), _velocity([2.0, 1e-3]), parameters
        )
        assert status == ContractStatus.OK
        assert np.all(np.isfinite(assisted.values[:2]))
        assert np.allclose(assisted.values[:2], [2.0, 1e-3], atol=1e-6)

    def test_full_rank_identity_span_is_unchanged(self) -> None:
        # Eight orthonormal desired directions span the whole 8D space, so
        # nothing is orthogonal to the manifold and the projector is the
        # identity: full impedance returns the command unchanged. This also
        # exercises the largest manifold the contract allows (count = dim = 8).
        space = _space(8)
        status, assisted = apply_ortho_impedance(
            space,
            _manifold([[1.0 if r == c else 0.0 for c in range(8)] for r in range(8)]),
            _velocity([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            _parameters([True] * 8, [1.0] * 8),
        )
        assert status == ContractStatus.OK
        assert np.allclose(
            assisted.values[:8], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], atol=1e-12
        )

    def test_row_ordering_does_not_change_projection(self) -> None:
        # Permuting the rows of a manifold does not change its span, so it must
        # not change the projection.
        space = _space(2)
        external = _velocity([0.6, 0.8])
        parameters = _parameters([True, True], [0.5, 0.25])
        status_a, a = apply_ortho_impedance(
            space, _manifold([[1.0, 0.0], [0.0, 1.0]]), external, parameters
        )
        status_b, b = apply_ortho_impedance(
            space, _manifold([[0.0, 1.0], [1.0, 0.0]]), external, parameters
        )
        assert status_a == ContractStatus.OK and status_b == ContractStatus.OK
        assert a.values == b.values


class TestOrthoImpedanceInputs:
    def test_empty_domain_is_rejected(self) -> None:
        # A caller that meant "no assistance" has AssistanceMethod.NONE. A
        # domain selecting no axis is a mistake, not a way of asking for it.
        assert validate(_parameters([False, False])) == ContractStatus.DOMAIN_MASK_INVALID

    def test_domain_naming_absent_axis_is_rejected(self) -> None:
        parameters = OrthoImpedanceParameters(
            dim=2, domain=[True, False, True], impedance=[0.0, 0.0, 0.0]
        )
        assert validate(parameters) == ContractStatus.DOMAIN_MASK_INVALID

    def test_parameters_must_describe_their_space(self) -> None:
        parameters = _parameters([True, True])
        assert validate_against(parameters, _space(2)) == ContractStatus.OK
        assert validate_against(parameters, _space(3)) == ContractStatus.DIMENSION_INVALID

    def test_partial_parameter_set_is_rejected(self) -> None:
        # Leaving an axis out would silently decide it is outside the domain
        # with an impedance of zero. Nobody decided that.
        with pytest.raises(ValueError):
            OrthoImpedanceParameters(dim=3, domain=[True, True], impedance=[0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            OrthoImpedanceParameters(dim=3, domain=[True] * 3, impedance=[0.0, 0.0])

    @pytest.mark.parametrize("impedance", [-1e-9, 1.0 + 1e-9, math.nan])
    def test_impedance_outside_interval_is_rejected(self, impedance: float) -> None:
        parameters = _parameters([True, True], [impedance, 0.0])
        assert validate(parameters) == ContractStatus.ASSISTANCE_OUT_OF_RANGE

    def test_tolerances_are_checked_for_finiteness_and_range(self) -> None:
        # rank_tol is a non-negative absolute floor; conditioning_tol
        # is a fraction of the largest eigenvalue, so it lies in [0, 1]. A finite
        # value outside the interval is parameter_out_of_range -- not
        # value_not_finite, which is reserved for NaN/infinite values.
        def with_tols(rank: float, cond: float) -> OrthoImpedanceParameters:
            return OrthoImpedanceParameters(
                dim=2,
                domain=[True, True],
                impedance=[0.0, 0.0],
                rank_tol=rank,
                conditioning_tol=cond,
            )

        for bad in (math.nan, math.inf, -math.inf):
            assert validate(with_tols(bad, 5e-6)) == ContractStatus.VALUE_NOT_FINITE
            assert validate(with_tols(1e-8, bad)) == ContractStatus.VALUE_NOT_FINITE

        assert validate(with_tols(-1e-9, 5e-6)) == ContractStatus.PARAMETER_OUT_OF_RANGE
        for value, ok in (
            (0.0, True),
            (1.0, True),
            (5e-6, True),
            (-1e-9, False),
            (1.0 + 1e-9, False),
            (2.0, False),
        ):
            expected = ContractStatus.OK if ok else ContractStatus.PARAMETER_OUT_OF_RANGE
            assert validate(with_tols(1e-8, value)) == expected

    def test_manifold_must_declare_its_vectors(self) -> None:
        assert validate(DesiredVelocitySet()) == ContractStatus.IDENTITY_MISSING
        assert (
            validate(DesiredVelocitySet(space=1, dim=2, count=0))
            == ContractStatus.DIMENSION_INVALID
        )
        with pytest.raises(ValueError):
            DesiredVelocitySet(space=1, dim=2, count=2, vectors=[[1.0, 0.0]])
        with pytest.raises(ValueError):
            DesiredVelocitySet(
                space=1,
                dim=2,
                count=1,
                vectors=[[1.0, 0.0]] * (MAX_DESIRED_VELOCITIES + 1),
            )

    def test_non_finite_manifold_is_rejected(self) -> None:
        assert validate(_manifold([[1.0, math.inf]])) == ContractStatus.VALUE_NOT_FINITE

    def test_manifold_from_another_space_is_rejected(self) -> None:
        status, _ = apply_ortho_impedance(
            _space(2),
            _manifold([[1.0, 0.0]], identifier=2),
            _velocity([0.6, 0.8]),
            _parameters([True, True]),
        )
        assert status == ContractStatus.IDENTITY_MISSING


class TestProvenance:
    def test_manifold_fingerprint_identifies_manifold(self) -> None:
        base = _manifold([[1.0, 0.0]])
        assert manifold_fingerprint(base) == manifold_fingerprint(_manifold([[1.0, 0.0]]))
        assert manifold_fingerprint(_manifold([[1.0, 1e-9]])) != manifold_fingerprint(base)
        assert manifold_fingerprint(_manifold([[1.0, 0.0], [0.0, 1.0]])) != manifold_fingerprint(
            base
        )
        # Negative zero and zero are the same direction, so the same manifold.
        assert manifold_fingerprint(_manifold([[1.0, -0.0]])) == manifold_fingerprint(base)

    def test_linear_record_states_method_version_and_parameter(self) -> None:
        space = _space(2)
        external = _velocity([0.25, -0.5])
        guidance = _velocity([1.0, 0.0])
        status, assisted = blend_velocity(space, external, guidance, LinearAssistance(0.4))
        assert status == ContractStatus.OK

        record = LinearAssistanceRecord(
            time_ns=1_000,
            sequence=7,
            trial=TrialIdentity(ordinal=3),
            space=space.id,
            dim=space.dim,
            assistance=0.4,
            external=list(external.values),
            guidance=list(guidance.values),
            assisted=list(assisted.values),
        )
        assert validate(record) == ContractStatus.OK
        assert record.method == AssistanceMethod.LINEAR_BLEND
        assert record.version == LINEAR_BLEND_VERSION_1

        # Everything the blend needed is in the record, so the velocity it
        # reports can be recomputed from it rather than taken on trust.
        recomputed = (1 - record.assistance) * np.array(record.external) + (
            record.assistance * np.array(record.guidance)
        )
        assert np.allclose(record.assisted, recomputed)

    def test_ortho_record_states_active_parameters(self) -> None:
        space = _space(2)
        desired = _manifold([[1.0, 0.0]])
        external = _velocity([0.6, 0.8])
        parameters = _parameters([True, True], [1.0, 1.0])
        status, assisted = apply_ortho_impedance(space, desired, external, parameters)
        assert status == ContractStatus.OK

        record = OrthoImpedanceRecord(
            time_ns=2_000,
            sequence=8,
            trial=TrialIdentity(ordinal=3),
            manifold=manifold_fingerprint(desired),
            space=space.id,
            dim=space.dim,
            rank_tol=parameters.rank_tol,
            conditioning_tol=parameters.conditioning_tol,
            domain=list(parameters.domain),
            impedance=list(parameters.impedance),
            external=list(external.values),
            assisted=list(assisted.values),
        )
        assert validate(record) == ContractStatus.OK
        assert record.method == AssistanceMethod.ORTHO_IMPEDANCE
        assert record.version == ORTHO_IMPEDANCE_VERSION_1
        assert record.manifold == manifold_fingerprint(desired)
        assert record.rank_tol == parameters.rank_tol

    def test_record_naming_wrong_method_is_rejected(self) -> None:
        record = LinearAssistanceRecord(
            space=1,
            dim=2,
            method=AssistanceMethod.ORTHO_IMPEDANCE,
            external=[0.0, 0.0],
            guidance=[0.0, 0.0],
            assisted=[0.0, 0.0],
        )
        assert validate(record) == ContractStatus.OUTCOME_INVALID

    def test_unknown_version_is_rejected(self) -> None:
        # Only the declared version is replayable. A record naming any other
        # version -- zero included -- is version_unsupported: the record is
        # structurally readable but this build cannot recompute the velocity it
        # reports, so it is not valid provenance.
        linear = dict(
            space=1,
            dim=2,
            assistance=0.5,
            external=[0.25, -0.5],
            guidance=[1.0, 0.0],
            assisted=[0.625, -0.25],
        )
        assert (
            validate(LinearAssistanceRecord(version=LINEAR_BLEND_VERSION_1, **linear))
            == ContractStatus.OK
        )
        assert (
            validate(LinearAssistanceRecord(version=0, **linear))
            == ContractStatus.VERSION_UNSUPPORTED
        )
        assert (
            validate(LinearAssistanceRecord(version=999, **linear))
            == ContractStatus.VERSION_UNSUPPORTED
        )

        ortho = dict(
            space=1,
            dim=2,
            manifold=0,
            domain=[True, True],
            impedance=[1.0, 1.0],
            rank_tol=1e-8,
            conditioning_tol=5e-6,
            external=[0.6, 0.8],
            assisted=[0.6, 0.0],
        )
        assert (
            validate(OrthoImpedanceRecord(version=ORTHO_IMPEDANCE_VERSION_1, **ortho))
            == ContractStatus.OK
        )
        assert (
            validate(OrthoImpedanceRecord(version=0, **ortho)) == ContractStatus.VERSION_UNSUPPORTED
        )
        assert (
            validate(OrthoImpedanceRecord(version=999, **ortho))
            == ContractStatus.VERSION_UNSUPPORTED
        )

    def test_method_enumeration_checks_declared_set(self) -> None:
        for method in AssistanceMethod.__members__.values():
            assert assistance_method_declared(method)

    def test_new_statuses_report_distinctly(self) -> None:
        assert contract_status_message(
            ContractStatus.ASSISTANCE_OUT_OF_RANGE
        ) != contract_status_message(ContractStatus.DOMAIN_MASK_INVALID)
        for status in (
            ContractStatus.ASSISTANCE_OUT_OF_RANGE,
            ContractStatus.DOMAIN_MASK_INVALID,
        ):
            assert contract_status_message(status)


class TestImmutability:
    _VALUE_TYPES = (
        VelocityVector,
        DesiredVelocitySet,
        LinearAssistance,
        OrthoImpedanceParameters,
        LinearAssistanceRecord,
        OrthoImpedanceRecord,
    )

    @pytest.mark.parametrize("value_type", _VALUE_TYPES, ids=lambda t: t.__name__)
    def test_no_field_can_be_set_after_construction(self, value_type: type) -> None:
        value = value_type()
        names = [name for name in dir(value) if not name.startswith("_")]
        assert names
        for name in names:
            attribute = getattr(type(value), name, None)
            if not isinstance(attribute, property):
                continue
            with pytest.raises(AttributeError):
                setattr(value, name, getattr(value, name))

    def test_result_cannot_be_edited(self) -> None:
        space = _space(2)
        status, assisted = blend_velocity(
            space, _velocity([0.25, -0.5]), _velocity([1.0, 0.0]), LinearAssistance(0.5)
        )
        assert status == ContractStatus.OK
        with pytest.raises(AttributeError):
            assisted.dim = 1
        with pytest.raises(AttributeError):
            assisted.values = [0.0, 0.0]
