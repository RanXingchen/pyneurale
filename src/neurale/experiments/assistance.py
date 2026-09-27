#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Shared velocity assistance for velocity-based BCI paradigms.

This submodule owns generic velocity shared control: the linear blend of an
external command with a guidance velocity, and the orthogonal-impedance
projection onto a desired manifold. It is applicable to any paradigm whose
movement command is a velocity vector, and it is owned neither by Center-Out,
nor by WebGrid, nor by a decoder, nor by an actuator.

It stays a submodule of :mod:`neurale.experiments` rather than becoming a
top-level ``neurale.control`` domain, because a small amount of velocity
shared-control mathematics is not enough to form one. That decision is rule 7 of
the experiments contract.

The transforms are blind. They receive vectors and parameters and return a
vector; they never learn target geometry, trial state, task phase, GUI state, or
experiment identity. A paradigm produces task-specific guidance, orchestration
composes guidance with the transform here and hands the result to the existing
runtime command path, and nothing in this module arms or releases a
``SafetyController``, changes a deadline policy, or retries an expired command.

The velocity vocabulary is not restated here. A velocity belongs to a
:class:`~neurale.experiments.CommandSpace`, which already fixes the dimension,
the per-axis names and their order, the units, and the coordinate frame, and
every transform validates its inputs against one prepared space. That is what
makes "dimensions, units and frame match exactly" a check rather than a hope.

``VelocityVector`` is deliberately not a ``CommandRequest``: a request carries
trial identity, an emission ordinal, and times, and handing those to a blind
transform would leave nothing but a comment stopping it from reading them.

Realtime: ``blend_velocity`` is bounded and allocates nothing, and a native test
asserts the second part. ``apply_ortho_impedance`` is also bounded and allocates
nothing, but no latency measurement has been executed for it and none is
claimed; it is not on the realtime data plane. Any future native adapter that
puts either on a streaming thread belongs to the integration layer, not here.

Importing this module is lightweight: the native extension loads on first use of
a name, not on import.
"""

from __future__ import annotations

_NATIVE_EXPORTS = {
    "AssistanceMethod",
    "DesiredVelocitySet",
    "LINEAR_BLEND_VERSION_1",
    "LinearAssistance",
    "LinearAssistanceRecord",
    "MAX_ASSISTANCE",
    "MAX_DESIRED_VELOCITIES",
    "MIN_ASSISTANCE",
    "ORTHO_IMPEDANCE_VERSION_1",
    "OrthoImpedanceParameters",
    "OrthoImpedanceRecord",
    "VelocityVector",
    "apply_ortho_impedance",
    "assistance_in_range",
    "assistance_method_declared",
    "blend_velocity",
    "manifold_fingerprint",
    "validate",
    "validate_against",
}


def __getattr__(name: str) -> object:
    if name not in _NATIVE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from neurale._native_loader import load_native_namespace

    value = getattr(load_native_namespace("experiments.assistance"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | _NATIVE_EXPORTS)


__all__ = sorted(_NATIVE_EXPORTS)
