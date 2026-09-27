#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Row, timestamp, and clock alignment checks shared by decoders.

Decoders *verify* alignment; they never establish it. Nothing here shifts,
resamples, interpolates, or converts a timeline: a target that does not already
line up with the feature frames is a caller error, because a decoder silently
repairing it would decide, invisibly, which frame a label belongs to.
"""

from __future__ import annotations

import numpy as np

from neurale.data import Clock, FeatureMatrix
from neurale.exceptions import ValidationError

# Timestamps derived through different but equivalent arithmetic differ by a
# few units in the last place, and that is all the slack allowed. The tolerance
# is therefore purely absolute: eight ULPs at the represented time scale, with
# a floor for timestamps near zero.
#
# There is deliberately no relative term. A relative tolerance is a tolerance
# on the *magnitude* of a timestamp, and a timestamp's magnitude is an epoch
# choice, not a precision. At Unix-epoch scale even 1e-12 relative is 1.7 ms --
# a whole observation bin of a 1 kHz feature stream would pass as roundoff.
#
# What the absolute term admits at that scale is what float64 cannot resolve
# there: one ULP of 1.7e9 s is about 0.4 us, so eight of them are a few
# microseconds. That is the representation's limit, not a policy. Session
# relative timestamps admit correspondingly less -- picoseconds at t < 1 s.
_TIME_ABSOLUTE_FLOOR = 1e-12
_TIME_ULP_FACTOR = 8.0

_CLOCK_IDENTITY_FIELDS = (
    "name",
    "type",
    "rate",
    "epoch",
    "offset",
    "drift",
    "synchronization_domain",
)


def feature_clock(X: FeatureMatrix) -> Clock | None:
    """Return the clock a feature matrix declares, if it declares one.

    :class:`~neurale.data.arrays.FeatureMatrix` has no clock field, so an
    extraction records one in ``attrs["clock"]``. Anything else stored under
    that key is a metadata error rather than an absent declaration.
    """

    if "clock" not in X.attrs:
        return None
    clock = X.attrs["clock"]
    if clock is not None and not isinstance(clock, Clock):
        raise ValidationError("X.attrs['clock'] must be a Clock or None.")
    return clock


def times_match(left: np.ndarray, right: np.ndarray) -> bool:
    """Report whether two timestamp vectors describe the same instants."""

    if left.shape != right.shape:
        return False
    scale = 0.0
    for values in (left, right):
        if values.size:
            scale = max(scale, float(np.max(np.abs(values))))
    tol = max(
        _TIME_ABSOLUTE_FLOOR,
        _TIME_ULP_FACTOR * float(np.finfo(np.float64).eps) * scale,
    )
    return bool(np.allclose(left, right, rtol=0.0, atol=tol))


def require_aligned(
    X: FeatureMatrix,
    *,
    n_rows: int,
    time: np.ndarray,
    clock: Clock | None,
    name: str = "y",
) -> None:
    """Raise unless a target already lines up with the feature frames."""

    if n_rows != X.n_frames:
        raise ValidationError(
            f"{name} must have one row per feature frame; got {n_rows} rows for "
            f"{X.n_frames} frames."
        )
    if not times_match(np.asarray(X.time, dtype=np.float64), np.asarray(time)):
        raise ValidationError(
            f"{name} timestamps must match the feature frame timestamps. Align the "
            "target to the frames before fitting; a decoder performs no resampling "
            "or time shifting."
        )
    require_same_clock(feature_clock(X), clock, name=name, reference="X")


def require_same_clock(
    left: Clock | None,
    right: Clock | None,
    *,
    name: str,
    reference: str,
) -> None:
    """Raise when two declared clocks describe different timelines.

    A clock that is not declared on one of the two sides is not checked: the
    absence of metadata is not evidence of a conflict. Two declared clocks that
    differ are a conflict, and converting between them is the caller's job --
    see :func:`neurale.data.convert_time`.
    """

    if left is None or right is None:
        return
    if _identity(left) != _identity(right):
        raise ValidationError(
            f"clock mismatch: {name} on {right.name!r}, {reference} on {left.name!r}. "
            "Convert the timestamps to one clock first; a decoder performs no clock "
            "conversion."
        )


def _identity(clock: Clock) -> tuple[object, ...]:
    return tuple(getattr(clock, attribute) for attribute in _CLOCK_IDENTITY_FIELDS)
