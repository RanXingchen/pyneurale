#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Statistical estimators and tests used by model and feature modules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neurale._validation import (
    validate_axis,
    validate_finite,
    validate_integer,
    validate_number,
    validate_real_array,
)
from neurale.exceptions import ValidationError

from ._dispatch import require_cpu_models_operation
from ._inference import validate_2d_array


@dataclass(frozen=True, slots=True)
class DipStatistic:
    """Result of Hartigan's dip statistic calculation."""

    dip: float
    lower: float
    upper: float


@dataclass(frozen=True, slots=True)
class DipTestResult:
    """Result of Hartigan's dip test."""

    statistic: float
    p_value: float
    n_boot: int


def empirical_cov(X: np.ndarray, *, center: bool = True) -> np.ndarray:
    """Compute the maximum-likelihood empirical covariance matrix."""
    require_cpu_models_operation("statistics.empirical_cov")
    X = validate_2d_array(X)

    if not isinstance(center, bool):
        raise ValidationError("center must be a bool.")

    n_samples = X.shape[0]
    if center:
        centered = X - np.mean(X, axis=0, keepdims=True)
        covariance = centered.T @ centered / n_samples
    else:
        covariance = X.T @ X / n_samples
    return np.atleast_2d(covariance)


class LedoitWolfCov:
    """Ledoit-Wolf covariance estimator."""

    def __init__(self, *, center: bool = True, block_size: int = 1000) -> None:
        if not isinstance(center, bool):
            raise ValidationError("center must be a bool.")
        self.center = center
        self.block_size = validate_integer(block_size, "block_size", minimum=1)

        self.covariance_: np.ndarray | None = None
        self.location_: np.ndarray | None = None
        self.shrinkage_: float | None = None

    def fit(self, X: np.ndarray) -> LedoitWolfCov:
        """Fit the estimator to a sample-by-feature matrix."""
        require_cpu_models_operation("statistics.LedoitWolfCov.fit")
        X = validate_2d_array(X)

        if self.center:
            self.location_ = np.mean(X, axis=0)
            centered = X - self.location_
        else:
            self.location_ = np.zeros(X.shape[1], dtype=X.dtype)
            centered = X
        covariance, shrinkage = self._ledoit_wolf(centered)
        self.covariance_ = covariance
        self.shrinkage_ = shrinkage
        self._device = "cpu"
        return self

    @property
    def device_(self) -> str:
        if not hasattr(self, "_device"):
            raise ValidationError("LedoitWolfCov instance is not fitted.")
        return self._device

    @staticmethod
    def shrunk(covariance: np.ndarray, shrinkage: float) -> np.ndarray:
        """Return a covariance matrix shrunk toward scaled identity."""
        require_cpu_models_operation("statistics.LedoitWolfCov.shrunk")
        values = validate_real_array(covariance, "covariance", ndim=2)
        if not np.all(values.shape):
            raise ValidationError("covariance must not be empty.")
        if values.shape[0] != values.shape[1]:
            raise ValidationError("covariance must be a square matrix.")

        shrinkage = float(
            validate_number(
                shrinkage,
                "shrinkage",
                kind="real",
                minimum=0,
                maximum=1,
                coerce=True,
            )
        )
        n_features = values.shape[0]
        mu = float(np.trace(values) / n_features)
        shrunk = (1.0 - shrinkage) * values.copy()
        shrunk.flat[:: n_features + 1] += shrinkage * mu
        return shrunk

    def _ledoit_wolf(self, X: np.ndarray) -> tuple[np.ndarray, float]:
        if X.shape[1] == 1:
            return np.atleast_2d(np.mean(X**2)), 0.0

        shrinkage = self._estimate_shrinkage(X)
        covariance = empirical_cov(X, center=False)
        return self.shrunk(covariance, shrinkage), shrinkage

    def _estimate_shrinkage(self, X: np.ndarray) -> float:
        n_samples, n_features = X.shape
        if n_features == 1:
            return 0.0

        block_size = min(self.block_size, n_features)
        squared = X**2
        trace_values = np.sum(squared, axis=0) / n_samples
        mu = float(np.sum(trace_values) / n_features)
        slices = tuple(_block_slices(n_features, block_size))
        transposed = X.T
        squared_transposed = squared.T

        beta_sum = 0.0
        delta_sum = 0.0
        for row_slice in slices:
            squared_rows = squared_transposed[row_slice]
            rows = transposed[row_slice]
            for col_slice in slices:
                beta_sum += float(np.sum(squared_rows @ squared[:, col_slice]))
                delta_sum += float(np.sum((rows @ X[:, col_slice]) ** 2))

        delta_sum /= n_samples**2
        beta = (beta_sum / n_samples - delta_sum) / (n_features * n_samples)
        delta = (
            delta_sum - 2.0 * mu * float(np.sum(trace_values)) + n_features * mu**2
        ) / n_features
        if delta <= 0.0:
            return 0.0
        beta = min(max(beta, 0.0), delta)
        return float(beta / delta)


def rayleigh_test(y: np.ndarray, axis: int = 0) -> np.ndarray | float:
    """Perform the Rayleigh test of circular uniformity for radian data."""
    require_cpu_models_operation("statistics.rayleigh_test")
    values = np.asarray(validate_real_array(y, "y"), dtype=float)
    validate_finite(y, "y")

    if values.size == 0:
        raise ValidationError("y must not be empty.")
    axis = validate_axis(axis, values.ndim)
    n = values.shape[axis]
    if n < 2:
        raise ValidationError("y must contain at least two samples along axis.")

    resultant = np.sum(np.exp(1j * values), axis=axis)
    resultant_length = np.abs(resultant)
    rayleigh_z = resultant_length**2 / n
    p_value = np.exp(np.sqrt(1.0 + 4.0 * n + 4.0 * (n**2 - n * rayleigh_z)) - (1.0 + 2.0 * n))
    return float(p_value) if np.ndim(p_value) == 0 else p_value


def dip_statistic(y: np.ndarray) -> DipStatistic:
    """Compute Hartigan's dip statistic for a 1D sample.

    Requires at least four samples to match the finite-sample calibration
    used by Hartigan and Hartigan's dip test.
    """
    require_cpu_models_operation("statistics.dip_statistic")
    return _dip_statistic(y)


def _dip_statistic(y: np.ndarray) -> DipStatistic:
    """Compute the dip statistic after the public device guard."""

    values = validate_real_array(y, "y", ndim=1, finite=True)
    values = values.astype(float, copy=False)
    values = np.sort(values)

    n = values.size
    if n < 4:
        raise ValidationError("y must contain at least four samples.")

    gcm = np.zeros(n, dtype=int)
    lcm = np.zeros(n, dtype=int)
    mn = np.zeros(n, dtype=int)
    mj = np.zeros(n, dtype=int)

    if values[-1] == values[0]:
        value = float(values[0])
        return DipStatistic(0.0, value, value)

    low = 0
    high = n - 1
    dip = 1.0 / n

    _build_greatest_convex_minorant(values, mn)
    _build_least_concave_majorant(values, mj)

    while True:
        icx = _collect_gcm(high, low, mn, gcm)
        icv = _collect_lcm(low, high, mj, lcm)
        d, ig, ih = _largest_gcm_lcm_distance(values, gcm, lcm, icx, icv)

        if d <= dip:
            break

        lower_dip = _convex_minorant_dip(values, gcm, ig, icx)
        upper_dip = _concave_majorant_dip(values, lcm, ih, icv)
        dip = max(dip, lower_dip, upper_dip)
        low = int(gcm[ig])
        high = int(lcm[ih])

    return DipStatistic(
        dip=float(dip * 0.5),
        lower=float(values[low]),
        upper=float(values[high]),
    )


def dip_test(
    y: np.ndarray,
    *,
    n_boot: int = 1000,
    random_state: int | np.random.Generator | None = None,
) -> DipTestResult:
    """Estimate the right-tail p-value under the uniform reference distribution."""
    require_cpu_models_operation("statistics.dip_test")
    values = validate_real_array(y, "y", ndim=1, finite=True)
    values = values.astype(float, copy=False)

    if values.size < 4:
        raise ValidationError("y must contain at least four samples.")

    n_boot = validate_integer(n_boot, "n_boot", minimum=1)
    observed = float(_dip_statistic(values).dip)
    if observed == 0.0:
        return DipTestResult(0.0, 1.0, n_boot)

    rng = _coerce_rng(random_state)
    boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        boot[i] = _dip_statistic(rng.uniform(0.0, 1.0, values.size)).dip
    exceedances = np.count_nonzero(boot >= observed)
    p_value = float((exceedances + 1.0) / (n_boot + 1.0))
    return DipTestResult(observed, p_value, n_boot)


def _block_slices(size: int, block_size: int) -> list[slice]:
    return [slice(start, min(start + block_size, size)) for start in range(0, size, block_size)]


def _coerce_rng(
    random_state: int | np.random.Generator | None,
) -> np.random.Generator:
    if random_state is None:
        return np.random.default_rng()
    if isinstance(random_state, np.random.Generator):
        return random_state
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise ValidationError("random_state must be None, an int, or numpy.random.Generator.")
    return np.random.default_rng(int(random_state))


def _build_greatest_convex_minorant(values: np.ndarray, mn: np.ndarray) -> None:
    for i in range(1, values.size):
        mn[i] = i - 1
        current = mn[i]
        previous = mn[current]
        while current != 0 and (
            (values[i] - values[current]) * (current - previous)
            >= (values[current] - values[previous]) * (i - current)
        ):
            mn[i] = previous
            current = mn[i]
            previous = mn[current]


def _build_least_concave_majorant(values: np.ndarray, mj: np.ndarray) -> None:
    n = values.size
    mj[-1] = n - 1
    for reverse_idx in range(1, n):
        idx = n - reverse_idx - 1
        mj[idx] = idx + 1
        current = mj[idx]
        previous = mj[current]
        while current != n - 1 and (
            (values[idx] - values[current]) * (current - previous)
            >= (values[current] - values[previous]) * (idx - current)
        ):
            mj[idx] = previous
            current = mj[idx]
            previous = mj[current]


def _collect_gcm(high: int, low: int, mn: np.ndarray, gcm: np.ndarray) -> int:
    count = 0
    gcm[count] = high
    count += 1
    gcm[count] = mn[high]
    while gcm[count] > low:
        count += 1
        gcm[count] = mn[gcm[count - 1]]
    return count


def _collect_lcm(low: int, high: int, mj: np.ndarray, lcm: np.ndarray) -> int:
    count = 0
    lcm[count] = low
    count += 1
    lcm[count] = mj[low]
    while lcm[count] < high:
        count += 1
        lcm[count] = mj[lcm[count - 1]]
    return count


def _largest_gcm_lcm_distance(
    values: np.ndarray,
    gcm: np.ndarray,
    lcm: np.ndarray,
    icx: int,
    icv: int,
) -> tuple[float, int, int]:
    n = values.size
    ix = icx - 1
    iv = 1
    ig = icx
    ih = icv
    distance = 1.0 / n if icx == 2 and icv == 2 else 0.0

    while gcm[ix] != lcm[iv]:
        if gcm[ix] <= lcm[iv]:
            lcm_previous = lcm[iv - 1]
            width = lcm[iv] - lcm_previous
            offset = gcm[ix] - lcm_previous - 1
            current = (values[gcm[ix]] - values[lcm_previous]) * width / (
                n * (values[lcm[iv]] - values[lcm_previous])
            ) - offset / n
            ix -= 1
            if current >= distance:
                distance = current
                ig = ix + 1
                ih = iv
        else:
            gcm_next = gcm[ix + 1]
            width = lcm[iv] - gcm_next + 1
            offset = gcm[ix] - gcm_next
            current = width / n - (
                (values[lcm[iv]] - values[gcm_next])
                * offset
                / (n * (values[gcm[ix]] - values[gcm_next]))
            )
            iv += 1
            if current >= distance:
                distance = current
                ig = ix + 1
                ih = iv - 1

        ix = max(ix, 0)
        iv = min(iv, icv)

    return float(distance), int(ig), int(ih)


def _convex_minorant_dip(
    values: np.ndarray,
    gcm: np.ndarray,
    ig: int,
    icx: int,
) -> float:
    return _envelope_dip(values, gcm, ig, icx, concave=False)


def _concave_majorant_dip(
    values: np.ndarray,
    lcm: np.ndarray,
    ih: int,
    icv: int,
) -> float:
    return _envelope_dip(values, lcm, ih, icv, concave=True)


def _envelope_dip(
    values: np.ndarray,
    envelope: np.ndarray,
    start_idx: int,
    stop_idx: int,
    *,
    concave: bool,
) -> float:
    largest = 0.0
    if start_idx == stop_idx:
        return largest
    n = values.size
    for i in range(start_idx, stop_idx):
        if concave:
            start = envelope[i]
            stop = envelope[i + 1]
        else:
            start = envelope[i + 1]
            stop = envelope[i]
        largest = max(largest, _segment_dip(values, start, stop, n, concave=concave))
    return float(largest)


def _segment_dip(
    values: np.ndarray,
    start: int,
    stop: int,
    n: int,
    *,
    concave: bool,
) -> float:
    if stop - start <= 1 or values[stop] == values[start]:
        return 1.0 / n
    scale = (stop - start) / (n * (values[stop] - values[start]))
    largest = 1.0 / n
    for i in range(start, stop + 1):
        projected = (values[i] - values[start]) * scale
        if concave:
            distance = projected - (i - start - 1) / n
        else:
            distance = (i - start + 1) / n - projected
        largest = max(largest, float(distance))
    return largest


__all__ = [
    "DipStatistic",
    "DipTestResult",
    "LedoitWolfCov",
    "dip_statistic",
    "dip_test",
    "empirical_cov",
    "rayleigh_test",
]
