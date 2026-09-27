#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Strongly typed filter coefficient objects."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from neurale._validation import validate_finite, validate_numeric_array
from neurale.exceptions import ValidationError
from neurale.signal._validation import validate_fs


@dataclass(frozen=True, slots=True, eq=False, init=False)
class FirCoefficients:
    """Store immutable FIR coefficients and optional sampling metadata.

    Parameters
    ----------
    taps : numpy.ndarray
        1D finite real or complex coefficients.
    fs : float or None, optional
        Positive sampling rate associated with the coefficients.

    Raises
    ------
    neurale.exceptions.ValidationError
        If coefficients are empty, non-finite, non-numeric, or not 1D,
        or if ``fs`` is invalid.
    """

    _taps: np.ndarray = field(repr=False)
    _dtype: np.dtype = field(repr=False)
    fs: float | None

    def __init__(
        self,
        taps: np.ndarray,
        fs: float | None = None,
    ) -> None:
        taps = validate_numeric_array(np.asarray(taps), "taps", ndim=1)
        if taps.size == 0:
            raise ValidationError("taps must contain at least one coefficient.")
        dtype = _coefficient_dtype(taps)
        copied = _frozen_vector(taps, dtype)
        if np.any(~np.isfinite(copied)):
            raise ValidationError("taps must contain finite values.")
        copied.setflags(write=False)
        object.__setattr__(self, "_taps", copied)
        object.__setattr__(self, "_dtype", copied.dtype)
        object.__setattr__(
            self,
            "fs",
            validate_fs(fs, optional=True),
        )

    @classmethod
    def _from_trusted_array(cls, taps, fs: float | None = None) -> FirCoefficients:
        dtype = _coefficient_dtype(taps)
        taps = _frozen_vector(taps, dtype)
        obj = object.__new__(cls)
        object.__setattr__(obj, "_taps", taps)
        object.__setattr__(obj, "_dtype", taps.dtype)
        object.__setattr__(obj, "fs", fs)
        return obj

    @property
    def taps(self) -> np.ndarray:
        return self._taps

    @property
    def order(self) -> int:
        return int(self.taps.size - 1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FirCoefficients):
            return NotImplemented
        return (
            self._dtype == other._dtype
            and np.array_equal(self.taps, other.taps)
            and self.fs == other.fs
        )

    def __hash__(self) -> int:
        return hash((self._dtype.str, self.taps.tobytes(), self.fs))

    def __repr__(self) -> str:
        return f"FirCoefficients(taps={self.taps!r}, fs={self.fs!r})"


@dataclass(frozen=True, slots=True, eq=False, init=False)
class IirCoefficients:
    """Immutable normalized direct-form IIR coefficients.

    Parameters
    ----------
    b : numpy.ndarray
        1D feedforward coefficients.
    a : numpy.ndarray
        1D feedback coefficients with non-zero leading value.
    fs : float or None, optional
        Positive sampling rate associated with the coefficients.

    Raises
    ------
    neurale.exceptions.ValidationError
        If coefficients are empty, non-finite, non-numeric, not
        1D, or if ``fs`` is invalid.

    Notes
    -----
    ``b`` and ``a`` are padded to the same length and normalized so that
    ``a[0] == 1``.
    """

    _b: np.ndarray = field(repr=False)
    _a: np.ndarray = field(repr=False)
    _dtype: np.dtype = field(repr=False)
    fs: float | None

    def __init__(
        self,
        b: np.ndarray,
        a: np.ndarray,
        fs: float | None = None,
    ) -> None:
        b = validate_numeric_array(np.asarray(b), "b", ndim=1)
        a = validate_numeric_array(np.asarray(a), "a", ndim=1)
        if b.size == 0 or a.size == 0:
            raise ValidationError("Coefficients must contain at least one coefficient.")
        validate_finite(b, "b")
        validate_finite(a, "a")

        if a[0] == 0:
            raise ValidationError("a[0] must be nonzero.")
        dtype = _coefficient_dtype(b, a)
        size = max(b.size, a.size)
        b_norm = np.zeros(size, dtype=dtype)
        a_norm = np.zeros(size, dtype=dtype)
        b_norm[: b.size] = b
        a_norm[: a.size] = a
        scale = a_norm[0]
        b_norm /= scale
        a_norm /= scale
        b_norm.setflags(write=False)
        a_norm.setflags(write=False)
        object.__setattr__(self, "_b", b_norm)
        object.__setattr__(self, "_a", a_norm)
        object.__setattr__(self, "_dtype", b_norm.dtype)
        object.__setattr__(
            self,
            "fs",
            validate_fs(fs, optional=True),
        )

    @classmethod
    def _from_trusted_arrays(cls, b, a, fs: float | None = None) -> IirCoefficients:
        dtype = _coefficient_dtype(b, a)
        b = _frozen_vector(b, dtype)
        a = _frozen_vector(a, dtype)
        obj = object.__new__(cls)
        object.__setattr__(obj, "_b", b)
        object.__setattr__(obj, "_a", a)
        object.__setattr__(obj, "_dtype", dtype)
        object.__setattr__(obj, "fs", fs)
        return obj

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def order(self) -> int:
        return int(self.b.size - 1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, IirCoefficients):
            return NotImplemented
        return (
            self._dtype == other._dtype
            and np.array_equal(self.b, other.b)
            and np.array_equal(self.a, other.a)
            and self.fs == other.fs
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._dtype.str,
                self.b.tobytes(),
                self.a.tobytes(),
                self.fs,
            )
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class SosCoefficients:
    """Immutable normalized second-order-section coefficients.

    Parameters
    ----------
    sos : numpy.ndarray
        2D array with shape ``(n_sections, 6)``. Each row is
        ``[b0, b1, b2, a0, a1, a2]``.
    fs : float or None, optional
        Positive sampling rate associated with the coefficients.

    Raises
    ------
    neurale.exceptions.ValidationError
        If ``sos`` has an invalid shape, contains non-finite values, has a
        zero denominator leading coefficient, or if ``fs`` is
        invalid.

    Notes
    -----
    Each section is normalized so that ``a0 == 1``.
    """

    _sos: np.ndarray = field(repr=False)
    _dtype: np.dtype = field(repr=False)
    _n_sections: int = field(repr=False)
    fs: float | None

    def __init__(
        self,
        sos: np.ndarray,
        fs: float | None = None,
    ) -> None:
        values = validate_numeric_array(np.asarray(sos), "sos", ndim=2)
        if values.shape[0] == 0 or values.shape[1] != 6:
            raise ValidationError("sos must have shape (n_sections, 6) with at least one section.")
        if np.any(~np.isfinite(values)):
            raise ValidationError("sos must contain finite values.")
        if np.any(values[:, 3] == 0):
            raise ValidationError("every SOS denominator a0 must be nonzero.")
        dtype = _coefficient_dtype(values)
        copied = np.array(values, dtype=dtype, order="C", copy=True)
        copied[:, :3] /= copied[:, 3, None]
        copied[:, 4:] /= copied[:, 3, None]
        copied[:, 3] = 1
        copied.setflags(write=False)
        object.__setattr__(self, "_sos", copied)
        object.__setattr__(self, "_dtype", copied.dtype)
        object.__setattr__(self, "_n_sections", copied.shape[0])
        object.__setattr__(
            self,
            "fs",
            validate_fs(fs, optional=True),
        )

    @classmethod
    def _from_trusted_array(cls, sos, fs: float | None = None) -> SosCoefficients:
        dtype = _coefficient_dtype(sos)
        values = _frozen_matrix(sos, dtype)
        obj = object.__new__(cls)
        object.__setattr__(obj, "_sos", values)
        object.__setattr__(obj, "_dtype", dtype)
        object.__setattr__(obj, "_n_sections", values.shape[0])
        object.__setattr__(obj, "fs", fs)
        return obj

    @property
    def sos(self) -> np.ndarray:
        return self._sos

    @property
    def n_sections(self) -> int:
        return self._n_sections

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SosCoefficients):
            return NotImplemented
        return (
            self._dtype == other._dtype
            and np.array_equal(self.sos, other.sos)
            and self.fs == other.fs
        )

    def __hash__(self) -> int:
        return hash(
            (
                self._dtype.str,
                self.sos.tobytes(),
                self.fs,
            )
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class ZpkCoefficients:
    """Immutable zero-pole-gain representation.

    Parameters
    ----------
    z : array_like
        Filter zeros.
    p : array_like
        Filter poles.
    k : float or complex
        System gain.
    fs : float or None, optional
        Positive sampling rate associated with the representation.

    Raises
    ------
    neurale.exceptions.ValidationError
        If zeros, poles, or gain are non-finite, if the representation is not
        proper, or if ``fs`` is invalid.
    """

    _z: np.ndarray = field(repr=False)
    _p: np.ndarray = field(repr=False)
    k: complex | float
    fs: float | None

    def __init__(self, z, p, k, fs: float | None = None) -> None:
        z = np.asarray(z, dtype=np.complex128).reshape(-1)
        p = np.asarray(p, dtype=np.complex128).reshape(-1)
        k = np.asarray(k).item()
        if np.any(~np.isfinite(z)) or np.any(~np.isfinite(p)) or not np.isfinite(k):
            raise ValidationError("z, p, and k must be finite.")
        if z.size > p.size:
            raise ValidationError("a proper system cannot have more zeros than poles.")
        z = _frozen_vector(z, np.complex128)
        p = _frozen_vector(p, np.complex128)
        object.__setattr__(self, "_z", z)
        object.__setattr__(self, "_p", p)
        object.__setattr__(self, "k", np.real_if_close(k).item())
        object.__setattr__(
            self,
            "fs",
            validate_fs(fs, optional=True),
        )

    @classmethod
    def _from_trusted_arrays(cls, z, p, k, fs: float | None = None) -> ZpkCoefficients:
        z = _frozen_vector(z, np.complex128)
        p = _frozen_vector(p, np.complex128)
        obj = object.__new__(cls)
        object.__setattr__(obj, "_z", z)
        object.__setattr__(obj, "_p", p)
        object.__setattr__(obj, "k", np.real_if_close(k).item())
        object.__setattr__(obj, "fs", fs)
        return obj

    @property
    def z(self) -> np.ndarray:
        return self._z

    @property
    def p(self) -> np.ndarray:
        return self._p

    @property
    def order(self) -> int:
        return int(self.p.size)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ZpkCoefficients):
            return NotImplemented
        return (
            np.array_equal(self.z, other.z)
            and np.array_equal(self.p, other.p)
            and self.k == other.k
            and self.fs == other.fs
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.z.tobytes(),
                self.p.tobytes(),
                self.k,
                self.fs,
            )
        )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class StateSpaceCoefficients:
    """Immutable SISO state-space representation.

    Parameters
    ----------
    a : array_like
        Square state-transition matrix.
    b : array_like
        Input vector with one entry per state.
    c : array_like
        Output vector with one entry per state.
    d : float
        Feedthrough scalar.
    fs : float or None, optional
        Positive sampling rate associated with the representation.

    Raises
    ------
    neurale.exceptions.ValidationError
        If shapes are not a valid SISO state-space system, values are
        non-finite, or ``fs`` is invalid.
    """

    _a: np.ndarray = field(repr=False)
    _b: np.ndarray = field(repr=False)
    _c: np.ndarray = field(repr=False)
    _order: int = field(repr=False)
    d: float
    fs: float | None

    def __init__(self, a, b, c, d, fs: float | None = None) -> None:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float).reshape(-1)
        c = np.asarray(c, dtype=float).reshape(-1)
        d = float(d)
        if (
            a.ndim != 2
            or a.shape[0] != a.shape[1]
            or b.shape != (a.shape[0],)
            or c.shape != (a.shape[0],)
        ):
            raise ValidationError("invalid SISO state-space shapes.")
        if (
            np.any(~np.isfinite(a))
            or np.any(~np.isfinite(b))
            or np.any(~np.isfinite(c))
            or not np.isfinite(d)
        ):
            raise ValidationError("state-space values must be finite.")
        a = _frozen_matrix(a, float)
        b = _frozen_vector(b, float)
        c = _frozen_vector(c, float)
        object.__setattr__(self, "_a", a)
        object.__setattr__(self, "_b", b)
        object.__setattr__(self, "_c", c)
        object.__setattr__(self, "_order", a.shape[0])
        object.__setattr__(self, "d", d)
        object.__setattr__(
            self,
            "fs",
            validate_fs(fs, optional=True),
        )

    @classmethod
    def _from_trusted_arrays(cls, a, b, c, d, fs: float | None = None) -> StateSpaceCoefficients:
        a = _frozen_matrix(a, float)
        b = _frozen_vector(b, float)
        c = _frozen_vector(c, float)
        obj = object.__new__(cls)
        object.__setattr__(obj, "_a", a)
        object.__setattr__(obj, "_b", b)
        object.__setattr__(obj, "_c", c)
        object.__setattr__(obj, "_order", a.shape[0])
        object.__setattr__(obj, "d", d)
        object.__setattr__(obj, "fs", fs)
        return obj

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def b(self) -> np.ndarray:
        return self._b

    @property
    def c(self) -> np.ndarray:
        return self._c

    @property
    def order(self) -> int:
        return self._order

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StateSpaceCoefficients):
            return NotImplemented
        return (
            np.array_equal(self.a, other.a)
            and np.array_equal(self.b, other.b)
            and np.array_equal(self.c, other.c)
            and self.d == other.d
            and self.fs == other.fs
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.a.tobytes(),
                self.b.tobytes(),
                self.c.tobytes(),
                self.d,
                self.fs,
            )
        )


def _coefficient_dtype(*arrays) -> np.dtype:
    values = [np.asarray(arr) for arr in arrays]
    complex_dtype = any(np.iscomplexobj(arr) for arr in values)
    floor = np.complex128 if complex_dtype else np.float64
    return np.result_type(*(arr.dtype for arr in values), floor)


def _frozen_vector(value, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True).reshape(-1)
    result.setflags(write=False)
    return result


def _frozen_matrix(value, dtype) -> np.ndarray:
    result = np.array(value, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result
