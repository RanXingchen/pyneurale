#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Internal validation helpers shared by PyNeurale modules."""

from __future__ import annotations

import math
from collections.abc import Iterable
from numbers import Integral, Real
from typing import Literal, TypeVar

import numpy as np

from .exceptions import ValidationError

T = TypeVar("T")
E = TypeVar("E", bound=Exception)
NumberKind = Literal["integer", "real"]


def _validate_ndim(array_ndim: int, ndim: int | None, name: str) -> None:
    if ndim is None:
        return
    if isinstance(ndim, bool) or not isinstance(ndim, Integral) or ndim < 0:
        raise ValidationError("ndim must be a non-negative integer or None.")
    n_dims = int(ndim)
    if array_ndim != n_dims:
        raise ValidationError(f"{name} must be a {n_dims}D array; got {array_ndim} dimensions.")


def validate_axis(axis: int, ndim: int, name: str = "axis") -> int:
    if isinstance(axis, bool) or not isinstance(axis, Integral):
        raise ValidationError(f"{name} must be an integer.")
    if isinstance(ndim, bool) or not isinstance(ndim, Integral) or ndim <= 0:
        raise ValidationError("ndim must be a positive integer.")

    normalized = int(axis)
    n_dims = int(ndim)
    if normalized < 0:
        normalized += n_dims
    if normalized < 0 or normalized >= n_dims:
        raise ValidationError(
            f"{name} {axis} is out of bounds for an array with {n_dims} dimensions."
        )
    return normalized


def validate_numeric_array(
    arr: np.ndarray,
    name: str = "value",
    *,
    ndim: int | None = None,
) -> np.ndarray:
    # Validate a numeric NumPy array with an optional exact dimension count.

    if not isinstance(arr, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray.")
    _validate_ndim(arr.ndim, ndim, name)
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise ValidationError(f"{name} must contain numeric data.")
    return arr


def validate_number(
    value: object,
    name: str = "value",
    *,
    kind: NumberKind,
    minimum: int | float | None = None,
    minimum_inclusive: bool = True,
    maximum: int | float | None = None,
    maximum_inclusive: bool = True,
    finite: bool = True,
    coerce: bool = False,
    error_type: type[E] = ValidationError,
) -> int | float:
    _validate_number_rules(
        kind=kind,
        minimum=minimum,
        minimum_inclusive=minimum_inclusive,
        maximum=maximum,
        maximum_inclusive=maximum_inclusive,
        finite=finite,
        coerce=coerce,
    )
    description = _number_description(
        kind,
        minimum=minimum,
        minimum_inclusive=minimum_inclusive,
        finite=finite,
    )
    converted = _convert_number(
        value,
        name=name,
        kind=kind,
        coerce=coerce,
        description=description,
        error_type=error_type,
    )
    _validate_number_value(
        converted,
        name=name,
        minimum=minimum,
        minimum_inclusive=minimum_inclusive,
        maximum=maximum,
        maximum_inclusive=maximum_inclusive,
        finite=finite,
        description=description,
        error_type=error_type,
    )
    return converted


def _validate_number_rules(
    *,
    kind: NumberKind,
    minimum: int | float | None,
    minimum_inclusive: bool,
    maximum: int | float | None,
    maximum_inclusive: bool,
    finite: bool,
    coerce: bool,
) -> None:
    if kind not in ("integer", "real"):
        raise ValueError(f"unsupported number kind: {kind!r}.")
    if not isinstance(minimum_inclusive, bool):
        raise TypeError("minimum_inclusive must be a bool.")
    if not isinstance(maximum_inclusive, bool):
        raise TypeError("maximum_inclusive must be a bool.")
    if not isinstance(finite, bool):
        raise TypeError("finite must be a bool.")
    if not isinstance(coerce, bool):
        raise TypeError("coerce must be a bool.")
    if minimum is not None and maximum is not None:
        if minimum > maximum:
            raise ValueError("minimum must not be greater than maximum.")
        if minimum == maximum and not (minimum_inclusive and maximum_inclusive):
            raise ValueError("equal bounds must both be inclusive.")


def _convert_number(
    value: object,
    *,
    name: str,
    kind: NumberKind,
    coerce: bool,
    description: str,
    error_type: type[E],
) -> int | float:
    if isinstance(value, bool):
        raise error_type(f"{name} must be {description}.")
    if kind == "integer":
        if not isinstance(value, Integral):
            raise error_type(f"{name} must be {description}.")
        converted = int(value) if coerce else value
    elif coerce:
        try:
            converted = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise error_type(f"{name} must be {description}.") from exc
    else:
        if not isinstance(value, Real):
            raise error_type(f"{name} must be {description}.")
        converted = value
    return converted


def _validate_number_value(
    converted: int | float,
    *,
    name: str,
    minimum: int | float | None,
    minimum_inclusive: bool,
    maximum: int | float | None,
    maximum_inclusive: bool,
    finite: bool,
    description: str,
    error_type: type[E],
) -> None:
    if finite and not math.isfinite(converted):
        raise error_type(f"{name} must be {description}.")
    if minimum is not None:
        below_minimum = converted < minimum if minimum_inclusive else converted <= minimum
        if below_minimum:
            relation = "at least" if minimum_inclusive else "greater than"
            raise error_type(f"{name} must be {relation} {minimum}; got {converted}.")
    if maximum is not None:
        above_maximum = converted > maximum if maximum_inclusive else converted >= maximum
        if above_maximum:
            relation = "at most" if maximum_inclusive else "less than"
            raise error_type(f"{name} must be {relation} {maximum}; got {converted}.")


def _number_description(
    kind: NumberKind,
    *,
    minimum: int | float | None,
    minimum_inclusive: bool,
    finite: bool,
) -> str:
    if kind == "integer":
        if minimum == 0 and minimum_inclusive:
            return "a non-negative integer"
        if minimum == 1 and minimum_inclusive:
            return "a positive integer"
        return "an integer"

    qualifiers: list[str] = []
    if finite:
        qualifiers.append("finite")
    if minimum == 0 and not minimum_inclusive:
        qualifiers.append("positive")
    qualifiers.append("number")
    return "a " + " ".join(qualifiers)


def validate_positive_float(
    value: float,
    name: str = "value",
    *,
    error_type: type[E] = ValidationError,
) -> float:
    try:
        return float(
            validate_number(
                value,
                name,
                kind="real",
                minimum=0,
                minimum_inclusive=False,
                finite=True,
                coerce=True,
                error_type=error_type,
            )
        )
    except error_type as exc:
        raise error_type(f"{name} must be a finite positive number.") from exc


def validate_choice(
    value: T,
    choices: Iterable[T],
    name: str = "value",
    *,
    error_type: type[E] = ValidationError,
) -> T:
    choice_tuple = tuple(choices)
    if not choice_tuple:
        raise error_type(f"{name} must define at least one valid choice.")

    if value in choice_tuple:
        return value

    formatted_choices = ", ".join(repr(choice) for choice in choice_tuple)
    raise error_type(f"{name} must be one of: {formatted_choices}; got {value!r}.")


def validate_integer(
    value: int,
    name: str = "value",
    *,
    minimum: int | None = None,
    error_type: type[E] = ValidationError,
) -> int:
    try:
        return int(
            validate_number(
                value,
                name,
                kind="integer",
                minimum=minimum,
                error_type=error_type,
            )
        )
    except error_type as exc:
        if isinstance(value, Integral) and not isinstance(value, bool):
            if minimum == 0:
                raise error_type(f"{name} must be non-negative.") from exc
            if minimum == 1:
                raise error_type(f"{name} must be positive.") from exc
        raise


def validate_finite(data: np.ndarray, name: str) -> None:
    if np.any(~np.isfinite(data)):
        raise ValidationError(f"{name} must contain finite values.")


def validate_real_array(
    arr: np.ndarray,
    name: str = "array",
    *,
    ndim: int | None = None,
    finite: bool = True,
) -> np.ndarray:
    validate_numeric_array(arr, name, ndim=ndim)

    if np.issubdtype(arr.dtype, np.complexfloating):
        raise ValidationError(f"{name} must contain real-valued data.")

    arr = np.require(arr, dtype=np.float64, requirements=("C", "A"))
    if finite:
        validate_finite(arr, name)

    return arr


def validate_labels(
    data: np.ndarray,
    name: str = "data",
    *,
    ndim: int | None = None,
    n_samples: int | None = None,
):
    if not isinstance(data, np.ndarray):
        raise ValidationError(f"{name} must be a numpy.ndarray.")
    if np.issubdtype(data.dtype, np.complexfloating):
        raise ValidationError(f"{name} must contain real-valued labels.")
    if np.issubdtype(data.dtype, np.inexact) and np.any(~np.isfinite(data)):
        raise ValidationError(f"{name} must contain only finite labels.")

    _validate_ndim(data.ndim, ndim, name)

    if n_samples is not None and data.shape[0] != n_samples:
        raise ValidationError(
            f"{name} expected have number of samples {n_samples}; got {data.shape[0]} samples."
        )

    return data
