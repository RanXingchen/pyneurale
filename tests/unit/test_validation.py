#!/usr/bin/env python3

import numpy as np
import pytest

from neurale._validation import (
    validate_axis,
    validate_choice,
    validate_integer,
    validate_number,
    validate_numeric_array,
    validate_positive_float,
)
from neurale.exceptions import ConfigurationError, ValidationError


def test_validate_choice_returns_valid_value() -> None:
    assert validate_choice("hilbert", ("pmtm", "hilbert"), "method") == "hilbert"


def test_validate_choice_rejects_invalid_value() -> None:
    with pytest.raises(ValidationError, match="method must be one of"):
        validate_choice("fft", ("pmtm", "hilbert"), "method")


def test_validate_choice_rejects_empty_choices() -> None:
    with pytest.raises(ValidationError, match="at least one valid choice"):
        validate_choice("anything", (), "method")


def test_validate_choice_supports_custom_error_type() -> None:
    with pytest.raises(ConfigurationError, match="device must be one of"):
        validate_choice(
            "gpu",
            ("cpu", "cuda"),
            "device",
            error_type=ConfigurationError,
        )


@pytest.mark.parametrize(
    ("value", "minimum"),
    [
        (0, None),
        (-10, None),
        (0, 0),
        (1, 1),
        (5, 3),
    ],
)
def test_validate_integer_returns_valid_value(value: int, minimum: int | None) -> None:
    assert validate_integer(value, minimum=minimum) == value


def test_validate_integer_rejects_bool_type_and_range() -> None:
    with pytest.raises(ValidationError, match="positive integer"):
        validate_integer(True, "count", minimum=1)

    with pytest.raises(ValidationError, match="positive"):
        validate_integer(0, "count", minimum=1)


def test_validate_integer_supports_custom_error_type() -> None:
    with pytest.raises(ConfigurationError, match="non-negative"):
        validate_integer(
            -1,
            "device",
            minimum=0,
            error_type=ConfigurationError,
        )


def test_validate_axis_normalizes_numpy_style_indices() -> None:
    assert validate_axis(0, 2) == 0
    assert validate_axis(-1, 2) == 1

    with pytest.raises(ValidationError, match="out of bounds"):
        validate_axis(2, 2)


def test_validate_numeric_array_checks_type_dimensions_and_dtype() -> None:
    values = np.ones((2, 3))
    assert validate_numeric_array(values, ndim=2) is values
    one_dim = np.ones(3)
    assert validate_numeric_array(one_dim) is one_dim
    assert validate_numeric_array(np.ones((2, 3), dtype=bool), ndim=2).dtype == bool

    with pytest.raises(ValidationError, match=r"numpy\.ndarray"):
        validate_numeric_array([1, 2, 3])  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="2D array"):
        validate_numeric_array(np.ones(3), ndim=2)
    with pytest.raises(ValidationError, match="numeric data"):
        validate_numeric_array(np.array(["x"], dtype=object))


def test_validate_positive_float_converts_and_rejects_invalid_values() -> None:
    assert validate_positive_float(1000) == 1000.0

    for value in (0, -1, np.inf, np.nan, True, "invalid"):
        with pytest.raises(ValidationError, match="finite positive number"):
            validate_positive_float(value)  # type: ignore[arg-type]


def test_validate_number_supports_kinds_and_bounds() -> None:
    assert validate_number(3, kind="integer", minimum=0, maximum=3) == 3
    assert (
        validate_number(
            "1.5",
            kind="real",
            minimum=0,
            minimum_inclusive=False,
            maximum=2,
            maximum_inclusive=False,
            coerce=True,
        )
        == 1.5
    )

    with pytest.raises(ValidationError, match="greater than 0"):
        validate_number(
            0.0,
            kind="real",
            minimum=0,
            minimum_inclusive=False,
        )
    with pytest.raises(ValidationError, match="less than 2"):
        validate_number(
            2.0,
            kind="real",
            maximum=2,
            maximum_inclusive=False,
        )


def test_validate_number_rejects_bool_nonfinite_and_integer_truncation() -> None:
    with pytest.raises(ValidationError, match="integer"):
        validate_number(True, kind="integer")
    with pytest.raises(ValidationError, match="integer"):
        validate_number(2.5, kind="integer", coerce=True)
    with pytest.raises(ValidationError, match="finite number"):
        validate_number(np.inf, kind="real")
    assert validate_number(np.inf, kind="real", finite=False) == np.inf


def test_validate_number_checks_rule_configuration() -> None:
    with pytest.raises(ValueError, match="unsupported number kind"):
        validate_number(1, kind="complex")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="minimum"):
        validate_number(1, kind="integer", minimum=2, maximum=1)
    with pytest.raises(ValueError, match="equal bounds"):
        validate_number(
            1,
            kind="integer",
            minimum=1,
            maximum=1,
            maximum_inclusive=False,
        )
