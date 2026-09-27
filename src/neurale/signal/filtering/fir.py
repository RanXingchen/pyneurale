#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""FIR filtering backed by the native signal extension."""

from __future__ import annotations

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import validate_integer, validate_real_array
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._arrays import native_array

from ._filter import (
    _check_state,
    _filter,
    _FilterBase,
)
from .coefficients import FirCoefficients


def fir_filter(
    x: np.ndarray | SignalArray,
    coefs: FirCoefficients,
    *,
    axis: int = 0,
    zi: np.ndarray | None = None,
    return_state: bool = False,
    check_finite: bool = True,
) -> np.ndarray | SignalArray | tuple[np.ndarray | SignalArray, np.ndarray]:
    """Apply a causal FIR filter to a complete signal.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional real-valued signal.
    coefs : neurale.signal.filtering.FirCoefficients
        FIR coefficients.
    axis : int, default=0
        Sample axis for ndarray input.
    zi : numpy.ndarray or None, optional
        Initial delay-line state with shape ``(order, n_channels)``.
    return_state : bool, default=False
        Return the final delay-line state together with the filtered signal.
    check_finite : bool, default=True
        Validate that ``x`` contains only finite values.

    Returns
    -------
    filtered : numpy.ndarray or neurale.data.SignalArray
        Filtered signal with input layout and metadata preserved.
    final_state : numpy.ndarray, optional
        Final delay-line state. Returned only when ``return_state=True``.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native filtering extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs, coefficients, state, or sampling metadata are invalid.
    """
    if not isinstance(coefs, FirCoefficients):
        raise ValidationError("coefficients must be FirCoefficients.")

    taps = native_array(coefs.taps, float)
    fs = coefs.fs

    native = load_native_namespace("signal.filtering")

    result, state = _filter(
        x,
        coefs.order,
        axis=axis,
        fs=fs,
        label="FIR",
        coef_arrays=(coefs.taps,),
        zi=zi,
        check_finite=check_finite,
        operation=lambda data, state, _check: native.fir_filter(
            data,
            taps,
            native_array(state, float),
            "auto",
        )[:2],
    )

    if return_state:
        return result, state
    return result


class FirFilter(_FilterBase):
    """Realtime FIR processor with persistent state.

    ``process()`` filters the supplied frame in-place and returns the same
    object.

    Parameters
    ----------
    coefs : neurale.signal.filtering.FirCoefficients
        FIR coefficients.
    n_channels : int
        Number of channels in each processed frame.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native filtering extension is unavailable.
    neurale.exceptions.ValidationError
        If ``coefs`` or ``n_channels`` are invalid.
    """

    def __init__(
        self,
        coefs: FirCoefficients,
        *,
        n_channels: int,
    ) -> None:
        self._native = load_native_namespace("signal.filtering")

        self.n_channels = validate_integer(n_channels, "n_channels", minimum=1)

        if not isinstance(coefs, FirCoefficients):
            raise ValidationError("coefficients must be FirCoefficients.")
        self._set_coefficients(coefs)

    @property
    def coefficients(self) -> FirCoefficients:
        return self._coefs

    def _set_coefficients(self, coefs: FirCoefficients) -> None:
        validate_real_array(coefs.taps, name="FIR filtering")
        self._coefs = coefs
        taps = native_array(self.coefficients.taps, float)
        self._processor = self._native._FirSampleProcessor(
            taps,
            np.zeros((coefs.order, self.n_channels), dtype=float),
        )

        self._attach_processor(self._processor)

    def _state(self, state: np.ndarray | None, name: str) -> np.ndarray:
        expected = (self.coefficients.order, self.n_channels)
        return _check_state(state, expected, name)
