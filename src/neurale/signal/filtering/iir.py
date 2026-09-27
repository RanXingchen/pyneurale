#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""IIR filtering backed by the native signal extension."""

from __future__ import annotations

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import (
    validate_integer,
    validate_real_array,
)
from neurale.data import SignalArray
from neurale.exceptions import ValidationError
from neurale.signal._arrays import native_array
from neurale.signal._input import normalize_signal_input, restore_signal_output

from ._filter import (
    _check_state,
    _filter,
    _FilterBase,
    validate_coefficient_rate,
)
from .coefficients import IirCoefficients, SosCoefficients

FilterResult = np.ndarray | SignalArray | tuple[np.ndarray | SignalArray, np.ndarray]


def iir_filter(
    x: np.ndarray | SignalArray,
    coefs: IirCoefficients,
    *,
    axis: int = 0,
    zi: np.ndarray | None = None,
    return_state: bool = False,
    check_finite: bool = True,
) -> FilterResult:
    """Apply a causal direct-form IIR filter to a complete signal.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional real-valued signal.
    coefs : neurale.signal.filtering.IirCoefficients
        Direct-form IIR coefficients.
    axis : int, default=0
        Sample axis for ndarray input.
    zi : numpy.ndarray or None, optional
        Initial filter state with shape ``(order, n_channels)``.
    return_state : bool, default=False
        Return the final filter state together with the filtered signal.
    check_finite : bool, default=True
        Validate finite input in the native filtering path.

    Returns
    -------
    filtered : numpy.ndarray or neurale.data.SignalArray
        Filtered signal with input layout and metadata preserved.
    final_state : numpy.ndarray, optional
        Final direct-form state. Returned only when ``return_state=True``.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native filtering extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs, coefficients, state, or sampling metadata are invalid.
    """
    if not isinstance(coefs, IirCoefficients):
        raise ValidationError("coefficients must be IirCoefficients.")

    b = native_array(coefs.b, float)
    a = native_array(coefs.a, float)
    fs = coefs.fs

    native = load_native_namespace("signal.filtering")

    result, state = _filter(
        x,
        coefs.order,
        axis=axis,
        fs=fs,
        label="IIR",
        coef_arrays=(coefs.b, coefs.a),
        zi=zi,
        check_finite=check_finite,
        operation=lambda data, state, check: native.iir_filter(
            data,
            b,
            a,
            state,
            "auto",
            check,
        )[:2],
    )

    if return_state:
        return result, state
    return result


def sos_filter(
    x: np.ndarray | SignalArray,
    coefs: SosCoefficients,
    *,
    axis: int = 0,
    zi: np.ndarray | None = None,
    return_state: bool = False,
    check_finite: bool = True,
) -> FilterResult:
    """Apply a causal second-order-section filter to a complete signal.

    Parameters
    ----------
    x : numpy.ndarray or neurale.data.SignalArray
        One- or two-dimensional real-valued signal.
    coefs : neurale.signal.filtering.SosCoefficients
        Second-order-section coefficients.
    axis : int, default=0
        Sample axis for ndarray input.
    zi : numpy.ndarray or None, optional
        Initial SOS state with shape ``(n_sections, 2, n_channels)``.
    return_state : bool, default=False
        Return the final SOS state together with the filtered signal.
    check_finite : bool, default=True
        Validate finite input in the native filtering path.

    Returns
    -------
    filtered : numpy.ndarray or neurale.data.SignalArray
        Filtered signal with input layout and metadata preserved.
    final_state : numpy.ndarray, optional
        Final SOS state. Returned only when ``return_state=True``.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native filtering extension is unavailable.
    neurale.exceptions.ValidationError
        If inputs, coefficients, state, or sampling metadata are invalid.
    """
    if not isinstance(coefs, SosCoefficients):
        raise ValidationError("coefficients must be SosCoefficients.")
    sos = native_array(coefs.sos, float)
    fs = coefs.fs

    native = load_native_namespace("signal.filtering")

    result, state = _filter(
        x,
        coefs.n_sections,
        axis=axis,
        fs=fs,
        label="SOS",
        coef_arrays=(coefs.sos,),
        zi=zi,
        check_finite=check_finite,
        operation=lambda data, state, check: native.sos_filter(
            data,
            sos,
            state,
            "auto",
            check,
        )[:2],
    )

    if return_state:
        return result, state
    return result


def sos_filtfilt(
    x: np.ndarray | SignalArray,
    coefs: SosCoefficients,
    *,
    axis: int = 0,
    check_finite: bool = True,
) -> np.ndarray | SignalArray:
    """Apply a forward-backward SOS filter to a complete signal.

    This offline operation preserves ndarray layout and ``SignalArray``
    metadata. Use :class:`SosFilter` for causal streaming processing.
    """
    if not isinstance(coefs, SosCoefficients):
        raise ValidationError("coefficients must be SosCoefficients.")
    if not isinstance(check_finite, bool):
        raise ValidationError("check_finite must be a bool.")

    data, context = normalize_signal_input(x, axis=axis)
    validate_coefficient_rate(x, coefs.fs, "SOS")
    validate_real_array(data, name="Signal", finite=False)

    native = load_native_namespace("signal.filtering")
    try:
        output = native.sos_filtfilt(
            native_array(data, float),
            native_array(coefs.sos, float),
            check_finite,
        )
    except ValueError as exc:
        raise ValidationError(str(exc)) from exc
    return restore_signal_output(output, context)


class IirFilter(_FilterBase):
    """Realtime direct-form IIR processor with persistent state.

    ``process()`` filters the supplied frame in-place and returns the same
    object.

    Parameters
    ----------
    coefs : neurale.signal.filtering.IirCoefficients
        Direct-form IIR coefficients.
    n_channels : int
        Number of channels in each processed frame.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native filtering extension is unavailable.
    neurale.exceptions.ValidationError
        If coefficients or ``n_channels`` are invalid.
    """

    def __init__(
        self,
        coefs: IirCoefficients,
        *,
        n_channels: int,
    ) -> None:
        self._native = load_native_namespace("signal.filtering")

        self.n_channels = validate_integer(n_channels, "n_channels", minimum=1)

        if not isinstance(coefs, IirCoefficients):
            raise ValidationError("coefficients must be IirCoefficients.")
        self._set_coefficients(coefs)

    @property
    def coefficients(self) -> IirCoefficients:
        return self._coefs

    def set_coefficients(self, coefs: IirCoefficients) -> None:
        """Replace the filter coefficients and reset the processor state.

        Parameters
        ----------
        coefs : neurale.signal.filtering.IirCoefficients
            Replacement direct-form coefficients.

        Raises
        ------
        neurale.exceptions.ValidationError
            If ``coefficients`` has an invalid type or contains complex values.
        """
        if not isinstance(coefs, IirCoefficients):
            raise ValidationError("coefficients must be IirCoefficients.")

        self._set_coefficients(coefs)

    def _set_coefficients(self, coefs: IirCoefficients) -> None:
        validate_real_array(coefs.b, "IIR coefficients b")
        validate_real_array(coefs.a, "IIR coefficients a")
        self._coefs = coefs
        b = native_array(coefs.b, float)
        a = native_array(coefs.a, float)
        self._processor = self._native._IirSampleProcessor(
            b,
            a,
            np.zeros((coefs.order, self.n_channels), dtype=float),
        )

        self._attach_processor(self._processor)

    def _state(self, state: np.ndarray | None, name: str) -> np.ndarray:
        expected = (self.coefficients.order, self.n_channels)
        return _check_state(state, expected, name)


class SosFilter(_FilterBase):
    """Realtime SOS processor with persistent state.

    ``process()`` filters the supplied frame in-place and returns the same
    object.

    Parameters
    ----------
    coefs : neurale.signal.filtering.SosCoefficients
        Second-order-section coefficients.
    n_channels : int
        Number of channels in each processed frame.

    Raises
    ------
    neurale.exceptions.NativeUnavailableError
        If the native filtering extension is unavailable.
    neurale.exceptions.ValidationError
        If coefficients or ``n_channels`` are invalid.
    """

    def __init__(
        self,
        coefs: SosCoefficients,
        *,
        n_channels: int,
    ) -> None:
        self._native = load_native_namespace("signal.filtering")

        self.n_channels = validate_integer(n_channels, "n_channels", minimum=1)

        if not isinstance(coefs, SosCoefficients):
            raise ValidationError("coefficients must be SosCoefficients.")
        self._set_sos(coefs)

    @property
    def coefficients(self) -> SosCoefficients:
        return self._coefs

    def set_sos(self, coefs: SosCoefficients) -> None:
        """Replace the SOS coefficients and reset the processor state.

        Parameters
        ----------
        coefs : neurale.signal.filtering.SosCoefficients
            Replacement SOS coefficients.

        Raises
        ------
        neurale.exceptions.ValidationError
            If ``coefs`` has an invalid type or contains complex values.
        """
        if not isinstance(coefs, SosCoefficients):
            raise ValidationError("coefficients must be SosCoefficients.")
        self._set_sos(coefs)

    def _set_sos(self, coefs: SosCoefficients) -> None:
        validate_real_array(coefs.sos, name="SOS filtering")
        self._coefs = coefs
        sos = native_array(coefs.sos, float)
        self._processor = self._native._SosSampleProcessor(
            sos,
            np.zeros((coefs.n_sections, 2, self.n_channels), dtype=float),
        )

        self._attach_processor(self._processor)

    def _state(self, state: np.ndarray | None, name: str) -> np.ndarray:
        expected = (self.coefficients.n_sections, 2, self.n_channels)
        return _check_state(state, expected, name)
