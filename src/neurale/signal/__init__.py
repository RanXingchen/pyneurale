#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Continuous-signal processing APIs.

The signal package uses ``(n_samples, n_channels)`` as its internal array
layout. Public algorithms may accept either a one- or two-dimensional
``numpy.ndarray`` or a :class:`neurale.data.SignalArray`:

* 1D arrays are treated as a single channel;
* ndarray results preserve the input dimensionality and sample-axis position;
* shape-preserving ``SignalArray`` operations return a new ``SignalArray`` and
  retain timing, channel, unit, clock, name, and attribute metadata;
* inputs are not modified unless the API explicitly documents in-place
  processing.

Signal algorithms with native implementations use the native signal extension
as their execution boundary. The native layer owns lower-level builtin, MKL,
BLAS, and FFT kernel selection; Python code handles validation, metadata, and
array layout.

Importing :mod:`neurale.signal` loads the NumPy and SciPy signal-processing
stack and exposes the stable public APIs implemented by this package. The
top-level :mod:`neurale` package remains the lightweight import boundary.
"""

import scipy as scipy

from .filtering import (
    FirCoefficients,
    FirFilter,
    IirCoefficients,
    IirFilter,
    SosCoefficients,
    SosFilter,
    StateSpaceCoefficients,
    ZpkCoefficients,
    bessel,
    bilinear_ss,
    bilinear_tf,
    bilinear_zpk,
    butter,
    common_reference,
    design_iir,
    ellip,
    fir_filter,
    fir_order,
    fir_transition_width,
    firls,
    firwin,
    iir_filter,
    lp2bp_ss,
    lp2bs_ss,
    lp2hp_ss,
    lp2lp_ss,
    notch,
    phase_comp_fir,
    sos_filter,
    sos_filtfilt,
    ss2tf,
    ss2zpk,
    tf2ss,
    tf2zpk,
    zpk2sos,
    zpk2ss,
    zpk2tf,
)
from .resampling import Resampler, resample
from .smoothing import WhittakerSmoother, gaussian_smooth, moving_average
from .spectral import (
    MultitaperPsdProcessor,
    analytic_signal,
    czt,
    freq_vector,
    multitaper_psd,
    multitaper_spectrogram,
    scale_spectrum,
    stft_spectrogram,
    welch_psd,
)
from .transforms import fft_integrate
from .windows import cosine_window, dpss, kaiser_window

__all__ = [
    "FirCoefficients",
    "FirFilter",
    "IirCoefficients",
    "IirFilter",
    "MultitaperPsdProcessor",
    "Resampler",
    "SosCoefficients",
    "SosFilter",
    "StateSpaceCoefficients",
    "WhittakerSmoother",
    "ZpkCoefficients",
    "analytic_signal",
    "bessel",
    "bilinear_ss",
    "bilinear_tf",
    "bilinear_zpk",
    "butter",
    "common_reference",
    "cosine_window",
    "czt",
    "design_iir",
    "dpss",
    "ellip",
    "fft_integrate",
    "fir_filter",
    "fir_order",
    "fir_transition_width",
    "firls",
    "firwin",
    "freq_vector",
    "gaussian_smooth",
    "iir_filter",
    "kaiser_window",
    "lp2bp_ss",
    "lp2bs_ss",
    "lp2hp_ss",
    "lp2lp_ss",
    "moving_average",
    "multitaper_psd",
    "multitaper_spectrogram",
    "notch",
    "phase_comp_fir",
    "resample",
    "scale_spectrum",
    "sos_filter",
    "sos_filtfilt",
    "ss2tf",
    "ss2zpk",
    "stft_spectrogram",
    "tf2ss",
    "tf2zpk",
    "welch_psd",
    "zpk2sos",
    "zpk2ss",
    "zpk2tf",
]
