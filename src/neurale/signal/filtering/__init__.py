#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""FIR and IIR filter design and execution APIs."""

from .coefficients import (
    FirCoefficients,
    IirCoefficients,
    SosCoefficients,
    StateSpaceCoefficients,
    ZpkCoefficients,
)
from .design import (
    fir_order,
    fir_transition_width,
    firls,
    firwin,
    phase_comp_fir,
)
from .fir import FirFilter, fir_filter
from .iir import (
    IirFilter,
    SosFilter,
    iir_filter,
    sos_filter,
    sos_filtfilt,
)
from .iir_design import (
    bessel,
    butter,
    design_iir,
    ellip,
    notch,
)
from .representations import (
    bilinear_ss,
    bilinear_tf,
    bilinear_zpk,
    lp2bp_ss,
    lp2bs_ss,
    lp2hp_ss,
    lp2lp_ss,
    ss2tf,
    ss2zpk,
    tf2ss,
    tf2zpk,
    zpk2sos,
    zpk2ss,
    zpk2tf,
)
from .spatial import common_reference

__all__ = [
    "FirCoefficients",
    "FirFilter",
    "IirCoefficients",
    "IirFilter",
    "SosCoefficients",
    "SosFilter",
    "StateSpaceCoefficients",
    "ZpkCoefficients",
    "bessel",
    "bilinear_ss",
    "bilinear_tf",
    "bilinear_zpk",
    "butter",
    "common_reference",
    "design_iir",
    "ellip",
    "fir_filter",
    "fir_order",
    "fir_transition_width",
    "firls",
    "firwin",
    "iir_filter",
    "lp2bp_ss",
    "lp2bs_ss",
    "lp2hp_ss",
    "lp2lp_ss",
    "notch",
    "phase_comp_fir",
    "sos_filter",
    "sos_filtfilt",
    "ss2tf",
    "ss2zpk",
    "tf2ss",
    "tf2zpk",
    "zpk2sos",
    "zpk2ss",
    "zpk2tf",
]
