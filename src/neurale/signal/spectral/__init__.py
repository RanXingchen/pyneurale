#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Spectral-analysis primitives."""

from .frequency import freq_vector
from .multitaper import (
    MultitaperPsdProcessor,
    multitaper_psd,
    multitaper_spectrogram,
)
from .scaling import scale_spectrum
from .stft import stft_spectrogram
from .transforms import analytic_signal, czt
from .welch import welch_psd

__all__ = [
    "MultitaperPsdProcessor",
    "analytic_signal",
    "czt",
    "freq_vector",
    "multitaper_psd",
    "multitaper_spectrogram",
    "scale_spectrum",
    "stft_spectrogram",
    "welch_psd",
]
