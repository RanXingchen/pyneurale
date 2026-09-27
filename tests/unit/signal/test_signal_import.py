#!/usr/bin/env python3

from __future__ import annotations

import importlib
import sys


def test_signal_import_loads_public_api_and_scipy() -> None:
    for module_name in list(sys.modules):
        if module_name == "neurale.signal" or module_name.startswith("neurale.signal."):
            del sys.modules[module_name]

    loaded_before = set(sys.modules)
    signal = importlib.import_module("neurale.signal")
    newly_loaded = set(sys.modules) - loaded_before

    assert_all = [
        "WhittakerSmoother",
        "analytic_signal",
        "bilinear_ss",
        "bilinear_tf",
        "bilinear_zpk",
        "common_reference",
        "czt",
        "dpss",
        "FirCoefficients",
        "FirFilter",
        "IirCoefficients",
        "IirFilter",
        "SosCoefficients",
        "SosFilter",
        "StateSpaceCoefficients",
        "ZpkCoefficients",
        "bessel",
        "butter",
        "ellip",
        "firwin",
        "design_iir",
        "notch",
        "phase_comp_fir",
        "fir_order",
        "fir_filter",
        "fir_transition_width",
        "firls",
        "iir_filter",
        "fft_integrate",
        "freq_vector",
        "gaussian_smooth",
        "cosine_window",
        "kaiser_window",
        "moving_average",
        "MultitaperPsdProcessor",
        "multitaper_psd",
        "multitaper_spectrogram",
        "Resampler",
        "resample",
        "scale_spectrum",
        "stft_spectrogram",
        "lp2bp_ss",
        "lp2bs_ss",
        "lp2hp_ss",
        "lp2lp_ss",
        "ss2tf",
        "ss2zpk",
        "sos_filter",
        "sos_filtfilt",
        "tf2ss",
        "tf2zpk",
        "welch_psd",
        "zpk2sos",
        "zpk2ss",
        "zpk2tf",
    ]
    for m in signal.__all__:
        assert m in assert_all

    assert "scipy" in sys.modules
    assert callable(signal.freq_vector)
    assert callable(signal.firls)
    assert callable(signal.firwin)
    assert callable(signal.common_reference)
    assert callable(signal.design_iir)
    assert callable(signal.notch)
    assert callable(signal.fir_filter)
    assert callable(signal.iir_filter)
    assert callable(signal.sos_filter)
    assert callable(signal.scale_spectrum)
    assert callable(signal.moving_average)
    assert callable(signal.gaussian_smooth)
    assert callable(signal.cosine_window)
    assert callable(signal.kaiser_window)
    assert callable(signal.fft_integrate)
    assert not hasattr(signal, "filter_signal")
    assert not hasattr(signal, "create_filter_processor")
    assert callable(signal.analytic_signal)
    assert callable(signal.czt)
    assert callable(signal.dpss)
    assert callable(signal.multitaper_psd)
    assert callable(signal.multitaper_spectrogram)
    assert callable(signal.welch_psd)
    assert callable(signal.stft_spectrogram)
    assert callable(signal.zpk2tf)
    assert callable(signal.tf2ss)
    assert callable(signal.tf2zpk)
    assert callable(signal.zpk2sos)
    assert callable(signal.bilinear_tf)
    for old_name in (
        "chirp_z_transform",
        "design_bessel",
        "design_butterworth",
        "design_elliptic",
        "design_fir",
        "design_notch",
        "design_phase_compensator",
        "estimate_fir_order",
        "fir_transition_bandwidth",
        "frequency_domain_integrate",
        "frequency_vector",
        "general_cosine_window",
    ):
        assert not hasattr(signal, old_name)
    assert not hasattr(signal, "zpk_to_transfer_function")
    assert not hasattr(signal, "spectrogram")
    assert callable(signal.resample)
    assert signal.WhittakerSmoother.__name__ == "WhittakerSmoother"
    assert "neurale._native" not in newly_loaded
    assert not any(name.startswith("neurale.runtime") for name in newly_loaded)


def test_simulation_module_import_does_not_load_native_or_runtime() -> None:
    for module_name in list(sys.modules):
        if module_name == "neurale._native" or module_name.startswith("neurale.runtime"):
            del sys.modules[module_name]
        if module_name == "neurale.signal.simulation":
            del sys.modules[module_name]

    loaded_before = set(sys.modules)
    module = importlib.import_module("neurale.signal.simulation")
    newly_loaded = set(sys.modules) - loaded_before

    assert module.SignalGenerator.__name__ == "SignalGenerator"
    assert "neurale._native" not in newly_loaded
    assert not any(name.startswith("neurale.runtime") for name in newly_loaded)
