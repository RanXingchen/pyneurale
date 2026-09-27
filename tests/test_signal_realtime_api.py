#!/usr/bin/env python3

from __future__ import annotations

import inspect

import numpy as np

import neurale.signal as public_signal
import neurale.signal.filtering as public_filtering
from neurale.signal import FirCoefficients, FirFilter, fir_filter


def test_filtering_api_does_not_export_into_names() -> None:
    for module in (public_signal, public_filtering):
        exported = set(getattr(module, "__all__", ()))
        visible = {name for name in dir(module) if not name.startswith("_")}
        assert not {name for name in exported | visible if name.endswith("_into")}


def test_fir_filter_has_no_backend_or_output_keywords() -> None:
    assert set(inspect.signature(fir_filter).parameters) == {
        "x",
        "coefs",
        "axis",
        "zi",
        "return_state",
        "check_finite",
    }


def test_fir_realtime_processes_supplied_frame_in_place() -> None:
    rng = np.random.default_rng(123)
    values = np.ascontiguousarray(rng.standard_normal((16, 2)), dtype=np.float64)
    source = values.copy()
    coefs = FirCoefficients([0.25, 0.5, 0.25])
    processor = FirFilter(coefs, n_channels=2)

    result = processor.process(values)

    assert result is values
    np.testing.assert_allclose(values, fir_filter(source, coefs))
