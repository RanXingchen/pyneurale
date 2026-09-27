#!/usr/bin/env python3

from __future__ import annotations

import importlib

from _subprocess_probe import probe_json


def test_importing_decoding_has_no_native_or_side_effects() -> None:
    """Decoders declare their model composition; none of it initializes hardware or I/O."""

    code = """
import importlib
import json
import sys

decoding = importlib.import_module("neurale.decoding")
forbidden = (
    "neurale._native",
    "neurale._native_cuda",
    "neurale.io",
    "neurale.recording",
    "neurale.streaming",
    "neurale.features",
    "neurale.signal",
    "neurale.devices",
    "torch",
    "scipy",
    "zarr",
)
print(json.dumps({
    "exports": sorted(decoding.__all__),
    "loaded_forbidden": [name for name in forbidden if name in sys.modules],
    "logging_configured": bool(
        importlib.import_module("neurale.runtime").is_logging_configured()
    ),
}))
"""
    result = probe_json(code)

    assert result["loaded_forbidden"] == []
    assert not result["logging_configured"]
    assert result["exports"] == [
        "BaseDecoder",
        "ClassificationDecoder",
        "ClassificationPrediction",
        "ClassificationTarget",
        "ContinuousDecoder",
        "ContinuousTargetSchema",
        "FeatureSchema",
        "FitSegment",
        "KalmanDecoder",
        "LDADecoder",
        "LinearDecoder",
        "RidgeDecoder",
    ]


def test_decoding_exports_are_importable() -> None:
    decoding = importlib.import_module("neurale.decoding")

    for name in decoding.__all__:
        assert getattr(decoding, name) is not None


def test_decoding_depends_on_data_models_and_runtime_only() -> None:
    """Typed data, the model library a decoder composes, and the device policy.

    ``neurale.models`` is a downward dependency and is pure Python at import
    time: a decoder is a lifecycle around estimators that live there, and it
    resolves their native kernels only when a fit actually runs.
    """

    code = """
import importlib
import json
import sys

importlib.import_module("neurale.decoding")
domains = (
    "data",
    "decoding",
    "features",
    "io",
    "models",
    "recording",
    "runtime",
    "signal",
    "sorting",
    "streaming",
)
print(json.dumps({
    "loaded": [name for name in domains if f"neurale.{name}" in sys.modules],
}))
"""
    assert probe_json(code)["loaded"] == ["data", "decoding", "models", "runtime"]
