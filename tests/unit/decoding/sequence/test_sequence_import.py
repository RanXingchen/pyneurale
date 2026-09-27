#!/usr/bin/env python3

from __future__ import annotations

import importlib

from _subprocess_probe import probe_json


def test_importing_sequence_decoding_pulls_in_nothing_heavy() -> None:
    """The search is arithmetic over arrays: no native code, no framework, no I/O."""

    code = """
import importlib
import json
import sys

sequence = importlib.import_module("neurale.decoding.sequence")
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
    "tkinter",
    "matplotlib",
)
print(json.dumps({
    "exports": sorted(sequence.__all__),
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
        "BeamSearchDecoder",
        "Hypothesis",
        "NgramLanguageModel",
        "Vocabulary",
    ]


def test_sequence_exports_are_importable() -> None:
    sequence = importlib.import_module("neurale.decoding.sequence")

    for name in sequence.__all__:
        assert getattr(sequence, name) is not None


def test_sequence_decoding_keeps_own_namespace() -> None:
    """It is a different kind of decoding, and shares no base class with the rest.

    Nothing here consumes a feature schema, a timeline, or a device, so these
    classes are not exported into ``neurale.decoding`` beside the decoders
    that do -- a caller reaches for them by the namespace that says what they
    are.
    """

    decoding = importlib.import_module("neurale.decoding")
    sequence = importlib.import_module("neurale.decoding.sequence")

    assert set(sequence.__all__).isdisjoint(decoding.__all__)
    assert not issubclass(sequence.BeamSearchDecoder, decoding.BaseDecoder)


def test_decoders_share_no_module_level_state() -> None:
    """Two decoders over one language model never see each other's search."""

    sequence = importlib.import_module("neurale.decoding.sequence")
    import numpy as np

    vocabulary = sequence.Vocabulary(["a", "b", "stop"], bos="<s>", eos="stop")
    model = sequence.NgramLanguageModel(vocabulary, smoothing=1.0).fit([["a", "b"], ["b", "a"]])

    first = sequence.BeamSearchDecoder(model)
    second = sequence.BeamSearchDecoder(model)
    first.update(np.array([[0.7, 0.2, 0.1]]))

    assert second.n_steps == 0
    assert [item.tokens for item in second.active] == [()]
