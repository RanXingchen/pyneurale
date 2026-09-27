#!/usr/bin/env python3

import importlib

from _subprocess_probe import probe_json


def test_models_import_does_not_load_unrelated_domains() -> None:
    code = """
import importlib
import json
import sys

models = importlib.import_module("neurale.models")
forbidden = (
    "neurale._native",
    "neurale._native_cuda",
    "neurale.io",
    "neurale.decoding",
    "neurale.devices",
    "neurale.visualization",
    "torch",
)
print(json.dumps({
    "has_empirical_cov": "empirical_cov" in vars(models),
    "has_evaluation": "confusion_matrix" in vars(models),
    "loaded_forbidden": [name for name in forbidden if name in sys.modules],
}))
"""
    result = probe_json(code)

    assert result["has_empirical_cov"]
    assert result["has_evaluation"]
    assert result["loaded_forbidden"] == []


def test_models_submodule_import_is_available() -> None:
    models = importlib.import_module("neurale.models")

    assert models.empirical_cov is not None
    assert models.PCA is not None
    assert models.LDA is not None
    assert models.knn is not None
    assert models.GaussianKDE is not None
    assert models.dtw is not None
    assert models.confusion_matrix is not None
    assert models.accuracy_score is not None
