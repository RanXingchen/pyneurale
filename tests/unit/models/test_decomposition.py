#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.exceptions import ValidationError
from neurale.models import PCA
from neurale.runtime import runtime_context


def _reference_pca(
    X: np.ndarray,
    n_components: int,
    *,
    center: bool,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    mean = np.mean(X, axis=0) if center else np.zeros(X.shape[1])
    centered = X - mean if center else X
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    components = components[:n_components].copy()
    for row in components:
        pivot = np.argmax(np.abs(row))
        if row[pivot] < 0.0:
            row *= -1.0
    dof = X.shape[0] - 1 if center else X.shape[0]
    rank_limit = min(X.shape[1], dof)
    all_variance = singular_values[:rank_limit] ** 2 / dof
    explained = singular_values[:n_components] ** 2 / dof
    ratio = explained / np.sum(all_variance)
    scores = centered @ components.T
    return mean, components, singular_values[:n_components], explained, ratio, scores


def test_pca_matches_svd_reference() -> None:
    X = np.array(
        [
            [2.5, 2.4, 0.5],
            [0.5, 0.7, 1.0],
            [2.2, 2.9, 1.5],
            [1.9, 2.2, 0.7],
            [3.1, 3.0, 2.0],
            [2.3, 2.7, 1.1],
        ],
        dtype=float,
    )

    pca = PCA(n_components=2).fit(X)
    mean, components, singular, explained, ratio, scores = _reference_pca(
        X,
        2,
        center=True,
    )

    assert np.allclose(pca.mean_, mean)
    assert np.allclose(pca.components_, components)
    assert np.allclose(pca.offset_, -mean @ components.T)
    assert np.allclose(pca.singular_values_, singular)
    assert np.allclose(pca.explained_variance_, explained)
    assert np.allclose(pca.explained_variance_ratio_, ratio)
    assert np.allclose(pca.transform(X), scores)
    assert hasattr(pca, "_model")
    assert pca.n_features_in_ == X.shape[1]
    assert pca.n_components_ == 2
    assert pca.device_ == "cpu"
    with pytest.raises(AttributeError):
        pca.device_ = "cuda"
    with runtime_context(device="cuda"):
        assert pca.device_ == "cpu"
        assert np.allclose(pca.transform(X), scores)


def test_pca_transform_out_reuses_output_buffer() -> None:
    X = np.array(
        [
            [2.5, 2.4, 0.5],
            [0.5, 0.7, 1.0],
            [2.2, 2.9, 1.5],
            [1.9, 2.2, 0.7],
        ],
        dtype=np.float64,
    )

    pca = PCA(n_components=2).fit(X)
    out = np.empty((X.shape[0], 2), dtype=np.float64)

    returned = pca.transform(X, out=out)

    assert returned is out
    assert np.allclose(out, pca.transform(X))


def test_pca_realtime_contract_rejects_bad_buffers() -> None:
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
        ],
        dtype=np.float64,
    )

    pca = PCA(n_components=2).fit(X)

    with pytest.raises(ValueError):
        pca.components_[0, 0] = 0.0
    with pytest.raises(AttributeError):
        pca.components_ = np.zeros_like(pca.components_)
    with pytest.raises(AttributeError):
        pca.n_samples_ = 0
    with pytest.raises(AttributeError):
        pca.n_features_in_ = 0
    with pytest.raises(AttributeError):
        pca.n_components_ = 1
    with pytest.raises(ValidationError, match="invalid shape"):
        pca.transform(X, out=np.empty((1, X.shape[0] * 2), dtype=np.float64))

    readonly = np.empty((X.shape[0], 2), dtype=np.float64)
    readonly.setflags(write=False)
    with pytest.raises(ValidationError, match="writable"):
        pca.transform(X, out=readonly)
    with pytest.raises(ValidationError, match="overlap"):
        pca.transform(X, out=X)
    with pytest.raises(TypeError):
        pca._model.transform(X.astype(np.float32))
    with pytest.raises(TypeError):
        pca._model.transform(np.asfortranarray(X))
    with pytest.raises(TypeError):
        pca._model.transform(X, np.empty((X.shape[0], 2), dtype=np.float32))


def test_pca_fitted_contract_uses_native_model_state() -> None:
    X = np.array(
        [
            [2.5, 2.4],
            [0.5, 0.7],
            [2.2, 2.9],
            [1.9, 2.2],
        ],
        dtype=np.float64,
    )
    pca = PCA(n_components=2).fit(X)

    del pca._model

    with pytest.raises(ValidationError, match="not fitted"):
        pca.transform(X)
    with pytest.raises(ValidationError, match="not fitted"):
        _ = pca.components_


def test_pca_supports_uncentered_transform() -> None:
    X = np.array(
        [
            [7.0, 26.0, 6.0, 60.0],
            [1.0, 29.0, 15.0, 52.0],
            [11.0, 56.0, 8.0, 20.0],
        ]
    )

    pca = PCA(n_components=2, center=False)
    transformed = pca.fit_transform(X)
    _, components, _, _, _, scores = _reference_pca(X, 2, center=False)

    assert np.allclose(pca.mean_, np.zeros(X.shape[1]))
    assert np.allclose(pca.offset_, np.zeros(2))
    assert np.allclose(pca.components_, components)
    assert np.allclose(transformed, scores)


def test_pca_supports_high_dimensional_dual_builtin_path() -> None:
    X = np.array(
        [
            [1.0, 0.0, 2.0, 4.0, 1.0],
            [0.0, 2.0, 1.0, 3.0, 5.0],
            [3.0, 1.0, 0.0, 2.0, 4.0],
        ]
    )

    pca = PCA(n_components=2).fit(X)
    _, components, singular, explained, ratio, scores = _reference_pca(
        X,
        2,
        center=True,
    )

    assert np.allclose(pca.components_, components)
    assert np.allclose(pca.singular_values_, singular)
    assert np.allclose(pca.explained_variance_, explained)
    assert np.allclose(pca.explained_variance_ratio_, ratio)
    assert np.allclose(pca.transform(X), scores)


def test_pca_components_are_scale_invariant_for_tiny_inputs() -> None:
    X = np.array(
        [
            [1.0, 2.0, 4.0],
            [2.0, 0.0, 3.0],
            [3.0, 1.0, 0.0],
            [4.0, 3.0, 2.0],
        ]
    )

    reference = PCA(n_components=2).fit(X)
    scaled = PCA(n_components=2).fit(X * 1e-12)

    assert np.allclose(scaled.components_, reference.components_)
    assert np.allclose(scaled.singular_values_, reference.singular_values_ * 1e-12)


def test_pca_dual_path_orthonormal_on_zero_rank_data() -> None:
    X = np.ones((3, 5))

    pca = PCA(n_components=2).fit(X)

    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(2))
    assert np.allclose(pca.transform(X), np.zeros((3, 2)))


@pytest.mark.parametrize("scale", [1.0, 1e6, 1e12, 1e15])
def test_pca_rank_deficient_dual_path_is_scale_invariant(scale: float) -> None:
    X = scale * np.array(
        [
            [-1.0, -2.0, -3.0, -4.0, -5.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ],
        dtype=np.float64,
    )

    pca = PCA(n_components=2).fit(X)

    assert np.all(np.isfinite(pca.components_))
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(2), atol=1e-12)


def test_pca_random_dual_path_components_are_orthonormal() -> None:
    rng = np.random.default_rng(1234)
    X = (rng.normal(size=(5, 2)) @ rng.normal(size=(2, 12))).astype(np.float64)

    pca = PCA(n_components=4).fit(X)

    assert np.all(np.isfinite(pca.components_))
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(4), atol=1e-12)


def test_pca_builtin_path_zeros_unreliable_singular_values() -> None:
    rng = np.random.default_rng(5678)
    X = (rng.normal(size=(10, 2)) @ rng.normal(size=(2, 5))).astype(np.float64)

    pca = PCA(n_components=4, center=False).fit(X)

    assert np.allclose(pca.singular_values_[2:], 0.0)
    assert np.allclose(pca.explained_variance_[2:], 0.0)


def test_pca_validates_inputs() -> None:
    with pytest.raises(ValidationError, match="n_components"):
        PCA(3).fit(np.ones((4, 2)))
    with pytest.raises(ValidationError, match="real-valued"):
        PCA(1).fit(np.array([[1.0 + 0.0j], [2.0 + 0.0j]]))
    with pytest.raises(ValidationError, match="not fitted"):
        PCA(1).transform(np.ones((2, 1)))
