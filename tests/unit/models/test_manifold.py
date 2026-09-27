# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

import copy
import pickle

import numpy as np
import pytest

from neurale._native_loader import load_native_namespace
from neurale.exceptions import ValidationError
from neurale.models import LPP
from neurale.runtime import runtime_context, runtime_info


def _require_cuda_manifold() -> None:
    native = load_native_namespace("models")

    if not hasattr(native, "cuda") or not hasattr(native.cuda, "manifold"):
        pytest.skip("CUDA LPP is not built")
    cuda = runtime_info(probe_cuda=True).cuda
    if cuda is None or not cuda.available:
        pytest.skip("no usable CUDA device is available")


def _graph_forms(X: np.ndarray, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    distances = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    indices = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbors]
    connectivity = np.zeros((X.shape[0], X.shape[0]))
    connectivity[np.arange(X.shape[0])[:, None], indices] = 1.0
    weights = 0.5 * (connectivity + connectivity.T)
    degree = np.diag(weights.sum(axis=1))
    return degree, degree - weights


def _assert_generalized_eigenvectors(
    components: np.ndarray,
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    eigenvalues = []
    for vector in components:
        left = numerator @ vector
        right = denominator @ vector
        eigenvalue = np.dot(vector, left) / np.dot(vector, right)
        eigenvalues.append(eigenvalue)
        np.testing.assert_allclose(left, eigenvalue * right, rtol=2e-6, atol=2e-7)
    return np.asarray(eigenvalues)


def _samples() -> np.ndarray:
    return np.array(
        [
            [-2.0, -1.0, 0.2, 0.0],
            [-1.3, -0.7, 0.5, 0.4],
            [-0.8, -0.2, 0.7, 0.9],
            [-0.1, 0.4, 1.0, 1.3],
            [0.6, 0.9, 0.4, 1.7],
            [1.2, 1.5, -0.1, 1.1],
            [1.8, 1.7, -0.6, 0.5],
            [2.4, 2.1, -0.9, -0.2],
        ],
        dtype=np.float64,
    )


def test_opp_satisfies_direct_projection_contract_and_is_stable() -> None:
    X = _samples()
    first = LPP(2, n_neighbors=4).fit(X)
    second = LPP(2, n_neighbors=4).fit(X)

    assert first.components_.shape == (2, X.shape[1])
    assert not first.components_.flags.writeable
    np.testing.assert_allclose(first.components_, second.components_)
    np.testing.assert_allclose(first.transform(X), X @ first.components_.T)

    degree, laplacian = _graph_forms(X, 4)
    centered = X - X.mean(axis=0)
    num = centered.T @ degree @ centered
    den = X.T @ laplacian @ X + 1e-6 * np.eye(X.shape[1])
    eigenvalues = _assert_generalized_eigenvectors(
        first.components_,
        num,
        den,
    )
    assert np.all(np.diff(eigenvalues) >= 0.0)


def test_lpp_path_uses_smallest_positive_generalized_directions() -> None:
    X = _samples()
    model = LPP(2, n_neighbors=4, proj_method="LPP").fit(X)
    degree, laplacian = _graph_forms(X, 4)

    assert model.components_.shape == (2, X.shape[1])
    _assert_generalized_eigenvectors(
        model.components_,
        X.T @ laplacian @ X,
        X.T @ degree @ X,
    )


def test_transform_reuses_exact_output_buffer() -> None:
    X = _samples()
    model = LPP(2, n_neighbors=3).fit(X)
    out = np.empty((X.shape[0], 2), dtype=np.float64)

    returned = model.transform(X, out=out)

    assert returned is out
    np.testing.assert_allclose(out, X @ model.components_.T)


@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, lambda value: pickle.loads(pickle.dumps(value))],
)
def test_lpp_copy_and_pickle_reconstruct_cpu_projection(clone) -> None:
    X = _samples()
    with runtime_context(device="cpu"):
        model = LPP(2, n_neighbors=4, proj_method="LPP").fit(X)

    with runtime_context(device="cuda"):
        restored = clone(model)
        actual = restored.transform(X)

    assert model.device_ == "cpu"
    assert restored.device_ == "cpu"
    assert restored._model is not model._model
    np.testing.assert_array_equal(restored.components_, model.components_)
    np.testing.assert_array_equal(actual, model.transform(X))


def test_lpp_validates_parameters_and_inputs() -> None:
    X = _samples()
    with pytest.raises(ValidationError, match="n_components"):
        LPP(0)
    with pytest.raises(ValidationError, match="n_neighbors"):
        LPP(1, n_neighbors=0)
    with pytest.raises(ValidationError, match="metric"):
        LPP(1, metric="cosine")
    with pytest.raises(ValidationError, match="proj_method"):
        LPP(1, proj_method="OLPP")
    with pytest.raises(ValidationError, match="2D"):
        LPP(1).fit(X[:, 0])
    with pytest.raises(ValidationError, match="real"):
        LPP(1).fit(X.astype(np.complex128))
    with pytest.raises(ValidationError, match="finite"):
        LPP(1).fit(np.where(np.arange(X.size).reshape(X.shape) == 0, np.nan, X))
    with pytest.raises(ValidationError, match="n_components"):
        LPP(X.shape[1] + 1).fit(X)
    with pytest.raises(ValidationError, match="n_neighbors"):
        LPP(1, n_neighbors=X.shape[0] + 1).fit(X)


def test_lpp_requires_fit_and_matching_transform_shapes() -> None:
    X = _samples()
    model = LPP(2, n_neighbors=3)
    with pytest.raises(ValidationError, match="not fitted"):
        model.transform(X)

    model.fit(X)
    with pytest.raises(ValidationError, match="same number of features"):
        model.transform(X[:, :-1])
    with pytest.raises(ValidationError, match="invalid shape"):
        model.transform(X, out=np.empty((X.shape[0], 1)))


@pytest.mark.parametrize("proj_method", ["OPP", "LPP"])
@pytest.mark.gpu
def test_cuda_knn_path_matches_cpu_projection(proj_method: str) -> None:
    _require_cuda_manifold()
    X = _samples()

    with runtime_context(device="cpu"):
        expected = LPP(
            2,
            n_neighbors=4,
            proj_method=proj_method,
        ).fit(X)
    with runtime_context(device="cuda", cuda_device=0):
        actual = LPP(
            2,
            n_neighbors=4,
            proj_method=proj_method,
        ).fit(X)

    np.testing.assert_array_equal(actual.components_, expected.components_)
    np.testing.assert_array_equal(actual.transform(X), expected.transform(X))
    assert expected.device_ == "cpu"
    assert actual.device_ == "cpu"


@pytest.mark.gpu
def test_cuda_lpp_preserves_knn_neighbor_limit() -> None:
    _require_cuda_manifold()
    X = np.random.default_rng(42).normal(size=(65, 4)).astype(np.float64)

    with runtime_context(device="cuda", cuda_device=0):
        with pytest.raises(ValidationError, match="at most 64"):
            LPP(1, n_neighbors=65).fit(X)
