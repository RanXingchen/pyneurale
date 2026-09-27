#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale._native_loader import load_native_namespace
from neurale.exceptions import NativeUnavailableError, ValidationError
from neurale.models import LDA
from neurale.runtime import runtime_context


def _native_models():
    try:
        return load_native_namespace("models")
    except NativeUnavailableError:
        return None


def test_lda_lsqr_matches_legacy_reference_without_shrinkage() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=float,
    )
    train_y = np.array([1, 1, 1, 2, 2, 2])
    test_x = np.array([[3, 1], [1, -1], [2, -3], [-1, 3]], dtype=float)

    lda = LDA(solver="lsqr", shrinkage=None).fit(train_x, train_y)

    assert hasattr(lda, "_model")
    assert np.allclose(lda.coef_, [[-2.64418859e-15, 1.2e01]])
    assert np.allclose(lda.intercept_, [0.0])
    assert lda.n_outputs_ == 1
    assert np.array_equal(lda.predict(test_x), [2, 1, 1, 2])
    assert np.allclose(
        lda.predict_proba(test_x),
        [
            [6.14417460e-06, 9.99993856e-01],
            [9.99993856e-01, 6.14417460e-06],
            [1.00000000e00, 2.31952283e-16],
            [2.22044605e-16, 1.00000000e00],
        ],
    )
    assert lda.decision_function(test_x).shape == (4,)
    assert lda.device_ == "cpu"
    with pytest.raises(AttributeError):
        lda.device_ = "cuda"
    with runtime_context(device="cuda"):
        assert lda.device_ == "cpu"
        assert np.array_equal(lda.predict(test_x), [2, 1, 1, 2])


def test_lda_inference_out_reuses_output_buffers() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=np.float64,
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])
    test_x = np.array([[3, 1], [1, -1], [2, -3], [-1, 3]], dtype=np.float64)

    lda = LDA().fit(train_x, train_y)
    scores = np.empty(test_x.shape[0], dtype=np.float64)
    proba = np.empty((test_x.shape[0], 2), dtype=np.float64)

    returned_scores = lda.decision_function(test_x, out=scores)
    returned_proba = lda.predict_proba(test_x, out=proba)

    assert returned_scores is scores
    assert returned_proba is proba
    assert np.allclose(scores, lda.decision_function(test_x))
    assert np.allclose(proba, lda.predict_proba(test_x))


def test_lda_realtime_contract_rejects_bad_buffers() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=np.float64,
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])
    test_x = np.array([[3, 1], [1, -1], [2, -3], [-1, 3]], dtype=np.float64)

    lda = LDA().fit(train_x, train_y)

    with pytest.raises(ValueError):
        lda.coef_[0, 0] = 0.0
    with pytest.raises(ValueError):
        lda.classes_[0] = -1
    with pytest.raises(AttributeError):
        lda.coef_ = np.zeros_like(lda.coef_)
    with pytest.raises(AttributeError):
        lda.classes_ = np.array([0, 1])
    with pytest.raises(AttributeError):
        lda.n_features_in_ = 0
    with pytest.raises(AttributeError):
        lda.n_outputs_ = 2
    with pytest.raises(ValidationError, match="invalid shape"):
        lda.decision_function(test_x, out=np.empty((test_x.shape[0], 1)))
    with pytest.raises(ValidationError, match="invalid shape"):
        lda.predict_proba(test_x, out=np.empty((2, test_x.shape[0])))

    readonly = np.empty(test_x.shape[0], dtype=np.float64)
    readonly.setflags(write=False)
    with pytest.raises(ValidationError, match="writable"):
        lda.decision_function(test_x, out=readonly)
    with pytest.raises(ValidationError, match="overlap"):
        lda.predict_proba(test_x, out=test_x)
    with pytest.raises(TypeError):
        lda._model.decision_function(test_x.astype(np.float32))
    with pytest.raises(TypeError):
        lda._model.predict_proba(np.asfortranarray(test_x))
    with pytest.raises(TypeError):
        lda._model.predict_proba(
            test_x,
            np.empty((test_x.shape[0], 2), dtype=np.float32),
        )


def test_lda_fitted_contract_uses_native_model_state() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=np.float64,
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])
    lda = LDA().fit(train_x, train_y)

    del lda._model

    with pytest.raises(ValidationError, match="not fitted"):
        lda.decision_function(train_x)
    with pytest.raises(ValidationError, match="not fitted"):
        _ = lda.coef_


def test_lda_predict_uses_decision_scores_without_probability() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=np.float64,
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])
    test_x = np.array([[3, 1], [1, -1], [2, -3], [-1, 3]], dtype=np.float64)

    lda = LDA().fit(train_x, train_y)

    def fail_predict_proba(_x):
        raise AssertionError("predict_proba should not be called")

    lda.predict_proba = fail_predict_proba

    assert np.array_equal(lda.predict(test_x), [1, 0, 0, 1])


def test_lda_binary_predict_tie_matches_probability_argmax() -> None:
    train_x = np.array(
        [
            [-2.0, -1.0],
            [-2.0, 1.0],
            [-1.0, 0.0],
            [2.0, 1.0],
            [2.0, -1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    train_y = np.array(["left", "left", "left", "right", "right", "right"])
    test_x = np.array([[0.0, 0.0]], dtype=np.float64)

    lda = LDA().fit(train_x, train_y)

    assert np.allclose(lda.decision_function(test_x), [0.0])
    assert np.allclose(lda.predict_proba(test_x), [[0.5, 0.5]])
    assert np.array_equal(lda.predict(test_x), ["left"])


def test_lda_lsqr_matches_legacy_reference_with_auto_shrinkage() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=float,
    )
    train_y = np.array([1, 1, 1, 2, 2, 2])
    test_x = np.array([[3, 1], [1, -1], [2, -3], [-1, 3]], dtype=float)

    lda = LDA(solver="lsqr", shrinkage="auto").fit(train_x, train_y)

    assert np.allclose(lda.coef_, [[3.46987952, 9.10843373]])
    assert np.allclose(lda.intercept_, [0.0])
    assert np.array_equal(lda.predict(test_x), [2, 1, 1, 2])
    assert np.allclose(
        lda.predict_proba(test_x),
        [
            [3.33740502e-09, 9.99999997e-01],
            [9.96454606e-01, 3.54539425e-03],
            [9.99999999e-01, 1.40175310e-09],
            [4.36235492e-11, 1.00000000e00],
        ],
    )


def test_lda_supports_string_labels_and_validates_inputs() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=float,
    )
    train_y = np.array(["left", "left", "left", "right", "right", "right"])

    lda = LDA().fit(train_x, train_y)

    assert np.array_equal(lda.classes_, ["left", "right"])
    assert np.array_equal(lda.predict(np.array([[2.0, 2.0]])), ["right"])
    with pytest.raises(ValidationError, match="solver"):
        LDA(solver="svd")
    with pytest.raises(ValidationError, match="shrinkage"):
        LDA(shrinkage=1.5).fit(train_x, train_y)


def test_lda_rejects_complex_and_nonfinite_numeric_labels() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [1, 1], [2, 1]],
        dtype=float,
    )

    with pytest.raises(ValidationError, match="real-valued"):
        LDA().fit(train_x, np.array([0.0, 0.0, 1.0, 1.0 + 0.0j]))
    with pytest.raises(ValidationError, match="finite"):
        LDA().fit(train_x, np.array([0.0, 0.0, 1.0, np.nan]))
    with pytest.raises(ValidationError, match="finite"):
        LDA().fit(train_x, np.array([0.0, 0.0, 1.0, np.inf]))


def test_lda_auto_shrinkage_handles_class_constant_features() -> None:
    train_x = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [3.0, 1.0],
            [4.0, 2.0],
        ]
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])

    lda = LDA(shrinkage="auto").fit(train_x, train_y)

    assert lda.predict_proba(train_x).shape == (6, 2)


def test_lda_auto_shrinkage_is_stable_for_tiny_scaled_inputs() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=float,
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])
    test_x = np.array([[3, 1], [1, -1], [2, -3], [-1, 3]], dtype=float)

    reference = LDA(shrinkage="auto").fit(train_x, train_y)
    scaled = LDA(shrinkage="auto").fit(train_x * 1e-12, train_y)

    assert np.allclose(
        scaled.predict_proba(test_x * 1e-12),
        reference.predict_proba(test_x),
    )


def test_lda_auto_shrinkage_is_stable_for_large_feature_offsets() -> None:
    train_x = np.array(
        [
            [0.0, 0.0],
            [100.0, 1.0],
            [200.0, 3.0],
            [300.0, 4.0],
            [400.0, 5.0],
            [500.0, 7.0],
        ],
        dtype=np.float64,
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])
    offset = np.array([1e8, 2e8], dtype=np.float64)

    reference = LDA(shrinkage="auto").fit(train_x, train_y)
    shifted = LDA(shrinkage="auto").fit(train_x + offset, train_y)

    assert np.allclose(shifted.covariance_, reference.covariance_)
    assert np.allclose(shifted.coef_, reference.coef_)


def test_lda_fixed_shrinkage_solve_is_stable_for_tiny_scaled_inputs() -> None:
    train_x = np.array(
        [[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]],
        dtype=float,
    )
    train_y = np.array([0, 0, 0, 1, 1, 1])

    lda = LDA(shrinkage=0.5).fit(train_x * 1e-12, train_y)

    assert np.all(np.isfinite(lda.coef_))
    assert np.all(np.isfinite(lda.intercept_))


def test_lda_native_rejects_invalid_fixed_shrinkage() -> None:
    native = _native_models()
    assert native is not None
    train_x = np.array(
        [[-1.0, -1.0], [-2.0, -1.0], [-3.0, -2.0], [1.0, 1.0], [2.0, 1.0], [3.0, 2.0]]
    )
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.uintp)

    with pytest.raises(ValueError, match="shrinkage"):
        native.classification.fit_lda(train_x, labels, 2, "fixed", 1.5)
