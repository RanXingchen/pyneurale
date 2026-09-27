#!/usr/bin/env python3

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from neurale.exceptions import ValidationError
from neurale.models.evaluation import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def _reference_confusion(
    y_true: list[object],
    y_pred: list[object],
    labels: list[object] | None,
) -> tuple[list[list[int]], list[object]]:
    # Small dependency-free reference mirroring the implementation's contract:
    # inference uses stable first-occurrence order (y_true then unseen y_pred),
    # explicit labels select a subset and drop samples whose true or predicted
    # label falls outside it. Returns (matrix, ordered_labels).
    if labels is None:
        ordered: list[object] = []
        idx: dict[object, int] = {}
        for value in list(y_true) + list(y_pred):
            if value not in idx:
                idx[value] = len(ordered)
                ordered.append(value)
    else:
        ordered = list(labels)
        idx = {value: i for i, value in enumerate(ordered)}
    n = len(ordered)
    matrix = [[0] * n for _ in range(n)]
    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        if true_value in idx and pred_value in idx:
            matrix[idx[true_value]][idx[pred_value]] += 1
    return matrix, ordered


def _reference_score(
    y_true: list[object],
    y_pred: list[object],
    labels: list[object] | None,
    average: str | None,
    metric: str,
) -> list[float] | float:
    # One-vs-rest reference: build the FULL union matrix (declared labels first,
    # then observed-not-declared) and compute per-selected-label tp / predicted
    # (column sum over all true classes) / support (row sum over all predicted
    # classes), so cross-subset FP/FN are counted. This deliberately differs from
    # the clipped ``_reference_confusion`` used for confusion_matrix parity.
    if labels is None:
        ordered: list[object] = []
        idx: dict[object, int] = {}
        for value in list(y_true) + list(y_pred):
            if value not in idx:
                idx[value] = len(ordered)
                ordered.append(value)
        selected = list(range(len(ordered)))
    else:
        ordered = list(labels)
        idx = {value: i for i, value in enumerate(ordered)}
        for value in list(y_true) + list(y_pred):
            if value not in idx:
                idx[value] = len(ordered)
                ordered.append(value)
        selected = list(range(len(labels)))
    n_full = len(ordered)
    matrix = [[0] * n_full for _ in range(n_full)]
    for true_value, pred_value in zip(y_true, y_pred, strict=True):
        matrix[idx[true_value]][idx[pred_value]] += 1
    tp = [matrix[i][i] for i in selected]
    predicted = [sum(matrix[r][c] for r in range(n_full)) for c in selected]
    support = [sum(matrix[r][c] for c in range(n_full)) for r in selected]
    precision = [tp[i] / predicted[i] if predicted[i] else 0.0 for i in range(len(selected))]
    recall = [tp[i] / support[i] if support[i] else 0.0 for i in range(len(selected))]
    f1 = [2 * p * r / (p + r) if (p + r) else 0.0 for p, r in zip(precision, recall, strict=True)]
    scores = {"precision": precision, "recall": recall, "f1": f1}[metric]
    if average is None:
        return scores
    if average == "micro":
        total_tp = sum(tp)
        micro_p = total_tp / sum(predicted) if sum(predicted) else 0.0
        micro_r = total_tp / sum(support) if sum(support) else 0.0
        if metric == "precision":
            return micro_p
        if metric == "recall":
            return micro_r
        return 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) else 0.0
    if average == "macro":
        return sum(scores) / len(scores) if scores else 0.0
    total_support = sum(support)
    if total_support == 0:
        return 0.0
    return (
        sum(score * weight for score, weight in zip(scores, support, strict=True)) / total_support
    )


def test_confusion_matrix_uses_explicit_label_order() -> None:
    y_true = np.array(["cat", "ant", "cat", "bird", "ant"])
    y_pred = np.array(["ant", "ant", "cat", "cat", "bird"])

    result = confusion_matrix(y_true, y_pred, labels=["bird", "cat", "ant"])

    assert np.array_equal(result, np.array([[0, 1, 0], [0, 1, 1], [1, 0, 1]]))


def test_explicit_tuple_labels_preserve_requested_order() -> None:
    # Equal-length tuple labels collapse to a 2D array under np.asarray, but the
    # inference path accepts the same labels as 1D object-array elements. The
    # explicit path must treat each tuple as one scalar label and honor the
    # requested order rather than rejecting the contract as non-1D.
    labels = [("left", 1), ("right", 2)]
    y_true = np.empty(2, dtype=object)
    y_true[:] = labels
    y_pred = np.empty(2, dtype=object)
    y_pred[:] = labels

    result = confusion_matrix(y_true, y_pred, labels=labels)

    assert np.array_equal(result, np.array([[1, 0], [0, 1]]))


def test_inferred_label_order_follows_first_occurrence() -> None:
    first = object()
    second = "second"
    third = 3
    y_true = np.array([first, second, first], dtype=object)
    y_pred = np.array([third, second, first], dtype=object)

    result = confusion_matrix(y_true, y_pred)

    assert np.array_equal(result, np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]]))


def test_accuracy_score_supports_generic_labels() -> None:
    y_true = np.array(["rest", "move", "rest", "move"])
    y_pred = np.array(["rest", "rest", "rest", "move"])

    assert accuracy_score(y_true, y_pred) == 0.75


def test_class_scores_follow_label_order() -> None:
    y_true = np.array(["a", "a", "b", "b", "c"])
    y_pred = np.array(["a", "b", "b", "c", "c"])
    labels = ["c", "a", "b"]

    assert np.allclose(precision_score(y_true, y_pred, labels=labels), [0.5, 1.0, 0.5])
    assert np.allclose(recall_score(y_true, y_pred, labels=labels), [1.0, 0.5, 0.5])
    assert np.allclose(f1_score(y_true, y_pred, labels=labels), [2 / 3, 2 / 3, 0.5])


def test_score_averages_have_multiclass_semantics() -> None:
    y_true = np.array(["a", "a", "b", "b", "c"])
    y_pred = np.array(["a", "b", "b", "c", "c"])
    labels = ["a", "b", "c"]

    assert precision_score(y_true, y_pred, labels=labels, average="micro") == 0.6
    assert recall_score(y_true, y_pred, labels=labels, average="micro") == 0.6
    assert f1_score(y_true, y_pred, labels=labels, average="micro") == 0.6
    assert precision_score(y_true, y_pred, labels=labels, average="macro") == pytest.approx(2 / 3)
    assert recall_score(y_true, y_pred, labels=labels, average="macro") == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred, labels=labels, average="weighted") == pytest.approx(0.6)


def test_degenerate_classes_score_zero() -> None:
    y_true = np.array(["present", "present"])
    y_pred = np.array(["predicted-only", "predicted-only"])
    labels = ["present", "predicted-only", "absent"]

    assert np.array_equal(
        confusion_matrix(y_true, y_pred, labels=labels),
        np.array([[0, 2, 0], [0, 0, 0], [0, 0, 0]]),
    )
    assert np.array_equal(precision_score(y_true, y_pred, labels=labels), [0.0, 0.0, 0.0])
    assert np.array_equal(recall_score(y_true, y_pred, labels=labels), [0.0, 0.0, 0.0])
    assert np.array_equal(f1_score(y_true, y_pred, labels=labels), [0.0, 0.0, 0.0])


def test_empty_targets_require_explicit_labels() -> None:
    empty = np.array([], dtype=int)

    with pytest.raises(ValidationError, match="cannot be inferred"):
        confusion_matrix(empty, empty)
    with pytest.raises(ValidationError, match="must not be empty"):
        accuracy_score(empty, empty)

    assert np.array_equal(confusion_matrix(empty, empty, labels=[0, 1]), np.zeros((2, 2)))
    assert np.array_equal(f1_score(empty, empty, labels=[0, 1]), [0.0, 0.0])
    assert f1_score(empty, empty, labels=[0, 1], average="micro") == 0.0
    assert f1_score(empty, empty, labels=[0, 1], average="macro") == 0.0
    assert f1_score(empty, empty, labels=[0, 1], average="weighted") == 0.0


@pytest.mark.parametrize(
    ("y_true", "y_pred", "labels", "message"),
    [
        (np.array([0]), np.array([0, 1]), [0, 1], "same number"),
        (np.array([0]), np.array([0]), [0, 0], "duplicates"),
        (np.array([0]), np.array([0]), [], "at least one"),
        (np.array([0]), np.array([0]), [0, None], "missing values"),
    ],
)
def test_evaluation_rejects_invalid_label_contracts(y_true, y_pred, labels, message) -> None:
    with pytest.raises(ValidationError, match=message):
        confusion_matrix(y_true, y_pred, labels=labels)


def test_explicit_labels_select_subset_and_drop_others() -> None:
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 2])
    # labels=[0, 1] keeps only samples where BOTH targets fall in the subset:
    # sample 0 (0,0) and sample 1 (1,1). Sample 2 (true=2) and sample 3 (pred=2)
    # are dropped because 2 is outside the subset.
    result = confusion_matrix(y_true, y_pred, labels=[0, 1])

    assert np.array_equal(result, np.array([[1, 0], [0, 1]]))


def test_explicit_labels_may_declare_absent_classes_as_zero_rows() -> None:
    y_true = np.array([0, 0])
    y_pred = np.array([0, 1])
    # "absent" appears in neither target -> its row and column stay all zero.
    result = confusion_matrix(y_true, y_pred, labels=[0, 1, "absent"])

    assert np.array_equal(result, np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]]))


def test_subset_scores_unaffected_by_correct_out_of_set_samples() -> None:
    y_true = np.array([0, 1, 2])
    y_pred = np.array([0, 1, 2])
    # One-vs-rest: the (2, 2) sample is a true positive for the unselected class
    # 2, so it contributes no false positive or false negative to the selected
    # classes 0 and 1. Their per-class precision/recall/f1 stay 1.0 — not because
    # the sample was dropped, but because it was correctly classified outside.
    labels = [0, 1]

    assert np.allclose(precision_score(y_true, y_pred, labels=labels), [1.0, 1.0])
    assert np.allclose(recall_score(y_true, y_pred, labels=labels), [1.0, 1.0])
    assert np.allclose(f1_score(y_true, y_pred, labels=labels), [1.0, 1.0])


def test_subset_labels_still_reject_invalid_sentinels_in_targets() -> None:
    # A valid-but-unselected label (int 2) is silently dropped by the subset
    # contract, but an invalid sentinel (None) is still rejected per-value so it
    # cannot be filtered away as if it were a legitimate class.
    y_true = np.empty(2, dtype=object)
    y_true[:] = [0, None]
    y_pred = np.array([0, 1])

    with pytest.raises(ValidationError, match="missing values"):
        confusion_matrix(y_true, y_pred, labels=[0, 1])


def test_scores_reject_invalid_average() -> None:
    with pytest.raises(ValidationError, match="average"):
        f1_score(np.array([0]), np.array([0]), labels=[0], average="binary")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (None, "missing values"),
        (float("nan"), "finite labels"),
        (float("inf"), "finite labels"),
        (1 + 2j, "real-valued labels"),
        (np.complex64(1 + 2j), "real-valued labels"),
        (np.datetime64("NaT"), "missing values"),
    ],
)
def test_object_dtype_labels_reject_invalid_sentinels(value, message) -> None:
    # validate_labels only inspects complex/finite/NaN for numeric dtypes, so an
    # object array could otherwise smuggle these sentinels in as a silent new
    # class. Per-value validation must reject them regardless of dtype.
    y_true = np.empty(1, dtype=object)
    y_true[:] = [value]

    with pytest.raises(ValidationError, match=message):
        confusion_matrix(y_true, y_true)


def test_object_dtype_rejects_invalid_sentinel() -> None:
    y_true = np.array([0, 1])
    y_pred = np.array([0, 1])

    with pytest.raises(ValidationError, match="missing values"):
        confusion_matrix(y_true, y_pred, labels=[0, None, 1])


def test_tuple_labels_score_in_requested_order() -> None:
    labels = [("left", 1), ("right", 2)]
    y_true = np.empty(4, dtype=object)
    y_true[:] = [("left", 1), ("right", 2), ("left", 1), ("right", 2)]
    y_pred = np.empty(4, dtype=object)
    y_pred[:] = [("left", 1), ("left", 1), ("left", 1), ("right", 2)]
    # matrix over [left, right]: row left = [2, 0], row right = [1, 1].
    assert np.array_equal(confusion_matrix(y_true, y_pred, labels=labels), [[2, 0], [1, 1]])
    # precision: left = 2/3, right = 1/1; recall: left = 2/2, right = 1/2.
    assert np.allclose(precision_score(y_true, y_pred, labels=labels), [2 / 3, 1.0])
    assert np.allclose(recall_score(y_true, y_pred, labels=labels), [1.0, 0.5])
    assert np.allclose(f1_score(y_true, y_pred, labels=labels), [0.8, 2 / 3])


def test_object_dtype_accepts_valid_labels_and_scores_them() -> None:
    y_true = np.empty(3, dtype=object)
    y_true[:] = ["x", "y", "x"]
    y_pred = np.empty(3, dtype=object)
    y_pred[:] = ["x", "x", "x"]
    # Inferred order [x, y]: row x = [2, 0], row y = [1, 0].
    assert np.array_equal(confusion_matrix(y_true, y_pred), [[2, 0], [1, 0]])
    # precision x = 2/3, y = 0/0 -> 0.0; recall x = 2/2, y = 0/1 -> 0.0.
    assert np.allclose(precision_score(y_true, y_pred), [2 / 3, 0.0])
    assert np.allclose(recall_score(y_true, y_pred), [1.0, 0.0])


def test_single_class_all_correct_yields_perfect_scores() -> None:
    y_true = np.array([5, 5, 5])
    y_pred = np.array([5, 5, 5])

    assert np.array_equal(confusion_matrix(y_true, y_pred), [[3]])
    assert accuracy_score(y_true, y_pred) == 1.0
    assert np.allclose(precision_score(y_true, y_pred), [1.0])
    assert np.allclose(recall_score(y_true, y_pred), [1.0])
    assert np.allclose(f1_score(y_true, y_pred), [1.0])
    for average in ("micro", "macro", "weighted"):
        assert f1_score(y_true, y_pred, average=average) == 1.0


def test_prediction_only_class_has_zero_precision_and_recall() -> None:
    y_true = np.array([0, 0])
    y_pred = np.array([0, 1])
    labels = [0, 1]
    # Class 1 appears only in predictions: zero true support -> recall 0.0, and
    # it is never correct -> precision 0.0. Class 0 retains its counts.
    assert np.array_equal(confusion_matrix(y_true, y_pred, labels=labels), [[1, 1], [0, 0]])
    assert np.allclose(precision_score(y_true, y_pred, labels=labels), [1.0, 0.0])
    assert np.allclose(recall_score(y_true, y_pred, labels=labels), [0.5, 0.0])
    assert np.allclose(f1_score(y_true, y_pred, labels=labels), [2 / 3, 0.0])


def test_subset_filter_and_absent_class_combined() -> None:
    # The two explicit-labels strategies in one call: subset filtering drops the
    # (2, 2) sample because 2 is outside labels, and "absent" is declared but
    # appears in neither target -> its row and column stay all zero.
    y_true = np.array([0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 1])
    labels = [0, 1, "absent"]

    assert np.array_equal(
        confusion_matrix(y_true, y_pred, labels=labels),
        [[1, 1, 0], [0, 1, 0], [0, 0, 0]],
    )


def test_mixed_type_labels_follow_explicit_order() -> None:
    y_true = np.array([0, "a", 1, "a"], dtype=object)
    y_pred = np.array([0, "a", "a", 1], dtype=object)
    labels = ["a", 1, 0]  # explicit mixed-type order, deliberately unsorted

    assert np.array_equal(
        confusion_matrix(y_true, y_pred, labels=labels),
        [[1, 1, 0], [1, 0, 0], [0, 0, 1]],
    )
    # Per-class scores follow the [a, 1, 0] order, not numeric or lexical order.
    assert np.allclose(precision_score(y_true, y_pred, labels=labels), [0.5, 0.0, 1.0])
    assert np.allclose(recall_score(y_true, y_pred, labels=labels), [0.5, 0.0, 1.0])
    assert np.allclose(f1_score(y_true, y_pred, labels=labels), [0.5, 0.0, 1.0])


def test_selected_label_recall_counts_predictions_outside_subset() -> None:
    # y_true=0, y_pred=2 is a false negative for class 0: the "rest" must include
    # label 2 even though 2 is outside labels=[0]. recall_0 = 1 / (1 + 1) = 0.5.
    y_true = np.array([0, 0])
    y_pred = np.array([0, 2])

    assert np.allclose(recall_score(y_true, y_pred, labels=[0]), [0.5])
    assert np.allclose(precision_score(y_true, y_pred, labels=[0]), [1.0])
    assert np.allclose(f1_score(y_true, y_pred, labels=[0]), [2 / 3])


def test_selected_label_precision_counts_truth_outside_subset() -> None:
    # y_true=2, y_pred=0 is a false positive for class 0: precision_0 = 1 / 2.
    y_true = np.array([0, 2])
    y_pred = np.array([0, 0])

    assert np.allclose(precision_score(y_true, y_pred, labels=[0]), [0.5])
    assert np.allclose(recall_score(y_true, y_pred, labels=[0]), [1.0])
    assert np.allclose(f1_score(y_true, y_pred, labels=[0]), [2 / 3])


def test_selected_label_f1_counts_cross_subset_errors() -> None:
    # Both error types at once: TP=1, FP=1, FN=1 -> precision=recall=f1=0.5.
    y_true = np.array([0, 0, 2])
    y_pred = np.array([0, 2, 0])

    assert np.allclose(f1_score(y_true, y_pred, labels=[0]), [0.5])
    assert np.allclose(precision_score(y_true, y_pred, labels=[0]), [0.5])
    assert np.allclose(recall_score(y_true, y_pred, labels=[0]), [0.5])


def test_selected_labels_count_cross_subset_errors() -> None:
    y_true = np.array([0, 0, 2, 1])
    y_pred = np.array([0, 2, 0, 1])
    labels = [0, 1]
    # Full union [0, 1, 2]; selected [0, 1].
    #   tp = [1, 1]; predicted (col sums over all rows) = [2, 1];
    #   support (row sums over all cols) = [2, 1].
    #   precision = [0.5, 1.0]; recall = [0.5, 1.0]; f1 = [0.5, 1.0].
    assert np.allclose(precision_score(y_true, y_pred, labels=labels), [0.5, 1.0])
    assert np.allclose(recall_score(y_true, y_pred, labels=labels), [0.5, 1.0])
    assert np.allclose(f1_score(y_true, y_pred, labels=labels), [0.5, 1.0])
    # micro: total_tp=2, total_predicted=3, total_support=3 -> 2/3 each.
    assert precision_score(y_true, y_pred, labels=labels, average="micro") == pytest.approx(2 / 3)
    assert recall_score(y_true, y_pred, labels=labels, average="micro") == pytest.approx(2 / 3)
    assert f1_score(y_true, y_pred, labels=labels, average="micro") == pytest.approx(2 / 3)
    # macro: mean([0.5, 1.0]) = 0.75.
    assert precision_score(y_true, y_pred, labels=labels, average="macro") == 0.75
    # weighted (support [2, 1], total 3): (0.5*2 + 1.0*1) / 3 = 2/3.
    assert f1_score(y_true, y_pred, labels=labels, average="weighted") == pytest.approx(2 / 3)


def test_absent_selected_class_still_counts_errors() -> None:
    y_true = np.array([0, 0])
    y_pred = np.array([0, 2])
    labels = [0, "absent"]
    # Class 0 keeps its FN from (0, 2); "absent" is declared but unobserved, so
    # it stays zero throughout while still occupying an output slot.
    assert np.allclose(precision_score(y_true, y_pred, labels=labels), [1.0, 0.0])
    assert np.allclose(recall_score(y_true, y_pred, labels=labels), [0.5, 0.0])
    assert np.allclose(f1_score(y_true, y_pred, labels=labels), [2 / 3, 0.0])


def test_evaluation_functions_are_exported_from_models_top_level() -> None:
    import neurale.models as models

    assert models.accuracy_score is accuracy_score
    assert models.confusion_matrix is confusion_matrix
    assert models.f1_score is f1_score
    assert models.precision_score is precision_score
    assert models.recall_score is recall_score


def test_randomized_parity_against_reference_helper() -> None:
    rng = np.random.default_rng(20260727)
    domain = ["a", "b", "c", 0, 1, 2]

    for _ in range(300):
        n = int(rng.integers(1, 30))
        y_true = np.array(
            [domain[int(i)] for i in rng.integers(0, len(domain), size=n)], dtype=object
        )
        y_pred = np.array(
            [domain[int(i)] for i in rng.integers(0, len(domain), size=n)], dtype=object
        )
        strategy = int(rng.integers(0, 3))
        if strategy == 0:
            labels = None
        elif strategy == 1:
            labels = list(domain)
        else:
            k = int(rng.integers(1, len(domain) + 1))
            chosen = rng.choice(len(domain), size=k, replace=False)
            labels = [domain[int(i)] for i in chosen]

        expected_matrix, _ = _reference_confusion(list(y_true), list(y_pred), labels)
        assert np.array_equal(
            confusion_matrix(y_true, y_pred, labels=labels),
            np.array(expected_matrix, dtype=np.int64),
        )
        for metric, func in [
            ("precision", precision_score),
            ("recall", recall_score),
            ("f1", f1_score),
        ]:
            for average in (None, "micro", "macro", "weighted"):
                expected = _reference_score(list(y_true), list(y_pred), labels, average, metric)
                actual = func(y_true, y_pred, labels=labels, average=average)
                if average is None:
                    assert np.allclose(actual, expected), (metric, average, labels, y_true, y_pred)
                else:
                    assert actual == pytest.approx(expected), (
                        metric,
                        average,
                        labels,
                        y_true,
                        y_pred,
                    )


def test_no_top_level_metrics_package_is_created() -> None:
    assert importlib.util.find_spec("neurale.metrics") is None
