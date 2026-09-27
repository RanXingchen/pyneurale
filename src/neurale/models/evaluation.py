#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""General classification evaluation independent of model implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from neurale._validation import validate_labels
from neurale.exceptions import ValidationError

Average = Literal["micro", "macro", "weighted"] | None


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[object] | np.ndarray | None = None,
) -> np.ndarray:
    """Return counts with true labels in rows and predicted labels in columns.

    Targets are single-label, 1D arrays; multilabel-indicator
    inputs are not supported. Labels are treated as opaque hashable scalars and
    are never sorted.

    When ``labels`` is ``None`` the label set is inferred in stable
    first-occurrence order — first from ``y_true``, then unseen labels from
    ``y_pred`` — and every sample is counted.

    When ``labels`` is supplied it selects a **subset**, not a strict superset:
    only samples whose true *and* predicted label both appear in ``labels`` are
    counted, and any other sample is silently dropped. ``labels`` also fixes the
    row and column order and may declare classes absent from both targets (their
    rows and columns stay all zero). ``labels`` itself may not contain missing
    values, non-finite floats, complex values, or duplicates.

    Parameters
    ----------
    y_true : numpy.ndarray
        One-dimensional array of ground-truth labels.
    y_pred : numpy.ndarray
        One-dimensional array of predicted labels, same length as ``y_true``.
    labels : sequence or numpy.ndarray or None, optional
        Class subset and row/column order. ``None`` infers the full label set
        from the targets.

    Returns
    -------
    numpy.ndarray
        Integer matrix of shape ``(n_labels, n_labels)`` with true labels in
        rows and predicted labels in columns, ordered per ``labels``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the targets are not 1D, differ in length, contain
        non-hashable / missing / non-finite / complex labels, ``labels`` is
        empty or duplicated, or labels cannot be inferred from empty targets.

    Notes
    -----
    An empty target with explicit ``labels`` returns a zero matrix. An empty
    target without ``labels`` raises, since the label set cannot be inferred.

    The matrix is clipped to the selected rows and columns. The one-vs-rest
    scores (:func:`precision_score`, :func:`recall_score`, :func:`f1_score`) do
    **not** share this clipping: cross-subset false positives and false
    negatives remain counted, so they are not derived from this matrix when
    ``labels`` is a proper subset.
    """
    true_indices, predicted_indices, n_labels = _encode_targets(y_true, y_pred, labels)
    matrix = np.zeros((n_labels, n_labels), dtype=np.int64)
    np.add.at(matrix, (true_indices, predicted_indices), 1)
    return matrix


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return the fraction of predictions equal to their targets.

    Accuracy is computed over the whole input — ``accuracy_score`` takes no
    ``labels`` argument and does not select a subset. Targets are single-label,
    1D arrays; multilabel-indicator inputs are not supported.

    Parameters
    ----------
    y_true : numpy.ndarray
        One-dimensional array of ground-truth labels.
    y_pred : numpy.ndarray
        One-dimensional array of predicted labels, same length as ``y_true``.

    Returns
    -------
    float
        Fraction of samples whose prediction equals its target, in ``[0, 1]``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the targets are empty, not 1D, differ in length, or
        contain non-hashable / missing / non-finite / complex labels.
    """
    true_values, predicted_values = _validate_targets(y_true, y_pred)
    if true_values.size == 0:
        raise ValidationError("y_true and y_pred must not be empty.")

    true_indices, predicted_indices, _ = _encode_validated_targets(
        true_values,
        predicted_values,
        labels=None,
    )
    return float(np.count_nonzero(true_indices == predicted_indices) / true_indices.size)


def precision_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[object] | np.ndarray | None = None,
    average: Average = None,
) -> np.ndarray | float:
    """Return precision per class or with the requested aggregation.

    Precision ``tp / (tp + fp)`` is a one-vs-rest metric: for each selected
    label the "rest" includes every label outside the selected set, so samples
    crossing the subset boundary count as false positives or false negatives
    rather than being dropped. ``labels`` selects which classes are reported and
    their order (``None`` infers the full set from the targets); it does **not**
    clip the counts the way :func:`confusion_matrix` does. Targets are
    single-label, 1D arrays; multilabel-indicator inputs are not
    supported.

    Parameters
    ----------
    y_true : numpy.ndarray
        One-dimensional array of ground-truth labels.
    y_pred : numpy.ndarray
        One-dimensional array of predicted labels, same length as ``y_true``.
    labels : sequence or numpy.ndarray or None, optional
        Selected classes and per-class output order. ``None`` infers the full
        label set from the targets.
    average : {None, "micro", "macro", "weighted"}, optional
        Aggregation mode:

        * ``None`` — per-class array in ``labels`` order (or inferred order).
        * ``"micro"`` — aggregate counts across classes into a single scalar.
        * ``"macro"`` — unweighted mean of the per-class scores.
        * ``"weighted"`` — support-weighted mean (support = per-class true
          count).

    Returns
    -------
    numpy.ndarray or float
        Per-class precision when ``average is None`` (shape ``(n_labels,)``),
        otherwise a single float.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the targets are not 1D, differ in length, contain
        non-hashable / missing / non-finite / complex labels, ``labels`` is
        empty or duplicated, labels cannot be inferred from empty targets, or
        ``average`` is not a supported value.

    Notes
    -----
    A class with no predicted samples has zero denominator; its precision is
    defined as ``0`` rather than raising. With explicit ``labels`` and empty
    targets the per-class result is an all-zero array and each aggregate is
    ``0.0``.
    """
    precision, _, _ = _precision_recall_f1(y_true, y_pred, labels=labels, average=average)
    return precision


def recall_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[object] | np.ndarray | None = None,
    average: Average = None,
) -> np.ndarray | float:
    """Return recall per class or with the requested aggregation.

    Recall ``tp / (tp + fn)`` is a one-vs-rest metric: for each selected label
    the "rest" includes every label outside the selected set, so samples
    crossing the subset boundary count as false positives or false negatives
    rather than being dropped. ``labels`` selects which classes are reported and
    their order (``None`` infers the full set from the targets); it does **not**
    clip the counts the way :func:`confusion_matrix` does. Targets are
    single-label, 1D arrays; multilabel-indicator inputs are not
    supported.

    Parameters
    ----------
    y_true : numpy.ndarray
        One-dimensional array of ground-truth labels.
    y_pred : numpy.ndarray
        One-dimensional array of predicted labels, same length as ``y_true``.
    labels : sequence or numpy.ndarray or None, optional
        Selected classes and per-class output order. ``None`` infers the full
        label set from the targets.
    average : {None, "micro", "macro", "weighted"}, optional
        Aggregation mode:

        * ``None`` — per-class array in ``labels`` order (or inferred order).
        * ``"micro"`` — aggregate counts across classes into a single scalar.
        * ``"macro"`` — unweighted mean of the per-class scores.
        * ``"weighted"`` — support-weighted mean (support = per-class true
          count).

    Returns
    -------
    numpy.ndarray or float
        Per-class recall when ``average is None`` (shape ``(n_labels,)``),
        otherwise a single float.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the targets are not 1D, differ in length, contain
        non-hashable / missing / non-finite / complex labels, ``labels`` is
        empty or duplicated, labels cannot be inferred from empty targets, or
        ``average`` is not a supported value.

    Notes
    -----
    A class with no true samples has zero denominator; its recall is defined as
    ``0`` rather than raising. With explicit ``labels`` and empty targets the
    per-class result is an all-zero array and each aggregate is ``0.0``.
    """
    _, recall, _ = _precision_recall_f1(y_true, y_pred, labels=labels, average=average)
    return recall


def f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[object] | np.ndarray | None = None,
    average: Average = None,
) -> np.ndarray | float:
    """Return the F1 score per class or with the requested aggregation.

    F1 ``2 * precision * recall / (precision + recall)`` is derived from the
    one-vs-rest precision and recall computed by :func:`precision_score` /
    :func:`recall_score`; for each selected label the "rest" includes every
    label outside the selected set, so samples crossing the subset boundary
    count as false positives or false negatives rather than being dropped.
    ``labels`` selects which classes are reported and their order (``None``
    infers the full set from the targets); it does **not** clip the counts the
    way :func:`confusion_matrix` does. Targets are single-label, 1D
    arrays; multilabel-indicator inputs are not supported.

    Parameters
    ----------
    y_true : numpy.ndarray
        One-dimensional array of ground-truth labels.
    y_pred : numpy.ndarray
        One-dimensional array of predicted labels, same length as ``y_true``.
    labels : sequence or numpy.ndarray or None, optional
        Selected classes and per-class output order. ``None`` infers the full
        label set from the targets.
    average : {None, "micro", "macro", "weighted"}, optional
        Aggregation mode:

        * ``None`` — per-class array in ``labels`` order (or inferred order).
        * ``"micro"`` — aggregate counts across classes into a single scalar.
        * ``"macro"`` — unweighted mean of the per-class scores.
        * ``"weighted"`` — support-weighted mean (support = per-class true
          count).

    Returns
    -------
    numpy.ndarray or float
        Per-class F1 when ``average is None`` (shape ``(n_labels,)``),
        otherwise a single float.

    Raises
    ------
    neurale.exceptions.ValidationError
        If the targets are not 1D, differ in length, contain
        non-hashable / missing / non-finite / complex labels, ``labels`` is
        empty or duplicated, labels cannot be inferred from empty targets, or
        ``average`` is not a supported value.

    Notes
    -----
    A class whose precision plus recall is zero has zero denominator; its F1 is
    defined as ``0`` rather than raising. With explicit ``labels`` and empty
    targets the per-class result is an all-zero array and each aggregate is
    ``0.0``.
    """
    _, _, f1 = _precision_recall_f1(y_true, y_pred, labels=labels, average=average)
    return f1


def _precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    labels: Sequence[object] | np.ndarray | None,
    average: Average,
) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float]:
    if average not in (None, "micro", "macro", "weighted"):
        raise ValidationError("average must be None, 'micro', 'macro', or 'weighted'.")

    true_positive, predicted, support = _selected_class_counts(y_true, y_pred, labels)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    den = precision + recall
    f1 = np.divide(
        2.0 * precision * recall,
        den,
        out=np.zeros_like(den),
        where=den != 0,
    )

    if average is None:
        return precision, recall, f1
    if average == "micro":
        total_true_positive = float(np.sum(true_positive))
        total_predicted = float(np.sum(predicted))
        total_support = float(np.sum(support))
        micro_precision = _safe_ratio(total_true_positive, total_predicted)
        micro_recall = _safe_ratio(total_true_positive, total_support)
        return (
            micro_precision,
            micro_recall,
            _safe_ratio(2.0 * micro_precision * micro_recall, micro_precision + micro_recall),
        )
    if average == "macro":
        return float(np.mean(precision)), float(np.mean(recall)), float(np.mean(f1))

    total_support = float(np.sum(support))
    if total_support == 0.0:
        return 0.0, 0.0, 0.0
    weights = support / total_support
    return (
        float(np.sum(precision * weights)),
        float(np.sum(recall * weights)),
        float(np.sum(f1 * weights)),
    )


def _selected_class_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[object] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Precision/recall/F1 are one-vs-rest: for a selected label the "rest" must
    # include labels outside the selected set, so a sample predicted as a
    # selected label but truly outside (a false positive) or truly a selected
    # label but predicted outside (a false negative) still counts. Build the
    # confusion matrix over the FULL union of declared and observed labels, then
    # extract per-selected-label ``tp`` (diagonal), ``predicted`` (column sum
    # over all true classes), and ``support`` (row sum over all predicted
    # classes). This deliberately differs from :func:`confusion_matrix`, which
    # clips to the selected rows and columns; reusing the clipped matrix would
    # drop cross-subset false positives/negatives and inflate the scores.
    true_values, predicted_values = _validate_targets(y_true, y_pred)
    if labels is None:
        true_indices, predicted_indices, full_size = _encode_validated_targets(
            true_values,
            predicted_values,
            labels=None,
        )
        selected_indices = np.arange(full_size, dtype=np.intp)
    else:
        declared = _validate_ordered_labels(labels)
        idx = _make_label_index(declared)
        for values in (true_values, predicted_values):
            for value in values:
                _validate_label(value)
                if value not in idx:
                    idx[value] = len(idx)
        selected_indices = np.arange(len(declared), dtype=np.intp)
        true_indices = _encode_labels(true_values, idx, "y_true", allow_missing=False)
        predicted_indices = _encode_labels(predicted_values, idx, "y_pred", allow_missing=False)
        full_size = len(idx)
    full_matrix = np.zeros((full_size, full_size), dtype=np.int64)
    np.add.at(full_matrix, (true_indices, predicted_indices), 1)
    true_positive = full_matrix[selected_indices, selected_indices].astype(np.float64)
    predicted = full_matrix[:, selected_indices].sum(axis=0).astype(np.float64)
    support = full_matrix[selected_indices, :].sum(axis=1).astype(np.float64)
    return true_positive, predicted, support


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if denominator == 0.0 else numerator / denominator


def _encode_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[object] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    true_values, predicted_values = _validate_targets(y_true, y_pred)
    return _encode_validated_targets(true_values, predicted_values, labels)


def _validate_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    true_values = validate_labels(y_true, "y_true", ndim=1)
    predicted_values = validate_labels(y_pred, "y_pred", ndim=1)
    if true_values.size != predicted_values.size:
        raise ValidationError(
            "y_true and y_pred must contain the same number of samples; "
            f"got {true_values.size} and {predicted_values.size}."
        )
    return true_values, predicted_values


def _encode_validated_targets(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[object] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    ordered_labels = _validate_ordered_labels(labels) if labels is not None else []
    label_to_idx = _make_label_index(ordered_labels)

    if labels is None:
        for values in (y_true, y_pred):
            for value in values:
                _append_unseen_label(value, ordered_labels, label_to_idx)
        if not ordered_labels:
            raise ValidationError("labels cannot be inferred from empty targets.")
        true_indices = _encode_labels(y_true, label_to_idx, "y_true", allow_missing=False)
        predicted_indices = _encode_labels(y_pred, label_to_idx, "y_pred", allow_missing=False)
        return true_indices, predicted_indices, len(ordered_labels)

    # Explicit ``labels`` selects a subset, not a strict superset: a sample is
    # counted only when both its true and predicted label fall inside the
    # declared set; any other sample is dropped (not counted). Invalid sentinels
    # (None/NaN/complex/NaT) are still rejected per-value via ``_validate_label``
    # so they cannot be silently filtered away — only valid-but-unselected
    # labels are dropped, by the subset contract.
    true_indices = _encode_labels(y_true, label_to_idx, "y_true", allow_missing=True)
    predicted_indices = _encode_labels(y_pred, label_to_idx, "y_pred", allow_missing=True)
    keep = (true_indices >= 0) & (predicted_indices >= 0)
    return true_indices[keep], predicted_indices[keep], len(ordered_labels)


def _validate_ordered_labels(labels: Sequence[object] | np.ndarray) -> list[object]:
    # ``np.asarray`` would collapse a sequence of equal-length tuples (e.g.
    # ``[("left", 1), ("right", 2)]``) into a 2D object array and reject it as
    # non-1D, even though the inferred path accepts the very same labels when they
    # arrive as a 1D object array. Materialize the sequence directly so each
    # element is treated as one scalar label, matching the inference path; only a
    # genuine ndarray is dimension-checked structurally.
    if isinstance(labels, (str, bytes)):
        raise ValidationError("labels must be a 1D sequence of labels.")
    if isinstance(labels, np.ndarray):
        if labels.ndim != 1:
            raise ValidationError("labels must be a 1D sequence of labels.")
        ordered = list(labels)
    else:
        try:
            ordered = list(labels)
        except TypeError as exc:
            raise ValidationError("labels must be a 1D sequence of labels.") from exc
    if not ordered:
        raise ValidationError("labels must contain at least one label.")
    return ordered


def _make_label_index(labels: list[object]) -> dict[object, int]:
    label_to_idx: dict[object, int] = {}
    for i, label in enumerate(labels):
        _validate_label(label)
        try:
            duplicate = label in label_to_idx
        except (TypeError, ValueError) as exc:
            raise ValidationError("labels must contain hashable scalar values.") from exc
        if duplicate:
            raise ValidationError("labels must not contain duplicates.")
        label_to_idx[label] = i
    return label_to_idx


def _append_unseen_label(
    label: object,
    labels: list[object],
    label_to_idx: dict[object, int],
) -> None:
    _validate_label(label)
    if label not in label_to_idx:
        label_to_idx[label] = len(labels)
        labels.append(label)


def _validate_label(label: object) -> None:
    # dtype-independent per-value checks. ``validate_labels`` only inspects
    # complex/finite/NaN for numeric dtypes, so an object array could otherwise
    # smuggle in None, Inf, NaN, complex, or NaT as a silent new class. Reject the
    # same sentinels here so object-dtype and numeric-dtype label domains behave
    # identically, then fall back to the hashability/reflexivity contract for
    # everything else (e.g. unhashable arrays, non-reflexive NaN-likes).
    if label is None:
        raise ValidationError("labels must not contain missing values.")
    if isinstance(label, (complex, np.complexfloating)):
        raise ValidationError("labels must contain real-valued labels.")
    if isinstance(label, (float, np.floating)) and not np.isfinite(label):
        raise ValidationError("labels must contain only finite labels.")
    if isinstance(label, np.datetime64) and np.isnat(label):
        raise ValidationError("labels must not contain missing values.")
    try:
        hash(label)
        reflexive = label == label
        valid = bool(reflexive)
    except (TypeError, ValueError) as exc:
        raise ValidationError("labels must contain hashable scalar values.") from exc
    if not valid:
        raise ValidationError("labels must not contain missing values.")


def _encode_labels(
    values: np.ndarray,
    label_to_idx: dict[object, int],
    name: str,
    *,
    allow_missing: bool,
) -> np.ndarray:
    encoded = np.empty(values.size, dtype=np.intp)
    for i, value in enumerate(values):
        if allow_missing:
            # Subset mode: reject invalid sentinels, then map unselected (but
            # valid) labels to -1 so the caller can drop the sample.
            _validate_label(value)
            encoded[i] = label_to_idx.get(value, -1)
        else:
            try:
                encoded[i] = label_to_idx[value]
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"{name} contains a label not present in labels: {value!r}."
                ) from exc
    return encoded


__all__ = [
    "accuracy_score",
    "confusion_matrix",
    "f1_score",
    "precision_score",
    "recall_score",
]
