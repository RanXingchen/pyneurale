#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Classification decoding by linear discriminant analysis."""

from __future__ import annotations

import numpy as np

from neurale.data import FeatureMatrix
from neurale.exceptions import ValidationError

# The shrinkage grammar belongs to the estimator that consumes it. Reusing its
# normalizer keeps a decoder from accepting -- or rejecting -- something the
# model would not, which a second copy of the rule here eventually would.
from neurale.models.classification import LDA, _normalize_shrinkage
from neurale.models.preprocessing import TemporalContext

from ._stages import FeatureScaler, FeatureStages, validate_stages
from ._validation import _hashable, readonly
from .base import ClassificationDecoder
from .results import ClassificationPrediction
from .targets import ClassificationTarget


class LDADecoder(ClassificationDecoder):
    """Decode discrete classes with linear discriminant analysis.

    The decoder composes owned pieces and adds no arithmetic of its own: an
    optional feature :attr:`scaler`, an optional temporal :attr:`context`, and
    a :class:`~neurale.models.classification.LDA` over the result. Labels come
    from the model's own decision rule, so a decoded label is the model's
    answer and not a rule layered on top of it.

    The class order is the one the fit target declared, and it is the column
    order of every :attr:`~ClassificationPrediction.probabilities` matrix the
    decoder produces. The model orders its own columns by sorted class value;
    a fit records the permutation onto the declared order once, so a caller
    reading column ``i`` always reads class ``classes_[i]`` however the target
    happened to be spelled.

    Prediction is stateless, and :meth:`~BaseDecoder.reset` is a no-op by
    inheritance.

    Parameters
    ----------
    scaler : neurale.models.preprocessing.StandardScaler or \
neurale.models.preprocessing.MinMaxScaler or None, optional
        Feature scaler to own. The decoder fits a *copy*, so the instance
        passed here stays a configuration; the fitted copy is :attr:`scaler_`,
        published frozen. A scaler that is already frozen is a published fit
        rather than a configuration, and is rejected.
    context : neurale.models.preprocessing.TemporalContext or None, optional
        Temporal context to stack before classifying. It drops the first
        ``left`` and last ``right`` frames, and the fit labels and every
        predicted timestamp lose exactly those same frames.
    shrinkage : None or {"auto"} or float, optional
        Covariance shrinkage passed to the model: ``None`` for none,
        ``"auto"`` for the Ledoit-Wolf estimate, or an explicit weight in
        ``[0, 1]``.

    Raises
    ------
    neurale.exceptions.ValidationError
        If any argument is invalid.

    Notes
    -----
    Nothing selects features here. There is no Kruskal-Wallis or ANOVA
    ranking, no greedy forward search, and no PCA stage the decoder inserts on
    its own: a caller who wants any of those applies them to the
    :class:`~neurale.data.arrays.FeatureMatrix` first and fits on the result,
    where the transform is visible and the fitted schema describes what the
    decoder actually consumed. Nothing post-processes the output either -- no
    probability smoothing, no acceptance threshold, no debounce across frames.
    Each frame is decoded on its own evidence, and a caller who wants a
    temporal rule applies it to the typed result.

    :attr:`~ClassificationPrediction.scores` is deliberately left empty. For
    three or more classes the solver produces one discriminant per class, but
    for two it produces a single discriminant whose split into two per-class
    values is not identified -- only their difference is. Reporting a
    fabricated split for the binary case, or a column count that changes with
    the number of classes, would both be worse than reporting the posterior
    probabilities, which are well defined either way.

    Examples
    --------
    >>> import numpy as np
    >>> from neurale.data import FeatureMatrix
    >>> from neurale.decoding import ClassificationTarget, LDADecoder
    >>> rng = np.random.default_rng(0)
    >>> rate = 20.0
    >>> labels = np.array(["left", "right"] * 60)
    >>> shift = np.where(labels == "left", -1.0, 1.0)[:, None]
    >>> X = FeatureMatrix(
    ...     data=shift * np.array([[1.0, -0.5, 0.25]]) + rng.normal(scale=0.4, size=(120, 3)),
    ...     fs=rate,
    ...     feature_names=["beta", "gamma", "hfa"],
    ...     unit="uV^2",
    ...     shift=1.0 / rate,
    ... )
    >>> y = ClassificationTarget(labels=labels, time=X.time.copy())
    >>> decoder = LDADecoder().fit(X, y)
    >>> decoded = decoder.predict(X)
    >>> [str(name) for name in decoded.classes], decoded.n_samples
    (['left', 'right'], 120)
    >>> posterior = decoder.predict_proba(X)
    >>> posterior.probabilities.shape
    (120, 2)
    """

    def __init__(
        self,
        *,
        scaler: FeatureScaler | None = None,
        context: TemporalContext | None = None,
        shrinkage: str | float | None = None,
    ) -> None:
        self.scaler, self.context = validate_stages(scaler, context)
        # Validated by the estimator's own normalizer rather than by a second
        # spelling of the same rule here, which could drift out of agreement
        # with what the model will accept. The configured value keeps the
        # caller's spelling; only its validity is checked.
        _normalize_shrinkage(shrinkage)
        self.shrinkage = shrinkage

    # ----------------------------------------------------------------- fitting

    def _validate_fit_arguments(
        self,
        X: FeatureMatrix,
        y: ClassificationTarget,
    ) -> None:
        # Every constructor argument stays a plain public attribute, so what
        # __init__ accepted is not necessarily what this fit will use; the fit
        # that consumes a configuration checks it. Everything here runs before
        # _clear_fitted, so a rejected refit leaves a fitted decoder untouched
        # rather than failing from inside the model with the old fit gone.
        validate_stages(self.scaler, self.context)
        _normalize_shrinkage(self.shrinkage)

        stages = FeatureStages(None, self.context)
        stages.require_frames(X.n_frames, "X")
        labels = stages.trim(np.asarray(y.labels), name="labels")
        _require_every_class_present(labels, np.asarray(y.classes), context=self.context)
        _require_more_rows_than_classes(
            int(labels.shape[0]),
            y.n_classes,
            context=self.context,
        )

    def _fit_decoder(self, X: np.ndarray, y: ClassificationTarget) -> None:
        stages, blocks = FeatureStages.fitted(self.scaler, self.context, [X])
        X = blocks[0]
        labels = stages.trim(np.asarray(y.labels), name="labels")
        classes = np.asarray(y.classes)

        model = LDA(shrinkage=self.shrinkage).fit(X, labels)

        self._stages = stages
        self._model = model
        self._order = _declared_order(classes, model.classes_)
        self._n_samples = int(X.shape[0])

    def _clear_fitted(self) -> None:
        super()._clear_fitted()
        self._stages = None
        self._model = None
        self._order = None
        self._n_samples = None

    # -------------------------------------------------------------- prediction

    def _decode(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        return self._model.predict(self._stages.transform(X)), None, None

    def predict_proba(self, X: FeatureMatrix) -> ClassificationPrediction:
        """Decode classes and their posterior probabilities.

        Parameters
        ----------
        X : neurale.data.arrays.FeatureMatrix
            Prepared feature frames, compatible with the fitted schema.

        Returns
        -------
        ClassificationPrediction
            The same result :meth:`predict` returns -- the same labels, class
            order, timestamps, and clock -- with
            :attr:`~ClassificationPrediction.probabilities` filled in. Column
            ``i`` is the posterior of ``classes_[i]``, and every row sums to
            one.

        Raises
        ------
        neurale.exceptions.ValidationError
            If the decoder is unfitted, or the features do not match the
            fitted schema or timeline.

        Notes
        -----
        Labels follow the model's own decision rule rather than an argmax over
        these columns. The two agree except on an exact probability tie, which
        each breaks in a different class order: the model in its own sorted
        one, an argmax here in the declared one. Reporting the decision rule's
        answer keeps a label and the rule that produced it from disagreeing,
        and the tie stays deterministic either way.
        """

        input_X = X
        X, clock = self._prepare_prediction(input_X)
        X = self._stages.transform(X)
        probabilities = np.ascontiguousarray(self._model.predict_proba(X)[:, self._order])
        return self._classification_result(
            self._model.predict(X),
            input_X,
            clock,
            probabilities=probabilities,
        )

    def _output_time(self, time: np.ndarray) -> np.ndarray:
        return self._stages.trim(np.asarray(time, dtype=np.float64), name="X")

    # ---------------------------------------------------------- fitted results

    @property
    def scaler_(self) -> FeatureScaler | None:
        """Fitted feature scaler, or ``None`` when the features are unscaled.

        The decoder's own frozen copy of the configured :attr:`scaler`; the
        instance handed to the constructor is never fitted or mutated.
        """

        self._check_is_fitted()
        return self._stages.scaler

    @property
    def context_(self) -> TemporalContext | None:
        """Temporal context the fit used, or ``None`` for one frame per prediction."""

        self._check_is_fitted()
        return self._stages.context

    @property
    def n_model_features_(self) -> int:
        """Number of columns the discriminant consumes."""

        self._check_is_fitted()
        return self._stages.n_model_features(self.n_features_in_)

    @property
    def n_samples_(self) -> int:
        """Number of rows the discriminant was fitted on, after any trimming."""

        self._check_is_fitted()
        return self._n_samples

    @property
    def priors_(self) -> np.ndarray:
        """``(n_classes_,)`` fitted class priors, in the declared class order."""

        self._check_is_fitted()
        return readonly(self._model.priors_[self._order])

    @property
    def means_(self) -> np.ndarray:
        """``(n_classes_, n_model_features_)`` class means, in the declared order."""

        self._check_is_fitted()
        return readonly(self._model.means_[self._order])

    @property
    def covariance_(self) -> np.ndarray:
        """``(n_model_features_, n_model_features_)`` shared within-class covariance.

        One matrix for every class, which is what makes the discriminant
        linear, so there is no class order to reconcile here.
        """

        self._check_is_fitted()
        return self._model.covariance_


def _declared_order(classes: np.ndarray, fitted: np.ndarray) -> np.ndarray:
    """Map each declared class onto its column in the model's own order.

    The model orders its columns by sorted class value; a target may declare
    any order it likes. Reconciling the two once at fit time is an indexing
    fact, not arithmetic, and it keeps every later result's columns aligned
    with the class order the caller was promised.
    """

    positions = {_hashable(value): idx for idx, value in enumerate(fitted.tolist())}
    return np.array([positions[_hashable(value)] for value in classes.tolist()], dtype=np.intp)


def _require_every_class_present(
    labels: np.ndarray,
    classes: np.ndarray,
    *,
    context: TemporalContext | None,
) -> None:
    """Raise when the fit rows do not contain every class the target declares.

    A target may declare a class its labels never take, and a temporal context
    can drop the only frames that carried one. Either way the model cannot
    estimate that class, and a decoder that promised one probability column per
    declared class cannot fill it. Refusing beats inventing an empty column.
    """

    present = {_hashable(value) for value in np.unique(labels).tolist()}
    missing = [value for value in classes.tolist() if _hashable(value) not in present]
    if not missing:
        return
    raise ValidationError(
        f"the fitted frames contain no example of every declared class; missing {missing!r}. "
        f"A decoder reports one column per declared class and cannot estimate one it never "
        f"saw.{_context_note(context, ', which may be where they were')}"
    )


def _require_more_rows_than_classes(
    n_rows: int,
    n_classes: int,
    *,
    context: TemporalContext | None,
) -> None:
    """Raise when the fit rows cannot support one mean per class.

    The discriminant estimates a mean per class and one covariance shared
    across them from the same rows, so a fit with no more rows than classes is
    not identified. The model refuses it as well, but from inside its own
    ``fit``, by which point a refitting decoder has already discarded the model
    that was working. A short calibration block, or one a temporal context
    trims below the bound, is exactly the case where that matters.
    """

    if n_rows > n_classes:
        return
    raise ValidationError(
        f"the fit must have more frames than classes; got {n_rows} frames for "
        f"{n_classes} classes.{_context_note(context)}"
    )


def _context_note(context: TemporalContext | None, tail: str = "") -> str:
    """Describe what a temporal context removed, for a message about the fit rows."""

    if context is None:
        return ""
    return (
        f" The temporal context drops the first {context.left} and last "
        f"{context.right} frames{tail}."
    )


__all__ = ["LDADecoder"]
