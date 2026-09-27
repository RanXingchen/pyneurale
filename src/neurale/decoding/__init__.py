#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Classical decoders and the typed contract they share.

The contract defines what a decoder consumes and produces, independently of how
any particular one works. A decoder takes prepared
:class:`~neurale.data.arrays.FeatureMatrix` frames and an already aligned
target, and returns a typed, metadata-complete result: a
:class:`~neurale.data.arrays.SignalArray` for continuous decoding, a
:class:`ClassificationPrediction` for classification.

Two boundaries are deliberate. A decoder *verifies* alignment and never
establishes it -- no trial slicing, feature extraction, resampling, file
loading, feature-subset search, or statistics adaptation happens inside one,
and clock conversion stays with :func:`neurale.data.convert_time`. A decoder
may consume a subset of its features, but only the one the caller names: which
features matter is never something a decoder decides. And a fitted
decoder is described by a :class:`FeatureSchema`, whose
:attr:`~FeatureSchema.fingerprint` is stable across processes, so predicting
from features the decoder was not fitted for fails before the model runs.

Both boundaries hold at prediction time, not only at fit time. A fitted
decoder belongs to one timeline, and a prediction input that declares a
different clock is rejected before the model runs; every result is stamped
with the clock of the timestamps it actually carries, so a result's ``time``
and ``clock`` never describe different domains.

A concrete decoder is a lifecycle around estimators that live in
:mod:`neurale.models`; it adds no arithmetic of its own. :class:`KalmanDecoder`
composes an owned feature scaler with :mod:`neurale.models.state_space`;
:class:`LinearDecoder` and :class:`RidgeDecoder` compose an owned scaler and
temporal context with :mod:`neurale.models.linear_model`; :class:`LDADecoder`
composes the same two stages with :class:`neurale.models.LDA`.

Those stages are the *only* preprocessing a decoder owns, and both are objects
the caller hands it. Dimensionality reduction and feature subsets are composed
outside, on the feature matrix: fit a :class:`~neurale.models.PCA` or select
columns first, build a :class:`~neurale.data.arrays.FeatureMatrix` from the
result, and fit the decoder on that. The transform stays visible, and the
fitted :class:`FeatureSchema` describes what the decoder actually consumed.

:mod:`neurale.decoding.sequence` is a different kind of decoding and keeps its
own namespace. It turns *per-step token scores* into token sequences under a
counted language model, so it runs after a classifier rather than beside one:
there is no feature schema, no timeline, and no device, and its classes are
deliberately not exported here beside the decoders that have all three.

:mod:`neurale.decoding.torch` is the optional Torch boundary and currently
holds no model at all -- only the helpers a future Torch-backed decoder will
ask for the framework with. It is a separate namespace so that installing a
deep-learning framework is never a condition of using the decoders here.

Importing this package pulls in :mod:`neurale.models`, which is pure Python at
import time, and no native extension, CUDA, Torch, I/O, or recording code. It
does not import or even *probe* for Torch, which is why the boundary above is
something a caller reaches for rather than something this import decides. A
decoder resolves the native kernels of the model it composes when a fit runs,
not when the module loads.
"""

from .base import BaseDecoder, ClassificationDecoder, ContinuousDecoder, FitSegment
from .kalman import KalmanDecoder
from .lda import LDADecoder
from .linear import LinearDecoder, RidgeDecoder
from .results import ClassificationPrediction
from .schema import ContinuousTargetSchema, FeatureSchema
from .targets import ClassificationTarget

__all__ = [
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
