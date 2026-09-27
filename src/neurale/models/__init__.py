#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Public model and statistics API for PyNeurale."""

from .alignment import DTWResult, dtw
from .classification import LDA
from .decomposition import PCA
from .density import GaussianKDE
from .evaluation import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from .linear_model import LinearRegression, Ridge
from .manifold import LPP
from .neighbors import knn
from .preprocessing import MinMaxScaler, StandardScaler, TemporalContext
from .state_space import (
    FilterResult,
    KalmanFilter,
    LinearGaussianParameters,
    LinearGaussianStateSpace,
)
from .statistics import (
    DipStatistic,
    DipTestResult,
    LedoitWolfCov,
    dip_statistic,
    dip_test,
    empirical_cov,
    rayleigh_test,
)

__all__ = [
    "LDA",
    "LPP",
    "PCA",
    "DTWResult",
    "DipStatistic",
    "DipTestResult",
    "FilterResult",
    "GaussianKDE",
    "KalmanFilter",
    "LedoitWolfCov",
    "LinearGaussianParameters",
    "LinearGaussianStateSpace",
    "LinearRegression",
    "MinMaxScaler",
    "Ridge",
    "StandardScaler",
    "TemporalContext",
    "accuracy_score",
    "confusion_matrix",
    "dip_statistic",
    "dip_test",
    "dtw",
    "empirical_cov",
    "f1_score",
    "knn",
    "precision_score",
    "rayleigh_test",
    "recall_score",
]
