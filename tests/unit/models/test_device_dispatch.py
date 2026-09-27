#!/usr/bin/env python3

from __future__ import annotations

import numpy as np
import pytest

from neurale.exceptions import DeviceUnavailableError
from neurale.models import (
    LDA,
    PCA,
    LedoitWolfCov,
    dip_statistic,
    dip_test,
    dtw,
    empirical_cov,
    rayleigh_test,
)
from neurale.models import _dispatch as dispatch_module
from neurale.runtime import runtime_context


def test_pca_rejects_explicit_cuda_before_native_dispatch() -> None:
    with runtime_context(device="cuda"):
        with pytest.raises(
            DeviceUnavailableError,
            match=r"models\.decomposition.*no CUDA implementation",
        ):
            PCA(n_components=1).fit(np.array([[0.0], [1.0]]))


def test_lda_rejects_explicit_cuda_before_native_dispatch() -> None:
    samples = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([0, 0, 1, 1])

    with runtime_context(device="cuda"):
        with pytest.raises(
            DeviceUnavailableError,
            match=r"models\.classification.*no CUDA implementation",
        ):
            LDA().fit(samples, labels)


def test_dtw_rejects_explicit_cuda_before_native_dispatch() -> None:
    with runtime_context(device="cuda"):
        with pytest.raises(
            DeviceUnavailableError,
            match=r"models\.alignment.*no CUDA implementation",
        ):
            dtw(np.array([0.0, 1.0]), np.array([0.0, 1.0]))


@pytest.mark.parametrize(
    "operation",
    [
        lambda: empirical_cov(np.eye(2)),
        lambda: LedoitWolfCov.shrunk(np.eye(2), 0.5),
        lambda: rayleigh_test(np.array([0.0, 1.0])),
        lambda: dip_statistic(np.arange(4.0)),
        lambda: dip_test(np.arange(4.0), n_boot=1, random_state=0),
    ],
    ids=[
        "empirical-cov",
        "shrunk-covariance",
        "rayleigh-test",
        "dip-statistic",
        "dip-test",
    ],
)
def test_stateless_cpu_only_operations_reject_explicit_cuda(operation) -> None:
    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match="no CUDA implementation"):
            operation()


@pytest.mark.parametrize("requested", ["auto", "cpu"])
def test_stateless_cpu_only_operations_accept_cpu_resolution(requested) -> None:
    with runtime_context(device=requested):
        assert np.allclose(empirical_cov(np.eye(2), center=False), np.eye(2) / 2.0)


def test_ledoit_wolf_rejects_explicit_cuda() -> None:
    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match="no CUDA implementation"):
            LedoitWolfCov().fit(np.eye(2))


def test_cpu_only_guard_rejects_cuda_before_native_namespace(monkeypatch) -> None:
    def fail_namespace(_name):
        raise AssertionError("CPU native namespace must not be loaded")

    monkeypatch.setattr(dispatch_module, "load_native_namespace", fail_namespace)
    with runtime_context(device="cuda"):
        with pytest.raises(DeviceUnavailableError, match="no CUDA implementation"):
            dispatch_module.load_cpu_models_operation("decomposition")
