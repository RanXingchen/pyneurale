#!/usr/bin/env python3

from __future__ import annotations

import os

import pytest

from neurale.exceptions import ConfigurationError
from neurale.runtime import configure_threading, thread_limit


def test_configure_threading_sets_mkl_and_openmp_environment(monkeypatch) -> None:
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_DYNAMIC", raising=False)

    with pytest.warns(RuntimeWarning, match="NumPy is already loaded"):
        info = configure_threading(3)

    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_DYNAMIC"] == "FALSE"
    assert dict(info.environment)["MKL_NUM_THREADS"] == "3"


def test_configure_threading_strict_rejects_late_configuration() -> None:
    with pytest.raises(ConfigurationError, match="NumPy is already loaded"):
        configure_threading(2, strict=True)


def test_thread_limit_restores_environment(monkeypatch) -> None:
    monkeypatch.setenv("MKL_NUM_THREADS", "8")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setenv("MKL_DYNAMIC", "TRUE")

    with pytest.warns(RuntimeWarning):
        with thread_limit(1):
            assert os.environ["MKL_NUM_THREADS"] == "1"
            assert os.environ["OMP_NUM_THREADS"] == "1"

    assert os.environ["MKL_NUM_THREADS"] == "8"
    assert os.environ["OMP_NUM_THREADS"] == "8"
    assert os.environ["MKL_DYNAMIC"] == "TRUE"
