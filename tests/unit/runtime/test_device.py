#!/usr/bin/env python3

from __future__ import annotations

import pytest

from neurale.exceptions import DeviceUnavailableError
from neurale.runtime import CudaDeviceInfo, CudaInfo, RuntimeConfig
from neurale.runtime import _device as device_module


@pytest.mark.parametrize("requested", ("auto", "cpu"))
def test_cpu_resolution_does_not_probe_cuda(monkeypatch, requested) -> None:
    def fail_probe():
        raise AssertionError("CUDA must not be probed")

    monkeypatch.setattr(device_module, "_probe_cuda", fail_probe)

    resolution = device_module.resolve_device(RuntimeConfig(device=requested))

    assert resolution.requested == requested
    assert resolution.resolved == "cpu"
    assert resolution.cuda is None


def test_cpu_only_operation_rejects_cuda_without_probe(monkeypatch) -> None:
    def fail_probe():
        raise AssertionError("CUDA must not be probed")

    monkeypatch.setattr(device_module, "_probe_cuda", fail_probe)

    with pytest.raises(DeviceUnavailableError, match="no CUDA implementation"):
        device_module.resolve_device(
            RuntimeConfig(device="cuda"),
            operation="models.decomposition",
            supports_cuda=False,
        )


def test_cuda_resolution_preserves_unavailable_reason(monkeypatch) -> None:
    monkeypatch.setattr(
        device_module,
        "_probe_cuda",
        lambda: CudaInfo(
            compiled=True,
            available=False,
            reason="driver initialization failed",
        ),
    )

    with pytest.raises(
        DeviceUnavailableError,
        match="driver initialization failed",
    ):
        device_module.resolve_device(RuntimeConfig(device="cuda"))


def test_cuda_resolution_rejects_unavailable_ordinal(monkeypatch) -> None:
    monkeypatch.setattr(
        device_module,
        "_probe_cuda",
        lambda: CudaInfo(
            compiled=True,
            available=True,
            devices=(CudaDeviceInfo(ordinal=0, name="test"),),
        ),
    )

    with pytest.raises(
        DeviceUnavailableError,
        match=r"device 1.*available devices are \[0\]",
    ):
        device_module.resolve_device(RuntimeConfig(device="cuda", cuda_device=1))
