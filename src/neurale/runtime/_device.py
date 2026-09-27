#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Resolve requested runtime devices without implicit fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neurale.exceptions import DeviceUnavailableError

from ._types import DevicePreference, ResolvedDevice

if TYPE_CHECKING:
    from .config import RuntimeConfig
    from .diagnostics import CudaInfo


@dataclass(frozen=True, slots=True)
class DeviceResolution:
    """Internal requested-to-resolved device result."""

    requested: DevicePreference
    resolved: ResolvedDevice
    cuda: CudaInfo | None = None
    cuda_device: int | None = None


def resolve_device(
    config: RuntimeConfig | None = None,
    *,
    operation: str | None = None,
    supports_cuda: bool = True,
) -> DeviceResolution:
    """Resolve one device request at an explicit execution boundary."""

    if config is None:
        from .context import get_runtime_config

        config = get_runtime_config()

    if config.device in {"auto", "cpu"}:
        return DeviceResolution(requested=config.device, resolved="cpu")

    if not supports_cuda:
        target = "this operation" if operation is None else operation
        raise DeviceUnavailableError(
            f"Requested device 'cuda' is unavailable for {target}: "
            "the operation has no CUDA implementation."
        )

    cuda = _probe_cuda()
    _require_usable_cuda(cuda, config.cuda_device, operation=operation)
    return DeviceResolution(
        requested="cuda",
        resolved="cuda",
        cuda=cuda,
        cuda_device=config.cuda_device,
    )


def _probe_cuda() -> CudaInfo:
    from .diagnostics import CudaInfo, runtime_info

    info = runtime_info(probe_native=True, probe_cuda=True)
    return info.cuda or CudaInfo(
        compiled=False,
        available=False,
        reason="CUDA diagnostics did not return capability information",
    )


def _raise_cuda_unavailable(cuda: CudaInfo, context: str, default_reason: str) -> None:
    reason = (cuda.reason or default_reason).rstrip(".")
    raise DeviceUnavailableError(f"Requested device 'cuda' is unavailable{context}: {reason}.")


def _require_usable_cuda(
    cuda: CudaInfo,
    cuda_device: int | None,
    *,
    operation: str | None,
) -> None:
    context = "" if operation is None else f" for {operation}"
    if not cuda.compiled:
        _raise_cuda_unavailable(
            cuda, context, "the native extension was built without CUDA support"
        )
    if not cuda.available:
        _raise_cuda_unavailable(cuda, context, "no usable CUDA runtime and device were found")
    if cuda_device is not None:
        ordinals = {device.ordinal for device in cuda.devices}
        if cuda_device not in ordinals:
            raise DeviceUnavailableError(
                f"Requested CUDA device {cuda_device} is unavailable{context}; "
                f"available devices are {sorted(ordinals)}."
            )
