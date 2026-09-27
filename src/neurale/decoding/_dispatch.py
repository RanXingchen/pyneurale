#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Device resolution for decoder fitting.

Resolution happens at the explicit execution boundary that :meth:`fit` is, not
at import time, and it never probes CUDA for a decoder that has no CUDA
implementation -- a requested ``cuda`` simply fails.
"""

from __future__ import annotations

from neurale.runtime._device import resolve_device
from neurale.runtime._types import ResolvedDevice


def resolve_decoder_device(operation: str, *, supports_cuda: bool = False) -> ResolvedDevice:
    """Resolve the device one decoder operation runs on."""

    return resolve_device(
        operation=f"decoding.{operation}",
        supports_cuda=supports_cuda,
    ).resolved
