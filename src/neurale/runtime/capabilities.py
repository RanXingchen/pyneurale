#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Define immutable runtime capability records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class Capability:
    """Describe a process-level runtime capability.

    Parameters
    ----------
    name : str
        Stable capability identifier.
    available : bool
        Whether the capability is currently available.
    compiled : bool or None, optional
        Whether support was compiled into the native extension.
    usable : bool or None, optional
        Whether the capability can currently execute work.
    provider : str or None, optional
        Library or runtime providing the capability.
    version : str or None, optional
        Provider version.
    reason : str or None, optional
        Explanation when the capability is unavailable.
    details : mapping, optional
        Additional immutable capability metadata.
    """

    name: str
    available: bool
    compiled: bool | None = None
    usable: bool | None = None
    provider: str | None = None
    version: str | None = None
    reason: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "details", MappingProxyType(dict(self.details)))
