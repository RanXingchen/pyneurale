#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Clock metadata for time alignment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from neurale.exceptions import ValidationError

from ._helpers import (
    ensure_finite_float,
    ensure_non_empty_string,
    ensure_positive_float,
    freeze_metadata,
)


@dataclass(frozen=True, slots=True)
class Clock:
    """Describe a timing source used by samples or events.

    Parameters
    ----------
    name : str
        Non-empty clock name.
    type : str
        neurale.data.time.Clock or synchronization-source type.
    rate : float or None, optional
        Positive nominal clock rate in hertz.
    epoch : str or None, optional
        Textual description of the clock epoch.
    offset : float, default=0.0
        Time offset relative to a reference clock.
    drift : float, default=0.0
        neurale.data.time.Clock drift relative to a reference.
    synchronization_domain : str or None, optional
        Identifier for clocks that share one reference synchronization domain.
    attrs : dict, optional
        Deep-frozen application metadata using the closed value domain
        documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If name, type, rate, epoch metadata, synchronization domain, offset, or
        drift are invalid. Drift must keep ``1 + drift`` positive so
        reference-time conversion remains monotonic and invertible.
    """

    name: str
    type: str
    rate: float | None = None
    epoch: str | None = None
    offset: float = 0.0
    drift: float = 0.0
    synchronization_domain: str | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = ensure_non_empty_string(self.name, "clock name")
        clock_type = ensure_non_empty_string(self.type, "clock type")
        rate = ensure_positive_float(self.rate, "clock rate")
        epoch = self.epoch
        if epoch is not None:
            epoch = ensure_non_empty_string(epoch, "clock epoch")
        synchronization_domain = self.synchronization_domain
        if synchronization_domain is not None:
            synchronization_domain = ensure_non_empty_string(
                synchronization_domain,
                "clock synchronization_domain",
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "type", clock_type)
        object.__setattr__(self, "rate", rate)
        object.__setattr__(self, "epoch", epoch)
        object.__setattr__(self, "synchronization_domain", synchronization_domain)
        drift = ensure_finite_float(self.drift, "clock drift")
        if 1.0 + drift <= 0.0:
            raise ValidationError("clock drift must keep 1 + drift positive.")
        object.__setattr__(self, "offset", ensure_finite_float(self.offset, "clock offset"))
        object.__setattr__(self, "drift", drift)
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))
