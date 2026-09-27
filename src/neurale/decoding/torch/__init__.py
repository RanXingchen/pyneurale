#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The boundary Torch-backed decoders will live behind, and nothing else.

This namespace exists so that the answer to "where does a Torch decoder go, and
how does it ask for Torch?" is fixed *before* the first one is written, rather
than settled by whichever module happens to import Torch first. It carries no
model, no network, no training loop, and no placeholder that could be mistaken
for one: importing it and finding it empty of estimators is the accurate
report, not an oversight.

Two import properties are the point of the boundary and are covered by tests:

* Importing :mod:`neurale.decoding` does not import or probe this namespace.
  The classical decoders next door are pure Python over NumPy, and installing
  Torch is not a condition of using them.
* Importing *this* namespace does not import Torch either. It is resolved at
  first use through :func:`require_torch`, the same way
  :mod:`neurale.io` resolves SciPy and :mod:`neurale.io.nrf` resolves Zarr, so
  a missing framework surfaces as :class:`~neurale.exceptions.DependencyError`
  when a capability is requested and never as an import-time failure of an
  unrelated one.

No Torch model, network, loss, scheduler, augmentation function, or training
loop is shipped. This namespace does not stand in for those capabilities with
stub classes.

There is no ``torch`` extra yet. A framework that no shipped capability uses is
not a distribution PyNeurale can honestly offer to install on the user's
behalf, so :func:`require_torch` names Torch itself in its installation hint --
as :mod:`neurale.io` does for SciPy. The extra belongs to the first release
that has something to run on it.

Examples
--------
Ask before branching, and let the error explain itself otherwise::

    from neurale.decoding.torch import has_torch, require_torch

    if has_torch():
        torch = require_torch()
"""

from __future__ import annotations

from types import ModuleType

from neurale.runtime.dependencies import DependencyStatus, dependency_status, require_dependency

#: Distribution and module name of the framework this namespace is gated on.
#: Both are ``"torch"``; naming it once is what keeps every future decoder in
#: this namespace asking the same question and reporting the same answer.
_TORCH = "torch"


def torch_status() -> DependencyStatus:
    """Report whether Torch is available, without importing it.

    Returns
    -------
    neurale.runtime.dependencies.DependencyStatus
        Availability, installed version when the distribution metadata is
        present, and the reason the framework is unusable when it is not.
    """

    return dependency_status(_TORCH)


def has_torch() -> bool:
    """Return whether Torch can be located.

    The check locates the module rather than importing it, so asking is cheap
    and has no side effect on a process that will not use Torch.

    Returns
    -------
    bool
        ``True`` when Torch can be imported by :func:`require_torch`.
    """

    return torch_status().available


def require_torch() -> ModuleType:
    """Import Torch or raise an actionable dependency error.

    This is the only supported way for code in this namespace to reach Torch.
    A module-level ``import torch`` would make an optional framework decide
    whether an unrelated import succeeds.

    Returns
    -------
    types.ModuleType
        The imported :mod:`torch` module.

    Raises
    ------
    neurale.exceptions.DependencyError
        If Torch is not installed, or is installed but cannot be imported.
    """

    return require_dependency(_TORCH)


__all__ = [
    "has_torch",
    "require_torch",
    "torch_status",
]
