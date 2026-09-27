#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Inspect and import optional Python dependencies."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from importlib import metadata
from types import ModuleType

from neurale.exceptions import DependencyError


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    """Describe whether an optional Python dependency is available.

    Parameters
    ----------
    distribution : str
        Installed distribution name.
    import_name : str
        Python module name.
    available : bool
        Whether the module can be located.
    version : str or None
        Installed distribution version.
    reason : str or None, optional
        Explanation when the dependency is unavailable.
    """

    distribution: str
    import_name: str
    available: bool
    version: str | None
    reason: str | None = None


def dependency_status(
    distribution: str,
    *,
    import_name: str | None = None,
) -> DependencyStatus:
    """Inspect an optional Python dependency without importing it.

    Parameters
    ----------
    distribution : str
        Distribution name used for metadata lookup.
    import_name : str or None, optional
        Module name. Hyphens are converted to underscores when omitted.

    Returns
    -------
    neurale.runtime.dependencies.DependencyStatus
        Availability, version, and failure details.
    """

    module_name = distribution.replace("-", "_") if import_name is None else import_name
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        version = None

    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError) as exc:
        return DependencyStatus(
            distribution=distribution,
            import_name=module_name,
            available=False,
            version=version,
            reason=str(exc),
        )

    if spec is None:
        reason = (
            f"distribution {distribution!r} is not installed."
            if version is None
            else f"module {module_name!r} cannot be found."
        )
        return DependencyStatus(
            distribution=distribution,
            import_name=module_name,
            available=False,
            version=version,
            reason=reason,
        )
    return DependencyStatus(
        distribution=distribution,
        import_name=module_name,
        available=True,
        version=version,
    )


def has_dependency(
    distribution: str,
    *,
    import_name: str | None = None,
) -> bool:
    """Return whether an optional Python dependency can be located.

    Parameters
    ----------
    distribution : str
        Distribution name used for metadata lookup.
    import_name : str or None, optional
        Module name override.

    Returns
    -------
    bool
        ``True`` when the module can be located.
    """

    return dependency_status(distribution, import_name=import_name).available


def require_dependency(
    distribution: str,
    *,
    import_name: str | None = None,
    extra: str | None = None,
) -> ModuleType:
    """Import an optional dependency or raise an actionable error.

    Parameters
    ----------
    distribution : str
        Distribution name used for availability checks.
    import_name : str or None, optional
        Module name override.
    extra : str or None, optional
        PyNeurale extra named in the installation hint.

    Returns
    -------
    types.ModuleType
        Imported module.

    Raises
    ------
    neurale.exceptions.DependencyError
        If the dependency is unavailable or cannot be imported.
    """

    status = dependency_status(distribution, import_name=import_name)
    if not status.available:
        install_target = f"pyneurale[{extra}]" if extra else distribution
        raise DependencyError(
            f"{distribution} is required for this operation. "
            f'Install it with: pip install "{install_target}". '
            f"Reason: {status.reason}"
        )
    try:
        return importlib.import_module(status.import_name)
    except (ImportError, OSError) as exc:
        raise DependencyError(
            f"{distribution} was found but {status.import_name!r} could not be imported."
        ) from exc
