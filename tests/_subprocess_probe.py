# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Running a probe script in a fresh interpreter.

The import contract (``docs/architecture/overview.md``) is about what a bare
``import neurale.<domain>`` pulls in, so it can only be observed from an
interpreter that has not imported anything yet -- inside pytest, ``neurale``
and half its dependencies are already in ``sys.modules``. Every such test
therefore runs a short script under ``sys.executable`` and reads back what it
saw; this module owns that one call so the tests themselves stay the script
plus the assertion.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from typing import Any


def probe_json(
    script: str,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
) -> Any:
    """Run ``script`` in a fresh interpreter and return the JSON it printed.

    A non-zero exit raises :class:`subprocess.CalledProcessError`, whose message
    carries the child's traceback -- a probe that cannot run is a failure, never
    an empty result.
    """
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
        timeout=timeout,
    )
    return json.loads(completed.stdout)
