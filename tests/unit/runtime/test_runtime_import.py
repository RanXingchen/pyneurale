#!/usr/bin/env python3

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_runtime_import_has_no_native_or_environment_side_effects() -> None:
    source = Path(__file__).resolve().parents[3] / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(source)
    environment.pop("MKL_NUM_THREADS", None)
    environment.pop("OMP_NUM_THREADS", None)
    environment.pop("MKL_DYNAMIC", None)
    script = """
import json
import logging
import os
import sys
import neurale.runtime

package_logger = logging.Logger.manager.loggerDict.get("neurale")
print(json.dumps({
    "native": "neurale._native" in sys.modules,
    "native_cuda": "neurale._native_cuda" in sys.modules,
    "mkl_threads": os.environ.get("MKL_NUM_THREADS"),
    "logging_handlers": (
        len(package_logger.handlers)
        if isinstance(package_logger, logging.Logger)
        else 0
    ),
    "logging_names": sorted(
        name
        for name in logging.Logger.manager.loggerDict
        if name == "neurale" or name.startswith("neurale.")
    ),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == (
        '{"native": false, "native_cuda": false, '
        '"mkl_threads": null, "logging_handlers": 0, "logging_names": []}'
    )
