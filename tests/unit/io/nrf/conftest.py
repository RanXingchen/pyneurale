#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Fixtures shared by the NRF test modules.

The builders themselves live in :mod:`.nrf_support`, which the modules import
directly because they need them at module scope as well.

This file sits one directory down from the MAT and CSV tests on purpose. A
``conftest.py`` is imported for every test collected beneath it, so an NRF
import here would make the optional ``nrf`` dependencies -- Zarr, jsonschema,
rfc8785 -- a precondition for collecting ``test_mat.py``. ``neurale.io`` keeps
that boundary in the package; the test tree has to keep it too, or the boundary
is only true of code nobody runs.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from neurale.io.nrf import NrfWriter

from .nrf_support import build_writer


@pytest.fixture
def writer(tmp_path: Path) -> Iterator[NrfWriter]:
    """A frozen writer on an empty session, closed however the test ends."""
    instance = build_writer(tmp_path / "writer.nrf")
    yield instance
    instance.close()
