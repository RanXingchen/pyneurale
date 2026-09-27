#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Tests for :mod:`neurale.io.nrf`.

This directory is a package so its ``conftest.py`` is imported as
``nrf.conftest`` rather than as the top-level name ``conftest``. Without it
pytest -- ``tests/`` has no ``__init__.py`` -- gives every ``conftest.py`` in
the tree the same module name, and whichever loads last owns
``sys.modules["conftest"]``, which is the root ``conftest`` that
``tests/unit/test_suite_classification.py`` imports by name.
"""
