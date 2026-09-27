#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""RFC 8785 JSON Canonicalization Scheme helpers for NRF v1 tooling."""

from __future__ import annotations

from typing import Any

import rfc8785


def canonical_json_bytes(value: Any) -> bytes:
    """Return the RFC 8785 canonical UTF-8 representation of *value*."""

    return rfc8785.dumps(value)


def canonical_json(value: Any) -> str:
    """Return the RFC 8785 canonical representation of *value* as text."""

    return canonical_json_bytes(value).decode("utf-8")
