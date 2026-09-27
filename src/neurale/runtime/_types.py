#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Define type aliases shared by runtime modules."""

from typing import Literal

DevicePreference = Literal["auto", "cpu", "cuda"]
ResolvedDevice = Literal["cpu", "cuda"]
