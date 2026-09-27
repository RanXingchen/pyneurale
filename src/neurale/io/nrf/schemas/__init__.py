#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Install destination of the normative NRF v1 JSON Schema documents.

This package holds no schema in the source tree, and nothing writes one back
into it. The only source of truth is ``specifications/nrf/v1/*.schema.json``;
the root ``CMakeLists.txt`` installs those files here when a wheel is built, so
an installed reader carries the contract without the repository carrying a
second copy of it to keep in step.

The consequence is that this package is empty in a plain source checkout.
Anything that loads a bundled schema -- :mod:`neurale.io.nrf._schemas`, and so
every reader, writer, and recovery path -- therefore needs a build, not just
``src`` on ``sys.path``.
"""
