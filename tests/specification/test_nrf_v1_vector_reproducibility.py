#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""The checked-in vectors must be exactly what the generator produces.

``test-vectors.json`` is normative and language-neutral: an implementation in
any language is entitled to treat it as the format's ground truth. That only
holds while the file and its generator say the same thing, so this asks the
generator to rebuild the vectors and compares them to the file on disk.

A failure here means one of two things, and the diff says which: the generator
changed and the file was not regenerated, or the file was edited by hand. Both
are the same defect -- a normative artifact nothing derives.
"""

from __future__ import annotations

import json

from _nrf_v1_reference import SPEC_DIR, load_generator


def test_generator_writes_file_byte_for_byte() -> None:
    """Not just equal as JSON -- equal as a file.

    The vectors are compared byte-for-byte by anyone diffing the repository, and
    they carry canonical-form cases whose whole point is exact bytes. Separator
    style, key order, escaping, and the trailing newline are therefore part of
    the artifact, not formatting noise.
    """
    generator = load_generator()
    rebuilt = (
        json.dumps(generator.build_vectors(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    assert rebuilt == (SPEC_DIR / "test-vectors.json").read_text(encoding="utf-8")


def test_generator_writes_beside_its_specification() -> None:
    """The generator's output path is the one the suite reads from."""
    generator = load_generator()
    assert generator.OUTPUT == SPEC_DIR / "test-vectors.json"
