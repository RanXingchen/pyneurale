#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Focused interoperability with external scientific file formats.

Importing :mod:`neurale.io` does not import SciPy. Format-specific dependencies
are resolved only when an I/O operation is called.

MAT structs inside top-level object arrays are converted recursively, ordinary
numeric arrays nested in structs remain NumPy arrays, and unsupported MATLAB
v7.3/HDF5 inputs raise an explicit format error.

CSV compatibility skips empty cells after the first row rather than inserting
placeholders, so affected columns become shorter. Headerless input retains
every first-row cell verbatim, including empty strings. CSV I/O fixes the
encoding to UTF-8 and explicitly rejects empty files/mappings, duplicate
headers, inconsistent row widths, non-mapping save inputs, non-iterable
columns, and missing output directories.
"""

from ._csv import loadcsv, savecsv
from ._mat import loadmat, savemat

__all__ = ["loadcsv", "loadmat", "savecsv", "savemat"]
