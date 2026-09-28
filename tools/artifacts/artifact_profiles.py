# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Capabilities promised by each built artifact profile."""

PROFILE_CAPABILITIES = {
    "core": frozenset(),
    "presentation": frozenset({"presentation"}),
    "cuda": frozenset({"cuda"}),
    "release": frozenset({"presentation", "cuda", "mkl"}),
}
PROFILE_MODULES = {
    profile: frozenset({"_native", "_native_cuda"} if "cuda" in capabilities else {"_native"})
    for profile, capabilities in PROFILE_CAPABILITIES.items()
}
RUNTIME_REQUIREMENTS = frozenset({"numpy", "scipy"})
