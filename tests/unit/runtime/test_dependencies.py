#!/usr/bin/env python3

from __future__ import annotations

import pytest

from neurale.exceptions import DependencyError
from neurale.runtime import dependency_status, has_dependency, require_dependency


def test_dependency_status_finds_standard_library_module() -> None:
    status = dependency_status("definitely-not-a-distribution", import_name="json")

    assert status.available
    assert status.import_name == "json"
    assert status.version is None
    assert has_dependency("unused", import_name="json")


def test_require_dependency_imports_module() -> None:
    module = require_dependency("unused", import_name="json")

    assert module.dumps({"value": 1}) == '{"value": 1}'


def test_require_dependency_raises_actionable_error() -> None:
    with pytest.raises(DependencyError, match=r"pyneurale\[viz\]"):
        require_dependency(
            "missing-pyneurale-test-package",
            import_name="_missing_pyneurale_test_package",
            extra="viz",
        )
