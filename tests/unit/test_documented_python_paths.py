from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = re.compile(r"(?<![/\w])(?:benchmarks|tests|tools)/[A-Za-z0-9_./-]+\.py")


def test_documented_python_paths_exist() -> None:
    missing = []
    for document in (_ROOT / "docs").rglob("*.md"):
        for line_number, line in enumerate(document.read_text(encoding="utf-8").splitlines(), 1):
            for match in _SCRIPT_PATH.finditer(line):
                if not (_ROOT / match.group()).is_file():
                    missing.append(f"{document.relative_to(_ROOT)}:{line_number}: {match.group()}")
    assert not missing, "Missing documented Python paths:\n" + "\n".join(missing)
