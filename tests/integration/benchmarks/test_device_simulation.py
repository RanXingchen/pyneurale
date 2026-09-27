from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.native

_ROOT = Path(__file__).resolve().parents[3]
_BENCHMARK = _ROOT / "benchmarks" / "devices" / "benchmark_simulation.py"


def test_recording_benchmark_cli_produces_both_scenarios(tmp_path: Path) -> None:
    output = tmp_path / "simulation.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(_BENCHMARK),
            "--skip-native",
            "--channels",
            "4",
            "--fs",
            "30000",
            "--record-fs",
            "15000",
            "--frames",
            "32",
            "--warmup-frames",
            "0",
            "--runs",
            "1",
            "--loss-period-frames",
            "16",
            "--output",
            str(output),
            "--work-dir",
            str(tmp_path / "sessions"),
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {(row["scenario"], row["metric"]) for row in rows} == {
        (scenario, metric)
        for scenario in ("continuous", "scheduled_sample_loss")
        for metric in ("record_session", "finalize_session")
    }
    assert all(row["fs"] == 15_000 for row in rows)
    assert all(row["session_bytes"] > 0 for row in rows)
