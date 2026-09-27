from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from _repository_module import load_repository_module

_ROOT = Path(__file__).resolve().parents[2]
_POLICY_PATH = _ROOT / "benchmarks" / "regression_policy.json"
_LABELS = ["self-hosted", "linux", "x64", "pyneurale-benchmark"]

_regression_policy = load_repository_module(
    "_regression_policy", _ROOT / "benchmarks" / "_regression_policy.py"
)
_benchmark_policy_cli = load_repository_module(
    "benchmark_policy", _ROOT / "tools" / "benchmark_policy.py"
)

BenchmarkPolicyError = _regression_policy.BenchmarkPolicyError
assess_history = _regression_policy.assess_history
build_environment_manifest = _regression_policy.build_environment_manifest
load_policy = _regression_policy.load_policy
validate_policy = _regression_policy.validate_policy


def _write_bundle(
    root: Path,
    *,
    run: int,
    values: tuple[float, float] = (100.0, 1_000.0),
    runner_name: str = "controlled-runner-1",
) -> Path:
    directory = root / f"run-{run}"
    directory.mkdir()
    rows = []
    for repetition in range(3):
        for feature, p99, throughput in (
            ("lmp", values[0], values[1]),
            ("bandpower", values[0] * 2, values[1] / 2),
        ):
            rows.append(
                {
                    "schema_version": 2,
                    "benchmark": "pipeline_realtime_physical_chain",
                    "scenario": "acquisition_4khz_256ch_1ms",
                    "feature": feature,
                    "bandpass": "sos" if feature == "lmp" else "none",
                    "resampler": False,
                    "run": repetition,
                    "p99": p99,
                    "scalar_samples_per_second": throughput,
                    "compiler": "test-compiler",
                    "build_type": "Release",
                    "native_version": "test-version",
                    "cpu_architecture": "x86_64",
                    "cpu_model": "test-cpu",
                    "logical_cores": 8,
                    "threading_backend": "builtin",
                    "threads": 8,
                    "cpu_math_backend": "none",
                    "build_fft_backend": "none",
                }
            )
    result_path = directory / "result.jsonl"
    result_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    environment = build_environment_manifest(
        results_path=result_path,
        runner_class="pyneurale-linux-x64-benchmark-v1",
        runner_labels=_LABELS,
        runner_name=runner_name,
        runner_image="test-image",
        evidence_run_id=f"evidence-{run}",
        git_sha="test-revision",
        exact_command="benchmark --warmup-seconds 1 --duration-seconds 60 --runs 3",
        provider="builtin",
        software_components=["cmake=test", "pybind11=test", "mkl=disabled"],
    )
    (directory / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    return directory


def test_policy_is_valid_without_unsubstantiated_gate() -> None:
    policy = load_policy(_POLICY_PATH)

    assert policy["active_performance_gates"] == []
    assert any(metric["status"] == "correctness_gate" for metric in policy["metrics"])


def test_hard_gate_requires_approved_historical_evidence() -> None:
    policy = load_policy(_POLICY_PATH)
    changed = copy.deepcopy(policy)
    changed["metrics"][2]["status"] = "hard_gate"
    changed["metrics"][2]["threshold"] = 100
    changed["active_performance_gates"] = [changed["metrics"][2]["id"]]

    with pytest.raises(BenchmarkPolicyError, match="requires approval evidence"):
        validate_policy(changed)


def test_controlled_history_requires_independent_runs(tmp_path: Path) -> None:
    policy = load_policy(_POLICY_PATH)
    inputs = [_write_bundle(tmp_path, run=idx) for idx in range(4)]

    report = assess_history(policy, inputs, "pyneurale-linux-x64-benchmark-v1")

    assert report["independent_runs"] == 4
    assert {metric["status"] for metric in report["metrics"]} == {"insufficient_history"}
    assert report["active_performance_gates"] == []


def test_stable_controlled_history_becomes_eligible_for_proposal(tmp_path: Path) -> None:
    policy = load_policy(_POLICY_PATH)
    inputs = [_write_bundle(tmp_path, run=idx) for idx in range(5)]

    report = assess_history(policy, inputs, "pyneurale-linux-x64-benchmark-v1")

    assert {metric["status"] for metric in report["metrics"]} == {"eligible_for_threshold_proposal"}
    assert all(
        scenario["relative_mad"] == 0
        for metric in report["metrics"]
        for scenario in metric["scenarios"]
    )


def test_variable_history_is_not_eligible(tmp_path: Path) -> None:
    policy = load_policy(_POLICY_PATH)
    values = (
        (100.0, 1_000.0),
        (100.0, 1_000.0),
        (100.0, 1_000.0),
        (130.0, 700.0),
        (140.0, 600.0),
    )
    inputs = [
        _write_bundle(tmp_path, run=idx, values=run_values) for idx, run_values in enumerate(values)
    ]

    report = assess_history(policy, inputs, "pyneurale-linux-x64-benchmark-v1")

    assert {metric["status"] for metric in report["metrics"]} == {"unstable"}


def test_history_rejects_host_fingerprint_changes(tmp_path: Path) -> None:
    policy = load_policy(_POLICY_PATH)
    inputs = [
        _write_bundle(tmp_path, run=0),
        _write_bundle(tmp_path, run=1, runner_name="different-host"),
    ]

    with pytest.raises(BenchmarkPolicyError, match="environment_fingerprint"):
        assess_history(policy, inputs, "pyneurale-linux-x64-benchmark-v1")


def test_hosted_characterization_is_not_controlled_history(tmp_path: Path) -> None:
    policy = load_policy(_POLICY_PATH)

    with pytest.raises(BenchmarkPolicyError, match="requires a controlled runner"):
        assess_history(policy, [_write_bundle(tmp_path, run=0)], "github-hosted-characterization")


def test_environment_capture_rejects_provider_metadata_contradiction(tmp_path: Path) -> None:
    result_path = tmp_path / "result.jsonl"
    row = {
        "schema_version": 2,
        "benchmark": "pipeline_realtime_physical_chain",
        "scenario": "acquisition_4khz_256ch_1ms",
        "compiler": "test-compiler",
        "build_type": "Release",
        "native_version": "test-version",
        "cpu_architecture": "x86_64",
        "cpu_model": "test-cpu",
        "logical_cores": 8,
        "threading_backend": "mkl",
        "threads": 8,
        "cpu_math_backend": "mkl",
        "build_fft_backend": "mkl",
    }
    result_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(BenchmarkPolicyError, match="contradicts"):
        build_environment_manifest(
            results_path=result_path,
            runner_class="github-hosted-characterization",
            runner_labels=["github-hosted"],
            evidence_run_id="run-1",
            git_sha="deadbeef",
            exact_command="benchmark --runs 3",
            provider="builtin",
        )


def test_benchmark_policy_cli_routes_all_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _benchmark_policy_cli.main(["validate", "--policy", str(_POLICY_PATH)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "valid"

    inputs = [_write_bundle(tmp_path, run=idx) for idx in range(5)]
    capture_output = tmp_path / "captured" / "environment.json"
    assert (
        _benchmark_policy_cli.main(
            [
                "capture",
                "--results",
                str(inputs[0] / "result.jsonl"),
                "--output",
                str(capture_output),
                "--runner-class",
                "pyneurale-linux-x64-benchmark-v1",
                "--runner-label",
                "self-hosted",
                "--evidence-run-id",
                "captured-run",
                "--git-sha",
                "deadbeef",
                "--exact-command",
                "benchmark --runs 3",
                "--provider",
                "builtin",
            ]
        )
        == 0
    )
    assert json.loads(capture_output.read_text(encoding="utf-8"))["result_file"] == "result.jsonl"

    assessment_output = tmp_path / "assessment.json"
    arguments = [
        "assess",
        "--policy",
        str(_POLICY_PATH),
        "--runner-class",
        "pyneurale-linux-x64-benchmark-v1",
        "--output",
        str(assessment_output),
    ]
    for input_path in inputs:
        arguments.extend(("--input", str(input_path)))
    assert _benchmark_policy_cli.main(arguments) == 0
    assert json.loads(assessment_output.read_text(encoding="utf-8"))["independent_runs"] == 5
