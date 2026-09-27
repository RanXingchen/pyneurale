#!/usr/bin/env python3

"""Validation and history assessment for the private benchmark policy."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_METRIC_STATUSES = {
    "correctness_gate",
    "history_required",
    "hard_gate",
    "smoke",
    "release_evidence",
    "deferred",
}
_SHARED_RESULT_FIELDS = (
    "schema_version",
    "benchmark",
    "scenario",
    "compiler",
    "build_type",
    "native_version",
    "cpu_architecture",
    "cpu_model",
    "logical_cores",
    "threading_backend",
    "threads",
    "cpu_math_backend",
    "build_fft_backend",
)


class BenchmarkPolicyError(ValueError):
    """Raised when policy or evidence violates the frozen contract."""


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _environment_fingerprint_payload(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "runner_class": environment.get("runner_class"),
        "runner_labels": environment.get("runner_labels"),
        "runner_name": environment.get("runner_name"),
        "runner_image": environment.get("runner_image"),
        "platform": environment.get("platform"),
        "os": environment.get("os"),
        "os_release": environment.get("os_release"),
        "kernel": environment.get("kernel"),
        "machine": environment.get("machine"),
        "provider": environment.get("provider"),
        "software_components": environment.get("software_components"),
        "timing_environment": environment.get("timing_environment"),
        "shared_result_metadata": environment.get("shared_result_metadata"),
    }


def _definition_fingerprint_payload(environment: Mapping[str, Any]) -> dict[str, Any]:
    shared = environment.get("shared_result_metadata")
    return {
        "git_sha": environment.get("git_sha"),
        "exact_command": environment.get("exact_command"),
        "benchmark": shared.get("benchmark") if isinstance(shared, Mapping) else None,
        "schema_version": shared.get("schema_version") if isinstance(shared, Mapping) else None,
    }


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BenchmarkPolicyError(f"cannot read JSON from {path}: {error}") from error


def load_policy(path: Path) -> dict[str, Any]:
    value = _load_json(path)
    if not isinstance(value, dict):
        raise BenchmarkPolicyError("benchmark policy must be a JSON object")
    validate_policy(value)
    return value


def validate_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("schema_version") != 1:
        raise BenchmarkPolicyError("benchmark policy schema_version must be 1")
    history = policy.get("historical_evidence")
    if not isinstance(history, Mapping):
        raise BenchmarkPolicyError("historical_evidence must be an object")
    for field in ("minimum_independent_runs", "minimum_repetitions_per_scenario"):
        if not isinstance(history.get(field), int) or history[field] < 1:
            raise BenchmarkPolicyError(f"{field} must be a positive integer")
    for field in ("maximum_relative_mad", "maximum_relative_range"):
        value = history.get(field)
        if not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise BenchmarkPolicyError(f"{field} must be in [0, 1)")

    runner_classes = policy.get("runner_classes")
    if not isinstance(runner_classes, Mapping) or not runner_classes:
        raise BenchmarkPolicyError("runner_classes must be a non-empty object")
    controlled_count = 0
    for name, runner in runner_classes.items():
        if not isinstance(name, str) or not name or not isinstance(runner, Mapping):
            raise BenchmarkPolicyError("runner class entries must be named objects")
        if not isinstance(runner.get("controlled"), bool):
            raise BenchmarkPolicyError(f"runner class {name!r} must declare controlled")
        controlled_count += int(runner["controlled"])
    if controlled_count == 0:
        raise BenchmarkPolicyError("at least one controlled runner class is required")

    metrics = policy.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise BenchmarkPolicyError("metrics must be a non-empty array")
    ids: set[str] = set()
    hard_gate_ids: set[str] = set()
    for metric in metrics:
        if not isinstance(metric, Mapping):
            raise BenchmarkPolicyError("each metric must be an object")
        metric_id = metric.get("id")
        status = metric.get("status")
        if not isinstance(metric_id, str) or not metric_id or metric_id in ids:
            raise BenchmarkPolicyError("metric IDs must be unique non-empty strings")
        ids.add(metric_id)
        if status not in _METRIC_STATUSES:
            raise BenchmarkPolicyError(f"metric {metric_id!r} has invalid status {status!r}")
        if status == "hard_gate":
            hard_gate_ids.add(metric_id)
            approval = metric.get("approval")
            if not isinstance(approval, Mapping):
                raise BenchmarkPolicyError(f"hard gate {metric_id!r} requires approval evidence")
            run_ids = approval.get("historical_run_ids")
            if (
                not isinstance(run_ids, list)
                or any(not isinstance(run_id, str) or not run_id for run_id in run_ids)
                or len(set(run_ids)) < history["minimum_independent_runs"]
            ):
                raise BenchmarkPolicyError(
                    f"hard gate {metric_id!r} lacks the required independent history"
                )
            if (
                not approval.get("approved_by_change")
                or not approval.get("baseline_sha256")
                or not approval.get("eligibility_report_sha256")
            ):
                raise BenchmarkPolicyError(
                    f"hard gate {metric_id!r} requires an approving change, baseline, and eligibility report"
                )
            threshold = metric.get("threshold")
            if (
                not isinstance(threshold, (int, float))
                or not math.isfinite(threshold)
                or threshold <= 0
            ):
                raise BenchmarkPolicyError(f"hard gate {metric_id!r} requires a positive threshold")

    active = policy.get("active_performance_gates")
    if not isinstance(active, list) or len(active) != len(set(active)):
        raise BenchmarkPolicyError("active_performance_gates must be a unique array")
    if set(active) != hard_gate_ids:
        raise BenchmarkPolicyError("active_performance_gates must exactly list hard_gate metrics")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise BenchmarkPolicyError(f"cannot read benchmark results {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise BenchmarkPolicyError(f"invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(row, dict):
            raise BenchmarkPolicyError(f"benchmark row at {path}:{line_number} must be an object")
        rows.append(row)
    if not rows:
        raise BenchmarkPolicyError(f"benchmark result is empty: {path}")
    return rows


def build_environment_manifest(
    *,
    results_path: Path,
    runner_class: str,
    runner_labels: list[str],
    evidence_run_id: str,
    git_sha: str,
    exact_command: str,
    provider: str,
    runner_name: str | None = None,
    runner_image: str | None = None,
    software_components: list[str] | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(results_path)
    first = rows[0]
    shared: dict[str, Any] = {}
    for field in _SHARED_RESULT_FIELDS:
        if field not in first:
            raise BenchmarkPolicyError(f"benchmark results are missing shared field {field!r}")
        shared[field] = first[field]
        if any(row.get(field) != first[field] for row in rows):
            raise BenchmarkPolicyError(f"benchmark field {field!r} changes within one evidence run")
    expected_backends = {"builtin": ("none", "none"), "mkl": ("mkl", "mkl")}
    if provider in expected_backends:
        expected_cpu, expected_fft = expected_backends[provider]
        if (shared["cpu_math_backend"], shared["build_fft_backend"]) != (
            expected_cpu,
            expected_fft,
        ):
            raise BenchmarkPolicyError(
                f"provider {provider!r} contradicts the benchmark build metadata"
            )
    result_sha256 = hashlib.sha256(results_path.read_bytes()).hexdigest()
    components: dict[str, str] = {}
    for component in software_components or ():
        name, separator, value = component.partition("=")
        if not separator or not name or not value or name in components:
            raise BenchmarkPolicyError("software components must be unique NAME=VALUE entries")
        components[name] = value
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))

    def optional_text(path: str) -> str | None:
        try:
            return Path(path).read_text(encoding="utf-8").strip() or None
        except OSError:
            return None

    environment = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "evidence_run_id": evidence_run_id,
        "runner_class": runner_class,
        "runner_labels": sorted(set(runner_labels)),
        "runner_name": runner_name,
        "runner_image": runner_image,
        "git_sha": git_sha,
        "exact_command": exact_command,
        "provider": provider,
        "result_file": results_path.name,
        "result_sha256": result_sha256,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "kernel": platform.version(),
        "machine": platform.machine(),
        "storage": "not_applicable",
        "software_components": components,
        "timing_environment": {
            "cpu_affinity": affinity,
            "linux_cpu_governor": optional_text(
                "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
            ),
            "linux_smt_control": optional_text("/sys/devices/system/cpu/smt/control"),
            "linux_kernel_command_line": optional_text("/proc/cmdline"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        },
        "shared_result_metadata": shared,
    }
    environment["environment_fingerprint"] = _canonical_hash(
        _environment_fingerprint_payload(environment)
    )
    environment["benchmark_definition_fingerprint"] = _canonical_hash(
        _definition_fingerprint_payload(environment)
    )
    return environment


def _load_bundle(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    environment_path = directory / "environment.json"
    environment = _load_json(environment_path)
    if not isinstance(environment, dict):
        raise BenchmarkPolicyError(f"environment manifest must be an object: {environment_path}")
    result_file = environment.get("result_file")
    if not isinstance(result_file, str) or Path(result_file).name != result_file:
        raise BenchmarkPolicyError(
            f"environment result_file must be a filename: {environment_path}"
        )
    result_path = directory / result_file
    try:
        digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    except OSError as error:
        raise BenchmarkPolicyError(
            f"cannot read benchmark results {result_path}: {error}"
        ) from error
    if digest != environment.get("result_sha256"):
        raise BenchmarkPolicyError(f"result checksum does not match {environment_path}")
    rows = read_jsonl(result_path)
    shared = environment.get("shared_result_metadata")
    if not isinstance(shared, Mapping):
        raise BenchmarkPolicyError(
            f"environment is missing shared result metadata: {environment_path}"
        )
    for field in _SHARED_RESULT_FIELDS:
        if shared.get(field) != rows[0].get(field) or any(
            row.get(field) != shared.get(field) for row in rows
        ):
            raise BenchmarkPolicyError(
                f"environment shared field {field!r} does not match {result_path}"
            )
    expected_environment = _canonical_hash(_environment_fingerprint_payload(environment))
    if environment.get("environment_fingerprint") != expected_environment:
        raise BenchmarkPolicyError(f"environment fingerprint is invalid: {environment_path}")
    expected_definition = _canonical_hash(_definition_fingerprint_payload(environment))
    if environment.get("benchmark_definition_fingerprint") != expected_definition:
        raise BenchmarkPolicyError(
            f"benchmark definition fingerprint is invalid: {environment_path}"
        )
    return environment, rows


def _relative_mad(values: Iterable[float]) -> float:
    samples = list(values)
    center = statistics.median(samples)
    deviation = statistics.median(abs(value - center) for value in samples)
    if center == 0:
        return 0.0 if deviation == 0 else math.inf
    return deviation / abs(center)


def _relative_range(values: Iterable[float]) -> float:
    samples = list(values)
    center = statistics.median(samples)
    spread = max(samples) - min(samples)
    if center == 0:
        return 0.0 if spread == 0 else math.inf
    return spread / abs(center)


def assess_history(
    policy: Mapping[str, Any], directories: list[Path], runner_class: str
) -> dict[str, Any]:
    validate_policy(policy)
    runner = policy["runner_classes"].get(runner_class)
    if not isinstance(runner, Mapping):
        raise BenchmarkPolicyError(f"unknown runner class {runner_class!r}")
    if not runner["controlled"]:
        raise BenchmarkPolicyError("historical eligibility requires a controlled runner class")
    if not directories:
        raise BenchmarkPolicyError("at least one evidence directory is required")

    bundles = [_load_bundle(path) for path in directories]
    environments = [bundle[0] for bundle in bundles]
    if any(environment.get("runner_class") != runner_class for environment in environments):
        raise BenchmarkPolicyError("evidence runner class does not match the requested class")
    required_labels = set(runner.get("required_labels", ()))
    if any(
        not required_labels.issubset(environment.get("runner_labels", ()))
        for environment in environments
    ):
        raise BenchmarkPolicyError("controlled evidence is missing required runner labels")
    run_ids = [environment.get("evidence_run_id") for environment in environments]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids):
        raise BenchmarkPolicyError("every evidence bundle requires a non-empty evidence_run_id")
    if len(set(run_ids)) != len(run_ids):
        raise BenchmarkPolicyError("historical evidence run IDs must be independent and unique")
    for field in ("environment_fingerprint", "benchmark_definition_fingerprint"):
        values = {environment.get(field) for environment in environments}
        if None in values or len(values) != 1:
            raise BenchmarkPolicyError(f"historical evidence has mismatched {field}")

    history = policy["historical_evidence"]
    metric_reports = []
    for metric in policy["metrics"]:
        if metric["status"] != "history_required" or "benchmark" not in metric:
            continue
        scenario_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
        expected_scenarios: set[tuple[Any, ...]] | None = None
        for _, rows in bundles:
            selected = [row for row in rows if row.get("benchmark") == metric["benchmark"]]
            grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
            for row in selected:
                scenario = tuple(row.get(field) for field in metric["scenario_fields"])
                value = row.get(metric["metric"])
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise BenchmarkPolicyError(
                        f"metric {metric['id']!r} contains a missing or non-finite value"
                    )
                grouped[scenario].append(float(value))
            if not grouped:
                raise BenchmarkPolicyError(f"metric {metric['id']!r} has no matching rows")
            if any(
                len(values) < history["minimum_repetitions_per_scenario"]
                for values in grouped.values()
            ):
                raise BenchmarkPolicyError(
                    f"metric {metric['id']!r} has too few repetitions for a scenario"
                )
            scenarios = set(grouped)
            if expected_scenarios is None:
                expected_scenarios = scenarios
            elif scenarios != expected_scenarios:
                raise BenchmarkPolicyError(f"metric {metric['id']!r} scenario sets differ")
            for scenario, values in grouped.items():
                scenario_values[scenario].append(float(statistics.median(values)))

        scenarios_report = []
        stable = True
        for scenario, values in sorted(scenario_values.items(), key=lambda item: repr(item[0])):
            relative_mad = _relative_mad(values)
            relative_range = _relative_range(values)
            stable = stable and (
                relative_mad <= history["maximum_relative_mad"]
                and relative_range <= history["maximum_relative_range"]
            )
            scenarios_report.append(
                {
                    "scenario": dict(zip(metric["scenario_fields"], scenario, strict=True)),
                    "independent_runs": len(values),
                    "median": statistics.median(values),
                    "relative_mad": relative_mad,
                    "relative_range": relative_range,
                }
            )
        enough_history = len(bundles) >= history["minimum_independent_runs"]
        status = (
            "eligible_for_threshold_proposal"
            if enough_history and stable
            else "insufficient_history"
            if not enough_history
            else "unstable"
        )
        metric_reports.append(
            {
                "id": metric["id"],
                "status": status,
                "hard_gate": False,
                "scenarios": scenarios_report,
            }
        )

    return {
        "schema_version": 1,
        "runner_class": runner_class,
        "evidence_run_ids": run_ids,
        "independent_runs": len(bundles),
        "environment_fingerprint": environments[0]["environment_fingerprint"],
        "benchmark_definition_fingerprint": environments[0]["benchmark_definition_fingerprint"],
        "active_performance_gates": list(policy["active_performance_gates"]),
        "metrics": metric_reports,
    }
