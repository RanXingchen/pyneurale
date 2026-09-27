#!/usr/bin/env python3

"""Capture benchmark evidence and evaluate the private regression policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks._regression_policy import (  # noqa: E402
    assess_history,
    build_environment_manifest,
    load_policy,
)


def _capture(arguments: argparse.Namespace) -> int:
    manifest = build_environment_manifest(
        results_path=arguments.results,
        runner_class=arguments.runner_class,
        runner_labels=arguments.runner_label,
        evidence_run_id=arguments.evidence_run_id,
        git_sha=arguments.git_sha,
        exact_command=arguments.exact_command,
        provider=arguments.provider,
        runner_name=arguments.runner_name,
        runner_image=arguments.runner_image,
        software_components=arguments.software_component,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    capture = subparsers.add_parser("capture")
    capture.add_argument("--results", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--runner-class", required=True)
    capture.add_argument("--runner-label", action="append", default=[])
    capture.add_argument("--runner-name")
    capture.add_argument("--runner-image")
    capture.add_argument("--evidence-run-id", required=True)
    capture.add_argument("--git-sha", required=True)
    capture.add_argument("--exact-command", required=True)
    capture.add_argument("--provider", required=True)
    capture.add_argument("--software-component", action="append", default=[])

    validate = subparsers.add_parser("validate")
    validate.add_argument("--policy", type=Path, required=True)

    assess = subparsers.add_parser("assess")
    assess.add_argument("--policy", type=Path, required=True)
    assess.add_argument("--runner-class", required=True)
    assess.add_argument("--input", type=Path, action="append", required=True)
    assess.add_argument("--output", type=Path, required=True)

    arguments = parser.parse_args(argv)
    if arguments.command == "capture":
        return _capture(arguments)

    policy = load_policy(arguments.policy)
    if arguments.command == "validate":
        print(json.dumps({"schema_version": 1, "status": "valid"}, sort_keys=True))
        return 0

    report = assess_history(policy, arguments.input, arguments.runner_class)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
