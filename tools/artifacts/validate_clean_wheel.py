# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Install a wheel into a temporary venv and validate only that installation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

if __package__:
    from .artifact_profiles import PROFILE_CAPABILITIES
    from .validate_wheel_profile import WheelProfileError, _select_wheel, validate_wheel_profile
else:
    from artifact_profiles import PROFILE_CAPABILITIES
    from validate_wheel_profile import WheelProfileError, _select_wheel, validate_wheel_profile


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return environment


def validate_clean_wheel(wheel: Path, profile: str) -> dict[str, object]:
    """Install *wheel* with normal dependencies and return validation evidence."""

    wheel = wheel.resolve()
    validate_wheel_profile(wheel, profile)
    validator = Path(__file__).with_name("validate_installed_artifact.py").resolve()
    environment = _clean_environment()

    with tempfile.TemporaryDirectory(prefix="pyneurale-clean-wheel-") as directory:
        root = Path(directory)
        virtual_environment = root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(virtual_environment)
        python = _venv_python(virtual_environment)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-input",
                str(wheel),
            ],
            check=True,
            cwd=root,
            env=environment,
        )
        subprocess.run(
            [str(python), "-m", "pip", "check"],
            check=True,
            cwd=root,
            env=environment,
        )
        completed = subprocess.run(
            [str(python), "-I", str(validator), "--profile", profile],
            capture_output=True,
            text=True,
            cwd=root,
            env=environment,
        )
        if completed.returncode != 0:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
            raise subprocess.CalledProcessError(
                completed.returncode,
                completed.args,
                output=completed.stdout,
                stderr=completed.stderr,
            )
    return json.loads(completed.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", nargs="?", type=Path)
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--profile", choices=sorted(PROFILE_CAPABILITIES), required=True)
    parser.add_argument("--require-cuda-device", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        wheel = _select_wheel(args.wheel, args.wheel_dir)
        result = validate_clean_wheel(wheel, args.profile)
        if args.require_cuda_device and not result["cuda"]["available"]:
            raise WheelProfileError("required CUDA device was unavailable in the clean wheel test")
    except (WheelProfileError, subprocess.CalledProcessError) as exc:
        print(f"clean wheel validation failed: {exc}", file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
