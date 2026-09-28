# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import sys
import tarfile
import tomllib
import zipfile
from io import BytesIO
from pathlib import Path
from types import ModuleType

import pytest
from _repository_module import load_repository_module
from packaging.version import Version

_ROOT = Path(__file__).resolve().parents[2]
_TOOLS = _ROOT / "tools" / "artifacts"

sys.path.insert(0, str(_TOOLS))


def _load_tool(name: str) -> ModuleType:
    """Load one repository tool by path, under its own module name."""
    return load_repository_module(name, _TOOLS / f"{name}.py")


_validate_wheel_profile = _load_tool("validate_wheel_profile")
_validate_native_binary = _load_tool("validate_native_binary")
_validate_clean_wheel = _load_tool("validate_clean_wheel")
_validate_installed_artifact = _load_tool("validate_installed_artifact")
_validate_release_dist = _load_tool("validate_release_dist")
_validate_required_pytest = _load_tool("validate_required_pytest")
LICENSE_FILES = _load_tool("artifact_profiles").LICENSE_FILES

_clean_environment = _validate_clean_wheel._clean_environment
_venv_python = _validate_clean_wheel._venv_python

NativeBinaryValidationError = _validate_native_binary.NativeBinaryValidationError
_parse_dumpbin_dependencies = _validate_native_binary._parse_dumpbin_dependencies
_parse_dumpbin_exports = _validate_native_binary._parse_dumpbin_exports
_parse_nm_exports = _validate_native_binary._parse_nm_exports
_parse_readelf_dependencies = _validate_native_binary._parse_readelf_dependencies
validate_audit_report = _validate_native_binary.validate_audit_report
validate_binary_contract = _validate_native_binary.validate_binary_contract

WheelProfileError = _validate_wheel_profile.WheelProfileError
validate_wheel_profile = _validate_wheel_profile.validate_wheel_profile
_typing_contract = _validate_installed_artifact._typing_contract

pytestmark = pytest.mark.python_only

_EXTRAS = ("dev", "docs", "lsl", "nrf", "test")


def _write_wheel(
    path: Path,
    *,
    modules: tuple[str, ...],
    requirements: tuple[str, ...] = ("numpy>=1.23", "scipy>=1.9"),
    extras: tuple[str, ...] = _EXTRAS,
    include_licenses: bool = True,
) -> Path:
    metadata = [
        "Metadata-Version: 2.3",
        "Name: pyneurale",
        "Version: 1.0",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        *(f"Provides-Extra: {extra}" for extra in extras),
        "",
        "",
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pyneurale-1.0.dist-info/METADATA", "\n".join(metadata))
        if include_licenses:
            for license_file in LICENSE_FILES:
                archive.writestr(f"pyneurale-1.0.dist-info/licenses/{license_file}", b"x")
        for module in modules:
            archive.writestr(f"neurale/{module}.cpython-312-x86_64-linux-gnu.so", b"")
    return path


def test_pyproject_freezes_minimal_runtime_and_explicit_extras() -> None:
    project = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert project["dependencies"] == ["numpy>=1.23", "scipy>=1.9"]
    assert set(project["optional-dependencies"]) == set(_EXTRAS)
    assert project["optional-dependencies"]["lsl"] == ["pylsl>=1.17,<2"]


def test_center_out_public_api_has_installed_typing_contract() -> None:
    report = _typing_contract(_ROOT / "src" / "neurale")

    assert report == {
        "pep561_marker": "py.typed",
        "stub": "experiments/center_out.pyi",
        "public_names": 62,
    }


def test_bundled_presentation_uses_supported_x11_profile() -> None:
    dependencies = (_ROOT / "cpp/cmake/ExperimentPresentationDependencies.cmake").read_text(
        encoding="utf-8"
    )

    assert 'set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)' in dependencies
    assert 'set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)' in dependencies


def test_bundled_presentation_disables_optional_freetype_dependencies() -> None:
    dependencies = (_ROOT / "cpp/cmake/ExperimentPresentationDependencies.cmake").read_text(
        encoding="utf-8"
    )

    for option in ("HARFBUZZ", "ZLIB", "PNG", "BZIP2", "BROTLI"):
        assert f'set(FT_DISABLE_{option} ON CACHE BOOL "" FORCE)' in dependencies


def test_bundled_license_texts_cover_native_dependencies() -> None:
    bundle = (_ROOT / "LICENSES_bundled.txt").read_text(encoding="utf-8")
    sections = bundle.split("\n===== ")[1:]
    names = {section.partition(" =====")[0] for section in sections}

    assert names == {
        "pybind11-LICENSE",
        "GLFW-LICENSE.md",
        "FreeType-FTL.TXT",
        "HarfBuzz-COPYING",
        "Intel-oneMKL-license.txt",
        "Intel-oneMKL-2026.0-third-party-programs.txt",
        "Intel-oneMKL-2026.1-third-party-programs.txt",
        "Intel-OpenMP-third-party-programs.txt",
        "Intel-oneAPI-EULA.htm",
    }
    assert all("\nSource: " in section and len(section) > 200 for section in sections)


@pytest.mark.parametrize(
    ("profile", "modules"),
    [
        ("core", ("_native",)),
        ("presentation", ("_native",)),
        ("cuda", ("_native", "_native_cuda")),
        ("release", ("_native", "_native_cuda")),
    ],
)
def test_valid_wheel_profiles_are_accepted(
    tmp_path: Path,
    profile: str,
    modules: tuple[str, ...],
) -> None:
    wheel = _write_wheel(tmp_path / "pyneurale.whl", modules=modules)

    validate_wheel_profile(wheel, profile)


def test_profile_rejects_unexpected_native_module(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale.whl",
        modules=("_native", "_unexpected_native"),
    )

    with pytest.raises(WheelProfileError, match="core profile requires native modules"):
        validate_wheel_profile(wheel, "core")


def test_profile_requires_third_party_license_texts(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale.whl", modules=("_native",), include_licenses=False
    )

    with pytest.raises(WheelProfileError, match="wheel is missing license files"):
        validate_wheel_profile(wheel, "core")


def test_windows_release_profile_requires_one_openmp_runtime(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale-1.0-cp312-cp312-win_amd64.whl",
        modules=("_native", "_native_cuda"),
    )
    with pytest.raises(WheelProfileError, match="one Intel OpenMP runtime"):
        validate_wheel_profile(wheel, "release")

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("neurale/libiomp5md.dll", b"")
    validate_wheel_profile(wheel, "release")

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("pyneurale.libs/libiomp5md-2299b046.dll", b"")
    with pytest.raises(WheelProfileError, match="one Intel OpenMP runtime"):
        validate_wheel_profile(wheel, "release")


def test_linux_release_profile_requires_repaired_runtimes(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale-1.0-cp312-cp312-manylinux_2_39_x86_64.whl",
        modules=("_native", "_native_cuda"),
    )
    with pytest.raises(WheelProfileError, match="bundled libiomp5"):
        validate_wheel_profile(wheel, "release")

    with zipfile.ZipFile(wheel, "a") as archive:
        archive.writestr("pyneurale.libs/libiomp5-19cd02fb.so", b"")
        archive.writestr("pyneurale.libs/libcudart-862020da.so.13.2.86", b"")
    validate_wheel_profile(wheel, "release")


def test_release_dist_requires_exactly_four_platform_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_validate_release_dist, "validate_wheel_profile", lambda *_args: None)
    sdist = tmp_path / "pyneurale-0.1.1.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "CMakeLists.txt",
            "PKG-INFO",
            "pyproject.toml",
            "tools/artifacts/artifact_profiles.py",
            *sorted(LICENSE_FILES),
        ):
            info = tarfile.TarInfo(f"pyneurale-0.1.1/{name}")
            info.size = 1
            archive.addfile(info, BytesIO(b"x"))
    for python in ("cp311", "cp312"):
        for platform in ("win_amd64", "manylinux_2_39_x86_64"):
            (tmp_path / f"pyneurale-0.1.1-{python}-{python}-{platform}.whl").touch()

    report = _validate_release_dist.validate_release_dist(tmp_path, "0.1.1")
    assert len(report["wheels"]) == 4

    (tmp_path / "pyneurale-0.1.1-cp312-cp312-manylinux_2_39_x86_64.whl").rename(
        tmp_path / "pyneurale-0.1.1-cp312-cp312-linux_x86_64.whl"
    )
    with pytest.raises(ValueError, match="unsupported wheel tags"):
        _validate_release_dist.validate_release_dist(tmp_path, "0.1.1")

    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "CMakeLists.txt",
            "PKG-INFO",
            "pyproject.toml",
            "tools/artifacts/artifact_profiles.py",
            *sorted(LICENSE_FILES),
            "AGENTS.md",
        ):
            info = tarfile.TarInfo(f"pyneurale-0.1.1/{name}")
            info.size = 1
            archive.addfile(info, BytesIO(b"x"))
    with pytest.raises(ValueError, match="local build file"):
        _validate_release_dist.validate_release_dist(tmp_path, "0.1.1")


def test_release_sdist_requires_third_party_license_texts(tmp_path: Path) -> None:
    sdist = tmp_path / "pyneurale-0.1.1.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name in (
            "CMakeLists.txt",
            "PKG-INFO",
            "pyproject.toml",
            "tools/artifacts/artifact_profiles.py",
            *(LICENSE_FILES - {"LICENSES_bundled.txt"}),
        ):
            info = tarfile.TarInfo(f"pyneurale-0.1.1/{name}")
            info.size = 1
            archive.addfile(info, BytesIO(b"x"))

    with pytest.raises(ValueError, match=r"LICENSES_bundled\.txt"):
        _validate_release_dist._validate_sdist(sdist, Version("0.1.1"))


def test_release_sdist_rejects_absolute_symlink(tmp_path: Path) -> None:
    sdist = tmp_path / "pyneurale-0.1.1.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        info = tarfile.TarInfo("pyneurale-0.1.1/bin/python")
        info.type = tarfile.SYMTYPE
        info.linkname = "/usr/bin/python3.12"
        archive.addfile(info)

    with pytest.raises(ValueError, match="sdist contains a link"):
        _validate_release_dist._validate_sdist(sdist, Version("0.1.1"))


def test_required_pytest_rejects_skipped_gpu_test(tmp_path: Path) -> None:
    report = tmp_path / "gpu.xml"
    report.write_text(
        '<testsuites><testsuite tests="31" skipped="0" failures="0" errors="0" /></testsuites>',
        encoding="utf-8",
    )
    assert _validate_required_pytest.validate_required_pytest(report, 31) == (31, 0)

    report.write_text(
        '<testsuites><testsuite tests="31" skipped="1" failures="0" errors="0" /></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="skipped=1"):
        _validate_required_pytest.validate_required_pytest(report, 31)


def test_profile_accepts_lsl_as_optional_dependency(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale.whl",
        modules=("_native",),
        requirements=("numpy>=1.23", "scipy>=1.9", 'pylsl>=1.17,<2; extra == "lsl"'),
    )
    validate_wheel_profile(wheel, "core")


def test_profile_rejects_missing_lsl_extra(tmp_path: Path) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale.whl",
        modules=("_native",),
        extras=tuple(extra for extra in _EXTRAS if extra != "lsl"),
    )
    with pytest.raises(WheelProfileError, match="declared extras"):
        validate_wheel_profile(wheel, "core")


@pytest.mark.parametrize("requirement", ["torch>=2", "matplotlib", "PySide6"])
def test_profile_rejects_non_core_runtime_dependency(
    tmp_path: Path,
    requirement: str,
) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale.whl",
        modules=("_native",),
        requirements=("numpy>=1.23", "scipy>=1.9", requirement),
    )

    with pytest.raises(WheelProfileError, match="forbidden core runtime requirements"):
        validate_wheel_profile(wheel, "core")


@pytest.mark.parametrize("extra", ["cuda", "presentation", "torch", "device-vendor"])
def test_profile_rejects_inert_native_capability_extra(tmp_path: Path, extra: str) -> None:
    wheel = _write_wheel(
        tmp_path / "pyneurale.whl",
        modules=("_native",),
        extras=(*_EXTRAS, extra),
    )

    with pytest.raises(WheelProfileError, match="must not be represented"):
        validate_wheel_profile(wheel, "core")


def test_clean_environment_removes_python_path_injection(monkeypatch) -> None:
    monkeypatch.setenv("PYTHONHOME", "foreign-home")
    monkeypatch.setenv("PYTHONPATH", "foreign-path")
    monkeypatch.setenv("VIRTUAL_ENV", "foreign-environment")

    environment = _clean_environment()

    assert "PYTHONHOME" not in environment
    assert "PYTHONPATH" not in environment
    assert "VIRTUAL_ENV" not in environment
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_clean_environment_python_path_is_platform_specific(tmp_path: Path) -> None:
    executable = _venv_python(tmp_path)

    assert executable.name in {"python", "python.exe"}
    assert executable.parent.name in {"bin", "Scripts"}


@pytest.mark.parametrize(
    ("profile", "report"),
    [
        ("core", "libc.so.6 libm.so.6 libstdc++.so.6"),
        ("presentation", "opengl32.dll user32.dll x11.dll"),
        ("cuda", "libcuda.so.1 libcudart.so libstdc++.so.6"),
        ("release", "libcuda.so.1 libiomp5.so libcudart.so libGL.so.1"),
    ],
)
def test_shared_library_report_accepts_profile_dependencies(profile: str, report: str) -> None:
    validate_audit_report(report, profile)


@pytest.mark.parametrize(
    ("profile", "dependency"),
    [
        ("core", "cudart64_130.dll"),
        ("core", "libmkl_rt.so"),
        ("presentation", "libharfbuzz.so"),
        ("presentation", "glfw3.dll"),
        ("cuda", "Qt6Core.dll"),
        ("release", "libharfbuzz.so"),
    ],
)
def test_shared_library_report_rejects_cross_profile_dependency(
    profile: str,
    dependency: str,
) -> None:
    with pytest.raises(NativeBinaryValidationError, match="forbidden dependencies"):
        validate_audit_report(dependency, profile)


def test_native_binary_cli_validates_audit_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    wheel = tmp_path / "pyneurale.whl"
    wheel.touch()
    audit_report = tmp_path / "audit.txt"
    audit_report.write_text("libmkl_rt.so", encoding="utf-8")
    monkeypatch.setattr(_validate_native_binary, "inspect_wheel", lambda *_args: {})

    result = _validate_native_binary.main(
        ["--profile", "core", "--wheel", str(wheel), "--audit-report", str(audit_report)]
    )

    assert result == 1
    assert "shared-library report contains forbidden dependencies" in capsys.readouterr().err


def test_windows_binary_reports_are_parsed() -> None:
    exports = """
          1    0 0000EF70 PyInit__native
    """
    dependencies = """
      Image has the following dependencies:

        python312.dll
        KERNEL32.dll

      Summary
    """

    assert _parse_dumpbin_exports(exports) == ["PyInit__native"]
    assert _parse_dumpbin_dependencies(dependencies) == ["python312.dll", "KERNEL32.dll"]


def test_linux_binary_reports_are_parsed() -> None:
    exports = "PyInit__native T 123 42\n_end B 456 8\n"
    dependencies = """
     0x0000000000000001 (NEEDED) Shared library: [libstdc++.so.6]
     0x0000000000000001 (NEEDED) Shared library: [libc.so.6]
    """

    assert _parse_nm_exports(exports) == ["PyInit__native", "_end"]
    assert _parse_readelf_dependencies(dependencies) == ["libstdc++.so.6", "libc.so.6"]


@pytest.mark.parametrize(
    ("module", "profile", "platform", "dependencies"),
    [
        ("_native", "core", "windows", ["python312.dll", "KERNEL32.dll", "VCOMP140.dll"]),
        ("_native", "core", "linux", ["libstdc++.so.6", "libgomp.so.1", "libc.so.6"]),
        (
            "_native",
            "presentation",
            "windows",
            ["python311.dll", "OPENGL32.dll", "USER32.dll"],
        ),
        (
            "_native",
            "presentation",
            "linux",
            ["libstdc++.so.6", "libOpenGL.so.0", "libX11.so.6"],
        ),
        ("_native_cuda", "cuda", "linux", ["libcudart.so.13", "libstdc++.so.6"]),
        (
            "_native",
            "release",
            "windows",
            ["python312.dll", "OPENGL32.dll", "libiomp5md.dll"],
        ),
        (
            "_native",
            "release",
            "linux",
            ["libcudart-862020da.so.13.2.86", "libOpenGL-9a0a6024.so.0.0.0"],
        ),
    ],
)
def test_native_binary_contract_accepts_owned_dependencies(
    module: str,
    profile: str,
    platform: str,
    dependencies: list[str],
) -> None:
    validate_binary_contract(
        module=module,
        profile=profile,
        platform=platform,
        exports=[f"PyInit_{module}"],
        dependencies=dependencies,
    )


def test_native_binary_contract_rejects_extra_export() -> None:
    with pytest.raises(NativeBinaryValidationError, match="must export only"):
        validate_binary_contract(
            module="_native",
            profile="core",
            platform="linux",
            exports=["PyInit__native", "neurale_internal_kernel"],
            dependencies=["libc.so.6"],
        )


@pytest.mark.parametrize("dependency", ["glfw3.dll", "vendor_acquisition_sdk.dll"])
def test_core_binary_contract_rejects_graphics_and_vendor_dependencies(dependency: str) -> None:
    with pytest.raises(NativeBinaryValidationError, match="outside the core/windows boundary"):
        validate_binary_contract(
            module="_native",
            profile="core",
            platform="windows",
            exports=["PyInit__native"],
            dependencies=["python312.dll", dependency],
        )


def test_presentation_binary_rejects_dynamic_font_dependency() -> None:
    with pytest.raises(
        NativeBinaryValidationError,
        match="outside the presentation/linux boundary",
    ):
        validate_binary_contract(
            module="_native",
            profile="presentation",
            platform="linux",
            exports=["PyInit__native"],
            dependencies=["libOpenGL.so.0", "libharfbuzz.so.0"],
        )


def test_release_binary_rejects_unowned_repaired_library() -> None:
    with pytest.raises(NativeBinaryValidationError, match="outside the release/linux boundary"):
        validate_binary_contract(
            module="_native",
            profile="release",
            platform="linux",
            exports=["PyInit__native"],
            dependencies=["libvendor-9a0a6024.so.1"],
        )


def test_release_cuda_extension_rejects_graphics_dependency() -> None:
    with pytest.raises(NativeBinaryValidationError, match="outside the release/linux boundary"):
        validate_binary_contract(
            module="_native_cuda",
            profile="release",
            platform="linux",
            exports=["PyInit__native_cuda"],
            dependencies=["libOpenGL-9a0a6024.so.0.0.0"],
        )
