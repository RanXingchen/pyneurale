#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

set -euo pipefail

repo=$(realpath "${1:?repository path required}")
output=$(realpath -m "${2:?output directory required}")
version=${3:?release version required}
evidence=$(realpath -m "${4:?evidence directory required}")
export PATH="/usr/local/cuda/bin:$PATH"

if [[ $(uname -m) != x86_64 ]]; then
    echo "Linux release builds require x86-64" >&2
    exit 1
fi
if ! command -v nvcc >/dev/null; then
    echo "CUDA nvcc is required" >&2
    exit 1
fi
if [[ ! -f /opt/intel/oneapi/mkl/latest/lib/cmake/mkl/MKLConfig.cmake ]]; then
    echo "Intel oneMKL development package is required" >&2
    exit 1
fi

export LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/latest/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export CMAKE_ARGS='-DNEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON -DNEURALE_ENABLE_MKL=ON -DNEURALE_ENABLE_CUDA=ON -DFETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER'
mkdir -p "$output"
mkdir -p "$evidence"

for minor in 11 12; do
    if [[ $minor == 11 ]]; then
        interpreter="$HOME/.pyenv/versions/3.11.16/bin/python3.11"
    else
        interpreter="$(command -v python3.12)"
    fi
    if [[ ! -x $interpreter ]]; then
        echo "CPython 3.$minor is required at $interpreter" >&2
        exit 1
    fi
    work="$HOME/.cache/pyneurale-release-build/$version/linux-py3.$minor"
    mkdir -p "$work"
    "$interpreter" -m venv "$work/venv"
    export PATH="$work/venv/bin:/usr/local/cuda/bin:$PATH"
    python -m pip install --upgrade pip 'cmake>=3.24,<4.4' pybind11 auditwheel patchelf packaging
    mkdir -p "$work/raw"
    python -m pip wheel "$repo" --no-deps --wheel-dir "$work/raw" \
        --config-settings="build-dir=$work/build"
    shopt -s nullglob
    raw=("$work/raw"/*.whl)
    if [[ ${#raw[@]} -ne 1 ]]; then
        echo "Expected one raw Linux wheel for CPython 3.$minor" >&2
        exit 1
    fi
    auditwheel show "${raw[0]}"
    auditwheel repair --plat manylinux_2_39_x86_64 -w "$output" "${raw[0]}"
    repaired=("$output"/pyneurale-"$version"-cp3"$minor"-cp3"$minor"-manylinux_2_39_x86_64.whl)
    if [[ ! -f ${repaired[0]} ]]; then
        echo "Repaired Linux wheel has an unexpected name/version" >&2
        exit 1
    fi
    auditwheel show "${repaired[0]}"
    python "$repo/tools/artifacts/validate_wheel_profile.py" --profile release "${repaired[0]}"
    python "$repo/tools/artifacts/validate_native_binary.py" --profile release --wheel "${repaired[0]}" \
        --report "$evidence/linux-py3.$minor-native.json"
    python "$repo/tools/artifacts/validate_clean_wheel.py" --profile release --require-cuda-device "${repaired[0]}" \
        --report "$evidence/linux-py3.$minor-clean.json"
    python -m pip install "${repaired[0]}[test]"
    cd "$repo"
    gpu_report="$evidence/linux-py3.$minor-gpu.xml"
    python -m pytest -o 'addopts=--require-native -ra' -m gpu \
        --deselect=tests/unit/models/test_neighbors.py::test_knn_cuda_restores_previous_device \
        --junitxml="$gpu_report" -q \
        tests/integration/runtime/test_native_extension.py \
        tests/unit/models/test_neighbors.py \
        tests/unit/models/test_density.py \
        tests/unit/models/test_manifold.py
    python "$repo/tools/artifacts/validate_required_pytest.py" "$gpu_report" --minimum 31
    if [[ $minor == 12 ]]; then
        window_report="$evidence/linux-py3.$minor-window.xml"
        xvfb-run -a env \
            NEURALE_PRESENTATION_TEST_FONT=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
            NEURALE_PRESENTATION_TEST_CJK_FONT=/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc \
            NEURALE_PRESENTATION_REQUIRE_DISPLAY=1 \
            python -m pytest -o 'addopts=--require-native -ra' \
            --junitxml="$window_report" -q \
            tests/integration/experiments/test_center_out_closed_loop.py::test_center_out_presentation
        python "$repo/tools/artifacts/validate_required_pytest.py" "$window_report" --minimum 1
    fi
done
