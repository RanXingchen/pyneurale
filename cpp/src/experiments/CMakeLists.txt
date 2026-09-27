# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# The shared experiment value contract. It links nothing: assistance and the
# paradigms must be usable without acquiring a dependency on the streaming
# runtime, and a value contract that linked one would hand them that dependency
# transitively. neurale_assert_experiments_dependencies() enforces it.
add_library(
    neurale_experiments STATIC
    abnormal.cpp
    assistance.cpp
    center_out.cpp
    center_out_guidance.cpp
    center_out_replay.cpp
    contract.cpp
    replay.cpp
    ssvep.cpp
    speech.cpp
    speech_replay.cpp
    validation.cpp
    webgrid.cpp
    webgrid_replay.cpp
)

add_library(neurale::experiments ALIAS neurale_experiments)

target_include_directories(
    neurale_experiments
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
)
