# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(
    neurale_sorting
    STATIC
    detection.cpp
    online_detection.cpp
    valley_seeking.cpp
)
add_library(neurale::sorting ALIAS neurale_sorting)

target_compile_features(neurale_sorting PUBLIC cxx_std_20)
target_include_directories(
    neurale_sorting
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
)

set_source_files_properties(
    detection.cpp
    online_detection.cpp
    valley_seeking.cpp
    PROPERTIES
    COMPILE_OPTIONS "${NEURALE_NO_FAST_MATH_COMPILE_OPTIONS}"
)
