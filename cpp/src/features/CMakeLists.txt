# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(
    neurale_features
    STATIC
    online.cpp
)
add_library(neurale::features ALIAS neurale_features)

target_compile_features(neurale_features PUBLIC cxx_std_20)
target_include_directories(
    neurale_features
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
)
target_link_libraries(neurale_features PRIVATE neurale::signal)
