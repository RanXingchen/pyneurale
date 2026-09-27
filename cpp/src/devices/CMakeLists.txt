# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(neurale_devices STATIC simulation.cpp provider.cpp)
add_library(neurale::devices ALIAS neurale_devices)

target_include_directories(
    neurale_devices
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src/neurale/devices/sdk>"
)
target_link_libraries(
    neurale_devices
    PUBLIC
    neurale::streaming
    neurale::signal
    ${CMAKE_DL_LIBS}
)
