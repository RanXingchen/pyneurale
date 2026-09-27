# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(
    neurale_streaming
    STATIC
    array_replay.cpp
    buffer_pool.cpp
    clock.cpp
    continuity.cpp
    discontinuity_pool.cpp
    frame_validation.cpp
    linear_processor_chain.cpp
    terminal_frame_emitter.cpp
    realtime_config.cpp
    realtime_thread.cpp
    runtime.cpp
    safety.cpp
    schema.cpp
)
add_library(neurale::streaming ALIAS neurale_streaming)

find_package(Threads REQUIRED)

target_compile_features(neurale_streaming PUBLIC cxx_std_20)

target_link_libraries(neurale_streaming PRIVATE Threads::Threads)

target_include_directories(
    neurale_streaming
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
)

# Deterministic lifecycle race and thread-start failure injection. The hook
# surface and branches are absent when native tests are not built.
if(NEURALE_BUILD_CPP_TESTS)
    target_compile_definitions(neurale_streaming PUBLIC NEURALE_STREAMING_TEST_HOOKS=1)
endif()
