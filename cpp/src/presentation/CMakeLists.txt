# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Private implementation boundary: not installed or exported. The separately
# installed Python extension is the only consumer; concrete paradigm
# presenters stay private implementation details.
add_library(
    neurale_experiment_presentation
    STATIC
    center_out_presenter.cpp
    coordinate_mapper.cpp
    dependency_versions.cpp
    font_identity.cpp
    input_runtime.cpp
    surface.cpp
    speech_presenter.cpp
    ssvep_presenter.cpp
    text_atlas.cpp
    time_mapper.cpp
    webgrid_presenter.cpp
)
set_property(
    TARGET neurale_experiment_presentation
    PROPERTY NEURALE_INTERNAL_ONLY TRUE
)
target_compile_features(neurale_experiment_presentation PRIVATE cxx_std_20)
target_include_directories(
    neurale_experiment_presentation
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    # Shared UTF-8 decoder header (utf8.h) from experiments/. The text atlas
    # validates catalog text with the same implementation the stimulus
    # validator and the Python constructor use, avoiding a second definition
    # of well-formed UTF-8.
    "${PROJECT_SOURCE_DIR}/cpp/src/experiments"
)
target_link_libraries(
    neurale_experiment_presentation
    PRIVATE
    neurale::experiments
    ${NEURALE_PRESENTATION_GLFW_TARGET}
    ${NEURALE_PRESENTATION_OPENGL_TARGET}
    ${NEURALE_PRESENTATION_FREETYPE_TARGET}
    ${NEURALE_PRESENTATION_HARFBUZZ_TARGET}
)

# Concrete seam between presentation and the trace bridge, kept as a
# separate target so the render target itself never acquires recording
# dependencies; the render loop only makes bounded calls into this seam
# when orchestration explicitly enables recording.
add_library(
    neurale_experiment_presentation_recording
    STATIC
    recording_bridge.cpp
)
set_property(
    TARGET neurale_experiment_presentation_recording
    PROPERTY NEURALE_INTERNAL_ONLY TRUE
)
target_compile_features(neurale_experiment_presentation_recording PRIVATE cxx_std_20)
target_include_directories(
    neurale_experiment_presentation_recording
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/cpp/src/execution"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
)
target_link_libraries(
    neurale_experiment_presentation_recording
    PRIVATE
    neurale_experiment_presentation
    neurale_execution
    neurale::experiments
    neurale::streaming
)
