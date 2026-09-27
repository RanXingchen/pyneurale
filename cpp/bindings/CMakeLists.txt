# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

function(neurale_configure_python_extension_surface target)
    # A wheel publishes Python module initializers, not the C++ symbols pulled
    # into each aggregate module from its static implementation libraries.
    # Keep this explicit even though pybind11 currently applies hidden
    # visibility itself, so a pybind11/CMake default change cannot widen the
    # binary surface silently.
    set_target_properties(
        ${target}
        PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON
        WINDOWS_EXPORT_ALL_SYMBOLS OFF
    )
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Hidden compile visibility does not affect objects already compiled
        # into the static domain archives. A linker version script closes that
        # second path and makes the ELF dynamic symbol table match the Python
        # module boundary exactly.
        set(_export_script "${CMAKE_CURRENT_BINARY_DIR}/${target}_exports.map")
        file(GENERATE
            OUTPUT "${_export_script}"
            CONTENT "{\n  global: PyInit_${target};\n  local: *;\n};\n"
        )
        target_link_options(
            ${target}
            PRIVATE "LINKER:--version-script=${_export_script}"
        )
        set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS "${_export_script}")
    endif()
endfunction()

pybind11_add_module(
    _native
    MODULE
    native_bindings.cpp
    runtime_bindings.cpp
    streaming/module_bindings.cpp
    streaming/array_replay_bindings.cpp
    streaming/observer_bridge.cpp
    streaming/adapters.cpp
    pipeline_bindings.cpp
    recording/module_bindings.cpp
    recording/replay_image_bindings.cpp
    recording/spool_bindings.cpp
    devices/module_bindings.cpp
    devices/provider_bindings.cpp
    signal/module_bindings.cpp
    signal/simulation_bindings.cpp
    signal/filtering_bindings.cpp
    signal/filter_design_bindings.cpp
    signal/spatial_bindings.cpp
    signal/representation_bindings.cpp
    signal/resampling_bindings.cpp
    signal/spectral_bindings.cpp
    signal/transform_bindings.cpp
    signal/windows_bindings.cpp
    features/module_bindings.cpp
    features/online_bindings.cpp
    models/module_bindings.cpp
    models/decomposition_bindings.cpp
    models/linear_model_bindings.cpp
    models/state_space_bindings.cpp
    models/classification_bindings.cpp
    models/neighbors_bindings.cpp
    models/density_bindings.cpp
    models/alignment_bindings.cpp
    models/manifold_bindings.cpp
    sorting/module_bindings.cpp
    sorting/detection_bindings.cpp
    sorting/online_detection_bindings.cpp
    sorting/valley_seeking_bindings.cpp
    experiments/module_bindings.cpp
    experiments/contract_bindings.cpp
    experiments/assistance_bindings.cpp
    experiments/center_out_bindings.cpp
    experiments/ssvep_bindings.cpp
    experiments/speech_bindings.cpp
    experiments/webgrid_bindings.cpp
)
neurale_configure_python_extension_surface(_native)
target_link_libraries(
    _native
    PRIVATE
    neurale::runtime
    neurale::streaming
    # Internal, unexported targets (no public alias by design -- see
    # cpp/src/recording/CMakeLists.txt). The extension module is inside this
    # repository, so it is allowed to link them; nothing outside can.
    neurale_recording
    neurale_recording_spool
    # The replay reader stays a separate library from the recorder on purpose
    # (cpp/src/recording/CMakeLists.txt): an image has nothing to say to the
    # recorder that wrote it. Linking both *here* does not weaken that -- the
    # extension module aggregates every domain by construction, and the
    # libraries themselves still cannot reach each other.
    neurale_recording_replay
    neurale::signal
    neurale::devices
    neurale::features
    neurale::models
    neurale::sorting
    neurale_pipeline
    neurale::experiments
    neurale_execution
)
if(NEURALE_CUDA_FOUND)
    target_compile_definitions(_native PRIVATE NEURALE_WITH_CUDA_EXTENSION=1)
endif()
if(NEURALE_CUDA_KERNELS_FOUND)
    target_sources(_native PRIVATE models/cuda_bindings.cpp)
    target_compile_definitions(
        _native
        PRIVATE
        NEURALE_MODELS_WITH_CUDA=1
    )
    target_link_libraries(_native PRIVATE neurale::models_cuda)
endif()
if(NEURALE_MKL_FOUND)
    target_compile_definitions(
        _native
        PRIVATE
        NEURALE_SIGNAL_WITH_MKL=1
        NEURALE_MODELS_WITH_MKL=1
    )
endif()
target_include_directories(
    _native
    PRIVATE
    "${PROJECT_SOURCE_DIR}/cpp/src/signal/filtering"
    "${PROJECT_SOURCE_DIR}/cpp/src/signal/spectral"
    "${PROJECT_SOURCE_DIR}/cpp/src/models"
    "${PROJECT_SOURCE_DIR}/cpp/src/pipeline"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
    "${PROJECT_SOURCE_DIR}/cpp/src/recording/replay"
    "${PROJECT_SOURCE_DIR}/cpp/src/experiments"
    "${PROJECT_SOURCE_DIR}/cpp/src/execution"
)
install(TARGETS _native LIBRARY DESTINATION neurale RUNTIME DESTINATION neurale)

if(NEURALE_ENABLE_EXPERIMENT_PRESENTATION STREQUAL "ON")
    target_sources(
        _native PRIVATE
        experiments/presentation_bindings.cpp
    )
    target_compile_definitions(_native PRIVATE NEURALE_WITH_EXPERIMENT_PRESENTATION=1)
    target_include_directories(
        _native
        PRIVATE
        "${PROJECT_SOURCE_DIR}/cpp/src/presentation"
    )
    target_link_libraries(
        _native
        PRIVATE
        neurale_experiment_presentation
        neurale_experiment_presentation_recording
    )
endif()

if(NEURALE_CUDA_FOUND)
    pybind11_add_module(
        _native_cuda
        MODULE
        cuda_runtime_bindings.cpp
    )
    neurale_configure_python_extension_surface(_native_cuda)
    target_link_libraries(_native_cuda PRIVATE neurale::runtime_cuda)
    install(
        TARGETS _native_cuda
        LIBRARY DESTINATION neurale
        RUNTIME DESTINATION neurale
    )
endif()
