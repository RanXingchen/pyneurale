# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Internal integration seam for algorithm-to-streaming adapters. This target
# deliberately has no public alias, public headers, install(), or export().
add_library(
    neurale_pipeline
    STATIC
    bad_channel_removal_adapter.cpp
    fir_filter_adapter.cpp
    feature_stack_adapter.cpp
    fitted_feature_contract.cpp
    hilbert_envelope_adapter.cpp
    iir_filter_adapter.cpp
    target_anchor.cpp
    kalman_decoder_adapter.cpp
    linear_decoder_adapter.cpp
    interval_mean_processor.cpp
    lmp_feature_adapter.cpp
    multitaper_bandpower_adapter.cpp
    resampler_adapter.cpp
    sos_filter_adapter.cpp
    spatial_reference_adapter.cpp
    spike_detector_adapter.cpp
)

set_property(TARGET neurale_pipeline PROPERTY NEURALE_INTERNAL_ONLY TRUE)
target_compile_features(neurale_pipeline PRIVATE cxx_std_20)
target_include_directories(
    neurale_pipeline
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${PROJECT_SOURCE_DIR}/cpp/src/signal/filtering"
)
target_link_libraries(
    neurale_pipeline
    PRIVATE
    neurale::streaming
    neurale::signal
    neurale::features
    neurale::models
    neurale::sorting
)
