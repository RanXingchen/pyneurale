# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

add_library(
    neurale_signal
    STATIC
    windows/basic.cpp
    windows/dpss.cpp
    filtering/fir_design.cpp
    filtering/fir_builtin.cpp
    filtering/iir_builtin.cpp
    filtering/fir_dispatch.cpp
    filtering/fir_realtime.cpp
    filtering/iir_dispatch.cpp
    filtering/iir_zero_phase.cpp
    filtering/iir_realtime.cpp
    filtering/sos_realtime.cpp
    filtering/iir_design.cpp
    filtering/representations.cpp
    filtering/spatial.cpp
    filtering/spatial_realtime.cpp
    neural_simulation.cpp
    resample.cpp
    simulation.cpp
    spectral/builtin/fft.cpp
    spectral/fft_dispatch.cpp
    spectral/czt.cpp
    spectral/analytic.cpp
    spectral/multitaper.cpp
    spectral/multitaper_builtin.cpp
    spectral/fft_integrate.cpp
)
add_library(neurale::signal ALIAS neurale_signal)

set_source_files_properties(
    neural_simulation.cpp
    simulation.cpp
    PROPERTIES COMPILE_OPTIONS "${NEURALE_NO_FAST_MATH_COMPILE_OPTIONS}"
)

target_include_directories(
    neurale_signal
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/cpp/include>"
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/filtering"
    "${PROJECT_SOURCE_DIR}/cpp/src"
)
target_link_libraries(
    neurale_signal
    PRIVATE
    neurale::runtime
)

if(NEURALE_MKL_FOUND)
    target_sources(
        neurale_signal
        PRIVATE
        filtering/fir_mkl.cpp
        filtering/iir_mkl.cpp
        spectral/mkl/fft.cpp
        spectral/multitaper_mkl.cpp
    )
    target_compile_definitions(
        neurale_signal
        PRIVATE
        NEURALE_SIGNAL_WITH_MKL=1
    )
    target_link_libraries(
        neurale_signal
        PRIVATE
        MKL::MKL
    )
    message(STATUS "PyNeurale signal FIR backend: builtin-direct, mkl-vsl-direct, mkl-vsl-fft")
    message(STATUS "PyNeurale signal IIR backend: builtin-df2t, mkl-blas-df2t")
    message(STATUS "PyNeurale signal FFT backend: builtin-fft, mkl-dfti")
else()
    message(STATUS "PyNeurale signal FIR backend: builtin-direct")
    message(STATUS "PyNeurale signal IIR backend: builtin-df2t")
    message(STATUS "PyNeurale signal FFT backend: builtin-fft")
endif()
