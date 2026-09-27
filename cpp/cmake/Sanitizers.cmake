# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Apply compile flags to every sanitized compile target and link flags to every
# sanitized link target. The compile flags are wrapped in a CXX-language guard so
# CUDA sources are left untouched; the link flags apply verbatim. Both flag
# lists are read from the calling neurale_configure_sanitizers scope.
function(_neurale_apply_sanitizer_flags)
    cmake_parse_arguments(PARSE_ARGV 0 _arg "" "" "COMPILE;LINK")
    foreach(target IN LISTS _neurale_sanitized_compile_targets)
        target_compile_options(${target}
            PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:${_arg_COMPILE}>
        )
    endforeach()
    foreach(target IN LISTS _neurale_sanitized_link_targets)
        target_link_options(${target} PRIVATE ${_arg_LINK})
    endforeach()
endfunction()

function(neurale_configure_sanitizers)
    if(NOT (NEURALE_ENABLE_ASAN OR NEURALE_ENABLE_UBSAN OR
            NEURALE_ENABLE_LSAN OR NEURALE_ENABLE_TSAN))
        return()
    endif()

    set(_neurale_sanitized_compile_targets
        neurale_runtime
        neurale_streaming
        neurale_signal
        neurale_devices
        neurale_features
        neurale_pipeline
        neurale_recording
        neurale_recording_spool
        neurale_recording_replay
        neurale_models
        neurale_sorting
        neurale_experiments
        neurale_execution
        _native
        ${NEURALE_NATIVE_TEST_TARGETS}
    )
    set(_neurale_sanitized_link_targets
        _native
        ${NEURALE_NATIVE_TEST_TARGETS}
    )

    foreach(target IN ITEMS neurale_runtime_cuda neurale_models_cuda _native_cuda)
        if(TARGET ${target})
            list(APPEND _neurale_sanitized_compile_targets ${target})
        endif()
    endforeach()

    foreach(target IN ITEMS
            neurale_experiment_presentation
            neurale_experiment_presentation_center_out_test
            neurale_experiment_presentation_dependency_test
            neurale_experiment_presentation_speech_test
            neurale_experiment_presentation_ssvep_test
            neurale_experiment_presentation_surface_test
            neurale_experiment_presentation_webgrid_test)
        if(TARGET ${target})
            list(APPEND _neurale_sanitized_compile_targets ${target})
        endif()
    endforeach()
    foreach(target IN ITEMS
            neurale_experiment_presentation_dependency_test
            neurale_experiment_presentation_center_out_test
            neurale_experiment_presentation_speech_test
            neurale_experiment_presentation_ssvep_test
            neurale_experiment_presentation_surface_test
            neurale_experiment_presentation_webgrid_test)
        if(TARGET ${target})
            list(APPEND _neurale_sanitized_link_targets ${target})
        endif()
    endforeach()
    if(TARGET _native_cuda)
        list(APPEND _neurale_sanitized_link_targets _native_cuda)
    endif()

    foreach(target IN ITEMS
            neurale_streaming_benchmark
            neurale_pipeline_multitaper_bandpower_benchmark
            neurale_pipeline_strict_realtime_benchmark)
        if(TARGET ${target})
            list(APPEND _neurale_sanitized_compile_targets ${target})
            list(APPEND _neurale_sanitized_link_targets ${target})
        endif()
    endforeach()

    if(TARGET neurale_streaming_soak_test)
        target_compile_definitions(
            neurale_streaming_soak_test
            PRIVATE
            NEURALE_SANITIZED_BUILD=1
        )
    endif()

    if(NEURALE_ENABLE_TSAN)
        if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" OR MSVC)
            message(FATAL_ERROR
                "NEURALE_ENABLE_TSAN requires GCC or Clang on a TSAN-supported platform"
            )
        endif()
        _neurale_apply_sanitizer_flags(
            COMPILE "-fsanitize=thread" "-fno-omit-frame-pointer"
            LINK   "-fsanitize=thread"
        )
        return()
    endif()

    set(_neurale_memory_sanitizers "")
    if(NEURALE_ENABLE_ASAN)
        list(APPEND _neurale_memory_sanitizers address)
    endif()
    if(NEURALE_ENABLE_UBSAN)
        if(MSVC)
            message(FATAL_ERROR "NEURALE_ENABLE_UBSAN is not supported by MSVC")
        endif()
        list(APPEND _neurale_memory_sanitizers undefined)
    endif()
    if(NEURALE_ENABLE_LSAN)
        if(MSVC OR APPLE)
            message(FATAL_ERROR "NEURALE_ENABLE_LSAN is not supported on this platform")
        endif()
        list(APPEND _neurale_memory_sanitizers leak)
    endif()

    if(MSVC)
        # Only address is reachable here: UBSAN/LSAN fatally error on MSVC above.
        _neurale_apply_sanitizer_flags(
            COMPILE "/fsanitize=address"
            LINK   "/INCREMENTAL:NO"
        )
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        list(JOIN _neurale_memory_sanitizers "," _neurale_sanitizer_list)
        set(_neurale_sanitizer_flag "-fsanitize=${_neurale_sanitizer_list}")
        _neurale_apply_sanitizer_flags(
            COMPILE "${_neurale_sanitizer_flag}" "-fno-omit-frame-pointer"
            LINK   "${_neurale_sanitizer_flag}"
        )
    else()
        message(FATAL_ERROR "Requested sanitizers require MSVC, GCC, or Clang")
    endif()
endfunction()
