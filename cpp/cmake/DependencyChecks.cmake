# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

function(_neurale_require_direct_links target)
    get_target_property(_links ${target} LINK_LIBRARIES)
    if(NOT _links)
        set(_links "")
    endif()
    foreach(_required IN LISTS ARGN)
        list(FIND _links "${_required}" _idx)
        if(_idx EQUAL -1)
            message(FATAL_ERROR
                "${target} must directly link the internal dependency ${_required}"
            )
        endif()
    endforeach()
endfunction()

# Exactly these links and no others. _neurale_forbid_direct_links matches by
# substring, so it cannot say "not neurale_recording" about a target that links
# neurale_recording_spool; stating the whole list is both stronger and the only
# way to say it.
function(_neurale_require_exact_links target)
    get_target_property(_links ${target} LINK_LIBRARIES)
    if(NOT _links)
        set(_links "")
    endif()
    set(_expected ${ARGN})
    list(SORT _links)
    list(SORT _expected)
    if(NOT "${_links}" STREQUAL "${_expected}")
        message(FATAL_ERROR
            "${target} must link exactly [${_expected}] and links [${_links}]"
        )
    endif()
endfunction()

# pybind11_add_module contributes its own Python/pybind11 support targets. They
# are implementation machinery rather than PyNeurale domain dependencies, so
# remove only those known entries before applying the same exact-set rule.
function(_neurale_require_exact_extension_links target)
    get_target_property(_links ${target} LINK_LIBRARIES)
    if(NOT _links)
        set(_links "")
    endif()
    set(_domain_links "")
    foreach(_link IN LISTS _links)
        if(NOT _link MATCHES "(pybind11::|Python::)")
            list(APPEND _domain_links "${_link}")
        endif()
    endforeach()
    set(_expected ${ARGN})
    list(SORT _domain_links)
    list(SORT _expected)
    if(NOT "${_domain_links}" STREQUAL "${_expected}")
        message(FATAL_ERROR
            "${target} must link exactly the domain targets [${_expected}] "
            "and links [${_domain_links}]"
        )
    endif()
endfunction()

function(_neurale_forbid_direct_links target)
    get_target_property(_links ${target} LINK_LIBRARIES)
    if(NOT _links)
        set(_links "")
    endif()
    get_target_property(_interface_links ${target} INTERFACE_LINK_LIBRARIES)
    if(_interface_links)
        list(APPEND _links ${_interface_links})
    endif()
    foreach(_forbidden IN LISTS ARGN)
        foreach(_link IN LISTS _links)
            string(FIND "${_link}" "${_forbidden}" _idx)
            if(NOT _idx EQUAL -1)
                message(FATAL_ERROR
                    "${target} must not link the higher-level target ${_forbidden}"
                )
            endif()
        endforeach()
    endforeach()
endfunction()

function(_neurale_require_no_public_surface target)
    foreach(_property IN ITEMS INTERFACE_INCLUDE_DIRECTORIES PUBLIC_HEADER)
        get_target_property(_value ${target} ${_property})
        if(_value)
            message(FATAL_ERROR
                "${target} is internal-only and must not set ${_property}"
            )
        endif()
    endforeach()
endfunction()

function(_neurale_forbid_domain_includes directory)
    file(GLOB_RECURSE _files
        "${directory}/*.cpp"
        "${directory}/*.h"
    )
    foreach(_file IN LISTS _files)
        file(READ "${_file}" _contents)
        if(_contents MATCHES
           "#[ \t]*include[ \t]*[<\"]neurale/(signal|features|models|sorting)/")
            message(FATAL_ERROR
                "source has a forbidden algorithm-domain dependency: ${_file}"
            )
        endif()
    endforeach()
endfunction()

# Non-recursive on purpose: the spool container is the top level of
# cpp/src/recording, and cpp/src/recording/core below it is the recorder core,
# which is allowed a streaming dependency the container is not.
function(_neurale_forbid_neurale_and_python_includes directory)
    file(GLOB _files
        "${directory}/*.cpp"
        "${directory}/*.h"
    )
    foreach(_file IN LISTS _files)
        file(READ "${_file}" _contents)
        if(_contents MATCHES "#[ \t]*include[ \t]*[<\"]neurale/")
            message(FATAL_ERROR
                "spool container source must not depend on a neurale domain: ${_file}"
            )
        endif()
        if(_contents MATCHES "#[ \t]*include[ \t]*[<\"](Python\\.h|pybind11/)")
            message(FATAL_ERROR
                "spool container source must not reach Python: ${_file}"
            )
        endif()
    endforeach()
endfunction()

function(_neurale_forbid_python_includes directory)
    file(GLOB_RECURSE _files
        "${directory}/*.cpp"
        "${directory}/*.h"
    )
    foreach(_file IN LISTS _files)
        file(READ "${_file}" _contents)
        if(_contents MATCHES "#[ \t]*include[ \t]*[<\"](Python\\.h|pybind11/)")
            message(FATAL_ERROR
                "source must not reach Python: ${_file}"
            )
        endif()
    endforeach()
endfunction()

# Recursive, and refuses every quoted include the allowlist does not name. The
# angle-bracket rules above police public headers under neurale/, which is the
# whole surface most targets have. A target that links an internal-only
# neighbour has no public headers to police: it includes that neighbour's
# private ones by path, and the only way to say which of them it may reach is to
# say so. Trailing arguments are the permitted basenames.
function(_neurale_restrict_quoted_includes directory)
    set(_allowed ${ARGN})
    file(GLOB_RECURSE _files "${directory}/*.cpp" "${directory}/*.h")
    foreach(_file IN LISTS _files)
        file(READ "${_file}" _contents)
        string(REGEX MATCHALL "#[ \t]*include[ \t]*\"[^\"]+\"" _includes "${_contents}")
        foreach(_include IN LISTS _includes)
            string(REGEX REPLACE "^.*\"([^\"]+)\".*$" "\\1" _name "${_include}")
            if(NOT _name IN_LIST _allowed)
                message(FATAL_ERROR
                    "${_file} includes \"${_name}\", which is not in the permitted set "
                    "[${_allowed}]"
                )
            endif()
        endforeach()
    endforeach()
endfunction()

# Recursive, and refuses every neurale domain except the directory's own. Used
# by a domain that must depend on no other: naming the forbidden domains one by
# one would silently pass the next domain somebody adds.
function(_neurale_forbid_foreign_domain_includes directory own_domain)
    file(GLOB_RECURSE _files
        "${directory}/*.cpp"
        "${directory}/*.h"
    )
    foreach(_file IN LISTS _files)
        file(READ "${_file}" _contents)
        string(REGEX MATCHALL "#[ \t]*include[ \t]*[<\"]neurale/[a-z_]+/" _includes "${_contents}")
        foreach(_include IN LISTS _includes)
            if(NOT _include MATCHES "neurale/${own_domain}/")
                message(FATAL_ERROR
                    "${own_domain} source must depend on no other neurale domain: ${_file}"
                )
            endif()
        endforeach()
    endforeach()
endfunction()

# The experiment schedule contract is a pinned, stateless sampler. A source that
# reached <random>, std::random_device, rand(), or a library generator would
# produce values no replay can regenerate, and the failure would be invisible
# until somebody tried to reproduce a session. Comments are stripped first: this
# is a rule about code, and the contract's own documentation has to be able to
# name the facilities it excludes.
function(_neurale_forbid_ambient_randomness directory)
    file(GLOB_RECURSE _files
        "${directory}/*.cpp"
        "${directory}/*.h"
    )
    foreach(_file IN LISTS _files)
        file(READ "${_file}" _contents)
        string(REGEX REPLACE "/\\*[^*]*\\*+([^/*][^*]*\\*+)*/" "" _code "${_contents}")
        string(REGEX REPLACE "//[^\n]*" "" _code "${_code}")
        if(_code MATCHES "#[ \t]*include[ \t]*[<\"](random|cstdlib)[>\"]")
            message(FATAL_ERROR
                "experiment source must not reach an ambient random or C library source: ${_file}"
            )
        endif()
        if(_code MATCHES "(random_device|mt19937|ranlux|knuth_b|minstd_rand|[^A-Za-z_]s?rand[ \t]*\\()")
            message(FATAL_ERROR
                "experiment source must not use ambient randomness: ${_file}"
            )
        endif()
    endforeach()
endfunction()

function(_neurale_forbid_device_includes directory)
    file(GLOB_RECURSE _files
        "${directory}/*.cpp"
        "${directory}/*.h"
    )
    foreach(_file IN LISTS _files)
        file(READ "${_file}" _contents)
        if(_contents MATCHES "#[ \t]*include[ \t]*[<\"]neurale/devices/")
            message(FATAL_ERROR
                "lower-level source must not depend on the Devices domain: ${_file}"
            )
        endif()
    endforeach()
endfunction()

function(neurale_assert_internal_recording_dependencies)
    foreach(_target IN ITEMS neurale_recording_spool neurale_recording neurale_recording_replay)
        if(NOT TARGET ${_target})
            message(FATAL_ERROR "the internal ${_target} target is required")
        endif()
        get_target_property(_internal ${_target} NEURALE_INTERNAL_ONLY)
        if(NOT _internal)
            message(FATAL_ERROR "${_target} must remain marked internal-only")
        endif()
        _neurale_require_no_public_surface(${_target})
    endforeach()

    # The spool container is a store, not a consumer of anything above it. It
    # must not grow a dependency on the streaming runtime or on any algorithm
    # domain: a writer that could reach them would be a writer that could be
    # made to do work on somebody else's thread.
    _neurale_forbid_direct_links(
        neurale_recording_spool
        neurale::streaming
        neurale_streaming
        neurale::signal
        neurale_signal
        neurale::features
        neurale_features
        neurale::models
        neurale_models
        neurale::sorting
        neurale_sorting
        neurale_pipeline
    )
    _neurale_forbid_neurale_and_python_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/recording"
    )

    # The recorder core consumes the streaming runtime's frame, discontinuity,
    # and fault types, and nothing above streaming: no algorithm domain, no
    # pipeline, and no Python. It is a critical recorder, and Python on its
    # callback would take the GIL, which the critical contract forbids.
    _neurale_require_direct_links(
        neurale_recording
        neurale_recording_spool
        neurale::streaming
    )
    _neurale_forbid_direct_links(
        neurale_streaming
        neurale_recording
        neurale_recording_spool
    )
    _neurale_forbid_direct_links(
        neurale_recording
        neurale::signal
        neurale_signal
        neurale::features
        neurale_features
        neurale::models
        neurale_models
        neurale::sorting
        neurale_sorting
        neurale_pipeline
    )
    _neurale_forbid_domain_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
    )
    _neurale_forbid_python_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/recording/core"
    )

    # The replay side reads a committed session and produces streaming's frames
    # and discontinuities. It links the spool for the container's checksum and
    # streaming for those types, and nothing else -- in particular it does not
    # link the recorder core: a replay has nothing to say to the writer that
    # produced the session, and a target that linked both would make that claim
    # unfalsifiable. Streaming still does not know either of them exists.
    _neurale_require_exact_links(
        neurale_recording_replay
        neurale_recording_spool
        neurale::streaming
    )
    _neurale_forbid_direct_links(
        neurale_streaming
        neurale_recording_replay
    )
    _neurale_forbid_domain_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/recording/replay"
    )
    _neurale_forbid_python_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/recording/replay"
    )
endfunction()

function(neurale_assert_internal_pipeline_dependencies)
    if(NOT TARGET neurale_pipeline)
        message(FATAL_ERROR "the internal neurale_pipeline target is required")
    endif()

    get_target_property(_internal neurale_pipeline NEURALE_INTERNAL_ONLY)
    if(NOT _internal)
        message(FATAL_ERROR "neurale_pipeline must remain marked internal-only")
    endif()

    # Streaming is the generic runtime boundary. Its only direct link is the
    # platform thread abstraction; an exact set prevents a new algorithm,
    # device, recorder, or presenter from entering through a name this file
    # does not yet know about.
    _neurale_require_exact_links(neurale_streaming Threads::Threads)

    _neurale_require_direct_links(
        neurale_pipeline
        neurale::streaming
        neurale::signal
        neurale::features
        neurale::models
        neurale::sorting
    )
    _neurale_require_no_public_surface(neurale_pipeline)
    _neurale_forbid_direct_links(
        neurale_streaming
        neurale_pipeline
        neurale::signal
        neurale_signal
        neurale::features
        neurale_features
        neurale::models
        neurale_models
        neurale::sorting
        neurale_sorting
    )
    _neurale_forbid_domain_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/streaming"
    )
    _neurale_forbid_domain_includes(
        "${PROJECT_SOURCE_DIR}/cpp/include/neurale/streaming"
    )
endfunction()

function(neurale_assert_experiments_dependencies)
    if(NOT TARGET neurale_experiments)
        message(FATAL_ERROR "the neurale_experiments target is required")
    endif()

    # The shared experiment value contract links nothing at all. Velocity
    # assistance must stay usable without the streaming runtime, and a paradigm
    # must not be able to reach a device; a value contract that linked either
    # would hand both of those dependencies to everything built on it. Stating
    # "links nothing" is the only form of that rule which survives somebody
    # adding a new lower-level target.
    get_target_property(_experiment_links neurale_experiments LINK_LIBRARIES)
    if(_experiment_links)
        message(FATAL_ERROR
            "neurale_experiments must link no internal target and links [${_experiment_links}]"
        )
    endif()
    _neurale_forbid_foreign_domain_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/experiments" experiments
    )
    _neurale_forbid_foreign_domain_includes(
        "${PROJECT_SOURCE_DIR}/cpp/include/neurale/experiments" experiments
    )

    # The contract must not reach Python, a device, or ambient randomness.
    foreach(_directory IN ITEMS
            "${PROJECT_SOURCE_DIR}/cpp/src/experiments"
            "${PROJECT_SOURCE_DIR}/cpp/include/neurale/experiments")
        _neurale_forbid_python_includes("${_directory}")
        _neurale_forbid_device_includes("${_directory}")
        _neurale_forbid_ambient_randomness("${_directory}")
    endforeach()

    # Nothing below experiments knows that experiments exists.
    foreach(_target IN ITEMS
            neurale_runtime
            neurale_streaming
            neurale_signal
            neurale_devices
            neurale_features
            neurale_models
            neurale_sorting
            neurale_pipeline
            neurale_recording
            neurale_recording_spool
            neurale_recording_replay)
        if(TARGET ${_target})
            _neurale_forbid_direct_links(${_target} neurale_experiments neurale::experiments)
        endif()
    endforeach()
endfunction()

function(neurale_assert_execution_dependencies)
    if(NOT TARGET neurale_execution)
        message(FATAL_ERROR "the internal neurale_execution target is required")
    endif()
    get_target_property(_internal neurale_execution NEURALE_INTERNAL_ONLY)
    if(NOT _internal)
        message(FATAL_ERROR "neurale_execution must remain internal-only")
    endif()
    _neurale_require_no_public_surface(neurale_execution)
    # Compose task semantics with recording and native interval processing.
    # Model kernels remain behind the pipeline boundary.
    _neurale_require_exact_links(
        neurale_execution
        neurale::experiments
        neurale::streaming
        neurale_recording
        neurale_pipeline
    )
    # neurale_recording publishes no header under neurale/, so the include rule
    # below cannot see the bridge reaching into it. This is what does.
    _neurale_restrict_quoted_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/execution"
        # The seam's own headers.
        adaptive_center_out_actuator.h
        ssvep_controller.h
        ../pipeline/interval_mean_processor.h
        abnormal_reporter.h
        bounded_trace_queue.h
        center_out_controller.h
        center_out_trace_writer.h
        interval_runner.h
        contract_status.h
        records.h
        session.h
        presentation_evidence.h
        speech_controller.h
        speech_trace_writer.h
        webgrid_controller.h
        webgrid_trace_writer.h
        # The recorder core's, and only these: the submission surface, the
        # status it reports, the plan it is held to, the spool handle a caller
        # hands over, and the producer-identity registry the record kinds are.
        recorder.h
        recorder_status.h
        recording_plan.h
        spool_file.h
        spool_layout.h
    )
    _neurale_forbid_python_includes(
        "${PROJECT_SOURCE_DIR}/cpp/src/execution"
    )
    _neurale_forbid_device_includes("${PROJECT_SOURCE_DIR}/cpp/src/execution")
    _neurale_forbid_ambient_randomness("${PROJECT_SOURCE_DIR}/cpp/src/execution")
    file(GLOB_RECURSE _execution_files
        "${PROJECT_SOURCE_DIR}/cpp/src/execution/*.cpp"
        "${PROJECT_SOURCE_DIR}/cpp/src/execution/*.h"
    )
    foreach(_file IN LISTS _execution_files)
        file(READ "${_file}" _contents)
        string(REGEX MATCHALL
            "#[ \t]*include[ \t]*[<\"]neurale/[a-z_]+/"
            _includes "${_contents}"
        )
        foreach(_include IN LISTS _includes)
            if(NOT _include MATCHES "neurale/(experiments|streaming)/")
                message(FATAL_ERROR
                    "experiment execution may include only experiments and streaming: ${_file}"
                )
            endif()
        endforeach()
    endforeach()

    # The execution target sits above the domain libraries. The aggregate
    # _native extension composes them; lower-level libraries must not link back.
    foreach(_target IN ITEMS
            neurale_runtime
            neurale_streaming
            neurale_signal
            neurale_devices
            neurale_features
            neurale_models
            neurale_sorting
            neurale_pipeline
            neurale_recording
            neurale_recording_spool
            neurale_recording_replay
            neurale_experiments
            _native_cuda)
        if(TARGET ${_target})
            _neurale_forbid_direct_links(${_target} neurale_execution)
        endif()
    endforeach()
endfunction()

function(neurale_assert_experiment_presentation_dependencies)
    if(NEURALE_ENABLE_EXPERIMENT_PRESENTATION STREQUAL "OFF")
        if(TARGET neurale_experiment_presentation OR
           TARGET neurale_experiment_presentation_recording)
            message(FATAL_ERROR
                "presentation targets must not exist when "
                "NEURALE_ENABLE_EXPERIMENT_PRESENTATION=OFF"
            )
        endif()
        return()
    endif()

    foreach(_target IN ITEMS neurale_experiment_presentation
            neurale_experiment_presentation_recording)
        if(NOT TARGET ${_target})
            message(FATAL_ERROR
                "NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON requires ${_target}"
            )
        endif()
    endforeach()

    get_target_property(
        _internal
        neurale_experiment_presentation
        NEURALE_INTERNAL_ONLY
    )
    if(NOT _internal)
        message(FATAL_ERROR
            "neurale_experiment_presentation must remain internal-only"
        )
    endif()
    _neurale_require_no_public_surface(neurale_experiment_presentation)
    get_target_property(
        _recording_internal
        neurale_experiment_presentation_recording
        NEURALE_INTERNAL_ONLY
    )
    if(NOT _recording_internal)
        message(FATAL_ERROR
            "neurale_experiment_presentation_recording must remain internal-only"
        )
    endif()
    _neurale_require_no_public_surface(neurale_experiment_presentation_recording)
    _neurale_require_exact_links(
        neurale_experiment_presentation
        neurale::experiments
        ${NEURALE_PRESENTATION_GLFW_TARGET}
        ${NEURALE_PRESENTATION_OPENGL_TARGET}
        ${NEURALE_PRESENTATION_FREETYPE_TARGET}
        ${NEURALE_PRESENTATION_HARFBUZZ_TARGET}
    )
    _neurale_require_exact_links(
        neurale_experiment_presentation_recording
        neurale_experiment_presentation
        neurale_execution
        neurale::experiments
        neurale::streaming
    )

    # Presentation sits above the experiments domain. Nothing in the core,
    # runtime, acquisition, or recording library may acquire it through a reverse
    # link. The aggregate _native extension owns optional presentation bindings.
    foreach(_target IN ITEMS
            neurale_runtime
            neurale_streaming
            neurale_signal
            neurale_devices
            neurale_features
            neurale_models
            neurale_sorting
            neurale_pipeline
            neurale_recording
            neurale_recording_spool
            neurale_recording_replay
            neurale_experiments
            neurale_execution
            _native_cuda)
        if(TARGET ${_target})
            _neurale_forbid_direct_links(
                ${_target}
                neurale_experiment_presentation
                neurale_experiment_presentation_recording
            )
        endif()
    endforeach()
endfunction()

function(neurale_assert_cuda_dependencies)
    if(NEURALE_ENABLE_CUDA STREQUAL "OFF")
        foreach(_target IN ITEMS neurale_runtime_cuda neurale_models_cuda _native_cuda)
            if(TARGET ${_target})
                message(FATAL_ERROR
                    "NEURALE_ENABLE_CUDA=OFF must not create ${_target}"
                )
            endif()
        endforeach()
        return()
    endif()

    if(NEURALE_CUDA_FOUND)
        foreach(_target IN ITEMS neurale_runtime_cuda _native_cuda)
            if(NOT TARGET ${_target})
                message(FATAL_ERROR "a CUDA-enabled build requires ${_target}")
            endif()
        endforeach()
        _neurale_require_exact_extension_links(_native_cuda neurale::runtime_cuda)
    endif()

    if(NEURALE_CUDA_KERNELS_FOUND)
        if(NOT TARGET neurale_models_cuda)
            message(FATAL_ERROR "CUDA model kernels require neurale_models_cuda")
        endif()
        _neurale_require_direct_links(_native neurale::models_cuda)
    endif()
endfunction()

function(neurale_assert_devices_dependencies)
    if(NOT TARGET neurale_devices)
        message(FATAL_ERROR "the neurale_devices target is required")
    endif()
    # SDKs remain in external plugins; the core only links the platform loader.
    _neurale_require_exact_links(
        neurale_devices
        neurale::streaming
        neurale::signal
        ${CMAKE_DL_LIBS}
    )
    _neurale_forbid_direct_links(
        neurale_devices
        neurale::features
        neurale_features
        neurale::models
        neurale_models
        neurale::sorting
        neurale_sorting
        neurale_pipeline
        neurale_recording
        neurale_recording_spool
        neurale_recording_replay
    )
    _neurale_forbid_direct_links(
        neurale_streaming
        neurale_devices
        neurale::devices
    )
    _neurale_forbid_direct_links(
        neurale_signal
        neurale_devices
        neurale::devices
    )
    foreach(_directory IN ITEMS
            "${PROJECT_SOURCE_DIR}/cpp/src/streaming"
            "${PROJECT_SOURCE_DIR}/cpp/include/neurale/streaming"
            "${PROJECT_SOURCE_DIR}/cpp/src/signal"
            "${PROJECT_SOURCE_DIR}/cpp/include/neurale/signal")
        _neurale_forbid_device_includes("${_directory}")
    endforeach()
    _neurale_forbid_python_includes("${PROJECT_SOURCE_DIR}/cpp/src/devices")
endfunction()
