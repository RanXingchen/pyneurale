# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

# Resolve the complete experiment-presentation stack. This function is called only
# when presentation was explicitly requested, so a missing dependency is a
# configuration error rather than a reason to silently disable the feature.
#
# For GLFW, FreeType, and HarfBuzz the discovery order is: first the system
# package (find_package via FIND_PACKAGE_ARGS), and only if that is not found
# does FetchContent download and build a private static copy inside the build
# tree. The system-package path stays the default for CI and packagers that
# prefer system packages; the bundled path makes a presentation build possible
# where no development package is installed (notably local Windows).
#
# Offline escape hatches use standard CMake, not a second PyNeurale option:
#   -DFETCHCONTENT_FULLY_DISCONNECTED=ON    forbid downloads; system package
#                                           missing + download forbidden = error
#   -DFETCHCONTENT_SOURCE_DIR_<NAME>=/dir   use a local unpacked source tree,
#                                           e.g. FETCHCONTENT_SOURCE_DIR_GLFW
#
# OpenGL is never bundled: it is the system graphics stack (runtime + driver +
# WGL/GLX/EGL/Cocoa), so find_package(OpenGL) stays REQUIRED.
#
# Pinned dependency versions (repo-controlled, not whatever the system has):
#   GLFW     3.4        https://github.com/glfw/glfw
#   FreeType 2.13.3     https://freetype.org
#   HarfBuzz 14.3.1     https://github.com/harfbuzz/harfbuzz
# The three are linked statically into the private neurale_experiment_presentation
# target; their compile-time version macros feed dependency_versions() directly,
# so bundled and system-package builds report the version actually in use.

include(FetchContent)

function(neurale_find_experiment_presentation_dependencies)
    # Build the three bundled libraries as static. BUILD_SHARED_LIBS is a
    # normal function-scope variable so it governs FreeType and HarfBuzz
    # add_library() defaults only while their MakeAvailable runs here, without
    # leaking into the surrounding project. GLFW uses its own GLFW_BUILD_SHARED.
    set(BUILD_SHARED_LIBS OFF)
    # FetchContent dependencies are private implementation details. Their
    # install rules must not publish headers, libraries, or package configs in
    # the PyNeurale wheel (and a later editable rebuild must not rediscover its
    # own bundled dependency as a system package).
    set(SKIP_INSTALL_ALL ON)

    # --- GLFW: static, no tests/examples/docs. The system CMake config package
    # is named glfw3 (glfw3Config.cmake), so NAMES glfw3 keeps the system path
    # working; the bundled CMake project creates the `glfw` target. ---
    set(GLFW_BUILD_SHARED OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    if(UNIX AND NOT APPLE)
        # The supported Linux presentation artifact is the X11/Xvfb profile.
        # Do not let a GLFW default change pull Wayland/xkbcommon libraries
        # into that wheel or imply a native-Wayland support claim.
        set(GLFW_BUILD_X11 ON CACHE BOOL "" FORCE)
        set(GLFW_BUILD_WAYLAND OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(glfw
        URL "https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip"
        URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9
        FIND_PACKAGE_ARGS NAMES glfw3 CONFIG
    )
    FetchContent_MakeAvailable(glfw)

    # --- FreeType: raster only. Disable optional Zlib/PNG/BZip2/Brotli and, in
    # particular, HarfBuzz, so the optional FT<->HB integration collapses to
    # the single FT->HB direction this stack actually uses (HB shaping + hb-ft +
    # FT raster, not FreeType's HarfBuzz-driven autohinter). The declared name
    # is `Freetype` (capital F) with no CONFIG so CMake's built-in FindFreetype
    # module stays the system-package path, matching the previous behavior. ---
    set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_ZLIB ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_PNG ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_BZIP2 ON CACHE BOOL "" FORCE)
    set(FT_DISABLE_BROTLI ON CACHE BOOL "" FORCE)
    FetchContent_Declare(Freetype
        URL "https://download-mirror.savannah.gnu.org/releases/freetype/freetype-2.13.3.tar.xz"
        URL_HASH SHA256=0550350666d427c74daeb85d5ac7bb353acba5f76956395995311a9c6f063289
        FIND_PACKAGE_ARGS
    )
    FetchContent_MakeAvailable(Freetype)

    # Bundled FreeType creates the `freetype` target but only exposes the
    # `Freetype::Freetype` namespaced target through install/export, which
    # FetchContent does not run. Add an alias so the consumer link line
    # (Freetype::Freetype) is identical on the bundled and system-package
    # paths. HarfBuzz detects `TARGET freetype` directly (its CMakeLists.txt
    # sets HB_HAVE_FREETYPE and links `freetype` itself), so no Freetype_DIR
    # wiring is needed for the FT->HB integration.
    if(TARGET freetype AND NOT TARGET Freetype::Freetype)
        add_library(Freetype::Freetype ALIAS freetype)
    endif()

    # --- HarfBuzz: shaping only. Keep FreeType integration, drop glib/icu/
    # graphite2/cairo so the bundled build needs nothing beyond FreeType. ---
    set(HB_HAVE_FREETYPE ON CACHE BOOL "" FORCE)
    set(HB_HAVE_GLIB OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_ICU OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_GRAPHITE2 OFF CACHE BOOL "" FORCE)
    set(HB_HAVE_CAIRO OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(harfbuzz
        URL "https://github.com/harfbuzz/harfbuzz/releases/download/14.3.1/harfbuzz-14.3.1.tar.xz"
        URL_HASH SHA256=9dae9538aae2ffdf70cec31f2c27bf68e2aaeeae3112688467697d5faf6194f7
        FIND_PACKAGE_ARGS CONFIG
    )
    FetchContent_MakeAvailable(harfbuzz)

    # Bundled HarfBuzz creates the `harfbuzz` target but only exposes the
    # `harfbuzz::harfbuzz` namespaced target through install/export. Add an
    # alias so the consumer link line matches the system-package path.
    if(TARGET harfbuzz AND NOT TARGET harfbuzz::harfbuzz)
        add_library(harfbuzz::harfbuzz ALIAS harfbuzz)
    endif()

    # --- OpenGL: system graphics stack, never bundled ---
    find_package(OpenGL REQUIRED)

    # Resolve the concrete target names. Bundled and system builds expose the
    # same names; both candidates are accepted to match existing packaging.
    if(TARGET glfw)
        set(_glfw_target glfw)
    elseif(TARGET glfw3::glfw)
        set(_glfw_target glfw3::glfw)
    else()
        message(FATAL_ERROR
            "NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON requires GLFW. "
            "Install a GLFW package that provides the glfw or glfw3::glfw CMake target, "
            "or allow FetchContent to download it (do not pass "
            "FETCHCONTENT_FULLY_DISCONNECTED=ON), or point glfw3_DIR or "
            "FETCHCONTENT_SOURCE_DIR_GLFW at a local source tree."
        )
    endif()

    if(NOT TARGET Freetype::Freetype)
        message(FATAL_ERROR
            "NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON requires FreeType. "
            "Install the FreeType development package, or allow FetchContent to "
            "download it, or point Freetype_DIR or FETCHCONTENT_SOURCE_DIR_FREETYPE "
            "at a local source tree."
        )
    endif()

    # Prefer the raw target created by FetchContent. A separately discovered
    # package may already have defined harfbuzz::harfbuzz (for example as an
    # optional transitive FreeType package in a Python environment); selecting
    # that imported target after MakeAvailable would lose the bundled source
    # target's BUILD_INTERFACE include path containing hb-ft.h.
    if(TARGET harfbuzz)
        set(_harfbuzz_target harfbuzz)
    elseif(TARGET harfbuzz::harfbuzz)
        set(_harfbuzz_target harfbuzz::harfbuzz)
    elseif(TARGET HarfBuzz::HarfBuzz)
        set(_harfbuzz_target HarfBuzz::HarfBuzz)
    else()
        message(FATAL_ERROR
            "NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON requires HarfBuzz. "
            "Install a HarfBuzz package that provides the harfbuzz::harfbuzz or "
            "HarfBuzz::HarfBuzz CMake target, or allow FetchContent to download "
            "it, or point harfbuzz_DIR or FETCHCONTENT_SOURCE_DIR_HARFBUZZ at a "
            "local source tree."
        )
    endif()

    if(TARGET OpenGL::GL)
        set(_opengl_target OpenGL::GL)
    elseif(TARGET OpenGL::OpenGL)
        set(_opengl_target OpenGL::OpenGL)
    else()
        message(FATAL_ERROR
            "NEURALE_ENABLE_EXPERIMENT_PRESENTATION=ON requires OpenGL. "
            "Install the platform OpenGL development package so CMake provides "
            "OpenGL::GL or OpenGL::OpenGL."
        )
    endif()

    set(NEURALE_PRESENTATION_GLFW_TARGET "${_glfw_target}" PARENT_SCOPE)
    set(NEURALE_PRESENTATION_OPENGL_TARGET "${_opengl_target}" PARENT_SCOPE)
    set(NEURALE_PRESENTATION_FREETYPE_TARGET Freetype::Freetype PARENT_SCOPE)
    set(NEURALE_PRESENTATION_HARFBUZZ_TARGET "${_harfbuzz_target}" PARENT_SCOPE)
endfunction()
