// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace neurale::experiment_presentation
{

using RendererTimeNs = std::uint64_t;

enum class SurfaceStatus : std::uint8_t
{
    ok = 0,
    invalid_configuration,
    invalid_state,
    wrong_thread,
    time_regressed,
    time_overflow,
    input_overflow,
    presentation_handoff_overflow,
    glfw_initialization_failed,
    monitor_unavailable,
    window_creation_failed,
    opengl_function_missing,
    shader_compilation_failed,
    shader_link_failed,
    resource_capacity_exceeded,
    font_load_failed,
    invalid_utf8,
    glyph_missing,
    atlas_capacity_exceeded,
    text_not_prepared,
    cancelled,
    window_closed,
    presentation_failed,
    /// The renderer completed the frame, but the software presentation instant
    /// it observed fell at or after the request's `valid_until_ns`.
    ///
    /// Distinct from presentation_failed on purpose: nothing about the renderer
    /// went wrong, so labelling it a renderer failure would put "the swap
    /// happened at T" and "the renderer failed" in one record and force a
    /// reader to guess which is true. The semantic verdict is still
    /// PresentationStatus::expired -- this was never a valid presentation -- and
    /// the swap instant is still preserved as software evidence.
    presentation_deadline_missed,
};

[[nodiscard]] constexpr const char* surface_status_name(SurfaceStatus status) noexcept
{
    switch (status)
    {
    case SurfaceStatus::ok:
        return "ok";
    case SurfaceStatus::invalid_configuration:
        return "invalid_configuration";
    case SurfaceStatus::invalid_state:
        return "invalid_state";
    case SurfaceStatus::wrong_thread:
        return "wrong_thread";
    case SurfaceStatus::time_regressed:
        return "time_regressed";
    case SurfaceStatus::time_overflow:
        return "time_overflow";
    case SurfaceStatus::input_overflow:
        return "input_overflow";
    case SurfaceStatus::presentation_handoff_overflow:
        return "presentation_handoff_overflow";
    case SurfaceStatus::glfw_initialization_failed:
        return "glfw_initialization_failed";
    case SurfaceStatus::monitor_unavailable:
        return "monitor_unavailable";
    case SurfaceStatus::window_creation_failed:
        return "window_creation_failed";
    case SurfaceStatus::opengl_function_missing:
        return "opengl_function_missing";
    case SurfaceStatus::shader_compilation_failed:
        return "shader_compilation_failed";
    case SurfaceStatus::shader_link_failed:
        return "shader_link_failed";
    case SurfaceStatus::resource_capacity_exceeded:
        return "resource_capacity_exceeded";
    case SurfaceStatus::font_load_failed:
        return "font_load_failed";
    case SurfaceStatus::invalid_utf8:
        return "invalid_utf8";
    case SurfaceStatus::glyph_missing:
        return "glyph_missing";
    case SurfaceStatus::atlas_capacity_exceeded:
        return "atlas_capacity_exceeded";
    case SurfaceStatus::text_not_prepared:
        return "text_not_prepared";
    case SurfaceStatus::cancelled:
        return "cancelled";
    case SurfaceStatus::window_closed:
        return "window_closed";
    case SurfaceStatus::presentation_failed:
        return "presentation_failed";
    case SurfaceStatus::presentation_deadline_missed:
        return "presentation_deadline_missed";
    }
    return "unknown";
}

struct Point2d
{
    double x{};
    double y{};
};

struct Size2i
{
    int width{};
    int height{};
};

struct Rect2d
{
    double left{};
    double bottom{};
    double width{};
    double height{};
};

struct Rect2i
{
    int left{};
    int top{};
    int width{};
    int height{};
};

struct Color
{
    float red{};
    float green{};
    float blue{};
    float alpha{1.0F};
};

enum class AspectPolicy : std::uint8_t
{
    fit_letterbox = 0,
    stretch,
};

struct ResourceStats
{
    bool window_open{};
    bool opengl_ready{};
    bool text_ready{};
    std::size_t vertex_capacity{};
    std::size_t batch_capacity{};
    std::size_t input_capacity{};
    std::size_t prepared_text_count{};
    std::size_t prepared_glyph_count{};
    std::uint64_t text_preparation_ns{};
    std::uint64_t shaping_preparation_ns{};
    std::uint64_t glyph_cache_preparation_ns{};
    std::uint64_t unexpected_runtime_glyph_cache_misses{};
};

struct PresentationEnvironment
{
    std::array<char, 128> gpu_vendor{};
    std::array<char, 256> gpu_renderer{};
    std::array<char, 128> opengl_version{};
    int monitor_idx{-1};
    int refresh_rate_hz{};
    int window_width{};
    int window_height{};
    int framebuffer_width{};
    int framebuffer_height{};
    int swap_interval{};
    bool fullscreen{};
};

} // namespace neurale::experiment_presentation
