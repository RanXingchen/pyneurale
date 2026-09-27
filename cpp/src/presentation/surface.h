// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <string_view>

#include <neurale/experiments/contract.h>

#include "coordinate_mapper.h"
#include "input_runtime.h"
#include "surface_types.h"
#include "text_atlas.h"
#include "time_mapper.h"

namespace neurale::experiment_presentation
{

enum class SurfaceLifecycle : std::uint8_t
{
    closed = 0,
    open,
    prepared,
    cancelled,
    faulted,
};

enum class SurfaceFailurePoint : std::uint8_t
{
    none = 0,
    glfw_initialization,
    window_creation,
    opengl_loading,
    shader_compilation,
    font_preparation,
    presentation,
};

struct WindowConfig
{
    std::string_view title{"PyNeurale experiment"};
    /// Windowed size; fullscreen preserves the selected monitor's current mode.
    Size2i window_size{800, 600};
    Rect2d logical_space{-1.0, -1.0, 2.0, 2.0};
    AspectPolicy aspect_policy{AspectPolicy::fit_letterbox};
    int monitor_idx{-1};
    int swap_interval{1};
    std::size_t input_capacity{256};
    InputAdmission input_admission{InputAdmission::all};
    bool fullscreen{};
    bool resizable{true};
    bool visible{true};
    /// Admit number-key presses for explicitly enabled SSVEP demonstrations.
};

struct SurfacePreparation
{
    std::size_t max_vertices{};
    std::size_t max_draw_batches{};
    std::size_t circle_segments{48};
    TextAtlasConfig text{};
};

struct SoftwarePresentationTimes
{
    SurfaceStatus status{SurfaceStatus::invalid_state};
    experiments::ExperimentTimeNs requested_ns{};
    experiments::ExperimentTimeNs intended_ns{};
    RendererTimeNs submitted_renderer_ns{};
    experiments::ExperimentTimeNs submitted_ns{};
    RendererTimeNs presented_renderer_ns{};
    experiments::ExperimentTimeNs presented_ns{};
};

class PresentationSurface
{
  public:
    PresentationSurface();
    ~PresentationSurface();
    PresentationSurface(const PresentationSurface&) = delete;
    PresentationSurface& operator=(const PresentationSurface&) = delete;
    PresentationSurface(PresentationSurface&&) = delete;
    PresentationSurface& operator=(PresentationSurface&&) = delete;

    [[nodiscard]] SurfaceStatus open(const WindowConfig& config, RendererTimeNs renderer_origin_ns,
                                     experiments::ExperimentTimeNs experiment_origin_ns);
    [[nodiscard]] SurfaceStatus prepare(const SurfacePreparation& config,
                                        std::span<const TextCatalogEntry> text_catalog);
    [[nodiscard]] SurfaceStatus pump_events() noexcept;
    [[nodiscard]] SurfaceStatus poll_input(InputEvent& event, bool& available) noexcept;

    [[nodiscard]] SurfaceStatus begin_frame(Color background) noexcept;
    [[nodiscard]] SurfaceStatus line(Point2d start, Point2d end, Color color) noexcept;
    [[nodiscard]] SurfaceStatus polyline(std::span<const Point2d> points, Color color) noexcept;
    [[nodiscard]] SurfaceStatus rectangle(Rect2d rectangle, Color color, bool filled) noexcept;
    [[nodiscard]] SurfaceStatus circle(Point2d center, double radius, Color color,
                                       bool filled) noexcept;
    [[nodiscard]] SurfaceStatus fixation_cross(Point2d center, double half_extent,
                                               Color color) noexcept;
    [[nodiscard]] SurfaceStatus shaped_text(PreparedTextId text, Point2d baseline,
                                            Color color) noexcept;
    [[nodiscard]] SoftwarePresentationTimes
    present(experiments::ExperimentTimeNs requested_ns,
            experiments::ExperimentTimeNs intended_ns) noexcept;

    [[nodiscard]] SurfaceStatus request_window_size(Size2i size) noexcept;
    [[nodiscard]] int monitor_count() const noexcept;
    void cancel() noexcept;
    void close() noexcept;

    [[nodiscard]] SurfaceStatus status() const noexcept;
    [[nodiscard]] SurfaceLifecycle lifecycle() const noexcept;
    [[nodiscard]] const CoordinateMapper& coordinates() const noexcept;
    [[nodiscard]] ResourceStats resource_stats() const noexcept;
    [[nodiscard]] PresentationEnvironment environment() const noexcept;

    // Private deterministic test seam. It is not bound or installed as API.
    void inject_failure(SurfaceFailurePoint point) noexcept;
    void inject_input(const PendingInputEvent& event) noexcept;

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace neurale::experiment_presentation
