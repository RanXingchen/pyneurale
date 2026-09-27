// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <neurale/experiments/webgrid.h>

#include "surface.h"

namespace neurale::experiment_presentation
{

enum class WebGridCellVisual : std::uint8_t
{
    inactive = 0,
    active,
    correct_feedback,
    incorrect_feedback,
};

struct WebGridPresentationStyle
{
    Color background{0.0F, 0.0F, 0.0F, 1.0F};
    Color cell{0.12F, 0.12F, 0.12F, 1.0F};
    Color active_target{0.95F, 0.75F, 0.1F, 1.0F};
    Color correct_feedback{0.15F, 0.8F, 0.25F, 1.0F};
    Color incorrect_feedback{0.9F, 0.2F, 0.2F, 1.0F};
    Color grid_line{0.55F, 0.55F, 0.55F, 1.0F};
    Color pointer{1.0F, 1.0F, 1.0F, 1.0F};
    double pointer_radius{};
    std::size_t circle_segments{32};
};

struct WebGridPresentationConfig
{
    WebGridPresentationStyle style{};
    std::string_view title{"PyNeurale WebGrid experiment"};
    Size2i window_size{800, 600};
    AspectPolicy aspect_policy{AspectPolicy::fit_letterbox};
    int monitor_idx{-1};
    int swap_interval{1};
    std::size_t input_capacity{256};
    int selection_button{};
    bool fullscreen{};
    bool resizable{true};
    bool visible{true};
};

struct WebGridRenderCell
{
    experiments::TargetId id{experiments::kUnsetTargetId};
    experiments::webgrid::CellBounds bounds{};
    WebGridCellVisual visual{WebGridCellVisual::inactive};
    Color fill{};
};

struct WebGridRenderPlan
{
    Color background{};
    experiments::webgrid::TaskBounds task_bounds{};
    std::uint16_t n_cells{};
    std::array<WebGridRenderCell, experiments::webgrid::kMaxWebGridCells> cells{};
    experiments::webgrid::PointerPosition pointer{};
    double pointer_radius{};
    Color pointer_color{};
};

[[nodiscard]] SurfaceStatus validate(const WebGridPresentationStyle& style) noexcept;
[[nodiscard]] SurfaceStatus validate(const WebGridPresentationConfig& config) noexcept;
[[nodiscard]] SurfaceStatus
validate_webgrid_presentation(const experiments::webgrid::WebGridConfig& task,
                              const WebGridPresentationConfig& presentation) noexcept;

/// Build drawing values by asking the task for each cell's frozen half-open
/// bounds. No pointer-to-cell mapping, progression, or metrics occur here.
/// When @p last_selection is non-null and valid, selection feedback is
/// colored from the task's already-decided `event.correct` and
/// `event.selected_id`; correctness is never decided here, so this is not a
/// second task-truth input.
[[nodiscard]] SurfaceStatus
build_webgrid_render_plan(const experiments::webgrid::WebGridConfig& task,
                          const WebGridPresentationStyle& style,
                          const experiments::webgrid::WebGridSnapshot& snapshot,
                          const experiments::webgrid::PointerPosition& pointer,
                          const experiments::webgrid::WebGridSelectionRecord* last_selection,
                          WebGridRenderPlan& plan) noexcept;

enum class WebGridPresentationInputKind : std::uint8_t
{
    pointer_update = 0,
    selection_request,
    escape_requested,
    window_close_requested,
};

/// Pointer/control value. A selection_request contains only the pointer at
/// button-press time. The task machine still constructs SelectionEvent and
/// resolves its cell.
struct WebGridPresentationInput
{
    WebGridPresentationInputKind kind{WebGridPresentationInputKind::pointer_update};
    experiments::webgrid::PointerPosition pointer{};
    bool inside_presentation{};
    std::uint64_t input_ordinal{};
    RendererTimeNs renderer_time_ns{};
    experiments::ExperimentTimeNs experiment_time_ns{};
    int button{};
    int modifiers{};
};

class WebGridPointerInputAdapter
{
  public:
    [[nodiscard]] SurfaceStatus prepare(int selection_button) noexcept;
    [[nodiscard]] SurfaceStatus adapt(const InputEvent& input, WebGridPresentationInput& output,
                                      bool& available) noexcept;
    void reset() noexcept;

  private:
    int selection_button_{};
    std::uint64_t last_ordinal_{};
    bool prepared_{};
    bool has_ordinal_{};
};

/// Private concrete presenter for one WebGrid session.
class WebGridPresenter
{
  public:
    WebGridPresenter() = default;
    ~WebGridPresenter() = default;
    WebGridPresenter(const WebGridPresenter&) = delete;
    WebGridPresenter& operator=(const WebGridPresenter&) = delete;
    WebGridPresenter(WebGridPresenter&&) = delete;
    WebGridPresenter& operator=(WebGridPresenter&&) = delete;

    [[nodiscard]] SurfaceStatus open(const experiments::webgrid::WebGridConfig& task,
                                     const WebGridPresentationConfig& presentation,
                                     RendererTimeNs renderer_origin_ns,
                                     experiments::ExperimentTimeNs experiment_origin_ns);
    [[nodiscard]] SurfaceStatus prepare();
    [[nodiscard]] SurfaceStatus
    update(const experiments::webgrid::WebGridSnapshot& snapshot,
           const experiments::webgrid::PointerPosition& pointer,
           const experiments::webgrid::WebGridSelectionRecord* last_selection = nullptr) noexcept;
    [[nodiscard]] SurfaceStatus pump_events() noexcept;
    [[nodiscard]] SurfaceStatus poll_input(WebGridPresentationInput& input,
                                           bool& available) noexcept;
    [[nodiscard]] SoftwarePresentationTimes
    render(experiments::ExperimentTimeNs requested_ns,
           experiments::ExperimentTimeNs intended_ns) noexcept;
    [[nodiscard]] SurfaceStatus request_window_size(Size2i size) noexcept;
    void cancel() noexcept;
    void close() noexcept;

    [[nodiscard]] const WebGridRenderPlan& latest_render_plan() const noexcept
    {
        return latest_plan_;
    }
    [[nodiscard]] std::uint64_t latest_update_ordinal() const noexcept
    {
        return latest_update_ordinal_;
    }
    [[nodiscard]] std::uint64_t rendered_update_ordinal() const noexcept
    {
        return rendered_update_ordinal_;
    }
    [[nodiscard]] const CoordinateMapper& coordinates() const noexcept
    {
        return surface_.coordinates();
    }
    [[nodiscard]] ResourceStats resource_stats() const noexcept
    {
        return surface_.resource_stats();
    }
    [[nodiscard]] PresentationEnvironment environment() const noexcept
    {
        return surface_.environment();
    }
    [[nodiscard]] SurfaceLifecycle lifecycle() const noexcept
    {
        return surface_.lifecycle();
    }
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }
    [[nodiscard]] const WebGridPresentationConfig& configuration() const noexcept
    {
        return presentation_;
    }
    [[nodiscard]] const experiments::webgrid::WebGridConfig& task_configuration() const noexcept
    {
        return task_;
    }

  private:
    PresentationSurface surface_{};
    WebGridPointerInputAdapter input_{};
    experiments::webgrid::WebGridConfig task_{};
    /// Owned backing bytes for presentation_.title, which is a non-owning view.
    /// validate(WebGridPresentationConfig) reads title again on the provenance
    /// path, long after open() returned and the caller's storage may be gone.
    std::string title_{};
    WebGridPresentationConfig presentation_{};
    WebGridRenderPlan latest_plan_{};
    std::uint64_t latest_update_ordinal_{};
    std::uint64_t rendered_update_ordinal_{};
    bool opened_{};
    bool prepared_{};
    bool has_update_{};
};

} // namespace neurale::experiment_presentation
