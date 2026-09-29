// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include <neurale/experiments/center_out.h>

#include "surface.h"

namespace neurale::experiment_presentation
{

inline constexpr std::size_t kMaxCenterOutRenderCircles =
    experiments::center_out::kMaxSurroundingTargets + 2;
inline constexpr std::size_t kMaxCenterOutCircleSegments = 256;

enum class CenterOutCircleRole : std::uint8_t
{
    center_target = 0,
    outward_target,
    cursor,
};

enum class CenterOutVisualState : std::uint8_t
{
    inactive = 0,
    active_move,
    active_hold,
    success,
    failure,
    cursor,
};

struct CenterOutPresentationStyle
{
    Color background{0.06F, 0.09F, 0.10F, 1.0F};
    Color center_target{0.32F, 0.43F, 0.44F, 1.0F};
    Color outward_target{0.27F, 0.37F, 0.38F, 1.0F};
    Color active_move{1.0F, 0.79F, 0.34F, 1.0F};
    Color active_hold{0.34F, 0.78F, 0.91F, 1.0F};
    Color success{0.34F, 0.82F, 0.59F, 1.0F};
    Color failure{0.95F, 0.39F, 0.38F, 1.0F};
    Color cursor{0.97F, 0.99F, 1.0F, 1.0F};
    // Visual radii are deliberately not defaulted: a value that suited only a
    // normalized workspace (0.04 / 0.025) would let a millimetre task silently
    // inherit sub-millimetre targets. validate(const CenterOutPresentationStyle&)
    // rejects a non-positive radius, so the caller must state one in the same
    // GeometryUnit as the task and the logical space.
    double target_radius{};
    double cursor_radius{};
    std::size_t circle_segments{96};
};

struct CenterOutRenderCircle
{
    CenterOutCircleRole role{CenterOutCircleRole::center_target};
    CenterOutVisualState visual{CenterOutVisualState::inactive};
    experiments::TargetId target_id{experiments::kUnsetTargetId};
    experiments::center_out::WorkspacePoint center{};
    double radius{};
    Color color{};
    bool filled{};
};

struct CenterOutRenderPlan
{
    Color background{};
    std::uint8_t n_circles{};
    std::array<CenterOutRenderCircle, kMaxCenterOutRenderCircles> circles{};
};

[[nodiscard]] SurfaceStatus validate(const CenterOutPresentationStyle& style) noexcept;

/// Concrete presentation coordinate contract for one Center-Out session.
///
/// This presenter does no unit conversion. The logical space, the visual radii in @p style,
/// and every target/cursor position the render plan carries are all measured in
/// @p geometry_unit, and that unit must equal the task's
/// ``CenterOut2DConfig::geometry_unit``. A mismatch is a configuration error
/// reported by validate_center_out_presentation() and prepare(), never a silent
/// rescale that draws a millimetre task at normalized coordinates (or vice
/// versa). The window/monitor/aspect fields are presentation-only and have no
/// task meaning.
struct CenterOutPresentationConfig
{
    /// Unit of @p logical_space and of @p style radii. Must be a declared,
    /// concrete GeometryUnit equal to the task's; ::GeometryUnit::unspecified is
    /// rejected.
    experiments::center_out::GeometryUnit geometry_unit{
        experiments::center_out::GeometryUnit::unspecified};
    /// Logical workspace the surface maps, in @p geometry_unit.
    Rect2d logical_space{};
    /// Visual radii and colours, in @p geometry_unit.
    CenterOutPresentationStyle style{};
    std::string_view title{"PyNeurale Center-Out experiment"};
    Size2i window_size{800, 600};
    AspectPolicy aspect_policy{AspectPolicy::fit_letterbox};
    int monitor_idx{-1};
    int swap_interval{1};
    std::size_t input_capacity{256};
    bool fullscreen{};
    bool resizable{true};
    bool visible{true};
};

[[nodiscard]] SurfaceStatus validate(const CenterOutPresentationConfig& config) noexcept;

/// Check that @p task and @p presentation share one geometry unit and that both
/// are individually valid. This is the frozen Center-Out presentation boundary:
/// it does not convert units, and a mismatch is reported rather than drawn. It
/// touches no surface and no window, so the contract can be exercised without a
/// presentation build.
[[nodiscard]] SurfaceStatus
validate_center_out_presentation(const experiments::center_out::CenterOut2DConfig& task,
                                 const CenterOutPresentationConfig& presentation) noexcept;

/// Build only drawing values. This function never performs hit testing, state
/// transitions, scheduling, assistance, guidance, clipping, or coordinate
/// feedback into the task state machine.
[[nodiscard]] SurfaceStatus build_center_out_render_plan(
    const experiments::center_out::CenterOut2DConfig& task, const CenterOutPresentationStyle& style,
    const experiments::center_out::CenterOutSnapshot& snapshot,
    const experiments::center_out::WorkspacePoint& cursor, CenterOutRenderPlan& plan) noexcept;

enum class CenterOutPresentationControlKind : std::uint8_t
{
    escape_requested = 0,
    window_close_requested,
};

struct CenterOutPresentationControlEvent
{
    CenterOutPresentationControlKind kind{CenterOutPresentationControlKind::escape_requested};
    std::uint64_t input_ordinal{};
    RendererTimeNs renderer_time_ns{};
    experiments::ExperimentTimeNs experiment_time_ns{};
};

/// Translate raw presentation input to Center-Out orchestration control. The
/// result is evidence for the caller; it never mutates task state.
[[nodiscard]] bool center_out_control_event(const InputEvent& input,
                                            CenterOutPresentationControlEvent& event) noexcept;

/// Private concrete presenter for one Center-Out session.
class CenterOut2DPresenter
{
  public:
    CenterOut2DPresenter() = default;
    ~CenterOut2DPresenter() = default;
    CenterOut2DPresenter(const CenterOut2DPresenter&) = delete;
    CenterOut2DPresenter& operator=(const CenterOut2DPresenter&) = delete;
    CenterOut2DPresenter(CenterOut2DPresenter&&) = delete;
    CenterOut2DPresenter& operator=(CenterOut2DPresenter&&) = delete;

    [[nodiscard]] SurfaceStatus open(const CenterOutPresentationConfig& config,
                                     RendererTimeNs renderer_origin_ns,
                                     experiments::ExperimentTimeNs experiment_origin_ns);
    [[nodiscard]] SurfaceStatus prepare(const experiments::center_out::CenterOut2DConfig& task);
    [[nodiscard]] SurfaceStatus
    update(const experiments::center_out::CenterOutSnapshot& snapshot,
           const experiments::center_out::WorkspacePoint& cursor) noexcept;

    [[nodiscard]] SurfaceStatus pump_events() noexcept;
    [[nodiscard]] SurfaceStatus poll_control(CenterOutPresentationControlEvent& event,
                                             bool& available) noexcept;
    [[nodiscard]] SoftwarePresentationTimes
    render(experiments::ExperimentTimeNs requested_ns,
           experiments::ExperimentTimeNs intended_ns) noexcept;

    [[nodiscard]] SurfaceStatus request_window_size(Size2i size) noexcept;
    void cancel() noexcept;
    void close() noexcept;

    [[nodiscard]] std::uint64_t latest_update_ordinal() const noexcept
    {
        return latest_update_ordinal_;
    }
    [[nodiscard]] std::uint64_t rendered_update_ordinal() const noexcept
    {
        return rendered_update_ordinal_;
    }
    [[nodiscard]] const CenterOutRenderPlan& latest_render_plan() const noexcept
    {
        return latest_plan_;
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
    [[nodiscard]] SurfaceStatus status() const noexcept
    {
        return surface_.status();
    }
    [[nodiscard]] SurfaceLifecycle lifecycle() const noexcept
    {
        return surface_.lifecycle();
    }
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }
    [[nodiscard]] const CenterOutPresentationConfig& configuration() const noexcept
    {
        return presentation_;
    }
    [[nodiscard]] const experiments::center_out::CenterOut2DConfig&
    task_configuration() const noexcept
    {
        return task_;
    }

  private:
    PresentationSurface surface_{};
    experiments::center_out::CenterOut2DConfig task_{};
    /// Owned backing bytes for presentation_.title, which is a non-owning view.
    /// The config outlives open(), so a view into the caller's storage would
    /// dangle as soon as that storage went away -- and it is read again later,
    /// by validate() on the provenance path. Same reason SpeechCuePresenter
    /// owns its font path.
    std::string title_{};
    CenterOutPresentationConfig presentation_{};
    CenterOutPresentationStyle style_{};
    CenterOutRenderPlan latest_plan_{};
    std::uint64_t latest_update_ordinal_{};
    std::uint64_t rendered_update_ordinal_{};
    bool prepared_{};
    bool has_update_{};
};

} // namespace neurale::experiment_presentation
