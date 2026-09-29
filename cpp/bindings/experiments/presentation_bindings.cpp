// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "center_out_presenter.h"
#include "center_out_session_context.h"
#include "dependency_versions.h"
#include "recording_bridge.h"
#include "speech_presenter.h"
#include "ssvep_presenter.h"
#include "webgrid_presenter.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <utility>

namespace py = pybind11;

namespace ep = neurale::experiment_presentation;
namespace ss = neurale::experiments::ssvep;
static void bind_ssvep_presentation(py::module_& module)
{
    py::class_<ep::SSVEPConcretePresenter>(module, "_SSVEPConcretePresenter")
        .def(py::init<>())
        .def("open",
             [](ep::SSVEPConcretePresenter& p, const ss::SSVEPConfig& task,
                const std::string& title, ep::Size2i size, int monitor, bool fullscreen,
                bool visible, ep::RendererTimeNs renderer_origin,
                neurale::experiments::ExperimentTimeNs experiment_origin,
                const std::vector<std::array<double, 2>>& positions, double target_size)
             {
                 if (positions.size() > ss::kMaxSSVEPTargets)
                     throw py::value_error("positions contains too many targets");
                 ep::SSVEPLayout layout{};
                 layout.count = positions.size();
                 layout.target_size = target_size;
                 for (std::size_t i = 0; i < positions.size(); ++i)
                     layout.positions[i] = {positions[i][0], positions[i][1]};
                 return p.open(
                     task,
                     {.title = title,
                      .window_size = size,
                      .logical_space = {-1, -1, 2, 2},
                      .monitor_idx = monitor,
                      .swap_interval = 1,
                      .input_admission = ep::InputAdmission::key | ep::InputAdmission::window_close,
                      .fullscreen = fullscreen,
                      .resizable = false,
                      .visible = visible},
                     renderer_origin, experiment_origin, layout);
             })
        .def("pump_events", &ep::SSVEPConcretePresenter::pump_events)
        .def("poll_input",
             [](ep::SSVEPConcretePresenter& p)
             {
                 ep::InputEvent event{};
                 bool available{};
                 const auto status = p.poll_input(event, available);
                 const bool stop =
                     available && (event.kind == ep::InputEventKind::window_close ||
                                   (event.kind == ep::InputEventKind::key &&
                                    event.action == ep::InputAction::press && event.code == 256));
                 return py::make_tuple(status, available, stop);
             })
        .def("render", &ep::SSVEPConcretePresenter::render, py::arg("snapshot"), py::arg("time_ns"),
             py::call_guard<py::gil_scoped_release>())
        .def("_render_relative",
             [](ep::SSVEPConcretePresenter& presenter, ss::SSVEPSnapshot snapshot,
                std::uint64_t time_ns, std::uint64_t epoch)
             {
                 const auto shift = [epoch](std::uint64_t value)
                 {
                     if (value > UINT64_MAX - epoch)
                         throw py::value_error("SSVEP presentation timestamp overflow");
                     return value + epoch;
                 };
                 snapshot.time_ns = shift(snapshot.time_ns);
                 snapshot.active = {shift(snapshot.active.start_ns), shift(snapshot.active.end_ns)};
                 snapshot.stimulation = {shift(snapshot.stimulation.start_ns),
                                         shift(snapshot.stimulation.end_ns)};
                 snapshot.decision_deadline_ns = shift(snapshot.decision_deadline_ns);
                 if (snapshot.has_selection)
                     snapshot.selection.time_ns = shift(snapshot.selection.time_ns);
                 const auto now = shift(time_ns);
                 py::gil_scoped_release release;
                 return presenter.render(snapshot, now);
             })
        .def_property_readonly("environment", &ep::SSVEPConcretePresenter::environment)
        .def("close", &ep::SSVEPConcretePresenter::close);
}

namespace ex = neurale::experiments;

namespace
{

struct CenterOutConfig
{
    ex::center_out::GeometryUnit geometry_unit{ex::center_out::GeometryUnit::unspecified};
    ep::Rect2d logical_space{-1.0, -1.0, 2.0, 2.0};
    ep::CenterOutPresentationStyle style{};
    std::string title{"PyNeurale Center-Out experiment"};
    ep::Size2i window_size{800, 600};
    ep::AspectPolicy aspect_policy{ep::AspectPolicy::fit_letterbox};
    int monitor_idx{-1};
    int swap_interval{1};
    std::size_t input_capacity{256};
    bool fullscreen{};
    bool resizable{true};
    bool visible{true};

    [[nodiscard]] ep::CenterOutPresentationConfig native() const noexcept
    {
        return {geometry_unit,  logical_space, style,       title,
                window_size,    aspect_policy, monitor_idx, swap_interval,
                input_capacity, fullscreen,    resizable,   visible};
    }
};

struct WebGridConfig
{
    ep::WebGridPresentationStyle style{};
    std::string title{"PyNeurale WebGrid experiment"};
    ep::Size2i window_size{800, 600};
    ep::AspectPolicy aspect_policy{ep::AspectPolicy::fit_letterbox};
    int monitor_idx{-1};
    int swap_interval{1};
    std::size_t input_capacity{256};
    int selection_button{};
    bool fullscreen{};
    bool resizable{true};
    bool visible{true};

    [[nodiscard]] ep::WebGridPresentationConfig native() const noexcept
    {
        return {style,       title,         window_size,    aspect_policy,
                monitor_idx, swap_interval, input_capacity, selection_button,
                fullscreen,  resizable,     visible};
    }
};

struct TextConfig
{
    std::string font_path{};
    long face_idx{};
    unsigned int pixel_height{32};
    int atlas_width{2048};
    int atlas_height{2048};
    std::size_t max_texts{64};
    std::size_t max_glyphs{1024};

    [[nodiscard]] ep::TextAtlasConfig native() const noexcept
    {
        return {.font_path = font_path,
                .face_idx = face_idx,
                .pixel_height = pixel_height,
                .atlas_width = atlas_width,
                .atlas_height = atlas_height,
                .max_texts = max_texts,
                .max_glyphs = max_glyphs};
    }
};

struct SpeechConfig
{
    ep::SpeechPresentationStyle style{};
    TextConfig text{};
    std::string title{"PyNeurale Speech cue experiment"};
    ep::Size2i window_size{800, 600};
    ep::Rect2d logical_space{-1.0, -1.0, 2.0, 2.0};
    ep::AspectPolicy aspect_policy{ep::AspectPolicy::fit_letterbox};
    int monitor_idx{-1};
    int swap_interval{1};
    std::size_t input_capacity{64};
    std::size_t outcome_capacity{64};
    bool fullscreen{};
    bool resizable{true};
    bool visible{true};

    [[nodiscard]] ep::SpeechPresentationConfig native() const noexcept
    {
        return {style,         text.native(), title,         window_size,    logical_space,
                aspect_policy, monitor_idx,   swap_interval, input_capacity, outcome_capacity,
                fullscreen,    resizable,     visible};
    }
};

struct CenterOutPresenter
{
    ep::CenterOut2DPresenter value{};
};

class CenterOutSessionPresentationBridge final
{
  public:
    explicit CenterOutSessionPresentationBridge(py::object session) : session_(std::move(session))
    {
        auto capsule = session_.attr("_presentation_context")().cast<py::capsule>();
        context_ = static_cast<neurale::bindings::experiments::CenterOutSessionContext*>(
            PyCapsule_GetPointer(capsule.ptr(),
                                 neurale::bindings::experiments::kCenterOutSessionContextCapsule));
        if (context_ == nullptr || context_->writer == nullptr || context_->controller == nullptr)
            throw py::value_error("invalid Center-Out session presentation context");
        context_->writer->require_presentation_evidence(true);
    }

    [[nodiscard]] bool record_config(const CenterOutPresenter& presenter) const noexcept
    {
        return ep::record_center_out_presentation_config(*context_->writer, presenter.value);
    }

    [[nodiscard]] bool report(const CenterOutPresenter& presenter,
                              const ep::SoftwarePresentationTimes& times,
                              std::uint64_t source_ordinal,
                              const ex::TrialIdentity& trial) const noexcept
    {
        return ep::report_center_out_presentation(*context_->writer, *context_->controller, times,
                                                  presenter.value.rendered_update_ordinal(),
                                                  source_ordinal, trial);
    }

    [[nodiscard]] bool report_failure(const CenterOutPresenter& presenter,
                                      ex::ExperimentTimeNs time_ns, ep::SurfaceStatus status,
                                      const ex::TrialIdentity& trial) const noexcept
    {
        return ep::report_center_out_runtime_failure(*context_->writer, *context_->controller,
                                                     time_ns, status, trial,
                                                     presenter.value.latest_update_ordinal());
    }

  private:
    py::object session_{};
    neurale::bindings::experiments::CenterOutSessionContext* context_{};
};
struct WebGridPresenter
{
    ep::WebGridPresenter value{};
};
struct SpeechPresenter
{
    ep::SpeechCuePresenter value{};
};

} // namespace

namespace
{
void bind_presentation_common(py::module_& module)
{
    module.doc() = "Optional concrete native experiment presentation.";

    module.def("dependency_versions",
               []()
               {
                   const auto versions = ep::dependency_versions();
                   py::dict result;
                   result["glfw"] = py::make_tuple(versions.glfw_major, versions.glfw_minor,
                                                   versions.glfw_revision);
                   result["opengl_minimum"] =
                       py::make_tuple(versions.minimum_opengl_major, versions.minimum_opengl_minor);
                   result["freetype"] = py::make_tuple(
                       versions.freetype_major, versions.freetype_minor, versions.freetype_patch);
                   result["harfbuzz"] = py::make_tuple(
                       versions.harfbuzz_major, versions.harfbuzz_minor, versions.harfbuzz_micro);
                   return result;
               });
    module.def("renderer_monotonic_now_ns", &ep::renderer_monotonic_now_ns);

    py::enum_<ep::SurfaceStatus>(module, "PresentationRuntimeStatus")
        .value("OK", ep::SurfaceStatus::ok)
        .value("INVALID_CONFIGURATION", ep::SurfaceStatus::invalid_configuration)
        .value("INVALID_STATE", ep::SurfaceStatus::invalid_state)
        .value("WRONG_THREAD", ep::SurfaceStatus::wrong_thread)
        .value("TIME_REGRESSED", ep::SurfaceStatus::time_regressed)
        .value("TIME_OVERFLOW", ep::SurfaceStatus::time_overflow)
        .value("INPUT_OVERFLOW", ep::SurfaceStatus::input_overflow)
        .value("PRESENTATION_HANDOFF_OVERFLOW", ep::SurfaceStatus::presentation_handoff_overflow)
        .value("GLFW_INITIALIZATION_FAILED", ep::SurfaceStatus::glfw_initialization_failed)
        .value("MONITOR_UNAVAILABLE", ep::SurfaceStatus::monitor_unavailable)
        .value("WINDOW_CREATION_FAILED", ep::SurfaceStatus::window_creation_failed)
        .value("OPENGL_FUNCTION_MISSING", ep::SurfaceStatus::opengl_function_missing)
        .value("SHADER_COMPILATION_FAILED", ep::SurfaceStatus::shader_compilation_failed)
        .value("SHADER_LINK_FAILED", ep::SurfaceStatus::shader_link_failed)
        .value("RESOURCE_CAPACITY_EXCEEDED", ep::SurfaceStatus::resource_capacity_exceeded)
        .value("FONT_LOAD_FAILED", ep::SurfaceStatus::font_load_failed)
        .value("INVALID_UTF8", ep::SurfaceStatus::invalid_utf8)
        .value("GLYPH_MISSING", ep::SurfaceStatus::glyph_missing)
        .value("ATLAS_CAPACITY_EXCEEDED", ep::SurfaceStatus::atlas_capacity_exceeded)
        .value("TEXT_NOT_PREPARED", ep::SurfaceStatus::text_not_prepared)
        .value("CANCELLED", ep::SurfaceStatus::cancelled)
        .value("WINDOW_CLOSED", ep::SurfaceStatus::window_closed)
        .value("PRESENTATION_FAILED", ep::SurfaceStatus::presentation_failed)
        .value("PRESENTATION_DEADLINE_MISSED", ep::SurfaceStatus::presentation_deadline_missed);
    py::enum_<ep::AspectPolicy>(module, "AspectPolicy")
        .value("FIT_LETTERBOX", ep::AspectPolicy::fit_letterbox)
        .value("STRETCH", ep::AspectPolicy::stretch);

    py::class_<ep::Point2d>(module, "Point2D")
        .def(py::init<double, double>(), py::arg("x") = 0.0, py::arg("y") = 0.0)
        .def_readonly("x", &ep::Point2d::x)
        .def_readonly("y", &ep::Point2d::y);
    py::class_<ep::Size2i>(module, "WindowSize")
        .def(py::init<int, int>(), py::arg("width") = 800, py::arg("height") = 600)
        .def_readonly("width", &ep::Size2i::width)
        .def_readonly("height", &ep::Size2i::height);
    py::class_<ep::Rect2d>(module, "LogicalRect")
        .def(py::init<double, double, double, double>(), py::arg("left") = -1.0,
             py::arg("bottom") = -1.0, py::arg("width") = 2.0, py::arg("height") = 2.0)
        .def_readonly("left", &ep::Rect2d::left)
        .def_readonly("bottom", &ep::Rect2d::bottom)
        .def_readonly("width", &ep::Rect2d::width)
        .def_readonly("height", &ep::Rect2d::height);
    py::class_<ep::Color>(module, "Color")
        .def(py::init<float, float, float, float>(), py::arg("red") = 0.0F, py::arg("green") = 0.0F,
             py::arg("blue") = 0.0F, py::arg("alpha") = 1.0F)
        .def_readonly("red", &ep::Color::red)
        .def_readonly("green", &ep::Color::green)
        .def_readonly("blue", &ep::Color::blue)
        .def_readonly("alpha", &ep::Color::alpha);

    py::class_<ep::ResourceStats>(module, "PresentationResourceStats")
        .def_readonly("window_open", &ep::ResourceStats::window_open)
        .def_readonly("opengl_ready", &ep::ResourceStats::opengl_ready)
        .def_readonly("text_ready", &ep::ResourceStats::text_ready)
        .def_readonly("vertex_capacity", &ep::ResourceStats::vertex_capacity)
        .def_readonly("batch_capacity", &ep::ResourceStats::batch_capacity)
        .def_readonly("input_capacity", &ep::ResourceStats::input_capacity)
        .def_readonly("prepared_text_count", &ep::ResourceStats::prepared_text_count)
        .def_readonly("prepared_glyph_count", &ep::ResourceStats::prepared_glyph_count)
        .def_readonly("text_preparation_ns", &ep::ResourceStats::text_preparation_ns)
        .def_readonly("shaping_preparation_ns", &ep::ResourceStats::shaping_preparation_ns)
        .def_readonly("glyph_cache_preparation_ns", &ep::ResourceStats::glyph_cache_preparation_ns)
        .def_readonly("unexpected_runtime_glyph_cache_misses",
                      &ep::ResourceStats::unexpected_runtime_glyph_cache_misses);
    py::class_<ep::PresentationEnvironment>(module, "PresentationEnvironment")
        .def_property_readonly("gpu_vendor", [](const ep::PresentationEnvironment& value)
                               { return std::string(value.gpu_vendor.data()); })
        .def_property_readonly("gpu_renderer", [](const ep::PresentationEnvironment& value)
                               { return std::string(value.gpu_renderer.data()); })
        .def_property_readonly("opengl_version", [](const ep::PresentationEnvironment& value)
                               { return std::string(value.opengl_version.data()); })
        .def_readonly("monitor_idx", &ep::PresentationEnvironment::monitor_idx)
        .def_readonly("refresh_rate_hz", &ep::PresentationEnvironment::refresh_rate_hz)
        .def_readonly("window_width", &ep::PresentationEnvironment::window_width)
        .def_readonly("window_height", &ep::PresentationEnvironment::window_height)
        .def_readonly("framebuffer_width", &ep::PresentationEnvironment::framebuffer_width)
        .def_readonly("framebuffer_height", &ep::PresentationEnvironment::framebuffer_height)
        .def_readonly("swap_interval", &ep::PresentationEnvironment::swap_interval)
        .def_readonly("fullscreen", &ep::PresentationEnvironment::fullscreen);
    py::class_<ep::SoftwarePresentationTimes>(module, "SoftwarePresentationTimes")
        .def_readonly("status", &ep::SoftwarePresentationTimes::status)
        .def_readonly("requested_ns", &ep::SoftwarePresentationTimes::requested_ns)
        .def_readonly("intended_ns", &ep::SoftwarePresentationTimes::intended_ns)
        .def_readonly("submitted_renderer_ns",
                      &ep::SoftwarePresentationTimes::submitted_renderer_ns)
        .def_readonly("submitted_ns", &ep::SoftwarePresentationTimes::submitted_ns)
        .def_readonly("presented_renderer_ns",
                      &ep::SoftwarePresentationTimes::presented_renderer_ns)
        .def_readonly("presented_ns", &ep::SoftwarePresentationTimes::presented_ns);
}

void bind_center_out_presentation(py::module_& module)
{
    py::class_<ep::CenterOutPresentationStyle>(module, "CenterOutPresentationStyle")
        .def(py::init(
                 [](double target_radius, double cursor_radius, std::size_t circle_segments,
                    ep::Color background, ep::Color center_target, ep::Color outward_target,
                    ep::Color active_move, ep::Color active_hold, ep::Color success,
                    ep::Color failure, ep::Color cursor)
                 {
                     ep::CenterOutPresentationStyle value{};
                     value.target_radius = target_radius;
                     value.cursor_radius = cursor_radius;
                     value.circle_segments = circle_segments;
                     value.background = background;
                     value.center_target = center_target;
                     value.outward_target = outward_target;
                     value.active_move = active_move;
                     value.active_hold = active_hold;
                     value.success = success;
                     value.failure = failure;
                     value.cursor = cursor;
                     return value;
                 }),
             py::arg("target_radius"), py::arg("cursor_radius"), py::arg("circle_segments") = 96,
             py::arg("background") = ep::Color{0.0F, 0.0F, 0.0F, 1.0F},
             py::arg("center_target") = ep::Color{0.35F, 0.35F, 0.35F, 1.0F},
             py::arg("outward_target") = ep::Color{0.25F, 0.25F, 0.25F, 1.0F},
             py::arg("active_move") = ep::Color{0.95F, 0.8F, 0.1F, 1.0F},
             py::arg("active_hold") = ep::Color{0.2F, 0.65F, 1.0F, 1.0F},
             py::arg("success") = ep::Color{0.15F, 0.8F, 0.25F, 1.0F},
             py::arg("failure") = ep::Color{0.9F, 0.2F, 0.2F, 1.0F},
             py::arg("cursor") = ep::Color{1.0F, 1.0F, 1.0F, 1.0F})
        .def_readonly("background", &ep::CenterOutPresentationStyle::background)
        .def_readonly("center_target", &ep::CenterOutPresentationStyle::center_target)
        .def_readonly("outward_target", &ep::CenterOutPresentationStyle::outward_target)
        .def_readonly("active_move", &ep::CenterOutPresentationStyle::active_move)
        .def_readonly("active_hold", &ep::CenterOutPresentationStyle::active_hold)
        .def_readonly("success", &ep::CenterOutPresentationStyle::success)
        .def_readonly("failure", &ep::CenterOutPresentationStyle::failure)
        .def_readonly("cursor", &ep::CenterOutPresentationStyle::cursor)
        .def_readonly("target_radius", &ep::CenterOutPresentationStyle::target_radius)
        .def_readonly("cursor_radius", &ep::CenterOutPresentationStyle::cursor_radius)
        .def_readonly("circle_segments", &ep::CenterOutPresentationStyle::circle_segments);
    py::class_<CenterOutConfig>(module, "CenterOutPresentationConfig")
        .def(py::init<ex::center_out::GeometryUnit, ep::Rect2d, ep::CenterOutPresentationStyle,
                      std::string, ep::Size2i, ep::AspectPolicy, int, int, std::size_t, bool, bool,
                      bool>(),
             py::arg("geometry_unit"), py::arg("logical_space"), py::arg("style"),
             py::arg("title") = "PyNeurale Center-Out experiment",
             py::arg("window_size") = ep::Size2i{800, 600},
             py::arg("aspect_policy") = ep::AspectPolicy::fit_letterbox,
             py::arg("monitor_idx") = -1, py::arg("swap_interval") = 1,
             py::arg("input_capacity") = 256, py::arg("fullscreen") = false,
             py::arg("resizable") = true, py::arg("visible") = true)
        .def_readonly("geometry_unit", &CenterOutConfig::geometry_unit)
        .def_readonly("logical_space", &CenterOutConfig::logical_space)
        .def_readonly("style", &CenterOutConfig::style)
        .def_readonly("title", &CenterOutConfig::title)
        .def_readonly("window_size", &CenterOutConfig::window_size)
        .def_readonly("aspect_policy", &CenterOutConfig::aspect_policy)
        .def_readonly("monitor_idx", &CenterOutConfig::monitor_idx)
        .def_readonly("swap_interval", &CenterOutConfig::swap_interval)
        .def_readonly("input_capacity", &CenterOutConfig::input_capacity)
        .def_readonly("fullscreen", &CenterOutConfig::fullscreen)
        .def_readonly("resizable", &CenterOutConfig::resizable)
        .def_readonly("visible", &CenterOutConfig::visible);
    py::enum_<ep::CenterOutPresentationControlKind>(module, "CenterOutPresentationControlKind")
        .value("ESCAPE_REQUESTED", ep::CenterOutPresentationControlKind::escape_requested)
        .value("WINDOW_CLOSE_REQUESTED",
               ep::CenterOutPresentationControlKind::window_close_requested);
    py::class_<ep::CenterOutPresentationControlEvent>(module, "CenterOutPresentationControlEvent")
        .def_readonly("kind", &ep::CenterOutPresentationControlEvent::kind)
        .def_readonly("input_ordinal", &ep::CenterOutPresentationControlEvent::input_ordinal)
        .def_readonly("renderer_time_ns", &ep::CenterOutPresentationControlEvent::renderer_time_ns)
        .def_readonly("experiment_time_ns",
                      &ep::CenterOutPresentationControlEvent::experiment_time_ns);
    py::class_<CenterOutPresenter>(module, "CenterOutPresenter")
        .def(py::init<>())
        .def(
            "open",
            [](CenterOutPresenter& self, const ex::center_out::CenterOut2DConfig& task,
               const CenterOutConfig& config, ep::RendererTimeNs renderer_origin_ns,
               ex::ExperimentTimeNs experiment_origin_ns)
            {
                auto status =
                    self.value.open(config.native(), renderer_origin_ns, experiment_origin_ns);
                return status == ep::SurfaceStatus::ok ? self.value.prepare(task) : status;
            },
            py::arg("task"), py::arg("config"), py::arg("renderer_origin_ns"),
            py::arg("experiment_origin_ns"))
        .def("update",
             [](CenterOutPresenter& self, const ex::center_out::CenterOutSnapshot& snapshot,
                const ex::center_out::WorkspacePoint& cursor)
             { return self.value.update(snapshot, cursor); })
        .def("pump_events", [](CenterOutPresenter& self) { return self.value.pump_events(); })
        .def("poll_control",
             [](CenterOutPresenter& self)
             {
                 ep::CenterOutPresentationControlEvent value{};
                 bool available{};
                 const auto status = self.value.poll_control(value, available);
                 return py::make_tuple(status, available ? py::cast(value) : py::none());
             })
        .def("render", [](CenterOutPresenter& self, ex::ExperimentTimeNs requested_ns,
                          ex::ExperimentTimeNs intended_ns)
             { return self.value.render(requested_ns, intended_ns); })
        .def("cancel", [](CenterOutPresenter& self) { self.value.cancel(); })
        .def("close", [](CenterOutPresenter& self) { self.value.close(); })
        .def_property_readonly("resource_stats", [](const CenterOutPresenter& self)
                               { return self.value.resource_stats(); })
        .def_property_readonly("environment", [](const CenterOutPresenter& self)
                               { return self.value.environment(); })
        .def_property_readonly("latest_update_ordinal", [](const CenterOutPresenter& self)
                               { return self.value.latest_update_ordinal(); })
        .def_property_readonly("rendered_update_ordinal", [](const CenterOutPresenter& self)
                               { return self.value.rendered_update_ordinal(); });
    py::class_<CenterOutSessionPresentationBridge>(module, "CenterOutSessionPresentationBridge",
                                                   py::is_final())
        .def(py::init<py::object>(), py::arg("session"))
        .def("record_config", &CenterOutSessionPresentationBridge::record_config,
             py::arg("presenter"))
        .def("report", &CenterOutSessionPresentationBridge::report, py::arg("presenter"),
             py::arg("times"), py::arg("source_ordinal"), py::arg("trial"))
        .def("report_failure", &CenterOutSessionPresentationBridge::report_failure,
             py::arg("presenter"), py::arg("time_ns"), py::arg("status"), py::arg("trial"));
}

void bind_webgrid_presentation(py::module_& module)
{
    py::class_<ep::WebGridPresentationStyle>(module, "WebGridPresentationStyle")
        .def(py::init(
                 [](double pointer_radius, std::size_t circle_segments, ep::Color background,
                    ep::Color cell, ep::Color active_target, ep::Color correct_feedback,
                    ep::Color incorrect_feedback, ep::Color grid_line, ep::Color pointer)
                 {
                     ep::WebGridPresentationStyle value{};
                     value.pointer_radius = pointer_radius;
                     value.circle_segments = circle_segments;
                     value.background = background;
                     value.cell = cell;
                     value.active_target = active_target;
                     value.correct_feedback = correct_feedback;
                     value.incorrect_feedback = incorrect_feedback;
                     value.grid_line = grid_line;
                     value.pointer = pointer;
                     return value;
                 }),
             py::arg("pointer_radius"), py::arg("circle_segments") = 32,
             py::arg("background") = ep::Color{0.0F, 0.0F, 0.0F, 1.0F},
             py::arg("cell") = ep::Color{0.12F, 0.12F, 0.12F, 1.0F},
             py::arg("active_target") = ep::Color{0.95F, 0.75F, 0.1F, 1.0F},
             py::arg("correct_feedback") = ep::Color{0.15F, 0.8F, 0.25F, 1.0F},
             py::arg("incorrect_feedback") = ep::Color{0.9F, 0.2F, 0.2F, 1.0F},
             py::arg("grid_line") = ep::Color{0.55F, 0.55F, 0.55F, 1.0F},
             py::arg("pointer") = ep::Color{1.0F, 1.0F, 1.0F, 1.0F})
        .def_readonly("background", &ep::WebGridPresentationStyle::background)
        .def_readonly("cell", &ep::WebGridPresentationStyle::cell)
        .def_readonly("active_target", &ep::WebGridPresentationStyle::active_target)
        .def_readonly("correct_feedback", &ep::WebGridPresentationStyle::correct_feedback)
        .def_readonly("incorrect_feedback", &ep::WebGridPresentationStyle::incorrect_feedback)
        .def_readonly("grid_line", &ep::WebGridPresentationStyle::grid_line)
        .def_readonly("pointer", &ep::WebGridPresentationStyle::pointer)
        .def_readonly("pointer_radius", &ep::WebGridPresentationStyle::pointer_radius)
        .def_readonly("circle_segments", &ep::WebGridPresentationStyle::circle_segments);
    py::class_<WebGridConfig>(module, "WebGridPresentationConfig")
        .def(py::init<ep::WebGridPresentationStyle, std::string, ep::Size2i, ep::AspectPolicy, int,
                      int, std::size_t, int, bool, bool, bool>(),
             py::arg("style"), py::arg("title") = "PyNeurale WebGrid experiment",
             py::arg("window_size") = ep::Size2i{800, 600},
             py::arg("aspect_policy") = ep::AspectPolicy::fit_letterbox,
             py::arg("monitor_idx") = -1, py::arg("swap_interval") = 1,
             py::arg("input_capacity") = 256, py::arg("selection_button") = 0,
             py::arg("fullscreen") = false, py::arg("resizable") = true, py::arg("visible") = true)
        .def_readonly("style", &WebGridConfig::style)
        .def_readonly("title", &WebGridConfig::title)
        .def_readonly("window_size", &WebGridConfig::window_size)
        .def_readonly("aspect_policy", &WebGridConfig::aspect_policy)
        .def_readonly("monitor_idx", &WebGridConfig::monitor_idx)
        .def_readonly("swap_interval", &WebGridConfig::swap_interval)
        .def_readonly("input_capacity", &WebGridConfig::input_capacity)
        .def_readonly("selection_button", &WebGridConfig::selection_button)
        .def_readonly("fullscreen", &WebGridConfig::fullscreen)
        .def_readonly("resizable", &WebGridConfig::resizable)
        .def_readonly("visible", &WebGridConfig::visible);
    py::enum_<ep::WebGridPresentationInputKind>(module, "WebGridPresentationInputKind")
        .value("POINTER_UPDATE", ep::WebGridPresentationInputKind::pointer_update)
        .value("SELECTION_REQUEST", ep::WebGridPresentationInputKind::selection_request)
        .value("ESCAPE_REQUESTED", ep::WebGridPresentationInputKind::escape_requested)
        .value("WINDOW_CLOSE_REQUESTED", ep::WebGridPresentationInputKind::window_close_requested);
    py::class_<ep::WebGridPresentationInput>(module, "WebGridPresentationInput")
        .def_readonly("kind", &ep::WebGridPresentationInput::kind)
        .def_readonly("pointer", &ep::WebGridPresentationInput::pointer)
        .def_readonly("inside_presentation", &ep::WebGridPresentationInput::inside_presentation)
        .def_readonly("input_ordinal", &ep::WebGridPresentationInput::input_ordinal)
        .def_readonly("renderer_time_ns", &ep::WebGridPresentationInput::renderer_time_ns)
        .def_readonly("experiment_time_ns", &ep::WebGridPresentationInput::experiment_time_ns)
        .def_readonly("button", &ep::WebGridPresentationInput::button);
    py::class_<WebGridPresenter>(module, "WebGridPresenter")
        .def(py::init<>())
        .def(
            "open",
            [](WebGridPresenter& self, const ex::webgrid::WebGridConfig& task,
               const WebGridConfig& config, ep::RendererTimeNs renderer_origin_ns,
               ex::ExperimentTimeNs experiment_origin_ns)
            {
                auto status = self.value.open(task, config.native(), renderer_origin_ns,
                                              experiment_origin_ns);
                return status == ep::SurfaceStatus::ok ? self.value.prepare() : status;
            },
            py::arg("task"), py::arg("config"), py::arg("renderer_origin_ns"),
            py::arg("experiment_origin_ns"))
        .def(
            "update",
            [](WebGridPresenter& self, const ex::webgrid::WebGridSnapshot& snapshot,
               const ex::webgrid::PointerPosition& pointer,
               const ex::webgrid::WebGridSelectionRecord* last_selection)
            { return self.value.update(snapshot, pointer, last_selection); },
            py::arg("snapshot"), py::arg("pointer"), py::arg("last_selection") = nullptr)
        .def("pump_events", [](WebGridPresenter& self) { return self.value.pump_events(); })
        .def("poll_input",
             [](WebGridPresenter& self)
             {
                 ep::WebGridPresentationInput value{};
                 bool available{};
                 const auto status = self.value.poll_input(value, available);
                 return py::make_tuple(status, available ? py::cast(value) : py::none());
             })
        .def("render", [](WebGridPresenter& self, ex::ExperimentTimeNs requested_ns,
                          ex::ExperimentTimeNs intended_ns)
             { return self.value.render(requested_ns, intended_ns); })
        .def("cancel", [](WebGridPresenter& self) { self.value.cancel(); })
        .def("close", [](WebGridPresenter& self) { self.value.close(); })
        .def_property_readonly("resource_stats", [](const WebGridPresenter& self)
                               { return self.value.resource_stats(); })
        .def_property_readonly("environment", [](const WebGridPresenter& self)
                               { return self.value.environment(); });
}

void bind_speech_presentation(py::module_& module)
{
    py::class_<TextConfig>(module, "SpeechTextConfig")
        .def(py::init<std::string, long, unsigned int, int, int, std::size_t, std::size_t>(),
             py::arg("font_path"), py::arg("face_idx") = 0, py::arg("pixel_height") = 32,
             py::arg("atlas_width") = 2048, py::arg("atlas_height") = 2048,
             py::arg("max_texts") = 64, py::arg("max_glyphs") = 1024)
        .def_readonly("font_path", &TextConfig::font_path)
        .def_readonly("face_idx", &TextConfig::face_idx)
        .def_readonly("pixel_height", &TextConfig::pixel_height)
        .def_readonly("atlas_width", &TextConfig::atlas_width)
        .def_readonly("atlas_height", &TextConfig::atlas_height)
        .def_readonly("max_texts", &TextConfig::max_texts)
        .def_readonly("max_glyphs", &TextConfig::max_glyphs);
    py::class_<ep::SpeechPresentationStyle>(module, "SpeechPresentationStyle")
        .def(py::init<ep::Color, ep::Color, ep::Color, ep::Point2d, double, ep::Point2d>(),
             py::arg("background") = ep::Color{0.0F, 0.0F, 0.0F, 1.0F},
             py::arg("fixation") = ep::Color{1.0F, 1.0F, 1.0F, 1.0F},
             py::arg("text") = ep::Color{1.0F, 1.0F, 1.0F, 1.0F},
             py::arg("fixation_center") = ep::Point2d{}, py::arg("fixation_half_extent") = 0.08,
             py::arg("text_baseline") = ep::Point2d{-0.8, 0.0})
        .def_readonly("background", &ep::SpeechPresentationStyle::background)
        .def_readonly("fixation", &ep::SpeechPresentationStyle::fixation)
        .def_readonly("text", &ep::SpeechPresentationStyle::text)
        .def_readonly("fixation_center", &ep::SpeechPresentationStyle::fixation_center)
        .def_readonly("fixation_half_extent", &ep::SpeechPresentationStyle::fixation_half_extent)
        .def_readonly("text_baseline", &ep::SpeechPresentationStyle::text_baseline);
    py::class_<SpeechConfig>(module, "SpeechPresentationConfig")
        .def(
            py::init(
                [](TextConfig text, ep::SpeechPresentationStyle style, std::string title,
                   ep::Size2i window_size, ep::Rect2d logical_space, ep::AspectPolicy aspect_policy,
                   int monitor_idx, int swap_interval, std::size_t input_capacity,
                   std::size_t outcome_capacity, bool fullscreen, bool resizable, bool visible)
                {
                    SpeechConfig value{};
                    value.text = std::move(text);
                    value.style = style;
                    value.title = std::move(title);
                    value.window_size = window_size;
                    value.logical_space = logical_space;
                    value.aspect_policy = aspect_policy;
                    value.monitor_idx = monitor_idx;
                    value.swap_interval = swap_interval;
                    value.input_capacity = input_capacity;
                    value.outcome_capacity = outcome_capacity;
                    value.fullscreen = fullscreen;
                    value.resizable = resizable;
                    value.visible = visible;
                    return value;
                }),
            py::arg("text"), py::arg("style") = ep::SpeechPresentationStyle{},
            py::arg("title") = "PyNeurale Speech cue experiment",
            py::arg("window_size") = ep::Size2i{800, 600},
            py::arg("logical_space") = ep::Rect2d{-1, -1, 2, 2},
            py::arg("aspect_policy") = ep::AspectPolicy::fit_letterbox, py::arg("monitor_idx") = -1,
            py::arg("swap_interval") = 1, py::arg("input_capacity") = 64,
            py::arg("outcome_capacity") = 64, py::arg("fullscreen") = false,
            py::arg("resizable") = true, py::arg("visible") = true)
        .def_readonly("text", &SpeechConfig::text)
        .def_readonly("style", &SpeechConfig::style)
        .def_readonly("title", &SpeechConfig::title)
        .def_readonly("window_size", &SpeechConfig::window_size)
        .def_readonly("logical_space", &SpeechConfig::logical_space)
        .def_readonly("aspect_policy", &SpeechConfig::aspect_policy)
        .def_readonly("monitor_idx", &SpeechConfig::monitor_idx)
        .def_readonly("swap_interval", &SpeechConfig::swap_interval)
        .def_readonly("input_capacity", &SpeechConfig::input_capacity)
        .def_readonly("outcome_capacity", &SpeechConfig::outcome_capacity)
        .def_readonly("fullscreen", &SpeechConfig::fullscreen)
        .def_readonly("resizable", &SpeechConfig::resizable)
        .def_readonly("visible", &SpeechConfig::visible);
    py::class_<ep::SpeechPresentationEvidence>(module, "SpeechPresentationEvidence")
        .def_readonly("request", &ep::SpeechPresentationEvidence::request)
        .def_readonly("outcome", &ep::SpeechPresentationEvidence::outcome)
        .def_readonly("software_times", &ep::SpeechPresentationEvidence::software_times)
        .def_readonly("render_status", &ep::SpeechPresentationEvidence::render_status);
    py::class_<ep::SpeechPresentationResult>(module, "SpeechPresentationResult")
        .def_readonly("status", &ep::SpeechPresentationResult::status)
        .def_readonly("evidence", &ep::SpeechPresentationResult::evidence)
        .def_readonly("outcome_produced", &ep::SpeechPresentationResult::outcome_produced)
        .def_readonly("outcome_enqueued", &ep::SpeechPresentationResult::outcome_enqueued);
    py::enum_<ep::SpeechPresentationControlKind>(module, "SpeechPresentationControlKind")
        .value("ESCAPE_REQUESTED", ep::SpeechPresentationControlKind::escape_requested)
        .value("WINDOW_CLOSE_REQUESTED", ep::SpeechPresentationControlKind::window_close_requested);
    py::class_<ep::SpeechPresentationControlEvent>(module, "SpeechPresentationControlEvent")
        .def_readonly("kind", &ep::SpeechPresentationControlEvent::kind)
        .def_readonly("input_ordinal", &ep::SpeechPresentationControlEvent::input_ordinal)
        .def_readonly("renderer_time_ns", &ep::SpeechPresentationControlEvent::renderer_time_ns)
        .def_readonly("experiment_time_ns",
                      &ep::SpeechPresentationControlEvent::experiment_time_ns);
    py::class_<SpeechPresenter>(module, "SpeechPresenter")
        .def(py::init<>())
        .def(
            "open",
            [](SpeechPresenter& self, const ex::speech::SpeechCatalog& catalog,
               const SpeechConfig& config, ep::RendererTimeNs renderer_origin_ns,
               ex::ExperimentTimeNs experiment_origin_ns)
            {
                auto status =
                    self.value.open(config.native(), renderer_origin_ns, experiment_origin_ns);
                return status == ep::SurfaceStatus::ok ? self.value.prepare(catalog) : status;
            },
            py::arg("catalog"), py::arg("config"), py::arg("renderer_origin_ns"),
            py::arg("experiment_origin_ns"))
        .def("present", [](SpeechPresenter& self, const ex::PresentationRequest& request,
                           ex::ExperimentTimeNs attempt_ns)
             { return self.value.present(request, attempt_ns); })
        .def("pump_events", [](SpeechPresenter& self) { return self.value.pump_events(); })
        .def("poll_control",
             [](SpeechPresenter& self)
             {
                 ep::SpeechPresentationControlEvent value{};
                 bool available{};
                 const auto status = self.value.poll_control(value, available);
                 return py::make_tuple(status, available ? py::cast(value) : py::none());
             })
        .def("poll_outcome",
             [](SpeechPresenter& self)
             {
                 ep::SpeechPresentationEvidence value{};
                 bool available{};
                 const auto status = self.value.poll_outcome(value, available);
                 return py::make_tuple(status, available ? py::cast(value) : py::none());
             })
        .def("reset", [](SpeechPresenter& self) { return self.value.reset(); })
        .def("cancel", [](SpeechPresenter& self) { self.value.cancel(); })
        .def("close", [](SpeechPresenter& self) { self.value.close(); })
        .def_property_readonly("resource_stats", [](const SpeechPresenter& self)
                               { return self.value.resource_stats(); })
        .def_property_readonly("environment", [](const SpeechPresenter& self)
                               { return self.value.environment(); });
}
} // namespace

void bind_experiments_presentation(py::module_& module)
{
    bind_presentation_common(module);
    bind_center_out_presentation(module);
    bind_webgrid_presentation(module);
    bind_speech_presentation(module);
    bind_ssvep_presentation(module);
}
