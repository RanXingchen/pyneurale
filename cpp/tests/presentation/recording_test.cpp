// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "check_counts.h"
#include "recording_bridge.h"
#include "skip_policy.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace
{
namespace ep = neurale::experiment_presentation;
namespace ex = neurale::experiments;
namespace execution = neurale::execution;

int failures{};

ex::webgrid::WebGridConfig webgrid_config() noexcept
{
    ex::webgrid::WebGridConfig config{};
    config.rows = 2;
    config.columns = 2;
    config.bounds = {0.0, 2.0, 0.0, 2.0};
    config.n_candidates = 4;
    config.candidates[0] = 1;
    config.candidates[1] = 2;
    config.candidates[2] = 3;
    config.candidates[3] = 4;
    config.schedule = ex::webgrid::TargetScheduleKind::explicit_sequence;
    config.immediate_repetition = ex::webgrid::ImmediateRepetitionPolicy::allow;
    config.correct_selection = ex::webgrid::CorrectSelectionPolicy::advance_target;
    config.incorrect_selection = ex::webgrid::IncorrectSelectionPolicy::keep_current_target;
    config.n_explicit = 2;
    config.explicit_targets[0] = 1;
    config.explicit_targets[1] = 2;
    config.initial_target = 1;
    config.target_count_limit = 2;
    config.metric_version = ex::webgrid::kMetricVersion1;
    return config;
}

execution::CenterOutControllerConfig center_out_config() noexcept
{
    execution::CenterOutControllerConfig config{};
    config.decoded_signal_id = 9;
    config.paradigm = 1;
    ex::center_out::RadialLayoutRequest request{};
    request.radius = 0.5;
    request.count = 2;
    request.center_id = 1;
    request.ids[0] = 2;
    request.ids[1] = 3;
    request.spokes[0] = 0;
    request.spokes[1] = 4;
    config.task.geometry_unit = ex::center_out::GeometryUnit::normalized;
    CHECK(ex::center_out::build_radial_layout(request, config.task.layout) ==
          ex::ContractStatus::ok);
    config.task.acceptance = {0.1, 0.1};
    config.task.cursor = {0.0};
    config.task.movement_timeout = {2'000'000'000, 2'000'000'000};
    config.task.selection = ex::center_out::TargetSelectionPolicy::repeat_until_success;
    config.task.seed = 17;
    config.task.trial_limit = 2;
    config.guidance = {ex::center_out::GeometryUnit::normalized, 1.0, 4.0, 4.0, 0.05};
    config.velocity_space.id = 11;
    config.velocity_space.dim = 2;
    config.velocity_space.frame = ex::CommandFrame::workspace_2d;
    config.velocity_space.axes[0] = {ex::CommandAxisName::x, ex::CommandUnit::normalized};
    config.velocity_space.axes[1] = {ex::CommandAxisName::y, ex::CommandUnit::normalized};
    config.initial_position = {0.0, 0.0};
    config.cursor_min = {-1.0, -1.0};
    config.cursor_max = {1.0, 1.0};
    config.trace_capacity = 8;
    return config;
}

std::string presentation_test_font()
{
    if (const auto* configured = std::getenv("NEURALE_PRESENTATION_TEST_FONT");
        configured != nullptr && *configured != '\0')
        return configured;
    for (const auto* candidate :
         {"C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"})
    {
        if (std::filesystem::is_regular_file(candidate))
            return candidate;
    }
    return {};
}

ex::speech::SpeechCatalog presentation_catalog()
{
    ex::speech::SpeechCatalog catalog{};
    catalog.count = 1;
    catalog.entries[0].id = 1;
    catalog.entries[0].content = ex::speech::SpeechContentKind::text;
    constexpr std::string_view text{"Speech"};
    std::memcpy(catalog.entries[0].text.data(), text.data(), text.size());
    catalog.entries[0].text_length = static_cast<std::uint8_t>(text.size());
    return catalog;
}

ex::speech::SpeechCueConfig presentation_speech_config()
{
    ex::speech::SpeechCueConfig config{};
    config.black_bound_ns = 100;
    config.content_bound_ns = 200;
    config.seed = 7;
    config.n_trials = 1;
    config.schedule = ex::speech::SpeechScheduleKind::seeded;
    config.stimulus_order = ex::speech::StimulusOrderPolicy::sequential;
    config.sampler_version = ex::kCurrentSamplerVersion;
    config.n_stimuli = 1;
    config.stimuli[0] = 1;
    return config;
}

neurale::streaming::StreamSchema decoded_schema()
{
    const neurale::streaming::SignalSchema signal{
        9,
        neurale::streaming::SignalDType::float64,
        2,
        2,
        4,
        neurale::streaming::RationalRate{10, 1},
        7,
        neurale::streaming::SignalLayout::sample_major,
        neurale::streaming::DeviceTickTracking::unavailable,
        neurale::streaming::PhysicalUnit::dimensionless,
        5};
    const std::array signals{signal};
    return {13, signals};
}

int test_concrete_bridges_preserve_identity_without_encoding(bool exercise_presenters)
{
    execution::CenterOutController center_controller{};
    CHECK(center_controller.prepare(decoded_schema(), center_out_config()) ==
          neurale::streaming::StreamStatus::ok);
    CHECK(center_controller.start(0, 100) == neurale::streaming::StreamStatus::ok);
    execution::CenterOutTraceWriter center_writer{center_controller, 0};
    ep::SoftwarePresentationTimes times{};
    times.status = ep::SurfaceStatus::ok;
    times.requested_ns = 100;
    times.intended_ns = 110;
    times.submitted_renderer_ns = 1'000;
    times.submitted_ns = 120;
    times.presented_renderer_ns = 1'010;
    times.presented_ns = 130;
    // The bridge seam carries both the presenter-local update_ordinal and the
    // controller-side source-ordinal join key; the caller supplies the
    // latter, never a default copied from the presenter's update counter.
    CHECK(ep::report_center_out_presentation(center_writer, center_controller, times, 7, 42,
                                             center_controller.snapshot().trial));
    const auto center_trial = center_controller.snapshot().trial;
    CHECK(!ep::report_center_out_runtime_failure(center_writer, center_controller, 140,
                                                 ep::SurfaceStatus::ok, center_trial));
    CHECK(ep::report_center_out_runtime_failure(center_writer, center_controller, 140,
                                                ep::SurfaceStatus::presentation_failed,
                                                center_trial, 8));
    CHECK(center_controller.apply_presentation_failures() == neurale::streaming::StreamStatus::ok);
    CHECK(center_controller.abnormal_summary().trials_affected == 1);
    auto wrong_center_trial = center_controller.snapshot().trial;
    ++wrong_center_trial.ordinal;
    CHECK(ep::report_center_out_runtime_failure(center_writer, center_controller, 141,
                                                ep::SurfaceStatus::presentation_failed,
                                                wrong_center_trial, 9));
    CHECK(center_controller.apply_presentation_failures() ==
          neurale::streaming::StreamStatus::invalid_frame);
    CHECK(center_controller.dropped_trace_count() == 1);

    ep::WebGridPresentationInput input{};
    input.kind = ep::WebGridPresentationInputKind::selection_request;
    input.pointer = {0.25, 0.75};
    input.inside_presentation = true;
    input.input_ordinal = 18;
    input.renderer_time_ns = 2'000;
    input.experiment_time_ns = 210;
    input.button = 0;
    input.modifiers = 2;
    const auto recorded = ep::webgrid_input_evidence(input);
    CHECK(recorded.available);
    CHECK(recorded.input_ordinal == 18);
    CHECK(recorded.renderer_time_ns == 2'000);
    CHECK(recorded.experiment_time_ns == 210);
    CHECK(recorded.kind ==
          static_cast<std::uint8_t>(ep::WebGridPresentationInputKind::selection_request));
    CHECK(recorded.inside_presentation);

    const auto config = webgrid_config();
    execution::WebGridHeadlessController webgrid_controller{};
    CHECK(webgrid_controller.prepare(config, 8) == neurale::streaming::StreamStatus::ok);
    CHECK(webgrid_controller.start(2, 100) == neurale::streaming::StreamStatus::ok);
    execution::WebGridTraceWriter webgrid_writer{webgrid_controller, 3};
    const auto snapshot = webgrid_controller.snapshot();
    ex::webgrid::GridCell active{};
    CHECK(ex::webgrid::grid_cell(config, snapshot.active_target, active) == ex::ContractStatus::ok);
    input.pointer = {(active.bounds.min_x + active.bounds.max_x) * 0.5,
                     (active.bounds.min_y + active.bounds.max_y) * 0.5};
    ex::SelectionEvent selection{};
    CHECK(ex::webgrid::make_selection_event(config, input.pointer, snapshot.active_target,
                                            snapshot.trial, 2, input.experiment_time_ns, 1,
                                            selection) == ex::ContractStatus::ok);
    CHECK(ep::process_webgrid_selection(webgrid_controller, input, selection) ==
          neurale::streaming::StreamStatus::ok);

    std::uint64_t pointer_ordinal{};
    CHECK(webgrid_controller.latest_pointer_update_ordinal(pointer_ordinal));
    ep::SoftwarePresentationTimes webgrid_times{};
    webgrid_times.status = ep::SurfaceStatus::ok;
    webgrid_times.requested_ns = 220;
    webgrid_times.intended_ns = 220;
    webgrid_times.submitted_renderer_ns = 2'100;
    webgrid_times.submitted_ns = 221;
    webgrid_times.presented_renderer_ns = 2'110;
    webgrid_times.presented_ns = 222;
    CHECK(ep::report_webgrid_presentation(webgrid_writer, webgrid_controller, webgrid_times, 8,
                                          pointer_ordinal, webgrid_controller.snapshot().trial));
    CHECK(ep::report_webgrid_runtime_failure(webgrid_writer, webgrid_controller, 230,
                                             ep::SurfaceStatus::presentation_failed,
                                             webgrid_controller.snapshot().trial, 9));
    CHECK(webgrid_controller.apply_presentation_failures() == neurale::streaming::StreamStatus::ok);
    auto wrong_webgrid_trial = webgrid_controller.snapshot().trial;
    ++wrong_webgrid_trial.ordinal;
    CHECK(ep::report_webgrid_runtime_failure(webgrid_writer, webgrid_controller, 231,
                                             ep::SurfaceStatus::presentation_failed,
                                             wrong_webgrid_trial, 10));
    CHECK(webgrid_controller.apply_presentation_failures() ==
          neurale::streaming::StreamStatus::invalid_frame);
    CHECK(webgrid_controller.dropped_trace_count() == 1);
    execution::WebGridHeadlessTrace failure_trace{};
    bool saw_invalidated = false;
    while (webgrid_controller.try_pop_trace(failure_trace) == neurale::streaming::StreamStatus::ok)
    {
        saw_invalidated =
            saw_invalidated ||
            (failure_trace.kind == execution::WebGridTraceKind::abnormal &&
             failure_trace.abnormal.response == ex::AbnormalResponse::trial_invalidated);
    }
    CHECK(saw_invalidated);

    const auto speech_task = presentation_speech_config();
    const auto speech_catalog = presentation_catalog();
    std::array<ex::speech::SpeechTrialSchedule, 1> speech_schedules{};
    CHECK(ex::speech::prepare_trial(speech_task, 0, speech_schedules[0]) == ex::ContractStatus::ok);
    execution::SpeechHeadlessScheduler scheduler{};
    CHECK(scheduler.prepare(speech_task, speech_catalog, speech_schedules, 8) ==
          neurale::streaming::StreamStatus::ok);
    execution::SpeechTraceWriter speech_writer{scheduler, 3};
    CHECK(speech_writer.start_execution(300) == neurale::streaming::StreamStatus::ok);
    ep::SpeechPresentationEvidence speech{};
    speech.request.requested_ns = 300;
    speech.request.onset_ns = 310;
    speech.request.valid_until_ns = 400;
    speech.request.duration_ns = 90;
    speech.request.sequence = 4;
    speech.request.paradigm = 3;
    speech.request.cue = ex::CueKind::black;
    speech.outcome.requested_ns = 300;
    speech.outcome.presented_ns = 330;
    speech.outcome.sequence = 8;
    speech.outcome.request_sequence = 4;
    speech.outcome.trial = speech.request.trial;
    speech.outcome.status = ex::PresentationStatus::presented;
    speech.software_times = {.status = ep::SurfaceStatus::ok,
                             .requested_ns = 300,
                             .intended_ns = 310,
                             .submitted_renderer_ns = 3'000,
                             .submitted_ns = 320,
                             .presented_renderer_ns = 3'010,
                             .presented_ns = 330};
    speech.render_status = ep::SurfaceStatus::ok;
    CHECK(ep::report_speech_presentation(speech_writer, speech));

    speech.outcome.request_sequence = 5;
    CHECK(!ep::report_speech_presentation(speech_writer, speech));
    CHECK(ep::report_speech_lifecycle(speech_writer, execution::PresentationLifecycleEvent::faulted,
                                      350, ep::SurfaceStatus::presentation_failed));
    CHECK(!ep::report_speech_runtime_failure(speech_writer, 360, ep::SurfaceStatus::ok,
                                             ex::TrialIdentity{}));
    CHECK(ep::report_speech_runtime_failure(
        speech_writer, 361, ep::SurfaceStatus::presentation_failed, scheduler.snapshot().trial));
    CHECK(scheduler.apply_presentation_failures() == neurale::streaming::StreamStatus::ok);
    CHECK(scheduler.abnormal_summary().trials_affected == 1);

    // Config provenance must come from a prepared presenter. The ordinary
    // contract test remains display-independent and verifies the rejection
    // path; the --window invocation below exercises the positive path.
    ep::CenterOut2DPresenter center_presenter{};
    ep::WebGridPresenter webgrid_presenter{};
    ep::SpeechCuePresenter speech_presenter{};
    CHECK(!ep::record_center_out_presentation_config(center_writer, center_presenter));
    CHECK(!ep::record_webgrid_presentation_config(webgrid_writer, webgrid_presenter));
    CHECK(!ep::record_speech_presentation_config(speech_writer, speech_presenter));
    if (!exercise_presenters)
        return 0;
    const auto font_path = presentation_test_font();
    if (font_path.empty())
        return neurale::presentation_test::skip_or_fail(
            "no font: set NEURALE_PRESENTATION_TEST_FONT");

    // The positive seam reads the exact config frozen by each concrete
    // presenter. Speech additionally reads the SHA-256 computed from the font
    // bytes that presenter prepared; no caller-provided digest is accepted.
    ep::CenterOutPresentationConfig center_out_presentation{};
    center_out_presentation.geometry_unit = ex::center_out::GeometryUnit::normalized;
    center_out_presentation.logical_space = {-1.0, -1.0, 2.0, 2.0};
    center_out_presentation.style.target_radius = 0.05;
    center_out_presentation.style.cursor_radius = 0.025;
    center_out_presentation.visible = false;
    CHECK(center_presenter.open(center_out_presentation, 0, 0) == ep::SurfaceStatus::ok);
    CHECK(center_presenter.prepare(center_writer.task_configuration()) == ep::SurfaceStatus::ok);
    CHECK(ep::record_center_out_presentation_config(center_writer, center_presenter));
    CHECK(!ep::record_center_out_presentation_config(center_writer, center_presenter));

    ep::WebGridPresentationConfig webgrid_presentation{};
    webgrid_presentation.style.pointer_radius = 0.03;
    webgrid_presentation.selection_button = 0;
    webgrid_presentation.visible = false;
    CHECK(webgrid_presenter.open(webgrid_writer.task_configuration(), webgrid_presentation, 0, 0) ==
          ep::SurfaceStatus::ok);
    CHECK(webgrid_presenter.prepare() == ep::SurfaceStatus::ok);
    CHECK(ep::record_webgrid_presentation_config(webgrid_writer, webgrid_presenter));
    CHECK(!ep::record_webgrid_presentation_config(webgrid_writer, webgrid_presenter));

    ep::SpeechPresentationConfig speech_presentation{};
    speech_presentation.text.font_path = font_path;
    speech_presentation.text.pixel_height = 32;
    speech_presentation.text.atlas_width = 256;
    speech_presentation.text.atlas_height = 256;
    speech_presentation.text.max_texts = 8;
    speech_presentation.text.max_glyphs = 64;
    speech_presentation.visible = false;
    CHECK(speech_presenter.open(speech_presentation, 0, 0) == ep::SurfaceStatus::ok);
    CHECK(speech_presenter.prepare(speech_catalog) == ep::SurfaceStatus::ok);
    CHECK(ep::record_speech_presentation_config(speech_writer, speech_presenter));
    CHECK(!ep::record_speech_presentation_config(speech_writer, speech_presenter));
    speech_presenter.close();

    execution::SpeechTraceWriter mismatched_writer{scheduler, 3};
    ep::SpeechCuePresenter mismatched_speech_presenter{};
    auto mismatched_catalog = speech_catalog;
    mismatched_catalog.entries[0].text = {};
    constexpr std::string_view mismatched_text{"Different speech"};
    std::memcpy(mismatched_catalog.entries[0].text.data(), mismatched_text.data(),
                mismatched_text.size());
    mismatched_catalog.entries[0].text_length = static_cast<std::uint8_t>(mismatched_text.size());
    CHECK(mismatched_catalog.entries[0].id == speech_catalog.entries[0].id);
    CHECK(ex::speech::catalog_fingerprint(mismatched_catalog) !=
          ex::speech::catalog_fingerprint(speech_catalog));
    CHECK(mismatched_speech_presenter.open(speech_presentation, 0, 0) == ep::SurfaceStatus::ok);
    CHECK(mismatched_speech_presenter.prepare(mismatched_catalog) == ep::SurfaceStatus::ok);
    CHECK(!ep::record_speech_presentation_config(mismatched_writer, mismatched_speech_presenter));
    mismatched_speech_presenter.close();
    webgrid_presenter.close();
    center_presenter.close();
    return 0;
}

void test_bridge_handoff_is_fixed_capacity_and_reports_loss()
{
    execution::CenterOutController controller{};
    execution::CenterOutTraceWriter writer{controller, 0};
    for (std::size_t i = 0; i < 64; ++i)
    {
        CHECK(ep::report_center_out_lifecycle(writer, execution::PresentationLifecycleEvent::opened,
                                              i, ep::SurfaceStatus::ok, i));
    }
    CHECK(!ep::report_center_out_lifecycle(writer, execution::PresentationLifecycleEvent::closed,
                                           65, ep::SurfaceStatus::ok));
    CHECK(writer.dropped_trace_count() == 1);
}
} // namespace

int main(int argc, char** argv)
{
    const bool exercise_presenters = argc == 2 && std::string_view(argv[1]) == "--window";
    if (const auto result =
            test_concrete_bridges_preserve_identity_without_encoding(exercise_presenters);
        result != 0)
        return result;
    test_bridge_handoff_is_fixed_capacity_and_reports_loss();
    return failures == 0 ? 0 : 1;
}
