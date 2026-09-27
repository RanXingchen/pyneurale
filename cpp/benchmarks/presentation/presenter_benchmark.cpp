/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Presentation characterization.  Every timestamp is a software timestamp;
/// this executable provides no evidence about physical pixel/photon onset.

#include "allocation_tracker.h"
#include "benchmark_report.h"
#include "center_out_presenter.h"
#include "speech_presenter.h"
#include "webgrid_presenter.h"

#include <neurale/experiments/center_out.h>
#include <neurale/experiments/speech.h>
#include <neurale/experiments/webgrid.h>
#include <neurale/runtime/runtime_info.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
// clang-format off
#include <windows.h>
#include <psapi.h>
// clang-format on
#else
#include <sys/resource.h>
#if defined(__linux__)
#include <unistd.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif
#endif

namespace
{
using neurale::benchmark::emit_json_string;
using neurale::benchmark::percentile;
using neurale::benchmark::text_field;

#ifndef NEURALE_BENCHMARK_OS_NAME
#define NEURALE_BENCHMARK_OS_NAME "unknown"
#endif
#ifndef NEURALE_BENCHMARK_OS_VERSION
#define NEURALE_BENCHMARK_OS_VERSION "unknown"
#endif
#ifndef NEURALE_BENCHMARK_SYSTEM_PROCESSOR
#define NEURALE_BENCHMARK_SYSTEM_PROCESSOR "unknown"
#endif

namespace ep = neurale::experiment_presentation;
namespace ex = neurale::experiments;
namespace co = neurale::experiments::center_out;
namespace wg = neurale::experiments::webgrid;
namespace sp = neurale::experiments::speech;
using Clock = std::chrono::steady_clock;

struct Options
{
    std::size_t warmups{64};
    std::size_t repetitions{512};
    std::size_t long_session_frames{100'000};
    std::string font_path{};
    std::string cjk_font_path{};
    std::string command{};
};

struct Summary
{
    std::size_t samples{};
    double minimum{};
    double p50{};
    double p95{};
    double p99{};
    double maximum{};
};

struct Measurement
{
    std::string scenario{};
    std::string text_kind{"not_applicable"};
    std::string metric{"operation_time"};
    std::string unit{"ns"};
    Summary summary{};
    std::size_t warmups{};
    std::size_t repetitions{};
    std::size_t allocations{};
    double cpu_percent{};
    std::uint64_t rss_before_bytes{};
    std::uint64_t rss_after_bytes{};
    std::uint64_t peak_rss_bytes{};
    std::uint64_t shaping_preparation_ns{};
    std::uint64_t glyph_cache_preparation_ns{};
    std::uint64_t total_preparation_ns{};
    std::uint64_t unexpected_runtime_glyph_cache_misses{};
    std::size_t prepared_text_count{};
    std::size_t prepared_glyph_count{};
    bool resources_stable{};
    bool passed{};
};

[[nodiscard]] Summary summarize(std::vector<std::uint64_t> values)
{
    std::sort(values.begin(), values.end());
    return {values.size(),
            static_cast<double>(values.front()),
            percentile(values, 0.5),
            percentile(values, 0.95),
            percentile(values, 0.99),
            static_cast<double>(values.back())};
}

[[nodiscard]] std::uint64_t current_rss_bytes() noexcept
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters)) == 0)
        return 0;
    return static_cast<std::uint64_t>(counters.WorkingSetSize);
#else
#if defined(__linux__)
    std::ifstream statm{"/proc/self/statm"};
    std::uint64_t total_pages{};
    std::uint64_t resident_pages{};
    if (!(statm >> total_pages >> resident_pages))
        return 0;
    const auto page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0 || resident_pages > (std::numeric_limits<std::uint64_t>::max)() /
                                               static_cast<std::uint64_t>(page_size))
        return 0;
    return resident_pages * static_cast<std::uint64_t>(page_size);
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) != KERN_SUCCESS)
        return 0;
    return static_cast<std::uint64_t>(info.resident_size);
#else
    return 0;
#endif
#endif
}

[[nodiscard]] std::uint64_t peak_rss_bytes() noexcept
{
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX counters{};
    counters.cb = sizeof(counters);
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&counters),
                             sizeof(counters)) == 0)
        return 0;
    return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#else
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0)
        return 0;
#if defined(__APPLE__)
    return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
    return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
#endif
#endif
}

template <typename Operation>
Measurement measure(const Options& options, std::string scenario, Operation&& operation)
{
    for (std::size_t i = 0; i < options.warmups; ++i)
        if (!operation(i, false))
            return {.scenario = std::move(scenario), .passed = false};

    std::vector<std::uint64_t> values(options.repetitions);
    const auto rss_before = current_rss_bytes();
    const auto cpu_before = std::clock();
    const auto wall_begin = Clock::now();
    bool passed = true;
    std::size_t allocations{};
    {
        neurale::benchmark::AllocationScope scope{};
        for (std::size_t i = 0; i < options.repetitions; ++i)
        {
            const auto begin = Clock::now();
            passed = operation(i, true) && passed;
            values[i] = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
        }
        allocations = scope.count();
    }
    const auto wall_seconds = std::chrono::duration<double>(Clock::now() - wall_begin).count();
    const auto cpu_seconds = static_cast<double>(std::clock() - cpu_before) / CLOCKS_PER_SEC;
    return {.scenario = std::move(scenario),
            .summary = summarize(std::move(values)),
            .warmups = options.warmups,
            .repetitions = options.repetitions,
            .allocations = allocations,
            .cpu_percent = wall_seconds > 0.0 ? 100.0 * cpu_seconds / wall_seconds : 0.0,
            .rss_before_bytes = rss_before,
            .rss_after_bytes = current_rss_bytes(),
            .peak_rss_bytes = peak_rss_bytes(),
            .passed = passed};
}

template <typename Operation>
Measurement measure_observed(const Options& options, std::string scenario, Operation&& operation)
{
    for (std::size_t i = 0; i < options.warmups; ++i)
        if (!operation(i).first)
            return {.scenario = std::move(scenario), .passed = false};
    std::vector<std::uint64_t> values(options.repetitions);
    const auto rss_before = current_rss_bytes();
    const auto cpu_before = std::clock();
    const auto wall_begin = Clock::now();
    bool passed = true;
    std::size_t allocations{};
    {
        neurale::benchmark::AllocationScope scope{};
        for (std::size_t i = 0; i < options.repetitions; ++i)
        {
            const auto [ok, value] = operation(i);
            passed = ok && passed;
            values[i] = value;
        }
        allocations = scope.count();
    }
    const auto wall_seconds = std::chrono::duration<double>(Clock::now() - wall_begin).count();
    const auto cpu_seconds = static_cast<double>(std::clock() - cpu_before) / CLOCKS_PER_SEC;
    auto result =
        Measurement{.scenario = std::move(scenario),
                    .metric = "request_to_present_software_latency",
                    .summary = summarize(std::move(values)),
                    .warmups = options.warmups,
                    .repetitions = options.repetitions,
                    .allocations = allocations,
                    .cpu_percent = wall_seconds > 0.0 ? 100.0 * cpu_seconds / wall_seconds : 0.0,
                    .rss_before_bytes = rss_before,
                    .rss_after_bytes = current_rss_bytes(),
                    .peak_rss_bytes = peak_rss_bytes(),
                    .resources_stable = true,
                    .passed = passed};
    return result;
}

template <typename Value> void field(std::string_view name, Value value)
{
    std::cout << ',';
    emit_json_string(name);
    std::cout << ':' << value;
}

[[nodiscard]] std::string_view c_string(const auto& value) noexcept
{
    return {value.data(), std::char_traits<char>::length(value.data())};
}

void emit(const Options& options, const ep::PresentationEnvironment& environment,
          const Measurement& result)
{
    const auto build = neurale::runtime::build_info();
    const auto cpu = neurale::runtime::cpu_info();
    std::cout << "{\"schema_version\":1";
    text_field("benchmark", "m9_experiment_presentation");
    text_field("scenario", result.scenario);
    text_field("text_kind", result.text_kind);
    text_field("metric", result.metric);
    text_field("unit", result.unit);
    field("samples", result.summary.samples);
    field("min_ns", result.summary.minimum);
    field("p50_ns", result.summary.p50);
    field("p95_ns", result.summary.p95);
    field("p99_ns", result.summary.p99);
    field("max_ns", result.summary.maximum);
    field("warmups", result.warmups);
    field("repetitions", result.repetitions);
    field("allocations", result.allocations);
    field("cpu_percent", result.cpu_percent);
    field("rss_before_bytes", result.rss_before_bytes);
    field("rss_after_bytes", result.rss_after_bytes);
    field("peak_rss_bytes", result.peak_rss_bytes);
    field("total_preparation_ns", result.total_preparation_ns);
    field("shaping_preparation_ns", result.shaping_preparation_ns);
    field("glyph_cache_preparation_ns", result.glyph_cache_preparation_ns);
    field("unexpected_runtime_glyph_cache_misses", result.unexpected_runtime_glyph_cache_misses);
    field("prepared_text_count", result.prepared_text_count);
    field("prepared_glyph_count", result.prepared_glyph_count);
    std::cout << ",\"resources_stable\":" << (result.resources_stable ? "true" : "false");
    std::cout << ",\"passed\":" << (result.passed ? "true" : "false");
    text_field("command", options.command);
    text_field("os_name", NEURALE_BENCHMARK_OS_NAME);
    text_field("os_version", NEURALE_BENCHMARK_OS_VERSION);
    text_field("system_processor", NEURALE_BENCHMARK_SYSTEM_PROCESSOR);
    text_field("cpu_architecture", cpu.architecture);
    text_field("cpu_vendor", cpu.vendor.value_or(""));
    text_field("cpu_model", cpu.model.value_or(""));
    text_field("compiler", build.compiler);
    text_field("build_type", build.build_type);
    text_field("gpu_vendor", c_string(environment.gpu_vendor));
    text_field("gpu_renderer", c_string(environment.gpu_renderer));
    text_field("opengl_version", c_string(environment.opengl_version));
    field("display_refresh_rate_hz", environment.refresh_rate_hz);
    field("monitor_index", environment.monitor_idx);
    field("window_width", environment.window_width);
    field("window_height", environment.window_height);
    field("framebuffer_width", environment.framebuffer_width);
    field("framebuffer_height", environment.framebuffer_height);
    field("swap_interval", environment.swap_interval);
    std::cout << ",\"fullscreen\":" << (environment.fullscreen ? "true" : "false");
    text_field("font_path", options.font_path);
    text_field("cjk_font_path", options.cjk_font_path);
    text_field("timing_claim", "software_swap_return_only");
    std::cout << "}\n";
}

[[nodiscard]] co::CenterOut2DConfig center_task()
{
    co::RadialLayoutRequest request{};
    request.radius = 0.7;
    request.count = 8;
    request.center_id = 1;
    for (std::uint8_t i = 0; i < request.count; ++i)
    {
        request.ids[i] = static_cast<ex::TargetId>(i + 2);
        request.spokes[i] = i;
    }
    co::CenterOut2DConfig task{};
    task.geometry_unit = co::GeometryUnit::normalized;
    if (co::build_radial_layout(request, task.layout) != ex::ContractStatus::ok)
        return {};
    task.acceptance = {0.08, 0.08};
    task.cursor = {0.02};
    task.movement_timeout = {1'000'000'000, 1'000'000'000};
    task.hold_ns = 100'000'000;
    task.reward_dwell = {100'000'000, 100'000'000};
    task.punish_dwell = {100'000'000, 100'000'000};
    task.selection = co::TargetSelectionPolicy::repeat_until_success;
    task.seed = 17;
    return task;
}

[[nodiscard]] ep::CenterOutPresentationConfig center_presentation()
{
    ep::CenterOutPresentationConfig config{};
    config.geometry_unit = co::GeometryUnit::normalized;
    config.logical_space = {-1.0, -1.0, 2.0, 2.0};
    config.style.target_radius = 0.06;
    config.style.cursor_radius = 0.025;
    config.style.circle_segments = 48;
    config.window_size = {800, 600};
    config.monitor_idx = 0;
    config.swap_interval = 0;
    config.input_capacity = 256;
    config.visible = false;
    return config;
}

[[nodiscard]] wg::WebGridConfig webgrid_task()
{
    wg::WebGridConfig task{};
    task.rows = 12;
    task.columns = 12;
    task.bounds = {0.0, 12.0, 0.0, 12.0};
    task.n_candidates = 144;
    for (std::uint16_t i = 0; i < task.n_candidates; ++i)
        task.candidates[i] = i + 1;
    task.schedule = wg::TargetScheduleKind::seeded;
    task.immediate_repetition = wg::ImmediateRepetitionPolicy::forbid;
    task.correct_selection = wg::CorrectSelectionPolicy::advance_target;
    task.incorrect_selection = wg::IncorrectSelectionPolicy::keep_current_target;
    task.initial_target = 1;
    task.seed = 23;
    task.metric_version = wg::kMetricVersion1;
    return task;
}

[[nodiscard]] ep::WebGridPresentationConfig webgrid_presentation()
{
    ep::WebGridPresentationConfig config{};
    config.style.pointer_radius = 0.08;
    config.style.circle_segments = 32;
    config.window_size = {800, 600};
    config.monitor_idx = 0;
    config.swap_interval = 0;
    config.input_capacity = 256;
    config.visible = false;
    return config;
}

void set_text(sp::SpeechStimulus& stimulus, ex::StimulusId id, std::string_view text)
{
    stimulus.id = id;
    stimulus.content = sp::SpeechContentKind::text;
    stimulus.text_length = static_cast<std::uint8_t>(text.size());
    std::memcpy(stimulus.text.data(), text.data(), text.size());
}

[[nodiscard]] sp::SpeechCatalog speech_catalog()
{
    sp::SpeechCatalog catalog{};
    catalog.count = 4;
    set_text(catalog.entries[0], 1, "Ready");
    set_text(catalog.entries[1], 2, "\xE8\xAF\xB7\xE5\x87\x86\xE5\xA4\x87");
    set_text(catalog.entries[2], 3, "Ready \xE5\x87\x86\xE5\xA4\x87 123");
    set_text(catalog.entries[3], 4, "Please prepare and speak the prompted phrase clearly.");
    return catalog;
}

[[nodiscard]] ep::SpeechPresentationConfig speech_presentation(const Options& options)
{
    ep::SpeechPresentationConfig config{};
    config.text = {.font_path = options.cjk_font_path,
                   .face_idx = 0,
                   .pixel_height = 32,
                   .atlas_width = 2048,
                   .atlas_height = 2048,
                   .max_texts = 16,
                   .max_glyphs = 512};
    config.window_size = {800, 600};
    config.monitor_idx = 0;
    config.swap_interval = 0;
    config.input_capacity = 64;
    config.outcome_capacity = 64;
    config.visible = false;
    return config;
}

[[nodiscard]] std::string environment_path(const char* name)
{
    const auto* value = std::getenv(name);
    return value == nullptr ? std::string{} : std::string(value);
}

[[nodiscard]] std::string first_existing(std::initializer_list<std::string_view> candidates)
{
    for (const auto candidate : candidates)
        if (!candidate.empty() && std::filesystem::is_regular_file(candidate))
            return std::string(candidate);
    return {};
}

[[nodiscard]] bool parse(Options& options, int argc, char** argv)
{
    options.font_path = environment_path("NEURALE_PRESENTATION_TEST_FONT");
    options.cjk_font_path = environment_path("NEURALE_PRESENTATION_TEST_CJK_FONT");
    options.command.clear();
    for (int i = 0; i < argc; ++i)
    {
        if (i != 0)
            options.command += ' ';
        options.command += argv[i];
    }
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view argument(argv[i]);
        if (i + 1 >= argc)
            return false;
        const std::string_view value(argv[++i]);
        if (argument == "--font")
            options.font_path = value;
        else if (argument == "--cjk-font")
            options.cjk_font_path = value;
        else if (argument == "--warmups")
            options.warmups = std::stoull(std::string(value));
        else if (argument == "--repetitions")
            options.repetitions = std::stoull(std::string(value));
        else if (argument == "--long-session-frames")
            options.long_session_frames = std::stoull(std::string(value));
        else
            return false;
    }
#ifdef _WIN32
    if (options.font_path.empty())
        options.font_path = first_existing({"C:/Windows/Fonts/arial.ttf"});
    if (options.cjk_font_path.empty())
        options.cjk_font_path =
            first_existing({"C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"});
#else
    if (options.font_path.empty())
        options.font_path = first_existing({"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"});
    if (options.cjk_font_path.empty())
        options.cjk_font_path =
            first_existing({"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                            "/usr/share/fonts/opentype/noto/NotoSansCJKSC-Regular.otf"});
#endif
    return options.warmups > 0 && options.repetitions > 0 && options.long_session_frames > 0 &&
           std::filesystem::is_regular_file(options.font_path) &&
           std::filesystem::is_regular_file(options.cjk_font_path);
}

int run(const Options& options)
{
    std::vector<Measurement> results;
    ep::PresentationEnvironment environment{};

    {
        auto task = center_task();
        co::CenterOutMachine machine{};
        co::CenterOutStepResult step{};
        if (machine.start(7, task, 0, step) != ex::ContractStatus::ok)
            return 1;
        ep::CenterOut2DPresenter presenter{};
        const auto origin = ep::renderer_monotonic_now_ns();
        if (presenter.open(center_presentation(), origin, 0) != ep::SurfaceStatus::ok ||
            presenter.prepare(task) != ep::SurfaceStatus::ok)
            return 1;
        environment = presenter.environment();
        auto result =
            measure(options, "center_out_snapshot_update_render",
                    [&](std::size_t, bool)
                    {
                        const co::WorkspacePoint cursor{0.01, -0.01};
                        if (presenter.update(step.snapshot, cursor) != ep::SurfaceStatus::ok)
                            return false;
                        const auto now = ep::renderer_monotonic_now_ns() - origin;
                        return presenter.render(now, now).status == ep::SurfaceStatus::ok;
                    });
        const auto stats = presenter.resource_stats();
        result.resources_stable = stats.vertex_capacity > 0 && stats.batch_capacity > 0;
        result.passed = result.passed && result.resources_stable;
        results.push_back(result);
        presenter.close();
    }

    {
        auto task = webgrid_task();
        wg::WebGridMachine machine{};
        wg::WebGridStepResult step{};
        if (machine.start(11, task, 0, step) != ex::ContractStatus::ok)
            return 1;
        ep::WebGridPresenter presenter{};
        const auto origin = ep::renderer_monotonic_now_ns();
        if (presenter.open(task, webgrid_presentation(), origin, 0) != ep::SurfaceStatus::ok ||
            presenter.prepare() != ep::SurfaceStatus::ok)
            return 1;
        environment = presenter.environment();
        results.push_back(
            measure(options, "webgrid_update_render_12x12",
                    [&](std::size_t, bool)
                    {
                        const wg::PointerPosition pointer{5.5, 5.5};
                        if (presenter.update(step.snapshot, pointer) != ep::SurfaceStatus::ok)
                            return false;
                        const auto now = ep::renderer_monotonic_now_ns() - origin;
                        return presenter.render(now, now).status == ep::SurfaceStatus::ok;
                    }));

        ep::WebGridPointerInputAdapter adapter{};
        if (adapter.prepare(0) != ep::SurfaceStatus::ok)
            return 1;
        std::uint64_t ordinal{};
        auto pointer_result =
            measure(options, "webgrid_pointer_to_logical_event",
                    [&](std::size_t, bool)
                    {
                        ep::Point2d logical{};
                        bool inside{};
                        if (!presenter.coordinates().window_pointer_to_logical_unclipped(
                                {400.0, 300.0}, logical, inside))
                            return false;
                        ep::InputEvent input{.ordinal = ++ordinal,
                                             .kind = ep::InputEventKind::pointer_moved,
                                             .pointer = logical,
                                             .pointer_inside = inside};
                        ep::WebGridPresentationInput output{};
                        bool available{};
                        return adapter.adapt(input, output, available) == ep::SurfaceStatus::ok &&
                               available && output.inside_presentation;
                    });
        pointer_result.resources_stable = true;
        results.back().resources_stable = true;
        results.push_back(std::move(pointer_result));
        presenter.close();
    }

    {
        const auto catalog = speech_catalog();
        ep::SpeechCuePresenter presenter{};
        const auto origin = ep::renderer_monotonic_now_ns();
        if (presenter.open(speech_presentation(options), origin, 0) != ep::SurfaceStatus::ok ||
            presenter.prepare(catalog) != ep::SurfaceStatus::ok)
            return 1;
        environment = presenter.environment();
        const auto prepared = presenter.resource_stats();
        const std::array text_kinds{"ascii", "chinese", "mixed", "representative"};
        for (std::size_t text_idx = 0; text_idx < text_kinds.size(); ++text_idx)
        {
            std::uint64_t sequence = text_idx * (options.warmups + options.repetitions) + 1;
            auto result = measure(
                options, "speech_prepared_text_render",
                [&](std::size_t, bool)
                {
                    const auto requested = ep::renderer_monotonic_now_ns() - origin;
                    ex::PresentationRequest request{
                        .requested_ns = requested,
                        .onset_ns = requested,
                        .valid_until_ns = requested + 1'000'000'000,
                        .duration_ns = 100'000'000,
                        .sequence = sequence++,
                        .trial = {.ordinal = sequence,
                                  .stimulus_id = static_cast<ex::StimulusId>(text_idx + 1)},
                        .paradigm = 31,
                        .phase = static_cast<ex::PhaseId>(sp::SpeechPhase::content),
                        .cue = ex::CueKind::text_content,
                        .stimulus_id = static_cast<ex::StimulusId>(text_idx + 1),
                    };
                    const auto presented = presenter.present(request, requested);
                    ep::SpeechPresentationEvidence evidence{};
                    bool available{};
                    return presented.status == ep::SurfaceStatus::ok &&
                           presenter.poll_outcome(evidence, available) == ep::SurfaceStatus::ok &&
                           available;
                });
            result.text_kind = text_kinds[text_idx];
            result.prepared_text_count = prepared.prepared_text_count;
            result.prepared_glyph_count = prepared.prepared_glyph_count;
            const auto current = presenter.resource_stats();
            result.unexpected_runtime_glyph_cache_misses =
                current.unexpected_runtime_glyph_cache_misses;
            result.resources_stable =
                current.prepared_text_count == prepared.prepared_text_count &&
                current.prepared_glyph_count == prepared.prepared_glyph_count &&
                current.unexpected_runtime_glyph_cache_misses == 0;
            result.passed = result.passed && result.resources_stable;
            results.push_back(std::move(result));
        }

        std::uint64_t latency_sequence = 900'000;
        auto latency = measure_observed(
            options, "speech_request_to_present_software_latency",
            [&](std::size_t)
            {
                const auto requested = ep::renderer_monotonic_now_ns() - origin;
                ex::PresentationRequest request{
                    .requested_ns = requested,
                    .onset_ns = requested,
                    .valid_until_ns = requested + 1'000'000'000,
                    .duration_ns = 100'000'000,
                    .sequence = latency_sequence++,
                    .trial = {.ordinal = latency_sequence, .stimulus_id = 3},
                    .paradigm = 31,
                    .phase = static_cast<ex::PhaseId>(sp::SpeechPhase::content),
                    .cue = ex::CueKind::text_content,
                    .stimulus_id = 3,
                };
                const auto presented = presenter.present(request, requested);
                ep::SpeechPresentationEvidence evidence{};
                bool available{};
                const bool ok =
                    presented.status == ep::SurfaceStatus::ok &&
                    presenter.poll_outcome(evidence, available) == ep::SurfaceStatus::ok &&
                    available && evidence.software_times.presented_ns >= requested;
                return std::pair{ok, ok ? evidence.software_times.presented_ns - requested
                                        : std::uint64_t{0}};
            });
        latency.text_kind = "mixed";
        latency.prepared_text_count = prepared.prepared_text_count;
        latency.prepared_glyph_count = prepared.prepared_glyph_count;
        results.push_back(std::move(latency));

        Measurement preparation{};
        preparation.scenario = "speech_catalog_prepare";
        preparation.metric = "preparation_time";
        preparation.summary = {1,
                               static_cast<double>(prepared.text_preparation_ns),
                               static_cast<double>(prepared.text_preparation_ns),
                               static_cast<double>(prepared.text_preparation_ns),
                               static_cast<double>(prepared.text_preparation_ns),
                               static_cast<double>(prepared.text_preparation_ns)};
        preparation.total_preparation_ns = prepared.text_preparation_ns;
        preparation.shaping_preparation_ns = prepared.shaping_preparation_ns;
        preparation.glyph_cache_preparation_ns = prepared.glyph_cache_preparation_ns;
        preparation.prepared_text_count = prepared.prepared_text_count;
        preparation.prepared_glyph_count = prepared.prepared_glyph_count;
        preparation.resources_stable = true;
        preparation.passed = prepared.text_preparation_ns > 0 &&
                             prepared.shaping_preparation_ns > 0 &&
                             prepared.glyph_cache_preparation_ns > 0;
        results.push_back(preparation);

        const auto rss_before = current_rss_bytes();
        const auto before = presenter.resource_stats();
        bool stable = true;
        std::uint64_t sequence = 1'000'000;
        const auto cpu_before = std::clock();
        const auto begin = Clock::now();
        std::size_t allocations{};
        {
            neurale::benchmark::AllocationScope scope{};
            for (std::size_t i = 0; i < options.long_session_frames; ++i)
            {
                const auto requested = ep::renderer_monotonic_now_ns() - origin;
                const auto stimulus = static_cast<ex::StimulusId>(i % 4 + 1);
                ex::PresentationRequest request{
                    .requested_ns = requested,
                    .onset_ns = requested,
                    .valid_until_ns = requested + 1'000'000'000,
                    .duration_ns = 100'000'000,
                    .sequence = sequence++,
                    .trial = {.ordinal = sequence, .stimulus_id = stimulus},
                    .paradigm = 31,
                    .phase = static_cast<ex::PhaseId>(sp::SpeechPhase::content),
                    .cue = ex::CueKind::text_content,
                    .stimulus_id = stimulus,
                };
                const auto result = presenter.present(request, requested);
                ep::SpeechPresentationEvidence evidence{};
                bool available{};
                stable = stable && result.status == ep::SurfaceStatus::ok &&
                         presenter.poll_outcome(evidence, available) == ep::SurfaceStatus::ok &&
                         available;
            }
            allocations = scope.count();
        }
        const auto after = presenter.resource_stats();
        Measurement long_session{};
        long_session.scenario = "speech_repeated_long_session";
        long_session.metric = "total_time";
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - begin).count());
        long_session.summary = {1,
                                static_cast<double>(elapsed),
                                static_cast<double>(elapsed),
                                static_cast<double>(elapsed),
                                static_cast<double>(elapsed),
                                static_cast<double>(elapsed)};
        long_session.repetitions = options.long_session_frames;
        long_session.allocations = allocations;
        const auto wall_seconds = static_cast<double>(elapsed) / 1'000'000'000.0;
        const auto cpu_seconds = static_cast<double>(std::clock() - cpu_before) / CLOCKS_PER_SEC;
        long_session.cpu_percent = wall_seconds > 0.0 ? 100.0 * cpu_seconds / wall_seconds : 0.0;
        long_session.rss_before_bytes = rss_before;
        long_session.rss_after_bytes = current_rss_bytes();
        long_session.peak_rss_bytes = peak_rss_bytes();
        long_session.prepared_text_count = after.prepared_text_count;
        long_session.prepared_glyph_count = after.prepared_glyph_count;
        long_session.unexpected_runtime_glyph_cache_misses =
            after.unexpected_runtime_glyph_cache_misses;
        long_session.resources_stable = before.vertex_capacity == after.vertex_capacity &&
                                        before.batch_capacity == after.batch_capacity &&
                                        before.prepared_text_count == after.prepared_text_count &&
                                        before.prepared_glyph_count == after.prepared_glyph_count &&
                                        after.unexpected_runtime_glyph_cache_misses == 0;
        long_session.passed = stable && long_session.resources_stable;
        results.push_back(long_session);
        presenter.close();
    }

    bool passed = true;
    for (const auto& result : results)
    {
        emit(options, environment, result);
        passed = result.passed && passed;
    }
    return passed ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    Options options{};
    try
    {
        if (!parse(options, argc, argv))
        {
            std::cerr << "usage: neurale_experiment_presentation_benchmark "
                         "[--font PATH] [--cjk-font PATH] [--warmups N] "
                         "[--repetitions N] [--long-session-frames N]\n";
            return 2;
        }
        return run(options);
    }
    catch (const std::exception& error)
    {
        std::cerr << "presentation benchmark failed: " << error.what() << '\n';
        return 1;
    }
}
