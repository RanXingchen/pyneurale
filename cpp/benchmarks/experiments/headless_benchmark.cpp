/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// Measured characterization for the public experiment semantics and the
/// private headless execution.  This is evidence for this executable and
/// host only; it is not a portable realtime threshold.

#include "allocation_tracker.h"
#include "benchmark_report.h"
#include "center_out_controller.h"
#include "speech_controller.h"
#include "webgrid_controller.h"

#include <neurale/experiments/assistance.h>
#include <neurale/experiments/center_out.h>
#include <neurale/experiments/center_out_guidance.h>
#include <neurale/experiments/center_out_replay.h>
#include <neurale/experiments/speech.h>
#include <neurale/experiments/speech_replay.h>
#include <neurale/experiments/webgrid.h>
#include <neurale/experiments/webgrid_replay.h>
#include <neurale/runtime/runtime_info.h>
#include <neurale/streaming/schema.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using neurale::benchmark::bool_field;
using neurale::benchmark::emit_json_string;
using neurale::benchmark::platform_name;
using neurale::benchmark::summarize;
using neurale::benchmark::Summary;
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

namespace ex = neurale::experiments;
namespace co = neurale::experiments::center_out;
namespace wg = neurale::experiments::webgrid;
namespace sp = neurale::experiments::speech;
namespace execution = neurale::execution;
namespace st = neurale::streaming;

using Clock = std::chrono::steady_clock;

struct Options
{
    std::size_t warmups{256};
    std::size_t repetitions{4096};
    std::size_t metric_records{4096};
    std::size_t replay_inputs{1024};
    std::size_t speech_trials{128};
};

struct Result
{
    std::string_view paradigm{};
    std::string_view scenario{};
    std::string_view configuration{};
    std::string_view metric{"latency"};
    std::string_view unit{"ns"};
    Summary summary{};
    double throughput_per_second{};
    std::string_view throughput_item{"operation"};
    std::size_t allocations{};
    std::string_view allocation_scope{"steady_state"};
    std::size_t n_inputs{};
    std::size_t warmups{};
    std::size_t repetitions{};
    std::uint32_t grid_rows{};
    std::uint32_t grid_columns{};
    std::size_t trace_capacity{};
    std::uint64_t trace_drops{};
    std::uint16_t metric_version{};
    std::size_t observations_per_frame{};
    std::size_t operations_per_sample{1};
    std::string_view replay_verdict{"not_applicable"};
    std::string_view replay_completeness{"not_applicable"};
    bool passed{};
};

std::string g_command{};

template <typename Operation>
[[nodiscard]] Result measure(const Options& options, Result result, Operation&& operation)
{
    std::vector<std::uint64_t> samples(options.repetitions);
    bool passed = true;
    for (std::size_t i = 0; i < options.warmups; ++i)
    {
        passed = operation(false, i) && passed;
    }

    std::size_t allocations = 0;
    {
        neurale::benchmark::AllocationScope scope{};
        for (std::size_t i = 0; i < options.repetitions; ++i)
        {
            const auto begin = Clock::now();
            passed = operation(true, i) && passed;
            const auto end = Clock::now();
            samples[i] =
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) /
                result.operations_per_sample;
        }
        allocations = scope.count();
    }
    result.summary = summarize(samples);
    result.throughput_per_second =
        result.summary.median > 0.0 ? 1.0e9 / result.summary.median : 0.0;
    result.allocations = allocations;
    result.warmups = options.warmups;
    result.repetitions = options.repetitions;
    result.passed = passed;
    return result;
}

template <typename Setup, typename Operation>
[[nodiscard]] Result measure_prepared(const Options& options, Result result, Setup&& setup,
                                      Operation&& operation)
{
    std::vector<std::uint64_t> samples(options.repetitions);
    bool passed = true;
    for (std::size_t i = 0; i < options.warmups; ++i)
    {
        passed = setup(false, i) && passed;
        passed = operation(false, i) && passed;
    }

    std::size_t allocations = 0;
    {
        neurale::benchmark::AllocationScope scope{};
        for (std::size_t i = 0; i < options.repetitions; ++i)
        {
            passed = setup(true, i) && passed;
            const auto begin = Clock::now();
            passed = operation(true, i) && passed;
            const auto end = Clock::now();
            samples[i] =
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count()) /
                result.operations_per_sample;
        }
        allocations = scope.count();
    }
    result.summary = summarize(samples);
    result.throughput_per_second =
        result.summary.median > 0.0 ? 1.0e9 / result.summary.median : 0.0;
    result.allocations = allocations;
    result.warmups = options.warmups;
    result.repetitions = options.repetitions;
    result.passed = passed;
    return result;
}

template <typename Value> void number_field(std::string_view name, Value value)
{
    std::cout << ',';
    emit_json_string(name);
    std::cout << ':' << value;
}

void emit(const Options& options, const Result& result)
{
    static const auto build = neurale::runtime::build_info();
    static const auto cpu = neurale::runtime::cpu_info();
    static const auto threading = neurale::runtime::threading_info();

    std::cout << "{\"schema_version\":1";
    text_field("benchmark", "m8_experiments");
    text_field("paradigm", result.paradigm);
    text_field("scenario", result.scenario);
    text_field("configuration", result.configuration);
    text_field("metric", result.metric);
    text_field("unit", result.unit);
    number_field("samples", result.summary.samples);
    number_field("min_ns", result.summary.minimum);
    number_field("median_ns", result.summary.median);
    number_field("p50_ns", result.summary.median);
    number_field("p95_ns", result.summary.p95);
    number_field("p99_ns", result.summary.p99);
    number_field("max_ns", result.summary.maximum);
    number_field("throughput_per_second", result.throughput_per_second);
    text_field("throughput_item", result.throughput_item);
    number_field("allocations", result.allocations);
    text_field("allocation_scope", result.allocation_scope);
    text_field("allocation_backend", neurale::benchmark::allocation_tracking_backend());
    number_field("warmups", result.warmups);
    number_field("repetitions", result.repetitions);
    number_field("n_inputs", result.n_inputs);
    number_field("metric_records", options.metric_records);
    number_field("replay_inputs_configured", options.replay_inputs);
    number_field("speech_trials_configured", options.speech_trials);
    number_field("grid_rows", result.grid_rows);
    number_field("grid_columns", result.grid_columns);
    number_field("trace_capacity", result.trace_capacity);
    number_field("trace_drops", result.trace_drops);
    number_field("metric_version", result.metric_version);
    number_field("observations_per_frame", result.observations_per_frame);
    number_field("operations_per_sample", result.operations_per_sample);
    text_field("replay_verdict", result.replay_verdict);
    text_field("replay_completeness", result.replay_completeness);
    text_field("command", g_command);
    text_field("platform", platform_name());
    text_field("os_name", NEURALE_BENCHMARK_OS_NAME);
    text_field("os_version", NEURALE_BENCHMARK_OS_VERSION);
    text_field("system_processor", NEURALE_BENCHMARK_SYSTEM_PROCESSOR);
    text_field("native_version", build.version);
    number_field("native_abi_version", build.abi_version);
    text_field("compiler", build.compiler);
    text_field("build_type", build.build_type);
    text_field("cpu_math_backend", build.cpu_math_backend);
    text_field("fft_backend", build.fft_backend);
    bool_field("cuda_compiled", build.cuda_compiled);
    text_field("cpu_architecture", cpu.architecture);
    text_field("cpu_vendor", cpu.vendor.value_or(""));
    text_field("cpu_model", cpu.model.value_or(""));
    number_field("logical_cores", cpu.logical_cores);
    text_field("threading_backend", threading.backend);
    number_field("threads", threading.num_threads);
    bool_field("passed", result.passed);
    std::cout << "}\n";
}

[[nodiscard]] co::CenterOut2DConfig center_config()
{
    co::RadialLayoutRequest request{};
    request.radius = 0.5;
    request.count = 8;
    request.center_id = 1;
    for (std::uint8_t i = 0; i < request.count; ++i)
    {
        request.ids[i] = static_cast<ex::TargetId>(i + 2);
        request.spokes[i] = i;
    }
    co::CenterOut2DConfig config{};
    config.geometry_unit = co::GeometryUnit::normalized;
    if (co::build_radial_layout(request, config.layout) != ex::ContractStatus::ok)
    {
        return {};
    }
    config.acceptance = {0.05, 0.05};
    config.cursor = {0.0};
    config.movement_timeout = {1'000'000'000'000ULL, 1'000'000'000'000ULL};
    config.selection = co::TargetSelectionPolicy::repeat_until_success;
    config.seed = 17;
    return config;
}

[[nodiscard]] ex::CommandSpace velocity_space()
{
    ex::CommandSpace space{};
    space.id = 11;
    space.dim = 2;
    space.frame = ex::CommandFrame::workspace_2d;
    space.axes[0] = {ex::CommandAxisName::x, ex::CommandUnit::normalized};
    space.axes[1] = {ex::CommandAxisName::y, ex::CommandUnit::normalized};
    return space;
}

[[nodiscard]] wg::WebGridConfig webgrid_config(std::uint16_t side)
{
    wg::WebGridConfig config{};
    config.rows = side;
    config.columns = side;
    config.bounds = {0.0, static_cast<double>(side), 0.0, static_cast<double>(side)};
    config.n_candidates = static_cast<std::uint16_t>(side * side);
    for (std::uint16_t i = 0; i < config.n_candidates; ++i)
    {
        config.candidates[i] = static_cast<ex::TargetId>(i + 1);
    }
    config.schedule = wg::TargetScheduleKind::seeded;
    config.immediate_repetition = wg::ImmediateRepetitionPolicy::forbid;
    config.correct_selection = wg::CorrectSelectionPolicy::advance_target;
    config.incorrect_selection = wg::IncorrectSelectionPolicy::keep_current_target;
    config.initial_target = 1;
    config.seed = 23;
    config.metric_version = wg::kMetricVersion1;
    return config;
}

[[nodiscard]] sp::SpeechCatalog speech_catalog(std::size_t trials)
{
    sp::SpeechCatalog catalog{};
    catalog.count = static_cast<std::uint16_t>(trials);
    for (std::size_t i = 0; i < trials; ++i)
    {
        auto& stimulus = catalog.entries[i];
        stimulus.id = static_cast<ex::StimulusId>(i + 1);
        stimulus.label = static_cast<std::uint32_t>(i + 100);
        stimulus.content = sp::SpeechContentKind::text;
        const auto text = std::to_string(i + 1);
        stimulus.text_length = static_cast<std::uint8_t>(text.size());
        std::memcpy(stimulus.text.data(), text.data(), text.size());
    }
    return catalog;
}

[[nodiscard]] sp::SpeechCueConfig speech_config(std::size_t trials)
{
    sp::SpeechCueConfig config{};
    config.black_bound_ns = 1'000'000;
    config.cross_bound_ns = 1'000'000;
    config.content_bound_ns = 2'000'000;
    config.seed = 29;
    config.n_trials = static_cast<ex::TrialOrdinal>(trials);
    config.schedule = sp::SpeechScheduleKind::seeded;
    config.stimulus_order = sp::StimulusOrderPolicy::sequential;
    config.sampler_version = ex::kCurrentSamplerVersion;
    config.cross_enabled = true;
    config.n_stimuli = static_cast<std::uint16_t>(trials);
    for (std::size_t i = 0; i < trials; ++i)
    {
        config.stimuli[i] = static_cast<ex::StimulusId>(i + 1);
    }
    return config;
}

[[nodiscard]] st::StreamSchema decoded_schema()
{
    const st::SignalSchema signal{9,
                                  st::SignalDType::float64,
                                  2,
                                  2,
                                  4,
                                  st::RationalRate{1'000, 1},
                                  7,
                                  st::SignalLayout::sample_major,
                                  st::DeviceTickTracking::unavailable,
                                  st::PhysicalUnit::dimensionless,
                                  5};
    const std::array signals{signal};
    return {13, signals};
}

struct FrameStorage
{
    std::array<st::SignalBlockHeader, 1> blocks{};
    alignas(double) std::array<std::byte, 8 * sizeof(double)> payload{};
};

[[nodiscard]] st::FrameView decoded_frame(FrameStorage& storage, std::uint64_t sequence,
                                          st::SampleIndex sample_idx,
                                          st::HostTimeNs observation_time_ns)
{
    constexpr std::array<double, 8> values{0.01, 0.0, 0.01, 0.0, 0.01, 0.0, 0.01, 0.0};
    std::memcpy(storage.payload.data(), values.data(), values.size() * sizeof(double));
    storage.blocks[0] = {.sample_idx_start = sample_idx,
                         .observation_time_start_ns = observation_time_ns,
                         .payload_offset = 0,
                         .payload_byte_count = storage.payload.size(),
                         .signal_id = 9,
                         .n_samples = 4};
    return {.header = {.session_id = 3,
                       .sequence = sequence,
                       .schema_id = 13,
                       .source_clock_domain = 7,
                       .signal_block_count = 1},
            .blocks = storage.blocks,
            .payload = storage.payload};
}

[[nodiscard]] std::vector<wg::WebGridSelectionRecord>
metric_records(std::size_t count) // all are valid misselections of one active target
{
    std::vector<wg::WebGridSelectionRecord> values(count);
    const ex::TrialIdentity trial{.ordinal = 0, .target_id = 1};
    for (std::size_t i = 0; i < count; ++i)
    {
        const auto time = static_cast<ex::ExperimentTimeNs>(i + 1);
        values[i] = {.event = {.time_ns = time,
                               .sequence = i + 1,
                               .trial = trial,
                               .paradigm = 52,
                               .kind = ex::SelectionKind::discrete,
                               .correct = false,
                               .selected_id = 2,
                               .intended_id = 1},
                     .target_onset_ns = 0,
                     .elapsed_since_target_onset_ns = time};
    }
    return values;
}

void benchmark_assistance(const Options& options, std::vector<Result>& results)
{
    const auto space = velocity_space();
    const ex::assistance::VelocityVector decoded{11, 2, {0.25, -0.5}};
    const ex::assistance::VelocityVector guidance{11, 2, {1.0, 0.5}};
    ex::assistance::VelocityVector output{};
    auto result =
        measure(options,
                {.paradigm = "shared_control",
                 .scenario = "linear_velocity_assistance",
                 .configuration =
                     "space=11;dimension=2;decoded=0.25,-0.5;guidance=1,0.5;alpha=0.25;batch=64",
                 .n_inputs = 2,
                 .operations_per_sample = 64},
                [&](bool, std::size_t)
                {
                    bool ok = true;
                    for (std::size_t batch = 0; batch < 64; ++batch)
                    {
                        ok = ex::assistance::blend_velocity(space, decoded, guidance,
                                                            ex::assistance::LinearAssistance{0.25},
                                                            output) == ex::ContractStatus::ok &&
                             ok;
                    }
                    return ok;
                });
    result.passed = result.passed && result.allocations == 0;
    results.push_back(result);
}

void benchmark_center_out(const Options& options, std::vector<Result>& results)
{
    const auto config = center_config();
    std::array<co::CenterOutMachine, 64> machines{};
    std::array<co::CenterOutStepResult, 64> steps{};
    auto machine_result = measure_prepared(
        options,
        {.paradigm = "center_out",
         .scenario = "state_machine_update",
         .configuration =
             "radial_targets=8;radius=0.5;acceptance=0.05;pointer=0.25,0.25;seed=17;batch=64",
         .n_inputs = 1,
         .operations_per_sample = machines.size()},
        [&](bool, std::size_t)
        {
            bool ok = true;
            for (std::size_t batch = 0; batch < machines.size(); ++batch)
            {
                machines[batch].reset();
                ok = machines[batch].start(41, config, 0, steps[batch]) == ex::ContractStatus::ok &&
                     ok;
            }
            return ok;
        },
        [&](bool, std::size_t idx)
        {
            bool ok = true;
            for (std::size_t batch = 0; batch < machines.size(); ++batch)
            {
                ok = machines[batch].step(static_cast<ex::ExperimentTimeNs>(idx + 1), {0.25, 0.25},
                                          steps[batch]) == ex::ContractStatus::ok &&
                     ok;
            }
            return ok;
        });
    machine_result.passed = machine_result.passed && machine_result.allocations == 0;
    results.push_back(machine_result);

    const co::CenterOutGuidanceConfig guidance_config{co::GeometryUnit::normalized, 1.0, 4.0, 4.0,
                                                      0.05};
    co::CenterOutGuidanceState guidance_state{};
    co::CenterOutGuidanceSample guidance_sample{};
    auto guidance_result =
        measure(options,
                {.paradigm = "center_out",
                 .scenario = "target_directed_guidance",
                 .configuration = "unit=normalized;acceleration=1;deceleration=4;max_"
                                  "speed=4;precision=0.05;dt_ns=1000000;batch=64",
                 .n_inputs = 2,
                 .operations_per_sample = 64},
                [&](bool, std::size_t)
                {
                    bool ok = true;
                    for (std::size_t batch = 0; batch < 64; ++batch)
                    {
                        guidance_state = {};
                        ok = co::evaluate_guidance(guidance_config, guidance_state, {2, {0.5, 0.0}},
                                                   {0.0, 0.0}, 1'000'000,
                                                   guidance_sample) == ex::ContractStatus::ok &&
                             ok;
                    }
                    return ok;
                });
    // Steady-state native rows require zero tracked allocations, like every
    // other steady-state row in this benchmark. evaluate_guidance is bounded
    // and allocates nothing, so a nonzero count here is a real regression
    // rather than a measurement artifact, and must fail the row.
    guidance_result.passed = guidance_result.passed && guidance_result.allocations == 0;
    results.push_back(guidance_result);

    execution::CenterOutController controller{};
    execution::CenterOutControllerConfig controller_config{};
    controller_config.decoded_signal_id = 9;
    controller_config.paradigm = 41;
    controller_config.task = config;
    controller_config.guidance = guidance_config;
    controller_config.velocity_space = velocity_space();
    controller_config.linear_assistance = {0.25};
    controller_config.cursor_min = {-1.0, -1.0};
    controller_config.cursor_max = {1.0, 1.0};
    controller_config.trace_capacity = 8;
    controller_config.assistance_method = ex::assistance::AssistanceMethod::linear_blend;
    bool setup_ok =
        controller.prepare(decoded_schema(), controller_config) == st::StreamStatus::ok &&
        controller.start(1'000'000'000, 0) == st::StreamStatus::ok;
    execution::CenterOutControlTrace trace{};
    setup_ok = setup_ok && controller.try_pop_trace(trace) == st::StreamStatus::ok;
    FrameStorage storage{};
    std::uint64_t sequence = 0;
    st::SampleIndex sample_idx = 0;
    auto result = measure(
        options,
        {.paradigm = "center_out",
         .scenario = "decoded_frame_to_headless_cursor",
         .configuration = "schema=13;signal=9;float64;sample_major;rate=1000/"
                          "1;observations=4;dt_ns=1000000;linear_alpha=0.25;trace_capacity=8",
         .throughput_item = "frame",
         .n_inputs = 4,
         .trace_capacity = 8,
         .observations_per_frame = 4},
        [&](bool, std::size_t)
        {
            const auto time = 1'000'000'000ULL + sample_idx * 1'000'000ULL;
            const auto status =
                controller.consume(decoded_frame(storage, sequence++, sample_idx, time));
            sample_idx += 4;
            bool ok = status == st::StreamStatus::ok;
            for (std::size_t observation = 0; observation < 4; ++observation)
            {
                ok = controller.try_pop_trace(trace) == st::StreamStatus::ok && ok;
            }
            return ok;
        });
    result.trace_drops = controller.dropped_trace_count();
    result.passed = setup_ok && result.passed && result.allocations == 0 && result.trace_drops == 0;
    results.push_back(result);
}

void benchmark_webgrid(const Options& options, std::vector<Result>& results)
{
    for (const auto side : {std::uint16_t{8}, std::uint16_t{16}})
    {
        const auto config = webgrid_config(side);
        ex::TargetId cell{};
        auto result =
            measure(options,
                    {.paradigm = "webgrid",
                     .scenario = side == 8 ? "geometry_hit_test_8x8" : "geometry_hit_test_16x16",
                     .configuration =
                         side == 8 ? "grid=8x8;bounds=0,8,0,8;candidates=64;seed=23;metric=1"
                                   : "grid=16x16;bounds=0,16,0,16;candidates=256;seed=23;metric=1",
                     .n_inputs = 1,
                     .grid_rows = side,
                     .grid_columns = side,
                     .metric_version = wg::kMetricVersion1},
                    [&](bool, std::size_t idx)
                    {
                        const auto pos = std::fmod(static_cast<double>(idx) * 0.61803398875,
                                                   static_cast<double>(side));
                        return wg::locate_cell(config, {pos, pos}, cell) == ex::ContractStatus::ok;
                    });
        result.passed = result.passed && result.allocations == 0;
        results.push_back(result);
    }

    const auto config = webgrid_config(16);
    wg::WebGridMachine machine{};
    wg::WebGridStepResult machine_step{};
    ex::SelectionEvent machine_selection{};
    auto machine_result = measure_prepared(
        options,
        {.paradigm = "webgrid",
         .scenario = "state_machine_selection",
         .configuration = "grid=16x16;bounds=0,16,0,16;seed=23;incorrect=keep_target;pointer=1.5,0."
                          "5;selected=2;target=1",
         .throughput_item = "selection",
         .n_inputs = 1,
         .grid_rows = 16,
         .grid_columns = 16,
         .metric_version = wg::kMetricVersion1},
        [&](bool, std::size_t)
        {
            machine.reset();
            if (machine.start(52, config, 0, machine_step) != ex::ContractStatus::ok)
            {
                return false;
            }
            machine_selection = {.time_ns = 1,
                                 .sequence = 1,
                                 .trial = machine_step.snapshot.trial,
                                 .paradigm = 52,
                                 .kind = ex::SelectionKind::discrete,
                                 .correct = false,
                                 .selected_id = 2,
                                 .intended_id = 1};
            return true;
        },
        [&](bool, std::size_t)
        {
            return machine.step(1, {1.5, 0.5}, machine_selection, machine_step) ==
                   ex::ContractStatus::ok;
        });
    machine_result.passed = machine_result.passed && machine_result.allocations == 0;
    results.push_back(machine_result);

    ex::TargetId target{};
    auto schedule_result = measure(
        options,
        {.paradigm = "webgrid",
         .scenario = "seeded_target_schedule",
         .configuration =
             "grid=16x16;candidates=256;seed=23;immediate_repetition=forbid;previous_target=1",
         .n_inputs = 256,
         .grid_rows = 16,
         .grid_columns = 16,
         .metric_version = wg::kMetricVersion1},
        [&](bool, std::size_t idx)
        {
            return wg::select_target(config, static_cast<ex::TrialOrdinal>(idx + 1), 1, target) ==
                   ex::ContractStatus::ok;
        });
    schedule_result.passed = schedule_result.passed && schedule_result.allocations == 0;
    results.push_back(schedule_result);

    const auto records = metric_records(options.metric_records);
    wg::WebGridMetrics metrics{};
    auto metric_result = measure(
        options,
        {.paradigm = "webgrid",
         .scenario = "metric_v1_recompute",
         .configuration = "metric=1;records=all_incorrect;target=1;selected=2;elapsed_end=count+1",
         .throughput_item = "record_set",
         .n_inputs = records.size(),
         .grid_rows = 16,
         .grid_columns = 16,
         .metric_version = wg::kMetricVersion1},
        [&](bool, std::size_t)
        {
            return wg::summarize(records, 0, records.size() + 1, wg::kMetricVersion1, metrics) ==
                   ex::ContractStatus::ok;
        });
    metric_result.passed = metric_result.passed && metric_result.allocations == 0;
    results.push_back(metric_result);

    execution::WebGridHeadlessController controller{};
    bool setup_ok = controller.prepare(config, 2) == st::StreamStatus::ok &&
                    controller.start(52, 0) == st::StreamStatus::ok;
    execution::WebGridHeadlessTrace trace{};
    setup_ok = setup_ok && controller.try_pop_trace(trace) == st::StreamStatus::ok;
    ex::SequenceOrdinal sequence = 1;
    auto controller_result = measure(
        options,
        {.paradigm = "webgrid",
         .scenario = "headless_pointer_selection",
         .configuration = "grid=16x16;seed=23;incorrect=keep_target;pointer=1.5,0.5;selected=2;"
                          "target=1;trace_capacity=2",
         .throughput_item = "selection",
         .n_inputs = 1,
         .grid_rows = 16,
         .grid_columns = 16,
         .trace_capacity = 2,
         .metric_version = wg::kMetricVersion1},
        [&](bool, std::size_t idx)
        {
            const auto time = static_cast<ex::ExperimentTimeNs>(sequence);
            const ex::SelectionEvent selection{.time_ns = time,
                                               .sequence = sequence,
                                               .trial = controller.snapshot().trial,
                                               .paradigm = 52,
                                               .kind = ex::SelectionKind::discrete,
                                               .correct = false,
                                               .selected_id = 2,
                                               .intended_id = 1};
            ++sequence;
            return controller.process(time, {1.5, 0.5}, selection) == st::StreamStatus::ok &&
                   controller.try_pop_trace(trace) == st::StreamStatus::ok;
        });
    controller_result.trace_drops = controller.dropped_trace_count();
    controller_result.passed = setup_ok && controller_result.passed &&
                               controller_result.allocations == 0 &&
                               controller_result.trace_drops == 0;
    results.push_back(controller_result);
}

void benchmark_speech(const Options& options, std::vector<Result>& results)
{
    const auto config = speech_config(options.speech_trials);
    std::vector<sp::SpeechTrialSchedule> schedules(options.speech_trials);
    bool schedules_ok = true;
    for (std::size_t i = 0; i < schedules.size(); ++i)
    {
        schedules_ok =
            sp::prepare_trial(config, i, schedules[i]) == ex::ContractStatus::ok && schedules_ok;
    }
    std::array<sp::SpeechMachine, 64> machines{};
    std::array<sp::SpeechStepResult, 64> steps{};
    auto transition_result = measure_prepared(
        options,
        {.paradigm = "speech",
         .scenario = "phase_transition",
         .configuration =
             "seed=29;sampler=current;black_bound_ns=1000000;cross_bound_ns=1000000;content_bound_"
             "ns=2000000;cross=true;trial=0;transition=black_end;batch=64",
         .n_inputs = 1,
         .operations_per_sample = machines.size()},
        [&](bool, std::size_t)
        {
            bool ok = true;
            for (std::size_t batch = 0; batch < machines.size(); ++batch)
            {
                machines[batch].reset();
                ok = machines[batch].start(63, config, schedules.front(), 0, steps[batch]) ==
                         ex::ContractStatus::ok &&
                     ok;
            }
            return ok;
        },
        [&](bool, std::size_t)
        {
            bool ok = true;
            for (std::size_t batch = 0; batch < machines.size(); ++batch)
            {
                ok = machines[batch].step(schedules.front().black_duration_ns, steps[batch]) ==
                         ex::ContractStatus::ok &&
                     ok;
            }
            return ok;
        });
    transition_result.passed =
        schedules_ok && transition_result.passed && transition_result.allocations == 0;
    results.push_back(transition_result);

    const auto catalog = speech_catalog(options.speech_trials);
    auto preparation_result = measure(
        options,
        {.paradigm = "speech",
         .scenario = "large_schedule_prepare",
         .configuration =
             "seed=29;sampler=current;stimulus_order=sequential;cross=true;trace_capacity=2",
         .throughput_item = "schedule",
         .allocation_scope = "prepare",
         .n_inputs = schedules.size()},
        [&](bool, std::size_t)
        {
            execution::SpeechHeadlessScheduler scheduler{};
            return scheduler.prepare(config, catalog, schedules, 2) == st::StreamStatus::ok;
        });
    preparation_result.throughput_per_second =
        preparation_result.summary.median > 0.0
            ? static_cast<double>(schedules.size()) * 1.0e9 / preparation_result.summary.median
            : 0.0;
    preparation_result.passed = schedules_ok && preparation_result.passed;
    results.push_back(preparation_result);
}

void benchmark_replay(const Options& options, std::vector<Result>& results)
{
    {
        co::CenterOutReplayConfig config{};
        config.paradigm = 41;
        config.task = center_config();
        config.guidance = {co::GeometryUnit::normalized, 1.0, 4.0, 4.0, 0.05};
        config.velocity_space = velocity_space();
        config.linear_assistance = {0.25};
        config.assistance_method = ex::assistance::AssistanceMethod::linear_blend;
        config.cursor_min = {-1.0, -1.0};
        config.cursor_max = {1.0, 1.0};
        co::CenterOutReplay replay{};
        bool setup_ok = replay.prepare(config) == ex::ContractStatus::ok;
        std::vector<co::CenterOutReplayInput> inputs(options.replay_inputs);
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            inputs[i] = {.kind = co::CenterOutReplayInputKind::observation,
                         .time_ns = i + 1,
                         .decoded = {0.001, 0.0},
                         .frame_sequence = i,
                         .sample_idx = i};
        }
        co::CenterOutReplayRecording recording{
            .provenance = replay.provenance(), .origin_ns = 0, .inputs = inputs};
        ex::ReplayReport report{};
        auto result =
            measure(options,
                    {.paradigm = "center_out",
                     .scenario = "deterministic_semantic_replay",
                     .configuration = "radial_targets=8;seed=17;guidance=1,4,4,0.05;linear_alpha=0."
                                      "25;inputs=observations;expected_streams=omitted",
                     .throughput_item = "session",
                     .n_inputs = inputs.size(),
                     .replay_verdict = "incomplete",
                     .replay_completeness = "stream_absent"},
                    [&](bool, std::size_t)
                    {
                        report = replay.run(recording);
                        return report.verdict == ex::ReplayVerdict::incomplete &&
                               report.completeness == ex::ReplayCompleteness::stream_absent &&
                               report.inputs_replayed == inputs.size();
                    });
        result.throughput_per_second =
            result.summary.median > 0.0 ? inputs.size() * 1.0e9 / result.summary.median : 0.0;
        result.throughput_item = "semantic_input";
        result.passed = setup_ok && result.passed && result.allocations == 0;
        results.push_back(result);
    }

    {
        wg::WebGridReplayConfig config{.paradigm = 52, .task = webgrid_config(16)};
        wg::WebGridReplay replay{};
        bool setup_ok = replay.prepare(config) == ex::ContractStatus::ok;
        std::vector<wg::WebGridReplayInput> inputs(options.replay_inputs);
        for (std::size_t i = 0; i < inputs.size(); ++i)
        {
            inputs[i] = {.kind = wg::WebGridReplayInputKind::pointer,
                         .time_ns = i + 1,
                         .pointer = {0.5, 0.5}};
        }
        wg::WebGridReplayRecording recording{
            .provenance = replay.provenance(), .origin_ns = 0, .inputs = inputs};
        ex::ReplayReport report{};
        auto result = measure(
            options,
            {.paradigm = "webgrid",
             .scenario = "deterministic_semantic_replay",
             .configuration =
                 "grid=16x16;seed=23;inputs=pointer;pointer=0.5,0.5;expected_streams=omitted",
             .throughput_item = "session",
             .n_inputs = inputs.size(),
             .grid_rows = 16,
             .grid_columns = 16,
             .metric_version = wg::kMetricVersion1,
             .replay_verdict = "incomplete",
             .replay_completeness = "stream_absent"},
            [&](bool, std::size_t)
            {
                report = replay.run(recording);
                return report.verdict == ex::ReplayVerdict::incomplete &&
                       report.completeness == ex::ReplayCompleteness::stream_absent &&
                       report.inputs_replayed == inputs.size();
            });
        result.throughput_per_second =
            result.summary.median > 0.0 ? inputs.size() * 1.0e9 / result.summary.median : 0.0;
        result.throughput_item = "semantic_input";
        result.passed = setup_ok && result.passed && result.allocations == 0;
        results.push_back(result);
    }

    {
        const auto trials = options.speech_trials;
        auto config_value = speech_config(trials);
        auto catalog = speech_catalog(trials);
        sp::SpeechReplayConfig config{.paradigm = 63, .task = config_value, .catalog = catalog};
        sp::SpeechReplay replay{};
        bool setup_ok = replay.prepare(config) == ex::ContractStatus::ok;
        std::vector<sp::SpeechTrialSchedule> schedules(trials);
        std::vector<sp::SpeechReplayInput> inputs{};
        inputs.reserve(trials * 3);
        ex::ExperimentTimeNs time = 0;
        for (std::size_t i = 0; i < trials; ++i)
        {
            setup_ok = sp::prepare_trial(config_value, i, schedules[i]) == ex::ContractStatus::ok &&
                       setup_ok;
            time += schedules[i].black_duration_ns;
            inputs.push_back({.kind = sp::SpeechReplayInputKind::advance, .time_ns = time});
            time += schedules[i].cross_duration_ns;
            inputs.push_back({.kind = sp::SpeechReplayInputKind::advance, .time_ns = time});
            time += schedules[i].content_duration_ns;
            inputs.push_back({.kind = sp::SpeechReplayInputKind::advance, .time_ns = time});
        }
        sp::SpeechReplayRecording recording{.provenance = replay.provenance(schedules),
                                            .origin_ns = 0,
                                            .inputs = inputs,
                                            .run_end_recorded = true};
        ex::ReplayReport report{};
        auto result =
            measure(options,
                    {.paradigm = "speech",
                     .scenario = "deterministic_semantic_replay",
                     .configuration = "seed=29;sampler=current;cross=true;inputs=phase_boundaries;"
                                      "expected_streams=omitted",
                     .throughput_item = "session",
                     .n_inputs = inputs.size(),
                     .replay_verdict = "incomplete",
                     .replay_completeness = "stream_absent"},
                    [&](bool, std::size_t)
                    {
                        report = replay.run(recording);
                        return report.verdict == ex::ReplayVerdict::incomplete &&
                               report.completeness == ex::ReplayCompleteness::stream_absent &&
                               report.inputs_replayed == inputs.size();
                    });
        result.throughput_per_second =
            result.summary.median > 0.0 ? inputs.size() * 1.0e9 / result.summary.median : 0.0;
        result.throughput_item = "semantic_input";
        result.passed = setup_ok && result.passed && result.allocations == 0;
        results.push_back(result);
    }
}

[[nodiscard]] bool parse_size(std::string_view value, std::size_t& output)
{
    std::string text{value};
    char* end = nullptr;
    const auto parsed = std::strtoull(text.c_str(), &end, 10);
    if (end == text.c_str() || *end != '\0' || parsed == 0)
    {
        return false;
    }
    output = static_cast<std::size_t>(parsed);
    return true;
}

[[nodiscard]] bool parse_options(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string_view flag{argv[i]};
        if (i + 1 >= argc)
        {
            return false;
        }
        std::size_t value{};
        if (!parse_size(argv[++i], value))
        {
            return false;
        }
        if (flag == "--warmups")
            options.warmups = value;
        else if (flag == "--repetitions")
            options.repetitions = value;
        else if (flag == "--metric-records")
            options.metric_records = value;
        else if (flag == "--replay-inputs")
            options.replay_inputs = value;
        else if (flag == "--speech-trials" && value <= sp::kMaxSpeechExplicitTrials)
            options.speech_trials = value;
        else
            return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    for (int i = 0; i < argc; ++i)
    {
        if (i != 0)
            g_command.push_back(' ');
        g_command.append(argv[i]);
    }

    Options options{};
    if (!parse_options(argc, argv, options))
    {
        std::cerr << "usage: neurale_experiments_benchmark [--warmups N] [--repetitions N] "
                     "[--metric-records N] [--replay-inputs N] [--speech-trials N]\n";
        return 2;
    }

    std::vector<Result> results{};
    results.reserve(15);
    benchmark_assistance(options, results);
    benchmark_center_out(options, results);
    benchmark_webgrid(options, results);
    benchmark_speech(options, results);
    benchmark_replay(options, results);

    bool passed = results.size() == 15;
    for (const auto& result : results)
    {
        emit(options, result);
        passed = result.passed && passed;
    }
    return passed ? 0 : 1;
}
