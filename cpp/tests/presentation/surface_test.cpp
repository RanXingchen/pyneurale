/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "check_returns.h"
#include "coordinate_mapper.h"
#include "input_runtime.h"
#include "skip_policy.h"
#include "surface.h"
#include "text_atlas.h"
#include "time_mapper.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <span>
#include <string_view>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace
{

namespace ep = neurale::experiment_presentation;

bool close(double left, double right) noexcept
{
    return std::abs(left - right) <= 1e-12;
}

int test_coordinates()
{
    ep::CoordinateMapper mapper;
    CHECK(mapper.configure({-1.0, -1.0, 2.0, 2.0}, {400, 200}, {800, 400},
                           ep::AspectPolicy::fit_letterbox) == ep::SurfaceStatus::ok);
    const auto viewport = mapper.viewport();
    CHECK(viewport.left == 200);
    CHECK(viewport.top == 0);
    CHECK(viewport.width == 400);
    CHECK(viewport.height == 400);

    const auto upper_left = mapper.logical_to_framebuffer({-1.0, 1.0});
    CHECK(close(upper_left.x, 200.0));
    CHECK(close(upper_left.y, 0.0));
    const auto lower_right = mapper.logical_to_framebuffer({1.0, -1.0});
    CHECK(close(lower_right.x, 600.0));
    CHECK(close(lower_right.y, 400.0));

    ep::Point2d logical{};
    CHECK(mapper.framebuffer_to_logical({400.0, 200.0}, logical));
    CHECK(close(logical.x, 0.0));
    CHECK(close(logical.y, 0.0));
    CHECK(!mapper.framebuffer_to_logical({199.0, 200.0}, logical));
    CHECK(!mapper.framebuffer_to_logical({600.0, 200.0}, logical));

    // Window coordinates are converted through the independent framebuffer
    // scale. Here the framebuffer is exactly 2x the logical window size.
    CHECK(mapper.window_pointer_to_logical({200.0, 100.0}, logical));
    CHECK(close(logical.x, 0.0));
    CHECK(close(logical.y, 0.0));
    CHECK(!mapper.window_pointer_to_logical({99.0, 100.0}, logical));

    CHECK(mapper.configure({0.0, 0.0, 4.0, 2.0}, {300, 300}, {600, 600},
                           ep::AspectPolicy::stretch) == ep::SurfaceStatus::ok);
    CHECK(mapper.viewport().width == 600);
    CHECK(mapper.viewport().height == 600);
    const auto center = mapper.logical_to_framebuffer({2.0, 1.0});
    CHECK(close(center.x, 300.0));
    CHECK(close(center.y, 300.0));

    CHECK(mapper.configure({0.0, 0.0, 0.0, 1.0}, {1, 1}, {1, 1}, ep::AspectPolicy::stretch) ==
          ep::SurfaceStatus::invalid_configuration);
    return 0;
}

int test_time_mapping()
{
    ep::RendererTimeMapper mapper;
    neurale::experiments::ExperimentTimeNs experiment_time{};
    CHECK(mapper.map(10, experiment_time) == ep::SurfaceStatus::invalid_state);
    mapper.configure(1'000, 5'000);
    CHECK(mapper.map(1'000, experiment_time) == ep::SurfaceStatus::ok);
    CHECK(experiment_time == 5'000);
    CHECK(mapper.map(1'025, experiment_time) == ep::SurfaceStatus::ok);
    CHECK(experiment_time == 5'025);
    CHECK(mapper.map(1'025, experiment_time) == ep::SurfaceStatus::ok);
    CHECK(mapper.map(1'024, experiment_time) == ep::SurfaceStatus::time_regressed);

    mapper.configure(100, (std::numeric_limits<neurale::experiments::ExperimentTimeNs>::max)() - 2);
    CHECK(mapper.map(103, experiment_time) == ep::SurfaceStatus::time_overflow);
    mapper.reset();
    CHECK(mapper.map(100, experiment_time) == ep::SurfaceStatus::invalid_state);
    return 0;
}

int test_input_ordering()
{
    ep::RendererTimeMapper time_mapper;
    time_mapper.configure(100, 1'000);
    ep::FixedInputQueue queue;
    CHECK(queue.prepare(2) == ep::SurfaceStatus::ok);
    CHECK(queue.push({.renderer_time_ns = 100,
                      .kind = ep::InputEventKind::pointer_moved,
                      .pointer = {0.25, -0.5},
                      .pointer_inside = true},
                     time_mapper) == ep::SurfaceStatus::ok);
    CHECK(queue.push({.renderer_time_ns = 100,
                      .kind = ep::InputEventKind::mouse_button,
                      .action = ep::InputAction::press,
                      .code = 1,
                      .pointer = {0.25, -0.5},
                      .pointer_inside = true},
                     time_mapper) == ep::SurfaceStatus::ok);
    CHECK(queue.push({.renderer_time_ns = 101, .kind = ep::InputEventKind::key}, time_mapper) ==
          ep::SurfaceStatus::input_overflow);

    ep::InputEvent event{};
    CHECK(queue.pop(event));
    CHECK(event.ordinal == 0);
    CHECK(event.experiment_time_ns == 1'000);
    CHECK(event.kind == ep::InputEventKind::pointer_moved);
    CHECK(queue.pop(event));
    CHECK(event.ordinal == 1);
    CHECK(event.experiment_time_ns == 1'000);
    CHECK(event.kind == ep::InputEventKind::mouse_button);
    CHECK(!queue.pop(event));

    queue.reset();
    time_mapper.configure(200, 2'000);
    CHECK(queue.push({.renderer_time_ns = 200,
                      .kind = ep::InputEventKind::key,
                      .action = ep::InputAction::press,
                      .code = 32,
                      .modifiers = 1},
                     time_mapper) == ep::SurfaceStatus::ok);
    CHECK(queue.push({.renderer_time_ns = 201,
                      .kind = ep::InputEventKind::framebuffer_resized,
                      .size = {640, 480}},
                     time_mapper) == ep::SurfaceStatus::ok);
    CHECK(queue.pop(event));
    CHECK(event.ordinal == 0);
    CHECK(event.experiment_time_ns == 2'000);
    CHECK(event.kind == ep::InputEventKind::key);
    CHECK(event.code == 32);
    CHECK(queue.pop(event));
    CHECK(event.ordinal == 1);
    CHECK(event.experiment_time_ns == 2'001);
    CHECK(event.kind == ep::InputEventKind::framebuffer_resized);
    CHECK(event.size.width == 640);

    queue.reset();
    time_mapper.configure(300, 3'000);
    CHECK(queue.push({.renderer_time_ns = 300, .kind = ep::InputEventKind::window_close},
                     time_mapper) == ep::SurfaceStatus::ok);
    CHECK(queue.pop(event));
    CHECK(event.ordinal == 0);
    CHECK(event.kind == ep::InputEventKind::window_close);
    queue.close();
    CHECK(queue.status() == ep::SurfaceStatus::invalid_state);
    return 0;
}

int test_explicit_failures()
{
    ep::PreparedTextAtlas atlas;
    const std::array catalog{ep::TextCatalogEntry{1, "text"}};
    CHECK(atlas.prepare({.font_path = "definitely-not-a-font.ttf",
                         .pixel_height = 24,
                         .atlas_width = 64,
                         .atlas_height = 64,
                         .max_texts = 1,
                         .max_glyphs = 16},
                        catalog) == ep::SurfaceStatus::font_load_failed);
    CHECK(!atlas.prepared());

    ep::PresentationSurface surface;
    // A renderer origin that lies in the future of the renderer clock cannot
    // have been sampled from it, which is the cheap way to catch an origin
    // taken from a foreign epoch -- a wall clock, a device clock, a raw
    // performance counter. RendererTimeMapper is a pure offset, so such an
    // origin would produce well-formed instants that are silently wrong by the
    // difference between the two epochs.
    CHECK(surface.open({}, (std::numeric_limits<ep::RendererTimeNs>::max)(), 0) ==
          ep::SurfaceStatus::invalid_configuration);
    CHECK(surface.lifecycle() == ep::SurfaceLifecycle::closed);

    surface.inject_failure(ep::SurfaceFailurePoint::glfw_initialization);
    CHECK(surface.open({}, 1, 2) == ep::SurfaceStatus::glfw_initialization_failed);
    CHECK(surface.lifecycle() == ep::SurfaceLifecycle::closed);
    surface.close();
    surface.close();
    CHECK(surface.lifecycle() == ep::SurfaceLifecycle::closed);
    CHECK(surface.status() == ep::SurfaceStatus::ok);
    CHECK(surface.request_window_size({100, 100}) == ep::SurfaceStatus::invalid_state);
    const auto resources = surface.resource_stats();
    CHECK(!resources.window_open);
    CHECK(!resources.opengl_ready);
    CHECK(resources.vertex_capacity == 0);
    return 0;
}

int test_text_prepare(std::string_view font_path)
{
    ep::PreparedTextAtlas atlas;
    const std::array catalog{
        ep::TextCatalogEntry{1, "Speech"},
        ep::TextCatalogEntry{2, "\xCE\xB1\xCE\xB2"},
    };
    const ep::TextAtlasConfig config{
        .font_path = font_path,
        .pixel_height = 28,
        .atlas_width = 256,
        .atlas_height = 256,
        .max_texts = catalog.size(),
        .max_glyphs = 64,
    };
    CHECK(atlas.prepare(config, catalog) == ep::SurfaceStatus::ok);
    CHECK(atlas.text_count() == catalog.size());
    CHECK(atlas.glyph_count() > 0);
    CHECK(atlas.find(1) != nullptr);
    CHECK(atlas.find(2) != nullptr);
    CHECK(atlas.find(2)->utf8 == catalog[1].utf8);

    const std::array invalid_utf8{ep::TextCatalogEntry{3, "\xC0\xAF"}};
    CHECK(atlas.prepare(config, invalid_utf8) == ep::SurfaceStatus::invalid_utf8);
    CHECK(!atlas.prepared());
    const std::array missing_glyph{ep::TextCatalogEntry{3, "\xF4\x8F\xBF\xBF"}};
    CHECK(atlas.prepare(config, missing_glyph) == ep::SurfaceStatus::glyph_missing);
    CHECK(!atlas.prepared());
    const std::array duplicate_ids{
        ep::TextCatalogEntry{4, "a"},
        ep::TextCatalogEntry{4, "b"},
    };
    CHECK(atlas.prepare(config, duplicate_ids) == ep::SurfaceStatus::invalid_configuration);
    return 0;
}

int test_text_cjk(std::string_view cjk_font_path)
{
    // CJK smoke against a CJK font. DejaVu (Latin/Greek only) cannot prove this
    // path, which 中文 -- a first-class Speech use case -- depends on. The
    // three prompts are UTF-8 byte escapes so the source stays ASCII, matching
    // the Greek entry above; the runtime property under test is that HarfBuzz
    // shapes each prompt to at least one glyph and preparation reports no
    // missing glyph rather than falling back.
    ep::PreparedTextAtlas atlas;
    const std::array catalog{
        ep::TextCatalogEntry{1, "\xE4\xB8\xAD\xE6\x96\x87"}, // 中文
        ep::TextCatalogEntry{2, "\xE6\x83\xB3\xE8\xB1\xA1\xE5\x90\x91\xE5\xB7\xA6\xE7\xA7\xBB\xE5"
                                "\x8A\xA8"},                 // 想象向左移动
        ep::TextCatalogEntry{3, "\xE6\x8F\xA1\xE6\x8B\xB3"}, // 握拳
    };
    const ep::TextAtlasConfig config{
        .font_path = cjk_font_path,
        .pixel_height = 28,
        .atlas_width = 512,
        .atlas_height = 512,
        .max_texts = catalog.size(),
        .max_glyphs = 256,
    };
    CHECK(atlas.prepare(config, catalog) == ep::SurfaceStatus::ok);
    CHECK(atlas.text_count() == catalog.size());
    CHECK(atlas.glyph_count() > 0);
    for (const auto& entry : catalog)
    {
        const auto* prepared = atlas.find(entry.id);
        CHECK(prepared != nullptr);
        CHECK(prepared->utf8 == entry.utf8);
        // HarfBuzz shaped at least one glyph for each CJK prompt.
        CHECK(!prepared->glyphs.empty());
    }

    // An atlas too narrow for the glyph it must hold is a capacity error, not a
    // blit that runs off the end of its row. Starting a new row cannot make a
    // glyph wider than the atlas fit, so the wrap has to be followed by a width
    // check as well as a height check; without it the row copy overwrites the
    // next glyph's pixels and, near the bottom of the atlas, writes past the
    // pixel buffer entirely. A Han glyph is about as wide as pixel_height, so
    // this is exactly the configuration a CJK stimulus makes reachable.
    ep::PreparedTextAtlas too_narrow;
    auto narrow_config = config;
    narrow_config.atlas_width = 8;
    narrow_config.atlas_height = 512;
    const std::array single{catalog[0]};
    CHECK(too_narrow.prepare(narrow_config, single) == ep::SurfaceStatus::atlas_capacity_exceeded);
    CHECK(!too_narrow.prepared());
    // The same glyph in an atlas wide enough for it still prepares, so the
    // check above rejects the unplaceable case and not the narrow-but-valid one.
    auto exact_config = config;
    exact_config.atlas_width = 128;
    exact_config.atlas_height = 128;
    ep::PreparedTextAtlas narrow_but_valid;
    CHECK(narrow_but_valid.prepare(exact_config, single) == ep::SurfaceStatus::ok);
    return 0;
}

int test_window(std::string_view font_path)
{
    ep::PresentationSurface surface;
    const auto renderer_origin = ep::renderer_monotonic_now_ns();
    auto status = surface.open({.title = "PyNeurale presentation test",
                                .window_size = {320, 240},
                                .logical_space = {-1.0, -1.0, 2.0, 2.0},
                                .aspect_policy = ep::AspectPolicy::fit_letterbox,
                                .monitor_idx = 0,
                                .swap_interval = 0,
                                .input_capacity = 32,
                                .fullscreen = false,
                                .resizable = true,
                                .visible = false},
                               renderer_origin, 10'000);
    if (status == ep::SurfaceStatus::glfw_initialization_failed ||
        status == ep::SurfaceStatus::window_creation_failed ||
        status == ep::SurfaceStatus::monitor_unavailable ||
        status == ep::SurfaceStatus::opengl_function_missing)
    {
        return neurale::presentation_test::skip_or_fail(ep::surface_status_name(status));
    }
    CHECK(status == ep::SurfaceStatus::ok);
    CHECK(surface.monitor_count() > 0);
    CHECK(surface.open({}, renderer_origin, 10'000) == ep::SurfaceStatus::invalid_state);

    const std::array catalog{
        ep::TextCatalogEntry{1, "Speech"},
        ep::TextCatalogEntry{2, "\xCE\xB1\xCE\xB2"},
    };
    CHECK(surface.prepare({.max_vertices = 1'024,
                           .max_draw_batches = 128,
                           .circle_segments = 32,
                           .text = {.font_path = font_path,
                                    .pixel_height = 28,
                                    .atlas_width = 256,
                                    .atlas_height = 256,
                                    .max_texts = catalog.size(),
                                    .max_glyphs = 64}},
                          catalog) == ep::SurfaceStatus::ok);
    const auto prepared_resources = surface.resource_stats();
    CHECK(prepared_resources.window_open);
    CHECK(prepared_resources.opengl_ready);
    CHECK(prepared_resources.text_ready);
    CHECK(prepared_resources.prepared_text_count == catalog.size());
    CHECK(prepared_resources.text_preparation_ns > 0);
    CHECK(prepared_resources.shaping_preparation_ns > 0);
    CHECK(prepared_resources.glyph_cache_preparation_ns > 0);
    CHECK(prepared_resources.unexpected_runtime_glyph_cache_misses == 0);
    const auto environment = surface.environment();
    CHECK(environment.gpu_vendor[0] != '\0');
    CHECK(environment.gpu_renderer[0] != '\0');
    CHECK(environment.opengl_version[0] != '\0');
    CHECK(environment.window_width == 320);
    CHECK(environment.window_height == 240);
    CHECK(environment.framebuffer_width > 0);
    CHECK(environment.framebuffer_height > 0);
    CHECK(environment.swap_interval == 0);
    CHECK(!environment.fullscreen);

    const std::array polyline{
        ep::Point2d{-0.8, -0.8},
        ep::Point2d{0.0, 0.8},
        ep::Point2d{0.8, -0.8},
    };
    CHECK(surface.begin_frame({0.05F, 0.1F, 0.15F, 1.0F}) == ep::SurfaceStatus::ok);
    CHECK(surface.line({-0.9, 0.0}, {0.9, 0.0}, {1.0F, 0.0F, 0.0F, 1.0F}) == ep::SurfaceStatus::ok);
    CHECK(surface.polyline(polyline, {0.0F, 1.0F, 0.0F, 1.0F}) == ep::SurfaceStatus::ok);
    CHECK(surface.rectangle({-0.8, -0.6, 0.4, 0.3}, {0.0F, 0.0F, 1.0F, 1.0F}, true) ==
          ep::SurfaceStatus::ok);
    CHECK(surface.circle({0.5, 0.4}, 0.2, {1.0F, 1.0F, 0.0F, 1.0F}, false) ==
          ep::SurfaceStatus::ok);
    CHECK(surface.fixation_cross({0.0, 0.0}, 0.1, {1.0F, 1.0F, 1.0F, 1.0F}) ==
          ep::SurfaceStatus::ok);
    CHECK(surface.shaped_text(1, {-0.8, 0.7}, {1.0F, 1.0F, 1.0F, 1.0F}) == ep::SurfaceStatus::ok);
    const auto first = surface.present(8'000, 9'000);
    CHECK(first.status == ep::SurfaceStatus::ok);
    CHECK(first.requested_ns == 8'000);
    CHECK(first.intended_ns == 9'000);
    CHECK(first.submitted_renderer_ns >= renderer_origin);
    CHECK(first.presented_renderer_ns >= first.submitted_renderer_ns);
    CHECK(first.presented_ns >= first.submitted_ns);

    // Repeated frames use the prepared capacities and shaped resources.
    for (int repetition = 0; repetition < 8; ++repetition)
    {
        CHECK(surface.begin_frame({0.0F, 0.0F, 0.0F, 1.0F}) == ep::SurfaceStatus::ok);
        CHECK(surface.shaped_text(2, {-0.2, 0.0}, {1.0F, 1.0F, 1.0F, 1.0F}) ==
              ep::SurfaceStatus::ok);
        CHECK(surface.present(10'000 + repetition, 11'000 + repetition).status ==
              ep::SurfaceStatus::ok);
    }
    const auto repeated_resources = surface.resource_stats();
    CHECK(repeated_resources.vertex_capacity == prepared_resources.vertex_capacity);
    CHECK(repeated_resources.batch_capacity == prepared_resources.batch_capacity);
    CHECK(repeated_resources.prepared_text_count == prepared_resources.prepared_text_count);
    CHECK(repeated_resources.prepared_glyph_count == prepared_resources.prepared_glyph_count);
    CHECK(repeated_resources.unexpected_runtime_glyph_cache_misses == 0);

    CHECK(surface.request_window_size({400, 300}) == ep::SurfaceStatus::ok);
    CHECK(surface.pump_events() == ep::SurfaceStatus::ok);
    CHECK(surface.coordinates().window_size().width > 0);
    CHECK(surface.coordinates().framebuffer_size().width > 0);
    ep::InputEvent event{};
    std::uint64_t previous_ordinal{};
    bool first_event = true;
    bool event_available{};
    CHECK(surface.poll_input(event, event_available) == ep::SurfaceStatus::ok);
    while (event_available)
    {
        if (!first_event)
        {
            CHECK(event.ordinal > previous_ordinal);
        }
        first_event = false;
        previous_ordinal = event.ordinal;
        CHECK(event.experiment_time_ns >= 10'000);
        CHECK(surface.poll_input(event, event_available) == ep::SurfaceStatus::ok);
    }

    surface.inject_failure(ep::SurfaceFailurePoint::presentation);
    CHECK(surface.begin_frame({0.0F, 0.0F, 0.0F, 1.0F}) == ep::SurfaceStatus::ok);
    CHECK(surface.present(1, 2).status == ep::SurfaceStatus::presentation_failed);
    CHECK(surface.lifecycle() == ep::SurfaceLifecycle::faulted);
    surface.close();
    surface.close();
    const auto closed = surface.resource_stats();
    CHECK(surface.lifecycle() == ep::SurfaceLifecycle::closed);
    CHECK(!closed.window_open);
    CHECK(!closed.opengl_ready);
    CHECK(!closed.text_ready);
    CHECK(closed.vertex_capacity == 0);

    ep::PresentationSurface capacity_limited;
    status = capacity_limited.open({.title = "PyNeurale capacity test",
                                    .window_size = {160, 120},
                                    .monitor_idx = 0,
                                    .swap_interval = 0,
                                    .input_capacity = 8,
                                    .visible = false},
                                   ep::renderer_monotonic_now_ns(), 0);
    CHECK(status == ep::SurfaceStatus::ok);
    CHECK(capacity_limited.prepare({.max_vertices = 4, .max_draw_batches = 2, .circle_segments = 3},
                                   {}) == ep::SurfaceStatus::ok);
    CHECK(capacity_limited.begin_frame({0.0F, 0.0F, 0.0F, 1.0F}) == ep::SurfaceStatus::ok);
    CHECK(capacity_limited.line({0.0, 0.0}, {0.5, 0.5}, {1.0F, 1.0F, 1.0F, 1.0F}) ==
          ep::SurfaceStatus::ok);
    CHECK(capacity_limited.line({0.0, 0.0}, {-0.5, -0.5}, {1.0F, 1.0F, 1.0F, 1.0F}) ==
          ep::SurfaceStatus::ok);
    CHECK(capacity_limited.line({0.0, 0.0}, {0.0, 0.5}, {1.0F, 1.0F, 1.0F, 1.0F}) ==
          ep::SurfaceStatus::resource_capacity_exceeded);
    capacity_limited.close();

    ep::PresentationSurface missing_text;
    status = missing_text.open({.title = "PyNeurale missing prepared text test",
                                .window_size = {160, 120},
                                .monitor_idx = 0,
                                .swap_interval = 0,
                                .input_capacity = 8,
                                .visible = false},
                               ep::renderer_monotonic_now_ns(), 0);
    CHECK(status == ep::SurfaceStatus::ok);
    CHECK(missing_text.prepare({.max_vertices = 64,
                                .max_draw_batches = 8,
                                .circle_segments = 3,
                                .text = {.font_path = font_path,
                                         .pixel_height = 28,
                                         .atlas_width = 256,
                                         .atlas_height = 256,
                                         .max_texts = catalog.size(),
                                         .max_glyphs = 64}},
                               catalog) == ep::SurfaceStatus::ok);
    CHECK(missing_text.begin_frame({0.0F, 0.0F, 0.0F, 1.0F}) == ep::SurfaceStatus::ok);
    CHECK(missing_text.shaped_text(999, {0.0, 0.0}, {1.0F, 1.0F, 1.0F, 1.0F}) ==
          ep::SurfaceStatus::text_not_prepared);
    CHECK(missing_text.resource_stats().unexpected_runtime_glyph_cache_misses == 1);
    missing_text.close();

    ep::PresentationSurface shader_failure;
    status = shader_failure.open({.title = "PyNeurale shader failure test",
                                  .window_size = {160, 120},
                                  .monitor_idx = 0,
                                  .swap_interval = 0,
                                  .input_capacity = 8,
                                  .visible = false},
                                 ep::renderer_monotonic_now_ns(), 0);
    CHECK(status == ep::SurfaceStatus::ok);
    shader_failure.inject_failure(ep::SurfaceFailurePoint::shader_compilation);
    CHECK(shader_failure.prepare({.max_vertices = 16, .max_draw_batches = 4, .circle_segments = 3},
                                 {}) == ep::SurfaceStatus::shader_compilation_failed);
    CHECK(!shader_failure.resource_stats().opengl_ready);
    shader_failure.close();

    ep::PresentationSurface cancellable;
    status = cancellable.open({.title = "PyNeurale cancellation test",
                               .window_size = {160, 120},
                               .monitor_idx = 0,
                               .swap_interval = 0,
                               .input_capacity = 8,
                               .visible = false},
                              ep::renderer_monotonic_now_ns(), 0);
    CHECK(status == ep::SurfaceStatus::ok);
    std::thread cancel_thread([&cancellable] { cancellable.cancel(); });
    cancel_thread.join();
    CHECK(cancellable.pump_events() == ep::SurfaceStatus::cancelled);
    cancellable.cancel();
    cancellable.close();

    ep::PresentationSurface wrong_thread;
    status = wrong_thread.open({.title = "PyNeurale thread-affinity test",
                                .window_size = {160, 120},
                                .monitor_idx = 0,
                                .swap_interval = 0,
                                .input_capacity = 8,
                                .visible = false},
                               ep::renderer_monotonic_now_ns(), 0);
    CHECK(status == ep::SurfaceStatus::ok);
    ep::SurfaceStatus cross_thread_poll_status{ep::SurfaceStatus::ok};
    std::thread poll_thread(
        [&wrong_thread, &cross_thread_poll_status]
        {
            ep::InputEvent ignored{};
            bool available{true};
            cross_thread_poll_status = wrong_thread.poll_input(ignored, available);
        });
    poll_thread.join();
    CHECK(cross_thread_poll_status == ep::SurfaceStatus::wrong_thread);
    CHECK(wrong_thread.status() == ep::SurfaceStatus::ok);
    CHECK(wrong_thread.lifecycle() == ep::SurfaceLifecycle::open);
    wrong_thread.close();

    // cross-thread close is a contract violation, not a teardown: a live
    // surface closed from the wrong thread must report wrong_thread, leave the
    // window open, and let the owner thread tear it down safely afterward. This
    // characterizes the lifetime rule the surface freezes -- close/destruction
    // is owner-thread work, and only cancel() is a cross-thread operation.
    ep::PresentationSurface cross_close;
    status = cross_close.open({.title = "PyNeurale cross-thread close test",
                               .window_size = {160, 120},
                               .monitor_idx = 0,
                               .swap_interval = 0,
                               .input_capacity = 8,
                               .visible = false},
                              ep::renderer_monotonic_now_ns(), 0);
    CHECK(status == ep::SurfaceStatus::ok);
    std::thread close_thread([&cross_close] { cross_close.close(); });
    close_thread.join();
    CHECK(cross_close.status() == ep::SurfaceStatus::wrong_thread);
    CHECK(cross_close.lifecycle() == ep::SurfaceLifecycle::open);
    CHECK(cross_close.resource_stats().window_open);
    CHECK(cross_close.pump_events() == ep::SurfaceStatus::ok);
    cross_close.close();
    CHECK(cross_close.lifecycle() == ep::SurfaceLifecycle::closed);
    CHECK(!cross_close.resource_stats().window_open);

    // Two live surfaces are independent. glfwInit()/glfwTerminate() are
    // library-global and GLFW keeps no reference count of its own, so a surface
    // that terminated the library on its own close() would destroy the other
    // surface's window behind its back -- leaving it holding a freed
    // GLFWwindow* that its own close() would then hand to
    // glfwMakeContextCurrent()/glfwDestroyWindow(). Center-Out, WebGrid, and
    // Speech presenters are separately constructible, so this is a supported
    // state and not a misuse. Closing the second must leave the first fully
    // usable: still open, still able to pump and render, and able to close
    // cleanly afterwards.
    ep::PresentationSurface coexisting_a;
    ep::PresentationSurface coexisting_b;
    const ep::WindowConfig shared{.title = "PyNeurale coexisting surface test",
                                  .window_size = {160, 120},
                                  .logical_space = {-1.0, -1.0, 2.0, 2.0},
                                  .monitor_idx = 0,
                                  .swap_interval = 0,
                                  .input_capacity = 8,
                                  .visible = false};
    CHECK(coexisting_a.open(shared, ep::renderer_monotonic_now_ns(), 0) == ep::SurfaceStatus::ok);
    CHECK(coexisting_b.open(shared, ep::renderer_monotonic_now_ns(), 0) == ep::SurfaceStatus::ok);
    const ep::SurfacePreparation coexisting{
        .max_vertices = 256, .max_draw_batches = 8, .circle_segments = 16};
    CHECK(coexisting_a.prepare(coexisting, {}) == ep::SurfaceStatus::ok);
    CHECK(coexisting_b.prepare(coexisting, {}) == ep::SurfaceStatus::ok);
    coexisting_b.close();
    CHECK(coexisting_b.lifecycle() == ep::SurfaceLifecycle::closed);
    CHECK(coexisting_a.lifecycle() == ep::SurfaceLifecycle::prepared);
    CHECK(coexisting_a.resource_stats().window_open);
    CHECK(coexisting_a.pump_events() == ep::SurfaceStatus::ok);
    // Rendering after the other surface closed also proves the context binding
    // followed the surface rather than staying wherever the last open() left it.
    CHECK(coexisting_a.begin_frame({0.0F, 0.0F, 0.0F, 1.0F}) == ep::SurfaceStatus::ok);
    CHECK(coexisting_a.circle({0.0, 0.0}, 0.25, {1.0F, 1.0F, 1.0F, 1.0F}, true) ==
          ep::SurfaceStatus::ok);
    CHECK(coexisting_a.present(1, 1).status == ep::SurfaceStatus::ok);
    coexisting_a.close();
    CHECK(coexisting_a.lifecycle() == ep::SurfaceLifecycle::closed);

    // Fullscreen is tested only when explicitly requested because many headless
    // display servers intentionally expose no switchable video mode.
    if (std::getenv("NEURALE_PRESENTATION_TEST_FULLSCREEN") != nullptr)
    {
#if defined(_WIN32)
        DEVMODEW desktop_mode{};
        desktop_mode.dmSize = sizeof(desktop_mode);
        CHECK(EnumDisplaySettingsW(nullptr, ENUM_CURRENT_SETTINGS, &desktop_mode) != 0);
#endif
        ep::PresentationSurface fullscreen;
        status = fullscreen.open({.title = "PyNeurale fullscreen presentation test",
                                  .window_size = {320, 240},
                                  .logical_space = {-1.0, -1.0, 2.0, 2.0},
                                  .monitor_idx = 0,
                                  .swap_interval = 0,
                                  .input_capacity = 8,
                                  .fullscreen = true,
                                  .resizable = false,
                                  .visible = false},
                                 ep::renderer_monotonic_now_ns(), 0);
        CHECK(status == ep::SurfaceStatus::ok);
#if defined(_WIN32)
        const auto fullscreen_environment = fullscreen.environment();
        CHECK(fullscreen_environment.window_width == static_cast<int>(desktop_mode.dmPelsWidth));
        CHECK(fullscreen_environment.window_height == static_cast<int>(desktop_mode.dmPelsHeight));
#endif
        fullscreen.close();
    }
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    if (const auto result = test_coordinates(); result != 0)
    {
        return result;
    }
    if (const auto result = test_time_mapping(); result != 0)
    {
        return result;
    }
    if (const auto result = test_input_ordering(); result != 0)
    {
        return result;
    }
    if (const auto result = test_explicit_failures(); result != 0)
    {
        return result;
    }

    const bool window_test = argc == 2 && std::string_view(argv[1]) == "--window";
    if (!window_test)
    {
        return 0;
    }
    const auto* font_path = std::getenv("NEURALE_PRESENTATION_TEST_FONT");
    if (font_path == nullptr || std::string_view(font_path).empty())
    {
        return neurale::presentation_test::skip_or_fail(
            "no font: set NEURALE_PRESENTATION_TEST_FONT");
    }
    if (const auto result = test_text_prepare(font_path); result != 0)
    {
        return result;
    }
    if (const auto* cjk_font_path = std::getenv("NEURALE_PRESENTATION_TEST_CJK_FONT");
        cjk_font_path != nullptr && std::string_view(cjk_font_path) != "")
    {
        if (const auto result = test_text_cjk(cjk_font_path); result != 0)
        {
            return result;
        }
    }
    return test_window(font_path);
}
