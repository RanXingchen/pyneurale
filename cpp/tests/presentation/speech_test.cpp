// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "allocation_counter.h"
#include "skip_policy.h"
#include "speech_presenter.h"

#include "check_returns.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <new>
#include <string_view>
#include <thread>

#include <neurale/experiments/speech.h>

// This suite drives code that allocates through the nothrow forms, which
// allocation_counter.h leaves alone; counting them here keeps the totals
// below honest without changing what every other suite measures.
void* operator new(std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept
{
    allocations.fetch_add(1, std::memory_order_relaxed);
    return std::malloc(size == 0 ? 1 : size);
}

namespace
{

namespace ep = neurale::experiment_presentation;
namespace speech = neurale::experiments::speech;
using neurale::experiments::ContractStatus;
using neurale::experiments::CueKind;
using neurale::experiments::ExperimentTimeNs;
using neurale::experiments::ParadigmId;
using neurale::experiments::PresentationRequest;
using neurale::experiments::PresentationStatus;
using neurale::experiments::StimulusId;

constexpr ParadigmId kParadigm = 77;
constexpr ExperimentTimeNs kOrigin = 1'000'000'000;

void set_text(speech::SpeechStimulus& stimulus, std::string_view text)
{
    stimulus.text = {};
    std::memcpy(stimulus.text.data(), text.data(), text.size());
    stimulus.text_length = static_cast<std::uint8_t>(text.size());
}

speech::SpeechStimulus stimulus(StimulusId id, std::string_view text)
{
    speech::SpeechStimulus result{};
    result.id = id;
    result.content = speech::SpeechContentKind::text;
    set_text(result, text);
    return result;
}

speech::SpeechCatalog catalog()
{
    speech::SpeechCatalog result{};
    result.count = 5;
    result.entries[0] = stimulus(1, "Speech");
    result.entries[1] = stimulus(2, "\xE4\xB8\xAD\xE6\x96\x87"); // 中文
    result.entries[2] = stimulus(3, "Move \xE4\xB8\xAD\xE6\x96\x87");
    result.entries[3] = stimulus(4, "e\xCC\x81"); // decomposed e + acute
    result.entries[4] = stimulus(5, "\xC3\xA9");  // precomposed e-acute
    return result;
}

ep::SpeechPresentationConfig presentation_config(std::string_view font_path,
                                                 std::size_t outcome_capacity = 16)
{
    ep::SpeechPresentationConfig config{};
    config.text = {.font_path = font_path,
                   .face_idx = 0,
                   .pixel_height = 32,
                   .atlas_width = 1024,
                   .atlas_height = 1024,
                   .max_texts = 16,
                   .max_glyphs = 256};
    config.window_size = {320, 240};
    config.monitor_idx = 0;
    config.swap_interval = 0;
    config.input_capacity = 16;
    config.outcome_capacity = outcome_capacity;
    config.visible = false;
    return config;
}

PresentationRequest request(CueKind cue, speech::SpeechPhase phase, StimulusId stimulus_id,
                            std::uint64_t sequence, ExperimentTimeNs requested_ns,
                            ExperimentTimeNs onset_ns, ExperimentTimeNs valid_until_ns)
{
    PresentationRequest result{};
    result.requested_ns = requested_ns;
    result.onset_ns = onset_ns;
    result.valid_until_ns = valid_until_ns;
    result.duration_ns = valid_until_ns - onset_ns;
    result.sequence = sequence;
    result.trial = {.ordinal = 0, .stimulus_id = stimulus_id == 0 ? 1 : stimulus_id};
    result.paradigm = kParadigm;
    result.phase = static_cast<neurale::experiments::PhaseId>(phase);
    result.cue = cue;
    result.stimulus_id = stimulus_id;
    return result;
}

int test_handoff_is_bounded_and_ordered()
{
    ep::SpeechPresentationHandoff handoff{};
    ep::SpeechPresentationEvidence evidence{};
    CHECK(handoff.push(evidence) == ep::SurfaceStatus::invalid_state);
    CHECK(handoff.prepare(2) == ep::SurfaceStatus::ok);
    evidence.outcome.sequence = 10;
    CHECK(handoff.push(evidence) == ep::SurfaceStatus::ok);
    evidence.outcome.sequence = 11;
    CHECK(handoff.push(evidence) == ep::SurfaceStatus::ok);
    CHECK(!handoff.can_push());
    evidence.outcome.sequence = 12;
    CHECK(handoff.push(evidence) == ep::SurfaceStatus::presentation_handoff_overflow);
    CHECK(handoff.dropped() == 1);
    CHECK(handoff.size() == 2);
    CHECK(handoff.pop(evidence) && evidence.outcome.sequence == 10);
    CHECK(handoff.pop(evidence) && evidence.outcome.sequence == 11);
    CHECK(!handoff.pop(evidence));
    handoff.reset();
    CHECK(handoff.dropped() == 0);
    handoff.close();
    CHECK(handoff.capacity() == 0);
    return 0;
}

int test_font_sha256()
{
    const auto path = std::filesystem::temp_directory_path() / "neurale-m9-font-sha256.bin";
    {
        std::ofstream output(path, std::ios::binary | std::ios::trunc);
        CHECK(output.good());
        output.write("abc", 3);
    }
    ep::PreparedFontResource resource{};
    const auto path_string = path.string();
    CHECK(ep::load_font_resource({.font_path = path_string}, resource) == ep::SurfaceStatus::ok);
    constexpr std::array<std::uint8_t, 32> expected{0xBA, 0x78, 0x16, 0xBF, 0x8F, 0x01, 0xCF, 0xEA,
                                                    0x41, 0x41, 0x40, 0xDE, 0x5D, 0xAE, 0x22, 0x23,
                                                    0xB0, 0x03, 0x61, 0xA3, 0x96, 0x17, 0x7A, 0x9C,
                                                    0xB4, 0x10, 0xFF, 0x61, 0xF2, 0x00, 0x15, 0xAD};
    CHECK(resource.identity.sha256 == expected);
    CHECK(resource.identity.path == path_string);
    CHECK(resource.bytes.size() == 3); // one read: hash and FT share these bytes
    std::filesystem::remove(path);
    return 0;
}

std::array<PresentationRequest, 2>
cross_disabled_requests(ExperimentTimeNs start_ns = kOrigin,
                        neurale::experiments::DurationNs black_duration_ns = 1)
{
    speech::SpeechCueConfig config{};
    config.black_bound_ns = black_duration_ns + 1;
    config.content_bound_ns = 6'000'000'000;
    config.n_trials = 1;
    config.schedule = speech::SpeechScheduleKind::explicit_sequence;
    config.sampler_version = neurale::experiments::kCurrentSamplerVersion;
    config.n_explicit = 1;
    config.explicit_schedule[0] = {.ordinal = 0,
                                   .black_duration_ns = black_duration_ns,
                                   .cross_duration_ns = 0,
                                   .content_duration_ns = 5'000'000'000,
                                   .stimulus_id = 1,
                                   .sampler_version = neurale::experiments::kCurrentSamplerVersion,
                                   .cross_enabled = false};
    speech::SpeechMachine machine{};
    speech::SpeechStepResult step{};
    if (machine.start(kParadigm, config, config.explicit_schedule[0], start_ns, step) !=
            ContractStatus::ok ||
        step.n_requests != 1)
    {
        std::abort();
    }
    const auto black = step.requests[0];
    if (machine.step(start_ns + black_duration_ns, step) != ContractStatus::ok ||
        step.n_requests != 1)
    {
        std::abort();
    }
    return {black, step.requests[0]};
}

int test_cross_disabled_sequence()
{
    const auto requests = cross_disabled_requests();
    CHECK(requests[0].cue == CueKind::black);
    CHECK(requests[1].cue == CueKind::text_content);
    return 0;
}

// GLFW_KEY_ESCAPE / GLFW_KEY_A without pulling <GLFW/glfw3.h> into the test.
constexpr int kGlfwKeyEscape = 256;
constexpr int kGlfwKeyA = 65;

int test_speech_control_event()
{
    ep::InputEvent input{};
    ep::SpeechPresentationControlEvent event{};
    // Non-control events are dropped: pointer, mouse, resize, key release, and
    // non-ESC keys never become orchestration control for a Speech run.
    input.kind = ep::InputEventKind::pointer_moved;
    CHECK(!ep::speech_control_event(input, event));
    input.kind = ep::InputEventKind::mouse_button;
    CHECK(!ep::speech_control_event(input, event));
    input.kind = ep::InputEventKind::framebuffer_resized;
    CHECK(!ep::speech_control_event(input, event));
    input.kind = ep::InputEventKind::key;
    input.action = ep::InputAction::release;
    input.code = kGlfwKeyEscape;
    CHECK(!ep::speech_control_event(input, event));
    input.action = ep::InputAction::press;
    input.code = kGlfwKeyA;
    CHECK(!ep::speech_control_event(input, event));
    // ESC press -> escape_requested, carrying identity and timestamps.
    input.code = kGlfwKeyEscape;
    input.ordinal = 7;
    input.renderer_time_ns = 100;
    input.experiment_time_ns = 1000;
    CHECK(ep::speech_control_event(input, event));
    CHECK(event.kind == ep::SpeechPresentationControlKind::escape_requested);
    CHECK(event.input_ordinal == 7);
    CHECK(event.experiment_time_ns == 1000);
    // window close -> window_close_requested.
    input.kind = ep::InputEventKind::window_close;
    input.ordinal = 9;
    CHECK(ep::speech_control_event(input, event));
    CHECK(event.kind == ep::SpeechPresentationControlKind::window_close_requested);
    CHECK(event.input_ordinal == 9);
    return 0;
}

int test_non_control_input_cannot_overflow_speech_queue()
{
    // A single glfwPollEvents() delivers every cursor/mouse/key callback before
    // the presenter drains, so a burst of irrelevant events can overflow the
    // FixedInputQueue (a sticky input_overflow fault) if they are queued. Speech
    // has no pointer/selection semantics, so the surface's admission policy drops
    // pointer_moved and mouse_button at enqueue -- before the queue. Keyboard
    // orchestration is ESC-press-only, so non-ESC keys and ESC release/repeat are
    // dropped at enqueue too. A Speech run cannot be faulted by pointer, mouse,
    // or keyboard motion regardless of drain timing. inject_input routes through
    // enqueue(), so it exercises the same admission a real GLFW callback would.
    // No font is needed: open prepares the input queue, and this test never calls
    // prepare (which would need a font).
    ep::SpeechCuePresenter presenter{};
    auto config = presentation_config("placeholder-font-path"); // non-empty; not read by open
    config.input_capacity = 8;
    CHECK(presenter.open(config, ep::renderer_monotonic_now_ns(), kOrigin) ==
          ep::SurfaceStatus::ok);

    // A pointer burst far beyond capacity, with NO drain between injections:
    // this is the scenario drain alone cannot survive (overflow is sticky).
    // Admission drops each at enqueue, so the queue stays empty and the surface
    // never faults.
    ep::PendingInputEvent pointer{};
    pointer.kind = ep::InputEventKind::pointer_moved;
    pointer.renderer_time_ns = ep::renderer_monotonic_now_ns();
    for (std::size_t i = 0; i < config.input_capacity * 4; ++i)
    {
        presenter.inject_input(pointer);
    }
    CHECK(presenter.lifecycle() == ep::SurfaceLifecycle::open);
    ep::SpeechPresentationControlEvent event{};
    bool available{};
    CHECK(presenter.poll_control(event, available) == ep::SurfaceStatus::ok);
    CHECK(!available); // queue empty: admission dropped every pointer

    // Mouse buttons are equally irrelevant to Speech and dropped the same way.
    ep::PendingInputEvent button{};
    button.kind = ep::InputEventKind::mouse_button;
    button.action = ep::InputAction::press;
    button.renderer_time_ns = ep::renderer_monotonic_now_ns();
    for (std::size_t i = 0; i < config.input_capacity * 4; ++i)
    {
        presenter.inject_input(button);
    }
    CHECK(presenter.lifecycle() == ep::SurfaceLifecycle::open);
    CHECK(presenter.poll_control(event, available) == ep::SurfaceStatus::ok);
    CHECK(!available);

    // Non-ESC keys, ESC release, and ESC repeat are dropped at enqueue: only an
    // ESC press is admitted. A keyboard burst cannot overflow the queue, and no
    // non-ESC key or non-press ESC ever reaches poll_control.
    ep::PendingInputEvent other_key{};
    other_key.kind = ep::InputEventKind::key;
    other_key.action = ep::InputAction::press;
    other_key.code = kGlfwKeyA;
    other_key.renderer_time_ns = ep::renderer_monotonic_now_ns();
    ep::PendingInputEvent esc_release = other_key;
    esc_release.code = kGlfwKeyEscape;
    esc_release.action = ep::InputAction::release;
    ep::PendingInputEvent esc_repeat = esc_release;
    esc_repeat.action = ep::InputAction::repeat;
    for (std::size_t i = 0; i < config.input_capacity * 4; ++i)
    {
        presenter.inject_input(other_key);
        presenter.inject_input(esc_release);
        presenter.inject_input(esc_repeat);
    }
    CHECK(presenter.lifecycle() == ep::SurfaceLifecycle::open);
    CHECK(presenter.poll_control(event, available) == ep::SurfaceStatus::ok);
    CHECK(!available); // queue still empty: only ESC press is admitted

    // ESC press and window-close are admitted and each produce one timestamped control.
    ep::PendingInputEvent esc{};
    esc.kind = ep::InputEventKind::key;
    esc.action = ep::InputAction::press;
    esc.code = kGlfwKeyEscape;
    esc.renderer_time_ns = ep::renderer_monotonic_now_ns();
    presenter.inject_input(esc);
    CHECK(presenter.poll_control(event, available) == ep::SurfaceStatus::ok);
    CHECK(available);
    CHECK(event.kind == ep::SpeechPresentationControlKind::escape_requested);

    ep::PendingInputEvent close{};
    close.kind = ep::InputEventKind::window_close;
    close.renderer_time_ns = ep::renderer_monotonic_now_ns();
    presenter.inject_input(close);
    CHECK(presenter.poll_control(event, available) == ep::SurfaceStatus::ok);
    CHECK(available);
    CHECK(event.kind == ep::SpeechPresentationControlKind::window_close_requested);

    presenter.close();
    return 0;
}

int check_repeated_glyph_bound(std::string_view font, std::string_view text,
                               std::size_t occurrences)
{
    // max_glyphs bounds shaped glyphs per prepared text, not just unique atlas
    // glyphs. "AAAAAAAA" has one unique atlas glyph but eight shaped occurrences.
    const std::array entry{ep::TextCatalogEntry{1, text}};
    ep::TextAtlasConfig too_small{.font_path = font,
                                  .pixel_height = 32,
                                  .atlas_width = 1024,
                                  .atlas_height = 1024,
                                  .max_texts = 1,
                                  .max_glyphs = occurrences - 1};
    ep::PreparedTextAtlas rejected{};
    CHECK(rejected.prepare(too_small, entry) == ep::SurfaceStatus::resource_capacity_exceeded);
    ep::TextAtlasConfig adequate{.font_path = font,
                                 .pixel_height = 32,
                                 .atlas_width = 1024,
                                 .atlas_height = 1024,
                                 .max_texts = 1,
                                 .max_glyphs = occurrences};
    ep::PreparedTextAtlas accepted{};
    CHECK(accepted.prepare(adequate, entry) == ep::SurfaceStatus::ok);
    const auto* prepared = accepted.find(1);
    CHECK(prepared != nullptr);
    CHECK(prepared->glyphs.size() == occurrences);
    return 0;
}

int test_repeated_glyph_bounds(std::string_view latin, std::string_view cjk)
{
    if (const auto r = check_repeated_glyph_bound(latin, "AAAAAAAA", 8); r != 0)
        return r;
    if (const auto r =
            check_repeated_glyph_bound(cjk, "\xE5\x93\x88\xE5\x93\x88\xE5\x93\x88\xE5\x93\x88", 4);
        r != 0)
    {
        return r;
    }
    return 0;
}

int test_text_contract(std::string_view latin_font, std::string_view cjk_font)
{
    const std::array entries{
        ep::TextCatalogEntry{1, "Speech"},
        ep::TextCatalogEntry{2, "\xE4\xB8\xAD\xE6\x96\x87"},
        ep::TextCatalogEntry{3, "Move \xE4\xB8\xAD\xE6\x96\x87"},
        ep::TextCatalogEntry{4, "e\xCC\x81"},
        ep::TextCatalogEntry{5, "\xC3\xA9"},
    };
    const ep::TextAtlasConfig config{.font_path = cjk_font,
                                     .pixel_height = 32,
                                     .atlas_width = 1024,
                                     .atlas_height = 1024,
                                     .max_texts = entries.size(),
                                     .max_glyphs = 256};
    ep::PreparedTextAtlas first{};
    ep::PreparedTextAtlas second{};
    CHECK(first.prepare(config, entries) == ep::SurfaceStatus::ok);
    CHECK(second.prepare(config, entries) == ep::SurfaceStatus::ok);
    for (const auto& entry : entries)
    {
        const auto* left = first.find(entry.id);
        const auto* right = second.find(entry.id);
        CHECK(left != nullptr && right != nullptr);
        CHECK(left->utf8 == entry.utf8 && right->utf8 == entry.utf8);
        CHECK(left->glyphs.size() == right->glyphs.size());
        for (std::size_t i = 0; i < left->glyphs.size(); ++i)
        {
            CHECK(left->glyphs[i].atlas_entry == right->glyphs[i].atlas_entry);
            CHECK(left->glyphs[i].x_offset == right->glyphs[i].x_offset);
            CHECK(left->glyphs[i].x_advance == right->glyphs[i].x_advance);
        }
    }
    CHECK(first.find(4)->utf8 != first.find(5)->utf8); // no Unicode normalization

    ep::PreparedTextAtlas missing{};
    const std::array chinese{entries[1]};
    auto latin_config = config;
    latin_config.font_path = latin_font;
    CHECK(missing.prepare(latin_config, chinese) == ep::SurfaceStatus::glyph_missing);
    const std::array invalid{ep::TextCatalogEntry{9, "\xC0\xAF"}};
    CHECK(missing.prepare(config, invalid) == ep::SurfaceStatus::invalid_utf8);
    return 0;
}

bool pop(ep::SpeechCuePresenter& presenter, ep::SpeechPresentationEvidence& evidence)
{
    bool available{};
    if (presenter.poll_outcome(evidence, available) != ep::SurfaceStatus::ok)
    {
        return false;
    }
    return available;
}

int test_presenter(std::string_view cjk_font)
{
    ep::SpeechCuePresenter presenter{};
    const auto config = presentation_config(cjk_font);
    const auto renderer_origin = ep::renderer_monotonic_now_ns();
    auto status = presenter.open(config, renderer_origin, kOrigin);
    if (status == ep::SurfaceStatus::glfw_initialization_failed ||
        status == ep::SurfaceStatus::window_creation_failed ||
        status == ep::SurfaceStatus::monitor_unavailable ||
        status == ep::SurfaceStatus::opengl_function_missing)
    {
        return neurale::presentation_test::skip_or_fail(ep::surface_status_name(status));
    }
    CHECK(status == ep::SurfaceStatus::ok);
    CHECK(presenter.prepare(catalog()) == ep::SurfaceStatus::ok);
    CHECK(presenter.font_identity().path == cjk_font);
    CHECK(std::any_of(presenter.font_identity().sha256.begin(),
                      presenter.font_identity().sha256.end(),
                      [](std::uint8_t value) { return value != 0; }));
    const auto prepared_resources = presenter.resource_stats();
    CHECK(prepared_resources.prepared_text_count == 5);

    const auto deadline = kOrigin + 5'000'000'000;
    const std::array requests{
        request(CueKind::black, speech::SpeechPhase::black, 0, 1, kOrigin, kOrigin, deadline),
        request(CueKind::fixation_cross, speech::SpeechPhase::cross, 0, 2, kOrigin, kOrigin + 1,
                deadline),
        request(CueKind::text_content, speech::SpeechPhase::content, 1, 3, kOrigin, kOrigin + 2,
                deadline),
        request(CueKind::text_content, speech::SpeechPhase::content, 2, 4, kOrigin, kOrigin + 3,
                deadline),
        request(CueKind::text_content, speech::SpeechPhase::content, 3, 5, kOrigin, kOrigin + 4,
                deadline),
    };
    for (const auto& item : requests)
    {
        const auto result = presenter.present(item, item.onset_ns);
        if (result.status != ep::SurfaceStatus::ok)
        {
            std::cerr << "request " << item.sequence
                      << " failed: " << ep::surface_status_name(result.status) << '\n';
        }
        CHECK(result.status == ep::SurfaceStatus::ok);
        CHECK(result.outcome_produced && result.outcome_enqueued);
        ep::SpeechPresentationEvidence evidence{};
        CHECK(pop(presenter, evidence));
        CHECK(evidence.request.sequence == item.sequence);
        CHECK(evidence.request.onset_ns == item.onset_ns);
        CHECK(evidence.outcome.request_sequence == item.sequence);
        CHECK(evidence.outcome.trial.ordinal == item.trial.ordinal);
        CHECK(evidence.outcome.stimulus_id == item.stimulus_id);
        CHECK(evidence.outcome.status == PresentationStatus::presented);
        CHECK(evidence.outcome.requested_ns == item.requested_ns);
        CHECK(evidence.outcome.presented_ns >= item.requested_ns);
        CHECK(evidence.software_times.requested_ns == item.requested_ns);
        CHECK(evidence.software_times.intended_ns == item.onset_ns);
    }
    const auto reused_resources = presenter.resource_stats();
    CHECK(reused_resources.prepared_text_count == prepared_resources.prepared_text_count);
    CHECK(reused_resources.prepared_glyph_count == prepared_resources.prepared_glyph_count);

    auto unknown = request(CueKind::text_content, speech::SpeechPhase::content, 99, 6, kOrigin,
                           kOrigin + 5, deadline);
    CHECK(presenter.present(unknown, unknown.onset_ns).status ==
          ep::SurfaceStatus::text_not_prepared);

    auto expired =
        request(CueKind::black, speech::SpeechPhase::black, 0, 6, kOrigin, kOrigin, kOrigin + 10);
    auto expired_result = presenter.present(expired, expired.valid_until_ns);
    CHECK(expired_result.status == ep::SurfaceStatus::ok);
    ep::SpeechPresentationEvidence evidence{};
    CHECK(pop(presenter, evidence));
    CHECK(evidence.outcome.status == PresentationStatus::expired);
    CHECK(evidence.outcome.presented_ns == 0);
    CHECK(presenter.present(expired, expired.valid_until_ns).status ==
          ep::SurfaceStatus::invalid_state);
    auto earlier = expired;
    earlier.sequence = 5;
    CHECK(presenter.present(earlier, earlier.requested_ns).status ==
          ep::SurfaceStatus::invalid_state);

    CHECK(presenter.reset() == ep::SurfaceStatus::ok);
    const auto allocation_baseline = allocations.load(std::memory_order_relaxed);
    for (std::uint64_t sequence = 1; sequence <= 32; ++sequence)
    {
        const auto repeated = request(CueKind::text_content, speech::SpeechPhase::content, 2,
                                      sequence, kOrigin, kOrigin, kOrigin + 20'000'000'000);
        const auto repeated_result = presenter.present(repeated, repeated.onset_ns);
        CHECK(repeated_result.status == ep::SurfaceStatus::ok);
        CHECK(pop(presenter, evidence));
        CHECK(evidence.outcome.sequence == sequence - 1);
    }
    CHECK(allocations.load(std::memory_order_relaxed) == allocation_baseline);
    CHECK(presenter.resource_stats().prepared_glyph_count ==
          prepared_resources.prepared_glyph_count);

    CHECK(presenter.reset() == ep::SurfaceStatus::ok);
    const auto now = ep::renderer_monotonic_now_ns();
    constexpr auto black_duration = std::chrono::milliseconds(100);
    const auto no_cross = cross_disabled_requests(
        kOrigin + (now - renderer_origin),
        static_cast<neurale::experiments::DurationNs>(black_duration.count()) * 1'000'000);
    for (const auto& item : no_cross)
    {
        if (item.cue == CueKind::text_content)
        {
            std::this_thread::sleep_for(black_duration);
        }
        CHECK(item.cue != CueKind::fixation_cross);
        const auto no_cross_result = presenter.present(item, item.onset_ns);
        if (no_cross_result.status != ep::SurfaceStatus::ok)
        {
            std::cerr << "cross-disabled request " << item.sequence
                      << " failed: " << ep::surface_status_name(no_cross_result.status) << '\n';
        }
        CHECK(no_cross_result.status == ep::SurfaceStatus::ok);
        CHECK(pop(presenter, evidence));
        CHECK(evidence.request.cue == item.cue);
    }

    presenter.cancel();
    presenter.cancel();
    CHECK(presenter.pump_events() == ep::SurfaceStatus::cancelled);
    presenter.close();
    presenter.close();
    CHECK(presenter.lifecycle() == ep::SurfaceLifecycle::closed);
    return 0;
}

int test_saturation_and_renderer_failure(std::string_view cjk_font)
{
    ep::SpeechCuePresenter saturated{};
    auto config = presentation_config(cjk_font, 1);
    CHECK(saturated.open(config, ep::renderer_monotonic_now_ns(), kOrigin) ==
          ep::SurfaceStatus::ok);
    CHECK(saturated.prepare(catalog()) == ep::SurfaceStatus::ok);
    const auto first = request(CueKind::black, speech::SpeechPhase::black, 0, 1, kOrigin, kOrigin,
                               kOrigin + 5'000'000'000);
    const auto second = request(CueKind::black, speech::SpeechPhase::black, 0, 2, kOrigin, kOrigin,
                                kOrigin + 5'000'000'000);
    CHECK(saturated.present(first, first.onset_ns).status == ep::SurfaceStatus::ok);
    const auto overflow = saturated.present(second, second.onset_ns);
    CHECK(overflow.status == ep::SurfaceStatus::presentation_handoff_overflow);
    CHECK(overflow.outcome_produced && !overflow.outcome_enqueued);
    CHECK(overflow.evidence.outcome.status == PresentationStatus::skipped);
    CHECK(saturated.dropped_outcomes() == 1);
    ep::SpeechPresentationEvidence evidence{};
    CHECK(pop(saturated, evidence));
    saturated.close();

    ep::SpeechCuePresenter failed{};
    config.outcome_capacity = 2;
    CHECK(failed.open(config, ep::renderer_monotonic_now_ns(), kOrigin) == ep::SurfaceStatus::ok);
    CHECK(failed.prepare(catalog()) == ep::SurfaceStatus::ok);
    failed.inject_failure(ep::SurfaceFailurePoint::presentation);
    const auto failure = failed.present(first, first.onset_ns);
    CHECK(failure.status == ep::SurfaceStatus::presentation_failed);
    CHECK(failure.outcome_enqueued);
    CHECK(pop(failed, evidence));
    CHECK(evidence.outcome.status == PresentationStatus::skipped);
    CHECK(evidence.render_status == ep::SurfaceStatus::presentation_failed);
    failed.close();
    return 0;
}

int test_missing_chinese_fails_prepare(std::string_view latin_font)
{
    ep::SpeechCuePresenter presenter{};
    auto config = presentation_config(latin_font);
    CHECK(presenter.open(config, ep::renderer_monotonic_now_ns(), kOrigin) ==
          ep::SurfaceStatus::ok);
    speech::SpeechCatalog chinese{};
    chinese.count = 1;
    chinese.entries[0] = stimulus(1, "\xE4\xB8\xAD\xE6\x96\x87");
    CHECK(presenter.prepare(chinese) == ep::SurfaceStatus::glyph_missing);
    presenter.close();
    return 0;
}

int test_invalid_utf8_fails_prepare(std::string_view cjk_font)
{
    ep::SpeechCuePresenter presenter{};
    const auto config = presentation_config(cjk_font);
    CHECK(presenter.open(config, ep::renderer_monotonic_now_ns(), kOrigin) ==
          ep::SurfaceStatus::ok);
    speech::SpeechCatalog invalid{};
    invalid.count = 1;
    invalid.entries[0] = stimulus(1, "\xC0\xAF");
    CHECK(presenter.prepare(invalid) == ep::SurfaceStatus::invalid_utf8);
    presenter.close();
    return 0;
}

int test_multiline_fails_prepare(std::string_view cjk_font)
{
    ep::SpeechCuePresenter presenter{};
    const auto config = presentation_config(cjk_font);
    CHECK(presenter.open(config, ep::renderer_monotonic_now_ns(), kOrigin) ==
          ep::SurfaceStatus::ok);
    speech::SpeechCatalog multiline{};
    multiline.count = 1;
    multiline.entries[0] = stimulus(1, "first\n\xE7\xAC\xAC\xE4\xBA\x8C");
    CHECK(presenter.prepare(multiline) == ep::SurfaceStatus::invalid_configuration);
    presenter.close();
    return 0;
}

int test_minimal_text_capacity_still_renders_cross(std::string_view cjk_font)
{
    ep::SpeechCuePresenter presenter{};
    auto config = presentation_config(cjk_font);
    config.text.max_texts = 1;
    config.text.max_glyphs = 1;
    CHECK(presenter.open(config, ep::renderer_monotonic_now_ns(), kOrigin) ==
          ep::SurfaceStatus::ok);
    speech::SpeechCatalog minimal{};
    minimal.count = 1;
    minimal.entries[0] = stimulus(1, "A");
    CHECK(presenter.prepare(minimal) == ep::SurfaceStatus::ok);
    const auto cross = request(CueKind::fixation_cross, speech::SpeechPhase::cross, 0, 1, kOrigin,
                               kOrigin, kOrigin + 5'000'000'000);
    CHECK(presenter.present(cross, cross.onset_ns).status == ep::SurfaceStatus::ok);
    ep::SpeechPresentationEvidence evidence{};
    CHECK(pop(presenter, evidence));
    CHECK(evidence.outcome.status == PresentationStatus::presented);
    presenter.close();
    return 0;
}

int test_window()
{
    if (const auto result = test_non_control_input_cannot_overflow_speech_queue(); result != 0)
    {
        return result;
    }
    const auto* latin = std::getenv("NEURALE_PRESENTATION_TEST_FONT");
    const auto* cjk = std::getenv("NEURALE_PRESENTATION_TEST_CJK_FONT");
    if (latin == nullptr || *latin == '\0' || cjk == nullptr || *cjk == '\0')
    {
        return neurale::presentation_test::skip_or_fail(
            "no fonts: set NEURALE_PRESENTATION_TEST_FONT and "
            "NEURALE_PRESENTATION_TEST_CJK_FONT");
    }
    if (const auto result = test_repeated_glyph_bounds(latin, cjk); result != 0)
    {
        return result;
    }
    if (const auto result = test_text_contract(latin, cjk); result != 0)
    {
        return result;
    }
    if (const auto result = test_missing_chinese_fails_prepare(latin); result != 0)
    {
        return result;
    }
    if (const auto result = test_invalid_utf8_fails_prepare(cjk); result != 0)
    {
        return result;
    }
    if (const auto result = test_multiline_fails_prepare(cjk); result != 0)
    {
        return result;
    }
    if (const auto result = test_minimal_text_capacity_still_renders_cross(cjk); result != 0)
    {
        return result;
    }
    if (const auto result = test_presenter(cjk); result != 0)
    {
        return result;
    }
    return test_saturation_and_renderer_failure(cjk);
}

} // namespace

int main(int argc, char** argv)
{
    if (const auto result = test_handoff_is_bounded_and_ordered(); result != 0)
    {
        return result;
    }
    if (const auto result = test_font_sha256(); result != 0)
    {
        return result;
    }
    if (const auto result = test_cross_disabled_sequence(); result != 0)
    {
        return result;
    }
    if (const auto result = test_speech_control_event(); result != 0)
    {
        return result;
    }
    if (argc == 2 && std::string_view(argv[1]) == "--window")
    {
        return test_window();
    }
    return 0;
}
