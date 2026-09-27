// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "speech_presenter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include <neurale/experiments/speech.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace neurale::experiment_presentation
{
namespace
{

namespace speech = experiments::speech;

[[nodiscard]] bool valid_color(Color color) noexcept
{
    return std::isfinite(color.red) && std::isfinite(color.green) && std::isfinite(color.blue) &&
           std::isfinite(color.alpha) && color.red >= 0.0F && color.red <= 1.0F &&
           color.green >= 0.0F && color.green <= 1.0F && color.blue >= 0.0F && color.blue <= 1.0F &&
           color.alpha >= 0.0F && color.alpha <= 1.0F;
}

[[nodiscard]] bool valid_point(Point2d point) noexcept
{
    return std::isfinite(point.x) && std::isfinite(point.y);
}

[[nodiscard]] bool contains_line_break(std::string_view text) noexcept
{
    return text.find('\n') != std::string_view::npos || text.find('\r') != std::string_view::npos;
}

} // namespace

SurfaceStatus SpeechPresentationHandoff::prepare(std::size_t capacity)
{
    if (capacity == 0 || capacity == (std::numeric_limits<std::size_t>::max)())
    {
        return SurfaceStatus::invalid_configuration;
    }
    storage_.assign(capacity + 1, SpeechPresentationEvidence{});
    prepared_ = true;
    reset();
    return SurfaceStatus::ok;
}

bool SpeechPresentationHandoff::can_push() const noexcept
{
    if (!prepared_)
    {
        return false;
    }
    const auto tail = tail_.load(std::memory_order_relaxed);
    const auto next = (tail + 1) % storage_.size();
    return next != head_.load(std::memory_order_acquire);
}

SurfaceStatus SpeechPresentationHandoff::push(const SpeechPresentationEvidence& evidence) noexcept
{
    if (!prepared_)
    {
        return SurfaceStatus::invalid_state;
    }
    const auto tail = tail_.load(std::memory_order_relaxed);
    const auto next = (tail + 1) % storage_.size();
    if (next == head_.load(std::memory_order_acquire))
    {
        note_dropped();
        return SurfaceStatus::presentation_handoff_overflow;
    }
    storage_[tail] = evidence;
    tail_.store(next, std::memory_order_release);
    return SurfaceStatus::ok;
}

bool SpeechPresentationHandoff::pop(SpeechPresentationEvidence& evidence) noexcept
{
    if (!prepared_)
    {
        return false;
    }
    const auto head = head_.load(std::memory_order_relaxed);
    if (head == tail_.load(std::memory_order_acquire))
    {
        return false;
    }
    evidence = storage_[head];
    head_.store((head + 1) % storage_.size(), std::memory_order_release);
    return true;
}

void SpeechPresentationHandoff::reset() noexcept
{
    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
    dropped_.store(0, std::memory_order_relaxed);
}

void SpeechPresentationHandoff::close() noexcept
{
    prepared_ = false;
    storage_.clear();
    storage_.shrink_to_fit();
    reset();
}

void SpeechPresentationHandoff::note_dropped() noexcept
{
    dropped_.fetch_add(1, std::memory_order_relaxed);
}

std::size_t SpeechPresentationHandoff::capacity() const noexcept
{
    return storage_.empty() ? 0 : storage_.size() - 1;
}

std::size_t SpeechPresentationHandoff::size() const noexcept
{
    if (!prepared_)
    {
        return 0;
    }
    const auto head = head_.load(std::memory_order_acquire);
    const auto tail = tail_.load(std::memory_order_acquire);
    return tail >= head ? tail - head : storage_.size() - head + tail;
}

SurfaceStatus validate(const SpeechPresentationStyle& style) noexcept
{
    if (!valid_color(style.background) || !valid_color(style.fixation) ||
        !valid_color(style.text) || !valid_point(style.fixation_center) ||
        !valid_point(style.text_baseline) || !std::isfinite(style.fixation_half_extent) ||
        style.fixation_half_extent <= 0.0)
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus validate(const SpeechPresentationConfig& config) noexcept
{
    if (validate(config.style) != SurfaceStatus::ok || config.title.empty() ||
        config.window_size.width <= 0 || config.window_size.height <= 0 ||
        !std::isfinite(config.logical_space.left) || !std::isfinite(config.logical_space.bottom) ||
        !std::isfinite(config.logical_space.width) || !std::isfinite(config.logical_space.height) ||
        config.logical_space.width <= 0.0 || config.logical_space.height <= 0.0 ||
        (config.aspect_policy != AspectPolicy::fit_letterbox &&
         config.aspect_policy != AspectPolicy::stretch) ||
        config.monitor_idx < -1 || config.input_capacity == 0 || config.outcome_capacity == 0 ||
        config.text.font_path.empty() || config.text.pixel_height == 0 ||
        config.text.atlas_width <= 0 || config.text.atlas_height <= 0 ||
        config.text.max_texts == 0 || config.text.max_glyphs == 0 ||
        config.text.max_glyphs > (std::numeric_limits<std::size_t>::max)() / 6)
    {
        return SurfaceStatus::invalid_configuration;
    }
    return SurfaceStatus::ok;
}

SurfaceStatus SpeechCuePresenter::open(const SpeechPresentationConfig& config,
                                       RendererTimeNs renderer_origin_ns,
                                       experiments::ExperimentTimeNs experiment_origin_ns)
{
    if (opened_)
    {
        return SurfaceStatus::invalid_state;
    }
    if (validate(config) != SurfaceStatus::ok)
    {
        return SurfaceStatus::invalid_configuration;
    }
    const auto status =
        surface_.open({.title = config.title,
                       .window_size = config.window_size,
                       .logical_space = config.logical_space,
                       .aspect_policy = config.aspect_policy,
                       .monitor_idx = config.monitor_idx,
                       .swap_interval = config.swap_interval,
                       .input_capacity = config.input_capacity,
                       .input_admission = InputAdmission::key | InputAdmission::window_close,
                       .fullscreen = config.fullscreen,
                       .resizable = config.resizable,
                       .visible = config.visible},
                      renderer_origin_ns, experiment_origin_ns);
    if (status != SurfaceStatus::ok)
    {
        return status;
    }
    // font_path crosses the open -> prepare boundary, and title crosses into
    // the provenance path (validate(config_) in the recording seam); see
    // font_path_'s declaration for why both are owned rather than viewed.
    font_path_.assign(config.text.font_path);
    title_.assign(config.title);
    config_ = config;
    config_.text.font_path = font_path_;
    config_.title = title_;
    opened_ = true;
    return SurfaceStatus::ok;
}

SurfaceStatus SpeechCuePresenter::prepare(const speech::SpeechCatalog& catalog)
{
    if (!opened_ || prepared_)
    {
        return SurfaceStatus::invalid_state;
    }
    std::array<TextCatalogEntry, speech::kMaxSpeechStimuli> entries{};
    if (catalog.count > catalog.entries.size())
    {
        return SurfaceStatus::invalid_configuration;
    }
    for (std::size_t i = 0; i < catalog.count; ++i)
    {
        const auto& stimulus = catalog.entries[i];
        if (stimulus.text_length > stimulus.text.size())
        {
            return SurfaceStatus::invalid_configuration;
        }
        const std::string_view text(stimulus.text.data(), stimulus.text_length);
        if (!valid_utf8(text))
        {
            return SurfaceStatus::invalid_utf8;
        }
        // Deliberately single-line until multiline layout has an explicit
        // baseline/line-height contract.
        if (contains_line_break(text))
        {
            return SurfaceStatus::invalid_configuration;
        }
        entries[i] = {.id = stimulus.id, .utf8 = text};
    }
    if (speech::validate(catalog) != experiments::ContractStatus::ok ||
        catalog.count > config_.text.max_texts)
    {
        return SurfaceStatus::invalid_configuration;
    }

    PreparedFontResource font_resource{};
    if (const auto status = load_font_resource(config_.text, font_resource);
        status != SurfaceStatus::ok)
    {
        return status;
    }
    if (const auto status = outcomes_.prepare(config_.outcome_capacity);
        status != SurfaceStatus::ok)
    {
        return status;
    }
    // The span is held only by this local for the duration of
    // surface_.prepare(); config_.text is left with an empty span so no member
    // ever references the buffer after it is released.
    auto text_config = config_.text;
    text_config.font_bytes = font_resource.bytes;
    const auto max_vertices = std::max<std::size_t>(4, config_.text.max_glyphs * 6);
    const auto surface_status =
        surface_.prepare({.max_vertices = max_vertices,
                          .max_draw_batches = std::max<std::size_t>(2, config_.text.max_glyphs),
                          .circle_segments = 3,
                          .text = text_config},
                         std::span<const TextCatalogEntry>(entries.data(), catalog.count));
    if (surface_status != SurfaceStatus::ok)
    {
        outcomes_.close();
        return surface_status;
    }
    catalog_ = catalog;
    catalog_fingerprint_ = speech::catalog_fingerprint(catalog);
    font_identity_ = std::move(font_resource.identity);
    outcome_sequences_.reset();
    has_request_ = false;
    prepared_ = true;
    return SurfaceStatus::ok;
}

SurfaceStatus
SpeechCuePresenter::validate_request(const experiments::PresentationRequest& request) const noexcept
{
    if (experiments::validate(request) != experiments::ContractStatus::ok ||
        request.phase > static_cast<experiments::PhaseId>(speech::SpeechPhase::content))
    {
        return SurfaceStatus::invalid_configuration;
    }
    const auto phase = static_cast<speech::SpeechPhase>(request.phase);
    if (speech::speech_cue_of(phase) != request.cue)
    {
        return SurfaceStatus::invalid_configuration;
    }
    if (request.cue == experiments::CueKind::text_content)
    {
        speech::SpeechStimulus ignored{};
        if (speech::find_stimulus(catalog_, request.stimulus_id, ignored) !=
            experiments::ContractStatus::ok)
        {
            return SurfaceStatus::text_not_prepared;
        }
    }
    return SurfaceStatus::ok;
}

SpeechPresentationResult SpeechCuePresenter::finish(const experiments::PresentationRequest& request,
                                                    experiments::PresentationStatus status,
                                                    SurfaceStatus render_status,
                                                    const SoftwarePresentationTimes& times,
                                                    bool enqueue) noexcept
{
    SpeechPresentationResult result{};
    experiments::SequenceOrdinal sequence{};
    if (outcome_sequences_.issue(sequence) != experiments::ContractStatus::ok)
    {
        result.status = SurfaceStatus::resource_capacity_exceeded;
        return result;
    }
    result.evidence = {
        .request = request,
        .outcome = {.requested_ns = request.requested_ns,
                    .presented_ns = status == experiments::PresentationStatus::presented
                                        ? times.presented_ns
                                        : 0,
                    .sequence = sequence,
                    .request_sequence = request.sequence,
                    .trial = request.trial,
                    .status = status,
                    .stimulus_id = request.stimulus_id},
        .software_times = times,
        .render_status = render_status,
    };
    result.outcome_produced = true;
    if (experiments::validate(result.evidence.outcome) != experiments::ContractStatus::ok)
    {
        result.status = SurfaceStatus::presentation_failed;
        return result;
    }
    if (!enqueue)
    {
        result.status = render_status;
        return result;
    }
    const auto handoff_status = outcomes_.push(result.evidence);
    result.outcome_enqueued = handoff_status == SurfaceStatus::ok;
    result.status = handoff_status == SurfaceStatus::ok ? render_status : handoff_status;
    return result;
}

bool speech_control_event(const InputEvent& input, SpeechPresentationControlEvent& event) noexcept
{
    SpeechPresentationControlKind kind{};
    if (input.kind == InputEventKind::window_close)
    {
        kind = SpeechPresentationControlKind::window_close_requested;
    }
    else if (input.kind == InputEventKind::key && input.action == InputAction::press &&
             input.code == GLFW_KEY_ESCAPE)
    {
        kind = SpeechPresentationControlKind::escape_requested;
    }
    else
    {
        return false;
    }
    event = SpeechPresentationControlEvent{
        .kind = kind,
        .input_ordinal = input.ordinal,
        .renderer_time_ns = input.renderer_time_ns,
        .experiment_time_ns = input.experiment_time_ns,
    };
    return true;
}

SpeechPresentationResult
SpeechCuePresenter::present(const experiments::PresentationRequest& request,
                            experiments::ExperimentTimeNs attempt_ns) noexcept
{
    SpeechPresentationResult result{};
    if (!prepared_)
    {
        result.status = SurfaceStatus::invalid_state;
        return result;
    }
    if (const auto status = validate_request(request); status != SurfaceStatus::ok)
    {
        result.status = status;
        return result;
    }
    if (attempt_ns < request.requested_ns)
    {
        result.status = SurfaceStatus::time_regressed;
        return result;
    }
    if (has_request_ && request.sequence <= last_request_sequence_)
    {
        result.status = SurfaceStatus::invalid_state;
        return result;
    }
    has_request_ = true;
    last_request_sequence_ = request.sequence;

    if (!outcomes_.can_push())
    {
        outcomes_.note_dropped();
        SoftwarePresentationTimes times{};
        times.requested_ns = request.requested_ns;
        times.intended_ns = request.onset_ns;
        return finish(request, experiments::PresentationStatus::skipped,
                      SurfaceStatus::presentation_handoff_overflow, times, false);
    }
    if (request.valid_until_ns != experiments::kNoExpiryNs && attempt_ns >= request.valid_until_ns)
    {
        SoftwarePresentationTimes times{};
        times.requested_ns = request.requested_ns;
        times.intended_ns = request.onset_ns;
        return finish(request, experiments::PresentationStatus::expired, SurfaceStatus::ok, times);
    }

    auto render_status = surface_.begin_frame(config_.style.background);
    if (render_status == SurfaceStatus::ok && request.cue == experiments::CueKind::fixation_cross)
    {
        render_status =
            surface_.fixation_cross(config_.style.fixation_center,
                                    config_.style.fixation_half_extent, config_.style.fixation);
    }
    else if (render_status == SurfaceStatus::ok &&
             request.cue == experiments::CueKind::text_content)
    {
        render_status = surface_.shaped_text(request.stimulus_id, config_.style.text_baseline,
                                             config_.style.text);
    }
    // Preserve the semantic requested/intended times even when begin_frame,
    // fixation_cross, or shaped_text fails before surface_.present() runs; a
    // failure must not zero out the request identity still needed as evidence.
    SoftwarePresentationTimes times{};
    times.requested_ns = request.requested_ns;
    times.intended_ns = request.onset_ns;
    if (render_status == SurfaceStatus::ok)
    {
        times = surface_.present(request.requested_ns, request.onset_ns);
        render_status = times.status;
    }
    if (render_status == SurfaceStatus::ok && request.valid_until_ns != experiments::kNoExpiryNs &&
        times.presented_ns >= request.valid_until_ns)
    {
        // The swap-return observation crossed the semantic deadline. The
        // software operation is retained in software_times, but it is never
        // reported as a valid presentation. The renderer itself did not
        // fail, so this carries its own status rather than borrowing
        // presentation_failed: a record saying "presented at T" beside
        // "implementation_status = presentation_failed" states two
        // contradictory things about one frame.
        return finish(request, experiments::PresentationStatus::expired,
                      SurfaceStatus::presentation_deadline_missed, times);
    }
    return finish(request,
                  render_status == SurfaceStatus::ok ? experiments::PresentationStatus::presented
                                                     : experiments::PresentationStatus::skipped,
                  render_status, times);
}

SurfaceStatus SpeechCuePresenter::poll_outcome(SpeechPresentationEvidence& evidence,
                                               bool& available) noexcept
{
    available = outcomes_.pop(evidence);
    return prepared_ ? SurfaceStatus::ok : SurfaceStatus::invalid_state;
}

SurfaceStatus SpeechCuePresenter::pump_events() noexcept
{
    return surface_.pump_events();
}

SurfaceStatus SpeechCuePresenter::poll_control(SpeechPresentationControlEvent& event,
                                               bool& available) noexcept
{
    // Pointer motion and mouse buttons never reach the queue: the surface's
    // admission policy (key | window_close) drops them at the GLFW callback, so
    // a pointer burst cannot overflow the FixedInputQueue regardless of drain
    // timing. Only key and window_close events are admitted; non-ESC keys are
    // drained here and ignored, ESC/window-close become orchestration control.
    available = false;
    InputEvent input{};
    bool input_available{};
    auto status = surface_.poll_input(input, input_available);
    while (status == SurfaceStatus::ok && input_available)
    {
        if (speech_control_event(input, event))
        {
            available = true;
            return SurfaceStatus::ok;
        }
        status = surface_.poll_input(input, input_available);
    }
    return status;
}

SurfaceStatus SpeechCuePresenter::request_window_size(Size2i size) noexcept
{
    return surface_.request_window_size(size);
}

SurfaceStatus SpeechCuePresenter::reset() noexcept
{
    if (!prepared_)
    {
        return SurfaceStatus::invalid_state;
    }
    outcomes_.reset();
    outcome_sequences_.reset();
    has_request_ = false;
    last_request_sequence_ = 0;
    return SurfaceStatus::ok;
}

void SpeechCuePresenter::cancel() noexcept
{
    surface_.cancel();
}

void SpeechCuePresenter::close() noexcept
{
    surface_.close();
    if (surface_.lifecycle() != SurfaceLifecycle::closed)
    {
        return;
    }
    outcomes_.close();
    config_ = {};
    font_path_.clear();
    title_.clear();
    catalog_ = {};
    catalog_fingerprint_ = 0;
    font_identity_ = {};
    outcome_sequences_.reset();
    last_request_sequence_ = 0;
    opened_ = false;
    prepared_ = false;
    has_request_ = false;
}

} // namespace neurale::experiment_presentation
