// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include <neurale/experiments/presentation.h>
#include <neurale/experiments/speech.h>

#include "font_identity.h"
#include "surface.h"

namespace neurale::experiment_presentation
{

struct SpeechPresentationStyle
{
    Color background{0.0F, 0.0F, 0.0F, 1.0F};
    Color fixation{1.0F, 1.0F, 1.0F, 1.0F};
    Color text{1.0F, 1.0F, 1.0F, 1.0F};
    Point2d fixation_center{};
    double fixation_half_extent{0.08};
    Point2d text_baseline{-0.8, 0.0};
};

struct SpeechPresentationConfig
{
    SpeechPresentationStyle style{};
    TextAtlasConfig text{};
    std::string_view title{"PyNeurale Speech cue experiment"};
    Size2i window_size{800, 600};
    Rect2d logical_space{-1.0, -1.0, 2.0, 2.0};
    AspectPolicy aspect_policy{AspectPolicy::fit_letterbox};
    int monitor_idx{-1};
    int swap_interval{1};
    std::size_t input_capacity{64};
    std::size_t outcome_capacity{64};
    bool fullscreen{};
    bool resizable{true};
    bool visible{true};
};

/// One presentation report plus the exact request it answers.
struct SpeechPresentationEvidence
{
    experiments::PresentationRequest request{};
    experiments::PresentationOutcome outcome{};
    SoftwarePresentationTimes software_times{};
    SurfaceStatus render_status{SurfaceStatus::invalid_state};
};

struct SpeechPresentationResult
{
    SurfaceStatus status{SurfaceStatus::invalid_state};
    SpeechPresentationEvidence evidence{};
    bool outcome_produced{};
    bool outcome_enqueued{};
};

/// Prepared SPSC seam from the presentation thread to orchestration.
class SpeechPresentationHandoff
{
  public:
    [[nodiscard]] SurfaceStatus prepare(std::size_t capacity);
    [[nodiscard]] SurfaceStatus push(const SpeechPresentationEvidence& evidence) noexcept;
    [[nodiscard]] bool pop(SpeechPresentationEvidence& evidence) noexcept;
    void reset() noexcept;
    void close() noexcept;

    [[nodiscard]] bool can_push() const noexcept;
    void note_dropped() noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::uint64_t dropped() const noexcept
    {
        return dropped_.load(std::memory_order_acquire);
    }

  private:
    std::vector<SpeechPresentationEvidence> storage_{};
    std::atomic<std::size_t> head_{};
    std::atomic<std::size_t> tail_{};
    std::atomic<std::uint64_t> dropped_{};
    bool prepared_{};
};

[[nodiscard]] SurfaceStatus validate(const SpeechPresentationStyle& style) noexcept;
[[nodiscard]] SurfaceStatus validate(const SpeechPresentationConfig& config) noexcept;

enum class SpeechPresentationControlKind : std::uint8_t
{
    escape_requested = 0,
    window_close_requested,
};

struct SpeechPresentationControlEvent
{
    SpeechPresentationControlKind kind{SpeechPresentationControlKind::escape_requested};
    std::uint64_t input_ordinal{};
    RendererTimeNs renderer_time_ns{};
    experiments::ExperimentTimeNs experiment_time_ns{};
};

/// Translate raw presentation input to Speech orchestration control. Speech has
/// no pointer/selection semantics, so only ESC and window-close become control;
/// every other event (pointer motion, mouse buttons, resize, other keys) is
/// dropped by the caller draining the queue. The result is evidence for the
/// caller; it never mutates SpeechMachine state.
[[nodiscard]] bool speech_control_event(const InputEvent& input,
                                        SpeechPresentationControlEvent& event) noexcept;

/// Private concrete presenter. It never advances SpeechMachine state.
class SpeechCuePresenter
{
  public:
    SpeechCuePresenter() = default;
    ~SpeechCuePresenter() = default;
    SpeechCuePresenter(const SpeechCuePresenter&) = delete;
    SpeechCuePresenter& operator=(const SpeechCuePresenter&) = delete;
    SpeechCuePresenter(SpeechCuePresenter&&) = delete;
    SpeechCuePresenter& operator=(SpeechCuePresenter&&) = delete;

    [[nodiscard]] SurfaceStatus open(const SpeechPresentationConfig& config,
                                     RendererTimeNs renderer_origin_ns,
                                     experiments::ExperimentTimeNs experiment_origin_ns);
    [[nodiscard]] SurfaceStatus prepare(const experiments::speech::SpeechCatalog& catalog);
    [[nodiscard]] SpeechPresentationResult
    present(const experiments::PresentationRequest& request,
            experiments::ExperimentTimeNs attempt_ns) noexcept;
    [[nodiscard]] SurfaceStatus poll_outcome(SpeechPresentationEvidence& evidence,
                                             bool& available) noexcept;
    [[nodiscard]] SurfaceStatus pump_events() noexcept;
    [[nodiscard]] SurfaceStatus poll_control(SpeechPresentationControlEvent& event,
                                             bool& available) noexcept;
    [[nodiscard]] SurfaceStatus request_window_size(Size2i size) noexcept;
    [[nodiscard]] SurfaceStatus reset() noexcept;
    void cancel() noexcept;
    void close() noexcept;

    [[nodiscard]] const FontResourceIdentity& font_identity() const noexcept
    {
        return font_identity_;
    }
    [[nodiscard]] bool prepared() const noexcept
    {
        return prepared_;
    }
    [[nodiscard]] const SpeechPresentationConfig& configuration() const noexcept
    {
        return config_;
    }
    [[nodiscard]] std::uint64_t catalog_fingerprint() const noexcept
    {
        return catalog_fingerprint_;
    }
    [[nodiscard]] ResourceStats resource_stats() const noexcept
    {
        return surface_.resource_stats();
    }
    [[nodiscard]] PresentationEnvironment environment() const noexcept
    {
        return surface_.environment();
    }
    [[nodiscard]] std::size_t outcome_capacity() const noexcept
    {
        return outcomes_.capacity();
    }
    [[nodiscard]] std::uint64_t dropped_outcomes() const noexcept
    {
        return outcomes_.dropped();
    }
    [[nodiscard]] SurfaceLifecycle lifecycle() const noexcept
    {
        return surface_.lifecycle();
    }

    // Private deterministic failure seam used by native tests.
    void inject_failure(SurfaceFailurePoint point) noexcept
    {
        surface_.inject_failure(point);
    }
    void inject_input(const PendingInputEvent& event) noexcept
    {
        surface_.inject_input(event);
    }

  private:
    [[nodiscard]] SurfaceStatus
    validate_request(const experiments::PresentationRequest& request) const noexcept;
    [[nodiscard]] SpeechPresentationResult finish(const experiments::PresentationRequest& request,
                                                  experiments::PresentationStatus status,
                                                  SurfaceStatus render_status,
                                                  const SoftwarePresentationTimes& times,
                                                  bool enqueue = true) noexcept;

    PresentationSurface surface_{};
    SpeechPresentationHandoff outcomes_{};
    /// Owned backing bytes for the two non-owning views config_ carries across
    /// the open() -> prepare() -> provenance boundary.
    std::string font_path_{};
    std::string title_{};
    SpeechPresentationConfig config_{};
    experiments::speech::SpeechCatalog catalog_{};
    std::uint64_t catalog_fingerprint_{};
    FontResourceIdentity font_identity_{};
    experiments::SequenceCounter outcome_sequences_{};
    experiments::SequenceOrdinal last_request_sequence_{};
    bool opened_{};
    bool prepared_{};
    bool has_request_{};
};

} // namespace neurale::experiment_presentation
