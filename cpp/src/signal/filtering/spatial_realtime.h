/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <span>
#include <string_view>
#include <vector>

namespace neurale::signal::detail
{

/// Reference statistic estimated across the reference channels at each sample.
enum class ReferenceStatistic
{
    mean,
    median
};

// Sample-wise common reference. The single definition of the arithmetic, used
// by both the real-time data plane and the batch `common_reference()` that
// spatial.cpp exposes.
//
// The batch entry point cannot simply be reused on a real-time thread: it
// validates every sample for finiteness and throws for an ordinary result, and
// its median acquired a workspace on every call. Here the workspace is sized
// once, in the constructor, and process() neither validates nor throws. What
// spatial.cpp keeps is the part that is genuinely batch: argument checking,
// copying the caller's input into the caller's output buffer, and asking for
// the per-sample reference values that the real-time chain has no consumer for.
//
// There is deliberately no backend dispatch. A common reference is a
// memory-bound per-sample reduction; routing it through MKL BLAS would pull the
// intel_thread pool onto the real-time thread for no arithmetic win, which is
// the pathology fft_dispatch.cpp already avoids for short transforms.
class CommonReferenceRealtimeProcessor
{
  public:
    // An empty `reference_channels` means every channel contributes. Throws if
    // the channel count is zero, or if a position repeats or is out of range.
    CommonReferenceRealtimeProcessor(std::size_t n_channels,
                                     std::span<const std::size_t> reference_channels,
                                     ReferenceStatistic statistic = ReferenceStatistic::mean);

    /// Reference in place. Nothing records what was subtracted.
    void process(std::span<double> frame, std::size_t n_samples) noexcept;
    /// Reference in place, recording the value subtracted at each sample.
    ///
    /// `reference` holds one value per sample. Only the batch caller wants it;
    /// the real-time chain has no consumer for it, and the overload above is a
    /// separate template instantiation in which the store does not exist.
    void process(std::span<double> frame, std::size_t n_samples,
                 std::span<double> reference) noexcept;
    void reset() noexcept;

    [[nodiscard]] std::size_t n_channels() const noexcept;
    [[nodiscard]] std::size_t n_reference_channels() const noexcept;
    [[nodiscard]] std::string_view kernel_name() const noexcept;
    [[nodiscard]] std::string_view statistic_name() const noexcept;

  private:
    template <bool EmitReference>
    void apply(double* data, std::size_t n_samples, double* reference) noexcept;
    template <bool EmitReference>
    void apply_mean(double* data, std::size_t n_samples, double* reference) noexcept;
    template <bool EmitReference>
    void apply_median(double* data, std::size_t n_samples, double* reference) noexcept;

    std::size_t n_channels_{1};
    ReferenceStatistic statistic_{ReferenceStatistic::mean};
    // The count itself, and the mean divides by it. Multiplying by a stored
    // reciprocal would be one operation cheaper but moves the last ulp of every
    // mean away from sum/n, and the batch `common_reference()` this kernel now
    // backs is a published API that computed sum/n exactly before the two were
    // merged. Keeping one arithmetic path for both callers is worth more than
    // the multiply: neurale_pipeline_spatial_reference_adapter_benchmark
    // (Release/MKL, 4 samples per frame, medians stable to 1 ns across runs)
    // puts the division at a fixed 2-3 ns per frame -- 91->93 ns at 64
    // channels, 427->426 at 256 where the reduction is memory bound anyway --
    // against a 1 ms frame budget.
    double reference_count_{1.0};
    // Whole-channel-set referencing sums contiguously; a subset pays one
    // indirection per term. Resolved once at construction so process() does not
    // re-derive it per sample.
    bool all_channels_{true};
    std::vector<std::size_t> reference_;
    // Median only. Sized at construction and never resized: moving this single
    // allocation off the per-call path is what lets median run on the data
    // plane at all.
    std::vector<double> workspace_;
    std::size_t middle_{};
    bool even_{};
};

} // namespace neurale::signal::detail
