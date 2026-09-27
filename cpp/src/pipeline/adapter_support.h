/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>

#include <neurale/streaming/clock.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/schema.h>

/// Prepare-time arithmetic and comparison shared by the pipeline adapters.
///
/// Every adapter under `cpp/src/pipeline/` resolves the same geometry before it
/// runs: overflow-checked sizes, an exact window duration in nanoseconds, the
/// observation rate a shift implies, and the host time a block's device tick
/// maps to.
///
/// Two boundaries keep this a support layer rather than a framework:
///
/// * **Nothing here runs on the real-time thread.** The throwing functions are
///   prepare-time by construction: the contract forbids throwing for an ordinary
///   result on the data plane, so a checked size can only be resolved before
///   `start()`. `scaled_duration`, `device_tick_host_time` and `equal_unit` are
///   `noexcept` and answer with a bool, which is what lets the first two be
///   called from a `process()` that must not throw.
/// * **No adapter lifecycle, no kernel.** These are plain functions over values,
///   and :class:`Checked` is a `constexpr` value holding a `string_view`. There
///   is no base class, no virtual dispatch, no type erasure and no
///   `std::function`, so an adapter that calls them compiles to what its own
///   copy compiled to.
///
/// The throwing functions hang off :class:`Checked`, which an adapter binds once
/// to its own name: three feature adapters can sit in one chain, and "size
/// overflows size_t" without a name does not say which one refused.
///
/// Deliberately **not** here: `checked_workspace_bytes` (the FIR, IIR and SOS
/// adapters each compute a different bound under one name), `checked_block_samples`
/// as the Kalman decoder spells it (it takes a `std::uint32_t`, checks only for
/// zero, and throws `std::invalid_argument`), and every adapter-specific helper.
/// A shared name is not a shared function.
namespace neurale::pipeline::adapter_support
{

[[nodiscard]] inline std::size_t ceil_div(std::size_t numerator, std::size_t denominator) noexcept
{
    return numerator / denominator + static_cast<std::size_t>(numerator % denominator != 0);
}

/// Overflow-checked prepare-time arithmetic, bound to one adapter's name.
///
/// `constexpr`-constructible and stateless but for the name, so an adapter
/// declares `constexpr Checked kChecked{"LMP"};` once and every refusal it
/// raises says which adapter raised it.
class Checked
{
  public:
    explicit constexpr Checked(std::string_view adapter) noexcept : adapter_{adapter} {}

    /// The adapter name this instance was bound to, for callers that compose
    /// their own refusal text out of it.
    [[nodiscard]] constexpr std::string_view name() const noexcept
    {
        return adapter_;
    }

    [[nodiscard]] std::size_t checked_multiply(std::size_t left, std::size_t right) const
    {
        if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
        {
            throw std::overflow_error(std::string{adapter_} + " adapter size overflows size_t");
        }
        return left * right;
    }

    [[nodiscard]] std::size_t checked_add(std::size_t left, std::size_t right) const
    {
        if (right > std::numeric_limits<std::size_t>::max() - left)
        {
            throw std::overflow_error(std::string{adapter_} + " adapter size overflows size_t");
        }
        return left + right;
    }

    [[nodiscard]] std::uint64_t checked_multiply_u64(std::uint64_t left, std::uint64_t right) const
    {
        if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
        {
            throw std::overflow_error(std::string{adapter_} + " adapter duration overflows uint64");
        }
        return left * right;
    }

    [[nodiscard]] std::uint32_t checked_block_samples(std::size_t value) const
    {
        if (value == 0 || value > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::overflow_error(std::string{adapter_} + " output block size exceeds uint32");
        }
        return static_cast<std::uint32_t>(value);
    }

    /// The exact duration of *samples* at *rate*, or a refusal.
    ///
    /// The cancellation is done before the multiply rather than after, so a rate
    /// whose product would overflow but whose reduced form does not is still
    /// answered. A geometry that does not land on whole nanoseconds is refused
    /// rather than rounded: a window that is 1.5 ns short each hop drifts, and the
    /// drift is invisible in a per-block result.
    [[nodiscard]] std::uint64_t exact_duration_ns(std::size_t samples,
                                                  streaming::RationalRate rate) const
    {
        if (samples == 0 || rate.numerator == 0 || rate.denominator == 0)
        {
            throw std::invalid_argument(std::string{adapter_} +
                                        " timing geometry must be positive");
        }
        auto divisor = rate.numerator;
        auto sample_factor = static_cast<std::uint64_t>(samples);
        auto rate_factor = rate.denominator;
        auto nanosecond_factor = std::uint64_t{1'000'000'000};
        for (auto* factor : {&sample_factor, &rate_factor, &nanosecond_factor})
        {
            const auto common = std::gcd(*factor, divisor);
            *factor /= common;
            divisor /= common;
        }
        if (divisor != 1)
        {
            throw std::invalid_argument(std::string{adapter_} +
                                        " window and shift must resolve to integral nanoseconds");
        }
        return checked_multiply_u64(checked_multiply_u64(sample_factor, rate_factor),
                                    nanosecond_factor);
    }

    /// The observation rate a *shift_samples* hop implies, in lowest terms.
    [[nodiscard]] streaming::RationalRate feature_rate(streaming::RationalRate input_rate,
                                                       std::size_t shift_samples) const
    {
        const auto shift = static_cast<std::uint64_t>(shift_samples);
        const auto cancellation = std::gcd(input_rate.numerator, shift);
        auto num = input_rate.numerator / cancellation;
        auto den = checked_multiply_u64(input_rate.denominator, shift / cancellation);
        const auto common = std::gcd(num, den);
        return {num / common, den / common};
    }

  private:
    std::string_view adapter_;
};

/// Convert *ticks* at *rate* into nanoseconds, reporting overflow as `false`.
///
/// `noexcept` and bool-returning because `device_tick_host_time` below calls it
/// from the data plane, where the contract forbids throwing for a result the
/// caller is expected to handle.
[[nodiscard]] inline bool scaled_duration(std::uint64_t ticks, streaming::RationalRate rate,
                                          bool round_up, std::uint64_t& result) noexcept
{
    if (rate.numerator == 0 || rate.denominator == 0)
    {
        return false;
    }
    auto divisor = rate.numerator;
    auto rate_factor = rate.denominator;
    auto nanosecond_factor = std::uint64_t{1'000'000'000};
    auto common = std::gcd(rate_factor, divisor);
    rate_factor /= common;
    divisor /= common;
    common = std::gcd(nanosecond_factor, divisor);
    nanosecond_factor /= common;
    divisor /= common;
    if (rate_factor != 0 &&
        nanosecond_factor > std::numeric_limits<std::uint64_t>::max() / rate_factor)
    {
        return false;
    }
    const auto multiplier = rate_factor * nanosecond_factor;
    const auto quotient = ticks / divisor;
    const auto remainder = ticks % divisor;
    if (quotient != 0 && multiplier > std::numeric_limits<std::uint64_t>::max() / quotient)
    {
        return false;
    }
    const auto whole = quotient * multiplier;
    if (remainder != 0 && multiplier > std::numeric_limits<std::uint64_t>::max() / remainder)
    {
        return false;
    }
    const auto product = remainder * multiplier;
    const auto fractional =
        round_up && product != 0 ? (product - 1) / divisor + 1 : product / divisor;
    if (fractional > std::numeric_limits<std::uint64_t>::max() - whole)
    {
        return false;
    }
    result = whole + fractional;
    return true;
}

/// The host time *block*'s device tick maps to, or `false` if it does not map.
///
/// Refuses rather than extrapolates when the snapshot is unsynchronized, has
/// generation zero, or carries a degenerate rate: a tick converted through a
/// snapshot that was never valid is a timestamp with no provenance, which is
/// worse than no timestamp. The two branches differ because the reference may
/// lie on either side of the block, and the earlier-than-reference branch
/// rounds up so the result never precedes what the reference established.
[[nodiscard]] inline bool device_tick_host_time(const streaming::SignalBlockHeader& block,
                                                streaming::HostTimeNs& result) noexcept
{
    const auto& sync = block.clock_sync;
    if (!streaming::has_flag(sync.flags, streaming::ClockSyncFlags::synchronized) ||
        sync.generation == 0 || sync.device_tick_rate.numerator == 0 ||
        sync.device_tick_rate.denominator == 0)
    {
        return false;
    }
    std::uint64_t duration{};
    if (block.device_tick_start >= sync.device_tick_reference)
    {
        if (!scaled_duration(block.device_tick_start - sync.device_tick_reference,
                             sync.device_tick_rate, false, duration) ||
            duration > std::numeric_limits<std::uint64_t>::max() - sync.host_time_reference_ns)
        {
            return false;
        }
        result = sync.host_time_reference_ns + duration;
        return true;
    }
    if (!scaled_duration(sync.device_tick_reference - block.device_tick_start,
                         sync.device_tick_rate, true, duration) ||
        duration > sync.host_time_reference_ns)
    {
        return false;
    }
    result = sync.host_time_reference_ns - duration;
    return true;
}

/// Whether two unit descriptors name the same unit, id and text alike.
[[nodiscard]] inline bool equal_unit(const streaming::UnitDescriptor& left,
                                     const streaming::UnitDescriptor& right) noexcept
{
    return left.id == right.id && left.symbol == right.symbol &&
           left.description == right.description;
}

/// The output geometry a windowed feature adapter declares at prepare time.
struct FeatureOutput
{
    /// The single signal the adapter emits.
    streaming::SignalSchema signal;
    /// Observations one nominal input block yields; at least one.
    std::size_t nominal_observations;
    /// Observations the largest allowed input block yields.
    std::size_t max_observations;
};

/// Resolve the output signal of an adapter that emits one observation per shift.
///
/// LMP, Hilbert envelope and multitaper bandpower differ in what they compute
/// and in how wide the result is; they do not differ in what kind of signal it
/// is. A windowed feature output is float64 and sample-major, carries no device
/// tick and no physical unit (its unit comes from the feature-set registry),
/// observes on a regular grid at the rate the shift implies, and inherits the
/// input's clock domain, channel set, calibration and reference -- because it
/// is that input, summarised. Stating it once is what keeps a fourth feature
/// adapter from declaring a subtly different kind of output and having the
/// difference show up as a schema mismatch at runtime.
///
/// @param checked           Bound to the calling adapter's name, for error text.
/// @param input             The single input signal the adapter accepts.
/// @param output_signal_id  The signal id the adapter was configured to emit.
/// @param feature_set_id    The descriptor that names and units the features.
/// @param n_output_channels Features per observation: one per input channel for
///                          a per-channel summary, one per name for a bank.
/// @param shift_samples     Input samples between observations; must be > 0.
[[nodiscard]] inline FeatureOutput
feature_output(const Checked& checked, const streaming::SignalSchema& input,
               streaming::SignalId output_signal_id, streaming::FeatureSetId feature_set_id,
               std::size_t n_output_channels, std::size_t shift_samples)
{
    const auto max_observations = ceil_div(input.max_block_samples, shift_samples);
    const auto nominal_observations =
        (std::max<std::size_t>)(1, ceil_div(input.nominal_block_samples, shift_samples));
    return {
        .signal =
            streaming::SignalSchema{
                output_signal_id,
                streaming::SignalDType::float64,
                checked.checked_block_samples(n_output_channels),
                checked.checked_block_samples(nominal_observations),
                checked.checked_block_samples(max_observations),
                checked.feature_rate(input.fs, shift_samples),
                input.clock_domain,
                streaming::SignalLayout::sample_major,
                streaming::DeviceTickTracking::unavailable,
                streaming::PhysicalUnit::unspecified,
                input.channel_set_id,
                input.calibration_id,
                input.reference_id,
                streaming::SignalKind::feature,
                feature_set_id,
                streaming::ObservationTiming::regular,
            },
        .nominal_observations = nominal_observations,
        .max_observations = max_observations,
    };
}

/// Validate the input an in-place sampled adapter accepts, and return its signal.
///
/// FIR, IIR, SOS and common referencing all accept exactly one sampled float64
/// sample-major signal, all need one process output and one frame lease, and all
/// refuse a schema that differs from the one an earlier `prepare` accepted.
/// Those four refusals were four verbatim copies differing only in the adapter
/// name, which :class:`Checked` already carries.
///
/// @param checked          Bound to the calling adapter's name, for error text.
/// @param context          The prepare context to validate.
/// @param prepared_schema  What an earlier `prepare` accepted, or `nullptr` on
///                         the first call.
///
/// Deliberately **not** here: the workspace bound, which every adapter computes
/// differently and passes to `forwarding_contract` itself, and any check that
/// only one adapter makes -- common referencing additionally rejects a
/// zero-channel signal and an out-of-range reference index, and keeps both in
/// its own `prepare`.
[[nodiscard]] inline const streaming::SignalSchema&
sampled_inplace_input(const Checked& checked, const streaming::ProcessorPrepareContext& context,
                      const streaming::StreamSchema* prepared_schema)
{
    const std::string adapter{checked.name()};
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument(adapter +
                                    " adapter requires exactly one signal in the stream schema");
    }
    const auto& signal = signals.front();
    if (signal.kind != streaming::SignalKind::sampled ||
        signal.dtype != streaming::SignalDType::float64 ||
        signal.layout != streaming::SignalLayout::sample_major)
    {
        throw std::invalid_argument(adapter +
                                    " adapter requires a sample-major float64 sampled signal");
    }
    if (context.max_process_outputs < 1 || context.available_frame_pool_leases < 1)
    {
        throw std::invalid_argument(adapter +
                                    " adapter requires one process output and one frame lease");
    }
    if (prepared_schema != nullptr && !prepared_schema->equivalent(context.input_schema))
    {
        throw std::invalid_argument(adapter + " adapter schema cannot change after prepare");
    }
    return signal;
}

/// The contract an adapter declares when it rewrites its input frame in place.
///
/// One input yields exactly one output, that output is the input frame itself
/// (`can_forward_input`, which is what lets `LinearProcessorChain` pass the
/// frame on without copying), the schema is unchanged, and nothing is emitted at
/// flush. *workspace_bytes* is the caller's own bound: what a workspace means
/// differs per algorithm, so it is computed by the adapter and only reported
/// here.
[[nodiscard]] inline streaming::PreparedProcessorContract
forwarding_contract(const streaming::StreamSchema& input_schema, std::size_t workspace_bytes)
{
    return {
        .accepted_input_schema = input_schema.clone(),
        .output_schema = input_schema.clone(),
        .max_process_outputs_per_input = 1,
        .max_flush_outputs = 0,
        .can_forward_input = true,
        .required_resources =
            streaming::ProcessorResourceBounds{
                .workspace_bytes = workspace_bytes,
                .frame_pool_leases = 1,
            },
    };
}

} // namespace neurale::pipeline::adapter_support
