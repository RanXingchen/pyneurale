/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/identity.h>

/**
 * @file
 * @brief The stable sampler, schedule identity, and replay authority.
 *
 * There is no global random state anywhere in the experiment contract: no
 * `std::random_device`, no file-scope generator, no `srand`, and no ambient
 * host RNG. Every randomized experiment value is a pure function of the
 * determining tuple
 *
 * ```text
 * (configuration, seed, trial index, sampler version)
 * ```
 *
 * and is therefore regenerated bit-exactly on any platform and toolchain. The
 * sampler is specified below by its transformation rather than named as a
 * library facility, because `<random>` distributions are not portable across
 * standard-library implementations.
 */
namespace neurale::experiments
{

/// Seed of one experiment schedule.
using ScheduleSeed = std::uint64_t;

/// Version of the sampler that produced a value.
///
/// It changes whenever either the generator transformation or the bounded
/// mapping changes. A determinism claim that is not conditioned on this value
/// is not made anywhere in the contract.
using SamplerVersion = std::uint32_t;

/// Purpose tag separating independent draw sequences under one seed.
///
/// A paradigm assigns its own tags -- one for trial durations, one for target
/// selection, and so on -- so that adding a draw to one sequence does not shift
/// the values of another.
using DrawStream = std::uint32_t;

/// The sampler specified in this header.
///
/// Version 1 names exactly this transformation, and nothing else is version 1:
///
/// ```text
/// sampler_mix64
///   + a value addressed by DrawKey rather than advanced
///   + bounded masked rejection onto the range
///   + at most kMaxRejectionDraws attempts
///   + ContractStatus::sampling_exhausted on exhaustion, with no fallback mapping
/// ```
///
/// Any change to any line of that is a different sampler and takes a different
/// version, because a determinism claim is only ever made about a named one.
inline constexpr SamplerVersion kSamplerVersion1 = 1;

/// Sampler version new schedules are created with.
inline constexpr SamplerVersion kCurrentSamplerVersion = kSamplerVersion1;

/// Rejection attempts a bounded draw makes before it gives up.
///
/// Each attempt succeeds with probability above one half, so exhausting all of
/// them has probability below 2^-64 for any range. The cap exists so the sampler
/// is bounded rather than merely expected-to-terminate.
inline constexpr std::uint32_t kMaxRejectionDraws = 64;

/// Counter slots one logical draw reserves.
///
/// Exactly one slot per attempt the rejection loop may make, so the attempts of
/// one logical draw never collide with the next draw's, and a caller's draw
/// ordinal advances by exactly one per logical draw whatever the loop did. That
/// is what makes a schedule resumable from a draw ordinal alone.
inline constexpr std::uint64_t kDrawSlotStride = kMaxRejectionDraws;

/// Exact coordinates of one randomized value.
///
/// The sampler is stateless: it is addressed, not advanced. Two callers holding
/// the same key obtain the same value in any order, and a replay obtains it
/// without having produced the draws before it.
///
/// Resetting a schedule is therefore not an operation on the sampler, because
/// the sampler holds nothing to reset. It is the caller returning its draw
/// ordinals to where they started -- an OrdinalCounter reset for the trial
/// index, and a `draw` cursor back to zero -- after which the same keys yield
/// the same values they yielded before. A schedule cannot be "re-randomized"
/// without changing the seed, and changing the seed makes it a different
/// schedule that a recorded session will not match.
struct DrawKey
{
    /// Schedule seed.
    ScheduleSeed seed{};
    /// Purpose tag of the draw sequence.
    DrawStream stream{};
    /// Trial this draw belongs to.
    TrialOrdinal trial_idx{};
    /// Draw ordinal within `(stream, trial_index)`.
    std::uint32_t draw{};
};

/// The sampler's mixing transformation, written out rather than named.
///
/// Three multiply-xorshift rounds over 64-bit unsigned arithmetic. Every
/// operation is exact on any conforming implementation, which is what makes the
/// result bit-identical across platforms.
[[nodiscard]] constexpr std::uint64_t sampler_mix64(std::uint64_t value) noexcept
{
    std::uint64_t z = value + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

/// Uniform 64-bit value at @p key, using counter slot @p slot of its stride.
[[nodiscard]] constexpr std::uint64_t sample_slot_bits(const DrawKey& key,
                                                       std::uint64_t slot) noexcept
{
    std::uint64_t state = sampler_mix64(key.seed ^ 0xA0761D6478BD642FULL);
    state = sampler_mix64(state + key.trial_idx);
    state = sampler_mix64(state ^ (static_cast<std::uint64_t>(key.stream) * 0x9E3779B97F4A7C15ULL));
    return sampler_mix64(state + static_cast<std::uint64_t>(key.draw) * kDrawSlotStride + slot);
}

/// Uniform 64-bit value at @p key.
[[nodiscard]] constexpr std::uint64_t sample_bits(const DrawKey& key) noexcept
{
    return sample_slot_bits(key, 0);
}

namespace detail
{

/// Masked-rejection mapping of an arbitrary bit source onto `[low, high]`.
///
/// Factored out of sample_inclusive() so that the exhaustion branch is
/// exercisable in a test. Whether some DrawKey reaches it is unknown and not
/// worth knowing: each attempt is accepted with probability above one half, so
/// searching the key space for one is not a test strategy. A test supplies a
/// source that rejects deliberately instead, and a branch no test can enter is
/// a branch no reviewer can check.
///
/// @tparam SlotBits Callable from a slot index to a uniform 64-bit value.
/// @param bits Bit source addressed by slot index.
/// @param low Smallest admissible value.
/// @param high Largest admissible value.
/// @param value Receives the sampled value on ContractStatus::ok, and is left
///              unmodified otherwise.
/// @return As sample_inclusive().
template <typename SlotBits>
[[nodiscard]] constexpr ContractStatus sample_inclusive_from(SlotBits bits, std::uint64_t low,
                                                             std::uint64_t high,
                                                             std::uint64_t& value) noexcept
{
    if (high < low)
        return ContractStatus::range_empty;
    const std::uint64_t span = high - low;
    if (span == ~std::uint64_t{0})
    {
        // The mask below would be all ones, so every draw is in range and the
        // rejection loop has nothing to reject.
        value = bits(0);
        return ContractStatus::ok;
    }
    std::uint64_t mask = span;
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;
    for (std::uint32_t attempt = 0; attempt < kMaxRejectionDraws; ++attempt)
    {
        const std::uint64_t candidate = bits(attempt) & mask;
        if (candidate <= span)
        {
            value = low + candidate;
            return ContractStatus::ok;
        }
    }
    return ContractStatus::sampling_exhausted;
}

} // namespace detail

/// Uniform integer in the closed range `[low, high]`.
///
/// Both endpoints are included. Bias is removed by masked rejection: the draw
/// is reduced to the smallest `2^k - 1` mask covering the span and retried
/// until it lands in range, for at most ::kMaxRejectionDraws attempts.
///
/// There is no fallback mapping. Every value this function returns is exactly
/// uniform on `[low, high]`, and exhausting the attempts is reported as
/// ContractStatus::sampling_exhausted instead of being folded into range --
/// reducing a further draw modulo the span would be uniform only when the span
/// divides 2^64, so a fallback would make the sampler biased on exactly the
/// path that is too rare for any test to catch it on.
///
/// @param key Draw coordinates.
/// @param low Smallest admissible value.
/// @param high Largest admissible value.
/// @param value Receives the sampled value when the result is ContractStatus::ok,
///              and is left unmodified otherwise.
/// @return ContractStatus::ok, ContractStatus::range_empty when `high < low`, or
///         ContractStatus::sampling_exhausted when every attempt was rejected.
[[nodiscard]] constexpr ContractStatus sample_inclusive(const DrawKey& key, std::uint64_t low,
                                                        std::uint64_t high,
                                                        std::uint64_t& value) noexcept
{
    return detail::sample_inclusive_from([&key](std::uint32_t slot)
                                         { return sample_slot_bits(key, slot); }, low, high, value);
}

/// Uniform integer in the open range `(low, high)`.
///
/// Both endpoints are excluded, so the result is strictly greater than @p low
/// and strictly less than @p high. This is the mapping the Speech cue paradigm
/// samples its per-trial durations with: a duration is never zero and never
/// equals its configured bound.
///
/// @param key Draw coordinates.
/// @param low Excluded lower bound.
/// @param high Excluded upper bound.
/// @param value Receives the sampled value when the result is ContractStatus::ok.
/// @return ContractStatus::ok, ContractStatus::range_empty when the open range
///         contains no integer, that is when `high < low + 2`, or
///         ContractStatus::sampling_exhausted as for sample_inclusive().
[[nodiscard]] constexpr ContractStatus sample_exclusive(const DrawKey& key, std::uint64_t low,
                                                        std::uint64_t high,
                                                        std::uint64_t& value) noexcept
{
    if (low == ~std::uint64_t{0} || high <= low + 1)
        return ContractStatus::range_empty;
    return sample_inclusive(key, low + 1, high - 1, value);
}

/// Uniform index in `[0, count)`.
///
/// @param key Draw coordinates.
/// @param count Number of admissible indices.
/// @param value Receives the index when the result is ContractStatus::ok.
/// @return ContractStatus::ok, ContractStatus::range_empty when @p count is
///         zero, or ContractStatus::sampling_exhausted as for sample_inclusive().
[[nodiscard]] constexpr ContractStatus sample_index(const DrawKey& key, std::uint64_t count,
                                                    std::uint64_t& value) noexcept
{
    if (count == 0)
        return ContractStatus::range_empty;
    return sample_inclusive(key, 0, count - 1, value);
}

/// Deterministic digest builder for configuration and catalog fingerprints.
///
/// Absorption is defined on integers and on individual bytes, so a fingerprint
/// does not depend on the host's endianness or on how a struct is padded. It
/// owns no storage and allocates nothing.
class FingerprintAccumulator
{
  public:
    /// Construct an accumulator at the default origin.
    constexpr FingerprintAccumulator() noexcept = default;

    /// Construct an accumulator at an explicit origin.
    constexpr explicit FingerprintAccumulator(std::uint64_t origin) noexcept : state_(origin) {}

    /// Absorb one integer.
    constexpr void absorb(std::uint64_t value) noexcept
    {
        state_ = sampler_mix64(state_ + sampler_mix64(value));
    }

    /// Absorb a byte sequence, one byte at a time.
    constexpr void absorb_bytes(std::span<const std::byte> bytes) noexcept
    {
        absorb(static_cast<std::uint64_t>(bytes.size()));
        for (const std::byte byte : bytes)
            absorb(static_cast<std::uint64_t>(std::to_integer<unsigned char>(byte)));
    }

    /// Digest of everything absorbed so far.
    [[nodiscard]] constexpr std::uint64_t value() const noexcept
    {
        return state_;
    }

  private:
    std::uint64_t state_{0x243F6A8885A308D3ULL};
};

/// Everything a reader needs in order to regenerate a session's random values.
///
/// Persisted once per session. The two fingerprints are produced by the owning
/// paradigm with a FingerprintAccumulator; the contract stores them and never
/// interprets them.
struct ScheduleIdentity
{
    /// Schedule seed.
    ScheduleSeed seed{};
    /// Sampler that produced the session's values.
    SamplerVersion sampler_version{kCurrentSamplerVersion};
    /// Digest of the paradigm's immutable configuration.
    std::uint64_t configuration_fingerprint{};
    /// Digest of the paradigm's immutable stimulus catalog, or zero when it has none.
    std::uint64_t catalog_fingerprint{};
};

/// Whether this build can regenerate values produced by @p version.
[[nodiscard]] constexpr bool sampler_version_supported(SamplerVersion version) noexcept
{
    return version == kSamplerVersion1;
}

/// Which of the two recorded artefacts a replay must treat as authoritative.
///
/// There is no third case, and neither is a matter of preference at replay
/// time: a session regenerated under a sampler version other than the one it
/// recorded would be a different experiment claiming to reproduce the recorded
/// one.
enum class ReplayAuthority : std::uint8_t
{
    /// Regenerate from the determining tuple. The recorded realized values are
    /// verification provenance: a reader checks the regeneration against them.
    regenerate = 0,
    /// The persisted realized schedule is the fallback authority, and resampling
    /// under any other sampler version is forbidden.
    recorded_schedule,
};

/// Replay authority for a session that recorded @p version.
[[nodiscard]] constexpr ReplayAuthority replay_authority(SamplerVersion version) noexcept
{
    return sampler_version_supported(version) ? ReplayAuthority::regenerate
                                              : ReplayAuthority::recorded_schedule;
}

/// Single value identifying a schedule, suitable for persistence and comparison.
[[nodiscard]] constexpr std::uint64_t
schedule_fingerprint(const ScheduleIdentity& identity) noexcept
{
    FingerprintAccumulator accumulator;
    accumulator.absorb(identity.seed);
    accumulator.absorb(identity.sampler_version);
    accumulator.absorb(identity.configuration_fingerprint);
    accumulator.absorb(identity.catalog_fingerprint);
    return accumulator.value();
}

/// Validate a schedule identity.
///
/// An unsupported sampler version is *not* rejected: a session recorded under
/// one is replayed through ReplayAuthority::recorded_schedule. Only an absent
/// version is invalid, because it names no sampler at all.
[[nodiscard]] constexpr ContractStatus validate(const ScheduleIdentity& identity) noexcept
{
    return identity.sampler_version == 0 ? ContractStatus::identity_missing : ContractStatus::ok;
}

/// One realized randomized value, emitted as replay provenance.
struct ScheduleDraw
{
    /// Experiment time the value was realized.
    ExperimentTimeNs time_ns{};
    /// Trial the value belongs to.
    TrialIdentity trial{};
    /// Purpose tag of the draw sequence.
    DrawStream stream{};
    /// Draw ordinal within `(stream, trial)`.
    std::uint32_t draw{};
    /// The realized value.
    std::uint64_t value{};
    /// Sampler that produced it.
    SamplerVersion sampler_version{kCurrentSamplerVersion};
};

/// Validate a realized-draw record.
[[nodiscard]] constexpr ContractStatus validate(const ScheduleDraw& draw) noexcept
{
    return draw.sampler_version == 0 ? ContractStatus::identity_missing : ContractStatus::ok;
}

} // namespace neurale::experiments
