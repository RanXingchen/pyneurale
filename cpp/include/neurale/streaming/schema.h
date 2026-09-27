/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

#include <neurale/streaming/clock.h>

namespace neurale::streaming
{

/// Stable identifier for an immutable stream schema.
using SchemaId = std::uint32_t;

/// Stable identifier for one signal within a session schema.
using SignalId = std::uint32_t;
using ChannelSetId = std::uint64_t;
using CalibrationId = std::uint64_t;
using ReferenceId = std::uint64_t;
using FeatureSetId = std::uint64_t;
using UnitId = std::uint32_t;

enum class PhysicalUnit : std::uint8_t
{
    unspecified,
    volts,
    amperes,
    dimensionless,
};

enum class SignalKind : std::uint8_t
{
    sampled,
    event,
    feature,
    /// Fixed-layout sparse records whose payload capacity is fixed at prepare time.
    spike,
};

/// Timing representation for observations in one signal block.
enum class ObservationTiming : std::uint8_t
{
    not_applicable,
    regular,
    /// One explicitly timestamped observation per block; no fixed cadence.
    irregular,
};

/// Meaning of the timestamp attached to a feature observation.
enum class FeatureTimestampReference : std::uint8_t
{
    window_center,
};

/// Session-level definition of one stable unit identifier.
struct UnitDescriptor
{
    UnitId id{};
    std::string symbol;
    std::string description;
};

/// Session-level metadata shared by every frame in one feature set.
struct FeatureSetDescriptor
{
    FeatureSetId id{};
    std::vector<std::string> feature_names;
    std::vector<UnitId> unit_ids;
    SignalId source_stream_id{};
    std::string source_stream;
    std::string algorithm_name;
    std::string algorithm_version;
    std::uint64_t window_length_ns{};
    std::uint64_t shift_ns{};
    FeatureTimestampReference timestamp_reference{FeatureTimestampReference::window_center};
};

/// Immutable owning unit registry fixed before a session starts.
class UnitRegistry
{
  public:
    UnitRegistry() = default;
    explicit UnitRegistry(std::span<const UnitDescriptor> units);
    ~UnitRegistry();

    UnitRegistry(const UnitRegistry&) = delete;
    UnitRegistry& operator=(const UnitRegistry&) = delete;
    UnitRegistry(UnitRegistry&&) noexcept = default;
    UnitRegistry& operator=(UnitRegistry&&) noexcept = default;

    [[nodiscard]] std::span<const UnitDescriptor> units() const noexcept
    {
        return {units_.get(), n_units_};
    }
    [[nodiscard]] const UnitDescriptor* find(UnitId id) const noexcept;

  private:
    std::unique_ptr<UnitDescriptor[]> units_;
    std::size_t n_units_{};
};

/// Immutable owning feature-set registry fixed before a session starts.
class FeatureSetDescriptorRegistry
{
  public:
    FeatureSetDescriptorRegistry() = default;
    explicit FeatureSetDescriptorRegistry(std::span<const FeatureSetDescriptor> descriptors);
    ~FeatureSetDescriptorRegistry();

    FeatureSetDescriptorRegistry(const FeatureSetDescriptorRegistry&) = delete;
    FeatureSetDescriptorRegistry& operator=(const FeatureSetDescriptorRegistry&) = delete;
    FeatureSetDescriptorRegistry(FeatureSetDescriptorRegistry&&) noexcept = default;
    FeatureSetDescriptorRegistry& operator=(FeatureSetDescriptorRegistry&&) noexcept = default;

    [[nodiscard]] std::span<const FeatureSetDescriptor> descriptors() const noexcept
    {
        return {descriptors_.get(), n_descriptors_};
    }
    [[nodiscard]] const FeatureSetDescriptor* find(FeatureSetId id) const noexcept;

  private:
    std::unique_ptr<FeatureSetDescriptor[]> descriptors_;
    std::size_t n_descriptors_{};
};

/// Scalar representation used by one native signal stream.
enum class SignalDType : std::uint8_t
{
    int16,
    int32,
    float32,
    float64,
};

[[nodiscard]] constexpr std::size_t signal_dtype_size(SignalDType dtype) noexcept
{
    switch (dtype)
    {
    case SignalDType::int16:
        return 2;
    case SignalDType::int32:
    case SignalDType::float32:
        return 4;
    case SignalDType::float64:
        return 8;
    }
    return 0;
}

/// In-block ordering of samples and channels.
enum class SignalLayout : std::uint8_t
{
    sample_major,
    channel_major,
};

/// Device tick semantics available to the continuity checker.
enum class DeviceTickTracking : std::uint8_t
{
    unavailable,
    sample_counter,
};

/// Fixed signal layout established before a native runtime starts.
struct SignalSchema
{
    SignalId id{};
    ClockDomainId clock_domain{};
    std::uint32_t n_channels{};
    std::uint32_t nominal_block_samples{};
    std::uint32_t max_block_samples{};
    RationalRate fs{};
    SignalDType dtype{SignalDType::float32};
    SignalLayout layout{SignalLayout::sample_major};
    DeviceTickTracking device_tick_tracking{DeviceTickTracking::unavailable};
    SignalKind kind{SignalKind::sampled};
    PhysicalUnit physical_unit{PhysicalUnit::unspecified};
    ChannelSetId channel_set_id{};
    CalibrationId calibration_id{};
    ReferenceId reference_id{};
    FeatureSetId feature_set_id{};
    ObservationTiming observation_timing{ObservationTiming::not_applicable};
    /// Exact payload bytes for one fixed-capacity sparse block. Zero for dense signals.
    std::uint64_t fixed_block_bytes{};
    std::uint64_t max_block_bytes{};
    std::vector<std::string> channel_names;
    /// Per-channel impedance magnitude in ohms. Empty when the source did not measure it.
    std::vector<double> channel_impedances_ohm;

    SignalSchema() = default;

    SignalSchema(SignalId signal_id, SignalDType signal_dtype, std::uint32_t channels,
                 std::uint32_t nominal_samples, std::uint32_t max_samples, RationalRate rate,
                 ClockDomainId signal_clock_domain,
                 SignalLayout signal_layout = SignalLayout::sample_major,
                 DeviceTickTracking tick_tracking = DeviceTickTracking::unavailable,
                 PhysicalUnit unit = PhysicalUnit::unspecified, ChannelSetId channels_id = 0,
                 CalibrationId signal_calibration_id = 0, ReferenceId signal_reference_id = 0,
                 SignalKind signal_kind = SignalKind::sampled,
                 FeatureSetId signal_feature_set_id = 0,
                 ObservationTiming signal_observation_timing = ObservationTiming::not_applicable,
                 std::uint64_t signal_fixed_block_bytes = 0,
                 std::vector<std::string> signal_channel_names = {},
                 std::vector<double> signal_channel_impedances_ohm = {});
};

/// Runtime payload-and-timing equivalence of one schema element.
///
/// These are the single definition of "the same schema element" in the
/// codebase: `StreamSchema::equivalent()` is built out of them, and so is any
/// out-of-tree check that has to compare a schema it decoded itself against the
/// one a runtime was configured with. Descriptive channel names and impedance
/// measurements are intentionally outside runtime equivalence; processors that
/// consume either field compare it explicitly during prepare.
[[nodiscard]] bool equivalent(const SignalSchema& left, const SignalSchema& right) noexcept;
[[nodiscard]] bool equivalent(const FeatureSetDescriptor& left,
                              const FeatureSetDescriptor& right) noexcept;
[[nodiscard]] bool equivalent(const UnitDescriptor& left, const UnitDescriptor& right) noexcept;

/// Immutable owning schema fixed before the streaming session starts.
class StreamSchema
{
  public:
    StreamSchema(SchemaId schema_id, std::span<const SignalSchema> signals);
    StreamSchema(SchemaId schema_id, std::span<const SignalSchema> signals,
                 std::span<const FeatureSetDescriptor> feature_sets,
                 std::span<const UnitDescriptor> units);
    ~StreamSchema();

    StreamSchema(const StreamSchema&) = delete;
    StreamSchema& operator=(const StreamSchema&) = delete;
    StreamSchema(StreamSchema&&) noexcept = default;
    StreamSchema& operator=(StreamSchema&&) noexcept = default;

    [[nodiscard]] SchemaId id() const noexcept
    {
        return id_;
    }
    [[nodiscard]] std::span<const SignalSchema> signals() const noexcept
    {
        return {signals_.get(), n_signals_};
    }
    [[nodiscard]] const FeatureSetDescriptorRegistry& feature_sets() const noexcept
    {
        return feature_sets_;
    }
    [[nodiscard]] const UnitRegistry& units() const noexcept
    {
        return units_;
    }
    [[nodiscard]] StreamSchema clone() const;
    [[nodiscard]] bool equivalent(const StreamSchema& other) const noexcept;

  private:
    SchemaId id_{};
    std::unique_ptr<SignalSchema[]> signals_;
    std::size_t n_signals_{};
    FeatureSetDescriptorRegistry feature_sets_;
    UnitRegistry units_;
};

static_assert(!std::is_copy_constructible_v<StreamSchema>);
static_assert(std::is_nothrow_move_constructible_v<StreamSchema>);

} // namespace neurale::streaming
