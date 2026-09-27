/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/streaming/schema.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace neurale::streaming
{
namespace
{

[[nodiscard]] std::uint64_t dtype_size(SignalDType dtype)
{
    const auto size = signal_dtype_size(dtype);
    if (size == 0)
    {
        throw std::invalid_argument("unsupported signal dtype");
    }
    return size;
}

void require_supported_layout(SignalLayout layout)
{
    switch (layout)
    {
    case SignalLayout::sample_major:
    case SignalLayout::channel_major:
        return;
    }
    throw std::invalid_argument("unsupported signal layout");
}

void require_supported_tick_tracking(DeviceTickTracking tracking)
{
    switch (tracking)
    {
    case DeviceTickTracking::unavailable:
    case DeviceTickTracking::sample_counter:
        return;
    }
    throw std::invalid_argument("unsupported device tick tracking mode");
}

[[nodiscard]] std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right)
{
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left)
    {
        throw std::overflow_error("signal block byte count overflows uint64");
    }
    return left * right;
}

[[nodiscard]] std::uint64_t block_byte_count(const SignalSchema& signal)
{
    const auto bytes_per_sample = dtype_size(signal.dtype);
    require_supported_layout(signal.layout);
    require_supported_tick_tracking(signal.device_tick_tracking);
    if (signal.n_channels == 0)
    {
        throw std::invalid_argument("signal channel count must be positive");
    }
    if (signal.nominal_block_samples == 0 || signal.max_block_samples == 0 ||
        signal.nominal_block_samples > signal.max_block_samples)
    {
        throw std::invalid_argument("invalid signal block sample limits");
    }
    if (!signal.channel_names.empty())
    {
        if (signal.channel_names.size() != signal.n_channels)
        {
            throw std::invalid_argument("signal channel names must match the channel count");
        }
        for (std::size_t i = 0; i < signal.channel_names.size(); ++i)
        {
            if (signal.channel_names[i].empty())
            {
                throw std::invalid_argument("signal channel names must be nonempty");
            }
            if (std::find(signal.channel_names.begin(), signal.channel_names.begin() + i,
                          signal.channel_names[i]) != signal.channel_names.begin() + i)
            {
                throw std::invalid_argument("signal channel names must be unique");
            }
        }
    }
    if (!signal.channel_impedances_ohm.empty())
    {
        if (signal.channel_impedances_ohm.size() != signal.n_channels)
        {
            throw std::invalid_argument("signal channel impedances must match the channel count");
        }
        for (const auto impedance : signal.channel_impedances_ohm)
        {
            if (!std::isfinite(impedance) || impedance < 0.0)
            {
                throw std::invalid_argument(
                    "signal channel impedances must be finite and non-negative");
            }
        }
    }
    if (signal.kind == SignalKind::sampled)
    {
        if (signal.fs.numerator == 0 || signal.fs.denominator == 0)
        {
            throw std::invalid_argument("signal sample rate must be positive");
        }
        if (signal.feature_set_id != 0 ||
            signal.observation_timing != ObservationTiming::not_applicable)
        {
            throw std::invalid_argument("sampled signals cannot reference feature metadata");
        }
    }
    else if (!signal.channel_impedances_ohm.empty())
    {
        throw std::invalid_argument("only sampled signals can declare channel impedances");
    }
    else if (signal.kind == SignalKind::event)
    {
        if (signal.fs.numerator != 0 || signal.fs.denominator != 1 ||
            signal.nominal_block_samples != 1 || signal.max_block_samples != 1 ||
            signal.feature_set_id != 0 ||
            signal.observation_timing != ObservationTiming::not_applicable)
        {
            throw std::invalid_argument("event signals require zero rate and one event per block");
        }
    }
    else if (signal.kind == SignalKind::feature)
    {
        if (signal.feature_set_id == 0)
        {
            throw std::invalid_argument("feature signals require a feature-set descriptor id");
        }
        if (signal.observation_timing != ObservationTiming::regular &&
            signal.observation_timing != ObservationTiming::irregular)
        {
            throw std::invalid_argument(
                "feature signals require regular or irregular observation timing");
        }
        if (signal.observation_timing == ObservationTiming::regular &&
            (signal.fs.numerator == 0 || signal.fs.denominator == 0))
        {
            throw std::invalid_argument("regular feature observation rate must be positive");
        }
        if (signal.observation_timing == ObservationTiming::irregular &&
            (signal.fs.numerator != 0 || signal.fs.denominator != 1 ||
             signal.nominal_block_samples != 1 || signal.max_block_samples != 1))
            throw std::invalid_argument(
                "irregular features require zero rate and one observation per block");
        if (signal.layout != SignalLayout::sample_major)
        {
            throw std::invalid_argument("feature payloads must use observation-major layout");
        }
        if (signal.physical_unit != PhysicalUnit::unspecified)
        {
            throw std::invalid_argument("feature units must be declared by the unit registry");
        }
    }
    else if (signal.kind == SignalKind::spike)
    {
        if (signal.fs.numerator != 0 || signal.fs.denominator != 1 || signal.feature_set_id != 0 ||
            signal.observation_timing != ObservationTiming::not_applicable ||
            signal.fixed_block_bytes == 0)
        {
            throw std::invalid_argument(
                "spike signals require zero rate and a fixed sparse-block payload size");
        }
        return signal.fixed_block_bytes;
    }
    else
    {
        throw std::invalid_argument("unsupported signal kind");
    }

    const auto n_scalars = checked_multiply(signal.n_channels, signal.max_block_samples);
    const auto bytes = checked_multiply(n_scalars, bytes_per_sample);
    if (bytes > std::numeric_limits<std::size_t>::max())
    {
        throw std::overflow_error("signal block byte count overflows size_t");
    }
    return bytes;
}

void validate_feature_set(const FeatureSetDescriptor& descriptor)
{
    if (descriptor.id == 0)
    {
        throw std::invalid_argument("feature-set descriptor id must be nonzero");
    }
    if (descriptor.feature_names.empty())
    {
        throw std::invalid_argument("feature set must contain feature names");
    }
    if (descriptor.feature_names.size() != descriptor.unit_ids.size())
    {
        throw std::invalid_argument("feature names and unit identifiers must have equal length");
    }
    for (std::size_t i = 0; i < descriptor.feature_names.size(); ++i)
    {
        if (descriptor.feature_names[i].empty())
        {
            throw std::invalid_argument("feature names must be nonempty");
        }
        if (descriptor.unit_ids[i] == 0)
        {
            throw std::invalid_argument("feature unit identifiers must be nonzero");
        }
        for (std::size_t previous = 0; previous < i; ++previous)
        {
            if (descriptor.feature_names[previous] == descriptor.feature_names[i])
            {
                throw std::invalid_argument("feature names must be unique");
            }
        }
    }
    if (descriptor.source_stream_id == 0 || descriptor.source_stream.empty())
    {
        throw std::invalid_argument("feature source stream id and name must be provided");
    }
    if (descriptor.window_length_ns == 0)
    {
        throw std::invalid_argument("feature window length must be positive");
    }
    if (descriptor.timestamp_reference != FeatureTimestampReference::window_center)
    {
        throw std::invalid_argument(
            "window-center is the only supported feature timestamp reference");
    }
}

void validate_regular_shift(const SignalSchema& signal, const FeatureSetDescriptor& descriptor)
{
    if (signal.observation_timing == ObservationTiming::irregular)
    {
        if (descriptor.shift_ns != 0)
            throw std::invalid_argument("irregular features must not declare a fixed shift");
        return;
    }
    const auto left = checked_multiply(signal.fs.numerator, descriptor.shift_ns);
    const auto right = checked_multiply(signal.fs.denominator, 1'000'000'000ULL);
    if (left != right)
    {
        throw std::invalid_argument("feature observation rate must equal the descriptor shift");
    }
}

} // namespace

bool equivalent(const UnitDescriptor& left, const UnitDescriptor& right) noexcept
{
    return left.id == right.id && left.symbol == right.symbol &&
           left.description == right.description;
}

bool equivalent(const FeatureSetDescriptor& left, const FeatureSetDescriptor& right) noexcept
{
    return left.id == right.id && left.feature_names == right.feature_names &&
           left.unit_ids == right.unit_ids && left.source_stream_id == right.source_stream_id &&
           left.source_stream == right.source_stream &&
           left.algorithm_name == right.algorithm_name &&
           left.algorithm_version == right.algorithm_version &&
           left.window_length_ns == right.window_length_ns && left.shift_ns == right.shift_ns &&
           left.timestamp_reference == right.timestamp_reference;
}

bool equivalent(const SignalSchema& left, const SignalSchema& right) noexcept
{
    return left.id == right.id && left.clock_domain == right.clock_domain &&
           left.n_channels == right.n_channels &&
           left.nominal_block_samples == right.nominal_block_samples &&
           left.max_block_samples == right.max_block_samples &&
           left.fs.numerator == right.fs.numerator && left.fs.denominator == right.fs.denominator &&
           left.dtype == right.dtype && left.layout == right.layout &&
           left.device_tick_tracking == right.device_tick_tracking && left.kind == right.kind &&
           left.physical_unit == right.physical_unit &&
           left.channel_set_id == right.channel_set_id &&
           left.calibration_id == right.calibration_id && left.reference_id == right.reference_id &&
           left.feature_set_id == right.feature_set_id &&
           left.observation_timing == right.observation_timing &&
           left.fixed_block_bytes == right.fixed_block_bytes &&
           left.max_block_bytes == right.max_block_bytes;
}

UnitRegistry::UnitRegistry(std::span<const UnitDescriptor> units) : n_units_(units.size())
{
    for (std::size_t i = 0; i < units.size(); ++i)
    {
        if (units[i].id == 0 || units[i].symbol.empty())
        {
            throw std::invalid_argument("unit id and canonical symbol must be provided");
        }
        for (std::size_t previous = 0; previous < i; ++previous)
        {
            if (units[previous].id == units[i].id)
            {
                throw std::invalid_argument("duplicate unit id");
            }
            if (units[previous].symbol == units[i].symbol)
            {
                throw std::invalid_argument("duplicate unit symbol");
            }
        }
    }
    units_ = std::make_unique<UnitDescriptor[]>(n_units_);
    std::copy(units.begin(), units.end(), units_.get());
}

UnitRegistry::~UnitRegistry() = default;

const UnitDescriptor* UnitRegistry::find(UnitId id) const noexcept
{
    for (const auto& unit : units())
    {
        if (unit.id == id)
        {
            return &unit;
        }
    }
    return nullptr;
}

FeatureSetDescriptorRegistry::FeatureSetDescriptorRegistry(
    std::span<const FeatureSetDescriptor> descriptors)
    : n_descriptors_(descriptors.size())
{
    for (std::size_t i = 0; i < descriptors.size(); ++i)
    {
        validate_feature_set(descriptors[i]);
        for (std::size_t previous = 0; previous < i; ++previous)
        {
            if (descriptors[previous].id == descriptors[i].id)
            {
                throw std::invalid_argument("duplicate feature-set descriptor id");
            }
        }
    }
    descriptors_ = std::make_unique<FeatureSetDescriptor[]>(n_descriptors_);
    std::copy(descriptors.begin(), descriptors.end(), descriptors_.get());
}

FeatureSetDescriptorRegistry::~FeatureSetDescriptorRegistry() = default;

const FeatureSetDescriptor* FeatureSetDescriptorRegistry::find(FeatureSetId id) const noexcept
{
    for (const auto& descriptor : descriptors())
    {
        if (descriptor.id == id)
        {
            return &descriptor;
        }
    }
    return nullptr;
}

SignalSchema::SignalSchema(SignalId signal_id, SignalDType signal_dtype, std::uint32_t channels,
                           std::uint32_t nominal_samples, std::uint32_t max_samples,
                           RationalRate rate, ClockDomainId signal_clock_domain,
                           SignalLayout signal_layout, DeviceTickTracking tick_tracking,
                           PhysicalUnit unit, ChannelSetId channels_id,
                           CalibrationId signal_calibration_id, ReferenceId signal_reference_id,
                           SignalKind signal_kind, FeatureSetId signal_feature_set_id,
                           ObservationTiming signal_observation_timing,
                           std::uint64_t signal_fixed_block_bytes,
                           std::vector<std::string> signal_channel_names,
                           std::vector<double> signal_channel_impedances_ohm)
    : id(signal_id), clock_domain(signal_clock_domain), n_channels(channels),
      nominal_block_samples(nominal_samples), max_block_samples(max_samples), fs(rate),
      dtype(signal_dtype), layout(signal_layout), device_tick_tracking(tick_tracking),
      kind(signal_kind), physical_unit(unit), channel_set_id(channels_id),
      calibration_id(signal_calibration_id), reference_id(signal_reference_id),
      feature_set_id(signal_feature_set_id), observation_timing(signal_observation_timing),
      fixed_block_bytes(signal_fixed_block_bytes), channel_names(std::move(signal_channel_names)),
      channel_impedances_ohm(std::move(signal_channel_impedances_ohm))
{
    max_block_bytes = block_byte_count(*this);
}

StreamSchema::StreamSchema(SchemaId schema_id, std::span<const SignalSchema> signals)
    : StreamSchema(schema_id, signals, {}, {})
{
}

StreamSchema::StreamSchema(SchemaId schema_id, std::span<const SignalSchema> signals,
                           std::span<const FeatureSetDescriptor> feature_sets,
                           std::span<const UnitDescriptor> units)
    : id_(schema_id), n_signals_(signals.size()), feature_sets_(feature_sets), units_(units)
{
    if (signals.empty())
    {
        throw std::invalid_argument("stream schema must contain a signal");
    }
    for (std::size_t i = 0; i < signals.size(); ++i)
    {
        if (block_byte_count(signals[i]) != signals[i].max_block_bytes)
        {
            throw std::invalid_argument("stale signal block byte count");
        }
        if (signals[i].kind == SignalKind::feature)
        {
            const auto* descriptor = feature_sets_.find(signals[i].feature_set_id);
            if (descriptor == nullptr)
            {
                throw std::invalid_argument("feature signal references a missing descriptor id");
            }
            if (descriptor->feature_names.size() != signals[i].n_channels)
            {
                throw std::invalid_argument("feature count does not match its descriptor");
            }
            for (const auto unit_id : descriptor->unit_ids)
            {
                if (units_.find(unit_id) == nullptr)
                {
                    throw std::invalid_argument("feature descriptor references a missing unit id");
                }
            }
            validate_regular_shift(signals[i], *descriptor);
        }
        for (std::size_t previous = 0; previous < i; ++previous)
        {
            if (signals[previous].id == signals[i].id)
            {
                throw std::invalid_argument("duplicate signal id");
            }
        }
    }
    signals_ = std::make_unique_for_overwrite<SignalSchema[]>(n_signals_);
    std::copy(signals.begin(), signals.end(), signals_.get());
}

StreamSchema::~StreamSchema() = default;

StreamSchema StreamSchema::clone() const
{
    return StreamSchema{id_, signals(), feature_sets_.descriptors(), units_.units()};
}

bool StreamSchema::equivalent(const StreamSchema& other) const noexcept
{
    if (id_ != other.id_ || n_signals_ != other.n_signals_)
    {
        return false;
    }
    for (std::size_t i = 0; i < n_signals_; ++i)
    {
        if (!streaming::equivalent(signals_[i], other.signals_[i]))
        {
            return false;
        }
    }
    const auto left_features = feature_sets_.descriptors();
    const auto right_features = other.feature_sets_.descriptors();
    if (left_features.size() != right_features.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < left_features.size(); ++i)
    {
        if (!streaming::equivalent(left_features[i], right_features[i]))
        {
            return false;
        }
    }
    const auto left_units = units_.units();
    const auto right_units = other.units_.units();
    if (left_units.size() != right_units.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < left_units.size(); ++i)
    {
        if (!streaming::equivalent(left_units[i], right_units[i]))
        {
            return false;
        }
    }
    return true;
}

} // namespace neurale::streaming
