/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The image's spellings of the schema's enumerated values, and nothing else.
///
/// The replay image stores a dtype, a layout, a signal kind and the rest as
/// interned strings, spelled the way the recording plan spells them. Reading
/// one back is a table lookup, and the table has to be in exactly one place:
/// `NativeReplaySource::check_schema()` reads it to compare an image against a
/// runtime schema, and the conformance harness reads it to rebuild the
/// recorded schema from an image alone. A second copy would let one of them accept a
/// spelling the other refuses, and both would look right on their own.
///
/// An unknown spelling is never guessed at. It does not mean the two schemas
/// differ; it means this build cannot compare them, which is a different answer
/// and gets a different status.

#include <string_view>

#include <neurale/streaming/schema.h>

namespace neurale::recording::names
{

[[nodiscard]] inline bool dtype_from(std::string_view name, streaming::SignalDType& value) noexcept
{
    if (name == "INT16")
    {
        value = streaming::SignalDType::int16;
        return true;
    }
    if (name == "INT32")
    {
        value = streaming::SignalDType::int32;
        return true;
    }
    if (name == "FLOAT32")
    {
        value = streaming::SignalDType::float32;
        return true;
    }
    if (name == "FLOAT64")
    {
        value = streaming::SignalDType::float64;
        return true;
    }
    return false;
}

[[nodiscard]] inline bool layout_from(std::string_view name,
                                      streaming::SignalLayout& value) noexcept
{
    if (name == "SAMPLE_MAJOR")
    {
        value = streaming::SignalLayout::sample_major;
        return true;
    }
    if (name == "CHANNEL_MAJOR")
    {
        value = streaming::SignalLayout::channel_major;
        return true;
    }
    return false;
}

[[nodiscard]] inline bool tick_tracking_from(std::string_view name,
                                             streaming::DeviceTickTracking& value) noexcept
{
    if (name == "UNAVAILABLE")
    {
        value = streaming::DeviceTickTracking::unavailable;
        return true;
    }
    if (name == "SAMPLE_COUNTER")
    {
        value = streaming::DeviceTickTracking::sample_counter;
        return true;
    }
    return false;
}

[[nodiscard]] inline bool signal_kind_from(std::string_view name,
                                           streaming::SignalKind& value) noexcept
{
    if (name == "SAMPLED")
    {
        value = streaming::SignalKind::sampled;
        return true;
    }
    if (name == "EVENT")
    {
        value = streaming::SignalKind::event;
        return true;
    }
    if (name == "FEATURE")
    {
        value = streaming::SignalKind::feature;
        return true;
    }
    if (name == "SPIKE")
    {
        value = streaming::SignalKind::spike;
        return true;
    }
    return false;
}

[[nodiscard]] inline bool physical_unit_from(std::string_view name,
                                             streaming::PhysicalUnit& value) noexcept
{
    if (name == "UNSPECIFIED")
    {
        value = streaming::PhysicalUnit::unspecified;
        return true;
    }
    if (name == "VOLTS")
    {
        value = streaming::PhysicalUnit::volts;
        return true;
    }
    if (name == "AMPERES")
    {
        value = streaming::PhysicalUnit::amperes;
        return true;
    }
    if (name == "DIMENSIONLESS")
    {
        value = streaming::PhysicalUnit::dimensionless;
        return true;
    }
    return false;
}

[[nodiscard]] inline bool observation_timing_from(std::string_view name,
                                                  streaming::ObservationTiming& value) noexcept
{
    if (name == "NOT_APPLICABLE")
    {
        value = streaming::ObservationTiming::not_applicable;
        return true;
    }
    if (name == "REGULAR")
    {
        value = streaming::ObservationTiming::regular;
        return true;
    }
    if (name == "IRREGULAR")
    {
        value = streaming::ObservationTiming::irregular;
        return true;
    }
    return false;
}

[[nodiscard]] inline bool
timestamp_reference_from(std::string_view name,
                         streaming::FeatureTimestampReference& value) noexcept
{
    if (name == "WINDOW_CENTER")
    {
        value = streaming::FeatureTimestampReference::window_center;
        return true;
    }
    return false;
}

} // namespace neurale::recording::names
