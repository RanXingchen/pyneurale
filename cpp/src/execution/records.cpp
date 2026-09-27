/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "records.h"

#include <charconv>
#include <cmath>
#include <system_error>

namespace neurale::execution
{

using namespace neurale::experiments;
namespace
{
constexpr std::string_view kNull{"null"};
constexpr std::string_view kTrue{"true"};
constexpr std::string_view kFalse{"false"};

/// Longest text `std::to_chars` produces for any of the types written here.
/// A shortest-round-trip double is the widest of them.
constexpr std::size_t kNumberBytes = 40;
} // namespace

void JsonWriter::raw(char byte) noexcept
{
    if (size_ >= storage_.size())
    {
        overflowed_ = true;
        return;
    }
    storage_[size_++] = byte;
}

void JsonWriter::raw(std::string_view bytes) noexcept
{
    if (bytes.size() > storage_.size() - size_)
    {
        overflowed_ = true;
        return;
    }
    for (const auto byte : bytes)
    {
        storage_[size_++] = byte;
    }
}

void JsonWriter::escaped(std::string_view value) noexcept
{
    raw('"');
    for (const auto byte : value)
    {
        const auto character = static_cast<unsigned char>(byte);
        if (character == '"' || character == '\\')
        {
            raw('\\');
            raw(static_cast<char>(character));
        }
        else if (character < 0x20 || character > 0x7E)
        {
            // Everything this layer writes is ASCII it chose itself, so a byte
            // outside that range is a bug rather than content. It is escaped
            // instead of dropped: a record that quietly lost a byte would still
            // parse, and would be wrong somewhere nobody looks.
            static constexpr std::string_view kDigits{"0123456789abcdef"};
            raw(std::string_view{"\\u00"});
            raw(kDigits[(character >> 4) & 0x0F]);
            raw(kDigits[character & 0x0F]);
        }
        else
        {
            raw(static_cast<char>(character));
        }
    }
    raw('"');
}

void JsonWriter::restart() noexcept
{
    size_ = 0;
    overflowed_ = false;
    started_ = false;
    closed_ = false;
    raw('{');
}

void JsonWriter::begin_key(std::string_view key) noexcept
{
    if (started_)
    {
        raw(',');
    }
    started_ = true;
    escaped(key);
    raw(':');
}

void JsonWriter::u64(std::string_view key, std::uint64_t value) noexcept
{
    begin_key(key);
    std::array<char, kNumberBytes> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
}

void JsonWriter::i64(std::string_view key, std::int64_t value) noexcept
{
    begin_key(key);
    std::array<char, kNumberBytes> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
}

void JsonWriter::f64(std::string_view key, double value) noexcept
{
    begin_key(key);
    if (!std::isfinite(value))
    {
        raw(kNull);
        return;
    }
    std::array<char, kNumberBytes> buffer{};
    const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
    if (result.ec != std::errc{})
    {
        raw(kNull);
        return;
    }
    raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
}

void JsonWriter::boolean(std::string_view key, bool value) noexcept
{
    begin_key(key);
    raw(value ? kTrue : kFalse);
}

void JsonWriter::null(std::string_view key) noexcept
{
    begin_key(key);
    raw(kNull);
}

void JsonWriter::text(std::string_view key, std::string_view value) noexcept
{
    begin_key(key);
    escaped(value);
}

void JsonWriter::array_f64(std::string_view key, std::span<const double> values) noexcept
{
    begin_key(key);
    raw('[');
    bool first = true;
    for (const auto value : values)
    {
        if (!first)
        {
            raw(',');
        }
        first = false;
        if (!std::isfinite(value))
        {
            raw(kNull);
            continue;
        }
        std::array<char, kNumberBytes> buffer{};
        const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        if (result.ec != std::errc{})
        {
            raw(kNull);
            continue;
        }
        raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
    }
    raw(']');
}

void JsonWriter::array_u64(std::string_view key, std::span<const std::uint64_t> values) noexcept
{
    begin_key(key);
    raw('[');
    bool first = true;
    for (const auto value : values)
    {
        if (!first)
        {
            raw(',');
        }
        first = false;
        std::array<char, kNumberBytes> buffer{};
        const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
    }
    raw(']');
}

void JsonWriter::trial(std::string_view key, const TrialIdentity& identity) noexcept
{
    begin_key(key);
    raw('{');
    const auto member = [this](std::string_view name, std::uint64_t value, bool first)
    {
        if (!first)
        {
            raw(',');
        }
        escaped(name);
        raw(':');
        std::array<char, kNumberBytes> buffer{};
        const auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value);
        raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
    };
    member("ordinal", identity.ordinal, true);
    member("key", identity.key, false);
    member("block", identity.block, false);
    member("target_id", identity.target_id, false);
    member("stimulus_id", identity.stimulus_id, false);
    raw('}');
}

void JsonWriter::interval(std::string_view key, const TimeInterval& value) noexcept
{
    begin_key(key);
    raw('[');
    std::array<char, kNumberBytes> buffer{};
    auto result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value.start_ns);
    raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
    raw(',');
    result = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value.end_ns);
    raw(std::string_view{buffer.data(), static_cast<std::size_t>(result.ptr - buffer.data())});
    raw(']');
}

std::string_view JsonWriter::finish() noexcept
{
    if (!closed_)
    {
        raw('}');
        closed_ = true;
    }
    return std::string_view{storage_.data(), size_};
}

// --- ControlRecordWriter ----------------------------------------------------

JsonWriter& ControlRecordWriter::named(ControlKind kind, std::string_view name,
                                       ExperimentTimeNs time_ns) noexcept
{
    shape_ = Shape::named;
    kind_ = kind;
    name_ = name;
    time_ns_ = time_ns;
    has_value_ = false;
    value_ = 0.0;
    text_.restart();
    return text_;
}

bool ControlRecordWriter::value(double quantity) noexcept
{
    if (!std::isfinite(quantity))
    {
        return false;
    }
    has_value_ = true;
    value_ = quantity;
    return true;
}

bool ControlRecordWriter::value(std::uint64_t quantity) noexcept
{
    if (quantity >= kExactIntegerLimit)
    {
        return false;
    }
    has_value_ = true;
    value_ = static_cast<double>(quantity);
    return true;
}

JsonWriter& ControlRecordWriter::trial(const TrialRecord& record) noexcept
{
    shape_ = Shape::trials;
    kind_ = ControlKind::trials;
    // A trial spans an interval and the record header carries one instant, so
    // the interval stays whole in the body and the header names the instant the
    // record is ordered by. That is the recording core's own rule for this kind.
    time_ns_ = record.interval.start_ns;
    interval_ = record.interval;
    outcome_ = static_cast<std::uint32_t>(record.outcome);
    reason_ = record.reason;
    text_.restart();
    return text_;
}

JsonWriter& ControlRecordWriter::fault(std::string_view code, std::string_view stage,
                                       ExperimentTimeNs time_ns) noexcept
{
    shape_ = Shape::faults;
    kind_ = ControlKind::faults;
    name_ = code;
    stage_ = stage;
    time_ns_ = time_ns;
    text_.restart();
    return text_;
}

EncodedControl ControlRecordWriter::finish() noexcept
{
    const auto nested = text_.finish();
    JsonWriter body{body_storage_};
    body.restart();
    switch (shape_)
    {
    case Shape::named:
        body.text("name", name_);
        if (has_value_)
        {
            body.f64("value", value_);
        }
        else
        {
            body.null("value");
        }
        body.text("text", nested);
        break;
    case Shape::trials:
        body.u64("start_ns", interval_.start_ns);
        body.u64("stop_ns", interval_.end_ns);
        body.text("label", nested);
        {
            // The contract's outcome and its paradigm-owned reason are two
            // numbers and the column is one string, so they travel as a
            // two-field document rather than as a number whose reason was
            // dropped.
            std::array<char, 64> storage{};
            JsonWriter outcome{storage};
            outcome.restart();
            outcome.u64("outcome", outcome_);
            outcome.u64("reason", reason_);
            body.text("outcome", outcome.finish());
        }
        break;
    case Shape::faults:
        body.text("code", name_);
        body.text("stage", stage_);
        body.null("frame_sequence");
        body.null("signal_id");
        body.text("text", nested);
        break;
    }
    const auto encoded = body.finish();
    const bool complete = !text_.overflowed() && !body.overflowed();
    return EncodedControl{.kind = kind_,
                          .time_ns = time_ns_,
                          .body = complete ? encoded : std::string_view{},
                          .complete = complete};
}

} // namespace neurale::execution
