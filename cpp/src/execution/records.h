/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// Bounded encoding of one NRF control record, off every critical path.
///
/// The paradigm controllers publish compact, trivially-copyable trace structs
/// into a bounded queue and do nothing else. This header is what the other end
/// of that queue uses: it turns one trace struct into the bytes of one control
/// record. Every function here formats text and is therefore forbidden on a
/// realtime thread; the whole point of the queue is that none of this runs
/// there.
///
/// It defines **no new record kind and no new format**. The nine control kinds
/// are the recording core's producer-identity registry numbers, and the three
/// body shapes are exactly the three the recording finalizer accepts -- a body
/// carrying a field those schemas have no column for is rejected there, so
/// inventing one here would produce a session that records fine and finalizes
/// never.
///
/// Numbers, not names. The experiment contract stores a paradigm's own state,
/// phase, cause, and outcome enumerators as opaque integers beside the
/// ::neurale::experiments::ParadigmId they belong to, and declares that it
/// never interprets them. Spelling them here would create a second vocabulary
/// that has to be kept in step with the paradigm's own, so the records carry
/// the contract's numbers and the record *name* -- which this layer does own --
/// says which field is which.

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/events.h>
#include <neurale/experiments/identity.h>
#include <neurale/experiments/time.h>

namespace neurale::execution
{

using namespace neurale::experiments;

/// The nine NRF control-plane kinds, as the recording core's producer-identity
/// numbers.
///
/// The numbers are the registry's (`spool_layout.h`), repeated here rather than
/// included so that the record layer stays compilable against the experiment
/// contract alone. `session.cpp` static_asserts the two agree, which
/// is what keeps the repetition from becoming a second source of truth.
enum class ControlKind : std::uint32_t
{
    events = 3,
    trials = 4,
    experiment_states = 5,
    commands = 6,
    targets = 7,
    labels = 8,
    assistance = 9,
    faults = 10,
    task_variables = 11,
};

/// Largest nested `text` document one record may carry.
inline constexpr std::size_t kMaxControlTextBytes = 1024;
/// Largest whole body one record may carry. Escaping can only grow the nested
/// document, so the body has to be able to hold a fully escaped one plus the
/// enclosing object's own field names.
inline constexpr std::size_t kMaxControlBodyBytes = 2 * kMaxControlTextBytes + 128;

/// Magnitude above which an integer stops being exact in a float64. The
/// uniform body's `value` column is float64 in NRF, so a 64-bit identity goes
/// in the nested `text` -- where JSON integers stay integers -- instead of
/// being rounded on its way into a column that cannot hold it.
inline constexpr std::uint64_t kExactIntegerLimit = 1ULL << 53;

/// A fixed-capacity JSON object writer.
///
/// Storage belongs to the caller and never grows. Overflow is latched and
/// reported by finish(): a truncated document is not a shorter record, it is an
/// unreadable one, and the caller turns the latch into a counted refusal rather
/// than offering bytes that do not parse.
class JsonWriter
{
  public:
    explicit JsonWriter(std::span<char> storage) noexcept : storage_(storage) {}

    /// Reopen the writer on an empty object.
    void restart() noexcept;

    void u64(std::string_view key, std::uint64_t value) noexcept;
    void i64(std::string_view key, std::int64_t value) noexcept;
    /// A finite double. NaN and infinity are written as `null`: JSON has no
    /// spelling for them, and a reader must not be handed one it cannot parse.
    void f64(std::string_view key, double value) noexcept;
    void boolean(std::string_view key, bool value) noexcept;
    void null(std::string_view key) noexcept;
    /// An ASCII string value, escaped.
    void text(std::string_view key, std::string_view value) noexcept;
    void array_f64(std::string_view key, std::span<const double> values) noexcept;
    void array_u64(std::string_view key, std::span<const std::uint64_t> values) noexcept;

    /// The shared identity block every experiment record carries.
    void trial(std::string_view key, const TrialIdentity& identity) noexcept;
    /// A half-open interval, as its two instants.
    void interval(std::string_view key, const TimeInterval& value) noexcept;

    /// Close the object. Idempotent; the view is stable afterwards.
    [[nodiscard]] std::string_view finish() noexcept;
    [[nodiscard]] bool overflowed() const noexcept
    {
        return overflowed_;
    }

  private:
    void raw(std::string_view bytes) noexcept;
    void raw(char byte) noexcept;
    void begin_key(std::string_view key) noexcept;
    void escaped(std::string_view value) noexcept;

    std::span<char> storage_;
    std::size_t size_{};
    bool overflowed_{};
    bool started_{};
    bool closed_{};
};

/// One encoded record, ready to be offered to the recorder.
///
/// `body` views storage inside the writer that produced it and stays valid
/// until that writer starts the next record. The recorder copies the bytes
/// during submission, so the view never has to outlive the call.
struct EncodedControl
{
    ControlKind kind{ControlKind::labels};
    ExperimentTimeNs time_ns{};
    std::string_view body{};
    /// Whether the record is offerable at all. False when the fixed storage
    /// could not hold it, which the session counts as a refusal.
    bool complete{};
};

/// Builds one control record at a time into storage it owns.
///
/// One instance serves a whole session: `finish()` hands out a view, the next
/// `named()`/`trial()`/`fault()` overwrites it. Nothing here allocates after
/// construction, which is why the bridge can run on a thread with a budget even
/// though it is not a realtime one.
class ControlRecordWriter
{
  public:
    ControlRecordWriter() noexcept : text_(text_storage_) {}

    ControlRecordWriter(const ControlRecordWriter&) = delete;
    ControlRecordWriter& operator=(const ControlRecordWriter&) = delete;

    /// Begin one of the seven uniform `{name, value, text}` kinds. The returned
    /// writer builds the nested `text` document.
    JsonWriter& named(ControlKind kind, std::string_view name, ExperimentTimeNs time_ns) noexcept;

    /// Attach the record's float64 `value`. Refused, and left unset, for a
    /// value a float64 cannot carry back unchanged.
    bool value(double quantity) noexcept;
    /// Attach an integral `value`, refused above ::kExactIntegerLimit.
    bool value(std::uint64_t quantity) noexcept;

    /// Begin a `trials` record. The returned writer builds the nested `label`
    /// document; `outcome` and `reason` are the contract's own numbers and are
    /// written into the record's `outcome` column as a two-field document.
    JsonWriter& trial(const TrialRecord& record) noexcept;

    /// Begin a `faults` record. The returned writer builds its nested `text`.
    JsonWriter& fault(std::string_view code, std::string_view stage,
                      ExperimentTimeNs time_ns) noexcept;

    /// Compose the body and report whether it fits.
    [[nodiscard]] EncodedControl finish() noexcept;

  private:
    enum class Shape : std::uint8_t
    {
        named,
        trials,
        faults,
    };

    std::array<char, kMaxControlTextBytes> text_storage_{};
    std::array<char, kMaxControlBodyBytes> body_storage_{};
    JsonWriter text_;

    Shape shape_{Shape::named};
    ControlKind kind_{ControlKind::labels};
    ExperimentTimeNs time_ns_{};
    std::string_view name_{};
    std::string_view stage_{};
    bool has_value_{};
    double value_{};
    TimeInterval interval_{};
    std::uint32_t outcome_{};
    std::uint32_t reason_{};
};

} // namespace neurale::execution
