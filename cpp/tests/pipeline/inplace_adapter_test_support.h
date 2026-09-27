/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <span>
#include <stdexcept>
#include <vector>

#include <neurale/streaming/buffer_pool.h>
#include <neurale/streaming/frame.h>
#include <neurale/streaming/frame_emitter.h>
#include <neurale/streaming/frame_validation.h>
#include <neurale/streaming/processor.h>
#include <neurale/streaming/schema.h>

/// \file
/// Scaffolding shared by the sampled in-place adapter tests (FIR, IIR, SOS,
/// common reference). The four adapters implement one streaming archetype --
/// one sampled float64 sample-major signal in, the same schema out, the input
/// frame forwarded -- and their tests had four verbatim copies of the schema
/// construction, prepare context, frame filling, near-equality and terminal
/// sink that archetype needs. Each adapter test keeps its algorithm-specific
/// pieces (coefficients, offline reference, the test bodies) and includes this
/// for the rest.
///
/// **The geometry stays with the test, not here.** Every entry point takes an
/// :class:`InplaceGeometry`, and each test binds its own once: the SOS and
/// common-reference tests run at 30 kHz with a tight one-output/one-lease
/// context, the FIR and IIR tests at 1 kHz with eight, because each was chosen
/// against what that adapter dispatches on. A shared default that quietly moved
/// one of them would change what the test covers without changing what it
/// asserts.
///
/// This header allocates only in constructors -- :class:`TerminalSink` sizes its
/// pool and value buffer when it is built, which every caller does before
/// snapshotting the allocation counter. Nothing it does inside `record()` or
/// `fill_frame()` allocates, so a test can wrap a steady-state loop in the
/// counter and measure the adapter rather than the scaffolding.
namespace neurale::pipeline::test_support
{

/// The schema geometry and clock anchor one adapter test builds frames against.
struct InplaceGeometry
{
    std::uint32_t n_channels{2};
    std::uint32_t nominal_block_samples{4};
    std::uint32_t max_block_samples{16};
    streaming::RationalRate fs{1'000, 1};
    streaming::SignalDType dtype{streaming::SignalDType::float64};
    streaming::SignalLayout layout{streaming::SignalLayout::sample_major};
    /// Host time the frames' clock snapshot anchors on. Feeds the observation
    /// timestamps a downstream feature adapter derives, so it is per-test.
    std::uint64_t host_time_reference_ns{4'000'000};
    /// Device tick the snapshot anchors on, and the base of every block's tick.
    std::uint64_t device_tick_reference{8'000};
};

/// Identifiers every in-place adapter test shares. They are arbitrary but must
/// agree between the schema and the frames filled against it.
inline constexpr streaming::SchemaId kSchemaId{29};
inline constexpr streaming::SignalId kSignalId{11};
inline constexpr streaming::ClockDomainId kClockDomain{41};
inline constexpr streaming::ChannelSetId kChannelSetId{17};
inline constexpr streaming::CalibrationId kCalibrationId{19};
inline constexpr streaming::ReferenceId kReferenceId{23};

/// The single-signal stream schema *geometry* describes.
[[nodiscard]] inline streaming::StreamSchema make_schema(const InplaceGeometry& geometry)
{
    const std::array signals{
        streaming::SignalSchema{
            kSignalId,
            geometry.dtype,
            geometry.n_channels,
            geometry.nominal_block_samples,
            geometry.max_block_samples,
            geometry.fs,
            kClockDomain,
            geometry.layout,
            streaming::DeviceTickTracking::sample_counter,
            streaming::PhysicalUnit::volts,
            kChannelSetId,
            kCalibrationId,
            kReferenceId,
        },
    };
    return streaming::StreamSchema{kSchemaId, signals};
}

/// A prepare context over *schema*.
///
/// `flush_outputs` is a parameter rather than a constant because the four tests
/// disagree deliberately: an in-place adapter declares `max_flush_outputs = 0`,
/// and the FIR and IIR tests offer eight anyway to prove the adapter still
/// declares zero rather than inheriting what it was offered.
[[nodiscard]] inline streaming::ProcessorPrepareContext
make_context(const streaming::StreamSchema& schema, std::size_t outputs, std::size_t leases,
             std::size_t flush_outputs) noexcept
{
    return {
        .input_schema = schema,
        .max_process_outputs = outputs,
        .max_flush_outputs = flush_outputs,
        .available_frame_pool_leases = leases,
    };
}

/// Write *values* into *frame* as one sample-major block under *schema*.
///
/// Refuses a value count that is not a whole number of samples rather than
/// filling a ragged block, which the adapters would report as `invalid_frame`
/// several steps later and harder to read.
[[nodiscard]] inline streaming::StreamStatus
fill_frame(streaming::MutableFrame& frame, const streaming::StreamSchema& schema,
           const InplaceGeometry& geometry, std::span<const double> values,
           std::uint64_t sequence = 7, streaming::SampleIndex sample_start = 100) noexcept
{
    const auto channels = schema.signals().front().n_channels;
    if (values.empty() || channels == 0 || values.size() % channels != 0)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    const auto n_samples = values.size() / channels;
    frame.header() = streaming::FrameHeader{
        .session_id = 5,
        .sequence = sequence,
        .host_received_ns = 900 + sequence,
        .source_tick = 700 + sequence,
        .valid_until_ns = 1'900 + sequence,
        .schema_id = schema.id(),
        .source_clock_domain = kClockDomain,
        .flags = streaming::FrameFlags::source_tick | streaming::FrameFlags::valid_until |
                 streaming::FrameFlags::source_received,
    };
    frame.block_storage()[0] = streaming::SignalBlockHeader{
        .sample_idx_start = sample_start,
        .device_tick_start = geometry.device_tick_reference + sample_start,
        .payload_offset = 0,
        .payload_byte_count = values.size_bytes(),
        .signal_id = kSignalId,
        .n_samples = static_cast<std::uint32_t>(n_samples),
        .clock_sync =
            streaming::ClockSyncSnapshot{
                .device_tick_reference = geometry.device_tick_reference,
                .host_time_reference_ns = geometry.host_time_reference_ns,
                .device_tick_rate = geometry.fs,
                .uncertainty_ns = 3,
                .clock_domain = kClockDomain,
                .generation = 2,
                .flags = streaming::ClockSyncFlags::synchronized,
            },
    };
    std::memcpy(frame.payload_storage().data(), values.data(), values.size_bytes());
    return frame.set_used_sizes(1, values.size_bytes());
}

/// The payload of *frame* read as doubles.
[[nodiscard]] inline std::span<const double>
frame_values(const streaming::MutableFrame& frame) noexcept
{
    const auto view = frame.view();
    return {
        reinterpret_cast<const double*>(view.payload.data()),
        view.payload.size() / sizeof(double),
    };
}

/// Whether two runs agree to within a relative tolerance.
///
/// Scaled by the larger magnitude rather than compared absolutely, because a
/// filter's output spans several decades across a chunk and an absolute epsilon
/// is either useless at the top or spurious at the bottom.
[[nodiscard]] inline bool nearly_equal(std::span<const double> left, std::span<const double> right,
                                       double tolerance = 1e-12) noexcept
{
    if (left.size() != right.size())
    {
        return false;
    }
    for (std::size_t i = 0; i < left.size(); ++i)
    {
        const auto scale = (std::max)({1.0, std::abs(left[i]), std::abs(right[i])});
        if (std::abs(left[i] - right[i]) > tolerance * scale)
        {
            return false;
        }
    }
    return true;
}

/// Whether *callable* refuses with `std::invalid_argument`.
template <typename Callable> [[nodiscard]] bool throws_invalid_argument(Callable&& callable)
{
    try
    {
        callable();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

/// The chain terminal an in-place adapter publishes into, recording one frame.
///
/// Constructed with a frame-pool capacity: an in-place adapter never acquires,
/// so a test of the adapter alone passes zero and any acquire is reported as
/// `invalid_state`. A test that puts a windowed-feature adapter downstream --
/// which does acquire -- passes a nonzero capacity and gets a real pool.
class TerminalSink : public streaming::FrameEmitter
{
  public:
    TerminalSink(const streaming::StreamSchema& schema, std::size_t pool_frames,
                 std::size_t max_values)
        : validator_(schema), values_(max_values, 0.0)
    {
        if (pool_frames != 0)
        {
            pool_ = std::make_unique<streaming::FramePool>(
                pool_frames, schema.signals().front().max_block_bytes, 1);
        }
    }

    /// Point the sink at the frame the adapter will be asked to forward.
    void begin(streaming::MutableFrame& input) noexcept
    {
        input_ = &input;
        publication_count_ = 0;
        value_count_ = 0;
        status_ = streaming::StreamStatus::ok;
        static_cast<void>(acquired_.reset());
    }

    streaming::StreamStatus try_acquire_frame(streaming::MutableFrame*& frame) noexcept override
    {
        frame = nullptr;
        if (pool_ == nullptr || acquired_)
        {
            return status_ = streaming::StreamStatus::invalid_state;
        }
        const auto status = pool_->try_acquire(acquired_);
        if (status == streaming::StreamStatus::ok)
        {
            frame = &acquired_.frame();
        }
        return status;
    }

    streaming::StreamStatus publish_acquired_frame() noexcept override
    {
        if (!acquired_)
        {
            return status_ = streaming::StreamStatus::invalid_state;
        }
        const auto view = acquired_.view();
        return record(view, std::move(acquired_));
    }

    streaming::StreamStatus publish_input() noexcept override
    {
        return input_ == nullptr ? status_ = streaming::StreamStatus::invalid_state
                                 : record(input_->view(), {});
    }

    [[nodiscard]] std::size_t publication_count() const noexcept
    {
        return publication_count_;
    }
    [[nodiscard]] std::span<const double> values() const noexcept
    {
        return {values_.data(), value_count_};
    }
    [[nodiscard]] const streaming::FrameHeader& header() const noexcept
    {
        return header_;
    }
    [[nodiscard]] const streaming::SignalBlockHeader& block() const noexcept
    {
        return block_;
    }
    /// The latest refusal this sink reported, or `ok` if it never refused.
    ///
    /// Latched rather than returned only, so a test can drive a whole run and
    /// then ask once whether the terminal ever rejected what reached it.
    [[nodiscard]] streaming::StreamStatus status() const noexcept
    {
        return status_;
    }

  private:
    streaming::StreamStatus publish_owned(streaming::FrameLease lease) noexcept override
    {
        const auto view = lease.view();
        return record(view, std::move(lease));
    }

    streaming::StreamStatus record(streaming::FrameView view, streaming::FrameLease lease) noexcept
    {
        if (validator_.validate(view) != streaming::FrameValidationError::none ||
            view.blocks.size() != 1 || view.payload.size() > values_.size() * sizeof(double))
        {
            return status_ = streaming::StreamStatus::invalid_frame;
        }
        header_ = view.header;
        block_ = view.blocks.front();
        value_count_ = view.payload.size() / sizeof(double);
        std::memcpy(values_.data(), view.payload.data(), view.payload.size());
        ++publication_count_;
        return lease ? lease.reset() : streaming::StreamStatus::ok;
    }

    std::unique_ptr<streaming::FramePool> pool_;
    streaming::FrameValidator validator_;
    streaming::MutableFrame* input_{};
    streaming::FrameLease acquired_{};
    streaming::FrameHeader header_{};
    streaming::SignalBlockHeader block_{};
    std::vector<double> values_;
    std::size_t value_count_{};
    std::size_t publication_count_{};
    streaming::StreamStatus status_{streaming::StreamStatus::ok};
};

} // namespace neurale::pipeline::test_support
