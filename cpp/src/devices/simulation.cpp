/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/devices/simulation.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace neurale::devices::simulation
{
namespace
{

using signal::simulation::GenerationStatus;
using streaming::SignalDType;
using streaming::SignalLayout;
using streaming::SignalSchema;
using streaming::StreamSchema;

constexpr streaming::HostTimeNs cancel_poll_interval_ns = 1'000'000;
constexpr std::size_t max_scheduled_events = 4'096;
constexpr std::int64_t parts_per_million = 1'000'000;
std::atomic<streaming::SessionId> next_session_id{1};

[[nodiscard]] streaming::SessionId allocate_session_id() noexcept
{
    streaming::SessionId result{};
    do
    {
        result = next_session_id.fetch_add(1, std::memory_order_relaxed);
    } while (result == 0);
    return result;
}

/// Exclusive access to the acquisition state, taken by every read and reset.
///
/// Same single compare-and-exchange gate as the native replay source's
/// `GateGuard`, and for the same reason: see there for why observing a flag
/// and then acting on it is not a mutual-exclusion protocol.
class GateGuard
{
  public:
    explicit GateGuard(std::atomic<bool>& gate) noexcept : gate_(&gate)
    {
        bool expected = false;
        held_ = gate.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                             std::memory_order_acquire);
    }

    ~GateGuard()
    {
        if (held_)
        {
            gate_->store(false, std::memory_order_release);
        }
    }

    GateGuard(const GateGuard&) = delete;
    GateGuard& operator=(const GateGuard&) = delete;
    GateGuard(GateGuard&&) = delete;
    GateGuard& operator=(GateGuard&&) = delete;

    [[nodiscard]] bool held() const noexcept
    {
        return held_;
    }

  private:
    std::atomic<bool>* gate_;
    bool held_{};
};

[[nodiscard]] constexpr streaming::HostTimeNs saturating_add(streaming::HostTimeNs left,
                                                             streaming::HostTimeNs right) noexcept
{
    return right > std::numeric_limits<streaming::HostTimeNs>::max() - left
               ? std::numeric_limits<streaming::HostTimeNs>::max()
               : left + right;
}

[[nodiscard]] std::uint64_t checked_payload_bytes(std::size_t channels, std::uint32_t samples,
                                                  SignalDType dtype)
{
    const auto scalar_bytes = streaming::signal_dtype_size(dtype);
    if (channels > std::numeric_limits<std::uint64_t>::max() / samples ||
        channels * samples > std::numeric_limits<std::uint64_t>::max() / scalar_bytes)
    {
        throw std::overflow_error("simulated frame payload size overflows uint64");
    }
    return static_cast<std::uint64_t>(channels) * samples * scalar_bytes;
}

/// Payload size for one read, without the construction-time overflow check.
/// validate_config already proved the product fits for max_samples_per_frame
/// and every read uses at most that many samples, so the check has nothing
/// left to catch here -- and read_impl is noexcept, where the throwing form
/// would terminate the process instead of reporting anything.
[[nodiscard]] constexpr std::uint64_t payload_bytes_for(std::size_t channels, std::uint32_t samples,
                                                        SignalDType dtype) noexcept
{
    return static_cast<std::uint64_t>(channels) * samples * streaming::signal_dtype_size(dtype);
}

/// Round one generated value to a saturating 16-bit count. Values outside the
/// representable range clamp rather than wrap; a non-finite value is a
/// generator misconfiguration and is reported instead of being coerced to zero.
[[nodiscard]] bool quantize_int16(double value, std::int16_t& result) noexcept
{
    if (!std::isfinite(value))
    {
        return false;
    }
    constexpr auto lowest = static_cast<double>(std::numeric_limits<std::int16_t>::min());
    constexpr auto highest = static_cast<double>(std::numeric_limits<std::int16_t>::max());
    result = static_cast<std::int16_t>(std::clamp(std::round(value), lowest, highest));
    return true;
}

/// Generation buffer for a dtype the generator cannot write in place. float64
/// generation targets the payload directly, so it needs no scratch at all.
[[nodiscard]] std::vector<double> make_scratch(const signal::simulation::SignalGenerator& generator,
                                               const SimulatedNeuralSourceConfig& config)
{
    if (config.sample_dtype == SignalDType::float64)
    {
        return {};
    }
    return std::vector<double>(generator.channel_count() * config.max_samples_per_frame);
}

[[nodiscard]] SimulatedNeuralSourceConfig
validate_config(const signal::simulation::SignalGenerator& generator,
                SimulatedNeuralSourceConfig config)
{
    if (config.session_id == 0)
    {
        config.session_id = allocate_session_id();
    }
    if (generator.channel_count() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument("channel count exceeds StreamSchema capacity");
    }
    if (config.schema_id == 0 || config.signal_id == 0 || config.clock_domain == 0)
    {
        throw std::invalid_argument("schema, signal, and clock-domain ids must be nonzero");
    }
    if (config.nominal_samples_per_frame == 0 || config.max_samples_per_frame == 0 ||
        config.nominal_samples_per_frame > config.max_samples_per_frame)
    {
        throw std::invalid_argument("invalid nominal or maximum samples per frame");
    }
    if (config.fs.numerator == 0 || config.fs.denominator == 0)
    {
        throw std::invalid_argument("sample rate must be a positive rational");
    }
    if (config.clock_drift_ppm <= -parts_per_million)
    {
        throw std::invalid_argument("clock drift must keep the device tick rate positive");
    }
    if (config.clock_drift_ppm > std::numeric_limits<std::int64_t>::max() - parts_per_million)
    {
        throw std::invalid_argument("clock drift is outside the supported integer range");
    }
    if (!config.device_ticks && (config.clock_offset_ns != 0 || config.clock_drift_ppm != 0 ||
                                 config.clock_sync_uncertainty_ns != 0))
    {
        throw std::invalid_argument(
            "device-clock configuration requires sample-counter device ticks");
    }
    if (config.events.size() > max_scheduled_events)
    {
        throw std::invalid_argument("acquisition event schedule exceeds the fixed limit");
    }
    bool terminal_seen = false;
    std::uint64_t previous_ordinal = 0;
    for (std::size_t i = 0; i < config.events.size(); ++i)
    {
        const auto& event = config.events[i];
        if ((i != 0 && event.frame_ordinal < previous_ordinal) || terminal_seen)
        {
            throw std::invalid_argument(
                "acquisition events must be ordered and cannot follow a terminal event");
        }
        previous_ordinal = event.frame_ordinal;
        const bool has_samples = event.n_samples != 0;
        const bool has_duration = event.duration_ns != 0;
        const bool has_delta = event.device_tick_delta != 0;
        const bool has_restart = event.restart_tick != 0;
        switch (event.kind)
        {
        case AcquisitionEventKind::sample_loss:
            if (!has_samples || has_duration || has_delta || has_restart)
                throw std::invalid_argument("sample-loss events require only n_samples");
            break;
        case AcquisitionEventKind::stall:
            if (!has_duration || has_samples || has_delta || has_restart)
                throw std::invalid_argument("stall events require only a positive duration");
            break;
        case AcquisitionEventKind::device_tick_jump:
            if (!config.device_ticks || !has_delta || has_samples || has_duration || has_restart)
                throw std::invalid_argument("tick-jump events require sample-counter device ticks");
            break;
        case AcquisitionEventKind::device_restart:
            if (!config.device_ticks || has_samples || has_duration || has_delta)
                throw std::invalid_argument("device-restart events require device ticks");
            break;
        case AcquisitionEventKind::would_block:
        case AcquisitionEventKind::disconnect:
        case AcquisitionEventKind::source_fault:
            if (has_samples || has_duration || has_delta || has_restart)
                throw std::invalid_argument("this acquisition event takes no payload");
            terminal_seen = event.kind == AcquisitionEventKind::disconnect ||
                            event.kind == AcquisitionEventKind::source_fault;
            break;
        }
    }
    const auto declared_rate =
        static_cast<double>(config.fs.numerator) / static_cast<double>(config.fs.denominator);
    if (!std::isfinite(declared_rate) || declared_rate != generator.sample_rate())
    {
        throw std::invalid_argument("source sample rate must match the signal generator");
    }
    switch (config.sample_dtype)
    {
    case SignalDType::float64:
    case SignalDType::int16:
        break;
    default:
        throw std::invalid_argument("simulated payload dtype must be float64 or int16");
    }
    switch (config.physical_unit)
    {
    case streaming::PhysicalUnit::unspecified:
    case streaming::PhysicalUnit::volts:
    case streaming::PhysicalUnit::amperes:
    case streaming::PhysicalUnit::dimensionless:
        break;
    default:
        throw std::invalid_argument("physical unit is not supported by StreamSchema");
    }
    static_cast<void>(checked_payload_bytes(generator.channel_count(), config.max_samples_per_frame,
                                            config.sample_dtype));

    if (const auto finite = generator.finite_sample_count(); finite.has_value())
    {
        if (config.initial_sample_idx > *finite)
        {
            throw std::invalid_argument("initial sample index exceeds supplied samples");
        }
        const auto available = *finite - config.initial_sample_idx;
        if (!config.total_sample_count.has_value())
        {
            config.total_sample_count = available;
        }
        else if (*config.total_sample_count > available)
        {
            throw std::invalid_argument("total sample count exceeds supplied samples");
        }
    }
    if (config.total_sample_count.has_value() && *config.total_sample_count != 0 &&
        *config.total_sample_count - 1 >
            std::numeric_limits<streaming::SampleIndex>::max() - config.initial_sample_idx)
    {
        throw std::overflow_error("configured sample range overflows SampleIndex");
    }
    return config;
}

[[nodiscard]] streaming::RationalRate
make_device_tick_rate(const SimulatedNeuralSourceConfig& config)
{
    const auto scale = static_cast<std::uint64_t>(parts_per_million + config.clock_drift_ppm);
    if (config.fs.numerator > std::numeric_limits<std::uint64_t>::max() / scale ||
        config.fs.denominator > std::numeric_limits<std::uint64_t>::max() /
                                    static_cast<std::uint64_t>(parts_per_million))
    {
        throw std::overflow_error("drifted device tick rate overflows RationalRate");
    }
    auto num = config.fs.numerator * scale;
    auto den = config.fs.denominator * static_cast<std::uint64_t>(parts_per_million);
    const auto divisor = std::gcd(num, den);
    return {num / divisor, den / divisor};
}

[[nodiscard]] bool add_tick_delta(streaming::DeviceTick value, std::int64_t delta,
                                  streaming::DeviceTick& result) noexcept
{
    if (delta >= 0)
    {
        const auto amount = static_cast<std::uint64_t>(delta);
        if (amount > std::numeric_limits<streaming::DeviceTick>::max() - value)
            return false;
        result = value + amount;
        return true;
    }
    const auto amount = static_cast<std::uint64_t>(-(delta + 1)) + 1;
    if (amount > value)
        return false;
    result = value - amount;
    return true;
}

[[nodiscard]] streaming::NativeClock& require_clock(const std::shared_ptr<ManualHostClock>& clock)
{
    if (!clock)
        throw std::invalid_argument("manual host clock cannot be null");
    return *clock;
}

[[nodiscard]] StreamSchema make_schema(const signal::simulation::SignalGenerator& generator,
                                       const SimulatedNeuralSourceConfig& config)
{
    const SignalSchema signal{
        config.signal_id,
        config.sample_dtype,
        static_cast<std::uint32_t>(generator.channel_count()),
        config.nominal_samples_per_frame,
        config.max_samples_per_frame,
        config.fs,
        config.clock_domain,
        SignalLayout::sample_major,
        config.device_ticks ? streaming::DeviceTickTracking::sample_counter
                            : streaming::DeviceTickTracking::unavailable,
        config.physical_unit,
        config.channel_set_id,
        config.calibration_id,
        config.reference_id,
        streaming::SignalKind::sampled,
        0,
        streaming::ObservationTiming::not_applicable,
        0,
        config.channel_names,
        config.channel_impedances_ohm,
    };
    return StreamSchema{config.schema_id, std::span<const SignalSchema>(&signal, 1)};
}

} // namespace

class ManualNeuralControlState final : public streaming::NeuralIntentSource
{
  public:
    struct Snapshot
    {
        streaming::NeuralIntentSnapshot intent;
        double drift_progress{};
    };

    void publish(const streaming::NeuralIntentSnapshot& intent, double drift_progress) noexcept
    {
        revision_.fetch_add(1, std::memory_order_acq_rel);
        std::atomic_thread_fence(std::memory_order_release);
        sequence_.store(intent.sequence, std::memory_order_relaxed);
        intent_x_.store(intent.intent_x, std::memory_order_relaxed);
        intent_y_.store(intent.intent_y, std::memory_order_relaxed);
        context_ordinal_.store(intent.context_ordinal, std::memory_order_relaxed);
        valid_.store(intent.valid, std::memory_order_relaxed);
        drift_progress_.store(drift_progress, std::memory_order_relaxed);
        revision_.fetch_add(1, std::memory_order_release);
    }

    [[nodiscard]] Snapshot read_control() const noexcept
    {
        Snapshot result{};
        for (;;)
        {
            const auto before = revision_.load(std::memory_order_acquire);
            if ((before & 1U) != 0U)
                continue;
            result.intent.sequence = sequence_.load(std::memory_order_relaxed);
            result.intent.intent_x = intent_x_.load(std::memory_order_relaxed);
            result.intent.intent_y = intent_y_.load(std::memory_order_relaxed);
            result.intent.context_ordinal = context_ordinal_.load(std::memory_order_relaxed);
            result.intent.valid = valid_.load(std::memory_order_relaxed);
            result.drift_progress = drift_progress_.load(std::memory_order_relaxed);
            std::atomic_thread_fence(std::memory_order_acquire);
            if (revision_.load(std::memory_order_acquire) == before)
                return result;
        }
    }

    [[nodiscard]] streaming::NeuralIntentSnapshot read_intent() const noexcept override
    {
        return read_control().intent;
    }

    void reset() noexcept
    {
        publish({}, 0.0);
    }

  private:
    std::atomic<std::uint64_t> revision_{};
    std::atomic<std::uint64_t> sequence_{};
    std::atomic<double> intent_x_{};
    std::atomic<double> intent_y_{};
    std::atomic<std::uint64_t> context_ordinal_{};
    std::atomic<bool> valid_{};
    std::atomic<double> drift_progress_{};
};

ManualHostClock::ManualHostClock(streaming::HostTimeNs initial_time_ns) noexcept
    : now_ns_(initial_time_ns)
{
}

streaming::HostTimeNs ManualHostClock::now_ns() noexcept
{
    return now_ns_.load(std::memory_order_acquire);
}

void ManualHostClock::wait_until(streaming::HostTimeNs deadline_ns) noexcept
{
    std::unique_lock lock{mutex_};
    const auto revision = revision_;
    condition_.wait(lock, [this, deadline_ns, revision]
                    { return revision_ != revision || now_ns() >= deadline_ns; });
}

void ManualHostClock::wake() noexcept
{
    {
        std::lock_guard lock{mutex_};
        ++revision_;
    }
    condition_.notify_all();
}

bool ManualHostClock::set(streaming::HostTimeNs time_ns) noexcept
{
    auto current = now_ns_.load(std::memory_order_relaxed);
    do
    {
        if (time_ns < current)
            return false;
    } while (!now_ns_.compare_exchange_weak(current, time_ns, std::memory_order_release,
                                            std::memory_order_relaxed));
    wake();
    return true;
}

void ManualHostClock::advance(streaming::HostTimeNs duration_ns) noexcept
{
    auto current = now_ns_.load(std::memory_order_relaxed);
    for (;;)
    {
        const auto next = saturating_add(current, duration_ns);
        if (now_ns_.compare_exchange_weak(current, next, std::memory_order_release,
                                          std::memory_order_relaxed))
            break;
    }
    wake();
}

SimulatedNeuralSource::SimulatedNeuralSource(signal::simulation::SignalGenerator generator,
                                             SimulatedNeuralSourceConfig config)
    : SimulatedNeuralSource(std::move(generator), std::move(config),
                            streaming::default_native_clock())
{
}

streaming::StreamStatus SimulatedNeuralSource::generate_samples(streaming::SampleIndex sample_start,
                                                                std::uint32_t n_samples,
                                                                std::span<double> output) noexcept
{
    const auto generated = generator_.generate(sample_start, n_samples, output);
    if (generated == GenerationStatus::ok)
        return streaming::StreamStatus::ok;
    return generated == GenerationStatus::invalid_output ? streaming::StreamStatus::invalid_frame
                                                         : streaming::StreamStatus::source_failure;
}

SimulatedNeuralSource::SimulatedNeuralSource(signal::simulation::SignalGenerator generator,
                                             SimulatedNeuralSourceConfig config,
                                             streaming::NativeClock& clock)
    : generator_(std::move(generator)), config_(validate_config(generator_, std::move(config))),
      schema_(make_schema(generator_, config_)),
      quantization_scratch_(make_scratch(generator_, config_)),
      device_tick_rate_(make_device_tick_rate(config_)), clock_(clock),
      device_tick_(config_.initial_device_tick)
{
    has_discontinuity_events_ =
        std::any_of(config_.events.begin(), config_.events.end(),
                    [](const AcquisitionEvent& event)
                    {
                        return event.kind == AcquisitionEventKind::sample_loss ||
                               event.kind == AcquisitionEventKind::device_tick_jump ||
                               event.kind == AcquisitionEventKind::device_restart;
                    });
}

SimulatedNeuralSource::SimulatedNeuralSource(signal::simulation::SignalGenerator generator,
                                             SimulatedNeuralSourceConfig config,
                                             std::shared_ptr<ManualHostClock> clock)
    : generator_(std::move(generator)), config_(validate_config(generator_, std::move(config))),
      schema_(make_schema(generator_, config_)),
      quantization_scratch_(make_scratch(generator_, config_)),
      device_tick_rate_(make_device_tick_rate(config_)), owned_host_clock_(std::move(clock)),
      clock_(require_clock(owned_host_clock_)), device_tick_(config_.initial_device_tick)
{
    has_discontinuity_events_ =
        std::any_of(config_.events.begin(), config_.events.end(),
                    [](const AcquisitionEvent& event)
                    {
                        return event.kind == AcquisitionEventKind::sample_loss ||
                               event.kind == AcquisitionEventKind::device_tick_jump ||
                               event.kind == AcquisitionEventKind::device_restart;
                    });
}

streaming::HostTimeNs
SimulatedNeuralSource::deadline_for(std::uint64_t sample_offset) const noexcept
{
    const auto seconds = static_cast<long double>(sample_offset) *
                         static_cast<long double>(device_tick_rate_.denominator) /
                         static_cast<long double>(device_tick_rate_.numerator);
    const auto offset_ns = std::round(seconds * 1'000'000'000.0L);
    const auto available = static_cast<long double>(
        std::numeric_limits<streaming::HostTimeNs>::max() - epoch_host_ns_);
    if (!std::isfinite(offset_ns) || offset_ns >= available)
    {
        return std::numeric_limits<streaming::HostTimeNs>::max();
    }
    return epoch_host_ns_ + static_cast<streaming::HostTimeNs>(offset_ns);
}

bool SimulatedNeuralSource::sync_host_time(streaming::HostTimeNs deadline,
                                           streaming::HostTimeNs& result) const noexcept
{
    if (config_.clock_offset_ns >= 0)
    {
        const auto offset = static_cast<std::uint64_t>(config_.clock_offset_ns);
        if (offset > std::numeric_limits<streaming::HostTimeNs>::max() - deadline)
            return false;
        result = deadline + offset;
        return true;
    }
    const auto offset = static_cast<std::uint64_t>(-(config_.clock_offset_ns + 1)) + 1;
    if (offset > deadline)
        return false;
    result = deadline - offset;
    return true;
}

streaming::StreamStatus SimulatedNeuralSource::wait_for(streaming::HostTimeNs deadline) noexcept
{
    for (;;)
    {
        if (closed_.load(std::memory_order_acquire) || cancelled_.load(std::memory_order_acquire))
        {
            return streaming::StreamStatus::stopped;
        }
        const auto now = clock_.now_ns();
        if (now >= deadline)
        {
            return streaming::StreamStatus::ok;
        }
        // NativeClock::wake() is deliberately generation-based. Bounded slices
        // also cover cancellation racing immediately before wait_until().
        clock_.wait_until(std::min(deadline, saturating_add(now, cancel_poll_interval_ns)));
    }
}

streaming::StreamStatus SimulatedNeuralSource::publish_gap(
    streaming::DiscontinuityLease* discontinuity, streaming::GapReason reason,
    streaming::SampleIndex expected_sample, streaming::SampleIndex actual_sample,
    streaming::DeviceTick expected_tick, streaming::DeviceTick actual_tick,
    std::uint64_t missing_samples) noexcept
{
    if (discontinuity == nullptr)
        return streaming::StreamStatus::invalid_state;
    auto flags = streaming::SignalGapFlags::missing_samples_known;
    if (config_.device_ticks)
        flags = flags | streaming::SignalGapFlags::device_ticks_available;
    const streaming::SignalGap gap{
        .expected_sample_idx = expected_sample,
        .actual_sample_idx = actual_sample,
        .missing_samples = missing_samples,
        .expected_device_tick = config_.device_ticks ? expected_tick : 0,
        .actual_device_tick = config_.device_ticks ? actual_tick : 0,
        .signal_id = config_.signal_id,
        .reason = reason,
        .flags = flags,
    };
    const auto ordinal = frames_emitted_.load(std::memory_order_relaxed);
    const auto previous = ordinal == 0 ? 0 : ordinal - 1;
    const auto status = discontinuity->assign(config_.session_id, previous, ordinal, reason,
                                              std::span<const streaming::SignalGap>(&gap, 1));
    return status == streaming::StreamStatus::ok ? streaming::StreamStatus::discontinuity : status;
}

streaming::StreamStatus
SimulatedNeuralSource::apply_scheduled_events(streaming::DiscontinuityLease* discontinuity) noexcept
{
    const auto ordinal = frames_emitted_.load(std::memory_order_relaxed);
    while (next_event_ < config_.events.size() &&
           config_.events[next_event_].frame_ordinal == ordinal)
    {
        const auto& event = config_.events[next_event_];
        switch (event.kind)
        {
        case AcquisitionEventKind::would_block:
            ++next_event_;
            return streaming::StreamStatus::would_block;
        case AcquisitionEventKind::stall:
        {
            ++next_event_;
            if (!epoch_initialized_)
            {
                epoch_host_ns_ = clock_.now_ns();
                epoch_initialized_ = true;
            }
            const auto expected = deadline_for(acquisition_sample_offset_);
            const auto base = std::max(expected, clock_.now_ns());
            const auto status = wait_for(saturating_add(base, event.duration_ns));
            if (status != streaming::StreamStatus::ok)
                return status;
            break;
        }
        case AcquisitionEventKind::disconnect:
        case AcquisitionEventKind::source_fault:
            ++next_event_;
            return streaming::StreamStatus::source_failure;
        case AcquisitionEventKind::sample_loss:
        {
            if (discontinuity == nullptr)
                return streaming::StreamStatus::invalid_state;
            if (event.n_samples > std::numeric_limits<streaming::SampleIndex>::max() -
                                      config_.initial_sample_idx - acquisition_sample_offset_)
                return streaming::StreamStatus::source_failure;
            if (config_.total_sample_count.has_value() &&
                event.n_samples > *config_.total_sample_count - acquisition_sample_offset_)
                return streaming::StreamStatus::source_failure;
            if (config_.device_ticks &&
                event.n_samples > std::numeric_limits<streaming::DeviceTick>::max() - device_tick_)
                return streaming::StreamStatus::source_failure;
            const auto expected_sample = config_.initial_sample_idx + acquisition_sample_offset_;
            const auto actual_sample = expected_sample + event.n_samples;
            const auto expected_tick = device_tick_;
            const auto actual_tick =
                config_.device_ticks ? device_tick_ + event.n_samples : streaming::DeviceTick{};
            const auto status =
                publish_gap(discontinuity, streaming::GapReason::sample_gap, expected_sample,
                            actual_sample, expected_tick, actual_tick, event.n_samples);
            if (status != streaming::StreamStatus::discontinuity)
                return status;
            acquisition_sample_offset_ += event.n_samples;
            if (config_.device_ticks)
                device_tick_ = actual_tick;
            ++next_event_;
            return status;
        }
        case AcquisitionEventKind::device_tick_jump:
        {
            if (discontinuity == nullptr)
                return streaming::StreamStatus::invalid_state;
            streaming::DeviceTick actual_tick{};
            if (!add_tick_delta(device_tick_, event.device_tick_delta, actual_tick) ||
                clock_sync_generation_ == std::numeric_limits<std::uint32_t>::max())
                return streaming::StreamStatus::source_failure;
            const auto sample = config_.initial_sample_idx + acquisition_sample_offset_;
            const auto status = publish_gap(discontinuity, streaming::GapReason::device_tick_gap,
                                            sample, sample, device_tick_, actual_tick, 0);
            if (status != streaming::StreamStatus::discontinuity)
                return status;
            device_tick_ = actual_tick;
            ++clock_sync_generation_;
            ++next_event_;
            return status;
        }
        case AcquisitionEventKind::device_restart:
        {
            if (discontinuity == nullptr)
                return streaming::StreamStatus::invalid_state;
            if (clock_sync_generation_ == std::numeric_limits<std::uint32_t>::max())
                return streaming::StreamStatus::source_failure;
            const auto sample = config_.initial_sample_idx + acquisition_sample_offset_;
            const auto status = publish_gap(discontinuity, streaming::GapReason::device_restart,
                                            sample, sample, device_tick_, event.restart_tick, 0);
            if (status != streaming::StreamStatus::discontinuity)
                return status;
            device_tick_ = event.restart_tick;
            ++clock_sync_generation_;
            ++next_event_;
            return status;
        }
        }
    }
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
SimulatedNeuralSource::read_impl(streaming::MutableFrame& frame,
                                 streaming::DiscontinuityLease* discontinuity) noexcept
{
    if (closed_.load(std::memory_order_acquire) || cancelled_.load(std::memory_order_acquire))
        return streaming::StreamStatus::stopped;

    const auto required_bytes = payload_bytes_for(
        generator_.channel_count(), config_.nominal_samples_per_frame, config_.sample_dtype);
    if (frame.block_storage().empty() || required_bytes > frame.payload_storage().size())
        return streaming::StreamStatus::invalid_frame;

    if (config_.total_sample_count.has_value() &&
        acquisition_sample_offset_ == *config_.total_sample_count)
        return streaming::StreamStatus::end_of_stream;

    if (const auto status = apply_scheduled_events(discontinuity);
        status != streaming::StreamStatus::ok)
        return status;

    if (config_.total_sample_count.has_value() &&
        acquisition_sample_offset_ == *config_.total_sample_count)
        return streaming::StreamStatus::end_of_stream;

    if (!epoch_initialized_)
    {
        epoch_host_ns_ = clock_.now_ns();
        epoch_initialized_ = true;
    }

    const auto emitted_samples = samples_emitted_.load(std::memory_order_relaxed);
    const auto frame_sequence = frames_emitted_.load(std::memory_order_relaxed);
    if (frame_sequence == std::numeric_limits<std::uint64_t>::max())
    {
        return streaming::StreamStatus::end_of_stream;
    }
    if (acquisition_sample_offset_ >
        std::numeric_limits<streaming::SampleIndex>::max() - config_.initial_sample_idx)
    {
        return streaming::StreamStatus::end_of_stream;
    }
    const auto sample_start = config_.initial_sample_idx + acquisition_sample_offset_;
    auto n_samples = config_.nominal_samples_per_frame;
    if (config_.total_sample_count.has_value())
    {
        const auto remaining = *config_.total_sample_count - acquisition_sample_offset_;
        n_samples = static_cast<std::uint32_t>(std::min<std::uint64_t>(n_samples, remaining));
    }
    else if (n_samples > std::numeric_limits<streaming::SampleIndex>::max() - sample_start)
    {
        return streaming::StreamStatus::end_of_stream;
    }

    const auto payload_bytes =
        payload_bytes_for(generator_.channel_count(), n_samples, config_.sample_dtype);
    const auto deadline = deadline_for(acquisition_sample_offset_);
    if (config_.paced)
        if (const auto status = wait_for(deadline); status != streaming::StreamStatus::ok)
            return status;
    if (closed_.load(std::memory_order_acquire) || cancelled_.load(std::memory_order_acquire))
    {
        return streaming::StreamStatus::stopped;
    }

    if (config_.device_ticks &&
        n_samples > std::numeric_limits<streaming::DeviceTick>::max() - device_tick_)
        return streaming::StreamStatus::source_failure;
    streaming::HostTimeNs sync_time{};
    if (config_.device_ticks && !sync_host_time(deadline, sync_time))
        return streaming::StreamStatus::source_failure;
    const auto device_tick = device_tick_;
    frame.header() = streaming::FrameHeader{
        .session_id = config_.session_id,
        .sequence = frame_sequence,
        .host_received_ns = 0,
        .source_tick = config_.device_ticks ? device_tick : 0,
        .schema_id = config_.schema_id,
        .source_clock_domain = config_.clock_domain,
        .flags =
            config_.device_ticks ? streaming::FrameFlags::source_tick : streaming::FrameFlags::none,
    };
    frame.block_storage()[0] = streaming::SignalBlockHeader{
        .sample_idx_start = sample_start,
        .device_tick_start = config_.device_ticks ? device_tick : 0,
        .payload_offset = 0,
        .payload_byte_count = payload_bytes,
        .signal_id = config_.signal_id,
        .n_samples = n_samples,
        .clock_sync =
            config_.device_ticks
                ? streaming::ClockSyncSnapshot{
                      .device_tick_reference = device_tick,
                      .host_time_reference_ns = sync_time,
                      .device_tick_rate = device_tick_rate_,
                      .uncertainty_ns = config_.clock_sync_uncertainty_ns,
                      .clock_domain = config_.clock_domain,
                      .generation = clock_sync_generation_,
                      .flags = streaming::ClockSyncFlags::synchronized,
                  }
                : streaming::ClockSyncSnapshot{},
    };

    const auto n_scalars = static_cast<std::size_t>(n_samples) * generator_.channel_count();
    // float64 generates straight into the payload. Any other dtype generates
    // into the scratch sized at construction and converts in place afterwards,
    // so neither path allocates and neither needs the payload to be wider than
    // the dtype the schema declares.
    const auto target =
        config_.sample_dtype == SignalDType::float64
            ? std::span<double>(reinterpret_cast<double*>(frame.payload_storage().data()),
                                n_scalars)
            : std::span<double>(quantization_scratch_.data(), n_scalars);
    if (const auto generated = generate_samples(sample_start, n_samples, target);
        generated != streaming::StreamStatus::ok)
        return generated;
    if (config_.sample_dtype == SignalDType::int16)
    {
        auto* counts = reinterpret_cast<std::int16_t*>(frame.payload_storage().data());
        for (std::size_t i = 0; i < n_scalars; ++i)
        {
            if (!quantize_int16(target[i], counts[i]))
            {
                return streaming::StreamStatus::source_failure;
            }
        }
    }
    if (const auto status = frame.set_used_sizes(1, static_cast<std::size_t>(payload_bytes));
        status != streaming::StreamStatus::ok)
    {
        return status;
    }

    samples_emitted_.store(emitted_samples + n_samples, std::memory_order_release);
    acquisition_sample_offset_ += n_samples;
    if (config_.device_ticks)
        device_tick_ += n_samples;
    frames_emitted_.store(frame_sequence + 1, std::memory_order_release);
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus SimulatedNeuralSource::read(streaming::MutableFrame& frame) noexcept
{
    GateGuard guard{operation_gate_};
    if (!guard.held())
        return streaming::StreamStatus::invalid_state;
    return read_impl(frame, nullptr);
}

streaming::StreamStatus
SimulatedNeuralSource::read_message(streaming::MutableFrame& frame,
                                    streaming::DiscontinuityLease& discontinuity) noexcept
{
    GateGuard guard{operation_gate_};
    if (!guard.held())
        return streaming::StreamStatus::invalid_state;
    return read_impl(frame, &discontinuity);
}

bool SimulatedNeuralSource::produces_discontinuities() const noexcept
{
    return has_discontinuity_events_;
}

void SimulatedNeuralSource::cancel() noexcept
{
    cancelled_.store(true, std::memory_order_release);
    clock_.wake();
}

void SimulatedNeuralSource::close() noexcept
{
    closed_.store(true, std::memory_order_release);
    cancel();
}

streaming::StreamStatus SimulatedNeuralSource::reset() noexcept
{
    // A reset concurrent with an in-flight read is refused rather than raced:
    // every field below is plain acquisition state the reading thread owns
    // while it is inside read_impl. cancel() remains the way to end that read,
    // and it stays outside the gate so it still works while one is held.
    GateGuard guard{operation_gate_};
    if (!guard.held())
        return streaming::StreamStatus::invalid_state;
    if (closed_.load(std::memory_order_acquire))
        return streaming::StreamStatus::invalid_state;
    frames_emitted_.store(0, std::memory_order_relaxed);
    samples_emitted_.store(0, std::memory_order_relaxed);
    epoch_host_ns_ = 0;
    epoch_initialized_ = false;
    acquisition_sample_offset_ = 0;
    device_tick_ = config_.initial_device_tick;
    clock_sync_generation_ = 1;
    next_event_ = 0;
    reset_generator();
    cancelled_.store(false, std::memory_order_release);
    return streaming::StreamStatus::ok;
}

IntentDrivenNeuralSource::IntentDrivenNeuralSource(
    signal::simulation::NeuralSignalConfig neural_config, SimulatedNeuralSourceConfig source_config,
    std::optional<NeuralDriftSchedule> drift_schedule, std::size_t evidence_capacity)
    : SimulatedNeuralSource(
          signal::simulation::SignalGenerator::zeros(neural_config.n_channels, neural_config.fs),
          source_config),
      generator_(std::move(neural_config)),
      manual_source_(std::make_shared<ManualNeuralControlState>()), drift_schedule_(drift_schedule)
{
    if (source_config.initial_sample_idx != 0)
        throw std::invalid_argument("intent-driven neural sources must start at sample index zero");
    if (std::any_of(source_config.events.begin(), source_config.events.end(),
                    [](const AcquisitionEvent& event)
                    { return event.kind == AcquisitionEventKind::sample_loss; }))
        throw std::invalid_argument(
            "intent-driven neural sources do not support simulated sample-loss events");
    if (evidence_capacity == 0 || evidence_capacity == std::numeric_limits<std::size_t>::max())
        throw std::invalid_argument("evidence_capacity must be positive");
    evidence_.resize(evidence_capacity + 1U);
    if (drift_schedule_.has_value())
    {
        const auto& schedule = *drift_schedule_;
        if (schedule.end_ordinal <= schedule.start_ordinal ||
            !std::isfinite(schedule.start_progress) || !std::isfinite(schedule.end_progress) ||
            schedule.start_progress < 0.0 || schedule.start_progress > 1.0 ||
            schedule.end_progress < 0.0 || schedule.end_progress > 1.0)
            throw std::invalid_argument(
                "drift schedule ordinals must increase and progress must be in [0, 1]");
    }
}

streaming::StreamStatus IntentDrivenNeuralSource::bind_intent_source(
    std::shared_ptr<streaming::NeuralIntentSource> source) noexcept
{
    if (!source)
        return streaming::StreamStatus::invalid_state;
    auto expected = ControlMode::undecided;
    if (!control_mode_.compare_exchange_strong(
            expected, ControlMode::binding, std::memory_order_acq_rel, std::memory_order_acquire))
        return streaming::StreamStatus::invalid_state;
    intent_source_ = std::move(source);
    control_mode_.store(ControlMode::bound, std::memory_order_release);
    control_mode_.notify_all();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
IntentDrivenNeuralSource::publish_control(double intent_x, double intent_y,
                                          double drift_progress_value,
                                          std::uint64_t context_ordinal) noexcept
{
    if (!std::isfinite(intent_x) || !std::isfinite(intent_y) ||
        !std::isfinite(drift_progress_value) || drift_progress_value < 0.0 ||
        drift_progress_value > 1.0 ||
        manual_sequence_.load(std::memory_order_relaxed) ==
            std::numeric_limits<std::uint64_t>::max())
        return streaming::StreamStatus::invalid_state;
    auto mode = control_mode_.load(std::memory_order_acquire);
    if (mode == ControlMode::undecided)
    {
        auto expected = ControlMode::undecided;
        if (control_mode_.compare_exchange_strong(expected, ControlMode::manual,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
            mode = ControlMode::manual;
        else
            mode = expected;
    }
    if (mode != ControlMode::manual)
        return streaming::StreamStatus::invalid_state;
    const auto sequence = manual_sequence_.fetch_add(1, std::memory_order_relaxed) + 1U;
    manual_source_->publish({sequence, intent_x, intent_y, context_ordinal, 0, 0, 0, true},
                            drift_progress_value);
    return streaming::StreamStatus::ok;
}

double IntentDrivenNeuralSource::drift_progress(std::uint64_t context_ordinal) const noexcept
{
    if (!drift_schedule_.has_value())
        return 0.0;
    const auto& schedule = *drift_schedule_;
    if (context_ordinal <= schedule.start_ordinal)
        return schedule.start_progress;
    if (context_ordinal >= schedule.end_ordinal)
        return schedule.end_progress;
    const auto fraction = static_cast<double>(context_ordinal - schedule.start_ordinal) /
                          static_cast<double>(schedule.end_ordinal - schedule.start_ordinal);
    return schedule.start_progress + fraction * (schedule.end_progress - schedule.start_progress);
}

void IntentDrivenNeuralSource::push_applied_control(const AppliedNeuralControl& control) noexcept
{
    const auto write = evidence_write_.load(std::memory_order_relaxed);
    const auto next = (write + 1U) % evidence_.size();
    if (next == evidence_read_.load(std::memory_order_acquire))
    {
        evidence_dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    evidence_[write] = control;
    evidence_write_.store(next, std::memory_order_release);
}

streaming::StreamStatus IntentDrivenNeuralSource::generate_samples(
    streaming::SampleIndex sample_start, std::uint32_t n_samples, std::span<double> output) noexcept
{
    auto mode = control_mode_.load(std::memory_order_acquire);
    if (mode == ControlMode::undecided)
    {
        auto expected = ControlMode::undecided;
        if (control_mode_.compare_exchange_strong(expected, ControlMode::manual,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_acquire))
            mode = ControlMode::manual;
        else
            mode = expected;
    }
    while (mode == ControlMode::binding)
    {
        control_mode_.wait(ControlMode::binding, std::memory_order_acquire);
        mode = control_mode_.load(std::memory_order_acquire);
    }
    const bool bound = mode == ControlMode::bound;
    const auto manual =
        bound ? ManualNeuralControlState::Snapshot{} : manual_source_->read_control();
    const auto intent = bound ? intent_source_->read_intent() : manual.intent;
    const auto progress = bound ? drift_progress(intent.context_ordinal) : manual.drift_progress;
    const auto status =
        generator_.generate(n_samples, {intent.intent_x, intent.intent_y}, progress, output);
    if (status != signal::simulation::NeuralGenerationStatus::ok)
        return status == signal::simulation::NeuralGenerationStatus::invalid_output
                   ? streaming::StreamStatus::invalid_frame
                   : streaming::StreamStatus::source_failure;
    push_applied_control({intent.sequence, intent.intent_x, intent.intent_y, intent.context_ordinal,
                          intent.valid, progress, sample_start, sample_start + n_samples - 1U,
                          generated_frame_sequence_++});
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus
IntentDrivenNeuralSource::try_pop_applied_control(AppliedNeuralControl& control) noexcept
{
    const auto read = evidence_read_.load(std::memory_order_relaxed);
    if (read == evidence_write_.load(std::memory_order_acquire))
        return streaming::StreamStatus::would_block;
    control = evidence_[read];
    evidence_read_.store((read + 1U) % evidence_.size(), std::memory_order_release);
    return streaming::StreamStatus::ok;
}

std::uint64_t IntentDrivenNeuralSource::dropped_applied_control_count() const noexcept
{
    return evidence_dropped_.load(std::memory_order_acquire);
}

std::uint64_t IntentDrivenNeuralSource::drift_fingerprint() const noexcept
{
    return generator_.drift_fingerprint();
}

void IntentDrivenNeuralSource::reset_generator() noexcept
{
    generator_.reset();
    auto mode = control_mode_.load(std::memory_order_acquire);
    while (mode == ControlMode::binding)
    {
        control_mode_.wait(ControlMode::binding, std::memory_order_acquire);
        mode = control_mode_.load(std::memory_order_acquire);
    }
    if (mode != ControlMode::bound)
    {
        manual_source_->reset();
        control_mode_.store(ControlMode::undecided, std::memory_order_release);
    }
    evidence_read_.store(0, std::memory_order_relaxed);
    evidence_write_.store(0, std::memory_order_relaxed);
    evidence_dropped_.store(0, std::memory_order_relaxed);
    manual_sequence_.store(0, std::memory_order_relaxed);
    generated_frame_sequence_ = 0;
}

} // namespace neurale::devices::simulation
