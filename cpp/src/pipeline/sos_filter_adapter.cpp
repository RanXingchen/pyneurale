/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "sos_filter_adapter.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <neurale/signal/iir.h>
#include <neurale/streaming/frame_validation.h>

#include "adapter_support.h"
#include "inplace_adapter_stream.h"
#include "sos_realtime.h"

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"SOS"};

[[nodiscard]] std::size_t checked_workspace_bytes(std::size_t n_coefs, std::size_t n_sections,
                                                  std::size_t n_channels)
{
    constexpr auto state_rows = std::size_t{3};
    if (n_sections != 0 &&
        n_channels > std::numeric_limits<std::size_t>::max() / (n_sections * state_rows))
    {
        throw std::overflow_error("SOS adapter state size overflows size_t");
    }
    const auto n_states = n_sections * state_rows * n_channels;
    if (n_states > std::numeric_limits<std::size_t>::max() - n_coefs ||
        n_states + n_coefs > std::numeric_limits<std::size_t>::max() / sizeof(double))
    {
        throw std::overflow_error("SOS adapter workspace size overflows size_t");
    }
    return (n_states + n_coefs) * sizeof(double);
}

} // namespace

struct SosFilterAdapter::Impl : adapter_support::InplaceStreamState
{
    explicit Impl(std::span<const double> coefs) : sos(coefs.begin(), coefs.end()) {}
    explicit Impl(FilterDesign value) : design(std::move(value)) {}
    explicit Impl(LineNoiseFilterDesign value) : line_noise_design(std::move(value)) {}

    std::vector<double> sos;
    std::optional<FilterDesign> design;
    std::optional<LineNoiseFilterDesign> line_noise_design;
    std::unique_ptr<signal::detail::SosRealtimeProcessor> processor;
};

SosFilterAdapter::SosFilterAdapter(std::span<const double> normalized_sos)
{
    if (normalized_sos.empty() || normalized_sos.size() % 6 != 0)
    {
        throw std::invalid_argument("SOS adapter coefficients must have shape (n_sections, 6)");
    }
    for (std::size_t i = 0; i < normalized_sos.size(); ++i)
    {
        if (!std::isfinite(normalized_sos[i]))
        {
            throw std::invalid_argument("SOS adapter coefficients must be finite");
        }
        if (i % 6 == 3 && normalized_sos[i] != 1.0)
        {
            throw std::invalid_argument("SOS adapter denominator must be normalized");
        }
    }
    impl_ = std::make_unique<Impl>(normalized_sos);
}

SosFilterAdapter::SosFilterAdapter(FilterDesign design)
{
    if (design.type != "lowpass" && design.type != "highpass" && design.type != "bandpass" &&
        design.type != "bandstop")
    {
        throw std::invalid_argument("filter type must be lowpass, highpass, bandpass, or bandstop");
    }
    const auto expected_cutoffs =
        design.type == "bandpass" || design.type == "bandstop" ? std::size_t{2} : std::size_t{1};
    if (design.cutoff_hz.size() != expected_cutoffs || design.order == 0)
    {
        throw std::invalid_argument("filter design has invalid cutoff or order");
    }
    impl_ = std::make_unique<Impl>(std::move(design));
}

SosFilterAdapter::SosFilterAdapter(LineNoiseFilterDesign design)
{
    if (!std::isfinite(design.frequency_hz) || design.frequency_hz <= 0.0 ||
        !std::isfinite(design.bandwidth_hz) || design.bandwidth_hz <= 0.0 ||
        design.bandwidth_hz >= 2.0 * design.frequency_hz || design.harmonics == 0 ||
        design.order == 0)
    {
        throw std::invalid_argument("line-noise filter design is invalid");
    }
    impl_ = std::make_unique<Impl>(std::move(design));
}

SosFilterAdapter::~SosFilterAdapter() = default;

streaming::PreparedProcessorContract
SosFilterAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto& signal_schema =
        adapter_support::sampled_inplace_input(kChecked, context, impl_->prepared_schema.get());
    // Not shared: only the SOS kernel takes an `int` channel count (the MKL
    // BLAS df2t path), so only this adapter has an upper bound to enforce.
    if (signal_schema.n_channels > static_cast<std::uint32_t>(std::numeric_limits<int>::max()))
    {
        throw std::invalid_argument("SOS adapter channel count exceeds the native kernel limit");
    }

    if (impl_->processor == nullptr)
    {
        if (impl_->design.has_value())
        {
            const auto& design = *impl_->design;
            const auto fs = static_cast<double>(signal_schema.fs.numerator) /
                            static_cast<double>(signal_schema.fs.denominator);
            impl_->sos =
                signal::iir_sos(design.kind, design.order, design.cutoff_hz, design.type, fs,
                                design.passband_ripple_db, design.stopband_attenuation_db);
            if (impl_->sos.empty() || impl_->sos.size() % 6 != 0)
            {
                throw std::runtime_error("filter design returned invalid SOS coefficients");
            }
        }
        else if (impl_->line_noise_design.has_value())
        {
            const auto& design = *impl_->line_noise_design;
            const auto fs = static_cast<double>(signal_schema.fs.numerator) /
                            static_cast<double>(signal_schema.fs.denominator);
            const auto nyquist = fs / 2.0;
            const auto half_width = design.bandwidth_hz / 2.0;
            if (design.frequency_hz + half_width >= nyquist)
            {
                throw std::invalid_argument(
                    "line-noise fundamental notch must be strictly below Nyquist");
            }
            const auto harmonic_limit = (nyquist - half_width) / design.frequency_hz;
            const auto count = harmonic_limit <= static_cast<double>(design.harmonics)
                                   ? static_cast<std::size_t>(std::ceil(harmonic_limit)) - 1
                                   : design.harmonics;
            for (std::size_t harmonic = 1; harmonic <= count; ++harmonic)
            {
                const auto center = static_cast<double>(harmonic) * design.frequency_hz;
                const std::array cutoffs{center - half_width, center + half_width};
                auto sections =
                    signal::iir_sos("butterworth", design.order, cutoffs, "bandstop", fs, 0.0, 0.0);
                impl_->sos.insert(impl_->sos.end(), sections.begin(), sections.end());
            }
            if (impl_->sos.empty() || impl_->sos.size() % 6 != 0)
            {
                throw std::runtime_error(
                    "line-noise filter design returned invalid SOS coefficients");
            }
        }
        impl_->processor = std::make_unique<signal::detail::SosRealtimeProcessor>(
            impl_->sos, signal_schema.n_channels, std::span<const double>{});
        impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
        impl_->prepared_schema =
            std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
        impl_->n_channels = signal_schema.n_channels;
    }
    else
    {
        impl_->processor->reset();
    }

    const auto workspace_bytes =
        checked_workspace_bytes(impl_->sos.size(), impl_->sos.size() / 6, signal_schema.n_channels);
    return adapter_support::forwarding_contract(context.input_schema, workspace_bytes);
}

streaming::StreamStatus SosFilterAdapter::process(streaming::FrameBorrow& frame,
                                                  streaming::FrameEmitter& emitter) noexcept
{
    return adapter_support::process_inplace_frame(
        *impl_, impl_->processor != nullptr, frame, emitter,
        [this](std::span<double> samples, std::size_t n_samples)
        { impl_->processor->process(samples, n_samples); });
}

streaming::StreamStatus
SosFilterAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->processor->reset();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus SosFilterAdapter::flush(streaming::FrameEmitter&) noexcept
{
    return impl_->processor == nullptr ? streaming::StreamStatus::invalid_state
                                       : streaming::StreamStatus::ok;
}

streaming::StreamStatus SosFilterAdapter::reset() noexcept
{
    if (impl_->processor == nullptr)
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->processor->reset();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
