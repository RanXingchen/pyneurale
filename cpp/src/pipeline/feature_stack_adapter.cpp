/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "feature_stack_adapter.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "adapter_support.h"
#include "feature_adapter_stream.h"

#include <neurale/features/online.h>
#include <neurale/signal/iir.h>
#include <neurale/signal/windows.h>
#include <neurale/streaming/frame_validation.h>

namespace neurale::pipeline
{
namespace
{

constexpr adapter_support::Checked kChecked{"feature stack"};

template <typename T> [[nodiscard]] T next_identifier(T value, const char* message)
{
    if (value == std::numeric_limits<T>::max())
    {
        throw std::invalid_argument(message);
    }
    return value + 1;
}

[[nodiscard]] std::size_t exact_samples(std::uint64_t duration_ns, streaming::RationalRate rate,
                                        const char* name)
{
    auto duration = duration_ns;
    auto numerator = rate.numerator;
    auto nanoseconds_per_second = std::uint64_t{1'000'000'000};
    auto denominator = rate.denominator;
    const auto cancel = [](auto& left, auto& right)
    {
        const auto divisor = std::gcd(left, right);
        left /= divisor;
        right /= divisor;
    };
    cancel(duration, nanoseconds_per_second);
    cancel(numerator, nanoseconds_per_second);
    cancel(duration, denominator);
    cancel(numerator, denominator);
    if (nanoseconds_per_second != 1 || denominator != 1)
    {
        throw std::invalid_argument(std::string{"feature stack "} + name +
                                    " does not resolve to an integer number of samples at the "
                                    "input sample rate");
    }
    if (numerator != 0 && duration > std::numeric_limits<std::uint64_t>::max() / numerator)
    {
        throw std::overflow_error(std::string{"feature stack "} + name +
                                  " sample count overflows uint64");
    }
    const auto samples = duration * numerator;
    if (samples == 0 || samples > std::numeric_limits<std::size_t>::max())
    {
        throw std::invalid_argument(std::string{"feature stack "} + name +
                                    " resolves outside the supported sample-count range");
    }
    return static_cast<std::size_t>(samples);
}

[[nodiscard]] std::size_t next_power_of_two(std::size_t value)
{
    auto result = std::size_t{1};
    while (result < value)
    {
        if (result > std::numeric_limits<std::size_t>::max() / 2)
        {
            throw std::overflow_error("feature stack FFT length overflows size_t");
        }
        result *= 2;
    }
    return result;
}

[[nodiscard]] bool valid_filter_design(std::string_view kind, std::size_t order,
                                       double passband_ripple_db,
                                       double stopband_attenuation_db) noexcept
{
    const auto known_kind = kind == "butterworth" || kind == "bessel" || kind == "elliptic";
    if (!known_kind || order == 0 || (kind == "bessel" && order > 16))
    {
        return false;
    }
    return kind != "elliptic" ||
           (std::isfinite(passband_ripple_db) && passband_ripple_db > 0.0 &&
            std::isfinite(stopband_attenuation_db) && stopband_attenuation_db > 0.0 &&
            passband_ripple_db < stopband_attenuation_db);
}

void require_unique_names(std::vector<std::string> names, const char* message)
{
    if (std::any_of(names.begin(), names.end(), [](const auto& name) { return name.empty(); }))
    {
        throw std::invalid_argument(message);
    }
    std::sort(names.begin(), names.end());
    if (std::adjacent_find(names.begin(), names.end()) != names.end())
    {
        throw std::invalid_argument(message);
    }
}

[[nodiscard]] streaming::UnitDescriptor amplitude_unit(streaming::UnitId id,
                                                       streaming::PhysicalUnit unit)
{
    switch (unit)
    {
    case streaming::PhysicalUnit::volts:
        return {id, "V", "volts"};
    case streaming::PhysicalUnit::amperes:
        return {id, "A", "amperes"};
    case streaming::PhysicalUnit::dimensionless:
        return {id, "1", "dimensionless"};
    case streaming::PhysicalUnit::unspecified:
        break;
    }
    throw std::invalid_argument("feature stack input amplitude unit must be specified");
}

[[nodiscard]] streaming::UnitDescriptor power_unit(streaming::UnitId id,
                                                   streaming::PhysicalUnit unit)
{
    switch (unit)
    {
    case streaming::PhysicalUnit::volts:
        return {id, "V^2", "volts squared"};
    case streaming::PhysicalUnit::amperes:
        return {id, "A^2", "amperes squared"};
    case streaming::PhysicalUnit::dimensionless:
        return {id, "1", "dimensionless"};
    case streaming::PhysicalUnit::unspecified:
        break;
    }
    throw std::invalid_argument("feature stack input amplitude unit must be specified");
}

using UnitFactory = streaming::UnitDescriptor (*)(streaming::UnitId, streaming::PhysicalUnit);

[[nodiscard]] streaming::UnitDescriptor
derived_unit(const std::vector<streaming::UnitDescriptor>& units,
             streaming::PhysicalUnit physical_unit, UnitFactory factory,
             const char* overflow_message)
{
    const auto expected = factory(1, physical_unit);
    for (const auto& unit : units)
    {
        if (unit.symbol == expected.symbol)
        {
            return unit;
        }
    }

    auto maximum_id = streaming::UnitId{};
    for (const auto& unit : units)
    {
        maximum_id = std::max(maximum_id, unit.id);
    }
    if (maximum_id == std::numeric_limits<streaming::UnitId>::max())
    {
        throw std::invalid_argument(overflow_message);
    }
    return factory(maximum_id + 1, physical_unit);
}

void merge_unit(std::vector<streaming::UnitDescriptor>& units,
                const streaming::UnitDescriptor& value)
{
    const auto by_id = std::find_if(units.begin(), units.end(),
                                    [&](const auto& unit) { return unit.id == value.id; });
    if (by_id != units.end())
    {
        if (!adapter_support::equal_unit(*by_id, value))
        {
            throw std::invalid_argument("feature stack unit id conflicts with the registry");
        }
        return;
    }
    if (std::any_of(units.begin(), units.end(),
                    [&](const auto& unit) { return unit.symbol == value.symbol; }))
    {
        throw std::invalid_argument("feature stack unit symbol uses another id");
    }
    units.push_back(value);
}

[[nodiscard]] std::vector<std::size_t> make_band_bins(std::span<const MultitaperBand> bands,
                                                      double fs, std::size_t fft_length)
{
    const auto n_bins = fft_length / 2 + 1;
    std::vector<std::size_t> result;
    result.reserve(kChecked.checked_multiply(bands.size(), 2));
    for (const auto& band : bands)
    {
        auto begin = n_bins;
        auto end = n_bins;
        for (std::size_t bin = 0; bin < n_bins; ++bin)
        {
            const auto frequency = static_cast<double>(bin) * fs / static_cast<double>(fft_length);
            if (begin == n_bins && frequency >= band.low_hz && frequency < band.high_hz)
            {
                begin = bin;
            }
            if (begin != n_bins && frequency >= band.high_hz)
            {
                end = bin;
                break;
            }
        }
        if (begin == n_bins)
        {
            throw std::invalid_argument("feature stack PMTM band contains no FFT bins");
        }
        result.push_back(begin);
        result.push_back(end);
    }
    return result;
}

using Processor = std::variant<std::unique_ptr<features::BandpowerProcessor>,
                               std::unique_ptr<features::HilbertEnvelopeProcessor>,
                               std::unique_ptr<features::LmpProcessor>>;

struct RuntimeBranch
{
    Processor processor;
    std::vector<double> output;
    std::size_t n_features{};
    std::size_t workspace_bytes{};

    [[nodiscard]] std::size_t output_count(std::size_t input_samples) const
    {
        return std::visit([&](const auto& value) { return value->output_count(input_samples); },
                          processor);
    }

    void process(std::span<const double> input, std::size_t input_samples,
                 std::size_t n_observations)
    {
        std::visit(
            [&](auto& value)
            { value->process(input, input_samples, {output.data(), n_observations * n_features}); },
            processor);
    }

    void reset() noexcept
    {
        std::visit([](auto& value) { value->reset(); }, processor);
    }
};

} // namespace

struct FeatureStackAdapter::Impl : adapter_support::FeatureStreamState
{
    explicit Impl(FeatureStackAdapterConfig adapter_config) : config(std::move(adapter_config)) {}

    void clear_stream_state() noexcept
    {
        for (auto& branch : branches)
        {
            branch.reset();
        }
        clear_timing();
    }

    FeatureStackAdapterConfig config;
    std::vector<RuntimeBranch> branches;
    std::size_t workspace_bytes{};
    std::size_t window_samples{};
    std::size_t shift_samples{};
    streaming::SchemaId output_schema_id{};
    streaming::SignalId output_signal_id{};
    streaming::FeatureSetId feature_set_id{};
};

FeatureStackAdapter::FeatureStackAdapter(FeatureStackAdapterConfig config)
    : impl_(std::make_unique<Impl>(std::move(config)))
{
    const auto& value = impl_->config;
    if (value.algorithm_version.empty() || value.window_ns == 0 || value.update_interval_ns == 0 ||
        value.update_interval_ns > value.window_ns || value.branches.empty() ||
        value.branches.size() > 3)
    {
        throw std::invalid_argument("invalid feature stack configuration");
    }
    std::array<bool, 3> seen{};
    for (const auto& branch : value.branches)
    {
        if (seen[branch.index()])
        {
            throw std::invalid_argument("each feature stack branch type may appear at most once");
        }
        seen[branch.index()] = true;
        std::visit(
            [&](const auto& item)
            {
                using T = std::decay_t<decltype(item)>;
                if constexpr (std::is_same_v<T, MultitaperBandpowerBranchConfig>)
                {
                    const auto maximum_tapers_value = std::floor(2.0 * item.time_bandwidth - 1.0);
                    if (item.bands.empty() || !std::isfinite(item.time_bandwidth) ||
                        item.time_bandwidth <= 0.0 || maximum_tapers_value < 2.0 ||
                        (item.n_tapers != 0 &&
                         (item.n_tapers < 2 ||
                          item.n_tapers > static_cast<std::size_t>(maximum_tapers_value))))
                    {
                        throw std::invalid_argument("invalid feature stack PMTM branch");
                    }
                    std::vector<std::string> band_names;
                    for (const auto& band : item.bands)
                    {
                        if (band.name.empty() || !std::isfinite(band.low_hz) ||
                            !std::isfinite(band.high_hz) || band.low_hz < 0.0 ||
                            band.high_hz <= band.low_hz)
                        {
                            throw std::invalid_argument("invalid feature stack PMTM band");
                        }
                        band_names.push_back(band.name);
                    }
                    require_unique_names(std::move(band_names),
                                         "feature stack PMTM band names must be unique");
                }
                else if constexpr (std::is_same_v<T, HilbertEnvelopeBranchConfig>)
                {
                    if (item.bands.empty() ||
                        !valid_filter_design(item.filter_kind, item.filter_order,
                                             item.passband_ripple_db, item.stopband_attenuation_db))
                    {
                        throw std::invalid_argument("invalid feature stack Hilbert branch");
                    }
                    std::vector<std::string> band_names;
                    for (const auto& band : item.bands)
                    {
                        if (band.name.empty() || !std::isfinite(band.low_hz) ||
                            !std::isfinite(band.high_hz) || band.low_hz <= 0.0 ||
                            band.high_hz <= band.low_hz)
                        {
                            throw std::invalid_argument("invalid feature stack Hilbert band");
                        }
                        band_names.push_back(band.name);
                    }
                    require_unique_names(std::move(band_names),
                                         "feature stack Hilbert band names must be unique");
                }
                else
                {
                    if (!std::isfinite(item.cutoff_hz) || item.cutoff_hz <= 0.0 ||
                        !valid_filter_design(item.filter_kind, item.filter_order,
                                             item.passband_ripple_db, item.stopband_attenuation_db))
                    {
                        throw std::invalid_argument("invalid feature stack LMP branch");
                    }
                }
            },
            branch);
    }
}

FeatureStackAdapter::~FeatureStackAdapter() = default;

streaming::PreparedProcessorContract
FeatureStackAdapter::prepare(const streaming::ProcessorPrepareContext& context)
{
    const auto signals = context.input_schema.signals();
    if (signals.size() != 1)
    {
        throw std::invalid_argument("feature stack requires exactly one input signal");
    }
    const auto& input = signals.front();
    if (input.kind != streaming::SignalKind::sampled ||
        input.dtype != streaming::SignalDType::float64 ||
        input.layout != streaming::SignalLayout::sample_major)
    {
        throw std::invalid_argument("feature stack requires a sample-major float64 sampled signal");
    }
    if (context.max_process_outputs < 1 || context.available_frame_pool_leases < 1)
    {
        throw std::invalid_argument(
            "feature stack requires one process output and one frame lease");
    }
    if (impl_->input_schema != nullptr)
    {
        if (!impl_->input_schema->equivalent(context.input_schema) ||
            impl_->input_schema->signals().front().channel_names != input.channel_names)
        {
            throw std::invalid_argument("feature stack input schema cannot change after prepare");
        }
        impl_->clear_stream_state();
        return {
            .accepted_input_schema = context.input_schema.clone(),
            .output_schema = impl_->output_schema->clone(),
            .max_process_outputs_per_input = 1,
            .max_flush_outputs = 0,
            .can_forward_input = false,
            .required_resources = {.workspace_bytes = impl_->workspace_bytes,
                                   .frame_pool_leases = 1},
        };
    }

    impl_->window_samples = exact_samples(impl_->config.window_ns, input.fs, "window_seconds");
    const auto fft_length = next_power_of_two(impl_->window_samples);
    impl_->shift_samples =
        exact_samples(impl_->config.update_interval_ns, input.fs, "update_interval_seconds");
    if (impl_->shift_samples > impl_->window_samples)
    {
        throw std::invalid_argument(
            "feature stack update_interval_seconds must not exceed window_seconds");
    }
    if (impl_->config.window_ns % 2 != 0)
    {
        throw std::invalid_argument("feature stack window center must resolve to integral ns");
    }
    impl_->window_center_ns = impl_->config.window_ns / 2;
    impl_->shift_ns = impl_->config.update_interval_ns;
    impl_->max_input_samples = input.max_block_samples;
    impl_->n_channels = input.n_channels;

    const auto fs =
        static_cast<double>(input.fs.numerator) / static_cast<double>(input.fs.denominator);
    if (!std::isfinite(fs) || fs <= 0.0)
    {
        throw std::invalid_argument("feature stack input sample rate must be finite and positive");
    }
    auto channel_names = input.channel_names;
    if (channel_names.empty())
    {
        channel_names.reserve(input.n_channels);
        for (std::size_t channel = 0; channel < input.n_channels; ++channel)
        {
            channel_names.push_back("channel_" + std::to_string(channel));
        }
    }

    impl_->output_schema_id = next_identifier(context.input_schema.id(),
                                              "feature stack cannot allocate an output schema id");
    impl_->output_signal_id =
        next_identifier(input.id, "feature stack cannot allocate an output signal id");
    auto maximum_feature_set_id = streaming::FeatureSetId{};
    for (const auto& descriptor : context.input_schema.feature_sets().descriptors())
    {
        maximum_feature_set_id = std::max(maximum_feature_set_id, descriptor.id);
    }
    impl_->feature_set_id =
        next_identifier(maximum_feature_set_id, "feature stack cannot allocate a feature-set id");

    std::vector<streaming::UnitDescriptor> units{context.input_schema.units().units().begin(),
                                                 context.input_schema.units().units().end()};
    std::vector<std::string> feature_names;
    std::vector<streaming::UnitId> unit_ids;
    std::vector<RuntimeBranch> branches;
    branches.reserve(impl_->config.branches.size());
    std::size_t branch_workspace_bytes{};

    for (const auto& configured : impl_->config.branches)
    {
        std::visit(
            [&](const auto& item)
            {
                using T = std::decay_t<decltype(item)>;
                RuntimeBranch branch;
                if constexpr (std::is_same_v<T, MultitaperBandpowerBranchConfig>)
                {
                    const auto maximum_tapers =
                        static_cast<std::size_t>(std::floor(2.0 * item.time_bandwidth - 1.0));
                    const auto n_tapers = item.n_tapers == 0 ? maximum_tapers : item.n_tapers;
                    if (item.time_bandwidth >= static_cast<double>(impl_->window_samples) / 2.0 ||
                        n_tapers > impl_->window_samples)
                    {
                        throw std::invalid_argument(
                            "feature stack PMTM branch does not support the prepared window");
                    }
                    const auto nyquist = fs / 2.0;
                    if (std::any_of(item.bands.begin(), item.bands.end(),
                                    [nyquist](const auto& band) { return band.high_hz > nyquist; }))
                    {
                        throw std::invalid_argument("feature stack PMTM bands exceed Nyquist");
                    }
                    auto band_bins = make_band_bins(item.bands, fs, fft_length);
                    std::vector<double> tapers(
                        kChecked.checked_multiply(impl_->window_samples, n_tapers));
                    std::vector<double> concentrations(n_tapers);
                    signal::multitap(impl_->window_samples, item.time_bandwidth, n_tapers, tapers,
                                     concentrations);
                    branch.n_features =
                        kChecked.checked_multiply(item.bands.size(), input.n_channels);
                    branch.processor = std::make_unique<features::BandpowerProcessor>(
                        impl_->window_samples, impl_->shift_samples, input.n_channels, fft_length,
                        fs, tapers, n_tapers, concentrations, item.weighting, band_bins,
                        fs / static_cast<double>(fft_length), item.detrend, item.backend);
                    const auto unit = derived_unit(units, input.physical_unit, power_unit,
                                                   "feature stack cannot allocate a power unit id");
                    merge_unit(units, unit);
                    for (const auto& band : item.bands)
                    {
                        for (const auto& channel : channel_names)
                        {
                            feature_names.push_back("pmtm:" + band.name + ":" + channel);
                            unit_ids.push_back(unit.id);
                        }
                    }
                    const auto spectrum_bins = fft_length / 2 + 1;
                    const auto retained = kChecked.checked_multiply(
                        4, kChecked.checked_add(
                               kChecked.checked_multiply(impl_->window_samples, input.n_channels),
                               kChecked.checked_add(
                                   kChecked.checked_multiply(impl_->window_samples, n_tapers),
                                   kChecked.checked_add(
                                       kChecked.checked_multiply(
                                           spectrum_bins,
                                           kChecked.checked_add(n_tapers, input.n_channels)),
                                       fft_length))));
                    branch.workspace_bytes = kChecked.checked_multiply(retained, sizeof(double));
                }
                else if constexpr (std::is_same_v<T, HilbertEnvelopeBranchConfig>)
                {
                    std::vector<double> sos;
                    std::size_t sos_sections{};
                    for (const auto& band : item.bands)
                    {
                        const std::array cutoff{band.low_hz, band.high_hz};
                        auto designed = signal::iir_sos(item.filter_kind, item.filter_order, cutoff,
                                                        "bandpass", fs, item.passband_ripple_db,
                                                        item.stopband_attenuation_db);
                        if (designed.empty() || designed.size() % 6 != 0)
                        {
                            throw std::runtime_error(
                                "feature stack Hilbert filter design returned invalid SOS");
                        }
                        const auto designed_sections = designed.size() / 6;
                        if (sos_sections == 0)
                        {
                            sos_sections = designed_sections;
                        }
                        else if (sos_sections != designed_sections)
                        {
                            throw std::runtime_error(
                                "feature stack Hilbert bands produced unequal SOS sections");
                        }
                        sos.insert(sos.end(), designed.begin(), designed.end());
                    }
                    branch.n_features =
                        kChecked.checked_multiply(item.bands.size(), input.n_channels);
                    branch.processor = std::make_unique<features::HilbertEnvelopeProcessor>(
                        impl_->window_samples, impl_->shift_samples, input.n_channels, sos,
                        item.bands.size(), sos_sections, fft_length,
                        signal::SpectralBackend::automatic);
                    const auto unit =
                        derived_unit(units, input.physical_unit, amplitude_unit,
                                     "feature stack cannot allocate an amplitude unit id");
                    merge_unit(units, unit);
                    for (const auto& band : item.bands)
                    {
                        for (const auto& channel : channel_names)
                        {
                            feature_names.push_back("hilbert:" + band.name + ":" + channel);
                            unit_ids.push_back(unit.id);
                        }
                    }
                    const auto window_features =
                        kChecked.checked_multiply(impl_->window_samples, branch.n_features);
                    const auto window_channels =
                        kChecked.checked_multiply(impl_->window_samples, input.n_channels);
                    const auto filter_state = kChecked.checked_multiply(
                        kChecked.checked_multiply(
                            kChecked.checked_multiply(item.bands.size(), sos_sections), 2),
                        input.n_channels);
                    const auto analytic = kChecked.checked_multiply(
                        kChecked.checked_multiply(2, fft_length), input.n_channels);
                    const auto retained = kChecked.checked_add(
                        kChecked.checked_multiply(2, window_features),
                        kChecked.checked_add(
                            window_channels,
                            kChecked.checked_add(
                                filter_state, kChecked.checked_add(
                                                  sos.size(), kChecked.checked_add(
                                                                  branch.n_features, analytic)))));
                    branch.workspace_bytes = kChecked.checked_multiply(retained, sizeof(double));
                }
                else
                {
                    const std::array cutoff{item.cutoff_hz};
                    auto sos =
                        signal::iir_sos(item.filter_kind, item.filter_order, cutoff, "lowpass", fs,
                                        item.passband_ripple_db, item.stopband_attenuation_db);
                    if (sos.empty() || sos.size() % 6 != 0)
                    {
                        throw std::runtime_error(
                            "feature stack LMP filter design returned invalid SOS");
                    }
                    const auto sos_sections = sos.size() / 6;
                    branch.n_features = input.n_channels;
                    branch.processor = std::make_unique<features::LmpProcessor>(
                        impl_->window_samples, impl_->shift_samples, input.n_channels, sos,
                        sos_sections);
                    const auto unit =
                        derived_unit(units, input.physical_unit, amplitude_unit,
                                     "feature stack cannot allocate an amplitude unit id");
                    merge_unit(units, unit);
                    for (const auto& name : channel_names)
                    {
                        feature_names.push_back("lmp:" + name);
                        unit_ids.push_back(unit.id);
                    }
                    const auto retained = kChecked.checked_add(
                        kChecked.checked_multiply(impl_->window_samples, input.n_channels),
                        kChecked.checked_add(
                            kChecked.checked_multiply(sos_sections, 5),
                            kChecked.checked_add(
                                kChecked.checked_multiply(
                                    kChecked.checked_multiply(sos_sections, 2), input.n_channels),
                                kChecked.checked_multiply(2, input.n_channels))));
                    branch.workspace_bytes = kChecked.checked_multiply(retained, sizeof(double));
                }
                branches.push_back(std::move(branch));
            },
            configured);
    }

    auto sorted_names = feature_names;
    std::sort(sorted_names.begin(), sorted_names.end());
    if (std::adjacent_find(sorted_names.begin(), sorted_names.end()) != sorted_names.end())
    {
        throw std::invalid_argument("feature stack feature names must be unique");
    }
    impl_->n_features = feature_names.size();
    const auto output = adapter_support::feature_output(kChecked, input, impl_->output_signal_id,
                                                        impl_->feature_set_id, impl_->n_features,
                                                        impl_->shift_samples);
    impl_->max_output_observations = output.max_observations;
    for (auto& branch : branches)
    {
        branch.output.resize(
            kChecked.checked_multiply(impl_->max_output_observations, branch.n_features));
        branch.workspace_bytes =
            kChecked.checked_add(branch.workspace_bytes,
                                 kChecked.checked_multiply(branch.output.size(), sizeof(double)));
        branch_workspace_bytes =
            kChecked.checked_add(branch_workspace_bytes, branch.workspace_bytes);
    }
    impl_->output_workspace.resize(
        kChecked.checked_multiply(impl_->max_output_observations, impl_->n_features));

    std::vector<streaming::FeatureSetDescriptor> descriptors{
        context.input_schema.feature_sets().descriptors().begin(),
        context.input_schema.feature_sets().descriptors().end()};
    descriptors.push_back(streaming::FeatureSetDescriptor{
        .id = impl_->feature_set_id,
        .feature_names = feature_names,
        .unit_ids = unit_ids,
        .source_stream_id = input.id,
        .source_stream = "signal-" + std::to_string(input.id),
        .algorithm_name = "feature-stack",
        .algorithm_version = impl_->config.algorithm_version,
        .window_length_ns = impl_->config.window_ns,
        .shift_ns = impl_->shift_ns,
        .timestamp_reference = streaming::FeatureTimestampReference::window_center,
    });
    const std::array output_signals{output.signal};
    auto output_schema =
        streaming::StreamSchema{impl_->output_schema_id, output_signals, descriptors, units};

    impl_->workspace_bytes = kChecked.checked_add(
        branch_workspace_bytes,
        kChecked.checked_multiply(impl_->output_workspace.size(), sizeof(double)));
    impl_->branches = std::move(branches);
    impl_->validator = std::make_unique<streaming::FrameValidator>(context.input_schema);
    impl_->input_schema = std::make_unique<streaming::StreamSchema>(context.input_schema.clone());
    impl_->output_schema = std::make_unique<streaming::StreamSchema>(output_schema.clone());
    impl_->clear_stream_state();
    return {
        .accepted_input_schema = context.input_schema.clone(),
        .output_schema = output_schema.clone(),
        .max_process_outputs_per_input = 1,
        .max_flush_outputs = 0,
        .can_forward_input = false,
        .required_resources = {.workspace_bytes = impl_->workspace_bytes, .frame_pool_leases = 1},
    };
}

streaming::StreamStatus FeatureStackAdapter::process(streaming::FrameBorrow& frame,
                                                     streaming::FrameEmitter& emitter) noexcept
{
    if (impl_->validator == nullptr || impl_->output_schema == nullptr || impl_->branches.empty())
    {
        return streaming::StreamStatus::invalid_state;
    }
    if (impl_->validator->validate(frame.view()) != streaming::FrameValidationError::none)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    const auto& block = frame.blocks().front();
    if (block.n_samples > impl_->max_input_samples)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    if (!impl_->timing_anchored)
    {
        if (!adapter_support::device_tick_host_time(block, impl_->stream_time_anchor_ns))
        {
            return streaming::StreamStatus::invalid_frame;
        }
        impl_->timing_anchored = true;
    }

    std::size_t n_observations{};
    try
    {
        n_observations = impl_->branches.front().output_count(block.n_samples);
        if (n_observations > impl_->max_output_observations)
        {
            return streaming::StreamStatus::output_limit;
        }
        if (std::any_of(impl_->branches.begin() + 1, impl_->branches.end(), [&](const auto& branch)
                        { return branch.output_count(block.n_samples) != n_observations; }))
        {
            return streaming::StreamStatus::processor_failure;
        }
        const auto* input =
            reinterpret_cast<const double*>(frame.payload().data() + block.payload_offset);
        const auto input_span = std::span<const double>{
            input, static_cast<std::size_t>(block.n_samples) * impl_->n_channels};
        for (auto& branch : impl_->branches)
        {
            branch.process(input_span, block.n_samples, n_observations);
        }
        for (std::size_t row = 0; row < n_observations; ++row)
        {
            std::size_t column{};
            for (const auto& branch : impl_->branches)
            {
                std::memcpy(impl_->output_workspace.data() + row * impl_->n_features + column,
                            branch.output.data() + row * branch.n_features,
                            branch.n_features * sizeof(double));
                column += branch.n_features;
            }
        }
    }
    catch (...)
    {
        return streaming::StreamStatus::processor_failure;
    }
    if (n_observations == 0)
    {
        return streaming::StreamStatus::ok;
    }

    if (impl_->emitted_observations != 0 &&
        impl_->shift_ns > std::numeric_limits<std::uint64_t>::max() / impl_->emitted_observations)
    {
        return streaming::StreamStatus::invalid_frame;
    }
    const auto offset = impl_->emitted_observations * impl_->shift_ns;
    if (impl_->window_center_ns > std::numeric_limits<std::uint64_t>::max() - offset ||
        impl_->stream_time_anchor_ns >
            std::numeric_limits<std::uint64_t>::max() - (offset + impl_->window_center_ns) ||
        n_observations > std::numeric_limits<std::uint64_t>::max() - impl_->emitted_observations)
    {
        return streaming::StreamStatus::invalid_frame;
    }

    streaming::FrameBorrow output{};
    auto status = emitter.try_acquire(output);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    output->header() = frame.header();
    output->header().schema_id = impl_->output_schema_id;
    const auto payload_bytes = n_observations * impl_->n_features * sizeof(double);
    output->block_storage()[0] = block;
    output->block_storage()[0].sample_idx_start = impl_->emitted_observations;
    output->block_storage()[0].device_tick_start = 0;
    output->block_storage()[0].observation_time_start_ns =
        impl_->stream_time_anchor_ns + offset + impl_->window_center_ns;
    output->block_storage()[0].payload_offset = 0;
    output->block_storage()[0].payload_byte_count = payload_bytes;
    output->block_storage()[0].signal_id = impl_->output_signal_id;
    output->block_storage()[0].n_samples = static_cast<std::uint32_t>(n_observations);
    std::memcpy(output->payload_storage().data(), impl_->output_workspace.data(), payload_bytes);
    status = output->set_used_sizes(1, payload_bytes);
    if (status != streaming::StreamStatus::ok)
    {
        return status;
    }
    status = emitter.publish_acquired();
    if (status == streaming::StreamStatus::ok)
    {
        impl_->emitted_observations += n_observations;
    }
    return status;
}

streaming::StreamStatus
FeatureStackAdapter::handle_discontinuity(const streaming::Discontinuity&) noexcept
{
    if (impl_->branches.empty())
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus FeatureStackAdapter::flush(streaming::FrameEmitter&) noexcept
{
    if (impl_->branches.empty())
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

streaming::StreamStatus FeatureStackAdapter::reset() noexcept
{
    if (impl_->branches.empty())
    {
        return streaming::StreamStatus::invalid_state;
    }
    impl_->clear_stream_state();
    return streaming::StreamStatus::ok;
}

} // namespace neurale::pipeline
