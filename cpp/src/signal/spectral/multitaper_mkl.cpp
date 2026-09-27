/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "mkl_utils.h"
#include "multitaper_internal.h"

#include <mkl.h>

#include <algorithm>
#include <complex>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace neurale::signal::detail
{
namespace
{

std::size_t worker_count_for(std::size_t n_channels)
{
    if (n_channels < 16)
    {
        return 1;
    }
    const auto hardware = std::max(1u, std::thread::hardware_concurrency());
    return std::min<std::size_t>({n_channels, hardware, 8});
}

class MklChannelBlock
{
  public:
    MklChannelBlock(const MultitaperPsdSpec& spec, std::size_t first_channel,
                    std::size_t last_channel)
        : first_channel_(first_channel), last_channel_(last_channel),
          n_channels_(last_channel - first_channel), spectra_(spec.n_tapers * spec.spectrum_bins),
          combined_(spec.spectrum_bins), next_(spec.spectrum_bins), weight_sum_(spec.spectrum_bins),
          channel_power_(n_channels_), tapered_(n_channels_ * spec.n_tapers * spec.fft_length),
          fft_output_(n_channels_ * spec.n_tapers * spec.spectrum_bins)
    {
        descriptor_ = create_descriptor(spec);
    }

    ~MklChannelBlock()
    {
        if (descriptor_ != nullptr)
        {
            DftiFreeDescriptor(&descriptor_);
        }
    }

    MklChannelBlock(const MklChannelBlock&) = delete;
    MklChannelBlock& operator=(const MklChannelBlock&) = delete;

    MklChannelBlock(MklChannelBlock&& other) noexcept
        : first_channel_(other.first_channel_), last_channel_(other.last_channel_),
          n_channels_(other.n_channels_), spectra_(std::move(other.spectra_)),
          combined_(std::move(other.combined_)), next_(std::move(other.next_)),
          weight_sum_(std::move(other.weight_sum_)),
          channel_power_(std::move(other.channel_power_)), tapered_(std::move(other.tapered_)),
          fft_output_(std::move(other.fft_output_)), descriptor_(other.descriptor_)
    {
        other.descriptor_ = nullptr;
    }

    MklChannelBlock& operator=(MklChannelBlock&& other) noexcept
    {
        if (this != &other)
        {
            if (descriptor_ != nullptr)
            {
                DftiFreeDescriptor(&descriptor_);
            }
            first_channel_ = other.first_channel_;
            last_channel_ = other.last_channel_;
            n_channels_ = other.n_channels_;
            spectra_ = std::move(other.spectra_);
            combined_ = std::move(other.combined_);
            next_ = std::move(other.next_);
            weight_sum_ = std::move(other.weight_sum_);
            channel_power_ = std::move(other.channel_power_);
            tapered_ = std::move(other.tapered_);
            fft_output_ = std::move(other.fft_output_);
            descriptor_ = other.descriptor_;
            other.descriptor_ = nullptr;
        }
        return *this;
    }

    void process(const MultitaperPsdSpec& spec, std::span<const double> x, std::span<double> output,
                 std::span<double> adaptive_state, std::span<unsigned char> adaptive_valid,
                 bool outer_parallel)
    {
        prepare_input(spec, x);
        mkl::check_mkl_status(DftiComputeForward(descriptor_, tapered_.data(), fft_output_.data()),
                              "DftiComputeForward");

        for (std::size_t local = 0; local < n_channels_; ++local)
        {
            const auto channel = first_channel_ + local;
            if (spec.weighting == MultitaperWeighting::adaptive)
            {
                pack_channel_power_spectra(spec, local);
                adaptive_channel(spec, channel_power_[local], channel, adaptive_state,
                                 adaptive_valid);
            }
            else
            {
                linear_channel(spec, local);
            }
            write_channel(spec, channel, output);
        }
    }

  private:
    DFTI_DESCRIPTOR_HANDLE create_descriptor(const MultitaperPsdSpec& spec) const
    {
        DFTI_DESCRIPTOR_HANDLE descriptor = nullptr;
        mkl::check_mkl_status(
            DftiCreateDescriptor(&descriptor, DFTI_DOUBLE, DFTI_REAL, 1,
                                 mkl::checked_mkl_long(spec.fft_length, "FFT length")),
            "DftiCreateDescriptor");
        mkl::check_mkl_status(DftiSetValue(descriptor, DFTI_PLACEMENT, DFTI_NOT_INPLACE),
                              "DftiSetValue(DFTI_PLACEMENT)");
        mkl::check_mkl_status(
            DftiSetValue(descriptor, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX),
            "DftiSetValue(DFTI_CONJUGATE_EVEN_STORAGE)");
        mkl::check_mkl_status(
            DftiSetValue(descriptor, DFTI_NUMBER_OF_TRANSFORMS,
                         mkl::checked_mkl_long(n_channels_ * spec.n_tapers, "transform count")),
            "DftiSetValue(DFTI_NUMBER_OF_TRANSFORMS)");
        mkl::check_mkl_status(DftiSetValue(descriptor, DFTI_INPUT_DISTANCE,
                                           mkl::checked_mkl_long(spec.fft_length, "FFT length")),
                              "DftiSetValue(DFTI_INPUT_DISTANCE)");
        mkl::check_mkl_status(
            DftiSetValue(descriptor, DFTI_OUTPUT_DISTANCE,
                         mkl::checked_mkl_long(spec.spectrum_bins, "spectrum bin count")),
            "DftiSetValue(DFTI_OUTPUT_DISTANCE)");
        mkl::check_mkl_status(DftiCommitDescriptor(descriptor), "DftiCommitDescriptor");
        return descriptor;
    }

    static double taper_at(const MultitaperPsdSpec& spec, std::size_t sample,
                           std::size_t taper) noexcept
    {
        return spec.tapers[sample * spec.n_tapers + taper];
    }

    std::size_t transform_index(const MultitaperPsdSpec& spec, std::size_t local_channel,
                                std::size_t taper) const noexcept
    {
        return local_channel * spec.n_tapers + taper;
    }

    void prepare_input(const MultitaperPsdSpec& spec, std::span<const double> x)
    {
        for (std::size_t local = 0; local < n_channels_; ++local)
        {
            const auto channel = first_channel_ + local;
            double power = 0.0;
            for (std::size_t sample = 0; sample < spec.n_samples; ++sample)
            {
                const auto value = x[sample * spec.n_channels + channel];
                power += value * value;
                for (std::size_t taper = 0; taper < spec.n_tapers; ++taper)
                {
                    tapered_[transform_index(spec, local, taper) * spec.fft_length + sample] =
                        value * taper_at(spec, sample, taper);
                }
            }
            channel_power_[local] = power / static_cast<double>(spec.n_samples);
        }
    }

    void pack_channel_power_spectra(const MultitaperPsdSpec& spec, std::size_t local_channel)
    {
        for (std::size_t taper = 0; taper < spec.n_tapers; ++taper)
        {
            const auto transform = transform_index(spec, local_channel, taper);
            const auto* source = fft_output_.data() + transform * spec.spectrum_bins;
            auto* target = spectra_.data() + taper * spec.spectrum_bins;
            for (std::size_t bin = 0; bin < spec.spectrum_bins; ++bin)
            {
                const auto real = source[bin].real();
                const auto imag = source[bin].imag();
                target[bin] = real * real + imag * imag;
            }
        }
    }

    std::span<const double> taper_spectrum(const MultitaperPsdSpec& spec,
                                           std::size_t taper) const noexcept
    {
        return std::span<const double>(spectra_.data() + taper * spec.spectrum_bins,
                                       spec.spectrum_bins);
    }

    void linear_channel(const MultitaperPsdSpec& spec, std::size_t local_channel)
    {
        std::fill(combined_.begin(), combined_.end(), 0.0);
        for (std::size_t taper = 0; taper < spec.n_tapers; ++taper)
        {
            const auto weight =
                spec.weighting == MultitaperWeighting::unity ? 1.0 : spec.ratios[taper];
            const auto transform = transform_index(spec, local_channel, taper);
            const auto* spectrum = fft_output_.data() + transform * spec.spectrum_bins;
            for (std::size_t bin = 0; bin < spec.spectrum_bins; ++bin)
            {
                const auto real = spectrum[bin].real();
                const auto imag = spectrum[bin].imag();
                combined_[bin] += weight * (real * real + imag * imag);
            }
        }
        const auto scale = 1.0 / static_cast<double>(spec.n_tapers);
        for (auto& value : combined_)
        {
            value *= scale;
        }
    }

    void adaptive_channel(const MultitaperPsdSpec& spec, double power, std::size_t channel,
                          std::span<double> adaptive_state, std::span<unsigned char> adaptive_valid)
    {
        auto state = std::span<double>(adaptive_state.data() + channel * spec.spectrum_bins,
                                       spec.spectrum_bins);
        if (power == 0.0)
        {
            std::fill(combined_.begin(), combined_.end(), 0.0);
            std::fill(state.begin(), state.end(), 0.0);
            adaptive_valid[channel] = 0;
            return;
        }

        if (adaptive_valid[channel] != 0)
        {
            std::copy(state.begin(), state.end(), combined_.begin());
        }
        else
        {
            const auto first = taper_spectrum(spec, 0);
            const auto second = taper_spectrum(spec, 1);
            for (std::size_t bin = 0; bin < spec.spectrum_bins; ++bin)
            {
                combined_[bin] = 0.5 * (first[bin] + second[bin]);
            }
        }

        const auto tol = 0.0005 * power;
        for (std::size_t iteration = 0; iteration < 100; ++iteration)
        {
            accumulate_adaptive_taper<true>(spec, 0, power);
            for (std::size_t taper = 1; taper < spec.n_tapers; ++taper)
            {
                accumulate_adaptive_taper<false>(spec, taper, power);
            }

            double difference = 0.0;
            for (std::size_t bin = 0; bin < spec.spectrum_bins; ++bin)
            {
                next_[bin] = weight_sum_[bin] == 0.0 ? 0.0 : next_[bin] / weight_sum_[bin];
                difference += std::abs(next_[bin] - combined_[bin]);
            }
            difference = real_two_sided_difference(spec, difference);
            combined_.swap(next_);
            if (difference <= tol)
            {
                break;
            }
        }

        std::copy(combined_.begin(), combined_.end(), state.begin());
        adaptive_valid[channel] = 1;
    }

    template <bool Initialize>
    void accumulate_adaptive_taper(const MultitaperPsdSpec& spec, std::size_t taper, double power)
    {
        const auto ratio = spec.ratios[taper];
        const auto leakage = power * spec.one_minus_ratios[taper];
        const auto spectrum = taper_spectrum(spec, taper);
        for (std::size_t bin = 0; bin < spec.spectrum_bins; ++bin)
        {
            const auto den = ratio * combined_[bin] + leakage;
            const auto weight = den == 0.0 ? 0.0 : combined_[bin] / den;
            const auto adaptive_weight = weight * weight * ratio;
            if constexpr (Initialize)
            {
                next_[bin] = adaptive_weight * spectrum[bin];
                weight_sum_[bin] = adaptive_weight;
            }
            else
            {
                next_[bin] += adaptive_weight * spectrum[bin];
                weight_sum_[bin] += adaptive_weight;
            }
        }
    }

    double real_two_sided_difference(const MultitaperPsdSpec& spec,
                                     double one_sided_difference) const
    {
        auto difference = 2.0 * one_sided_difference;
        if (!next_.empty())
        {
            difference -= std::abs(next_[0] - combined_[0]);
        }
        if (spec.fft_length % 2 == 0 && spec.spectrum_bins > 1)
        {
            difference -=
                std::abs(next_[spec.spectrum_bins - 1] - combined_[spec.spectrum_bins - 1]);
        }
        return difference;
    }

    void write_channel(const MultitaperPsdSpec& spec, std::size_t channel,
                       std::span<double> output) const
    {
        for (std::size_t bin = 0; bin < spec.output_bins; ++bin)
        {
            output[bin * spec.n_channels + channel] =
                combined_[spec.output_source_bin[bin]] * spec.output_scale[bin];
        }
    }

    std::size_t first_channel_;
    std::size_t last_channel_;
    std::size_t n_channels_;
    std::vector<double> spectra_;
    std::vector<double> combined_;
    std::vector<double> next_;
    std::vector<double> weight_sum_;
    std::vector<double> channel_power_;
    std::vector<double> tapered_;
    std::vector<std::complex<double>> fft_output_;
    DFTI_DESCRIPTOR_HANDLE descriptor_ = nullptr;
};

class MklMultitaperPsdBackend final : public MultitaperPsdBackend
{
  public:
    explicit MklMultitaperPsdBackend(MultitaperPsdSpec spec)
        : spec_(std::move(spec)), adaptive_state_(spec_.n_channels * spec_.spectrum_bins),
          adaptive_valid_(spec_.n_channels)
    {
        const auto count = worker_count_for(spec_.n_channels);
        blocks_.reserve(count);
        auto first = std::size_t{0};
        for (std::size_t i = 0; i < count; ++i)
        {
            const auto remaining = spec_.n_channels - first;
            const auto blocks_left = count - i;
            const auto size = (remaining + blocks_left - 1) / blocks_left;
            blocks_.emplace_back(spec_, first, first + size);
            first += size;
        }
        start_workers();
    }

    ~MklMultitaperPsdBackend() override
    {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
            ++generation_;
        }
        work_ready_.notify_all();
    }

    [[nodiscard]] std::size_t output_bins() const noexcept override
    {
        return spec_.output_bins;
    }

    void reset_adaptive_state() override
    {
        std::fill(adaptive_state_.begin(), adaptive_state_.end(), 0.0);
        std::fill(adaptive_valid_.begin(), adaptive_valid_.end(), 0);
    }

    void process_real(std::span<const double> x, std::span<double> output) override
    {
        if (x.size() != spec_.n_samples * spec_.n_channels)
        {
            throw std::invalid_argument("multitaper PSD input shape is incorrect");
        }
        if (output.size() != spec_.output_bins * spec_.n_channels)
        {
            throw std::invalid_argument("multitaper PSD output shape is incorrect");
        }

        if (blocks_.size() == 1)
        {
            // The same one-thread limit the workers hold below, for the same
            // two reasons. It keeps MKL's pool off the caller's thread, which
            // is a realtime thread here and must not synchronise a pool it does
            // not own; and the batched real->CCE transform this descriptor
            // drives (`n_channels * n_tapers` transforms in one call) does not
            // return a reproducible result when MKL executes it across threads
            // in a process that also carries a second OpenMP runtime, which a
            // build linking both static MKL (Intel OpenMP) and an OpenMP-using
            // target (GNU OpenMP) does. Without this the PSD of one identical
            // window differs from call to call.
            mkl::LocalThreadLimit thread_limit;
            blocks_.front().process(spec_, x, output, adaptive_state_, adaptive_valid_, false);
            return;
        }

        {
            std::lock_guard lock(mutex_);
            input_ = x;
            output_ = output;
            completed_ = 0;
            worker_error_ = nullptr;
            ++generation_;
        }
        work_ready_.notify_all();

        std::unique_lock lock(mutex_);
        work_done_.wait(lock, [this] { return completed_ == blocks_.size(); });
        if (worker_error_)
        {
            std::rethrow_exception(worker_error_);
        }
    }

    void process_complex(std::span<const std::complex<double>>, std::span<double>) override
    {
        throw std::invalid_argument("MKL multitaper PSD backend only supports real input");
    }

  private:
    void start_workers()
    {
        if (blocks_.size() == 1)
        {
            return;
        }
        workers_.reserve(blocks_.size());
        for (std::size_t i = 0; i < blocks_.size(); ++i)
        {
            workers_.emplace_back([this, i] { worker_loop(i); });
        }
    }

    void worker_loop(std::size_t idx)
    {
        mkl::LocalThreadLimit thread_limit;
        auto seen_generation = std::size_t{0};
        while (true)
        {
            std::span<const double> x;
            std::span<double> output;
            {
                std::unique_lock lock(mutex_);
                work_ready_.wait(lock, [this, seen_generation]
                                 { return stopping_ || generation_ != seen_generation; });
                if (stopping_)
                {
                    return;
                }
                seen_generation = generation_;
                x = input_;
                output = output_;
            }

            std::exception_ptr error;
            try
            {
                blocks_[idx].process(spec_, x, output, adaptive_state_, adaptive_valid_, true);
            }
            catch (...)
            {
                error = std::current_exception();
            }

            {
                std::lock_guard lock(mutex_);
                if (error && !worker_error_)
                {
                    worker_error_ = error;
                }
                ++completed_;
            }
            work_done_.notify_one();
        }
    }

    MultitaperPsdSpec spec_;
    std::vector<double> adaptive_state_;
    std::vector<unsigned char> adaptive_valid_;
    std::vector<MklChannelBlock> blocks_;
    std::vector<std::jthread> workers_;
    std::mutex mutex_;
    std::condition_variable work_ready_;
    std::condition_variable work_done_;
    std::span<const double> input_;
    std::span<double> output_;
    std::exception_ptr worker_error_;
    std::size_t generation_ = 0;
    std::size_t completed_ = 0;
    bool stopping_ = false;
};

} // namespace

bool can_use_mkl_multitaper_psd_backend(const MultitaperPsdSpec& spec) noexcept
{
    return spec.real_input && spec.n_samples <= spec.fft_length;
}

std::unique_ptr<MultitaperPsdBackend> create_mkl_multitaper_psd_backend(MultitaperPsdSpec spec)
{
    return std::make_unique<MklMultitaperPsdBackend>(std::move(spec));
}

} // namespace neurale::signal::detail
