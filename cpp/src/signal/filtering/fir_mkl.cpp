/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "fir_mkl.h"
#include "fir_state.h"
#include "mkl_utils.h"

#include <algorithm>
#include <array>
#include <memory>
#include <thread>
#include <vector>

namespace neurale::signal::detail
{
namespace
{

constexpr std::size_t kMaxChannelWorkers = 4;
constexpr std::size_t kParallelMinChannels = 16;

class ConvTask
{
  public:
    ConvTask(MKL_INT mode, MKL_INT input_length, MKL_INT taps_length, MKL_INT output_length,
             MKL_INT output_start)
    {
        mkl::check_vsl_status(
            vsldConvNewTask1D(&task_, mode, input_length, taps_length, output_length),
            "vsldConvNewTask1D");
        mkl::check_vsl_status(vslConvSetStart(task_, &output_start), "vslConvSetStart");
    }

    explicit ConvTask(VSLConvTaskPtr source)
    {
        mkl::check_vsl_status(vslConvCopyTask(&task_, source), "vslConvCopyTask");
    }

    ~ConvTask()
    {
        if (task_ != nullptr)
        {
            vslConvDeleteTask(&task_);
        }
    }

    ConvTask(const ConvTask&) = delete;
    ConvTask& operator=(const ConvTask&) = delete;

    VSLConvTaskPtr get() const noexcept
    {
        return task_;
    }

  private:
    VSLConvTaskPtr task_ = nullptr;
};

std::size_t channel_worker_count(std::size_t n_channels)
{
    if (n_channels < kParallelMinChannels)
    {
        return 1;
    }
    const auto hardware = std::thread::hardware_concurrency();
    const auto available = hardware == 0 ? kMaxChannelWorkers : hardware;
    return std::min({n_channels, kMaxChannelWorkers, available});
}

struct CachedTask
{
    std::size_t input_length = 0;
    std::size_t taps_length = 0;
    std::size_t output_length = 0;
    std::size_t output_start = 0;
    std::unique_ptr<ConvTask> task;
};

ConvTask& cached_task(MklFirMode mode, std::size_t input_length, std::size_t taps_length,
                      std::size_t output_length, std::size_t output_start)
{
    thread_local std::array<CachedTask, 2> cache;
    auto& entry = cache[mode == MklFirMode::direct ? 0 : 1];
    if (entry.task == nullptr || entry.input_length != input_length ||
        entry.taps_length != taps_length || entry.output_length != output_length ||
        entry.output_start != output_start)
    {
        const MKL_INT mkl_mode =
            mode == MklFirMode::direct ? VSL_CONV_MODE_DIRECT : VSL_CONV_MODE_FFT;
        entry.task =
            std::make_unique<ConvTask>(mkl_mode, mkl::checked_mkl_int(input_length, "input length"),
                                       mkl::checked_mkl_int(taps_length, "tap count"),
                                       mkl::checked_mkl_int(output_length, "output length"),
                                       mkl::checked_mkl_int(output_start, "output start"));
        entry.input_length = input_length;
        entry.taps_length = taps_length;
        entry.output_length = output_length;
        entry.output_start = output_start;
    }
    return *entry.task;
}

bool state_is_zero(std::span<const double> state)
{
    return std::all_of(state.begin(), state.end(), [](double value) { return value == 0.0; });
}

void run_channel_range(VSLConvTaskPtr task, std::span<const double> x, MKL_INT input_stride,
                       std::span<const double> taps, std::span<double> output,
                       MKL_INT output_stride, std::size_t first_channel, std::size_t step,
                       std::size_t n_channels, int& status)
{
    for (std::size_t channel = first_channel; channel < n_channels; channel += step)
    {
        status = vsldConvExec1D(task, x.data() + channel, input_stride, taps.data(), 1,
                                output.data() + channel, output_stride);
        if (status != VSL_STATUS_OK)
        {
            return;
        }
    }
}

void run_channels(ConvTask& task, std::span<const double> x, std::span<const double> taps,
                  std::span<double> output, std::size_t n_channels, std::size_t workers)
{
    const MKL_INT channel_stride = mkl::checked_mkl_int(n_channels, "channel count");
    if (workers == 1)
    {
        int status = VSL_STATUS_OK;
        run_channel_range(task.get(), x, channel_stride, taps, output, channel_stride, 0, 1,
                          n_channels, status);
        mkl::check_vsl_status(status, "vsldConvExec1D");
        return;
    }

    std::vector<std::unique_ptr<ConvTask>> tasks;
    std::vector<std::thread> threads;
    std::vector<int> statuses(workers, VSL_STATUS_OK);
    tasks.reserve(workers);
    threads.reserve(workers);

    for (std::size_t worker = 0; worker < workers; ++worker)
    {
        tasks.push_back(std::make_unique<ConvTask>(task.get()));
    }
    for (std::size_t worker = 0; worker < workers; ++worker)
    {
        threads.emplace_back(
            [&, worker]
            {
                run_channel_range(tasks[worker]->get(), x, channel_stride, taps, output,
                                  channel_stride, worker, workers, n_channels, statuses[worker]);
            });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }
    for (const int status : statuses)
    {
        mkl::check_vsl_status(status, "vsldConvExec1D");
    }
}

} // namespace

void fir_filter_mkl(std::span<const double> x, std::size_t n_samples, std::size_t n_channels,
                    std::span<const double> taps, std::span<const double> state,
                    std::span<double> output, std::span<double> final_state, MklFirMode mode)
{
    const std::size_t history_length = taps.size() - 1;
    if (n_samples == 0)
    {
        if (state.data() != final_state.data())
        {
            std::copy(state.begin(), state.end(), final_state.begin());
        }
        return;
    }

    const std::size_t combined_length = history_length + n_samples;
    const bool zero_state = state_is_zero(state);
    const std::size_t input_length = zero_state ? n_samples : combined_length;
    const std::size_t output_start = zero_state ? 0 : history_length;
    const std::size_t workers = channel_worker_count(n_channels);
    mkl::LocalThreadLimit thread_limit(mode == MklFirMode::direct || workers > 1);
    ConvTask& task = cached_task(mode, input_length, taps.size(), n_samples, output_start);

    thread_local std::vector<double> combined;
    std::span<const double> source = x;
    if (!zero_state)
    {
        combined.resize(combined_length * n_channels);
        std::copy(state.begin(), state.end(), combined.begin());
        std::copy(x.begin(), x.end(), combined.begin() + state.size());
        source = combined;
    }
    run_channels(task, source, taps, output, n_channels, workers);

    update_fir_state(x, n_samples, n_channels, history_length, state, final_state);
}

} // namespace neurale::signal::detail
