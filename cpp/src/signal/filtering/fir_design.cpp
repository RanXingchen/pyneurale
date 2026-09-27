/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/fir.h>
#include <neurale/signal/windows.h>

#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neurale::signal
{
namespace
{

using Band = std::pair<double, double>;
constexpr double tol = 1e-14;

double sinc(double value)
{
    if (value == 0.0)
    {
        return 1.0;
    }
    const double argument = std::numbers::pi_v<double> * value;
    return std::sin(argument) / argument;
}

std::vector<double> solve(std::vector<double> matrix, std::vector<double> rhs, std::size_t size)
{
    for (std::size_t column = 0; column < size; ++column)
    {
        std::size_t pivot = column;
        double pivot_value = std::abs(matrix[column * size + column]);
        for (std::size_t row = column + 1; row < size; ++row)
        {
            const double candidate = std::abs(matrix[row * size + column]);
            if (candidate > pivot_value)
            {
                pivot = row;
                pivot_value = candidate;
            }
        }
        if (pivot_value <= tol)
        {
            throw std::runtime_error("FIRLS normal matrix is singular");
        }
        if (pivot != column)
        {
            for (std::size_t item = column; item < size; ++item)
            {
                std::swap(matrix[column * size + item], matrix[pivot * size + item]);
            }
            std::swap(rhs[column], rhs[pivot]);
        }
        const double diagonal = matrix[column * size + column];
        for (std::size_t row = column + 1; row < size; ++row)
        {
            const double factor = matrix[row * size + column] / diagonal;
            matrix[row * size + column] = 0.0;
            for (std::size_t item = column + 1; item < size; ++item)
            {
                matrix[row * size + item] -= factor * matrix[column * size + item];
            }
            rhs[row] -= factor * rhs[column];
        }
    }

    std::vector<double> result(size);
    for (std::size_t reverse = 0; reverse < size; ++reverse)
    {
        const std::size_t row = size - reverse - 1;
        double value = rhs[row];
        for (std::size_t column = row + 1; column < size; ++column)
        {
            value -= matrix[row * size + column] * result[column];
        }
        result[row] = value / matrix[row * size + row];
    }
    return result;
}

bool nearly_equal(double left, double right)
{
    return std::abs(left - right) <= tol;
}

bool is_full_band(std::span<const double> bands)
{
    if (!nearly_equal(bands.front(), 0.0) || !nearly_equal(bands.back(), 1.0))
    {
        return false;
    }
    for (std::size_t i = 1; i + 1 < bands.size(); i += 2)
    {
        if (!nearly_equal(bands[i], bands[i + 1]))
        {
            return false;
        }
    }
    return true;
}

bool has_constant_weight(std::span<const double> weights)
{
    for (const double weight : weights.subspan(1))
    {
        if (!nearly_equal(weight, weights.front()))
        {
            return false;
        }
    }
    return true;
}

std::vector<double> solve_full_band_firls(std::vector<double> rhs, bool type_one, double weight)
{
    for (std::size_t i = 0; i < rhs.size(); ++i)
    {
        const double diagonal = type_one && i == 0 ? 0.5 * weight : 0.25 * weight;
        rhs[i] /= diagonal;
    }
    return rhs;
}

std::vector<Band> passbands(std::span<const double> cutoff, std::string_view band)
{
    if (band == "lowpass")
    {
        if (cutoff.size() != 1)
        {
            throw std::invalid_argument("lowpass requires one cutoff");
        }
        return {{0.0, cutoff[0]}};
    }
    if (band == "highpass")
    {
        if (cutoff.size() != 1)
        {
            throw std::invalid_argument("highpass requires one cutoff");
        }
        return {{cutoff[0], 1.0}};
    }
    if (band == "bandpass")
    {
        if (cutoff.size() != 2)
        {
            throw std::invalid_argument("bandpass requires two cutoffs");
        }
        return {{cutoff[0], cutoff[1]}};
    }
    if (band == "bandstop")
    {
        if (cutoff.size() != 2)
        {
            throw std::invalid_argument("bandstop requires two cutoffs");
        }
        return {{0.0, cutoff[0]}, {cutoff[1], 1.0}};
    }
    throw std::invalid_argument("unknown FIR band");
}

void validate_cutoff(std::span<const double> cutoff)
{
    for (std::size_t i = 0; i < cutoff.size(); ++i)
    {
        if (!(cutoff[i] > 0.0 && cutoff[i] < 1.0) || !std::isfinite(cutoff[i]))
        {
            throw std::invalid_argument("cutoff must lie in (0, 1)");
        }
        if (i > 0 && cutoff[i] <= cutoff[i - 1])
        {
            throw std::invalid_argument("cutoff must be strictly increasing");
        }
    }
}

double scale_frequency(const std::vector<Band>& bands)
{
    if (bands.front().first == 0.0)
    {
        return 0.0;
    }
    if (bands.front().second == 1.0)
    {
        return 1.0;
    }
    return 0.5 * (bands.front().first + bands.front().second);
}

} // namespace

void firls(std::size_t order, std::span<const double> bands, std::span<const double> desired,
           std::span<const double> weights, std::span<double> output)
{
    if (order < 1 || output.size() != order + 1)
    {
        throw std::invalid_argument("output size must equal order + 1");
    }
    if (bands.size() < 2 || bands.size() % 2 != 0 || desired.size() != bands.size() ||
        weights.size() * 2 != bands.size())
    {
        throw std::invalid_argument("invalid FIRLS array sizes");
    }

    const std::size_t num_taps = order + 1;
    const bool type_one = num_taps % 2 != 0;
    const std::size_t half_order = (num_taps - 1) / 2;
    const std::size_t basis_size = half_order + 1;
    std::vector<double> basis(basis_size);
    for (std::size_t i = 0; i < basis_size; ++i)
    {
        basis[i] = static_cast<double>(i) + (type_one ? 0.0 : 0.5);
    }

    const bool use_full_band_path = is_full_band(bands) && has_constant_weight(weights);
    std::vector<double> gram;
    if (!use_full_band_path)
    {
        gram.assign(basis_size * basis_size, 0.0);
    }
    std::vector<double> rhs(basis_size, 0.0);
    for (std::size_t band = 0; band < weights.size(); ++band)
    {
        const double start = 0.5 * bands[2 * band];
        const double stop = 0.5 * bands[2 * band + 1];
        const double gain_start = desired[2 * band];
        const double gain_stop = desired[2 * band + 1];
        const double slope = (gain_stop - gain_start) / (stop - start);
        const double intercept = gain_start - slope * start;
        const double weight = weights[band];

        for (std::size_t row = 0; row < basis_size; ++row)
        {
            const double freq_idx = basis[row];
            if (freq_idx == 0.0)
            {
                rhs[row] += weight * (intercept * (stop - start) +
                                      0.5 * slope * (stop * stop - start * start));
            }
            else
            {
                rhs[row] +=
                    weight *
                    (slope /
                         (4.0 * std::numbers::pi_v<double> * std::numbers::pi_v<double>)*(
                             std::cos(2.0 * std::numbers::pi_v<double> * freq_idx * stop) -
                             std::cos(2.0 * std::numbers::pi_v<double> * freq_idx * start)) /
                         (freq_idx * freq_idx) +
                     stop * (slope * stop + intercept) * sinc(2.0 * freq_idx * stop) -
                     start * (slope * start + intercept) * sinc(2.0 * freq_idx * start));
            }
            if (!use_full_band_path)
            {
                for (std::size_t column = 0; column < basis_size; ++column)
                {
                    const double difference = basis[row] - basis[column];
                    const double sum = basis[row] + basis[column];
                    gram[row * basis_size + column] +=
                        weight *
                        (0.5 * stop * (sinc(2.0 * difference * stop) + sinc(2.0 * sum * stop)) -
                         0.5 * start * (sinc(2.0 * difference * start) + sinc(2.0 * sum * start)));
                }
            }
        }
    }

    const auto solution = use_full_band_path
                              ? solve_full_band_firls(std::move(rhs), type_one, weights.front())
                              : solve(std::move(gram), std::move(rhs), basis_size);
    if (type_one)
    {
        for (std::size_t i = 0; i < half_order; ++i)
        {
            output[i] = 0.5 * solution[half_order - i];
        }
        output[half_order] = solution[0];
        for (std::size_t i = 1; i <= half_order; ++i)
        {
            output[half_order + i] = 0.5 * solution[i];
        }
    }
    else
    {
        for (std::size_t i = 0; i < basis_size; ++i)
        {
            output[i] = 0.5 * solution[basis_size - i - 1];
            output[basis_size + i] = 0.5 * solution[i];
        }
    }
}

void firwin(std::size_t order, std::span<const double> cutoff, std::string_view band,
            std::string_view window, bool scale, std::span<double> output)
{
    if (order < 1 || output.size() != order + 1)
    {
        throw std::invalid_argument("FIR output size must equal order + 1");
    }
    validate_cutoff(cutoff);
    const auto bands = passbands(cutoff, band);
    const auto num_taps = output.size();
    const double center = 0.5 * static_cast<double>(num_taps - 1);

    for (std::size_t i = 0; i < num_taps; ++i)
    {
        const double sample = static_cast<double>(i) - center;
        double value = 0.0;
        for (const auto [left, right] : bands)
        {
            value += right * sinc(right * sample) - left * sinc(left * sample);
        }
        output[i] = value;
    }

    std::vector<double> weights(num_taps);
    cosine_window(window, weights, true);
    for (std::size_t i = 0; i < num_taps; ++i)
    {
        output[i] *= weights[i];
    }

    if (!scale)
    {
        return;
    }
    const double freq = scale_frequency(bands);
    double response = 0.0;
    for (std::size_t i = 0; i < num_taps; ++i)
    {
        const double sample = static_cast<double>(i) - center;
        response += output[i] * std::cos(std::numbers::pi_v<double> * sample * freq);
    }
    if (response == 0.0 || !std::isfinite(response))
    {
        throw std::runtime_error("FIR scaling response is singular");
    }
    for (auto& value : output)
    {
        value /= response;
    }
}

} // namespace neurale::signal
