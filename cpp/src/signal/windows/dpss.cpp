/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/windows.h>

#include "../numerics.h"
#include "../spectral/fft_dispatch.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace neurale::signal
{
namespace
{

struct Tridiagonal
{
    std::vector<double> diagonal;
    std::vector<double> off_diagonal;
};

Tridiagonal make_dpss_matrix(std::size_t length, double nw)
{
    Tridiagonal matrix{
        std::vector<double>(length),
        std::vector<double>(length - 1),
    };
    const auto scale = 1.0 / (static_cast<double>(length) * static_cast<double>(length));
    const auto cosine =
        std::cos(2.0 * std::numbers::pi_v<double> * nw / static_cast<double>(length));
    for (std::size_t i = 0; i < length; ++i)
    {
        const auto centered =
            (static_cast<double>(length - 1) - 2.0 * static_cast<double>(i)) / 2.0;
        matrix.diagonal[i] = centered * centered * cosine * scale;
    }
    for (std::size_t i = 1; i < length; ++i)
    {
        matrix.off_diagonal[i - 1] =
            0.5 * static_cast<double>(i) * static_cast<double>(length - i) * scale;
    }
    return matrix;
}

std::pair<double, double> eigenvalue_bounds(const Tridiagonal& matrix)
{
    auto lower = matrix.diagonal.front();
    auto upper = matrix.diagonal.front();
    for (std::size_t i = 0; i < matrix.diagonal.size(); ++i)
    {
        double radius = 0.0;
        if (i > 0)
        {
            radius += std::abs(matrix.off_diagonal[i - 1]);
        }
        if (i + 1 < matrix.diagonal.size())
        {
            radius += std::abs(matrix.off_diagonal[i]);
        }
        lower = std::min(lower, matrix.diagonal[i] - radius);
        upper = std::max(upper, matrix.diagonal[i] + radius);
    }
    const auto padding = 32.0 * std::numeric_limits<double>::epsilon() *
                         std::max({1.0, std::abs(lower), std::abs(upper)});
    return {lower - padding, upper + padding};
}

std::size_t sturm_count(const Tridiagonal& matrix, double value)
{
    const auto pivot_floor = 64.0 * std::numeric_limits<double>::min();
    double pivot = matrix.diagonal[0] - value;
    std::size_t count = pivot < 0.0 ? 1 : 0;
    for (std::size_t i = 1; i < matrix.diagonal.size(); ++i)
    {
        if (std::abs(pivot) < pivot_floor)
        {
            pivot = std::copysign(pivot_floor, pivot == 0.0 ? -1.0 : pivot);
        }
        const auto off = matrix.off_diagonal[i - 1];
        pivot = matrix.diagonal[i] - value - off * off / pivot;
        if (pivot < 0.0)
        {
            ++count;
        }
    }
    return count;
}

double eigenvalue_at(const Tridiagonal& matrix, std::size_t ascending_idx, double lower_bound,
                     double upper_bound)
{
    auto lower = lower_bound;
    auto upper = upper_bound;
    for (int iteration = 0; iteration < 96; ++iteration)
    {
        const auto middle = std::midpoint(lower, upper);
        if (sturm_count(matrix, middle) <= ascending_idx)
        {
            lower = middle;
        }
        else
        {
            upper = middle;
        }
        if (upper - lower <= 8.0 * std::numeric_limits<double>::epsilon() *
                                 std::max({1.0, std::abs(lower), std::abs(upper)}))
        {
            break;
        }
    }
    return std::midpoint(lower, upper);
}

double dot(std::span<const double> left, std::span<const double> right)
{
    return std::inner_product(left.begin(), left.end(), right.begin(), 0.0);
}

double normalize(std::span<double> values)
{
    const auto norm = std::sqrt(dot(values, values));
    if (!(norm > 0.0) || !std::isfinite(norm))
    {
        throw std::runtime_error("DPSS inverse iteration produced zero norm");
    }
    for (auto& value : values)
    {
        value /= norm;
    }
    return norm;
}

void orthogonalize(std::span<double> values, std::span<const double> previous, std::size_t length,
                   std::size_t n_vectors)
{
    for (std::size_t vector = 0; vector < n_vectors; ++vector)
    {
        const auto existing = previous.subspan(vector * length, length);
        const auto projection = dot(values, existing);
        for (std::size_t i = 0; i < length; ++i)
        {
            values[i] -= projection * existing[i];
        }
    }
}

void solve_shifted(const Tridiagonal& matrix, double shift, std::span<const double> right,
                   std::span<double> solution, std::vector<double>& modified_off)
{
    const auto length = matrix.diagonal.size();
    const auto pivot_floor = 128.0 * std::numeric_limits<double>::epsilon();
    auto pivot = matrix.diagonal[0] - shift;
    if (std::abs(pivot) < pivot_floor)
    {
        pivot = std::copysign(pivot_floor, pivot == 0.0 ? 1.0 : pivot);
    }
    modified_off[0] = matrix.off_diagonal[0] / pivot;
    solution[0] = right[0] / pivot;
    for (std::size_t i = 1; i < length; ++i)
    {
        pivot = matrix.diagonal[i] - shift - matrix.off_diagonal[i - 1] * modified_off[i - 1];
        if (std::abs(pivot) < pivot_floor)
        {
            pivot = std::copysign(pivot_floor, pivot == 0.0 ? 1.0 : pivot);
        }
        if (i + 1 < length)
        {
            modified_off[i] = matrix.off_diagonal[i] / pivot;
        }
        solution[i] = (right[i] - matrix.off_diagonal[i - 1] * solution[i - 1]) / pivot;
    }
    for (std::size_t i = length - 1; i-- > 0;)
    {
        solution[i] -= modified_off[i] * solution[i + 1];
    }
}

double rayleigh_quotient(const Tridiagonal& matrix, std::span<const double> vector)
{
    double result = 0.0;
    for (std::size_t i = 0; i < vector.size(); ++i)
    {
        result += matrix.diagonal[i] * vector[i] * vector[i];
        if (i + 1 < vector.size())
        {
            result += 2.0 * matrix.off_diagonal[i] * vector[i] * vector[i + 1];
        }
    }
    return result;
}

std::vector<double> inverse_iteration(const Tridiagonal& matrix, double eigenvalue,
                                      std::size_t order, std::span<const double> previous)
{
    const auto length = matrix.diagonal.size();
    std::vector<double> vector(length);
    for (std::size_t i = 0; i < length; ++i)
    {
        vector[i] =
            std::sin(std::numbers::pi_v<double> * static_cast<double>((order + 1) * (i + 1)) /
                     static_cast<double>(length + 1));
    }
    orthogonalize(vector, previous, length, order);
    normalize(vector);

    std::vector<double> next(length);
    std::vector<double> modified_off(length - 1);
    auto shift = eigenvalue + (order + 1) * 32.0 * std::numeric_limits<double>::epsilon();
    for (int iteration = 0; iteration < 16; ++iteration)
    {
        solve_shifted(matrix, shift, vector, next, modified_off);
        orthogonalize(next, previous, length, order);
        normalize(next);
        const auto alignment = std::abs(dot(vector, next));
        vector.swap(next);
        shift = rayleigh_quotient(matrix, vector) +
                (order + 1) * 32.0 * std::numeric_limits<double>::epsilon();
        if (1.0 - alignment < 1e-13)
        {
            break;
        }
    }
    return vector;
}

void apply_sign_convention(std::span<double> taper, std::size_t order)
{
    if (order % 2 == 0)
    {
        if (std::accumulate(taper.begin(), taper.end(), 0.0) < 0.0)
        {
            for (auto& value : taper)
            {
                value = -value;
            }
        }
        return;
    }
    const auto threshold = std::max(1e-7, 1.0 / static_cast<double>(taper.size()));
    for (const auto value : taper)
    {
        if (value * value > threshold)
        {
            if (value < 0.0)
            {
                for (auto& item : taper)
                {
                    item = -item;
                }
            }
            return;
        }
    }
}

double concentration_ratio(std::span<const double> taper, double nw)
{
    const auto length = taper.size();
    const auto fft_length = detail::next_power_of_two(2 * length - 1);
    std::vector<std::complex<double>> autocorrelation(fft_length, 0.0);
    for (std::size_t i = 0; i < length; ++i)
    {
        autocorrelation[i] = taper[i];
    }
    detail::fft_inplace(autocorrelation, false);
    for (auto& value : autocorrelation)
    {
        value = std::norm(value);
    }
    detail::fft_inplace(autocorrelation, true);

    const auto bandwidth = nw / static_cast<double>(length);
    auto ratio = 2.0 * bandwidth * autocorrelation[0].real();
    for (std::size_t lag = 1; lag < length; ++lag)
    {
        const auto argument = 2.0 * bandwidth * static_cast<double>(lag);
        const auto sinc = std::sin(std::numbers::pi_v<double> * argument) /
                          (std::numbers::pi_v<double> * argument);
        ratio += 4.0 * bandwidth * sinc * autocorrelation[lag].real();
    }
    return std::clamp(ratio, 0.0, 1.0);
}

} // namespace

void multitap(std::size_t length, double nw, std::size_t n_tapers, std::span<double> tapers,
              std::span<double> concentration_ratios)
{
    if (length < 2)
    {
        throw std::invalid_argument("DPSS length must be at least two");
    }
    if (!std::isfinite(nw) || nw <= 0.0 || nw >= static_cast<double>(length) / 2.0)
    {
        throw std::invalid_argument("DPSS time bandwidth must be in (0, length / 2)");
    }
    if (n_tapers == 0 || n_tapers > length)
    {
        throw std::invalid_argument("DPSS taper count must be in [1, length]");
    }
    if (tapers.size() != length * n_tapers || concentration_ratios.size() != n_tapers)
    {
        throw std::invalid_argument("DPSS output buffer shape is incorrect");
    }

    const auto matrix = make_dpss_matrix(length, nw);
    const auto [lower, upper] = eigenvalue_bounds(matrix);
    std::vector<double> eigenvectors(n_tapers * length);
    for (std::size_t order = 0; order < n_tapers; ++order)
    {
        const auto ascending_idx = length - 1 - order;
        const auto eigenvalue = eigenvalue_at(matrix, ascending_idx, lower, upper);
        auto vector =
            inverse_iteration(matrix, eigenvalue, order,
                              std::span<const double>(eigenvectors.data(), order * length));
        apply_sign_convention(vector, order);
        std::copy(vector.begin(), vector.end(), eigenvectors.begin() + order * length);
        concentration_ratios[order] = concentration_ratio(vector, nw);
        for (std::size_t sample = 0; sample < length; ++sample)
        {
            tapers[sample * n_tapers + order] = vector[sample];
        }
    }
}

} // namespace neurale::signal
