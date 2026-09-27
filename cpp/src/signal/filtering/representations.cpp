/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/representations.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <utility>

#ifdef NEURALE_SIGNAL_WITH_MKL
#include <mkl.h>
#endif

namespace neurale::signal
{
namespace
{

constexpr double kTolerance = 100.0 * std::numeric_limits<double>::epsilon();

void require_finite(double value, const char* name)
{
    if (!std::isfinite(value))
    {
        throw std::invalid_argument(std::string(name) + " must be finite");
    }
}

void require_finite(std::complex<double> value, const char* name)
{
    require_finite(value.real(), name);
    require_finite(value.imag(), name);
}

void require_positive(double value, const char* name)
{
    require_finite(value, name);
    if (!(value > 0.0))
    {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }
}

void validate_state(const StateSpace& state)
{
    const auto n = state.order;
    if (state.a.size() != n * n || state.b.size() != n || state.c.size() != n)
    {
        throw std::invalid_argument("state-space buffer shape is invalid");
    }
    for (const auto value : state.a)
        require_finite(value, "a");
    for (const auto value : state.b)
        require_finite(value, "b");
    for (const auto value : state.c)
        require_finite(value, "c");
    require_finite(state.d, "d");
}

std::vector<double> trim_leading(std::span<const double> coefs, const char* name)
{
    if (coefs.empty())
    {
        throw std::invalid_argument(std::string(name) + " must not be empty");
    }
    std::size_t first = 0;
    while (first < coefs.size() && coefs[first] == 0.0)
    {
        ++first;
    }
    if (first == coefs.size())
    {
        throw std::invalid_argument(std::string(name) + " must contain a nonzero coefficient");
    }
    std::vector<double> result(coefs.begin() + static_cast<std::ptrdiff_t>(first), coefs.end());
    for (const auto value : result)
        require_finite(value, name);
    return result;
}

std::vector<double> solve(std::vector<double> matrix, std::vector<double> right, std::size_t n,
                          std::size_t nrhs)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    std::vector<MKL_INT> pivots(n);
    const auto status = LAPACKE_dgesv(
        LAPACK_ROW_MAJOR, static_cast<MKL_INT>(n), static_cast<MKL_INT>(nrhs), matrix.data(),
        static_cast<MKL_INT>(n), pivots.data(), right.data(), static_cast<MKL_INT>(nrhs));
    if (status != 0)
    {
        throw std::invalid_argument("state matrix is singular");
    }
    return right;
#else
    for (std::size_t pivot = 0; pivot < n; ++pivot)
    {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < n; ++row)
        {
            if (std::abs(matrix[row * n + pivot]) > std::abs(matrix[best * n + pivot]))
            {
                best = row;
            }
        }
        if (std::abs(matrix[best * n + pivot]) <= kTolerance)
        {
            throw std::invalid_argument("state matrix is singular");
        }
        if (best != pivot)
        {
            for (std::size_t column = 0; column < n; ++column)
            {
                std::swap(matrix[pivot * n + column], matrix[best * n + column]);
            }
            for (std::size_t column = 0; column < nrhs; ++column)
            {
                std::swap(right[pivot * nrhs + column], right[best * nrhs + column]);
            }
        }
        const auto diagonal = matrix[pivot * n + pivot];
        for (std::size_t row = pivot + 1; row < n; ++row)
        {
            const auto factor = matrix[row * n + pivot] / diagonal;
            matrix[row * n + pivot] = 0.0;
            for (std::size_t column = pivot + 1; column < n; ++column)
            {
                matrix[row * n + column] -= factor * matrix[pivot * n + column];
            }
            for (std::size_t column = 0; column < nrhs; ++column)
            {
                right[row * nrhs + column] -= factor * right[pivot * nrhs + column];
            }
        }
    }
    for (std::size_t reverse = n; reverse-- > 0;)
    {
        for (std::size_t column = 0; column < nrhs; ++column)
        {
            auto value = right[reverse * nrhs + column];
            for (std::size_t known = reverse + 1; known < n; ++known)
            {
                value -= matrix[reverse * n + known] * right[known * nrhs + column];
            }
            right[reverse * nrhs + column] = value / matrix[reverse * n + reverse];
        }
    }
    return right;
#endif
}

void solve_into(std::vector<double> matrix, std::span<double> right, std::size_t n,
                std::size_t nrhs)
{
#ifdef NEURALE_SIGNAL_WITH_MKL
    std::vector<MKL_INT> pivots(n);
    const auto status = LAPACKE_dgesv(
        LAPACK_ROW_MAJOR, static_cast<MKL_INT>(n), static_cast<MKL_INT>(nrhs), matrix.data(),
        static_cast<MKL_INT>(n), pivots.data(), right.data(), static_cast<MKL_INT>(nrhs));
    if (status != 0)
    {
        throw std::invalid_argument("state matrix is singular");
    }
#else
    auto result =
        solve(std::move(matrix), std::vector<double>(right.begin(), right.end()), n, nrhs);
    std::copy(result.begin(), result.end(), right.begin());
#endif
}

double effective_rate(double rate, double prewarp)
{
    require_positive(rate, "fs");
    if (std::isnan(prewarp))
        return rate;
    require_positive(prewarp, "prewarp_frequency");
    if (prewarp >= rate / 2.0)
    {
        throw std::invalid_argument("prewarp_frequency must be below Nyquist");
    }
    return std::numbers::pi_v<double> * prewarp /
           std::tan(std::numbers::pi_v<double> * prewarp / rate);
}

std::complex<double> evaluate_polynomial(std::span<const double> coefs, std::complex<double> value)
{
    std::complex<double> result = 0.0;
    for (const auto coef : coefs)
    {
        result = result * value + coef;
    }
    return result;
}

std::complex<double> evaluate_derivative(std::span<const double> coefs, std::complex<double> value)
{
    std::complex<double> result = 0.0;
    const auto degree = coefs.size() - 1;
    for (std::size_t i = 0; i < degree; ++i)
    {
        result = result * value + coefs[i] * static_cast<double>(degree - i);
    }
    return result;
}

std::vector<double> multiply_square(std::span<const double> left, std::span<const double> right,
                                    std::size_t n)
{
    std::vector<double> result(n * n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t inner = 0; inner < n; ++inner)
        {
            const auto value = left[row * n + inner];
            for (std::size_t column = 0; column < n; ++column)
            {
                result[row * n + column] += value * right[inner * n + column];
            }
        }
    }
    return result;
}

std::vector<double> characteristic_polynomial(const StateSpace& state)
{
    const auto n = state.order;
    std::vector<double> coefs(n + 1, 0.0);
    coefs[0] = 1.0;
    if (n == 0)
        return coefs;

    std::vector<double> current(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        current[i * n + i] = 1.0;
    }
    for (std::size_t step = 1; step <= n; ++step)
    {
        current = multiply_square(state.a, current, n);
        double trace = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            trace += current[i * n + i];
        }
        const double coef = -trace / static_cast<double>(step);
        coefs[step] = coef;
        for (std::size_t i = 0; i < n; ++i)
        {
            current[i * n + i] += coef;
        }
    }
    return coefs;
}

std::vector<std::complex<double>>
canonicalize_real_polynomial_roots(std::vector<std::complex<double>> roots)
{
    constexpr double root_tol = 1e-6;
    std::vector<std::complex<double>> real;
    std::vector<std::complex<double>> positive;
    std::vector<std::complex<double>> negative;
    for (auto root : roots)
    {
        if (std::abs(root.imag()) <= root_tol * std::max(1.0, std::abs(root)))
        {
            real.emplace_back(root.real(), 0.0);
        }
        else if (root.imag() > 0.0)
        {
            positive.push_back(root);
        }
        else
        {
            negative.push_back(root);
        }
    }
    std::vector<std::complex<double>> result = real;
    for (const auto root : positive)
    {
        if (negative.empty())
        {
            result.push_back(root);
            continue;
        }
        const auto iterator = std::min_element(
            negative.begin(), negative.end(), [&](const auto left, const auto right)
            { return std::abs(left - std::conj(root)) < std::abs(right - std::conj(root)); });
        if (std::abs(*iterator - std::conj(root)) <=
            root_tol * std::max({1.0, std::abs(root), std::abs(*iterator)}))
        {
            const std::complex<double> canonical{
                0.5 * (root.real() + iterator->real()),
                0.5 * (std::abs(root.imag()) + std::abs(iterator->imag()))};
            result.push_back(std::conj(canonical));
            result.push_back(canonical);
            negative.erase(iterator);
        }
        else
        {
            result.push_back(root);
        }
    }
    result.insert(result.end(), negative.begin(), negative.end());
    return result;
}

bool is_effectively_real_root(std::complex<double> value)
{
    return std::abs(value.imag()) <= kTolerance * std::abs(value);
}

bool same_effective_real_part(std::complex<double> left, std::complex<double> right)
{
    return std::abs(left.real() - right.real()) <=
           kTolerance * std::max(std::abs(left), std::abs(right));
}

std::vector<std::complex<double>> finite_roots(std::span<const std::complex<double>> roots)
{
    std::vector<std::complex<double>> result;
    result.reserve(roots.size());
    for (const auto root : roots)
    {
        if (std::isfinite(root.real()) && std::isfinite(root.imag()))
        {
            result.push_back(root);
        }
    }
    return result;
}

std::vector<std::complex<double>> conjugate_pair_order(std::span<const std::complex<double>> roots)
{
    std::vector<std::complex<double>> real;
    std::vector<std::complex<double>> positive;
    std::vector<std::complex<double>> negative;
    for (const auto root : roots)
    {
        require_finite(root, "roots");
        if (is_effectively_real_root(root))
        {
            real.emplace_back(root.real(), 0.0);
        }
        else if (root.imag() > 0.0)
        {
            positive.push_back(root);
        }
        else
        {
            negative.push_back(root);
        }
    }

    const auto compare_real = [](const auto left, const auto right)
    { return left.real() < right.real(); };
    const auto compare_complex = [](const auto left, const auto right)
    {
        if (left.real() != right.real())
            return left.real() < right.real();
        return std::abs(left.imag()) > std::abs(right.imag());
    };
    std::sort(real.begin(), real.end(), compare_real);
    std::sort(positive.begin(), positive.end(), compare_complex);

    std::vector<std::complex<double>> result;
    result.reserve(roots.size());
    for (const auto root : positive)
    {
        if (negative.empty())
        {
            throw std::invalid_argument("roots must contain conjugate pairs");
        }
        const auto match = std::min_element(
            negative.begin(), negative.end(), [&](const auto left, const auto right)
            { return std::abs(left - std::conj(root)) < std::abs(right - std::conj(root)); });
        if (!same_effective_real_part(root, *match) ||
            std::abs(match->imag() + root.imag()) >
                kTolerance * std::max(std::abs(root), std::abs(*match)))
        {
            throw std::invalid_argument("roots must contain conjugate pairs");
        }
        const std::complex<double> canonical{0.5 * (root.real() + match->real()),
                                             0.5 *
                                                 (std::abs(root.imag()) + std::abs(match->imag()))};
        result.push_back(std::conj(canonical));
        result.push_back(canonical);
        negative.erase(match);
    }
    if (!negative.empty())
    {
        throw std::invalid_argument("roots must contain conjugate pairs");
    }
    result.insert(result.end(), real.begin(), real.end());
    return result;
}

std::vector<double> polynomial_from_ordered_roots(std::span<const std::complex<double>> roots)
{
    std::vector<std::complex<double>> coefs{1.0};
    for (const auto root : roots)
    {
        std::vector<std::complex<double>> next(coefs.size() + 1, 0.0);
        for (std::size_t i = 0; i < coefs.size(); ++i)
        {
            next[i] += coefs[i];
            next[i + 1] -= root * coefs[i];
        }
        coefs.swap(next);
    }
    std::vector<double> result(coefs.size());
    for (std::size_t i = 0; i < coefs.size(); ++i)
    {
        result[i] = coefs[i].real();
    }
    return result;
}

std::array<double, 3> padded_quadratic(const std::vector<double>& coefs)
{
    std::array<double, 3> result{};
    std::copy(coefs.begin(), coefs.end(),
              result.begin() + static_cast<std::ptrdiff_t>(result.size() - coefs.size()));
    return result;
}

std::array<double, 2> padded_linear(const std::vector<double>& coefs)
{
    std::array<double, 2> result{};
    std::copy(coefs.begin(), coefs.end(),
              result.begin() + static_cast<std::ptrdiff_t>(result.size() - coefs.size()));
    return result;
}

double section_wn(std::span<const std::complex<double>> poles, std::size_t first)
{
    auto value = std::sqrt(std::abs(poles[first]) * std::abs(poles[first + 1]));
    return value == 0.0 ? 1.0 : value;
}

struct OrderedSplitRoots
{
    std::vector<std::complex<double>> complex;
    std::vector<double> real;
};

OrderedSplitRoots split_ordered_roots(std::span<const std::complex<double>> roots)
{
    OrderedSplitRoots result;
    result.complex.reserve(roots.size());
    result.real.reserve(roots.size());
    for (const auto root : roots)
    {
        if (root.imag() != 0.0)
        {
            result.complex.push_back(root);
        }
        else
        {
            result.real.push_back(root.real());
        }
    }
    return result;
}

double unit_sign(double value)
{
    return value < 0.0 ? -1.0 : 1.0;
}

void order_sos_roots_for_pairing(std::vector<std::complex<double>>& complex_poles,
                                 std::vector<double>& real_poles,
                                 std::vector<std::complex<double>>& complex_zeros,
                                 std::vector<double>& real_zeros,
                                 std::vector<std::complex<double>>& zeros,
                                 std::vector<std::complex<double>>& poles)
{
    std::stable_sort(complex_poles.begin(), complex_poles.end(),
                     [](const auto left, const auto right)
                     {
                         const auto left_unit =
                             std::polar(1.0, std::atan2(left.imag(), left.real()));
                         const auto right_unit =
                             std::polar(1.0, std::atan2(right.imag(), right.real()));
                         return std::abs(left - left_unit) < std::abs(right - right_unit);
                     });
    std::stable_sort(
        real_poles.begin(), real_poles.end(), [](double left, double right)
        { return std::abs(left - unit_sign(left)) < std::abs(right - unit_sign(right)); });

    poles.assign(complex_poles.begin(), complex_poles.end());
    poles.insert(poles.end(), real_poles.begin(), real_poles.end());

    zeros.clear();
    zeros.reserve(complex_zeros.size() + real_zeros.size());
    const auto n_complex_zeros = complex_zeros.size();
    const auto n_real_zeros = real_zeros.size();
    for (std::size_t i = 0; i < n_complex_zeros / 2; ++i)
    {
        if (!complex_poles.empty())
        {
            auto first = std::min_element(
                complex_zeros.begin(), complex_zeros.end(), [&](const auto left, const auto right)
                { return std::abs(left - complex_poles[0]) < std::abs(right - complex_poles[0]); });
            zeros.push_back(*first);
            complex_zeros.erase(first);

            auto second = std::min_element(
                complex_zeros.begin(), complex_zeros.end(), [&](const auto left, const auto right)
                { return std::abs(left - complex_poles[1]) < std::abs(right - complex_poles[1]); });
            zeros.push_back(*second);
            complex_zeros.erase(second);
            complex_poles.erase(complex_poles.begin(), complex_poles.begin() + 2);
        }
        else if (!real_poles.empty())
        {
            auto first = std::min_element(
                complex_zeros.begin(), complex_zeros.end(), [&](const auto left, const auto right)
                { return std::abs(left - real_poles[0]) < std::abs(right - real_poles[0]); });
            const auto offset =
                static_cast<std::size_t>(std::distance(complex_zeros.begin(), first));
            zeros.push_back(complex_zeros[offset]);
            zeros.push_back(complex_zeros[offset + 1]);
            complex_zeros.erase(complex_zeros.begin() + static_cast<std::ptrdiff_t>(offset));
            real_poles.erase(real_poles.begin(), real_poles.begin() + 2);
        }
        else
        {
            zeros.insert(zeros.end(), complex_zeros.begin(), complex_zeros.end());
            break;
        }
    }

    for (std::size_t i = 0; i < n_real_zeros; ++i)
    {
        if (!complex_poles.empty())
        {
            auto first = std::min_element(
                real_zeros.begin(), real_zeros.end(), [&](double left, double right)
                { return std::abs(left - complex_poles[0]) < std::abs(right - complex_poles[0]); });
            zeros.emplace_back(*first, 0.0);
            real_zeros.erase(first);
            complex_poles.erase(complex_poles.begin());
        }
        else if (!real_poles.empty())
        {
            auto first = std::min_element(
                real_zeros.begin(), real_zeros.end(), [&](double left, double right)
                { return std::abs(left - real_poles[0]) < std::abs(right - real_poles[0]); });
            zeros.emplace_back(*first, 0.0);
            real_zeros.erase(first);
            real_poles.erase(real_poles.begin());
        }
        else
        {
            for (const auto zero : real_zeros)
            {
                zeros.emplace_back(zero, 0.0);
            }
            break;
        }
    }
}

void append_ordered_sos_pair(std::span<const std::complex<double>> zeros,
                             std::span<const std::complex<double>> poles, std::size_t idx,
                             std::vector<double>& sos)
{
    std::vector<std::complex<double>> section_poles{poles[idx], poles[idx + 1]};
    std::vector<std::complex<double>> section_zeros;
    if (idx + 2 <= zeros.size())
    {
        section_zeros.assign(zeros.begin() + static_cast<std::ptrdiff_t>(idx),
                             zeros.begin() + static_cast<std::ptrdiff_t>(idx + 2));
    }
    const auto num = padded_quadratic(polynomial_from_ordered_roots(section_zeros));
    const auto den = padded_quadratic(polynomial_from_ordered_roots(section_poles));
    sos.insert(sos.end(), {den[2], den[1], den[0], num[2], num[1], num[0]});
}

void append_ordered_sos_pairs(std::span<const std::complex<double>> zeros,
                              std::span<const std::complex<double>> poles, std::size_t first,
                              std::size_t last, std::vector<double>& sos)
{
    for (auto i = first; i < last; i += 2)
    {
        append_ordered_sos_pair(zeros, poles, i, sos);
    }
}

void append_ordered_last_pole(std::span<const std::complex<double>> zeros,
                              std::span<const std::complex<double>> poles, std::vector<double>& sos)
{
    const auto num = padded_linear(polynomial_from_ordered_roots(zeros));
    const auto den = padded_linear(polynomial_from_ordered_roots(poles));
    sos.insert(sos.end(), {0.0, den[1], den[0], 0.0, num[1], num[0]});
}

} // namespace

std::vector<std::complex<double>> sort_conjugate_pairs(std::span<const std::complex<double>> x)
{
    std::vector<std::complex<double>> real;
    std::vector<std::complex<double>> positive;
    std::vector<std::complex<double>> negative;
    for (const auto value : x)
    {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            throw std::invalid_argument("roots must be finite");
        }
        if (std::abs(value.imag()) <= kTolerance * std::max(1.0, std::abs(value)))
        {
            real.emplace_back(value.real(), 0.0);
        }
        else if (value.imag() > 0.0)
        {
            positive.push_back(value);
        }
        else
        {
            negative.push_back(value);
        }
    }
    const auto compare = [](const auto left, const auto right)
    {
        if (left.real() != right.real())
            return left.real() < right.real();
        return std::abs(left.imag()) < std::abs(right.imag());
    };
    std::sort(real.begin(), real.end(), compare);
    std::sort(positive.begin(), positive.end(), compare);
    std::sort(negative.begin(), negative.end(), compare);
    std::vector<std::complex<double>> result = real;
    for (const auto value : positive)
    {
        if (negative.empty())
        {
            throw std::invalid_argument("roots must contain conjugate pairs");
        }
        const auto iterator = std::min_element(
            negative.begin(), negative.end(), [&](const auto left, const auto right)
            { return std::abs(left - std::conj(value)) < std::abs(right - std::conj(value)); });
        if (std::abs(*iterator - std::conj(value)) >
            kTolerance * std::max({1.0, std::abs(value), std::abs(*iterator)}))
        {
            throw std::invalid_argument("roots must contain conjugate pairs");
        }
        const std::complex<double> canonical{
            0.5 * (value.real() + iterator->real()),
            0.5 * (std::abs(value.imag()) + std::abs(iterator->imag()))};
        result.push_back(std::conj(canonical));
        result.push_back(canonical);
        negative.erase(iterator);
    }
    if (!negative.empty())
    {
        throw std::invalid_argument("roots must contain conjugate pairs");
    }
    return result;
}

std::vector<double> polynomial_from_roots(std::span<const std::complex<double>> roots)
{
    const auto ordered = sort_conjugate_pairs(roots);
    std::vector<std::complex<double>> coefs{1.0};
    for (const auto root : ordered)
    {
        std::vector<std::complex<double>> next(coefs.size() + 1, 0.0);
        for (std::size_t i = 0; i < coefs.size(); ++i)
        {
            next[i] += coefs[i];
            next[i + 1] -= root * coefs[i];
        }
        coefs.swap(next);
    }
    std::vector<double> result(coefs.size());
    for (std::size_t i = 0; i < coefs.size(); ++i)
    {
        if (std::abs(coefs[i].imag()) > 1000.0 * kTolerance * std::max(1.0, std::abs(coefs[i])))
        {
            throw std::invalid_argument("roots do not define a real polynomial");
        }
        result[i] = coefs[i].real();
    }
    return result;
}

std::vector<std::complex<double>> polynomial_roots(std::span<const double> coefs)
{
    const auto values = trim_leading(coefs, "coefficients");
    const auto order = values.size() - 1;
    if (order == 0)
        return {};
#ifdef NEURALE_SIGNAL_WITH_MKL
    std::vector<double> companion(order * order, 0.0);
    for (std::size_t column = 0; column < order; ++column)
    {
        companion[column] = -values[column + 1] / values[0];
    }
    for (std::size_t row = 1; row < order; ++row)
    {
        companion[row * order + row - 1] = 1.0;
    }
    std::vector<double> real(order);
    std::vector<double> imag(order);
    const auto status =
        LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', static_cast<MKL_INT>(order), companion.data(),
                      static_cast<MKL_INT>(order), real.data(), imag.data(), nullptr,
                      static_cast<MKL_INT>(order), nullptr, static_cast<MKL_INT>(order));
    if (status != 0)
    {
        throw std::runtime_error("polynomial root solver did not converge");
    }
    std::vector<std::complex<double>> roots(order);
    for (std::size_t i = 0; i < order; ++i)
    {
        roots[i] = {real[i], imag[i]};
    }
    return roots;
#else
    std::vector<double> normalized(values);
    for (auto& value : normalized)
        value /= values[0];
    double radius = 1.0;
    for (std::size_t i = 1; i < normalized.size(); ++i)
    {
        radius = std::max(radius, std::abs(normalized[i]));
    }
    radius += 1.0;
    std::vector<std::complex<double>> roots(order);
    constexpr double pi = std::numbers::pi_v<double>;
    for (std::size_t i = 0; i < order; ++i)
    {
        const double angle = 2.0 * pi * (static_cast<double>(i) + 0.5) / static_cast<double>(order);
        roots[i] = std::polar(radius, angle);
    }
    for (std::size_t iteration = 0; iteration < 2000; ++iteration)
    {
        double max_update = 0.0;
        for (std::size_t i = 0; i < order; ++i)
        {
            const auto value = evaluate_polynomial(normalized, roots[i]);
            const auto derivative = evaluate_derivative(normalized, roots[i]);
            if (std::abs(derivative) <= kTolerance)
                continue;
            const auto newton = value / derivative;
            std::complex<double> repulsion = 0.0;
            for (std::size_t other = 0; other < order; ++other)
            {
                if (other != i)
                {
                    repulsion += 1.0 / (roots[i] - roots[other]);
                }
            }
            const auto update = newton / (1.0 - newton * repulsion);
            roots[i] -= update;
            max_update = std::max(max_update, std::abs(update));
        }
        if (max_update <= 1000.0 * kTolerance * std::max(1.0, static_cast<double>(order)))
        {
            return canonicalize_real_polynomial_roots(std::move(roots));
        }
    }
    throw std::runtime_error("polynomial root solver did not converge");
#endif
}

void zpk2tf(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
            std::complex<double> k, std::span<double> num, std::span<double> den)
{
    if (z.size() > p.size())
    {
        throw std::invalid_argument("system must be proper");
    }
    const auto size = p.size() + 1;
    if (num.size() != size || den.size() != size)
    {
        throw std::invalid_argument("transfer-function output size is invalid");
    }
    if (std::abs(k.imag()) > kTolerance * std::max(1.0, std::abs(k)))
    {
        throw std::invalid_argument("gain must be real");
    }
    auto num_poly = polynomial_from_roots(z);
    auto den_poly = polynomial_from_roots(p);
    for (auto& value : num_poly)
        value *= k.real();
    std::fill(num.begin(), num.end(), 0.0);
    std::copy(num_poly.begin(), num_poly.end(),
              num.begin() + static_cast<std::ptrdiff_t>(den_poly.size() - num_poly.size()));
    std::copy(den_poly.begin(), den_poly.end(), den.begin());
}

void tf2ss(std::span<const double> num_input, std::span<const double> den_input,
           std::span<double> a, std::span<double> b, std::span<double> c, double& d)
{
    auto num = trim_leading(num_input, "num");
    auto den = trim_leading(den_input, "den");
    if (num.size() > den.size())
    {
        throw std::invalid_argument("system must be proper");
    }
    const auto scale = den[0];
    for (auto& value : den)
        value /= scale;
    for (auto& value : num)
        value /= scale;
    num.insert(num.begin(), den.size() - num.size(), 0.0);
    const auto order = den.size() - 1;
    if (a.size() != order * order || b.size() != order || c.size() != order)
    {
        throw std::invalid_argument("state-space output size is invalid");
    }
    if (order == 0)
    {
        d = num[0];
        return;
    }
    std::fill(a.begin(), a.end(), 0.0);
    std::fill(b.begin(), b.end(), 0.0);
    std::fill(c.begin(), c.end(), 0.0);
    d = num[0];
    for (std::size_t column = 0; column < order; ++column)
    {
        a[column] = -den[column + 1];
        c[column] = num[column + 1] - d * den[column + 1];
    }
    for (std::size_t row = 1; row < order; ++row)
    {
        a[row * order + row - 1] = 1.0;
    }
    b[0] = 1.0;
}

void zpk2ss(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
            std::complex<double> k, std::span<double> a, std::span<double> b, std::span<double> c,
            double& d)
{
    if (z.size() > p.size())
    {
        throw std::invalid_argument("system must be proper");
    }
    if (std::abs(k.imag()) > kTolerance * std::max(1.0, std::abs(k)))
    {
        throw std::invalid_argument("gain must be real");
    }
    auto zeros = conjugate_pair_order(finite_roots(z));
    auto poles = conjugate_pair_order(finite_roots(p));
    const auto n = poles.size();
    if (a.size() != p.size() * p.size() || b.size() != p.size() || c.size() != p.size())
    {
        throw std::invalid_argument("state-space output size is invalid");
    }
    std::fill(a.begin(), a.end(), 0.0);
    std::fill(b.begin(), b.end(), 0.0);
    std::fill(c.begin(), c.end(), 0.0);
    d = 1.0;
    if (n == 0)
    {
        d = k.real();
        return;
    }
    if (n != p.size())
    {
        throw std::invalid_argument("state-space output cannot represent non-finite poles");
    }

    auto ip = static_cast<int>(poles.size());
    auto iz = static_cast<int>(zeros.size());
    bool odd_poles = false;
    bool odd_zeros = false;

    if (poles.size() % 2 == 1 && zeros.size() % 2 == 1)
    {
        a[0] = poles[n - 1].real();
        b[0] = 1.0;
        c[0] = poles[n - 1].real() - zeros[zeros.size() - 1].real();
        --ip;
        --iz;
        odd_poles = true;
    }
    else if (poles.size() % 2 == 1)
    {
        a[0] = poles[n - 1].real();
        b[0] = 1.0;
        c[0] = 1.0;
        d = 0.0;
        --ip;
        odd_poles = true;
    }
    else if (zeros.size() % 2 == 1)
    {
        const std::vector<std::complex<double>> section_zeros{zeros[zeros.size() - 1]};
        const std::vector<std::complex<double>> section_poles{poles[n - 2], poles[n - 1]};
        const auto num = polynomial_from_ordered_roots(section_zeros);
        const auto den = polynomial_from_ordered_roots(section_poles);
        const auto wn = section_wn(poles, n - 2);

        a[0] = -den[1];
        a[1] = -den[2] / wn;
        a[n] = wn;
        b[0] = 1.0;
        c[0] = 1.0;
        c[1] = num[1] / wn;
        d = 0.0;
        --iz;
        ip -= 2;
        odd_zeros = true;
    }

    auto section_start = [&](int idx)
    {
        if (odd_poles)
            return idx - 1;
        if (odd_zeros)
            return idx;
        return idx - 2;
    };
    auto append_section = [&](int j, const double section_a[4], const double section_b[2],
                              const double section_c[2], double section_d)
    {
        if (j == -1)
        {
            a[0] = section_a[0];
            a[1] = section_a[1];
            a[n] = section_a[2];
            a[n + 1] = section_a[3];
        }
        else
        {
            const auto offset = static_cast<std::size_t>(j + 1);
            for (std::size_t column = 0; column < offset; ++column)
            {
                a[offset * n + column] = section_b[0] * c[column];
                a[(offset + 1) * n + column] = section_b[1] * c[column];
            }
            a[offset * n + offset] = section_a[0];
            a[offset * n + offset + 1] = section_a[1];
            a[(offset + 1) * n + offset] = section_a[2];
            a[(offset + 1) * n + offset + 1] = section_a[3];
            if (section_d == 0.0)
            {
                std::fill(c.begin(), c.begin() + static_cast<std::ptrdiff_t>(offset), 0.0);
            }
        }

        const auto out = static_cast<std::size_t>(j + 1);
        b[out] = section_b[0] * d;
        b[out + 1] = section_b[1] * d;
        c[out] = section_c[0];
        c[out + 1] = section_c[1];
        d *= section_d;
    };

    int idx = 1;
    for (; idx < iz; idx += 2)
    {
        const std::vector<std::complex<double>> section_zeros{
            zeros[static_cast<std::size_t>(idx - 1)], zeros[static_cast<std::size_t>(idx)]};
        const std::vector<std::complex<double>> section_poles{
            poles[static_cast<std::size_t>(idx - 1)], poles[static_cast<std::size_t>(idx)]};
        const auto num = polynomial_from_ordered_roots(section_zeros);
        const auto den = polynomial_from_ordered_roots(section_poles);
        const auto wn = section_wn(poles, static_cast<std::size_t>(idx - 1));
        const double section_a[4] = {-den[1], -den[2] / wn, wn, 0.0};
        const double section_b[2] = {1.0, 0.0};
        const double section_c[2] = {num[1] - den[1], (num[2] - den[2]) / wn};
        append_section(section_start(idx), section_a, section_b, section_c, 1.0);
    }

    for (; idx < ip; idx += 2)
    {
        const std::vector<std::complex<double>> section_poles{
            poles[static_cast<std::size_t>(idx - 1)], poles[static_cast<std::size_t>(idx)]};
        const auto den = polynomial_from_ordered_roots(section_poles);
        const auto wn = section_wn(poles, static_cast<std::size_t>(idx - 1));
        const double section_a[4] = {-den[1], -den[2] / wn, wn, 0.0};
        const double section_b[2] = {1.0, 0.0};
        const double section_c[2] = {0.0, 1.0 / wn};
        append_section(section_start(idx), section_a, section_b, section_c, 0.0);
    }

    for (auto& value : c)
        value *= k.real();
    d *= k.real();
}

TransferFunction ss2tf(const StateSpace& state)
{
    validate_state(state);
    if (state.order == 0)
        return {{state.d}, {1.0}};
    const auto n = state.order;
    auto den = characteristic_polynomial(state);
    std::vector<double> impulse(n + 1, 0.0);
    impulse[0] = state.d;
    auto state_power = state.b;
    for (std::size_t step = 1; step <= n; ++step)
    {
        for (std::size_t i = 0; i < n; ++i)
        {
            impulse[step] += state.c[i] * state_power[i];
        }
        std::vector<double> next(n, 0.0);
        for (std::size_t row = 0; row < n; ++row)
        {
            for (std::size_t column = 0; column < n; ++column)
            {
                next[row] += state.a[row * n + column] * state_power[column];
            }
        }
        state_power.swap(next);
    }
    std::vector<double> num(n + 1, 0.0);
    for (std::size_t i = 0; i <= n; ++i)
    {
        for (std::size_t lag = 0; lag <= i; ++lag)
        {
            num[i] += den[lag] * impulse[i - lag];
        }
    }
    return {std::move(num), std::move(den)};
}

Zpk tf2zpk(std::span<const double> num_input, std::span<const double> den_input)
{
    auto den = trim_leading(den_input, "den");
    for (const auto value : num_input)
        require_finite(value, "num");
    const bool zero_num =
        std::all_of(num_input.begin(), num_input.end(), [](double value) { return value == 0.0; });
    if (zero_num)
    {
        return {{}, polynomial_roots(den), {0.0, 0.0}};
    }
    auto num = trim_leading(num_input, "num");
    if (num.size() > den.size())
    {
        throw std::invalid_argument("system must be proper");
    }
    const auto k = num[0] / den[0];
    return {polynomial_roots(num), polynomial_roots(den), {k, 0.0}};
}

Zpk ss2zpk(const StateSpace& state)
{
    validate_state(state);
#ifdef NEURALE_SIGNAL_WITH_MKL
    const auto n = state.order;
    if (n == 0)
        return {{}, {}, {state.d, 0.0}};
    auto matrix = state.a;
    std::vector<double> pole_real(n), pole_imag(n);
    auto status = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'N', static_cast<MKL_INT>(n), matrix.data(),
                                static_cast<MKL_INT>(n), pole_real.data(), pole_imag.data(),
                                nullptr, static_cast<MKL_INT>(n), nullptr, static_cast<MKL_INT>(n));
    if (status != 0)
        throw std::runtime_error("pole solver failed");
    std::vector<std::complex<double>> p(n);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = {pole_real[i], pole_imag[i]};

    const auto dim = n + 1;
    std::vector<double> left(dim * dim, 0.0);
    std::vector<double> right(dim * dim, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t column = 0; column < n; ++column)
        {
            left[row * dim + column] = state.a[row * n + column];
        }
        left[row * dim + n] = state.b[row];
        left[n * dim + row] = state.c[row];
        right[row * dim + row] = 1.0;
    }
    left[n * dim + n] = state.d;
    std::vector<double> alpha_real(dim), alpha_imag(dim);
    std::vector<double> beta(dim);
    status = LAPACKE_dggev(LAPACK_ROW_MAJOR, 'N', 'N', static_cast<MKL_INT>(dim), left.data(),
                           static_cast<MKL_INT>(dim), right.data(), static_cast<MKL_INT>(dim),
                           alpha_real.data(), alpha_imag.data(), beta.data(), nullptr,
                           static_cast<MKL_INT>(dim), nullptr, static_cast<MKL_INT>(dim));
    if (status != 0)
        throw std::runtime_error("zero solver failed");
    std::vector<std::complex<double>> z;
    for (std::size_t i = 0; i < dim; ++i)
    {
        if (std::abs(beta[i]) >
            kTolerance * std::max(1.0, std::hypot(alpha_real[i], alpha_imag[i])))
        {
            z.emplace_back(alpha_real[i] / beta[i], alpha_imag[i] / beta[i]);
        }
    }
    double k = state.d;
    if (std::abs(k) <= std::numeric_limits<double>::epsilon())
    {
        auto power_b = state.b;
        for (std::size_t iteration = 0; iteration < n; ++iteration)
        {
            k = 0.0;
            for (std::size_t i = 0; i < n; ++i)
                k += state.c[i] * power_b[i];
            if (std::abs(k) > std::numeric_limits<double>::epsilon())
                break;
            std::vector<double> next(n, 0.0);
            for (std::size_t row = 0; row < n; ++row)
                for (std::size_t column = 0; column < n; ++column)
                    next[row] += state.a[row * n + column] * power_b[column];
            power_b.swap(next);
        }
    }
    return {std::move(z), std::move(p), {k, 0.0}};
#else
    const auto tf = ss2tf(state);
    return tf2zpk(tf.num, tf.den);
#endif
}

void zpk2sos(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
             double k, std::span<double> sos)
{
    if (z.size() > p.size())
        throw std::invalid_argument("system must be proper");
    const auto ordered_zeros = conjugate_pair_order(z);
    const auto ordered_poles = conjugate_pair_order(p);
    const auto sections = std::max<std::size_t>(1, (ordered_poles.size() + 1) / 2);
    if (sos.size() != sections * 6)
    {
        throw std::invalid_argument("SOS output size is invalid");
    }

    auto split_zeros = split_ordered_roots(ordered_zeros);
    auto split_poles = split_ordered_roots(ordered_poles);
    std::vector<std::complex<double>> zeros;
    std::vector<std::complex<double>> poles;
    order_sos_roots_for_pairing(split_poles.complex, split_poles.real, split_zeros.complex,
                                split_zeros.real, zeros, poles);

    std::vector<double> output;
    output.reserve(sections * 6);
    if (zeros.empty())
    {
        if (poles.empty())
        {
            output.insert(output.end(), {0.0, 0.0, 1.0, 0.0, 0.0, 1.0});
        }
        else if (poles.size() % 2 == 0)
        {
            append_ordered_sos_pairs(zeros, poles, 0, 2 * sections - 1, output);
        }
        else
        {
            append_ordered_sos_pairs(zeros, poles, 0, 2 * (sections - 1) - 1, output);
            append_ordered_last_pole(
                {}, {poles.data() + static_cast<std::ptrdiff_t>(poles.size() - 1), 1}, output);
        }
    }
    else if (zeros.size() % 2 == 0)
    {
        append_ordered_sos_pairs(zeros, poles, 0, zeros.size() - 1, output);
        if (poles.size() % 2 == 0)
        {
            append_ordered_sos_pairs(zeros, poles, zeros.size(), poles.size(), output);
        }
        else
        {
            append_ordered_sos_pairs(zeros, poles, zeros.size(), poles.size() - 1, output);
            append_ordered_last_pole(
                {}, {poles.data() + static_cast<std::ptrdiff_t>(poles.size() - 1), 1}, output);
        }
    }
    else
    {
        append_ordered_sos_pairs(zeros, poles, 0, zeros.size() - 1, output);
        std::span<const std::complex<double>> last_zero{
            zeros.data() + static_cast<std::ptrdiff_t>(zeros.size() - 1), 1};
        if (zeros.size() == poles.size())
        {
            std::span<const std::complex<double>> last_pole{
                poles.data() + static_cast<std::ptrdiff_t>(poles.size() - 1), 1};
            append_ordered_last_pole(last_zero, last_pole, output);
        }
        else
        {
            const auto start = zeros.size() - 1;
            std::span<const std::complex<double>> section_poles{
                poles.data() + static_cast<std::ptrdiff_t>(start), 2};
            const auto num = padded_quadratic(polynomial_from_ordered_roots(last_zero));
            const auto den = padded_quadratic(polynomial_from_ordered_roots(section_poles));
            output.insert(output.end(), {den[2], den[1], den[0], num[2], num[1], num[0]});
            if (poles.size() % 2 == 0)
            {
                append_ordered_sos_pairs(zeros, poles, zeros.size() + 1, poles.size(), output);
            }
            else
            {
                append_ordered_sos_pairs(zeros, poles, zeros.size() + 1, poles.size() - 1, output);
                append_ordered_last_pole(
                    {}, {poles.data() + static_cast<std::ptrdiff_t>(poles.size() - 1), 1}, output);
            }
        }
    }
    std::reverse(output.begin(), output.end());
    std::copy(output.begin(), output.end(), sos.begin());
    for (std::size_t i = 0; i < 3; ++i)
        sos[i] *= k;
}

StateSpace lp2lp_ss(const StateSpace& state, double cutoff)
{
    validate_state(state);
    require_positive(cutoff, "cutoff");
    auto result = state;
    for (auto& value : result.a)
        value *= cutoff;
    for (auto& value : result.b)
        value *= cutoff;
    return result;
}

StateSpace lp2hp_ss(const StateSpace& state, double cutoff)
{
    validate_state(state);
    require_positive(cutoff, "cutoff");
    const auto n = state.order;
    std::vector<double> identity(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        identity[i * n + i] = 1.0;
    const auto inverse = solve(state.a, identity, n, n);
    StateSpace result{n, inverse, {}, {}, state.d};
    for (auto& value : result.a)
        value *= cutoff;
    result.b.assign(n, 0.0);
    result.c.assign(n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t column = 0; column < n; ++column)
        {
            result.b[row] -= cutoff * inverse[row * n + column] * state.b[column];
            result.c[column] += state.c[row] * inverse[row * n + column];
        }
    }
    for (std::size_t i = 0; i < n; ++i)
        result.d -= result.c[i] * state.b[i];
    return result;
}

StateSpace lp2bp_ss(const StateSpace& state, double center, double bandwidth)
{
    validate_state(state);
    require_positive(center, "center_frequency");
    require_positive(bandwidth, "bandwidth");
    const auto n = state.order;
    StateSpace result{2 * n, {}, {}, {}, state.d};
    result.a.assign(4 * n * n, 0.0);
    result.b.assign(2 * n, 0.0);
    result.c.assign(2 * n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t column = 0; column < n; ++column)
            result.a[row * 2 * n + column] = bandwidth * state.a[row * n + column];
        result.a[row * 2 * n + n + row] = center;
        result.a[(n + row) * 2 * n + row] = -center;
        result.b[row] = bandwidth * state.b[row];
        result.c[row] = state.c[row];
    }
    return result;
}

StateSpace lp2bs_ss(const StateSpace& state, double center, double bandwidth)
{
    validate_state(state);
    require_positive(center, "center_frequency");
    require_positive(bandwidth, "bandwidth");
    const auto n = state.order;
    std::vector<double> identity(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        identity[i * n + i] = 1.0;
    const auto inverse = solve(state.a, identity, n, n);
    StateSpace result{2 * n, {}, {}, {}, state.d};
    result.a.assign(4 * n * n, 0.0);
    result.b.assign(2 * n, 0.0);
    result.c.assign(2 * n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t column = 0; column < n; ++column)
        {
            result.a[row * 2 * n + column] = bandwidth * inverse[row * n + column];
            result.b[row] -= bandwidth * inverse[row * n + column] * state.b[column];
            result.c[column] += state.c[row] * inverse[row * n + column];
        }
        result.a[row * 2 * n + n + row] = center;
        result.a[(n + row) * 2 * n + row] = -center;
    }
    for (std::size_t i = 0; i < n; ++i)
        result.d -= result.c[i] * state.b[i];
    return result;
}

void bilinear_zpk(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
                  std::complex<double> k, double rate, double prewarp,
                  std::span<std::complex<double>> digital_z,
                  std::span<std::complex<double>> digital_p, std::complex<double>& digital_k)
{
    if (z.size() > p.size())
    {
        throw std::invalid_argument("system must be proper");
    }
    if (digital_z.size() != p.size() || digital_p.size() != p.size())
    {
        throw std::invalid_argument("bilinear ZPK output size is invalid");
    }
    require_finite(k, "k");
    const auto effective = effective_rate(rate, prewarp);
    const auto scale = 2.0 * effective;
    std::complex<double> num_product{1.0, 0.0};
    std::complex<double> den_product{1.0, 0.0};
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        const auto zero = z[i];
        require_finite(zero, "z");
        digital_z[i] = (scale + zero) / (scale - zero);
        require_finite(digital_z[i], "digital z");
        num_product *= scale - zero;
    }
    std::fill(digital_z.begin() + static_cast<std::ptrdiff_t>(z.size()), digital_z.end(),
              std::complex<double>{-1.0, 0.0});
    for (std::size_t i = 0; i < p.size(); ++i)
    {
        const auto pole = p[i];
        require_finite(pole, "p");
        digital_p[i] = (scale + pole) / (scale - pole);
        require_finite(digital_p[i], "digital p");
        den_product *= scale - pole;
    }
    digital_k = k * num_product / den_product;
    require_finite(digital_k, "digital k");
}

void bilinear_ss(std::span<const double> a, std::span<const double> b, std::span<const double> c,
                 double d, std::size_t n, double rate, double prewarp, std::span<double> digital_a,
                 std::span<double> digital_b, std::span<double> digital_c, double& digital_d)
{
    if (a.size() != n * n || b.size() != n || c.size() != n)
    {
        throw std::invalid_argument("state-space buffer shape is invalid");
    }
    if (digital_a.size() != n * n || digital_b.size() != n || digital_c.size() != n)
    {
        throw std::invalid_argument("bilinear state-space output size is invalid");
    }
    for (const auto value : a)
        require_finite(value, "a");
    for (const auto value : b)
        require_finite(value, "b");
    for (const auto value : c)
        require_finite(value, "c");
    require_finite(d, "d");
    const auto effective = effective_rate(rate, prewarp);
    if (n == 0)
    {
        digital_d = d;
        return;
    }
    const auto step = 1.0 / effective;
    std::vector<double> left(n * n, 0.0), right(n * n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t column = 0; column < n; ++column)
        {
            const auto value = 0.5 * step * a[row * n + column];
            left[row * n + column] = -value;
            right[row * n + column] = value;
        }
        left[row * n + row] += 1.0;
        right[row * n + row] += 1.0;
    }
    std::copy(right.begin(), right.end(), digital_a.begin());
    std::copy(b.begin(), b.end(), digital_b.begin());
    std::copy(c.begin(), c.end(), digital_c.begin());
    solve_into(left, digital_a, n, n);
    solve_into(left, digital_b, n, 1);
    const auto root_step = std::sqrt(step);
    for (auto& value : digital_b)
        value *= root_step;
    std::vector<double> left_transpose(n * n);
    for (std::size_t row = 0; row < n; ++row)
        for (std::size_t column = 0; column < n; ++column)
            left_transpose[row * n + column] = left[column * n + row];
    solve_into(left_transpose, digital_c, n, 1);
    for (auto& value : digital_c)
        value *= root_step;
    digital_d = d;
    for (std::size_t i = 0; i < n; ++i)
        digital_d += 0.5 * root_step * digital_c[i] * b[i];
}

std::size_t bilinear_tf_size(std::span<const double> num, std::span<const double> den)
{
    const auto trimmed_num = trim_leading(num, "num");
    const auto trimmed_den = trim_leading(den, "den");
    if (trimmed_num.size() > trimmed_den.size())
    {
        throw std::invalid_argument("system must be proper");
    }
    return trimmed_den.size();
}

void bilinear_tf(std::span<const double> num, std::span<const double> den, double rate,
                 double prewarp, std::span<double> digital_num, std::span<double> digital_den)
{
    const auto size = bilinear_tf_size(num, den);
    if (digital_num.size() != size || digital_den.size() != size)
    {
        throw std::invalid_argument("bilinear TF output size is invalid");
    }
    const auto zpk = tf2zpk(num, den);
    std::vector<std::complex<double>> digital_z(zpk.p.size());
    std::vector<std::complex<double>> digital_p(zpk.p.size());
    std::complex<double> digital_k;
    bilinear_zpk(zpk.z, zpk.p, zpk.k, rate, prewarp, digital_z, digital_p, digital_k);
    zpk2tf(digital_z, digital_p, digital_k, digital_num, digital_den);
}

} // namespace neurale::signal
