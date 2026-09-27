/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/signal/iir.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <vector>

namespace neurale::signal
{
namespace
{

using Complex = std::complex<double>;
constexpr double pi = std::numbers::pi_v<double>;
constexpr double tol = 1e-14;

double product_real(std::span<const Complex> num, std::span<const Complex> den)
{
    Complex value = 1.0;
    for (const auto root : num)
        value *= -root;
    for (const auto root : den)
        value /= -root;
    return value.real();
}

double polynomial_value(double x, std::initializer_list<double> values)
{
    double result = 0.0;
    for (auto iterator = values.end(); iterator != values.begin();)
    {
        result = result * x + *--iterator;
    }
    return result;
}

std::pair<Complex, Complex> polynomial_and_derivative(std::span<const double> coefs, Complex value)
{
    Complex polynomial = coefs[0];
    Complex derivative = 0.0;
    for (std::size_t i = 1; i < coefs.size(); ++i)
    {
        derivative = derivative * value + polynomial;
        polynomial = polynomial * value + coefs[i];
    }
    return {polynomial, derivative};
}

std::vector<Complex> bessel_zeros(std::span<const double> coefs)
{
    const std::size_t order = coefs.size() - 1;
    if (order == 1)
        return {{-1.0, 0.0}};
    const double n = static_cast<double>(order);
    const double s = polynomial_value(n, {0, 0, 2, 0, -3, 1});
    const double b3 = polynomial_value(n, {16, -8}) / s;
    const double b2 = polynomial_value(n, {-24, -12, 12}) / s;
    const double b1 = polynomial_value(n, {8, 24, -12, -2}) / s;
    const double b0 = polynomial_value(n, {0, -6, 0, 5, -1}) / s;
    const double r = polynomial_value(n, {0, 0, 2, 1});
    const double a1 = polynomial_value(n, {-6, -6}) / r;
    const double a2 = 6.0 / r;
    std::vector<Complex> roots(order);
    for (std::size_t i = 0; i < order; ++i)
    {
        const double k = static_cast<double>(i + 1);
        roots[i] = {polynomial_value(k, {0, a1, a2}), polynomial_value(k, {b0, b1, b2, b3})};
    }
    for (int iteration = 0; iteration < 500; ++iteration)
    {
        double maximum_change = 0.0;
        const auto previous = roots;
        for (std::size_t i = 0; i < order; ++i)
        {
            const auto [value, derivative] = polynomial_and_derivative(coefs, previous[i]);
            const Complex newton = value / derivative;
            Complex sum = 0.0;
            for (std::size_t other = 0; other < order; ++other)
            {
                if (other != i)
                {
                    sum += 1.0 / (previous[i] - previous[other]);
                }
            }
            const Complex correction = newton / (1.0 - newton * sum);
            roots[i] -= correction;
            maximum_change = std::max(maximum_change, std::abs(correction));
        }
        if (maximum_change < 1e-13)
            break;
    }
    for (auto& root : roots)
    {
        for (int iteration = 0; iteration < 10; ++iteration)
        {
            const auto [value, derivative] = polynomial_and_derivative(coefs, root);
            const Complex correction = value / derivative;
            root -= correction;
            if (std::abs(correction) < 1e-15)
                break;
        }
    }
    std::sort(roots.begin(), roots.end(),
              [](Complex left, Complex right) { return left.imag() < right.imag(); });
    for (std::size_t i = 0; i < order / 2; ++i)
    {
        const Complex averaged = 0.5 * (roots[i] + std::conj(roots[order - i - 1]));
        roots[i] = averaged;
        roots[order - i - 1] = std::conj(averaged);
    }
    if (order % 2)
    {
        roots[order / 2] = {roots[order / 2].real(), 0.0};
    }
    return roots;
}

Zpk butterworth_prototype(std::size_t order)
{
    Zpk result;
    result.k = 1.0;
    for (std::ptrdiff_t value = -static_cast<std::ptrdiff_t>(order) + 1;
         value < static_cast<std::ptrdiff_t>(order); value += 2)
    {
        result.p.push_back(-std::exp(
            Complex{0.0, pi * static_cast<double>(value) / (2.0 * static_cast<double>(order))}));
    }
    return result;
}

Zpk bessel_prototype(std::size_t order)
{
    std::vector<double> ascending(order + 1);
    for (std::size_t k = 0; k <= order; ++k)
    {
        const double logarithm = std::lgamma(2.0 * order - k + 1.0) - (order - k) * std::log(2.0) -
                                 std::lgamma(k + 1.0) - std::lgamma(order - k + 1.0);
        ascending[k] = std::exp(logarithm);
    }
    auto ordinary_zeros = bessel_zeros(ascending);
    std::vector<Complex> poles(order);
    std::transform(ordinary_zeros.begin(), ordinary_zeros.end(), poles.begin(),
                   [](Complex zero) { return 1.0 / zero; });
    const double a_last = ascending[0];
    const double scale = std::pow(10.0, -std::log10(a_last) / order);
    for (auto& pole : poles)
        pole *= scale;
    return {{}, std::move(poles), 1.0};
}

std::vector<double> landen(double modulus)
{
    std::vector<double> values;
    if (modulus == 0.0 || modulus == 1.0)
        return {modulus};
    while (modulus > tol)
    {
        modulus = std::pow(modulus / (1.0 + std::sqrt(1.0 - modulus * modulus)), 2.0);
        values.push_back(modulus);
        if (values.size() > 100)
        {
            throw std::runtime_error("Landen sequence did not converge");
        }
    }
    return values;
}

double complete_elliptic_k(double modulus, double& complement)
{
    const auto values = landen(modulus);
    double product = 1.0;
    for (double value : values)
        product *= 1.0 + value;
    const double result =
        modulus == 1.0 ? std::numeric_limits<double>::infinity() : product * pi / 2.0;
    const double complementary_modulus = std::sqrt((1.0 - modulus) * (1.0 + modulus));
    if (modulus == 0.0)
    {
        complement = std::numeric_limits<double>::infinity();
    }
    else if (complementary_modulus < 1e-6)
    {
        const double log_value = -std::log(complementary_modulus / 4.0);
        complement =
            log_value + (log_value - 1.0) * complementary_modulus * complementary_modulus / 4.0;
    }
    else
    {
        const auto complementary_values = landen(complementary_modulus);
        double complementary_product = 1.0;
        for (double value : complementary_values)
        {
            complementary_product *= 1.0 + value;
        }
        complement = complementary_product * pi / 2.0;
    }
    return result;
}

Complex jacobi_sn(Complex value, double modulus)
{
    auto values = landen(modulus);
    Complex result = std::sin(value * pi / 2.0);
    for (auto iterator = values.rbegin(); iterator != values.rend(); ++iterator)
    {
        result = (1.0 + *iterator) * result / (1.0 + *iterator * result * result);
    }
    return result;
}

Complex jacobi_cd(Complex value, double modulus)
{
    auto values = landen(modulus);
    Complex result = std::cos(value * pi / 2.0);
    for (auto iterator = values.rbegin(); iterator != values.rend(); ++iterator)
    {
        result = (1.0 + *iterator) * result / (1.0 + *iterator * result * result);
    }
    return result;
}

Complex inverse_jacobi_sn(Complex value, double modulus)
{
    auto values = landen(modulus);
    for (std::size_t i = 0; i < values.size(); ++i)
    {
        const double previous = i == 0 ? modulus : values[i - 1];
        value = value / (1.0 + std::sqrt(1.0 - value * value * previous * previous)) * 2.0 /
                (1.0 + values[i]);
    }
    Complex result =
        value == Complex{1.0, 0.0} ? Complex{1.0, 0.0} : 1.0 - (2.0 / pi) * std::acos(value);
    double complement = 0.0;
    const double k = complete_elliptic_k(modulus, complement);
    const double period = complement / k;
    result.real(result.real() - 4.0 * std::round(result.real() / 4.0));
    result.imag(result.imag() - 2.0 * period * std::round(result.imag() / (2.0 * period)));
    return result;
}

double elliptic_degree(std::size_t order, double k1)
{
    double complement = 0.0;
    const double k = complete_elliptic_k(k1, complement);
    const double q = std::exp(-pi * complement / k);
    const double qn = std::pow(q, 1.0 / order);
    double num = 1.0;
    double den = 1.0;
    for (int i = 1; i <= 7; ++i)
    {
        num += std::pow(qn, i * (i + 1));
        den += 2.0 * std::pow(qn, i * i);
    }
    return 4.0 * std::sqrt(qn) * std::pow(num / den, 2.0);
}

Zpk elliptic_prototype(std::size_t order, double rp, double rs)
{
    if (!(rp > 0.0) || !(rs > rp))
    {
        throw std::invalid_argument("elliptic ripple must satisfy 0 < rp < rs");
    }
    if (order == 1)
    {
        const double pole = -std::sqrt(1.0 / std::expm1(0.1 * rp * std::log(10.0)));
        return {{}, {{pole, 0.0}}, {-pole, 0.0}};
    }
    const double eps = std::sqrt(std::expm1(0.1 * rp * std::log(10.0)));
    const double stop = std::sqrt(std::expm1(0.1 * rs * std::log(10.0)));
    const double k1 = eps / stop;
    const double modulus = elliptic_degree(order, k1);
    const std::size_t half = order / 2;
    std::vector<Complex> zeros;
    std::vector<Complex> poles;
    for (std::size_t i = 1; i <= half; ++i)
    {
        const double u = (2.0 * i - 1.0) / static_cast<double>(order);
        const Complex zeta = jacobi_cd({u, 0.0}, modulus);
        const Complex zero = Complex{0.0, 1.0} / (modulus * zeta);
        zeros.push_back(zero);
        zeros.push_back(std::conj(zero));
    }
    Complex v0 = Complex{0.0, -1.0} * inverse_jacobi_sn(Complex{0.0, 1.0 / eps}, k1) /
                 static_cast<double>(order);
    for (std::size_t i = 1; i <= half; ++i)
    {
        const double u = (2.0 * i - 1.0) / static_cast<double>(order);
        const Complex pole =
            Complex{0.0, 1.0} * jacobi_cd(Complex{u, 0.0} - Complex{0.0, 1.0} * v0, modulus);
        poles.push_back(pole);
        poles.push_back(std::conj(pole));
    }
    if (order % 2)
    {
        poles.push_back(Complex{0.0, 1.0} * jacobi_sn(Complex{0.0, 1.0} * v0, modulus));
    }
    Complex gain = 1.0;
    for (const auto pole : poles)
        gain *= -pole;
    for (const auto zero : zeros)
        gain /= -zero;
    if (order % 2 == 0)
    {
        gain /= std::sqrt(1.0 + eps * eps);
    }
    return {std::move(zeros), std::move(poles), gain.real()};
}

Zpk transform_lowpass(Zpk prototype, std::span<const double> cutoff, std::string_view band)
{
    const auto degree = prototype.p.size() - prototype.z.size();
    if (band == "lowpass")
    {
        const double freq = cutoff[0];
        for (auto& zero : prototype.z)
            zero *= freq;
        for (auto& pole : prototype.p)
            pole *= freq;
        prototype.k *= std::pow(freq, degree);
        return prototype;
    }
    if (band == "highpass")
    {
        const double freq = cutoff[0];
        const double gain = prototype.k.real() * product_real(prototype.z, prototype.p);
        for (auto& zero : prototype.z)
            zero = freq / zero;
        for (auto& pole : prototype.p)
            pole = freq / pole;
        prototype.z.insert(prototype.z.end(), degree, 0.0);
        prototype.k = gain;
        return prototype;
    }
    const double center = std::sqrt(cutoff[0] * cutoff[1]);
    const double bandwidth = cutoff[1] - cutoff[0];
    std::vector<Complex> zeros;
    std::vector<Complex> poles;
    const bool pass = band == "bandpass";
    for (const auto root : prototype.z)
    {
        const Complex scaled = pass ? root * bandwidth / 2.0 : (bandwidth / 2.0) / root;
        const Complex radical = std::sqrt(scaled * scaled - center * center);
        zeros.push_back(scaled + radical);
        zeros.push_back(scaled - radical);
    }
    for (const auto root : prototype.p)
    {
        const Complex scaled = pass ? root * bandwidth / 2.0 : (bandwidth / 2.0) / root;
        const Complex radical = std::sqrt(scaled * scaled - center * center);
        poles.push_back(scaled + radical);
        poles.push_back(scaled - radical);
    }
    if (pass)
    {
        zeros.insert(zeros.end(), degree, 0.0);
        prototype.k *= std::pow(bandwidth, degree);
    }
    else
    {
        zeros.insert(zeros.end(), degree, Complex{0.0, center});
        zeros.insert(zeros.end(), degree, Complex{0.0, -center});
        prototype.k *= product_real(prototype.z, prototype.p);
    }
    return {std::move(zeros), std::move(poles), prototype.k};
}

Zpk bilinear(Zpk analog, double fs)
{
    const auto degree = analog.p.size() - analog.z.size();
    const double twice_rate = 2.0 * fs;
    Complex ratio = 1.0;
    for (auto& zero : analog.z)
    {
        ratio *= twice_rate - zero;
        zero = (twice_rate + zero) / (twice_rate - zero);
    }
    for (auto& pole : analog.p)
    {
        ratio /= twice_rate - pole;
        pole = (twice_rate + pole) / (twice_rate - pole);
    }
    analog.z.insert(analog.z.end(), degree, -1.0);
    analog.k *= ratio.real();
    return analog;
}

std::vector<Complex> sort_real_last_conjugate_pairs(std::span<const Complex> x)
{
    constexpr double pair_tol = 100.0 * std::numeric_limits<double>::epsilon();
    std::vector<Complex> real;
    std::vector<Complex> complex_roots;
    real.reserve(x.size());
    complex_roots.reserve(x.size());
    for (const auto value : x)
    {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag()))
        {
            continue;
        }
        if (std::abs(value.imag()) <= pair_tol * std::abs(value))
        {
            real.emplace_back(value.real(), 0.0);
        }
        else
        {
            complex_roots.push_back(value);
        }
    }
    std::sort(real.begin(), real.end(),
              [](Complex left, Complex right) { return left.real() < right.real(); });
    std::sort(complex_roots.begin(), complex_roots.end(),
              [](Complex left, Complex right) { return left.real() < right.real(); });
    if (complex_roots.size() % 2 != 0)
    {
        throw std::invalid_argument("roots must contain conjugate pairs");
    }
    std::vector<Complex> result(complex_roots.size() + real.size());
    std::size_t group_begin = 0;
    while (group_begin < complex_roots.size())
    {
        const double current_real = complex_roots[group_begin].real();
        std::size_t group_end = group_begin;
        while (group_end < complex_roots.size() &&
               std::abs(complex_roots[group_end].real() - current_real) <=
                   pair_tol * std::abs(complex_roots[group_end]))
        {
            ++group_end;
        }
        const auto count = group_end - group_begin;
        if (count % 2 != 0)
        {
            throw std::invalid_argument("roots must contain conjugate pairs");
        }
        std::vector<Complex> group(complex_roots.begin() + static_cast<std::ptrdiff_t>(group_begin),
                                   complex_roots.begin() + static_cast<std::ptrdiff_t>(group_end));
        std::sort(group.begin(), group.end(),
                  [](Complex left, Complex right) { return left.imag() < right.imag(); });
        for (std::size_t i = 0; i < count; ++i)
        {
            if (std::abs(group[i].imag() + group[count - 1 - i].imag()) >
                pair_tol * std::abs(group[i]))
            {
                throw std::invalid_argument("roots must contain conjugate pairs");
            }
        }
        for (std::size_t i = 0; i < count; i += 2)
        {
            const auto pole = group[count - 1 - i / 2];
            result[group_begin + i] = std::conj(pole);
            result[group_begin + i + 1] = pole;
        }
        group_begin = group_end;
    }
    std::copy(real.begin(), real.end(),
              result.begin() + static_cast<std::ptrdiff_t>(complex_roots.size()));
    return result;
}

std::vector<double> polynomial_from_ordered_roots(std::span<const Complex> roots)
{
    std::vector<Complex> coefs{1.0};
    for (const auto root : roots)
    {
        std::vector<Complex> next(coefs.size() + 1, 0.0);
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

double pair_natural_frequency(std::span<const Complex> poles, std::size_t first)
{
    double value = std::sqrt(std::abs(poles[first]) * std::abs(poles[first + 1]));
    return value == 0.0 ? 1.0 : value;
}

void write_section_cascade(StateSpace& state, std::size_t offset,
                           std::span<const double, 4> section_a,
                           std::span<const double, 2> section_b,
                           std::span<const double, 2> section_c, double section_d)
{
    const auto n = state.order;
    if (offset == 0)
    {
        state.a[0] = section_a[0];
        state.a[1] = section_a[1];
        state.a[n] = section_a[2];
        state.a[n + 1] = section_a[3];
    }
    else
    {
        for (std::size_t row = 0; row < 2; ++row)
        {
            for (std::size_t column = 0; column < offset; ++column)
            {
                state.a[(offset + row) * n + column] = section_b[row] * state.c[column];
            }
        }
        state.a[offset * n + offset] = section_a[0];
        state.a[offset * n + offset + 1] = section_a[1];
        state.a[(offset + 1) * n + offset] = section_a[2];
        state.a[(offset + 1) * n + offset + 1] = section_a[3];
        for (std::size_t column = 0; column < offset; ++column)
        {
            state.c[column] *= section_d;
        }
    }
    state.b[offset] = section_b[0] * state.d;
    state.b[offset + 1] = section_b[1] * state.d;
    state.c[offset] = section_c[0];
    state.c[offset + 1] = section_c[1];
    state.d *= section_d;
}

StateSpace zpk_state_space(const Zpk& prototype)
{
    const auto zeros = sort_real_last_conjugate_pairs(prototype.z);
    const auto poles = sort_real_last_conjugate_pairs(prototype.p);
    if (zeros.size() > poles.size())
    {
        throw std::invalid_argument("system must be proper");
    }
    const auto n = poles.size();
    StateSpace state{n, std::vector<double>(n * n, 0.0), std::vector<double>(n, 0.0),
                     std::vector<double>(n, 0.0), 1.0};
    if (n == 0)
    {
        state.d = prototype.k.real();
        return state;
    }

    auto remaining_poles = static_cast<int>(poles.size());
    auto remaining_zeros = static_cast<int>(zeros.size());
    bool odd_poles = false;
    bool odd_zeros = false;
    if (poles.size() % 2 == 1 && zeros.size() % 2 == 1)
    {
        state.a[0] = poles[n - 1].real();
        state.b[0] = 1.0;
        state.c[0] = poles[n - 1].real() - zeros[zeros.size() - 1].real();
        state.d = 1.0;
        --remaining_poles;
        --remaining_zeros;
        odd_poles = true;
    }
    else if (poles.size() % 2 == 1)
    {
        state.a[0] = poles[n - 1].real();
        state.b[0] = 1.0;
        state.c[0] = 1.0;
        state.d = 0.0;
        --remaining_poles;
        odd_poles = true;
    }
    else if (zeros.size() % 2 == 1)
    {
        const auto num =
            polynomial_from_ordered_roots(std::span<const Complex>(&zeros[zeros.size() - 1], 1));
        const auto den = polynomial_from_ordered_roots(std::span<const Complex>(&poles[n - 2], 2));
        const double natural = pair_natural_frequency(poles, n - 2);
        state.a[0] = -den[1];
        state.a[1] = -den[2] / natural;
        state.a[n] = natural;
        state.b[0] = 1.0;
        state.c[0] = 1.0;
        state.c[1] = num[1] / natural;
        state.d = 0.0;
        --remaining_zeros;
        remaining_poles -= 2;
        odd_zeros = true;
    }

    int idx = 1;
    for (; idx < remaining_zeros; idx += 2)
    {
        const auto num = polynomial_from_ordered_roots(
            std::span<const Complex>(&zeros[static_cast<std::size_t>(idx - 1)], 2));
        const auto den = polynomial_from_ordered_roots(
            std::span<const Complex>(&poles[static_cast<std::size_t>(idx - 1)], 2));
        const double natural = pair_natural_frequency(poles, static_cast<std::size_t>(idx - 1));
        const double section_a_values[4]{-den[1], -den[2] / natural, natural, 0.0};
        const double section_b_values[2]{1.0, 0.0};
        const double section_c_values[2]{num[1] - den[1], (num[2] - den[2]) / natural};
        const int offset = odd_poles ? idx : (odd_zeros ? idx + 1 : idx - 1);
        write_section_cascade(state, static_cast<std::size_t>(offset),
                              std::span<const double, 4>(section_a_values),
                              std::span<const double, 2>(section_b_values),
                              std::span<const double, 2>(section_c_values), 1.0);
    }

    for (; idx < remaining_poles; idx += 2)
    {
        const auto den = polynomial_from_ordered_roots(
            std::span<const Complex>(&poles[static_cast<std::size_t>(idx - 1)], 2));
        const double natural = pair_natural_frequency(poles, static_cast<std::size_t>(idx - 1));
        const double section_a_values[4]{-den[1], -den[2] / natural, natural, 0.0};
        const double section_b_values[2]{1.0, 0.0};
        const double section_c_values[2]{0.0, 1.0 / natural};
        const int offset = odd_poles ? idx : (odd_zeros ? idx + 1 : idx - 1);
        write_section_cascade(state, static_cast<std::size_t>(offset),
                              std::span<const double, 4>(section_a_values),
                              std::span<const double, 2>(section_b_values),
                              std::span<const double, 2>(section_c_values), 0.0);
    }

    for (auto& value : state.c)
        value *= prototype.k.real();
    state.d *= prototype.k.real();
    return state;
}

Zpk analog_prototype(std::string_view kind, std::size_t order, double rp, double rs)
{
    if (kind == "butterworth")
    {
        return butterworth_prototype(order);
    }
    if (kind == "bessel")
    {
        if (order > 16)
        {
            throw std::invalid_argument("native Bessel order must not exceed 16");
        }
        return bessel_prototype(order);
    }
    if (kind == "elliptic")
    {
        return elliptic_prototype(order, rp, rs);
    }
    throw std::invalid_argument("unknown IIR design kind");
}

std::vector<double> validate_and_warp(std::size_t order, std::span<const double> cutoff,
                                      std::string_view band, double fs)
{
    if (order == 0)
    {
        throw std::invalid_argument("order must be positive");
    }
    if (!(fs > 0.0) || !std::isfinite(fs))
    {
        throw std::invalid_argument("fs must be positive");
    }
    const bool single = band == "lowpass" || band == "highpass";
    if ((!single && band != "bandpass" && band != "bandstop") || cutoff.size() != (single ? 1 : 2))
    {
        throw std::invalid_argument("invalid IIR band or cutoff count");
    }
    std::vector<double> warped(cutoff.size());
    for (std::size_t i = 0; i < cutoff.size(); ++i)
    {
        if (!(cutoff[i] > 0.0 && cutoff[i] < fs / 2.0))
        {
            throw std::invalid_argument("cutoff must lie strictly below Nyquist");
        }
        if (i > 0 && cutoff[i] <= cutoff[i - 1])
        {
            throw std::invalid_argument("cutoffs must be strictly increasing");
        }
        warped[i] = 2.0 * fs * std::tan(pi * cutoff[i] / fs);
    }
    return warped;
}

} // namespace

Zpk notch_zpk(double freq, double bandwidth)
{
    if (!(freq > 0.0 && freq < 1.0) || !(bandwidth > 0.0 && bandwidth < 1.0))
    {
        throw std::invalid_argument("notch frequency and bandwidth must lie in (0, 1)");
    }
    const double omega = pi * freq;
    const double cos_omega = std::cos(omega);
    const double beta = std::tan(pi * bandwidth / 2.0);
    const double gain = 1.0 / (1.0 + beta);
    const Complex zero{cos_omega, std::sin(omega)};

    const double a1 = -2.0 * gain * cos_omega;
    const double a2 = 2.0 * gain - 1.0;
    const Complex root = std::sqrt(Complex{a1 * a1 - 4.0 * a2, 0.0});
    const std::vector<Complex> poles{(-a1 - root) / 2.0, (-a1 + root) / 2.0};
    return {{std::conj(zero), zero}, poles, {gain, 0.0}};
}

Zpk iir_zpk(std::string_view kind, std::size_t order, std::span<const double> cutoff,
            std::string_view band, double fs, double rp, double rs)
{
    const auto warped = validate_and_warp(order, cutoff, band, fs);
    auto prototype = analog_prototype(kind, order, rp, rs);
    return bilinear(transform_lowpass(std::move(prototype), warped, band), fs);
}

TransferFunction iir_tf(std::string_view kind, std::size_t order, std::span<const double> cutoff,
                        std::string_view band, double fs, double rp, double rs)
{
    const auto zpk = iir_zpk(kind, order, cutoff, band, fs, rp, rs);
    TransferFunction tf;
    const auto size = zpk.p.size() + 1;
    tf.num.resize(size);
    tf.den.resize(size);
    zpk2tf(zpk.z, zpk.p, zpk.k, tf.num, tf.den);
    return tf;
}

std::vector<double> iir_sos(std::string_view kind, std::size_t order,
                            std::span<const double> cutoff, std::string_view band, double fs,
                            double rp, double rs)
{
    const auto zpk = iir_zpk(kind, order, cutoff, band, fs, rp, rs);
    std::vector<double> sos(std::max<std::size_t>(1, (zpk.p.size() + 1) / 2) * 6);
    zpk2sos(zpk.z, zpk.p, zpk.k.real(), sos);
    return sos;
}

StateSpace iir_ss(std::string_view kind, std::size_t order, std::span<const double> cutoff,
                  std::string_view band, double fs, double rp, double rs)
{
    if (!(fs > 0.0) || !std::isfinite(fs))
    {
        throw std::invalid_argument("fs must be positive");
    }
    std::vector<double> normalized(cutoff.size());
    const double nyquist = fs / 2.0;
    for (std::size_t i = 0; i < cutoff.size(); ++i)
    {
        normalized[i] = cutoff[i] / nyquist;
    }
    const auto warped = validate_and_warp(order, normalized, band, 2.0);
    auto state = zpk_state_space(analog_prototype(kind, order, rp, rs));
    if (band == "lowpass")
    {
        state = lp2lp_ss(state, warped[0]);
    }
    else if (band == "highpass")
    {
        state = lp2hp_ss(state, warped[0]);
    }
    else
    {
        const auto center = std::sqrt(warped[0] * warped[1]);
        const auto bandwidth = warped[1] - warped[0];
        state = band == "bandpass" ? lp2bp_ss(state, center, bandwidth)
                                   : lp2bs_ss(state, center, bandwidth);
    }
    StateSpace digital{state.order, std::vector<double>(state.a.size()),
                       std::vector<double>(state.b.size()), std::vector<double>(state.c.size()),
                       0.0};
    bilinear_ss(state.a, state.b, state.c, state.d, state.order, 2.0,
                std::numeric_limits<double>::quiet_NaN(), digital.a, digital.b, digital.c,
                digital.d);
    return digital;
}

} // namespace neurale::signal
