/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/**
 * @file
 * @brief Public SISO filter representations and conversion functions.
 */

#pragma once

#include <complex>
#include <cstddef>
#include <span>
#include <vector>

namespace neurale::signal
{

/**
 * @brief Real SISO transfer-function coefficients.
 */
struct TransferFunction
{
    /// Numerator coefficients in descending powers.
    std::vector<double> num;
    /// Denominator coefficients in descending powers.
    std::vector<double> den;
};

/**
 * @brief Real SISO state-space realization.
 *
 * Matrices use row-major flattened storage. `b` and `c` are vectors.
 */
struct StateSpace
{
    /// Number of state variables.
    std::size_t order{};
    /// Flattened square state-transition matrix.
    std::vector<double> a;
    /// Input vector.
    std::vector<double> b;
    /// Output vector.
    std::vector<double> c;
    /// Direct-feedthrough scalar.
    double d{};
};

/**
 * @brief SISO zeros, poles, and gain.
 */
struct Zpk
{
    /// System zeros.
    std::vector<std::complex<double>> z;
    /// System poles.
    std::vector<std::complex<double>> p;
    /// System gain.
    std::complex<double> k{1.0, 0.0};
};

/**
 * @brief Validate and deterministically order real and conjugate roots.
 *
 * @param values Finite roots describing a real-coefficient polynomial.
 * @return Roots ordered into conjugate pairs.
 * @throws std::invalid_argument If roots are non-finite or not conjugate
 *         paired.
 */
std::vector<std::complex<double>>
sort_conjugate_pairs(std::span<const std::complex<double>> values);

/**
 * @brief Construct real polynomial coefficients from roots.
 *
 * @param roots Finite roots forming real or conjugate-paired values.
 * @return Coefficients in descending powers.
 * @throws std::invalid_argument If roots cannot form a real polynomial.
 */
std::vector<double> polynomial_from_roots(std::span<const std::complex<double>> roots);

/**
 * @brief Compute polynomial roots from real coefficients.
 *
 * @param coefs Non-empty real coefficients in descending powers.
 * @return Complex polynomial roots.
 * @throws std::invalid_argument If coefficients are invalid.
 * @throws std::runtime_error If the numerical root solver fails.
 */
std::vector<std::complex<double>> polynomial_roots(std::span<const double> coefs);

/**
 * @brief Convert zeros, poles, and gain into caller-owned TF buffers.
 *
 * `num` and `den` must each contain `p.size() + 1` elements.
 */
void zpk2tf(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
            std::complex<double> k, std::span<double> num, std::span<double> den);

/**
 * @brief Convert a transfer function into caller-owned state-space buffers.
 *
 * Output spans must match the trimmed denominator order.
 */
void tf2ss(std::span<const double> num, std::span<const double> den, std::span<double> a,
           std::span<double> b, std::span<double> c, double& d);

/**
 * @brief Convert zeros, poles, and gain into caller-owned state-space buffers.
 *
 * Output spans must match `p.size()` states.
 */
void zpk2ss(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
            std::complex<double> k, std::span<double> a, std::span<double> b, std::span<double> c,
            double& d);

/**
 * @brief Convert a SISO state-space realization to a transfer function.
 *
 * @param state Valid state-space realization.
 * @return Equivalent real numerator and denominator coefficients.
 * @throws std::invalid_argument If the realization is malformed.
 */
TransferFunction ss2tf(const StateSpace& state);

/**
 * @brief Convert a transfer function to zeros, poles, and gain.
 *
 * @param num Real numerator coefficients.
 * @param den Real denominator coefficients.
 * @return Equivalent zeros, poles, and gain.
 * @throws std::invalid_argument If coefficients are invalid or improper.
 * @throws std::runtime_error If root computation fails.
 */
Zpk tf2zpk(std::span<const double> num, std::span<const double> den);

/**
 * @brief Convert a SISO state-space realization to zeros, poles, and gain.
 *
 * @param state Valid state-space realization.
 * @return Equivalent zeros, poles, and gain.
 * @throws std::invalid_argument If the realization is malformed.
 * @throws std::runtime_error If the generalized eigenvalue computation fails.
 */
Zpk ss2zpk(const StateSpace& state);

/**
 * @brief Convert zeros, poles, and gain into caller-owned SOS buffers.
 *
 * `sos` stores flattened `[b0, b1, b2, a0, a1, a2]` sections and must contain
 * `max(1, (p.size() + 1) / 2) * 6` elements.
 */
void zpk2sos(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
             double k, std::span<double> sos);

/**
 * @brief Frequency-scale an analog lowpass state-space prototype.
 *
 * @param state Analog lowpass prototype.
 * @param cutoff Positive target angular cutoff frequency.
 * @return Scaled lowpass realization.
 * @throws std::invalid_argument If the realization or cutoff is invalid.
 */
StateSpace lp2lp_ss(const StateSpace& state, double cutoff);

/**
 * @brief Transform an analog lowpass prototype to highpass.
 *
 * @param state Analog lowpass prototype.
 * @param cutoff Positive target angular cutoff frequency.
 * @return Highpass realization.
 * @throws std::invalid_argument If the realization or cutoff is invalid.
 * @throws std::runtime_error If a required linear solve fails.
 */
StateSpace lp2hp_ss(const StateSpace& state, double cutoff);

/**
 * @brief Transform an analog lowpass prototype to bandpass.
 *
 * @param state Analog lowpass prototype.
 * @param center_freq Positive target center angular frequency.
 * @param bandwidth Positive target angular bandwidth.
 * @return Bandpass realization with doubled state order.
 * @throws std::invalid_argument If parameters are invalid.
 */
StateSpace lp2bp_ss(const StateSpace& state, double center_freq, double bandwidth);

/**
 * @brief Transform an analog lowpass prototype to bandstop.
 *
 * @param state Analog lowpass prototype.
 * @param center_freq Positive target center angular frequency.
 * @param bandwidth Positive target angular bandwidth.
 * @return Bandstop realization with doubled state order.
 * @throws std::invalid_argument If parameters are invalid.
 * @throws std::runtime_error If a required linear solve fails.
 */
StateSpace lp2bs_ss(const StateSpace& state, double center_freq, double bandwidth);

/**
 * @brief Apply the bilinear transform directly into caller-owned ZPK buffers.
 *
 * `digital_z` and `digital_p` must each contain `p.size()` elements.
 */
void bilinear_zpk(std::span<const std::complex<double>> z, std::span<const std::complex<double>> p,
                  std::complex<double> k, double fs, double prewarp_freq,
                  std::span<std::complex<double>> digital_z,
                  std::span<std::complex<double>> digital_p, std::complex<double>& digital_k);

/**
 * @brief Apply the bilinear transform directly into caller-owned state buffers.
 */
void bilinear_ss(std::span<const double> a, std::span<const double> b, std::span<const double> c,
                 double d, std::size_t order, double fs, double prewarp_freq,
                 std::span<double> digital_a, std::span<double> digital_b,
                 std::span<double> digital_c, double& digital_d);

/**
 * @brief Return the output coefficient count for a proper transfer function.
 */
std::size_t bilinear_tf_size(std::span<const double> num, std::span<const double> den);

/**
 * @brief Apply the bilinear transform directly into caller-owned TF buffers.
 *
 * Output spans must each contain `bilinear_tf_size(num, den)` elements.
 */
void bilinear_tf(std::span<const double> num, std::span<const double> den, double fs,
                 double prewarp_freq, std::span<double> digital_num, std::span<double> digital_den);

} // namespace neurale::signal
