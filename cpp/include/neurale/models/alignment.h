/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace neurale::models
{

struct DtwResult
{
    double cost{};
    std::vector<std::size_t> path_x;
    std::vector<std::size_t> path_y;
};

DtwResult dtw(std::span<const double> x, std::size_t n_x, std::size_t x_dims,
              std::span<const double> y, std::size_t n_y, std::size_t y_dims,
              std::optional<std::size_t> radius, const std::string& metric);

} // namespace neurale::models
