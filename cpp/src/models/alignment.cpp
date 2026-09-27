/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/models/alignment.h>

#include "numeric_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && defined(_M_X64)
#include <intrin.h>
#endif

namespace neurale::models
{
namespace
{

enum class DtwMetric
{
    Euclidean,
    SquaredEuclidean,
};

enum class Direction : std::uint8_t
{
    None = 0,
    Start = 1,
    Diagonal = 2,
    Up = 3,
    Left = 4,
};

struct Bounds
{
    std::size_t first{};
    std::size_t last{};
};

struct DtwProblem
{
    std::span<const double> x;
    std::span<const double> y;
    std::size_t n_x{};
    std::size_t n_y{};
    std::size_t n_dims{};
    std::optional<std::size_t> radius;
    DtwMetric metric{};
};

std::size_t checked_add(std::size_t left, std::size_t right, const char* message)
{
    if (right > std::numeric_limits<std::size_t>::max() - left)
    {
        throw std::length_error(message);
    }
    return left + right;
}

std::size_t checked_mul(std::size_t left, std::size_t right, const char* message)
{
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left)
    {
        throw std::length_error(message);
    }
    return left * right;
}

struct DivResult
{
    std::size_t quotient{};
    std::size_t remainder{};
};

DivResult mul_div(std::size_t left, std::size_t right, std::size_t divisor, const char* message)
{
#if defined(_MSC_VER) && defined(_M_X64)
    unsigned __int64 remainder = 0;
    const unsigned __int64 high = __umulh(left, right);
    if (high >= divisor)
    {
        throw std::length_error(message);
    }
    const auto quotient = _udiv128(high, left * right, divisor, &remainder);
    return DivResult{.quotient = quotient, .remainder = remainder};
#elif defined(__SIZEOF_INT128__)
    static_cast<void>(message);

    using Wide = unsigned __int128;
    const Wide product = static_cast<Wide>(left) * static_cast<Wide>(right);
    const Wide wide_divisor = static_cast<Wide>(divisor);
    return DivResult{
        .quotient = static_cast<std::size_t>(product / wide_divisor),
        .remainder = static_cast<std::size_t>(product % wide_divisor),
    };
#else
#error "DTW integer window bounds require MSVC x64 or compiler __int128 support"
#endif
}

std::size_t mul_div_floor(std::size_t left, std::size_t right, std::size_t divisor,
                          const char* message)
{
    return mul_div(left, right, divisor, message).quotient;
}

std::size_t mul_div_ceil(std::size_t left, std::size_t right, std::size_t divisor,
                         const char* message)
{
    const DivResult result = mul_div(left, right, divisor, message);
    return result.remainder == 0 ? result.quotient : checked_add(result.quotient, 1, message);
}

void require_sequence(std::span<const double> values, std::size_t n_steps, std::size_t n_dims,
                      const char* name)
{
    const std::size_t size = checked_mul(n_steps, n_dims, "DTW input shape is too large");
    if (n_steps == 0 || n_dims == 0 || values.size() != size)
    {
        throw std::invalid_argument(std::string(name) + " shape is invalid");
    }
    for (double value : values)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument(std::string(name) + " must contain finite values");
        }
    }
}

DtwMetric parse_metric(const std::string& metric)
{
    if (metric == "euclidean")
    {
        return DtwMetric::Euclidean;
    }
    if (metric == "sqeuclidean")
    {
        return DtwMetric::SquaredEuclidean;
    }
    throw std::invalid_argument("metric is invalid");
}

Bounds full_bounds(std::size_t n_cols) noexcept
{
    return Bounds{.first = 0, .last = n_cols};
}

Bounds row_bounds(std::size_t n_rows, std::size_t n_cols, std::size_t row,
                  std::optional<std::size_t> radius)
{
    if (!radius.has_value() || n_cols == 1)
    {
        return full_bounds(n_cols);
    }

    const std::size_t den = n_rows - 1;
    const std::size_t scale = n_cols - 1;
    if (*radius >= den)
    {
        return full_bounds(n_cols);
    }

    const std::size_t lower_row = row > *radius ? row - *radius : 0;
    const std::size_t upper_row = *radius >= den - row ? den : row + *radius;
    const std::size_t first =
        mul_div_ceil(lower_row, scale, den, "DTW window bounds are too large");
    const std::size_t last =
        checked_add(mul_div_floor(upper_row, scale, den, "DTW window bounds are too large"), 1,
                    "DTW window bounds are too large");
    return Bounds{.first = first, .last = std::min(last, n_cols)};
}

class DirectionTable
{
  public:
    DirectionTable(std::size_t n_rows, std::size_t n_cols, std::optional<std::size_t> radius)
        : bounds_(n_rows), offsets_(checked_add(n_rows, 1, "DTW path metadata is too large"))
    {
        std::size_t offset = 0;
        for (std::size_t row = 0; row < n_rows; ++row)
        {
            bounds_[row] = row_bounds(n_rows, n_cols, row, radius);
            offsets_[row] = offset;
            offset = checked_add(offset, bounds_[row].last - bounds_[row].first,
                                 "DTW path metadata is too large");
        }
        offsets_[n_rows] = offset;
        directions_.assign(offset, static_cast<std::uint8_t>(Direction::None));
    }

    bool contains(std::size_t row, std::size_t col) const noexcept
    {
        const Bounds bounds = bounds_[row];
        return col >= bounds.first && col < bounds.last;
    }

    Bounds bounds(std::size_t row) const noexcept
    {
        return bounds_[row];
    }

    Direction get(std::size_t row, std::size_t col) const
    {
        return static_cast<Direction>(directions_[index(row, col)]);
    }

    void set(std::size_t row, std::size_t col, Direction direction)
    {
        directions_[index(row, col)] = static_cast<std::uint8_t>(direction);
    }

  private:
    std::size_t index(std::size_t row, std::size_t col) const
    {
        const Bounds bounds = bounds_[row];
        return offsets_[row] + (col - bounds.first);
    }

    std::vector<Bounds> bounds_;
    std::vector<std::size_t> offsets_;
    std::vector<std::uint8_t> directions_;
};

double local_cost(const double* x_row, const double* y_row, std::size_t n_dims, DtwMetric metric)
{
    double squared = 0.0;
    for (std::size_t dim = 0; dim < n_dims; ++dim)
    {
        const double delta = x_row[dim] - y_row[dim];
        squared += delta * delta;
    }
    if (metric == DtwMetric::SquaredEuclidean)
    {
        return squared;
    }
    if (std::isnormal(squared))
    {
        return std::sqrt(squared);
    }
    if (squared == 0.0 && std::equal(x_row, x_row + n_dims, y_row))
    {
        return 0.0;
    }
    return detail::scaled_euclidean_distance(x_row, y_row, n_dims);
}

bool reachable(const DirectionTable& directions, std::size_t row, std::size_t col)
{
    return directions.contains(row, col) && directions.get(row, col) != Direction::None;
}

void choose_predecessor(bool candidate_reachable, double candidate_cost,
                        Direction candidate_direction, bool& best_reachable, double& best_cost,
                        Direction& best_direction)
{
    if (!candidate_reachable)
    {
        return;
    }
    if (!best_reachable || candidate_cost < best_cost)
    {
        best_reachable = true;
        best_cost = candidate_cost;
        best_direction = candidate_direction;
    }
}

DtwResult traceback(double cost, const DirectionTable& directions, std::size_t n_x, std::size_t n_y)
{
    auto reserve_size = checked_add(n_x, n_y, "DTW path is too large");
    auto path_x = std::vector<std::size_t>();
    auto path_y = std::vector<std::size_t>();
    path_x.reserve(reserve_size);
    path_y.reserve(reserve_size);

    std::size_t row = n_x - 1;
    std::size_t col = n_y - 1;
    while (true)
    {
        path_x.push_back(row);
        path_y.push_back(col);
        const Direction direction = directions.get(row, col);
        if (direction == Direction::Start)
        {
            break;
        }
        if (direction == Direction::Diagonal)
        {
            --row;
            --col;
        }
        else if (direction == Direction::Up)
        {
            --row;
        }
        else if (direction == Direction::Left)
        {
            --col;
        }
        else
        {
            throw std::runtime_error("DTW traceback encountered an invalid cell");
        }
    }

    std::reverse(path_x.begin(), path_x.end());
    std::reverse(path_y.begin(), path_y.end());
    return DtwResult{cost, std::move(path_x), std::move(path_y)};
}

DtwResult run_dtw(const DtwProblem& problem)
{
    constexpr double infinity = std::numeric_limits<double>::infinity();
    auto directions = DirectionTable(problem.n_x, problem.n_y, problem.radius);
    auto previous = std::vector<double>(problem.n_y, infinity);
    auto current = std::vector<double>(problem.n_y, infinity);

    for (std::size_t row = 0; row < problem.n_x; ++row)
    {
        const Bounds bounds = directions.bounds(row);
        double left_cost = infinity;
        bool left_reachable = false;
        const double* x_row = problem.x.data() + row * problem.n_dims;

        for (std::size_t col = bounds.first; col < bounds.last; ++col)
        {
            const double* y_row = problem.y.data() + col * problem.n_dims;
            const double distance = local_cost(x_row, y_row, problem.n_dims, problem.metric);

            if (row == 0 && col == 0)
            {
                current[col] = distance;
                directions.set(row, col, Direction::Start);
                left_cost = current[col];
                left_reachable = true;
                continue;
            }

            bool best_reachable = false;
            double best_cost = infinity;
            Direction best_direction = Direction::None;

            if (row > 0 && col > 0)
            {
                choose_predecessor(reachable(directions, row - 1, col - 1), previous[col - 1],
                                   Direction::Diagonal, best_reachable, best_cost, best_direction);
            }
            if (row > 0)
            {
                choose_predecessor(reachable(directions, row - 1, col), previous[col],
                                   Direction::Up, best_reachable, best_cost, best_direction);
            }
            if (col > bounds.first)
            {
                choose_predecessor(left_reachable, left_cost, Direction::Left, best_reachable,
                                   best_cost, best_direction);
            }

            if (best_reachable)
            {
                current[col] = best_cost + distance;
                directions.set(row, col, best_direction);
                left_cost = current[col];
                left_reachable = true;
            }
            else
            {
                current[col] = infinity;
                directions.set(row, col, Direction::None);
                left_cost = infinity;
                left_reachable = false;
            }
        }

        std::swap(previous, current);
    }

    if (!directions.contains(problem.n_x - 1, problem.n_y - 1) ||
        directions.get(problem.n_x - 1, problem.n_y - 1) == Direction::None)
    {
        throw std::invalid_argument("radius does not allow a complete DTW path");
    }

    const double final_cost = previous[problem.n_y - 1];
    if (!std::isfinite(final_cost))
    {
        throw std::overflow_error("DTW accumulated cost overflowed");
    }
    return traceback(final_cost, directions, problem.n_x, problem.n_y);
}

DtwResult swap_result_paths(DtwResult result)
{
    std::swap(result.path_x, result.path_y);
    return result;
}

} // namespace

DtwResult dtw(std::span<const double> x, std::size_t n_x, std::size_t x_dims,
              std::span<const double> y, std::size_t n_y, std::size_t y_dims,
              std::optional<std::size_t> radius, const std::string& metric)
{
    require_sequence(x, n_x, x_dims, "x");
    require_sequence(y, n_y, y_dims, "y");
    if (x_dims != y_dims)
    {
        throw std::invalid_argument("x and y must have the same number of dimensions");
    }

    const DtwMetric metric_value = parse_metric(metric);
    if (n_x < n_y)
    {
        return swap_result_paths(run_dtw(DtwProblem{
            .x = y,
            .y = x,
            .n_x = n_y,
            .n_y = n_x,
            .n_dims = x_dims,
            .radius = radius,
            .metric = metric_value,
        }));
    }
    return run_dtw(DtwProblem{
        .x = x,
        .y = y,
        .n_x = n_x,
        .n_y = n_y,
        .n_dims = x_dims,
        .radius = radius,
        .metric = metric_value,
    });
}

} // namespace neurale::models
