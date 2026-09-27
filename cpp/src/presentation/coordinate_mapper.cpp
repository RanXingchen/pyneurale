// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#include "coordinate_mapper.h"

#include <algorithm>
#include <cmath>

namespace neurale::experiment_presentation
{

SurfaceStatus CoordinateMapper::configure(Rect2d logical_space, Size2i window_size,
                                          Size2i framebuffer_size, AspectPolicy policy) noexcept
{
    if (!std::isfinite(logical_space.left) || !std::isfinite(logical_space.bottom) ||
        !std::isfinite(logical_space.width) || !std::isfinite(logical_space.height) ||
        logical_space.width <= 0.0 || logical_space.height <= 0.0 || window_size.width <= 0 ||
        window_size.height <= 0 || framebuffer_size.width <= 0 || framebuffer_size.height <= 0)
    {
        return SurfaceStatus::invalid_configuration;
    }

    Rect2i viewport{0, 0, framebuffer_size.width, framebuffer_size.height};
    if (policy == AspectPolicy::fit_letterbox)
    {
        const auto logical_aspect = logical_space.width / logical_space.height;
        const auto framebuffer_aspect = static_cast<double>(framebuffer_size.width) /
                                        static_cast<double>(framebuffer_size.height);
        if (framebuffer_aspect > logical_aspect)
        {
            viewport.width = std::max(
                1, static_cast<int>(std::llround(framebuffer_size.height * logical_aspect)));
            viewport.left = (framebuffer_size.width - viewport.width) / 2;
        }
        else
        {
            viewport.height = std::max(
                1, static_cast<int>(std::llround(framebuffer_size.width / logical_aspect)));
            viewport.top = (framebuffer_size.height - viewport.height) / 2;
        }
    }
    else if (policy != AspectPolicy::stretch)
    {
        return SurfaceStatus::invalid_configuration;
    }

    logical_space_ = logical_space;
    window_size_ = window_size;
    framebuffer_size_ = framebuffer_size;
    viewport_ = viewport;
    configured_ = true;
    return SurfaceStatus::ok;
}

Point2d CoordinateMapper::logical_to_framebuffer(Point2d logical) const noexcept
{
    const auto x_fraction = (logical.x - logical_space_.left) / logical_space_.width;
    const auto y_fraction = (logical.y - logical_space_.bottom) / logical_space_.height;
    return Point2d{
        static_cast<double>(viewport_.left) + x_fraction * viewport_.width,
        static_cast<double>(viewport_.top) + (1.0 - y_fraction) * viewport_.height,
    };
}

bool CoordinateMapper::framebuffer_to_logical(Point2d framebuffer, Point2d& logical) const noexcept
{
    Point2d candidate{};
    bool inside{};
    if (!framebuffer_to_logical_unclipped(framebuffer, candidate, inside) || !inside)
    {
        return false;
    }
    logical = candidate;
    return true;
}

bool CoordinateMapper::framebuffer_to_logical_unclipped(Point2d framebuffer, Point2d& logical,
                                                        bool& inside_viewport) const noexcept
{
    if (!configured_ || !std::isfinite(framebuffer.x) || !std::isfinite(framebuffer.y))
    {
        inside_viewport = false;
        return false;
    }
    // Pointer-domain membership mirrors the half-open logical contract
    // [min_x, max_x) x [min_y, max_y). Framebuffer Y points down, so the top
    // edge maps to logical max_y (excluded) and the bottom edge to logical
    // min_y (included); X is unaffected by the flip. This is the continuous
    // pointer domain only -- the OpenGL scissor/raster viewport in
    // begin_frame() stays the standard integer-pixel [left, right) x [top, bottom).
    inside_viewport =
        framebuffer.x >= viewport_.left && framebuffer.x < viewport_.left + viewport_.width &&
        framebuffer.y > viewport_.top && framebuffer.y <= viewport_.top + viewport_.height;
    const auto x_fraction = (framebuffer.x - viewport_.left) / viewport_.width;
    const auto y_fraction = 1.0 - (framebuffer.y - viewport_.top) / viewport_.height;
    logical = Point2d{
        logical_space_.left + x_fraction * logical_space_.width,
        logical_space_.bottom + y_fraction * logical_space_.height,
    };
    return true;
}

bool CoordinateMapper::window_pointer_to_logical(Point2d window_pointer,
                                                 Point2d& logical) const noexcept
{
    Point2d candidate{};
    bool inside{};
    if (!window_pointer_to_logical_unclipped(window_pointer, candidate, inside) || !inside)
    {
        return false;
    }
    logical = candidate;
    return true;
}

bool CoordinateMapper::window_pointer_to_logical_unclipped(Point2d window_pointer, Point2d& logical,
                                                           bool& inside_viewport) const noexcept
{
    if (!configured_ || !std::isfinite(window_pointer.x) || !std::isfinite(window_pointer.y))
    {
        inside_viewport = false;
        return false;
    }
    const auto framebuffer = Point2d{
        window_pointer.x * framebuffer_size_.width / window_size_.width,
        window_pointer.y * framebuffer_size_.height / window_size_.height,
    };
    return framebuffer_to_logical_unclipped(framebuffer, logical, inside_viewport);
}

Point2d CoordinateMapper::framebuffer_to_ndc(Point2d framebuffer) const noexcept
{
    return Point2d{
        2.0 * framebuffer.x / framebuffer_size_.width - 1.0,
        1.0 - 2.0 * framebuffer.y / framebuffer_size_.height,
    };
}

} // namespace neurale::experiment_presentation
