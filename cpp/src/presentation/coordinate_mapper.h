// SPDX-FileCopyrightText: 2026 pyneurale contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "surface_types.h"

namespace neurale::experiment_presentation
{

class CoordinateMapper
{
  public:
    [[nodiscard]] SurfaceStatus configure(Rect2d logical_space, Size2i window_size,
                                          Size2i framebuffer_size, AspectPolicy policy) noexcept;

    [[nodiscard]] Point2d logical_to_framebuffer(Point2d logical) const noexcept;
    [[nodiscard]] bool framebuffer_to_logical(Point2d framebuffer, Point2d& logical) const noexcept;
    /// Map any finite framebuffer position, including letterbox/outside pixels.
    /// @p inside_viewport reports the half-open presentation viewport separately.
    [[nodiscard]] bool framebuffer_to_logical_unclipped(Point2d framebuffer, Point2d& logical,
                                                        bool& inside_viewport) const noexcept;
    [[nodiscard]] bool window_pointer_to_logical(Point2d window_pointer,
                                                 Point2d& logical) const noexcept;
    /// HiDPI-aware counterpart of framebuffer_to_logical_unclipped().
    [[nodiscard]] bool window_pointer_to_logical_unclipped(Point2d window_pointer, Point2d& logical,
                                                           bool& inside_viewport) const noexcept;
    [[nodiscard]] Point2d framebuffer_to_ndc(Point2d framebuffer) const noexcept;

    [[nodiscard]] Rect2d logical_space() const noexcept
    {
        return logical_space_;
    }
    [[nodiscard]] Rect2i viewport() const noexcept
    {
        return viewport_;
    }
    [[nodiscard]] Size2i window_size() const noexcept
    {
        return window_size_;
    }
    [[nodiscard]] Size2i framebuffer_size() const noexcept
    {
        return framebuffer_size_;
    }
    [[nodiscard]] bool configured() const noexcept
    {
        return configured_;
    }

  private:
    Rect2d logical_space_{};
    Rect2i viewport_{};
    Size2i window_size_{};
    Size2i framebuffer_size_{};
    bool configured_{};
};

} // namespace neurale::experiment_presentation
