/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <stdexcept>

namespace neurale::models::cuda
{

class DeviceUnavailableError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

} // namespace neurale::models::cuda
