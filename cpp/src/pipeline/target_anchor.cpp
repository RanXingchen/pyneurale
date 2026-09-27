/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

// This translation unit keeps a target-level anchor without publishing the
// private adapter implementation as a public CMake or Python API.
namespace neurale::pipeline::detail
{

void integration_target_anchor() noexcept {}

} // namespace neurale::pipeline::detail
