/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace neurale::execution
{

class CenterOutController;
class CenterOutTraceWriter;
} // namespace neurale::execution

namespace neurale::bindings::experiments
{

inline constexpr const char* kCenterOutSessionContextCapsule =
    "neurale.experiments.center_out_session_context.v1";

/// Non-owning link between the Center-Out session and presentation bindings.
/// The Python bridge retains the owning session while it uses this context.
struct CenterOutSessionContext
{
    neurale::execution::CenterOutTraceWriter* writer{};
    neurale::execution::CenterOutController* controller{};
};

} // namespace neurale::bindings::experiments
