/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The one translation from a paradigm's ContractStatus to a runtime
/// StreamStatus.
///
/// A paradigm answers in its own vocabulary -- the time went backwards, the
/// value is not a number, the machine is not running -- and the streaming
/// runtime answers in its own. Every controller and the session have to cross
/// that boundary, and the crossing is the same one: it is a property of the two
/// enumerations, not of any paradigm. Four copies of the switch were four
/// chances for one paradigm to classify a condition as a bad frame while
/// another called the identical condition a failed consumer, which the runtime
/// then treats differently -- one is recoverable, the other stops the session.
///
/// Total and allocation-free, so it is callable from a realtime thread.

#include <neurale/experiments/contract.h>
#include <neurale/streaming/fault.h>

namespace neurale::execution
{

using namespace neurale::experiments;

/// Classify @p status for the streaming runtime.
///
/// Anything the runtime has no more specific answer for is a consumer failure:
/// the paradigm refused work it was asked to do, and the runtime cannot tell
/// from here whether continuing would be sound.
[[nodiscard]] constexpr streaming::StreamStatus map_contract_status(ContractStatus status) noexcept
{
    switch (status)
    {
    case ContractStatus::ok:
        return streaming::StreamStatus::ok;
    case ContractStatus::time_regressed:
    case ContractStatus::value_not_finite:
        return streaming::StreamStatus::invalid_frame;
    case ContractStatus::not_running:
    case ContractStatus::already_running:
        return streaming::StreamStatus::invalid_state;
    default:
        return streaming::StreamStatus::consumer_failure;
    }
}

} // namespace neurale::execution
