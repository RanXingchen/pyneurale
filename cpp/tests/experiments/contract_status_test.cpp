/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

/// \file
/// The ContractStatus -> StreamStatus table, pinned enumerator by enumerator.
///
/// This mapping used to be four copies of one switch, one per controller and
/// one in the session, and the copies had drifted: the speech copy had no
/// `value_not_finite` case and fell through to `consumer_failure`, while the
/// other three answered `invalid_frame`. The difference is not cosmetic.
/// `invalid_frame` is a recoverable verdict -- the runtime rejects the frame
/// and keeps running -- and `consumer_failure` stops the session. A paradigm
/// that hands the runtime a NaN command should lose the frame, not the session,
/// and every paradigm should lose the same thing.
///
/// The unified `map_contract_status` is that agreement, so it is what is tested
/// here rather than any one controller. `SpeechMachine`'s own translation unit
/// cannot reach `value_not_finite` today -- the only producer is
/// `validate(const CommandRequest&)`, and `command.h` is not in that include
/// closure -- so no reachable speech scenario would notice the old copy
/// silently coming back. A direct test of the table does.

#include "contract_status.h"

#include "check_returns.h"

#include <neurale/experiments/contract.h>
#include <neurale/streaming/fault.h>

#include <exception>
#include <iostream>

namespace
{

using neurale::execution::map_contract_status;
using neurale::experiments::ContractStatus;
using neurale::streaming::StreamStatus;

/// The one enumerator the four copies disagreed about.
int check_value_not_finite_is_recoverable()
{
    CHECK(map_contract_status(ContractStatus::value_not_finite) == StreamStatus::invalid_frame);
    CHECK(map_contract_status(ContractStatus::value_not_finite) != StreamStatus::consumer_failure);
    return 0;
}

/// A bad value and a bad time are the same kind of answer: reject the frame.
int check_bad_frame_statuses()
{
    CHECK(map_contract_status(ContractStatus::time_regressed) == StreamStatus::invalid_frame);
    CHECK(map_contract_status(ContractStatus::value_not_finite) == StreamStatus::invalid_frame);
    return 0;
}

/// Lifecycle answers are about the machine, not the frame.
int check_lifecycle_statuses()
{
    CHECK(map_contract_status(ContractStatus::not_running) == StreamStatus::invalid_state);
    CHECK(map_contract_status(ContractStatus::already_running) == StreamStatus::invalid_state);
    return 0;
}

int check_ok_passes_through()
{
    CHECK(map_contract_status(ContractStatus::ok) == StreamStatus::ok);
    return 0;
}

/// Every remaining enumerator is a consumer failure. Listed one by one rather
/// than looped over a range, so adding an enumerator to ContractStatus without
/// deciding where it belongs leaves this file visibly silent about it instead
/// of quietly asserting the default.
int check_remaining_statuses_are_consumer_failures()
{
    CHECK(map_contract_status(ContractStatus::interval_inverted) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::duration_overflow) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::dimension_invalid) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::expiry_before_generation) ==
          StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::range_empty) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::sampling_exhausted) ==
          StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::ordinal_exhausted) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::identity_missing) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::enum_undeclared) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::presentation_invalid) ==
          StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::outcome_invalid) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::assistance_out_of_range) ==
          StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::domain_mask_invalid) ==
          StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::version_unsupported) ==
          StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::numerical_failure) == StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::parameter_out_of_range) ==
          StreamStatus::consumer_failure);
    CHECK(map_contract_status(ContractStatus::target_set_invalid) ==
          StreamStatus::consumer_failure);
    return 0;
}

int run()
{
    if (const int failure = check_value_not_finite_is_recoverable(); failure != 0)
        return failure;
    if (const int failure = check_bad_frame_statuses(); failure != 0)
        return failure;
    if (const int failure = check_lifecycle_statuses(); failure != 0)
        return failure;
    if (const int failure = check_ok_passes_through(); failure != 0)
        return failure;
    if (const int failure = check_remaining_statuses_are_consumer_failures(); failure != 0)
        return failure;
    return 0;
}

} // namespace

int main()
{
    try
    {
        return run();
    }
    catch (const std::exception& exception)
    {
        std::cerr << exception.what() << '\n';
        return 1;
    }
}
