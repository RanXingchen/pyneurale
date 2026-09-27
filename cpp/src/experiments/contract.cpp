/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/contract.h>

namespace neurale::experiments
{

const char* contract_status_message(ContractStatus status) noexcept
{
    // Every arm returns a string literal, so reporting a rejection allocates
    // nothing and stays callable from a bounded context.
    switch (status)
    {
    case ContractStatus::ok:
        return "ok";
    case ContractStatus::time_regressed:
        return "supplied experiment time moved backwards";
    case ContractStatus::interval_inverted:
        return "interval ends before it starts";
    case ContractStatus::duration_overflow:
        return "start plus duration exceeds the representable nanosecond range";
    case ContractStatus::dimension_invalid:
        return "command dimension is zero, too large, or contradicted by the values";
    case ContractStatus::value_not_finite:
        return "command value is not finite";
    case ContractStatus::expiry_before_generation:
        return "request expires no later than it was generated";
    case ContractStatus::range_empty:
        return "sampling range contains no admissible value";
    case ContractStatus::sampling_exhausted:
        return "bounded draw rejected every attempt";
    case ContractStatus::ordinal_exhausted:
        return "ordinal sequence has no unissued ordinal left";
    case ContractStatus::identity_missing:
        return "a required identifier is unset";
    case ContractStatus::enum_undeclared:
        return "an enumerated field names no declared value";
    case ContractStatus::presentation_invalid:
        return "presentation fields describe no possible presentation";
    case ContractStatus::outcome_invalid:
        return "outcome contradicts the record that carries it";
    case ContractStatus::assistance_out_of_range:
        return "assistance parameter lies outside its declared interval";
    case ContractStatus::domain_mask_invalid:
        return "domain mask selects no axis, or an axis the command does not have";
    case ContractStatus::version_unsupported:
        return "assistance record names an algorithm version this build does not replay";
    case ContractStatus::numerical_failure:
        return "bounded numerical procedure did not converge within its budget";
    case ContractStatus::parameter_out_of_range:
        return "configuration parameter is finite but outside its declared interval";
    case ContractStatus::target_set_invalid:
        return "target set is empty, too large, or names one identifier twice";
    case ContractStatus::not_running:
        return "the procedure has not been started, or has already finished";
    case ContractStatus::already_running:
        return "start() called on a machine that already holds a session";
    }
    return "unknown contract status";
}

} // namespace neurale::experiments
