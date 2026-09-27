/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include "recorder_status.h"

namespace neurale::recording
{

std::string_view to_string(RecorderLifecycleState state) noexcept
{
    switch (state)
    {
    case RecorderLifecycleState::created:
        return "created";
    case RecorderLifecycleState::prepared:
        return "prepared";
    case RecorderLifecycleState::ready:
        return "ready";
    case RecorderLifecycleState::recording:
        return "recording";
    case RecorderLifecycleState::draining:
        return "draining";
    case RecorderLifecycleState::stopped:
        return "stopped";
    case RecorderLifecycleState::finalizing:
        return "finalizing";
    case RecorderLifecycleState::finalization_failed:
        return "finalization_failed";
    case RecorderLifecycleState::closed:
        return "closed";
    case RecorderLifecycleState::failed:
        return "failed";
    }
    return "unknown";
}

std::string_view to_string(CompletenessVerdict verdict) noexcept
{
    switch (verdict)
    {
    case CompletenessVerdict::verified_complete:
        return "verified_complete";
    case CompletenessVerdict::verified_incomplete:
        return "verified_incomplete";
    case CompletenessVerdict::unverified_legacy:
        return "unverified_legacy";
    }
    return "unknown";
}

std::string_view to_string(FinalizationStatus status) noexcept
{
    switch (status)
    {
    case FinalizationStatus::not_started:
        return "not_started";
    case FinalizationStatus::running:
        return "running";
    case FinalizationStatus::failed_retryable:
        return "failed_retryable";
    case FinalizationStatus::succeeded:
        return "succeeded";
    case FinalizationStatus::abandoned:
        return "abandoned";
    }
    return "unknown";
}

std::string_view to_string(EffectiveSessionOutcome outcome) noexcept
{
    switch (outcome)
    {
    case EffectiveSessionOutcome::normal:
        return "normal";
    case EffectiveSessionOutcome::aborted:
        return "aborted";
    case EffectiveSessionOutcome::faulted:
        return "faulted";
    }
    return "unknown";
}

std::string_view to_string(RecoverabilityAnswer answer) noexcept
{
    switch (answer)
    {
    case RecoverabilityAnswer::not_applicable:
        return "not_applicable";
    case RecoverabilityAnswer::recoverable:
        return "recoverable";
    case RecoverabilityAnswer::unrecoverable:
        return "unrecoverable";
    }
    return "unknown";
}

std::string_view to_string(RecorderStatusCode code) noexcept
{
    switch (code)
    {
    case RecorderStatusCode::ok:
        return "ok";
    case RecorderStatusCode::wrong_state:
        return "wrong_state";
    case RecorderStatusCode::invalid_plan:
        return "invalid_plan";
    case RecorderStatusCode::prepare_failed:
        return "prepare_failed";
    case RecorderStatusCode::not_ready:
        return "not_ready";
    case RecorderStatusCode::writer_failed:
        return "writer_failed";
    case RecorderStatusCode::drain_timed_out:
        return "drain_timed_out";
    case RecorderStatusCode::not_implemented:
        return "not_implemented";
    }
    return "unknown";
}

} // namespace neurale::recording
