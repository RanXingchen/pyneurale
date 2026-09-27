/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#pragma once

/// \file
/// The verdict rules every paradigm replay settles identically.
///
/// What each paradigm replays is its own -- Center-Out compares cursor and
/// guidance streams, WebGrid compares selections and metrics, Speech compares
/// phases and reports. How the outcome of that comparison becomes a verdict is
/// not: a run that could not regenerate is rejected rather than mismatched, a
/// comparator that saw a difference reports the first one, and a recording that
/// holds more of a stream than the run produced is a stream-length mismatch.
/// Those are properties of ::ReplayReport, and three copies of them were three
/// chances for one paradigm to call rejected what another called mismatch --
/// a distinction a caller acts on, since only one of the two says the
/// recording and the implementation disagree.

#include <cstddef>

#include <neurale/experiments/contract.h>
#include <neurale/experiments/replay.h>

namespace neurale::experiments
{

/// Fold the comparator's totals into @p report and settle the verdict when it
/// is already decided.
///
/// A non-ok @p regeneration_status means the replay could not produce an output
/// at all: the configuration and the recorded inputs together describe a step
/// the paradigm refuses. That is not a difference between two runs, so it is
/// reported as a rejection and never as a mismatch.
///
/// @return true when @p report is final and the caller should return it.
[[nodiscard]] inline bool note_regeneration_outcome(ReplayReport& report,
                                                    const ReplayComparator& comparator,
                                                    ContractStatus regeneration_status) noexcept
{
    report.items_compared = comparator.items_compared();
    report.fields_compared = comparator.fields_compared();

    if (regeneration_status != ContractStatus::ok)
    {
        report.rejection = regeneration_status == ContractStatus::version_unsupported
                               ? ReplayRejection::sampler_version_unsupported
                               : ReplayRejection::evidence_invalid;
        report.verdict = ReplayVerdict::rejected;
        return true;
    }
    if (comparator.differed())
    {
        report.verdict = ReplayVerdict::mismatch;
        report.first_mismatch = comparator.first_mismatch();
        return true;
    }
    return false;
}

/// Report a recorded stream that holds more entries than the run regenerated.
///
/// Every field agreed as far as the run went, so there is no item to hang the
/// difference on -- which is what ::ReplayItem::stream_length exists for.
///
/// @param present  Whether the recording carries this stream at all; an absent
///                 stream is not a longer one.
/// @return true when @p report was set to a mismatch.
[[nodiscard]] inline bool note_recorded_extra(ReplayReport& report, bool present,
                                              std::size_t recorded,
                                              std::size_t regenerated) noexcept
{
    if (!present || recorded <= regenerated)
    {
        return false;
    }
    report.verdict = ReplayVerdict::mismatch;
    report.first_mismatch.item = ReplayItem::stream_length;
    report.first_mismatch.field = "recorded_extra";
    report.first_mismatch.recorded = recorded;
    report.first_mismatch.regenerated = regenerated;
    return true;
}

} // namespace neurale::experiments
