/*
 * SPDX-FileCopyrightText: 2026 pyneurale contributors
 * SPDX-License-Identifier: MIT
 */

#include <neurale/experiments/replay.h>

namespace neurale::experiments
{

const char* replay_verdict_name(ReplayVerdict verdict) noexcept
{
    switch (verdict)
    {
    case ReplayVerdict::match:
        return "match";
    case ReplayVerdict::mismatch:
        return "mismatch";
    case ReplayVerdict::incomplete:
        return "incomplete";
    case ReplayVerdict::rejected:
        return "rejected";
    }
    return "undeclared";
}

const char* replay_rejection_name(ReplayRejection rejection) noexcept
{
    switch (rejection)
    {
    case ReplayRejection::none:
        return "none";
    case ReplayRejection::paradigm_mismatch:
        return "paradigm-mismatch";
    case ReplayRejection::experiment_version_mismatch:
        return "experiment-version-mismatch";
    case ReplayRejection::configuration_fingerprint_mismatch:
        return "configuration-fingerprint-mismatch";
    case ReplayRejection::schedule_fingerprint_mismatch:
        return "schedule-fingerprint-mismatch";
    case ReplayRejection::seed_mismatch:
        return "seed-mismatch";
    case ReplayRejection::sampler_version_mismatch:
        return "sampler-version-mismatch";
    case ReplayRejection::sampler_version_unsupported:
        return "sampler-version-unsupported";
    case ReplayRejection::realized_schedule_mismatch:
        return "realized-schedule-mismatch";
    case ReplayRejection::metric_version_mismatch:
        return "metric-version-mismatch";
    case ReplayRejection::policy_version_mismatch:
        return "policy-version-mismatch";
    case ReplayRejection::provenance_incomplete:
        return "provenance-incomplete";
    case ReplayRejection::configuration_invalid:
        return "configuration-invalid";
    case ReplayRejection::evidence_invalid:
        return "evidence-invalid";
    case ReplayRejection::metric_version_unsupported:
        return "metric-version-unsupported";
    }
    return "undeclared";
}

const char* replay_item_name(ReplayItem item) noexcept
{
    switch (item)
    {
    case ReplayItem::none:
        return "none";
    case ReplayItem::transition:
        return "transition";
    case ReplayItem::event:
        return "event";
    case ReplayItem::trial:
        return "trial";
    case ReplayItem::target:
        return "target";
    case ReplayItem::cursor:
        return "cursor";
    case ReplayItem::assisted_velocity:
        return "assisted-velocity";
    case ReplayItem::guidance_sample:
        return "guidance-sample";
    case ReplayItem::selection:
        return "selection";
    case ReplayItem::metrics:
        return "metrics";
    case ReplayItem::phase:
        return "phase";
    case ReplayItem::presentation_request:
        return "presentation-request";
    case ReplayItem::schedule:
        return "schedule";
    case ReplayItem::stream_length:
        return "stream-length";
    }
    return "undeclared";
}

const char* replay_completeness_name(ReplayCompleteness completeness) noexcept
{
    switch (completeness)
    {
    case ReplayCompleteness::complete:
        return "complete";
    case ReplayCompleteness::stream_absent:
        return "stream-absent";
    case ReplayCompleteness::trace_loss_recorded:
        return "trace-loss-recorded";
    case ReplayCompleteness::run_end_missing:
        return "run-end-missing";
    }
    return "undeclared";
}

ReplayRejection check_provenance(const ReplayProvenance& recorded,
                                 const ReplayProvenance& candidate) noexcept
{
    // Usability before agreement. A recording that names no paradigm and no
    // sampler agrees with a candidate that names none either, and reporting
    // that as "nothing to reject" would let a replay proceed against provenance
    // that identifies nothing.
    if (recorded.paradigm == kUnsetParadigmId || candidate.paradigm == kUnsetParadigmId ||
        recorded.sampler_version == 0 || candidate.sampler_version == 0)
    {
        return ReplayRejection::provenance_incomplete;
    }
    if (recorded.paradigm != candidate.paradigm)
    {
        return ReplayRejection::paradigm_mismatch;
    }
    if (recorded.experiment_version != candidate.experiment_version)
    {
        return ReplayRejection::experiment_version_mismatch;
    }
    // The seed and the sampler before the digests that cover them. Both orders
    // reject the same recordings; this one names the parameter a caller can act
    // on rather than the digest that reflects it.
    if (recorded.seed != candidate.seed)
    {
        return ReplayRejection::seed_mismatch;
    }
    if (recorded.sampler_version != candidate.sampler_version)
    {
        return ReplayRejection::sampler_version_mismatch;
    }
    if (recorded.configuration_fingerprint != candidate.configuration_fingerprint)
    {
        return ReplayRejection::configuration_fingerprint_mismatch;
    }
    if (recorded.schedule_fingerprint != candidate.schedule_fingerprint)
    {
        return ReplayRejection::schedule_fingerprint_mismatch;
    }
    if (recorded.realized_schedule_fingerprint != candidate.realized_schedule_fingerprint)
    {
        return ReplayRejection::realized_schedule_mismatch;
    }
    if (recorded.metric_version != candidate.metric_version)
    {
        return ReplayRejection::metric_version_mismatch;
    }
    if (recorded.policy_version != candidate.policy_version)
    {
        return ReplayRejection::policy_version_mismatch;
    }
    return ReplayRejection::none;
}

void ReplayComparator::begin(ReplayItem item, std::uint64_t idx, ExperimentTimeNs time_ns,
                             const TrialIdentity& trial) noexcept
{
    item_ = item;
    idx_ = idx;
    time_ns_ = time_ns;
    trial_ = trial;
    ++items_;
}

void ReplayComparator::latch(std::string_view field) noexcept
{
    if (differed_)
    {
        return;
    }
    differed_ = true;
    first_.item = item_;
    first_.idx = idx_;
    first_.time_ns = time_ns_;
    first_.trial = trial_;
    first_.field = field;
}

void ReplayComparator::integer(std::string_view field, std::uint64_t recorded,
                               std::uint64_t regenerated) noexcept
{
    ++fields_;
    if (recorded == regenerated)
    {
        return;
    }
    const bool first = !differed_;
    latch(field);
    if (first)
    {
        first_.recorded = recorded;
        first_.regenerated = regenerated;
        first_.real_valued = false;
    }
}

void ReplayComparator::real(std::string_view field, double recorded, double regenerated) noexcept
{
    ++fields_;
    // Bit patterns, not values. Two doubles that differ in any bit came out of
    // different arithmetic, which is exactly what a determinism claim is about;
    // and comparing bits is also the only comparison under which two NaNs are
    // the same NaN.
    if (std::bit_cast<std::uint64_t>(recorded) == std::bit_cast<std::uint64_t>(regenerated))
    {
        return;
    }
    const bool first = !differed_;
    latch(field);
    if (first)
    {
        first_.recorded_real = recorded;
        first_.regenerated_real = regenerated;
        first_.recorded = std::bit_cast<std::uint64_t>(recorded);
        first_.regenerated = std::bit_cast<std::uint64_t>(regenerated);
        first_.real_valued = true;
    }
}

void ReplayComparator::flag(std::string_view field, bool recorded, bool regenerated) noexcept
{
    integer(field, recorded ? 1U : 0U, regenerated ? 1U : 0U);
}

void ReplayComparator::identity(std::string_view field, const TrialIdentity& recorded,
                                const TrialIdentity& regenerated) noexcept
{
    // One field name for five numbers, because a reader chasing a trial
    // identity mismatch wants the item and the trial, and the report already
    // carries both. Splitting it into five names would make the field column
    // say `trial.stimulus_id` for what is, every time, the same finding.
    integer(field, recorded.ordinal, regenerated.ordinal);
    integer(field, recorded.key, regenerated.key);
    integer(field, recorded.block, regenerated.block);
    integer(field, recorded.target_id, regenerated.target_id);
    integer(field, recorded.stimulus_id, regenerated.stimulus_id);
}

} // namespace neurale::experiments
