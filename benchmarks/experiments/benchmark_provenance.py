#!/usr/bin/env python3

"""Offline provenance-query characterization.

The three indexes are prepared before timing.  The measured operation is one
offline ``resolve`` call over a complete representative evidence chain; this
script makes no realtime or allocation claim.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from functools import partial
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import benchmark_metadata, measure_call, write_records

from neurale.experiments import traceability as tr


def _id(kind: tr.EvidenceKind, ordinal: int) -> tr.EvidenceId:
    return tr.EvidenceId(kind, ordinal)


def _recording(*streams: str) -> tr.RecordingEvidence:
    return tr.RecordingEvidence("m8-benchmark", True, True, True, streams)


def _trial(*, target: int = 0, stimulus: int = 0) -> tr.TrialIdentityEvidence:
    return tr.TrialIdentityEvidence(0, 0, 0, target, stimulus)


def _center_out_index() -> tr.CenterOutProvenanceIndex:
    trial = _trial(target=1)
    source_a = tr.SourceRangeEvidence(
        _id(tr.EvidenceKind.SOURCE_RANGE, 0), "neural", 0, 100, 164, 10, 11
    )
    source_b = tr.SourceRangeEvidence(
        _id(tr.EvidenceKind.SOURCE_RANGE, 1), "neural", 0, 164, 228, 11, 12
    )
    feature = tr.FeatureObservationEvidence(
        _id(tr.EvidenceKind.FEATURE_OBSERVATION, 0),
        0,
        1_000,
        (source_a.id, source_b.id),
    )
    decoder = tr.DecoderOutputEvidence(
        _id(tr.EvidenceKind.DECODER_OUTPUT, 0), feature.id, 0, 1_000, 12, 200
    )
    guidance = tr.GuidanceDecisionEvidence(_id(tr.EvidenceKind.GUIDANCE_DECISION, 0), decoder.id, 1)
    assistance = tr.AssistanceDecisionEvidence(
        _id(tr.EvidenceKind.ASSISTANCE_DECISION, 0), decoder.id, guidance.id, trial
    )
    state = tr.ExperimentStateEvidence(_id(tr.EvidenceKind.EXPERIMENT_STATE, 0), trial, 1, 2)
    command = tr.CenterOutCommandEvidence(
        _id(tr.EvidenceKind.COMMAND_REQUEST, 0),
        decoder.id,
        state.id,
        1_000,
        trial,
        guidance.id,
        assistance.id,
        True,
        True,
    )
    application = tr.CommandApplicationEvidence(
        _id(tr.EvidenceKind.COMMAND_APPLICATION, 0),
        command.id,
        1_000,
        1_001,
        1,
        0,
        trial,
    )
    return tr.CenterOutProvenanceIndex(
        _recording("neural"),
        sources=(source_a, source_b),
        features=(feature,),
        decoder_outputs=(decoder,),
        guidance=(guidance,),
        assistance=(assistance,),
        states=(state,),
        commands=(command,),
        applications=(application,),
    )


def _webgrid_index() -> tr.WebGridProvenanceIndex:
    trial = _trial(target=1)
    pointer_source = tr.PointerSourceEvidence(
        _id(tr.EvidenceKind.POINTER_SOURCE, 0), "pointer", 0, True
    )
    update = tr.PointerUpdateEvidence(
        _id(tr.EvidenceKind.POINTER_UPDATE, 0), 100, pointer_source.id, trial
    )
    target = tr.TargetOnsetEvidence(_id(tr.EvidenceKind.TARGET_ONSET, 0), trial, 1, 0)
    selection = tr.WebGridSelectionEvidence(
        _id(tr.EvidenceKind.SELECTION, 0),
        update.id,
        target.id,
        trial,
        52,
        1,
        1,
        True,
        100,
        0,
    )
    completed = tr.WebGridTrialEvidence(
        _id(tr.EvidenceKind.TRIAL, 0),
        selection.id,
        target.id,
        trial,
        52,
        1,
        1,
        1,
        True,
        0,
        100,
        True,
    )
    metric = tr.MetricContributionEvidence(
        _id(tr.EvidenceKind.METRIC_CONTRIBUTION, 0), selection.id, 1, 0, 100
    )
    return tr.WebGridProvenanceIndex(
        _recording(),
        pointer_sources=(pointer_source,),
        pointer_updates=(update,),
        targets=(target,),
        selections=(selection,),
        trials=(completed,),
        metrics=(metric,),
    )


def _speech_index() -> tr.SpeechProvenanceIndex:
    trial = _trial(stimulus=1)
    schedule = tr.SpeechScheduleEvidence(_id(tr.EvidenceKind.SPEECH_SCHEDULE, 0), trial, 1)
    phases = (
        tr.SpeechPhaseEvidence(_id(tr.EvidenceKind.SPEECH_PHASE, 0), schedule.id, 0, 0, 10),
        tr.SpeechPhaseEvidence(_id(tr.EvidenceKind.SPEECH_PHASE, 1), schedule.id, 1, 10, 20),
        tr.SpeechPhaseEvidence(_id(tr.EvidenceKind.SPEECH_PHASE, 2), schedule.id, 2, 20, 30),
    )
    requests = tuple(
        tr.SpeechPresentationRequestEvidence(
            _id(tr.EvidenceKind.PRESENTATION_REQUEST, idx),
            phase.id,
            trial,
            phase.start_ns,
            phase.start_ns,
            1 if idx == 2 else 0,
        )
        for idx, phase in enumerate(phases)
    )
    outcome = tr.SpeechPresentationOutcomeEvidence(
        _id(tr.EvidenceKind.PRESENTATION_OUTCOME, 0),
        requests[-1].id,
        trial,
        True,
        20,
        1,
        21,
        1,
    )
    marker = tr.AcquisitionMarkerEvidence(_id(tr.EvidenceKind.ACQUISITION_MARKER, 0), trial, 20, 1)
    acquisition = tr.SpeechAcquisitionEvidence(
        _id(tr.EvidenceKind.ACQUISITION_INTERVAL, 0), trial, 0, 30, (marker.id,)
    )
    completed = tr.SpeechTrialEvidence(_id(tr.EvidenceKind.TRIAL, 0), schedule.id, trial, 1)
    return tr.SpeechProvenanceIndex(
        _recording(),
        schedules=(schedule,),
        phases=phases,
        requests=requests,
        outcomes=(outcome,),
        acquisitions=(acquisition,),
        markers=(marker,),
        trials=(completed,),
    )


def _native_module() -> Any:
    import neurale._native as native

    return native


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("build/m8-provenance.jsonl"))
    parser.add_argument("--warmups", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=4096)
    args = parser.parse_args()
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be non-negative and repeats must be positive")

    cases = (
        (
            "center_out",
            _center_out_index(),
            lambda idx: idx.resolve(0),
            9,
            "command_ordinal=0;source_ranges=2;recording_complete=true",
        ),
        (
            "webgrid",
            _webgrid_index(),
            lambda idx: idx.resolve(0),
            6,
            "selection_ordinal=0;pointer_stream=pointer;recording_complete=true",
        ),
        (
            "speech",
            _speech_index(),
            lambda idx: idx.resolve(0),
            11,
            "trial_ordinal=0;phases=black,cross,content;actual_content_outcome=true;recording_complete=true",
        ),
    )
    metadata = benchmark_metadata(_native_module())
    command = shlex.join([sys.executable, *sys.argv])
    rows: list[dict[str, Any]] = []
    for paradigm, idx, resolve, evidence_records, configuration in cases:
        query = partial(resolve, idx)
        summary = measure_call(query, args.repeats, warmups=args.warmups)
        resolved = resolve(idx)
        rows.append(
            {
                **metadata,
                "benchmark": "m8_provenance_query",
                "paradigm": paradigm,
                "scenario": "complete_representative_trace",
                "configuration": configuration,
                "metric": "offline_query_latency",
                "unit": "ns",
                "samples": summary.samples,
                **summary.fields("ns"),
                "p50_ns": summary.median_ns,
                "queries_per_second": summary.throughput(1),
                "warmups": args.warmups,
                "repeats": args.repeats,
                "evidence_records": evidence_records,
                "provenance_verdict": resolved.verdict.name.lower(),
                "provenance_gap_count": len(resolved.gaps),
                "execution_class": "offline_noncritical",
                "allocation_metric": "not_measured",
                "command": command,
                "passed": resolved.verdict is tr.ProvenanceVerdict.COMPLETE,
            }
        )
    write_records(args.output, rows)
    if not all(row["passed"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
