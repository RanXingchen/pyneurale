# Benchmark regression policy

PyNeurale separates deterministic contract gates from statistical performance
regression gates. The machine-readable inventory is
`benchmarks/regression_policy.json`.

- Steady-state allocation requirements are correctness contracts. Their existing
  CTest allocation gates continue to require zero allocations on the documented
  prepared paths.
- Latency, throughput, RSS, storage, pacing, and presentation measurements are
  not hard gates until independent runs on one controlled runner demonstrate
  reproducibility and a maintainer approves a baseline and threshold.
- GitHub-hosted benchmark jobs are characterization only. They validate output
  structure and archive evidence, but changing hosted hardware cannot fail a
  release on timing.
- Developer-laptop results may diagnose a change but cannot create or update a
  repository threshold.

## Historical eligibility

A performance metric is only eligible for a threshold proposal when all inputs:

1. come from a runner class marked `controlled`;
2. have distinct evidence-run IDs;
3. have identical hardware/software and benchmark-definition fingerprints;
4. contain every scenario with at least the configured repetitions;
5. include at least the configured number of independent runs; and
6. keep each scenario's relative median absolute deviation and full relative
   range within the configured bounds.

The eligibility report is evidence, not approval. A hard performance gate also
requires an explicit checked-in baseline, threshold, approving change, and the
historical run IDs that justify it. The current policy intentionally has no
active latency, throughput, memory, recording/replay, acquisition, or
presentation performance gate because that historical evidence is not present
in the repository.

The controlled history command is:

```text
python tools/benchmark_policy.py assess \
  --policy benchmarks/regression_policy.json \
  --runner-class pyneurale-linux-x64-benchmark-v1 \
  --input <evidence-directory> [--input <evidence-directory> ...] \
  --output benchmark-eligibility.json
```

Each evidence directory contains `environment.json` plus the JSONL file named by
that manifest. The environment manifest records the exact command, revision,
runner class/labels, OS/kernel, CPU, compiler, build/provider/thread settings,
and a checksum of the result file. Raw JSONL and the manifest must be retained
together.

Smoke runs may use reduced duration or repetitions to prove that an entry point
still executes and emits valid machine-readable output. Smoke results must never
be mixed into history or used to justify a hard threshold.
