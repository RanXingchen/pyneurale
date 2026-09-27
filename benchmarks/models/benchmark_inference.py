#!/usr/bin/env python3

"""Fitted PCA transform and LDA predict.

Fit is setup, not measurement: both estimators are fitted before timing starts,
so each record describes inference on already-trained parameters.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# The shared benchmark helpers stay at the root of ``benchmarks/``; this script
# lives one level down, in the directory of the domain that owns it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _harness import benchmark_metadata, timed_row, write_records

import neurale._native as runtime_native
from neurale.models import LDA, PCA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("build/models_benchmark.jsonl"),
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--samples", type=int, default=2048)
    parser.add_argument("--features", type=int, default=32)
    parser.add_argument("--queries", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    metadata = {
        **benchmark_metadata(runtime_native),
        "input_seed": args.seed,
    }
    rng = np.random.default_rng(args.seed)
    training = np.ascontiguousarray(
        rng.standard_normal((args.samples, args.features)),
        dtype=np.float64,
    )
    queries = np.ascontiguousarray(
        rng.standard_normal((args.queries, args.features)),
        dtype=np.float64,
    )
    labels = np.arange(args.samples, dtype=np.intp) % 4
    components = min(8, args.features)
    pca = PCA(n_components=components).fit(training)
    pca_out = np.empty((args.queries, components), dtype=np.float64)
    lda = LDA().fit(training, labels)

    rows = [
        timed_row(
            metadata,
            benchmark="pca",
            operation="transform",
            function=lambda: pca.transform(queries, out=pca_out),
            warmups=args.warmups,
            repeats=args.repeats,
            dtype=str(queries.dtype),
            shape=f"{args.queries}x{args.features}",
            items_per_call=args.queries,
            n_components=components,
        ),
        timed_row(
            metadata,
            benchmark="lda",
            operation="predict",
            function=lambda: lda.predict(queries),
            warmups=args.warmups,
            repeats=args.repeats,
            dtype=str(queries.dtype),
            shape=f"{args.queries}x{args.features}",
            items_per_call=args.queries,
            classes=4,
        ),
    ]

    write_records(args.output, rows)
    print(args.output)


if __name__ == "__main__":
    main()
