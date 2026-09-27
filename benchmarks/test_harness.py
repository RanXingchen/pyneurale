#!/usr/bin/env python3

from __future__ import annotations

import csv
import json

import pytest

from benchmarks._harness import measure_call, summarize_ns, write_records


def test_timing_summary_reports_median_and_tail_statistics() -> None:
    summary = summarize_ns([30, 10, 20])

    assert summary.samples == 3
    assert summary.minimum_ns == 10
    assert summary.median_ns == 20
    assert summary.p95_ns == pytest.approx(29)
    assert summary.p99_ns == pytest.approx(29.8)
    assert summary.maximum_ns == 30
    assert summary.throughput(2) == pytest.approx(100_000_000)


def test_measure_call_keeps_warmups_outside_reported_samples() -> None:
    calls = 0

    def operation() -> None:
        nonlocal calls
        calls += 1

    summary = measure_call(operation, repeats=2, warmups=3)

    assert calls == 5
    assert summary.samples == 2


def test_machine_readable_writers_preserve_flat_records(tmp_path) -> None:
    rows = [{"benchmark": "example", "median_ms": 1.25}]
    csv_path = tmp_path / "result.csv"
    jsonl_path = tmp_path / "result.jsonl"

    write_records(csv_path, rows)
    write_records(jsonl_path, rows)

    with csv_path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream)) == [{"benchmark": "example", "median_ms": "1.25"}]
    assert [json.loads(line) for line in jsonl_path.read_text().splitlines()] == rows
