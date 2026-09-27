#!/usr/bin/env python3

"""Summarize a FIR benchmark matrix and validate automatic dispatch."""

from __future__ import annotations

import argparse

from _dispatch_analysis import add_arguments, analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser, max_regression=1.05)
    analyze(
        parser.parse_args(),
        key_fields=("threads", "samples", "channels", "taps"),
        builtin_kernel="builtin-direct",
        report_auto_overhead=True,
    )


if __name__ == "__main__":
    main()
