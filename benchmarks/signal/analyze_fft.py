#!/usr/bin/env python3

"""Summarize the signal FFT benchmark and validate automatic dispatch."""

from __future__ import annotations

import argparse

from _dispatch_analysis import add_arguments, analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    add_arguments(parser, max_regression=1.10)
    analyze(
        parser.parse_args(),
        key_fields=("threads", "length", "channels"),
        builtin_kernel="builtin-fft",
    )


if __name__ == "__main__":
    main()
