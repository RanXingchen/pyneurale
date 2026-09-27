#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Build the PyNeurale C++ and Python documentation."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "docs"
BUILD_DIR = DOCS_DIR / "_build"
DOXYGEN_XML = BUILD_DIR / "doxygen" / "xml" / "index.xml"
HTML_DIR = BUILD_DIR / "html"
GENERATED_API_DIR = DOCS_DIR / "api" / "_generated"


def run(command: list[str]) -> None:
    """Run a documentation command from the repository root."""

    print("+", " ".join(command), flush=True)

    environment = os.environ.copy()
    source_path = str(ROOT / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def clean() -> None:
    """Remove generated documentation artifacts."""

    docs_root = DOCS_DIR.resolve()

    for generated_path in (BUILD_DIR, GENERATED_API_DIR):
        resolved_path = generated_path.resolve()

        if docs_root not in resolved_path.parents:
            raise RuntimeError(f"Refusing to remove unexpected path: {resolved_path}")

        shutil.rmtree(resolved_path, ignore_errors=True)


def build_doxygen() -> None:
    """Generate the Doxygen XML consumed by Breathe."""

    executable = shutil.which("doxygen")
    if executable is None:
        raise RuntimeError(
            "Doxygen is required but was not found on PATH. "
            "See docs/development/index.md for installation instructions."
        )
    (BUILD_DIR / "doxygen").mkdir(parents=True, exist_ok=True)
    run([executable, str(DOCS_DIR / "Doxyfile")])
    if not DOXYGEN_XML.is_file():
        raise RuntimeError(f"Doxygen did not generate {DOXYGEN_XML}")
    normalize_doxygen_xml()


def normalize_doxygen_xml() -> None:
    """Remove duplicate declarations emitted by Doxygen for Breathe."""

    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    removed_ids: set[str] = set()
    for path in DOXYGEN_XML.parent.glob("class*.xml"):
        tree = ET.parse(path)
        compound = tree.getroot().find("compounddef")
        if compound is None:
            continue
        class_name = compound.findtext("compoundname", "")
        constructor_name = class_name.rsplit("::", 1)[-1]
        changed = False
        for section in compound.findall("sectiondef"):
            members = section.findall("memberdef")
            override_ids = {
                member.find("reimplements").get("refid")
                for member in members
                if member.find("reimplements") is not None
                and (member.findtext("definition") or "").endswith(
                    f"{class_name}::{member.findtext('name')}"
                )
            }
            for member in members:
                name = member.findtext("name")
                type_node = member.find("type")
                if (
                    name == constructor_name
                    and member.get("constexpr") == "yes"
                    and type_node is not None
                    and type_node.text == "constexpr"
                ):
                    type_node.text = None
                    changed = True
                inherited = member.find("reimplements")
                if (
                    inherited is not None
                    and inherited.get("refid") in override_ids
                    and not (member.findtext("definition") or "").endswith(f"{class_name}::{name}")
                ):
                    removed_ids.add(member.get("id", ""))
                    section.remove(member)
                    changed = True
        if changed:
            for entry in compound.findall("./listofallmembers/member"):
                if entry.get("refid") in removed_ids:
                    compound.find("listofallmembers").remove(entry)
            tree.write(path, encoding="UTF-8", xml_declaration=True)

    if removed_ids:
        tree = ET.parse(DOXYGEN_XML)
        for compound in tree.getroot().findall("compound"):
            for member in compound.findall("member"):
                if member.get("refid") in removed_ids:
                    compound.remove(member)
        tree.write(DOXYGEN_XML, encoding="UTF-8", xml_declaration=True)


def build_sphinx() -> None:
    """Build strict Sphinx HTML documentation."""

    if not DOXYGEN_XML.is_file():
        raise RuntimeError("Doxygen XML is missing. Run the full build or --doxygen-only first.")
    run(
        [
            sys.executable,
            "-m",
            "sphinx",
            "-W",
            "--keep-going",
            "-j",
            "auto",
            "-b",
            "html",
            str(DOCS_DIR),
            str(HTML_DIR),
        ]
    )
    (HTML_DIR / ".nojekyll").touch()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove generated files before building",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--doxygen-only",
        action="store_true",
        help="generate only the Doxygen XML",
    )
    mode.add_argument(
        "--sphinx-only",
        action="store_true",
        help="build Sphinx using existing Doxygen XML",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.clean:
            clean()
        if not args.sphinx_only:
            build_doxygen()
        if not args.doxygen_only:
            build_sphinx()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"documentation build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
