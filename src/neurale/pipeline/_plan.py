#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Versioned plans and compiler for fixed built-in native pipelines."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from neurale._native_loader import load_native_namespace
from neurale.exceptions import ValidationError
from neurale.streaming import ExecutionProfile, RealtimeConfig, StreamRunner
from neurale.streaming._config import _native_realtime_config

from ._stages import StageSpec, stage_from_document

_PLAN_VERSION = 1


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"pipeline plan document repeats field {key!r}.")
        result[key] = value
    return result


def _canonical(document: Mapping[str, object]) -> bytes:
    try:
        return json.dumps(
            dict(document),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"pipeline plan is not canonical JSON data: {exc}") from exc


@dataclass(frozen=True, slots=True)
class PipelinePlan:
    """A fixed ordered list of reviewed native processing stages."""

    stages: tuple[StageSpec, ...]

    def __post_init__(self) -> None:
        values = tuple(self.stages)
        if not values:
            raise ValidationError("PipelinePlan.stages must not be empty.")
        object.__setattr__(self, "stages", values)

    def to_document(self) -> dict[str, object]:
        """Return a fresh versioned JSON-compatible plan document."""

        value = {
            "kind": "pipeline",
            "version": _PLAN_VERSION,
            "stages": [stage.to_document() for stage in self.stages],
        }
        return value

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> PipelinePlan:
        """Validate and construct one version-1 built-in pipeline plan."""

        if not isinstance(document, Mapping):
            raise ValidationError("pipeline plan document must be an object.")
        values = dict(document)
        version = values.get("version")
        if values.get("kind") != "pipeline" or type(version) is not int or version != _PLAN_VERSION:
            raise ValidationError("only pipeline plan version 1 is supported.")
        if set(values) != {"kind", "version", "stages"} or not isinstance(values["stages"], list):
            raise ValidationError("pipeline plan document has an invalid top-level shape.")
        return cls(tuple(stage_from_document(value) for value in values["stages"]))

    @property
    def fingerprint(self) -> str:
        """SHA-256 of the canonical versioned plan document."""

        return sha256(_canonical(self.to_document())).hexdigest()


class CompiledPipeline:
    """One owning native processor chain compiled from a :class:`PipelinePlan`."""

    __slots__ = ("_attached", "_native", "plan")

    def __init__(self, plan: PipelinePlan, native: object) -> None:
        self.plan = plan
        self._native = native
        self._attached = False

    @property
    def stage_count(self) -> int:
        return int(self._native.stage_count)  # type: ignore[attr-defined]

    def output_schema(self, input_schema: object) -> object:
        """Prepare the immutable native stage contracts and return the final schema."""

        probe = _native_realtime_config(
            RealtimeConfig(), ExecutionProfile.RESEARCH, schema_probe=True
        )
        return self._native.output_schema(input_schema, probe)  # type: ignore[attr-defined]

    def enable_training_capture(self, capacity: int) -> None:
        """Allocate a bounded capture of the decoder's selected feature rows."""

        self._native.enable_training_capture(capacity)  # type: ignore[attr-defined]

    def pop_training_observation(self) -> tuple[int, tuple[float, ...]] | None:
        """Pop one selected feature row outside the realtime path."""

        value = self._native.pop_training_observation()  # type: ignore[attr-defined]
        if value is None:
            return None
        return int(value[0]), tuple(float(item) for item in value[1])

    @property
    def training_capture_drops(self) -> int:
        return int(self._native.training_capture_drops)  # type: ignore[attr-defined]

    def create_runner(
        self,
        schema: object,
        config: RealtimeConfig,
        source: object,
        actuator: object,
        *,
        profile: ExecutionProfile | str,
        safety_controller: object | None = None,
    ) -> StreamRunner:
        """Create the existing runtime with this native chain as its processor."""

        if self._attached:
            raise RuntimeError("a compiled pipeline can be attached to only one StreamRunner")
        runner = StreamRunner(
            schema,
            config,
            source,
            self._native,
            actuator,
            profile=profile,
            safety_controller=safety_controller,
        )
        self._attached = True
        return runner


def compile_pipeline(plan: PipelinePlan) -> CompiledPipeline:
    """Construct all native stages without starting or preparing a runtime."""

    if not isinstance(plan, PipelinePlan):
        raise TypeError("plan must be a PipelinePlan")
    native = load_native_namespace("pipeline")
    streaming = load_native_namespace("streaming")
    builder = native._PipelineBuilder()
    for stage in plan.stages:
        stage._add_to(builder, native, streaming)
    return CompiledPipeline(plan, builder.build())


def save_plan(plan: PipelinePlan, path: str | Path) -> None:
    """Write canonical UTF-8 JSON with one trailing newline."""

    destination = Path(path)
    destination.write_bytes(_canonical(plan.to_document()) + b"\n")


def load_plan(path: str | Path) -> PipelinePlan:
    """Load one version-1 built-in pipeline plan without executing code."""

    try:
        document = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=_unique_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"could not read pipeline plan: {exc}") from exc
    return PipelinePlan.from_document(document)


__all__ = ["CompiledPipeline", "PipelinePlan", "compile_pipeline", "load_plan", "save_plan"]
