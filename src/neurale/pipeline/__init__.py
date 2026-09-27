#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Typed control plane for composing reviewed realtime BCI components.

Importing this module does not load a native extension, discover hardware,
create a window, or allocate runtime resources. Native stages are constructed
only by :func:`compile_pipeline`.
"""

from ._deployment import decoder_stage
from ._plan import CompiledPipeline, PipelinePlan, compile_pipeline, load_plan, save_plan
from ._stages import (
    BadChannelRemovalStage,
    Band,
    FeatureSpec,
    FeatureStage,
    FilterStage,
    FittedFeatureContract,
    HilbertEnvelopeFeature,
    KalmanDecoderStage,
    LdaDecoderStage,
    LinearDecoderStage,
    LineNoiseFilterStage,
    LmpFeature,
    MultitaperBandpowerFeature,
    ResampleStage,
    SpatialReferenceStage,
    SpikeDetectorStage,
    StageSpec,
    supported_stage_kinds,
)

__all__ = [
    "BadChannelRemovalStage",
    "Band",
    "CompiledPipeline",
    "FeatureSpec",
    "FeatureStage",
    "FilterStage",
    "FittedFeatureContract",
    "HilbertEnvelopeFeature",
    "KalmanDecoderStage",
    "LdaDecoderStage",
    "LineNoiseFilterStage",
    "LinearDecoderStage",
    "LmpFeature",
    "MultitaperBandpowerFeature",
    "PipelinePlan",
    "ResampleStage",
    "SpatialReferenceStage",
    "SpikeDetectorStage",
    "StageSpec",
    "compile_pipeline",
    "decoder_stage",
    "load_plan",
    "save_plan",
    "supported_stage_kinds",
]
