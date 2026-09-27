#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import json
import sys
from itertools import combinations, permutations

import pytest

import neurale.pipeline as realtime
import neurale.streaming as streaming
from neurale.exceptions import ValidationError


def _schema():
    signal = streaming.SignalSchema(
        1,
        streaming.SignalDType.FLOAT64,
        2,
        4,
        4,
        streaming.RationalRate(1_000, 1),
        7,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
    )
    return streaming.StreamSchema(11, [signal])


def _named_schema():
    signal = streaming.SignalSchema(
        1,
        streaming.SignalDType.FLOAT64,
        4,
        4,
        4,
        streaming.RationalRate(1_000, 1),
        7,
        device_tick_tracking=streaming.DeviceTickTracking.SAMPLE_COUNTER,
        channel_names=("C1", "C2", "C3", "C4"),
        channel_impedances_ohm=(1_000.0, 5_000.0, 20_000.0, 100_000.0),
    )
    return streaming.StreamSchema(11, [signal])


def _config():
    return streaming.RealtimeConfig()


def _plan() -> realtime.PipelinePlan:
    return realtime.PipelinePlan(
        (
            realtime.SpatialReferenceStage(),
            realtime.FilterStage(filter_type="lowpass", cutoff_hz=100.0, filter_order=2),
        )
    )


def _features():
    return (
        realtime.MultitaperBandpowerFeature(
            bands=(realtime.Band("alpha", 8.0, 12.0),),
            n_tapers=3,
        ),
        realtime.HilbertEnvelopeFeature(
            bands=(realtime.Band("alpha", 8.0, 12.0),),
            filter_order=2,
        ),
        realtime.LmpFeature(
            cutoff_hz=4.0,
            filter_order=2,
        ),
    )


def _feature_stage(features) -> realtime.FeatureStage:
    return realtime.FeatureStage(
        window_seconds=0.008,
        update_interval_seconds=0.004,
        features=tuple(features),
    )


def test_import_is_lightweight() -> None:
    code = "import sys; import neurale.pipeline; print('neurale._native' in sys.modules)"
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    assert result.stdout.strip() == "False"


def test_compile_pipeline_requires_a_plan() -> None:
    with pytest.raises(TypeError, match="plan must be a PipelinePlan"):
        realtime.compile_pipeline(realtime.FeatureStage())


def test_plan_round_trip_is_canonical_and_fingerprinted(tmp_path) -> None:
    plan = _plan()
    path = tmp_path / "pipeline.json"

    realtime.save_plan(plan, path)
    restored = realtime.load_plan(path)

    assert restored == plan
    assert restored.fingerprint == plan.fingerprint
    assert plan.fingerprint == "2f3e16dd834ff345e47359b1c51b36b9c0f998b96f32d1d79f83d9b20023b252"
    assert path.read_bytes().endswith(b"\n")
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1
    assert realtime.PipelinePlan.from_document(plan.to_document()) == plan
    assert plan.to_document() is not plan.to_document()


def test_supported_stage_kinds_are_fixed_and_complete() -> None:
    assert realtime.supported_stage_kinds() == (
        "bad_channel_removal",
        "spatial_reference",
        "filter",
        "line_noise_filter",
        "resample",
        "feature",
        "spike_detector",
        "linear_decoder",
        "lda_decoder",
        "kalman_decoder",
    )


def test_coefficient_and_standalone_feature_stages_are_not_public() -> None:
    removed = (
        "FirFilterStage",
        "IirFilterStage",
        "SosFilterStage",
        "LmpStage",
        "HilbertEnvelopeStage",
        "MultitaperBandpowerStage",
        "UnitSpec",
        "FeatureBranchSpec",
        "FeatureStackStage",
        "HilbertEnvelopeBranch",
        "LmpBranch",
        "MultitaperBandpowerBranch",
    )

    assert all(name not in realtime.__all__ for name in removed)
    assert all(not hasattr(realtime, name) for name in removed)


def test_filter_stage_has_minimal_keyword_only_design_api() -> None:
    stage = realtime.FilterStage(filter_type="lowpass", cutoff_hz=30)

    assert stage.cutoff_hz == 30.0
    assert stage.filter_order == 4
    assert stage.filter_kind == "butterworth"
    assert stage.to_document() == {
        "kind": "filter",
        "filter_type": "lowpass",
        "cutoff_hz": 30.0,
        "filter_order": 4,
        "filter_kind": "butterworth",
    }
    with pytest.raises(TypeError):
        realtime.FilterStage("lowpass", 30.0)  # type: ignore[misc]


@pytest.mark.parametrize(
    ("filter_type", "cutoff_hz"),
    (
        ("lowpass", 30.0),
        ("highpass", 1.0),
        ("bandpass", (8.0, 30.0)),
        ("bandstop", (48.0, 52.0)),
    ),
)
def test_filter_stage_round_trips_and_prepares(filter_type, cutoff_hz) -> None:
    stage = realtime.FilterStage(filter_type=filter_type, cutoff_hz=cutoff_hz)
    restored = realtime.PipelinePlan.from_document(
        realtime.PipelinePlan((stage,)).to_document()
    ).stages[0]

    assert restored == stage
    assert realtime.compile_pipeline(realtime.PipelinePlan((stage,))).output_schema(_schema())


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: realtime.FilterStage(filter_type="lowpass", cutoff_hz=(1.0, 2.0)),
            "one frequency",
        ),
        (lambda: realtime.FilterStage(filter_type="bandpass", cutoff_hz=1.0), "two frequencies"),
        (
            lambda: realtime.FilterStage(filter_type="bandstop", cutoff_hz=(20.0, 10.0)),
            "strictly increasing",
        ),
        (lambda: realtime.FilterStage(filter_type="lowpass", cutoff_hz=0.0), "one frequency"),
        (
            lambda: realtime.FilterStage(
                filter_type="lowpass", cutoff_hz=10.0, filter_kind="bessel", filter_order=17
            ),
            "must not exceed 16",
        ),
        (
            lambda: realtime.FilterStage(
                filter_type="lowpass", cutoff_hz=10.0, filter_kind="elliptic"
            ),
            "elliptic filters require",
        ),
    ),
)
def test_filter_stage_validates_design(factory, message) -> None:
    with pytest.raises(ValidationError, match=message):
        factory()


def test_filter_stage_accepts_elliptic_design_parameters() -> None:
    stage = realtime.FilterStage(
        filter_type="bandpass",
        cutoff_hz=(8.0, 30.0),
        filter_order=6,
        filter_kind="elliptic",
        passband_ripple_db=0.5,
        stopband_attenuation_db=40.0,
    )

    assert stage.to_document()["passband_ripple_db"] == 0.5
    assert stage.to_document()["stopband_attenuation_db"] == 40.0


def test_filter_stage_validates_cutoff_against_prepared_sampling_rate() -> None:
    pipeline = realtime.compile_pipeline(
        realtime.PipelinePlan((realtime.FilterStage(filter_type="lowpass", cutoff_hz=500.0),))
    )

    with pytest.raises(ValueError, match="strictly below Nyquist"):
        pipeline.output_schema(_schema())


def test_line_noise_filter_has_practical_defaults_and_round_trips() -> None:
    stage = realtime.LineNoiseFilterStage()

    assert stage.to_document() == {
        "kind": "line_noise_filter",
        "frequency_hz": 50.0,
        "bandwidth_hz": 2.0,
        "harmonics": 3,
        "filter_order": 2,
    }
    restored = realtime.PipelinePlan.from_document(
        realtime.PipelinePlan((stage,)).to_document()
    ).stages[0]
    assert restored == stage
    assert realtime.compile_pipeline(realtime.PipelinePlan((stage,))).output_schema(_schema())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"frequency_hz": 0.0}, "frequency_hz"),
        ({"bandwidth_hz": 0.0}, "bandwidth_hz"),
        ({"frequency_hz": 1.0, "bandwidth_hz": 2.0}, "less than twice"),
        ({"harmonics": 0}, "harmonics"),
        ({"filter_order": 0}, "filter_order"),
    ),
)
def test_line_noise_filter_validates_design(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        realtime.LineNoiseFilterStage(**kwargs)


def test_line_noise_filter_validates_notch_against_sampling_rate() -> None:
    stage = realtime.LineNoiseFilterStage(frequency_hz=500.0)
    pipeline = realtime.compile_pipeline(realtime.PipelinePlan((stage,)))

    with pytest.raises(ValueError, match="strictly below Nyquist"):
        pipeline.output_schema(_schema())


@pytest.mark.parametrize("bad_channels", ((1, 3), ("C2", "C4")))
def test_bad_channel_removal_round_trips_and_updates_schema(bad_channels) -> None:
    stage = realtime.BadChannelRemovalStage(bad_channels)
    plan = realtime.PipelinePlan((stage,))

    assert realtime.PipelinePlan.from_document(plan.to_document()) == plan
    input_schema = _named_schema()
    output_schema = realtime.compile_pipeline(plan).output_schema(input_schema)
    assert output_schema.id == input_schema.id + 1
    assert output_schema.signals[0].n_channels == 2
    assert output_schema.signals[0].channel_names == ["C1", "C3"]
    assert output_schema.signals[0].channel_impedances_ohm == [1_000.0, 20_000.0]
    assert output_schema.signals[0].channel_set_id == input_schema.signals[0].channel_set_id + 1


@pytest.mark.parametrize(
    ("bad_channels", "message"),
    (
        ((1, "C2"), "not both"),
        ((True,), "not both"),
        ((1, 1), "duplicates"),
        (("C2", "C2"), "duplicates"),
    ),
)
def test_bad_channel_removal_validates_selectors(bad_channels, message) -> None:
    with pytest.raises(ValidationError, match=message):
        realtime.BadChannelRemovalStage(bad_channels)


def test_bad_channel_removal_filters_schema_impedances_with_inclusive_limits() -> None:
    stage = realtime.BadChannelRemovalStage(
        min_impedance_ohm=5_000,
        max_impedance_ohm=20_000,
    )

    output = realtime.compile_pipeline(realtime.PipelinePlan((stage,))).output_schema(
        _named_schema()
    )

    assert output.signals[0].channel_names == ["C2", "C3"]
    assert output.signals[0].channel_impedances_ohm == [5_000.0, 20_000.0]
    assert (
        realtime.PipelinePlan.from_document(realtime.PipelinePlan((stage,)).to_document()).stages[0]
        == stage
    )


def test_bad_channel_removal_combines_explicit_and_impedance_rules() -> None:
    stage = realtime.BadChannelRemovalStage(
        ("C2",),
        min_impedance_ohm=1_000,
        max_impedance_ohm=20_000,
    )

    output = realtime.compile_pipeline(realtime.PipelinePlan((stage,))).output_schema(
        _named_schema()
    )

    assert output.signals[0].channel_names == ["C1", "C3"]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({}, "requires bad_channels or an impedance limit"),
        ({"min_impedance_ohm": -1}, "min_impedance_ohm"),
        ({"max_impedance_ohm": float("inf")}, "max_impedance_ohm"),
        (
            {"min_impedance_ohm": 20_000, "max_impedance_ohm": 5_000},
            "must not exceed",
        ),
    ),
)
def test_bad_channel_removal_validates_impedance_limits(kwargs, message) -> None:
    with pytest.raises(ValidationError, match=message):
        realtime.BadChannelRemovalStage(**kwargs)


def test_bad_channel_removal_requires_impedance_metadata_when_filtering() -> None:
    pipeline = realtime.compile_pipeline(
        realtime.PipelinePlan((realtime.BadChannelRemovalStage(max_impedance_ohm=20_000),))
    )

    with pytest.raises(ValueError, match="channel_impedances_ohm"):
        pipeline.output_schema(_schema())


def test_bad_channel_removal_rejects_unknown_and_all_channels() -> None:
    for bad_channels, message in ((("missing",), "absent"), ((0, 1, 2, 3), "every")):
        pipeline = realtime.compile_pipeline(
            realtime.PipelinePlan((realtime.BadChannelRemovalStage(bad_channels),))
        )
        with pytest.raises(ValueError, match=message):
            pipeline.output_schema(_named_schema())


def test_resample_stage_does_not_accept_filter_coefficients() -> None:
    stage = realtime.ResampleStage(2, 2, 1)

    assert not hasattr(stage, "filter")
    assert stage.to_document() == {"kind": "resample", "output_schema_id": 2, "up": 2, "down": 1}
    output = realtime.compile_pipeline(realtime.PipelinePlan((stage,))).output_schema(_schema())
    assert output.signals[0].fs.numerator == 2_000
    assert output.signals[0].fs.denominator == 1
    named_output = realtime.compile_pipeline(realtime.PipelinePlan((stage,))).output_schema(
        _named_schema()
    )
    assert named_output.signals[0].channel_names == ["C1", "C2", "C3", "C4"]
    assert named_output.signals[0].channel_impedances_ohm == [
        1_000.0,
        5_000.0,
        20_000.0,
        100_000.0,
    ]
    with pytest.raises(TypeError):
        realtime.ResampleStage(2, 2, 1, (1.0,))  # type: ignore[call-arg]


def test_filter_related_public_apis_have_no_coefficient_parameters() -> None:
    forbidden = {"a", "b", "coefficients", "filter", "sos", "sos_sections", "taps"}

    for public_type in (
        realtime.FilterStage,
        realtime.LineNoiseFilterStage,
        realtime.ResampleStage,
        realtime.HilbertEnvelopeFeature,
        realtime.LmpFeature,
    ):
        assert forbidden.isdisjoint(inspect.signature(public_type).parameters)


def test_loader_rejects_unknown_components(tmp_path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text(
        '{"kind":"pipeline","version":1,"stages":[{"kind":"plugin"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unsupported built-in"):
        realtime.load_plan(path)


def test_loader_does_not_accept_old_feature_stack_kind(tmp_path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text(
        '{"kind":"pipeline","version":1,"stages":[{"kind":"feature_stack"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unsupported built-in"):
        realtime.load_plan(path)


def test_loader_rejects_duplicate_fields(tmp_path) -> None:
    path = tmp_path / "pipeline.json"
    path.write_text(
        '{"kind":"pipeline","kind":"pipeline","version":1,"stages":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="repeats field"):
        realtime.load_plan(path)


@pytest.mark.parametrize("version", [True, 1.0])
def test_plan_document_rejects_non_integer_version(version) -> None:
    document = {"kind": "pipeline", "version": version, "stages": []}

    with pytest.raises(ValidationError, match="version 1"):
        realtime.PipelinePlan.from_document(document)


@pytest.mark.parametrize("value", ["8, 30", b"8, 30"])
def test_plan_document_rejects_scalar_filter_cutoff_sequence(value) -> None:
    document = {
        "kind": "pipeline",
        "version": 1,
        "stages": [
            {
                "kind": "filter",
                "filter_type": "bandpass",
                "cutoff_hz": value,
            }
        ],
    }

    with pytest.raises(ValidationError, match="sequence of finite numbers"):
        realtime.PipelinePlan.from_document(document)


@pytest.mark.parametrize(
    "stage",
    [
        realtime.BadChannelRemovalStage((1,)),
        realtime.FilterStage(filter_type="lowpass", cutoff_hz=30.0),
        realtime.LineNoiseFilterStage(),
        realtime.ResampleStage(2, 2, 1),
        _feature_stage(_features()),
        realtime.SpikeDetectorStage(
            2,
            2,
            8,
            2,
            1,
            1,
            2,
            (0.0,),
            (3.0,),
            ((0,),),
        ),
    ],
)
def test_every_preprocessing_stage_round_trips_and_compiles(stage) -> None:
    plan = realtime.PipelinePlan((stage,))

    restored = realtime.PipelinePlan.from_document(plan.to_document())

    assert restored == plan
    assert realtime.compile_pipeline(restored).stage_count == 1


def test_plan_document_rejects_implicit_scalar_conversion() -> None:
    document = {
        "kind": "pipeline",
        "version": 1,
        "stages": [{"kind": "spatial_reference", "reference_channels": [0.5]}],
    }

    with pytest.raises(ValidationError, match="non-negative integers"):
        realtime.PipelinePlan.from_document(document)


def test_loader_wraps_malformed_stage_fields(tmp_path) -> None:
    invalid_kind = {
        "kind": "pipeline",
        "version": 1,
        "stages": [{"kind": []}],
    }
    malformed_filter = {
        "kind": "pipeline",
        "version": 1,
        "stages": [
            {
                "kind": "filter",
                "filter_type": "lowpass",
                "cutoff_hz": 30.0,
                "unexpected": True,
            }
        ],
    }

    path = tmp_path / "pipeline.json"
    for document, message in (
        (invalid_kind, "unsupported built-in pipeline stage kind"),
        (malformed_filter, "invalid 'filter' stage document"),
    ):
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(ValidationError, match=message):
            realtime.load_plan(path)


def test_loader_wraps_malformed_feature_fields(tmp_path) -> None:
    path = tmp_path / "pipeline.json"
    document = realtime.PipelinePlan((_feature_stage((_features()[-1],)),)).to_document()
    document["stages"][0]["features"][0]["kind"] = []
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError, match="unsupported feature kind"):
        realtime.load_plan(path)

    document = realtime.PipelinePlan((_feature_stage((_features()[-1],)),)).to_document()
    document["stages"][0]["features"][0]["feature_unit"] = {"id": 23}
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError, match="invalid 'lmp' feature document"):
        realtime.load_plan(path)

    document = realtime.PipelinePlan((_feature_stage((_features()[-1],)),)).to_document()
    document["stages"][0]["features"][0]["algorithm_version"] = "custom"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValidationError, match="unsupported LMP algorithm_version"):
        realtime.load_plan(path)

    for feature, algorithm_name in zip(_features()[:2], ("PMTM", "Hilbert"), strict=True):
        document = realtime.PipelinePlan((_feature_stage((feature,)),)).to_document()
        document["stages"][0]["features"][0]["algorithm_version"] = "custom"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(
            ValidationError, match=f"unsupported {algorithm_name} algorithm_version"
        ):
            realtime.load_plan(path)


def test_feature_stage_supports_every_ordered_nonempty_subset() -> None:
    features = _features()
    ordered_subsets = [
        ordering
        for size in range(1, len(features) + 1)
        for subset in combinations(features, size)
        for ordering in permutations(subset)
    ]

    assert len(ordered_subsets) == 15
    for ordering in ordered_subsets:
        stage = _feature_stage(ordering)
        restored = realtime.PipelinePlan.from_document(
            realtime.PipelinePlan((stage,)).to_document()
        ).stages[0]

        assert restored == stage
        assert restored.algorithm_version == stage.algorithm_version
        assert [feature.kind for feature in restored.features] == [
            feature.kind for feature in ordering
        ]


def test_features_serialize_filter_design_not_sos() -> None:
    pmtm, hilbert, lmp = _features()

    pmtm_document = pmtm.to_document()
    hilbert_document = hilbert.to_document()
    lmp_document = lmp.to_document()

    assert hilbert_document["bands"] == ({"name": "alpha", "low_hz": 8.0, "high_hz": 12.0},)
    assert hilbert_document["filter_order"] == 2
    assert pmtm_document["algorithm_version"] == "multitaper-bandpower-v1"
    assert hilbert_document["algorithm_version"] == "hilbert-envelope-v1"
    assert lmp_document["cutoff_hz"] == 4.0
    assert lmp_document["filter_kind"] == "butterworth"
    assert lmp_document["algorithm_version"] == "lmp-v1"
    assert "feature_unit" not in lmp_document
    assert "feature_names" not in lmp_document
    assert "passband_ripple_db" not in lmp_document
    assert "stopband_attenuation_db" not in lmp_document
    assert "sos" not in hilbert_document
    assert "sos_sections" not in hilbert_document
    assert "sos" not in lmp_document
    assert "sos_sections" not in lmp_document
    for document in (pmtm_document, hilbert_document):
        assert "feature_unit_id" not in document
        assert "fft_length" not in document


def test_pmtm_feature_has_minimal_keyword_only_api() -> None:
    feature = realtime.MultitaperBandpowerFeature(bands=(realtime.Band("alpha", 8.0, 12.0),))

    assert feature.time_bandwidth == 2.5
    assert feature.n_tapers is None
    assert feature.weighting == "adaptive"
    assert (
        realtime.PipelinePlan.from_document(
            realtime.PipelinePlan((_feature_stage((feature,)),)).to_document()
        )
        .stages[0]
        .features[0]
        == feature
    )
    for removed in ("feature_unit_id", "fft_length", "algorithm_version"):
        assert not hasattr(feature, removed)
    with pytest.raises(TypeError):
        realtime.MultitaperBandpowerFeature((realtime.Band("alpha", 8.0, 12.0),))  # type: ignore[misc]
    with pytest.raises(TypeError):
        realtime.MultitaperBandpowerFeature(  # type: ignore[call-arg]
            bands=(realtime.Band("alpha", 8.0, 12.0),), fft_length=8
        )
    with pytest.raises(ValidationError, match="valid DPSS set"):
        realtime.MultitaperBandpowerFeature(bands=(realtime.Band("alpha", 8.0, 12.0),), n_tapers=5)


def test_hilbert_feature_has_minimal_keyword_only_api() -> None:
    feature = realtime.HilbertEnvelopeFeature(bands=(realtime.Band("alpha", 8.0, 12.0),))

    assert feature.filter_order == 4
    assert feature.filter_kind == "butterworth"
    for removed in ("feature_unit_id", "fft_length", "algorithm_version"):
        assert not hasattr(feature, removed)
    with pytest.raises(TypeError):
        realtime.HilbertEnvelopeFeature((realtime.Band("alpha", 8.0, 12.0),))  # type: ignore[misc]


def test_hilbert_feature_requires_elliptic_parameters_only_for_elliptic() -> None:
    bands = (realtime.Band("alpha", 8.0, 12.0),)
    with pytest.raises(ValidationError, match="elliptic filters require"):
        realtime.HilbertEnvelopeFeature(bands=bands, filter_kind="elliptic")
    with pytest.raises(ValidationError, match="only valid for elliptic"):
        realtime.HilbertEnvelopeFeature(bands=bands, passband_ripple_db=0.5)

    feature = realtime.HilbertEnvelopeFeature(
        bands=bands,
        filter_kind="elliptic",
        passband_ripple_db=0.5,
        stopband_attenuation_db=30.0,
    )
    assert feature.to_document()["passband_ripple_db"] == 0.5
    assert feature.to_document()["stopband_attenuation_db"] == 30.0


def test_lmp_feature_has_minimal_keyword_only_api() -> None:
    feature = realtime.LmpFeature(cutoff_hz=4.0)

    assert feature.filter_order == 4
    assert feature.filter_kind == "butterworth"
    assert not hasattr(feature, "algorithm_version")
    with pytest.raises(TypeError):
        realtime.LmpFeature(4.0)  # type: ignore[misc]
    with pytest.raises(TypeError):
        realtime.LmpFeature(cutoff_hz=4.0, algorithm_version="custom")  # type: ignore[call-arg]


def test_lmp_feature_requires_elliptic_parameters_only_for_elliptic() -> None:
    with pytest.raises(ValidationError, match="elliptic filters require"):
        realtime.LmpFeature(cutoff_hz=4.0, filter_kind="elliptic")
    with pytest.raises(ValidationError, match="only valid for elliptic"):
        realtime.LmpFeature(cutoff_hz=4.0, passband_ripple_db=0.5)

    feature = realtime.LmpFeature(
        cutoff_hz=4.0,
        filter_kind="elliptic",
        passband_ripple_db=0.5,
        stopband_attenuation_db=30.0,
    )
    assert feature.to_document()["passband_ripple_db"] == 0.5
    assert feature.to_document()["stopband_attenuation_db"] == 30.0


@pytest.mark.parametrize(
    ("factory", "message"),
    (
        (
            lambda: realtime.LmpFeature(cutoff_hz=0.0),
            "cutoff_hz",
        ),
        (
            lambda: realtime.LmpFeature(cutoff_hz=4.0, filter_order=0),
            "filter_order",
        ),
        (
            lambda: realtime.HilbertEnvelopeFeature(bands=(realtime.Band("dc", 0.0, 4.0),)),
            "low_hz",
        ),
        (
            lambda: realtime.LmpFeature(
                cutoff_hz=4.0,
                filter_kind="elliptic",
                passband_ripple_db=30.0,
                stopband_attenuation_db=20.0,
            ),
            "passband_ripple_db",
        ),
    ),
)
def test_features_validate_filter_design(factory, message) -> None:
    with pytest.raises(ValidationError, match=message):
        factory()


def test_feature_stage_rejects_duplicate_or_unknown_features() -> None:
    lmp = _features()[-1]

    with pytest.raises(ValidationError, match="at most once"):
        _feature_stage((lmp, lmp))
    with pytest.raises(ValidationError, match="unsupported feature kind"):
        realtime.PipelinePlan.from_document(
            {
                "kind": "pipeline",
                "version": 1,
                "stages": [
                    {
                        "kind": "feature",
                        "window_seconds": 0.008,
                        "update_interval_seconds": 0.004,
                        "features": [{"kind": "plugin"}],
                    }
                ],
            }
        )


def test_feature_stage_rejects_scalar_sequences() -> None:
    pmtm, _, _ = _features()

    with pytest.raises(ValidationError, match="features must be a sequence"):
        realtime.FeatureStage(features="lmp")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="bands must be a sequence"):
        realtime.MultitaperBandpowerFeature(bands="alpha")  # type: ignore[arg-type]
    document = realtime.PipelinePlan((_feature_stage((pmtm,)),)).to_document()
    document["stages"][0]["features"][0]["bands"] = "alpha"
    with pytest.raises(ValidationError, match="bands must be a sequence"):
        realtime.PipelinePlan.from_document(document)


def test_feature_stage_has_minimal_keyword_only_api() -> None:
    stage = realtime.FeatureStage()

    assert stage.window_seconds == 0.2
    assert stage.update_interval_seconds == 0.05
    assert stage.features == (realtime.LmpFeature(cutoff_hz=4.0),)
    assert stage.to_document()["kind"] == "feature"
    assert stage.to_document()["features"][0]["kind"] == "lmp"
    for removed in (
        "branches",
        "output_schema_id",
        "output_signal_id",
        "feature_set_id",
        "source_stream",
        "channel_names",
        "window_samples",
        "shift_samples",
    ):
        assert not hasattr(stage, removed)
    with pytest.raises(TypeError):
        realtime.FeatureStage(0.2)  # type: ignore[misc]
    with pytest.raises(TypeError):
        realtime.FeatureStage(branches=stage.features)  # type: ignore[call-arg]


def test_feature_stage_validates_durations() -> None:
    with pytest.raises(ValidationError, match="update_interval_seconds"):
        realtime.FeatureStage(window_seconds=0.1, update_interval_seconds=0.2)
    with pytest.raises(ValidationError, match="integer number of nanoseconds"):
        realtime.FeatureStage(window_seconds=1 / 3)


def test_feature_stage_prepare_derives_metadata_and_exact_samples() -> None:
    signal = streaming.SignalSchema(
        7,
        streaming.SignalDType.FLOAT64,
        2,
        4,
        4,
        streaming.RationalRate(1_000, 1),
        9,
        physical_unit=streaming.PhysicalUnit.DIMENSIONLESS,
        channel_names=["C3", "C4"],
    )
    schema = streaming.StreamSchema(11, [signal])
    pipeline = realtime.compile_pipeline(
        realtime.PipelinePlan(
            (
                realtime.FeatureStage(
                    window_seconds=0.008,
                    update_interval_seconds=0.004,
                ),
            )
        )
    )

    output = pipeline.output_schema(schema)

    assert output.id == 12
    assert output.signals[0].id == 8
    assert output.signals[0].feature_set_id == 1
    assert output.feature_sets[0].source_stream_id == 7
    assert output.feature_sets[0].source_stream == "signal-7"
    assert output.feature_sets[0].feature_names == ["lmp:C3", "lmp:C4"]

    nonintegral = realtime.compile_pipeline(
        realtime.PipelinePlan(
            (
                realtime.FeatureStage(
                    window_seconds=0.0015,
                    update_interval_seconds=0.001,
                ),
            )
        )
    )
    with pytest.raises(ValueError, match="does not resolve to an integer number of samples"):
        nonintegral.output_schema(schema)


def test_feature_stage_algorithm_version_covers_order_and_parameters() -> None:
    pmtm, hilbert, _ = _features()

    assert (
        _feature_stage((pmtm, hilbert)).algorithm_version
        != _feature_stage((hilbert, pmtm)).algorithm_version
    )
    changed = realtime.MultitaperBandpowerFeature(
        bands=(realtime.Band("beta", 13.0, 20.0),),
        time_bandwidth=pmtm.time_bandwidth,
        n_tapers=pmtm.n_tapers,
    )
    assert _feature_stage((pmtm,)).algorithm_version != _feature_stage((changed,)).algorithm_version


def test_compiled_native_chain_runs_under_realtime_profile() -> None:
    schema = _schema()
    source = streaming.SyntheticNativeSource(schema, 3)
    consumer = streaming.CountingNativeConsumer()
    pipeline = realtime.compile_pipeline(_plan())
    runner = pipeline.create_runner(
        schema,
        _config(),
        source,
        consumer,
        profile=streaming.ExecutionProfile.REALTIME,
        safety_controller=streaming.RecordingNativeSafetyController(),
    )

    assert pipeline.stage_count == 2
    assert runner.prepare() == streaming.StreamStatus.OK
    assert runner.arm() == streaming.StreamStatus.OK
    assert runner.run() == streaming.StreamStatus.OK
    assert consumer.frame_count == 3
    assert runner.outstanding_frames == 0

    with pytest.raises(RuntimeError, match="only one StreamRunner"):
        pipeline.create_runner(
            schema,
            _config(),
            streaming.SyntheticNativeSource(schema, 1),
            streaming.CountingNativeConsumer(),
            profile=streaming.ExecutionProfile.REALTIME,
            safety_controller=streaming.RecordingNativeSafetyController(),
        )
