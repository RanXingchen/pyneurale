#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""``Recording -> NRF -> Recording`` must not change what it was given.

The first typed writer round-tripped the payload but not its meaning: every
stream was pinned to one synthetic clock, every stream was declared regular
regardless of its timestamps, and ``Recording.metadata`` and a standalone
``Recording.subject`` were dropped. Data that survives while its time base and
provenance quietly change is worse than data that fails to write, so each
property below is asserted on the restored object rather than on the bytes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from neurale.data import (
    ChannelInfo,
    ChannelTable,
    Clock,
    Event,
    EventSeries,
    FeatureMatrix,
    Recording,
    SignalArray,
)
from neurale.io.nrf import NrfReader, NrfSemanticError, write_recording
from neurale.io.nrf._semantics import validate_manifest

from .nrf_support import CREATED_AT, SESSION_ID

ACQUISITION = Clock(
    name="acquisition",
    type="device",
    rate=30000.0,
    epoch="power-on",
    offset=0.25,
    drift=1.5e-6,
    synchronization_domain="rig",
)
HOST = Clock(name="host", type="utc", rate=1e9, epoch="unix", offset=-0.5)


def _channels(count: int, unit: str = "uV") -> ChannelTable:
    return ChannelTable(
        [ChannelInfo(name=f"E{idx}", index=idx, type="ecog", unit=unit) for idx in range(count)]
    )


def _signal(
    name: str = "neural",
    *,
    rows: int = 8,
    fs: float = 1000.0,
    time: np.ndarray | None = None,
    t0: float | None = 0.0,
    clock: Clock | None = None,
) -> SignalArray:
    return SignalArray(
        data=np.arange(rows * 2, dtype="float64").reshape(rows, 2),
        fs=fs,
        time=time,
        t0=t0,
        clock=clock,
        channels=_channels(2),
        unit="uV",
        name=name,
    )


def _round_trip(tmp_path: Path, recording: Recording, tag: str) -> tuple[Recording, dict[str, Any]]:
    root = write_recording(
        tmp_path / f"{tag}.nrf", recording, session_id=SESSION_ID, created_at=CREATED_AT
    )
    with NrfReader.open(root, verify_checksums=True) as reader:
        assert reader.legacy_termination_normal
        validate_manifest(reader.manifest)
        return reader.to_recording(), dict(reader.manifest)


def _timing(manifest: dict[str, Any], stream_id: str) -> dict[str, Any]:
    return next(stream["timing"] for stream in manifest["streams"] if stream["id"] == stream_id)


# --- 1. clocks ------------------------------------------------------------


def test_distinct_clocks_stay_distinct(tmp_path: Path) -> None:
    recording = Recording(
        signals={
            "neural": _signal("neural", clock=ACQUISITION),
            "cursor": _signal("cursor", fs=500.0, clock=HOST),
        }
    )
    restored, manifest = _round_trip(tmp_path, recording, "clocks")

    assert {clock["id"] for clock in manifest["clocks"]} >= {"acquisition", "host"}
    assert restored.signals["neural"].clock == ACQUISITION
    assert restored.signals["cursor"].clock == HOST


def test_one_shared_clock_is_registered_once(tmp_path: Path) -> None:
    recording = Recording(
        signals={
            "neural": _signal("neural", clock=ACQUISITION),
            "lfp": _signal("lfp", clock=ACQUISITION),
        }
    )
    _, manifest = _round_trip(tmp_path, recording, "shared")

    clock_ids = [stream["clock_id"] for stream in manifest["streams"]]
    assert clock_ids[0] == clock_ids[1]
    assert [clock["id"] for clock in manifest["clocks"]].count(clock_ids[0]) == 1


def test_signal_without_clock_restores_without_one(tmp_path: Path) -> None:
    # NRF requires every stream to name a clock, so the writer supplies a
    # placeholder. Restoring it as a real clock would invent provenance.
    restored, manifest = _round_trip(
        tmp_path, Recording(signals={"neural": _signal(clock=None)}), "noclock"
    )
    assert restored.signals["neural"].clock is None
    assert manifest["clocks"]  # NRF-SEM-008: the registry is never empty


def test_clock_type_outside_vocabulary_round_trips(tmp_path: Path) -> None:
    clock = Clock(name="amplifier", type="fpga-counter", rate=25000.0)
    restored, manifest = _round_trip(
        tmp_path, Recording(signals={"neural": _signal(clock=clock)}), "vocab"
    )
    descriptor = next(entry for entry in manifest["clocks"] if entry["id"] == "amplifier")
    assert descriptor["type"] == "external"  # the format's own vocabulary
    assert restored.signals["neural"].clock == clock  # the recording's spelling


def test_events_keep_their_recorded_clock(tmp_path: Path) -> None:
    recording = Recording(
        signals={"neural": _signal(clock=ACQUISITION)},
        events=EventSeries([Event(onset=0.1, duration=0.0, label="go")], clock=HOST),
    )
    restored, _ = _round_trip(tmp_path, recording, "evclock")
    assert restored.events is not None
    assert restored.events.clock == HOST


# --- 2. timing ------------------------------------------------------------


def test_regular_timing_is_kept_regular(tmp_path: Path) -> None:
    restored, manifest = _round_trip(
        tmp_path, Recording(signals={"neural": _signal(t0=0.5)}), "regular"
    )
    assert _timing(manifest, "neural")["mode"] == "regular"
    assert restored.signals["neural"].t0 == pytest.approx(0.5)
    assert restored.signals["neural"].fs == 1000.0


def test_irregular_signal_timestamps_survive(tmp_path: Path) -> None:
    stamps = np.array([0.0, 0.001, 0.005, 0.0051, 0.02, 0.021, 0.03, 0.1])
    recording = Recording(signals={"neural": _signal(time=stamps, t0=None)})
    restored, manifest = _round_trip(tmp_path, recording, "irregular")

    assert _timing(manifest, "neural")["mode"] == "explicit"
    assert np.allclose(restored.signals["neural"].time, stamps)
    # The declared rate is a declaration, not an average of the timestamps.
    assert restored.signals["neural"].fs == 1000.0


def test_irregular_features_are_not_written_at_one_hertz(tmp_path: Path) -> None:
    stamps = np.array([0.0, 0.3, 0.9, 2.5])
    features = FeatureMatrix(
        data=np.arange(8, dtype="float64").reshape(4, 2),
        fs=None,
        time=stamps,
        feature_names=["f0", "f1"],
        window_size=0.1,
        shift=0.05,
    )
    restored, manifest = _round_trip(tmp_path, Recording(features={"bp": features}), "irregfeat")

    assert _timing(manifest, "bp")["mode"] == "explicit"
    assert restored.features["bp"].fs is None
    assert np.allclose(restored.features["bp"].time, stamps)


def test_regular_features_keep_their_start_time(tmp_path: Path) -> None:
    features = FeatureMatrix(
        data=np.zeros((4, 2)),
        fs=10.0,
        t0=5.0,
        feature_names=["f0", "f1"],
        window_size=0.1,
        shift=0.05,
    )
    restored, _ = _round_trip(tmp_path, Recording(features={"bp": features}), "featt0")
    assert restored.features["bp"].t0 == pytest.approx(5.0)
    assert np.allclose(restored.features["bp"].time, 5.0 + np.arange(4) / 10.0)


def test_unrepresentable_decimal_rate_stays_exact(tmp_path: Path) -> None:
    # A fixed 1e-6 scaling turns 1000/3 Hz into 333.333333 Hz, which drifts by
    # milliseconds across a long stream. The stored ratio must reproduce the
    # declared float, and the stream must therefore stay regular.
    rate = 1000.0 / 3.0
    recording = Recording(signals={"neural": _signal(rows=2000, fs=rate)})
    restored, manifest = _round_trip(tmp_path, recording, "thirds")

    assert _timing(manifest, "neural")["mode"] == "regular"
    assert restored.signals["neural"].fs == rate


# --- 3. recording metadata ------------------------------------------------


def _annotated() -> Recording:
    return Recording(
        signals={"neural": _signal(clock=ACQUISITION)},
        metadata={"experiment": "center-out", "targets": 8, "notes": {"rig": "A"}},
        subject={"id": "subject-01", "species": "macaque"},
        session={"id": "session-01", "operator": "rx"},
    )


def test_recording_metadata_and_subject_survive_independently(tmp_path: Path) -> None:
    restored, _ = _round_trip(tmp_path, _annotated(), "annotated")

    assert restored.metadata["experiment"] == "center-out"
    assert restored.metadata["targets"] == 8
    assert restored.metadata["notes"] == {"rig": "A"}
    # The subject is not smuggled in through the session dict.
    assert restored.subject == {"id": "subject-01", "species": "macaque"}
    assert restored.session == {"id": "session-01", "operator": "rx"}


def test_free_form_session_dict_does_not_corrupt_session(tmp_path: Path) -> None:
    # manifest.session is a closed vocabulary whose id is a UUID, so an
    # arbitrary Recording.session cannot simply be merged into it.
    _, manifest = _round_trip(tmp_path, _annotated(), "sessionshape")
    assert manifest["session"]["id"] == SESSION_ID
    assert manifest["session"]["subject"] == {"id": "subject-01", "species": "macaque"}
    assert manifest["session"]["metadata"] == {"id": "session-01", "operator": "rx"}


def test_rewriting_read_recording_is_idempotent(tmp_path: Path) -> None:
    once, _ = _round_trip(tmp_path, _annotated(), "once")
    twice, _ = _round_trip(tmp_path, once, "twice")

    assert twice.metadata["experiment"] == "center-out"
    assert twice.subject == once.subject
    assert twice.session == once.session
    # Reader-generated keys are stripped on the way back in rather than nested.
    assert "nrf_metadata" not in twice.metadata.get("nrf_metadata", {})


def test_metadata_nrf_cannot_store_is_refused_by_name(tmp_path: Path) -> None:
    recording = Recording(
        signals={"neural": _signal()},
        metadata={"trace": np.arange(4)},
    )
    with pytest.raises(NrfSemanticError, match=r"Recording\.metadata\.trace"):
        write_recording(
            tmp_path / "badmeta.nrf", recording, session_id=SESSION_ID, created_at=CREATED_AT
        )
