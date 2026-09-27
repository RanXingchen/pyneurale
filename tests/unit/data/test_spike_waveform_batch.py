#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

import copy
import dataclasses
import pickle
from dataclasses import FrozenInstanceError
from datetime import datetime
from enum import Enum
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, Clock, SpikeWaveformBatch
from neurale.data._helpers import is_strongly_immutable
from neurale.exceptions import ValidationError


def _batch(n_spikes: int = 3, **changes) -> SpikeWaveformBatch:
    sample_indices = np.asarray([100, 110, 110][:n_spikes], dtype=np.int64)
    times = 1.0 + (sample_indices.astype(float) - 100.0) / 1_000.0
    values = {
        "waveforms": np.arange(n_spikes * 5 * 2, dtype=np.float32).reshape(n_spikes, 5, 2),
        "sample_indices": sample_indices,
        "times": times,
        "peak_channel_indices": np.asarray([0, 1, 0][:n_spikes], dtype=np.int64),
        "electrode_group_ids": np.asarray([4, 4, 7][:n_spikes], dtype=np.int64),
        "amps": np.asarray([-12.0, 9.0, -7.0][:n_spikes], dtype=np.float32),
        "channels": ChannelTable(
            (
                ChannelInfo("left", 10, "spike", "uV", attrs={"bank": "A"}),
                ChannelInfo("right", 20, "spike", "uV", attrs={"bank": "B"}),
            )
        ),
        "fs": 1_000.0,
        "clock": Clock(
            "acquisition",
            "device",
            rate=1_000.0,
            synchronization_domain="session",
            attrs={"serial": "synthetic"},
        ),
        "source_stream": "wideband",
        "segment_id": "segment-2",
        "pre_samples": 2,
        "post_samples": 2,
        "polarity": "both",
        "spike_polarities": np.asarray([-1, 1, -1][:n_spikes], dtype=np.int8),
        "attrs": {"detector": {"name": "threshold", "levels": np.array([4.5, 5.0])}},
    }
    values.update(changes)
    return SpikeWaveformBatch(**values)


def test_public_batch_layout_alignment_metadata_and_repr() -> None:
    batch = _batch()

    assert batch.waveforms.shape == (3, 5, 2)
    assert batch.axis_order == ("spike", "sample", "channel")
    assert (batch.n_spikes, batch.n_samples, batch.n_channels) == (3, 5, 2)
    assert len(batch) == 3
    np.testing.assert_array_equal(batch.waveforms[:, batch.pre_samples, :], batch.waveforms[:, 2])
    assert batch.channels.names == ["left", "right"]
    assert batch.channels.indices == [10, 20]
    assert batch.clock is not None
    assert batch.clock.synchronization_domain == "session"
    assert batch.source_stream == "wideband"
    assert batch.segment_id == "segment-2"
    assert repr(batch) == (
        "SpikeWaveformBatch(n_spikes=3, n_samples=5, n_channels=2, "
        "source_stream='wideband', segment_id='segment-2', polarity='both')"
    )


def test_batch_owns_immutable_arrays_and_deep_frozen_metadata() -> None:
    waveforms = np.zeros((1, 5, 2), dtype=np.float64)
    attrs = {"nested": {"labels": ["a", "b"], "array": np.array([1, 2])}}
    batch = _batch(
        1,
        waveforms=waveforms,
        attrs=attrs,
        spike_polarities=np.array([-1], dtype=np.int8),
    )

    waveforms[0, 0, 0] = 99.0
    attrs["nested"]["labels"].append("changed")
    assert batch.waveforms[0, 0, 0] == 0.0
    assert batch.attrs["nested"]["labels"] == ("a", "b")
    for arr in (
        batch.waveforms,
        batch.sample_indices,
        batch.times,
        batch.peak_channel_indices,
        batch.electrode_group_ids,
        batch.amps,
        batch.spike_polarities,
        batch.attrs["nested"]["array"],
    ):
        assert arr is not None
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.flags.writeable = True
    with pytest.raises(TypeError):
        batch.attrs["new"] = "value"
    with pytest.raises(FrozenInstanceError):
        batch.segment_id = "other"


def test_batch_reuses_strongly_immutable_waveform_backing() -> None:
    source = np.arange(10, dtype=np.float64).reshape(1, 5, 2)
    backing = source.tobytes()
    immutable = np.frombuffer(backing, dtype=np.float64).reshape(source.shape)

    batch = _batch(1, waveforms=immutable, spike_polarities=np.array([-1], dtype=np.int8))

    assert batch.waveforms is immutable
    assert batch.waveforms.base is not None
    assert np.shares_memory(batch.waveforms, immutable)
    with pytest.raises(ValueError):
        batch.waveforms.flags.writeable = True


def test_batch_channel_impedance_metadata_is_deeply_immutable() -> None:
    channels = ChannelTable(
        (
            ChannelInfo("left", 10, "spike", "uV", impedance=1.5 + 2.0j),
            ChannelInfo("right", 20, "spike", "uV", impedance=np.float64(3.0)),
        )
    )
    channel_attr = ChannelInfo("reference", 30, "spike", "uV", impedance=4.0 + 0.5j)
    batch = _batch(channels=channels, attrs={"reference_channel": channel_attr})

    assert batch.channels[0].impedance == 1.5 + 2.0j
    assert batch.channels[1].impedance == 3.0
    assert batch.attrs["reference_channel"].impedance == 4.0 + 0.5j
    with pytest.raises(FrozenInstanceError):
        batch.channels[0].impedance = 9.0
    with pytest.raises(FrozenInstanceError):
        batch.attrs["reference_channel"].impedance = 9.0

    for transported in (pickle.loads(pickle.dumps(batch)), copy.deepcopy(batch)):
        assert transported.channels[0].impedance == 1.5 + 2.0j
        assert transported.attrs["reference_channel"].impedance == 4.0 + 0.5j
        with pytest.raises(FrozenInstanceError):
            transported.channels[0].impedance = 9.0


def test_batch_stays_immutable_through_pickle_and_copy() -> None:
    # ``pickle.loads`` and ``copy.deepcopy`` skip ``__post_init__``, so without
    # an explicit rebuild protocol they restore writable arrays and mutable
    # metadata, breaking the immutable batch contract. The rebuild protocol
    # routes both back through the public constructor, which re-freezes every
    # array and metadata container.
    batch = _batch()

    restored = pickle.loads(pickle.dumps(batch))
    copied = copy.deepcopy(batch)

    for arr in (
        restored.waveforms,
        restored.sample_indices,
        restored.times,
        restored.peak_channel_indices,
        restored.electrode_group_ids,
        restored.amps,
        restored.spike_polarities,
        restored.attrs["detector"]["levels"],
        copied.waveforms,
        copied.sample_indices,
        copied.times,
        copied.peak_channel_indices,
        copied.electrode_group_ids,
        copied.amps,
        copied.spike_polarities,
        copied.attrs["detector"]["levels"],
    ):
        assert arr is not None
        assert not arr.flags.writeable
        with pytest.raises(ValueError):
            arr.flags.writeable = True

    # Re-entering the constructor also validates the rebuilt state, so the
    # restored copies still satisfy the public invariants.
    assert restored.axis_order == ("spike", "sample", "channel")
    assert copied.axis_order == ("spike", "sample", "channel")
    np.testing.assert_array_equal(restored.waveforms, batch.waveforms)
    np.testing.assert_array_equal(copied.waveforms, batch.waveforms)
    assert restored.attrs["detector"]["name"] == "threshold"
    assert copied.attrs["detector"]["name"] == "threshold"

    # The deep copy is an independent object graph, not shared references.
    assert copied.waveforms is not batch.waveforms
    assert copied.channels is not batch.channels
    assert copied.clock is not batch.clock


def test_pickle_and_deepcopy_refreeze_channel_and_clock_attrs() -> None:
    # ``FrozenMapping.__reduce__`` re-freezes nested ndarrays, so arrays stashed
    # inside ``ChannelInfo.attrs`` and ``Clock.attrs`` stay strongly immutable
    # after pickle and deep copy -- not only the batch-level ``attrs``. Without
    # the rebuild protocol, ``np.ndarray`` reconstruction restores them as
    # writable arrays inside the frozen metadata containers.
    channels = ChannelTable(
        (
            ChannelInfo("left", 10, "spike", "uV", attrs={"calibration": np.array([1.0, 2.0])}),
            ChannelInfo("right", 20, "spike", "uV", attrs={"calibration": np.array([3.0, 4.0])}),
        )
    )
    clock = Clock(
        "acquisition",
        "device",
        rate=1_000.0,
        synchronization_domain="session",
        attrs={"coefficients": np.array([5.0, 6.0])},
    )
    batch = _batch(channels=channels, clock=clock)

    for restored in (pickle.loads(pickle.dumps(batch)), copy.deepcopy(batch)):
        channel_array = restored.channels[0].attrs["calibration"]
        clock_array = restored.clock.attrs["coefficients"]
        for arr in (channel_array, clock_array):
            assert not arr.flags.writeable
            with pytest.raises(ValueError):
                arr.flags.writeable = True
        np.testing.assert_array_equal(channel_array, [1.0, 2.0])
        np.testing.assert_array_equal(clock_array, [5.0, 6.0])


def test_batch_accepts_supported_metadata_leaf_types() -> None:
    # The metadata value domain is closed but covers the immutable leaves and
    # recursive containers used in practice: scalars, NumPy scalars, bytes,
    # sequences, sets, ndarrays, and nested combinations. Object ndarrays are
    # converted with ``tolist()`` and recursively frozen as tuples.
    supported = {
        "none": None,
        "bool": True,
        "int": 7,
        "float": 2.5,
        "str": "text",
        "bytes": b"raw",
        "np_int": np.int64(3),
        "np_float": np.float64(1.25),
        "np_bool": np.bool_(False),
        "array": np.array([1.0, 2.0]),
        "object_array": np.array([["a", 1], ["b", 2]], dtype=object),
        "tuple": (1, "a"),
        "frozenset": frozenset({1, 2}),
        "nested": {"deep": [1, 2, {"x": np.array([0])}]},
    }
    batch = _batch(1, attrs=supported)

    assert set(batch.attrs) == set(supported)
    assert not batch.attrs["array"].flags.writeable
    assert batch.attrs["object_array"] == (("a", 1), ("b", 2))
    assert not batch.attrs["nested"]["deep"][2]["x"].flags.writeable
    assert batch.attrs["tuple"] == (1, "a")
    assert batch.attrs["frozenset"] == frozenset({1, 2})


def test_batch_rejects_mutable_metadata_payload() -> None:
    # ``freeze_metadata`` only freezes known containers; a mutable leaf such as
    # ``bytearray`` would survive ``copy.deepcopy`` and remain writable inside
    # the batch, breaking the deep-immutability contract. Unsupported mutable
    # types are therefore rejected at construction.
    with pytest.raises(ValidationError, match=r"metadata value of type 'bytearray'"):
        _batch(1, attrs={"payload": bytearray(b"ab")})


class _MetadataEnum(Enum):
    VALUE = "value"


@pytest.mark.parametrize(
    "payload",
    [
        datetime(2026, 7, 28),
        UUID("12345678-1234-5678-1234-567812345678"),
        Path("session.json"),
        _MetadataEnum.VALUE,
    ],
)
def test_batch_rejects_unsupported_metadata_leaf_types(payload) -> None:
    with pytest.raises(ValidationError, match="metadata value of type"):
        _batch(1, attrs={"payload": payload})


@dataclasses.dataclass
class _MutablePayload:
    values: list


def test_batch_rejects_non_frozen_dataclass_metadata() -> None:
    # A non-frozen dataclass is mutable; only frozen dataclasses are admitted
    # as metadata leaves.
    with pytest.raises(ValidationError, match=r"metadata value of type '_MutablePayload'"):
        _batch(1, attrs={"payload": _MutablePayload([1, 2])})


def test_batch_accepts_frozen_dataclass_metadata() -> None:
    # Whitelisted project dataclasses (e.g. a ``ChannelTable``) are admitted
    # as metadata leaves. The stored value is an independent frozen copy, not
    # an alias of the caller's object.
    table = ChannelTable(
        (ChannelInfo("left", 10, "spike", "uV"), ChannelInfo("right", 20, "spike", "uV"))
    )
    batch = _batch(1, attrs={"channels": table})

    stored = batch.attrs["channels"]
    assert isinstance(stored, ChannelTable)
    assert stored is not table
    assert stored.names == ["left", "right"]
    with pytest.raises(FrozenInstanceError):
        stored.channels = ()


@dataclasses.dataclass(frozen=True)
class _FrozenButMutable:
    values: list


def test_batch_rejects_frozen_dataclass_with_mutable_fields() -> None:
    # ``frozen=True`` only forbids rebinding fields; it does not make the
    # referenced objects immutable. A frozen dataclass carrying a ``list`` can
    # be mutated in place (``payload.values.append(...)``), so it is not a
    # trusted immutable leaf and must be rejected just like a non-frozen
    # dataclass. Only the whitelisted project dataclasses are admitted.
    payload = _FrozenButMutable([1, 2])
    with pytest.raises(ValidationError, match=r"metadata value of type '_FrozenButMutable'"):
        _batch(1, attrs={"payload": payload})


def test_dataclass_metadata_stays_immutable_after_copy() -> None:
    # A whitelisted dataclass stored in attrs must stay deeply immutable across
    # pickle and deepcopy. Its ndarrays live inside ``FrozenMapping`` attrs,
    # which ``_rebuild_frozen_mapping`` re-freezes on reconstruction -- this
    # guards against ``copy.deepcopy`` restoring a read-only ndarray as
    # writable, which is the failure mode that disqualifies arbitrary frozen
    # dataclasses with direct ndarray fields.
    table = ChannelTable(
        (
            ChannelInfo("left", 10, "spike", "uV", attrs={"cal": np.array([1.0, 2.0])}),
            ChannelInfo("right", 20, "spike", "uV", attrs={"cal": np.array([3.0, 4.0])}),
        )
    )
    batch = _batch(1, attrs={"channels": table})

    def assert_immutable(transported: SpikeWaveformBatch) -> None:
        restored = transported.attrs["channels"]
        assert isinstance(restored, ChannelTable)
        assert restored is not table
        for channel in restored:
            arr = channel.attrs["cal"]
            assert not arr.flags.writeable
            with pytest.raises(ValueError):
                arr.flags.writeable = True
        np.testing.assert_array_equal(restored[0].attrs["cal"], [1.0, 2.0])

    assert_immutable(pickle.loads(pickle.dumps(batch)))
    assert_immutable(copy.deepcopy(batch))


def test_indexing_and_slicing_preserve_batch_contract() -> None:
    batch = _batch()

    single = batch[1]
    assert isinstance(single, SpikeWaveformBatch)
    assert single.waveforms.shape == (1, 5, 2)
    np.testing.assert_array_equal(single.sample_indices, [110])
    assert single.channels == batch.channels
    assert single.clock == batch.clock
    assert single.attrs["detector"]["name"] == "threshold"
    np.testing.assert_array_equal(single.attrs["detector"]["levels"], [4.5, 5.0])

    np.testing.assert_array_equal(batch[:2].sample_indices, [100, 110])
    np.testing.assert_array_equal(batch[-1].peak_channel_indices, [0])
    np.testing.assert_array_equal(
        batch[np.array([True, False, True])].electrode_group_ids,
        [4, 7],
    )
    assert batch[0:0].waveforms.shape == (0, 5, 2)
    with pytest.raises(ValidationError, match="monotonic"):
        batch[[2, 0]]
    with pytest.raises(IndexError, match="out of range"):
        batch[3]
    with pytest.raises(ValidationError, match="wrong length"):
        batch[np.array([True, False])]
    with pytest.raises(TypeError, match="boolean scalar"):
        batch[True]


def test_contiguous_slice_shares_immutable_backing_without_copy() -> None:
    # A contiguous slice (default step) returns views of the already-immutable
    # backing arrays, so the subset shares memory with the original instead of
    # being double-copied. The views remain strongly immutable. Non-contiguous
    # selectors still copy and re-freeze.
    batch = _batch()

    subset = batch[:2]
    assert np.shares_memory(subset.waveforms, batch.waveforms)
    assert np.shares_memory(subset.sample_indices, batch.sample_indices)
    assert not subset.waveforms.flags.writeable
    with pytest.raises(ValueError):
        subset.waveforms.flags.writeable = True
    np.testing.assert_array_equal(subset.waveforms, batch.waveforms[:2])

    strided = batch[::2]
    assert not np.shares_memory(strided.waveforms, batch.waveforms)
    assert not strided.waveforms.flags.writeable


def test_strong_immutability_distinguishes_bytes_backed() -> None:
    # ``flags.writeable == False`` is not proof of strong immutability: a view
    # of a writable base can be marked read-only yet re-enabled. The robust
    # proof is a ``bytes`` backing, which NumPy cannot expose for writing.
    base = np.zeros((1, 3, 1))
    soft = base.view()
    soft.flags.writeable = False
    assert not soft.flags.writeable
    assert not is_strongly_immutable(soft)
    # Re-enabling succeeds and mutates the base -- exactly the hole the trusted
    # path must close.
    soft.flags.writeable = True
    soft[0, 0, 0] = 5.0
    assert base[0, 0, 0] == 5.0

    batch = _batch(1)
    for arr in (
        batch.waveforms,
        batch.sample_indices,
        batch.times,
        batch.spike_polarities,
    ):
        assert is_strongly_immutable(arr)
    # Contiguous slice views of a bytes-backed batch stay strongly immutable.
    for arr in (
        batch.waveforms[:1],
        batch.sample_indices[:1],
        batch.times[:1],
    ):
        assert is_strongly_immutable(arr)


def test_contiguous_slice_derives_immutable_arrays() -> None:
    # ``_contiguous_slice`` derives the subset from ``self`` (structural
    # provenance) instead of accepting caller-supplied arrays, so every array in
    # the subset is a strongly immutable view of the batch's bytes backing.
    batch = _batch(3)
    subset = batch[:2]
    assert np.shares_memory(subset.waveforms, batch.waveforms)
    for arr in (
        subset.waveforms,
        subset.sample_indices,
        subset.times,
        subset.peak_channel_indices,
        subset.electrode_group_ids,
        subset.amps,
        subset.spike_polarities,
    ):
        assert is_strongly_immutable(arr)
        with pytest.raises(ValueError):
            arr.flags.writeable = True


def test_trusted_initialize_rejects_soft_frozen_view() -> None:
    # Defense-in-depth: even if a soft-frozen view reached the trusted path,
    # ``_initialize(freeze_arrays=False)`` rejects it via ``is_strongly_immutable``
    # rather than accepting it on ``flags.writeable`` alone. This is the exact
    # scenario the removed general ``_from_trusted`` entry let through.
    base = np.zeros((1, 5, 2), dtype=np.float32)
    soft = base.view()
    soft.flags.writeable = False

    valid = _batch(1)
    obj = object.__new__(SpikeWaveformBatch)
    for name in (
        "waveforms",
        "sample_indices",
        "times",
        "peak_channel_indices",
        "electrode_group_ids",
        "amps",
        "channels",
        "fs",
        "clock",
        "source_stream",
        "segment_id",
        "pre_samples",
        "post_samples",
        "polarity",
        "spike_polarities",
        "attrs",
    ):
        object.__setattr__(obj, name, getattr(valid, name))
    object.__setattr__(obj, "waveforms", soft)

    with pytest.raises(RuntimeError, match="strongly immutable"):
        obj._initialize(freeze_arrays=False)


def test_empty_and_unsigned_spike_selectors_select_correctly() -> None:
    # An empty selector selects no spikes regardless of the dtype ``np.asarray``
    # inferred: a Python ``[]`` becomes a zero-length float64 array, and an
    # empty typed array is already integer, but both are legal empty subsets.
    batch = _batch()

    for empty in (
        [],
        np.array([], dtype=np.int64),
        np.array([], dtype=np.uint64),
    ):
        subset = batch[empty]
        assert subset.n_spikes == 0
        assert subset.waveforms.shape == (0, 5, 2)

    # An unsigned selector that fits in int64 selects the same spikes as its
    # signed counterpart.
    np.testing.assert_array_equal(
        batch[np.array([0, 2], dtype=np.uint64)].sample_indices,
        batch[np.array([0, 2], dtype=np.int64)].sample_indices,
    )

    # An unsigned value above int64 max must be rejected on the original dtype
    # before the int64 cast: otherwise it wraps to a negative int64 and the
    # negative-index rewrite silently returns a trailing spike.
    with pytest.raises(IndexError, match="out of range"):
        batch[np.array([np.iinfo(np.uint64).max], dtype=np.uint64)]
    with pytest.raises(IndexError, match="out of range"):
        batch[np.array([np.iinfo(np.int64).max + 1], dtype=np.uint64)]


def test_concatenate_rejects_incompatible_batches() -> None:
    batch = _batch()
    joined = SpikeWaveformBatch.concatenate((batch[:2], batch[2:]))

    np.testing.assert_array_equal(joined.waveforms, batch.waveforms)
    np.testing.assert_array_equal(joined.sample_indices, batch.sample_indices)
    np.testing.assert_array_equal(joined.spike_polarities, batch.spike_polarities)
    assert joined.channels == batch.channels
    assert joined.clock == batch.clock
    assert joined.attrs["detector"]["name"] == "threshold"
    np.testing.assert_array_equal(joined.attrs["detector"]["levels"], [4.5, 5.0])

    with pytest.raises(ValidationError, match="at least one"):
        SpikeWaveformBatch.concatenate(())
    with pytest.raises(ValidationError, match="segment_id"):
        SpikeWaveformBatch.concatenate((batch, _batch(segment_id="segment-3")))
    with pytest.raises(ValidationError, match="source_stream"):
        SpikeWaveformBatch.concatenate((batch, _batch(source_stream="filtered")))
    with pytest.raises(ValidationError, match="attrs"):
        SpikeWaveformBatch.concatenate((batch, _batch(attrs={"detector": "other"})))
    other_channels = ChannelTable.from_names(
        ("different-left", "different-right"),
        type="spike",
        unit="uV",
    )
    with pytest.raises(ValidationError, match="channels"):
        SpikeWaveformBatch.concatenate((batch, _batch(channels=other_channels)))
    with pytest.raises(ValidationError, match="monotonic"):
        SpikeWaveformBatch.concatenate((batch[2:], batch[:2]))


def test_concatenate_ignores_attr_insertion_order() -> None:
    # Mapping equality is order-independent, so two batches whose attrs differ
    # only in key insertion order are semantically identical and must
    # concatenate. This holds for the batch attrs, the ``Clock.attrs`` nested
    # mapping, and the per-channel ``ChannelInfo.attrs`` nested mapping. Each
    # pair uses single-spike batches with continuing sample indices so the
    # concatenated result stays on one regular grid.

    def continuing_batch(
        *, sample_idx: int, time: float, attrs=None, clock=None, channels=None
    ) -> SpikeWaveformBatch:
        overrides = {
            "sample_indices": np.asarray([sample_idx], dtype=np.int64),
            "times": np.asarray([time], dtype=float),
        }
        if attrs is not None:
            overrides["attrs"] = attrs
        if clock is not None:
            overrides["clock"] = clock
        if channels is not None:
            overrides["channels"] = channels
        return _batch(1, **overrides)

    joined_attrs = SpikeWaveformBatch.concatenate(
        (
            continuing_batch(sample_idx=100, time=1.0, attrs={"a": 1, "b": 2}),
            continuing_batch(sample_idx=110, time=1.01, attrs={"b": 2, "a": 1}),
        )
    )
    assert joined_attrs.n_spikes == 2

    joined_nested = SpikeWaveformBatch.concatenate(
        (
            continuing_batch(sample_idx=100, time=1.0, attrs={"detector": {"x": 1, "y": 2}}),
            continuing_batch(sample_idx=110, time=1.01, attrs={"detector": {"y": 2, "x": 1}}),
        )
    )
    assert joined_nested.n_spikes == 2

    clock_first = Clock(
        "acquisition",
        "device",
        rate=1_000.0,
        synchronization_domain="session",
        attrs={"serial": "synthetic", "gain": 2.0},
    )
    clock_second = Clock(
        "acquisition",
        "device",
        rate=1_000.0,
        synchronization_domain="session",
        attrs={"gain": 2.0, "serial": "synthetic"},
    )
    joined_clock = SpikeWaveformBatch.concatenate(
        (
            continuing_batch(sample_idx=100, time=1.0, clock=clock_first),
            continuing_batch(sample_idx=110, time=1.01, clock=clock_second),
        )
    )
    assert joined_clock.n_spikes == 2

    channels_first = ChannelTable(
        (
            ChannelInfo("left", 10, "spike", "uV", attrs={"bank": "A", "gain": 2.0}),
            ChannelInfo("right", 20, "spike", "uV", attrs={"bank": "B", "gain": 1.0}),
        )
    )
    channels_second = ChannelTable(
        (
            ChannelInfo("left", 10, "spike", "uV", attrs={"gain": 2.0, "bank": "A"}),
            ChannelInfo("right", 20, "spike", "uV", attrs={"gain": 1.0, "bank": "B"}),
        )
    )
    joined_channels = SpikeWaveformBatch.concatenate(
        (
            continuing_batch(sample_idx=100, time=1.0, channels=channels_first),
            continuing_batch(sample_idx=110, time=1.01, channels=channels_second),
        )
    )
    assert joined_channels.n_spikes == 2

    # Genuinely different attrs are still rejected, regardless of order.
    with pytest.raises(ValidationError, match="attrs"):
        SpikeWaveformBatch.concatenate(
            (
                continuing_batch(sample_idx=100, time=1.0, attrs={"a": 1}),
                continuing_batch(sample_idx=110, time=1.01, attrs={"a": 2}),
            )
        )
    with pytest.raises(ValidationError, match="attrs"):
        SpikeWaveformBatch.concatenate(
            (
                continuing_batch(sample_idx=100, time=1.0, attrs={"a": 1, "b": 2}),
                continuing_batch(sample_idx=110, time=1.01, attrs={"a": 1}),
            )
        )


def test_empty_batch_is_valid_and_preserves_declared_shape() -> None:
    batch = _batch(0)

    assert batch.waveforms.shape == (0, 5, 2)
    assert len(batch) == 0
    assert batch.spike_polarities is not None
    assert batch.spike_polarities.shape == (0,)
    with pytest.raises(IndexError, match="out of range"):
        batch[0]


def test_large_sample_indices_keep_relative_time_valid() -> None:
    indices = np.asarray(
        [np.iinfo(np.int64).max - 2, np.iinfo(np.int64).max - 1, np.iinfo(np.int64).max - 1],
        dtype=np.int64,
    )
    batch = _batch(
        sample_indices=indices,
        times=np.asarray([1.0, 1.001, 1.001], dtype=np.float64),
    )

    np.testing.assert_array_equal(batch.sample_indices, indices)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_indices", np.array([100, 110], dtype=np.int64), "length"),
        ("times", np.array([1.0, 1.01]), "length"),
        ("peak_channel_indices", np.array([0, 1], dtype=np.int64), "length"),
        ("electrode_group_ids", np.array([4, 4], dtype=np.int64), "length"),
        ("amps", np.array([1.0, 2.0]), "length"),
    ],
)
def test_all_per_spike_arrays_must_match_waveform_count(field, value, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _batch(**{field: value})


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"waveforms": np.zeros((3, 5))}, "3D"),
        ({"waveforms": np.zeros((3, 0, 2))}, "at least one sample"),
        ({"waveforms": np.zeros((3, 5, 0))}, "at least one channel"),
        ({"waveforms": np.full((3, 5, 2), np.nan)}, "finite"),
        ({"waveforms": np.zeros((3, 5, 2), dtype=complex)}, "real numeric"),
        ({"sample_indices": np.array([100, 110, 110], dtype=np.int32)}, "int64"),
        ({"sample_indices": np.array([100, -1, 110], dtype=np.int64)}, "non-negative"),
        ({"sample_indices": np.array([100, 99, 110], dtype=np.int64)}, "monotonic"),
        ({"times": np.array([1.0, np.nan, 1.01])}, "finite"),
        ({"times": np.array([1.0, 1.01 + 1j, 1.01])}, "real"),
        ({"times": np.array([1.0, -0.01, 1.01])}, "non-negative"),
        ({"times": np.array([1.0, 0.9, 1.01])}, "monotonic"),
        ({"times": np.array([1.0, 1.02, 1.02])}, "regular grid"),
        ({"peak_channel_indices": np.array([0, 1, 0], dtype=np.int32)}, "int64"),
        ({"peak_channel_indices": np.array([0, 2, 0], dtype=np.int64)}, "positions"),
        ({"peak_channel_indices": np.array([0, -1, 0], dtype=np.int64)}, "positions"),
        ({"electrode_group_ids": np.array([4, 4, 7], dtype=np.int32)}, "int64"),
        ({"electrode_group_ids": np.array([4, -1, 7], dtype=np.int64)}, "non-negative"),
        ({"amps": np.zeros((3, 1))}, "1D"),
        ({"amps": np.array([-1.0, np.inf, 2.0])}, "finite"),
        ({"fs": 0.0}, "positive"),
        ({"clock": "device"}, "Clock"),
        ({"source_stream": ""}, "non-empty"),
        ({"segment_id": -1}, "non-negative"),
        ({"segment_id": ""}, "non-empty"),
        ({"pre_samples": -1}, "non-negative"),
        ({"post_samples": -1}, "non-negative"),
        ({"pre_samples": 1}, "pre_samples"),
        ({"polarity": "falling"}, "polarity"),
        ({"spike_polarities": None}, "required"),
        ({"spike_polarities": np.array([-1, 1, -1], dtype=np.int64)}, "int8"),
        ({"spike_polarities": np.array([-1, 0, 1], dtype=np.int8)}, "-1 or 1"),
        ({"attrs": ["not", "a", "mapping"]}, "mapping"),
    ],
)
def test_invalid_batch_contracts_are_rejected(changes, message) -> None:
    with pytest.raises(ValidationError, match=message):
        _batch(**changes)


def test_channel_metadata_and_polarity_are_validated() -> None:
    one_channel = ChannelTable.from_names(("only",), type="spike", unit="uV")
    with pytest.raises(ValidationError, match="channels length"):
        _batch(channels=one_channel)
    with pytest.raises(ValidationError, match="ChannelTable"):
        _batch(channels=("left", "right"))
    with pytest.raises(ValidationError, match="only valid"):
        _batch(polarity="negative", spike_polarities=np.array([-1, -1, -1], dtype=np.int8))

    batch = _batch(polarity="negative", spike_polarities=None)
    assert batch.spike_polarities is None
