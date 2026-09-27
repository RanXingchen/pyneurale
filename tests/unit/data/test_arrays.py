#!/usr/bin/env python3

import pickle
from typing import get_type_hints

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, FeatureMatrix, SignalArray, SpikeTrain
from neurale.exceptions import ValidationError


def _channels():
    return ChannelTable(
        [
            ChannelInfo("A-000", 0, "ecog", "uV"),
            ChannelInfo("A-001", 1, "ecog", "uV"),
            ChannelInfo("A-002", 2, "ecog", "uV"),
        ]
    )


def test_signal_array_derives_times_duration_and_channel_names():
    signal = SignalArray(
        data=np.zeros((4, 3)),
        fs=2.0,
        time=None,
        t0=10.0,
        clock=None,
        channels=_channels(),
        unit="uV",
        name="neural",
    )

    np.testing.assert_allclose(signal.time, [10.0, 10.5, 11.0, 11.5])
    assert signal.duration == 1.5
    assert signal.n_samples == 4
    assert signal.n_channels == 3
    assert signal.channel_names == ["A-000", "A-001", "A-002"]
    assert not hasattr(signal, "times")
    assert not hasattr(signal, "time_span")


def test_empty_and_single_sample_signal_duration_is_zero():
    empty = SignalArray.from_array(
        np.empty((0, 1)),
        fs=2.0,
    )
    single = SignalArray.from_array(
        np.zeros((1, 1)),
        fs=2.0,
    )

    assert empty.duration == 0.0
    assert single.duration == 0.0


def test_signal_array_defaults_t0_to_zero_when_constructing_time():
    signal = SignalArray(
        data=np.zeros((3, 1)),
        fs=2.0,
        time=None,
        t0=None,
        clock=None,
        channels=ChannelTable.default(1),
        unit="a.u.",
        name="neural",
    )

    assert signal.t0 == 0.0
    np.testing.assert_allclose(signal.time, [0.0, 0.5, 1.0])


def test_select_channels_preserves_order_and_metadata():
    signal = SignalArray(
        data=np.arange(12).reshape(4, 3),
        fs=1000.0,
        time=None,
        t0=0.0,
        clock=None,
        channels=_channels(),
        unit=["uV", "uV", "uV"],
        name="neural",
    )

    selected = signal.select_channels(["A-002", "A-000"])

    assert selected.channel_names == ["A-002", "A-000"]
    np.testing.assert_array_equal(selected.data, signal.data[:, [2, 0]])
    assert selected.unit == ("uV", "uV")


def test_signal_array_requires_fs_even_with_explicit_time():
    with pytest.raises(ValidationError, match="fs"):
        SignalArray(
            data=np.zeros((3, 1)),
            fs=None,
            time=np.array([0.0, 0.2, 0.5]),
            t0=None,
            clock=None,
            channels=ChannelTable([ChannelInfo("aux0", 0, "aux", "a.u.")]),
            unit="a.u.",
            name="aux",
        )


def test_signal_array_accepts_explicit_irregular_time_with_fs():
    signal = SignalArray(
        data=np.zeros((3, 1)),
        fs=10.0,
        time=np.array([0.0, 0.2, 0.5]),
        t0=None,
        clock=None,
        channels=ChannelTable([ChannelInfo("aux0", 0, "aux", "a.u.")]),
        unit="a.u.",
        name="aux",
    )

    assert signal.duration == 0.5
    assert signal.t0 == 0.0


def test_signal_array_rejects_invalid_shapes_and_time():
    with pytest.raises(ValidationError):
        SignalArray(
            data=np.zeros(3),
            fs=1.0,
            time=None,
            t0=0.0,
            clock=None,
            channels=ChannelTable.default(1),
            unit="a.u.",
            name="bad",
        )

    with pytest.raises(ValidationError):
        SignalArray(
            data=np.zeros((3, 1)),
            fs=1.0,
            time=np.array([0.0, 0.3, 0.2]),
            t0=0.0,
            clock=None,
            channels=ChannelTable.default(1),
            unit="a.u.",
            name="bad",
        )


def test_feature_matrix_generates_default_names_and_validates_time():
    features = FeatureMatrix(
        data=np.ones((5, 2)),
        fs=10.0,
        t0=None,
        source_signal="neural",
        window_size=0.1,
        shift=0.05,
    )

    assert features.feature_names == ["feature000", "feature001"]
    assert features.t0 == 0.0
    np.testing.assert_allclose(features.time, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert not hasattr(features, "times")

    with pytest.raises(ValidationError):
        FeatureMatrix(
            data=np.ones((2, 2)),
            fs=10.0,
            feature_names=["x", "x"],
        )


def test_feature_matrix_accepts_explicit_irregular_time_without_fs():
    features = FeatureMatrix(
        data=np.ones((3, 1)),
        fs=None,
        time=np.array([0.0, 0.1, 0.35]),
    )

    assert features.fs is None
    np.testing.assert_array_equal(features.time, [0.0, 0.1, 0.35])


def test_feature_matrix_rejects_irregular_time_with_declared_rate():
    with pytest.raises(ValidationError, match="use fs=None"):
        FeatureMatrix(
            data=np.ones((3, 1)),
            fs=10.0,
            time=np.array([0.0, 0.1, 0.35]),
        )


def test_feature_matrix_requires_fs_when_time_is_omitted():
    with pytest.raises(ValidationError, match="fs"):
        FeatureMatrix(data=np.ones((2, 1)), fs=None)


def test_spike_train_validates_times_and_waveform_shape():
    train = SpikeTrain(
        times=[np.array([0.1, 0.2])],
        units=["unit0"],
        channels=[0],
        waveforms=[np.zeros((2, 40, 1))],
        fs=30000.0,
    )

    assert train.units == ["unit0"]

    with pytest.raises(ValidationError):
        SpikeTrain(times=[np.array([0.2, 0.1])])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_data_models_reject_non_finite_contract_values(value):
    with pytest.raises(ValidationError):
        SignalArray(
            data=np.zeros((1, 1)),
            fs=value,
            time=None,
            t0=0.0,
            clock=None,
            channels=ChannelTable.default(1),
            unit="a.u.",
            name="invalid",
        )
    with pytest.raises(ValidationError):
        SpikeTrain(times=[np.array([value])])


def test_spike_channels_require_actual_integers():
    with pytest.raises(ValidationError, match="integer"):
        SpikeTrain(times=[np.array([0.1])], channels=[1.5])


def test_signal_unit_must_match_channel_metadata():
    with pytest.raises(ValidationError, match="must match channel units"):
        SignalArray(
            data=np.zeros((2, 2)),
            fs=1.0,
            time=None,
            t0=0.0,
            clock=None,
            channels=ChannelTable.default(2),
            unit=["uV", "mV"],
            name="signal",
        )


def test_signal_array_owns_immutable_metadata():
    data = np.zeros((3, 1))
    time = np.array([0.0, 0.1, 0.2])
    attrs = {"nested": {"labels": ["a", "b"]}}
    signal = SignalArray.from_array(
        data,
        fs=10.0,
        time=time,
        attrs=attrs,
    )

    time[1] = 99.0
    attrs["nested"]["labels"].append("c")
    np.testing.assert_array_equal(signal.time, [0.0, 0.1, 0.2])
    assert signal.attrs["nested"]["labels"] == ("a", "b")
    assert not signal.time.flags.writeable
    with pytest.raises((AttributeError, TypeError)):
        signal.name = "changed"
    with pytest.raises(TypeError):
        signal.attrs["new"] = True

    data.shape = (1, 3)
    with pytest.raises(ValidationError, match="shape changed"):
        signal.validate()


def test_signal_array_copy_policy_and_controlled_replacement():
    source = np.arange(6.0).reshape(3, 2)
    borrowed = SignalArray.from_array(source, fs=10.0)
    owned = SignalArray.from_array(source, fs=10.0, copy_data=True)
    source[0, 0] = 99.0

    assert borrowed.data[0, 0] == 99.0
    assert owned.data[0, 0] == 0.0
    replacement = borrowed.with_data(np.ones((3, 2)), copy_data=True)
    replacement.validate()
    assert replacement is not borrowed
    np.testing.assert_array_equal(replacement.data, 1.0)


def test_signal_array_metadata_is_pickleable_and_type_hints_resolve():
    signal = SignalArray.from_array(
        np.ones((2, 1)),
        fs=10.0,
        attrs={"nested": {"value": 1}},
    )

    restored = pickle.loads(pickle.dumps(signal))
    assert restored.attrs == signal.attrs
    hints = get_type_hints(SignalArray)
    assert hints["channels"] is ChannelTable


def test_signal_from_array_crop_time_semantics():
    signal = SignalArray.from_array(
        np.arange(20.0).reshape(10, 2),
        fs=10.0,
        t0=2.0,
        channel_names=["C3", "C4"],
        channel_types="eeg",
        units="uV",
        name="eeg",
    )

    assert signal.channel_names == ["C3", "C4"]
    assert [channel.type for channel in signal.channels] == ["eeg", "eeg"]
    assert signal.duration == pytest.approx(0.9)

    selected = signal.select_time(2.2, 2.6)
    cropped = signal.crop(0.2, 0.6)
    np.testing.assert_array_equal(selected.data, signal.data[2:6])
    np.testing.assert_array_equal(cropped.data, selected.data)
    assert selected.t0 == pytest.approx(2.2)


def test_signal_from_array_accepts_one_channel_and_validates_metadata():
    signal = SignalArray.from_array(
        np.arange(4.0),
        fs=100.0,
    )
    assert signal.data.shape == (4, 1)
    with pytest.raises(ValidationError, match="channel_types"):
        SignalArray.from_array(
            np.ones((4, 2)),
            fs=100.0,
            channel_types=["eeg"],
        )
