#!/usr/bin/env python3

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neurale.data import ChannelInfo, ChannelTable, ElectrodeArray
from neurale.exceptions import ValidationError


def test_channel_table_selects_by_name_index_and_mask():
    channels = ChannelTable(
        [
            ChannelInfo("A-000", 0, "ecog", "uV"),
            ChannelInfo("A-001", 1, "ecog", "uV", bad=True),
            ChannelInfo("A-002", 2, "ecog", "uV", valid=False),
        ]
    )

    assert channels.names == ["A-000", "A-001", "A-002"]
    assert channels.position_of("A-001") == 1
    assert channels.position_of(2) == 2
    assert channels.select(["A-002", "A-000"]).names == ["A-002", "A-000"]
    assert channels.select(np.array([True, False, True])).names == ["A-000", "A-002"]
    assert channels.good_mask.tolist() == [True, False, False]


def test_channel_table_rejects_duplicate_names_and_indices():
    with pytest.raises(ValidationError):
        ChannelTable(
            [
                ChannelInfo("A-000", 0, "ecog", "uV"),
                ChannelInfo("A-000", 1, "ecog", "uV"),
            ]
        )

    with pytest.raises(ValidationError):
        ChannelTable(
            [
                ChannelInfo("A-000", 0, "ecog", "uV"),
                ChannelInfo("A-001", 0, "ecog", "uV"),
            ]
        )


def test_channel_info_rejects_unknown_channel_type():
    with pytest.raises(ValidationError):
        ChannelInfo("A-000", 0, "unknown", "uV")


@pytest.mark.parametrize(
    "impedance",
    [
        [],
        {},
        np.array([1.0]),
        bytearray(b"x"),
        True,
        np.bool_(False),
        "1.0",
    ],
)
def test_channel_info_rejects_non_scalar_or_non_numeric_impedance(impedance):
    with pytest.raises(ValidationError, match="finite real or complex scalar"):
        ChannelInfo("A-000", 0, "ecog", "uV", impedance=impedance)


@pytest.mark.parametrize(
    "impedance",
    [
        np.inf,
        -np.inf,
        np.nan,
        complex(np.nan, 1.0),
        complex(1.0, np.inf),
    ],
)
def test_channel_info_rejects_non_finite_impedance(impedance):
    with pytest.raises(ValidationError, match="impedance must be finite"):
        ChannelInfo("A-000", 0, "ecog", "uV", impedance=impedance)


def test_channel_info_normalizes_finite_scalar_impedance() -> None:
    real = ChannelInfo("A-000", 0, "ecog", "uV", impedance=np.float32(1.5))
    real_complex = ChannelInfo("A-001", 1, "ecog", "uV", impedance=2.0 + 0.0j)
    complex_value = ChannelInfo("A-002", 2, "ecog", "uV", impedance=1.5 + 2.0j)

    assert real.impedance == 1.5
    assert type(real.impedance) is float
    assert real_complex.impedance == 2.0
    assert type(real_complex.impedance) is float
    assert complex_value.impedance == 1.5 + 2.0j
    assert type(complex_value.impedance) is complex


def test_default_channel_table_names_are_stable():
    channels = ChannelTable.default(3, prefix="A-", type="ecog", unit="uV")

    assert channels.names == ["A-000", "A-001", "A-002"]
    assert channels.units == ["uV", "uV", "uV"]


def test_channel_table_from_names_accepts_per_channel_units():
    channels = ChannelTable.from_names(
        ["C3", "C4"],
        type="eeg",
        unit=["uV", "mV"],
    )

    assert channels.units == ["uV", "mV"]
    with pytest.raises(ValidationError, match="unit length"):
        ChannelTable.from_names(["C3", "C4"], unit=["uV"])


def test_electrode_array_validates_contact_mask():
    electrode = ElectrodeArray(
        name="NXEcog-test",
        type="ecog_grid",
        contacts=[0, 1],
        channels=[10, 11],
        map_channel_to_contact={10: 0, 11: 1},
        contact_mask=np.array([[True, False]]),
        spacing=(1.5, 0.8),
        spacing_unit="mm",
    )

    assert electrode.contact_mask.shape == (1, 2)
    assert electrode.map_channel_to_contact[11] == 1


def test_channel_identity_cannot_be_mutated_after_table_validation():
    channels = ChannelTable.default(1)

    with pytest.raises(FrozenInstanceError):
        channels[0].name = "duplicate"
    with pytest.raises(AttributeError):
        channels.channels.append(channels[0])
