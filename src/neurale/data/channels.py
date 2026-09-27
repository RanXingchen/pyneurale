#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Channel and electrode metadata for PyNeurale data objects."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from neurale._validation import validate_integer
from neurale.exceptions import ValidationError

from ._helpers import (
    ALLOWED_CHANNEL_TYPES,
    copy_attrs,
    ensure_finite_float,
    ensure_non_empty_string,
    freeze_metadata,
    validate_unit,
)


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Describe one recorded or derived channel.

    Parameters
    ----------
    name : str
        Unique non-empty channel name.
    index : int
        Non-negative channel identifier.
    type : str
        Channel type from the supported PyNeurale channel categories.
    unit : str
        Physical or logical measurement unit.
    valid : bool, default=True
        Whether the channel contains usable data.
    bad : bool, default=False
        Whether the channel has been marked as bad.
    impedance : complex or float or None, optional
        Measured channel impedance as a finite numeric scalar. Real values are
        normalized to ``float`` and values with a non-zero imaginary component
        to ``complex``. Boolean, array, container, and non-finite values are
        rejected.
    electrode : str or None, optional
        Name of the associated electrode array.
    contact : int or None, optional
        Non-negative electrode-contact identifier.
    position : tuple of float or None, optional
        3D channel position.
    attrs : dict, optional
        Deep-frozen application metadata using the closed value domain
        documented by :mod:`neurale.data`.

    Raises
    ------
    neurale.exceptions.ValidationError
        If required values, channel type, impedance, contact, position, or
        metadata are invalid.
    """

    name: str
    index: int
    type: str
    unit: str
    valid: bool = True
    bad: bool = False
    impedance: complex | float | None = None
    electrode: str | None = None
    contact: int | None = None
    pos: tuple[float, float, float] | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = ensure_non_empty_string(self.name, "channel name")
        idx = validate_integer(self.index, "channel index", minimum=0)
        channel_type = ensure_non_empty_string(self.type, "channel type").lower()
        if channel_type not in ALLOWED_CHANNEL_TYPES:
            allowed = ", ".join(sorted(ALLOWED_CHANNEL_TYPES))
            raise ValidationError(f"channel type must be one of: {allowed}; got {channel_type!r}.")
        unit = validate_unit(self.unit, 1, "channel unit")
        impedance = _validated_impedance(self.impedance)

        contact = None
        if self.contact is not None:
            contact = validate_integer(self.contact, "contact", minimum=0)

        electrode = self.electrode
        if electrode is not None:
            electrode = ensure_non_empty_string(electrode, "electrode")
        pos = self.pos
        if pos is not None:
            if len(pos) != 3:
                raise ValidationError("position must contain three coordinates.")
            pos = tuple(ensure_finite_float(value, "position coordinate") for value in pos)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "index", idx)
        object.__setattr__(self, "type", channel_type)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "valid", bool(self.valid))
        object.__setattr__(self, "bad", bool(self.bad))
        object.__setattr__(self, "impedance", impedance)
        object.__setattr__(self, "contact", contact)
        object.__setattr__(self, "electrode", electrode)
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "attrs", freeze_metadata(self.attrs))


def _validated_impedance(value: object) -> float | complex | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValidationError("impedance must be a finite real or complex scalar.")
    arr = np.asarray(value)
    if not np.issubdtype(arr.dtype, np.number):
        raise ValidationError("impedance must be a finite real or complex scalar.")
    impedance = complex(value)
    if not np.isfinite(impedance.real) or not np.isfinite(impedance.imag):
        raise ValidationError("impedance must be finite.")
    if impedance.imag == 0.0:
        return float(impedance.real)
    return impedance


@dataclass(frozen=True, slots=True)
class ChannelTable:
    """Store an ordered collection of unique channel metadata.

    Parameters
    ----------
    channels : iterable of neurale.data.channels.ChannelInfo, optional
        Channels in data-column order.

    Raises
    ------
    neurale.exceptions.ValidationError
        If channel names or indices are duplicated.
    """

    channels: tuple[ChannelInfo, ...] = field(default_factory=tuple)

    def __init__(self, channels: Iterable[ChannelInfo] = ()) -> None:
        values = tuple(channels)
        if not all(isinstance(channel, ChannelInfo) for channel in values):
            raise ValidationError("channels must contain ChannelInfo instances.")
        object.__setattr__(self, "channels", values)
        self._validate()

    def _validate(self) -> None:
        names = [channel.name for channel in self.channels]
        if len(names) != len(set(names)):
            raise ValidationError("channel names must be unique.")

        indices = [channel.index for channel in self.channels]
        if len(indices) != len(set(indices)):
            raise ValidationError("channel indices must be unique.")

    def __len__(self) -> int:
        return len(self.channels)

    def __iter__(self) -> Iterator[ChannelInfo]:
        return iter(self.channels)

    def __getitem__(self, key: int | str | slice) -> ChannelInfo | ChannelTable:
        if isinstance(key, slice):
            return ChannelTable(self.channels[key])
        if isinstance(key, str):
            return self.channels[self.position_of(key)]
        return self.channels[key]

    @property
    def names(self) -> list[str]:
        return [channel.name for channel in self.channels]

    @property
    def indices(self) -> list[int]:
        return [channel.index for channel in self.channels]

    @property
    def units(self) -> list[str]:
        return [channel.unit for channel in self.channels]

    @property
    def valid_mask(self) -> np.ndarray:
        return np.asarray([channel.valid for channel in self.channels], dtype=bool)

    @property
    def bad_mask(self) -> np.ndarray:
        return np.asarray([channel.bad for channel in self.channels], dtype=bool)

    @property
    def good_mask(self) -> np.ndarray:
        return self.valid_mask & ~self.bad_mask

    def position_of(self, key: int | str) -> int:
        """Return the table position of a channel.

        Parameters
        ----------
        key : int or str
            Channel index or channel name.

        Returns
        -------
        int
            Zero-based position in the table.

        Raises
        ------
        KeyError
            If the channel name or index is unknown.
        """

        if isinstance(key, str):
            try:
                return self.names.index(key)
            except ValueError as exc:
                raise KeyError(f"unknown channel name: {key!r}") from exc

        for pos, channel in enumerate(self.channels):
            if channel.index == key:
                return pos
        raise KeyError(f"unknown channel index: {key!r}")

    def positions(self, keys: Sequence[int | str] | np.ndarray) -> list[int]:
        """Resolve channel selectors to table positions.

        Parameters
        ----------
        keys : sequence of int or str, or numpy.ndarray
            Channel indices, names, or a boolean mask in table order.

        Returns
        -------
        list of int
            Zero-based table positions.

        Raises
        ------
        KeyError
            If a channel name or index is unknown.
        neurale.exceptions.ValidationError
            If a boolean mask is not 1D or has the wrong length.
        """
        if isinstance(keys, np.ndarray) and keys.dtype == bool:
            if keys.ndim != 1 or keys.size != len(self.channels):
                raise ValidationError("boolean channel mask has the wrong shape.")
            return np.nonzero(keys)[0].tolist()

        return [self.position_of(key) for key in list(keys)]

    def select(self, keys: Sequence[int | str] | np.ndarray) -> ChannelTable:
        """Return a table containing selected channels.

        Parameters
        ----------
        keys : sequence of int or str, or numpy.ndarray
            Channel indices, names, or a boolean mask.

        Returns
        -------
        neurale.data.channels.ChannelTable
            New table preserving requested channel order.
        """
        return ChannelTable([self.channels[pos] for pos in self.positions(keys)])

    def copy(self) -> ChannelTable:
        """Return a copy of the channel table.

        Returns
        -------
        neurale.data.channels.ChannelTable
            New table with copied channel objects and attribute mappings.
        """
        return ChannelTable(
            [replace(channel, attrs=dict(channel.attrs)) for channel in self.channels]
        )

    @classmethod
    def from_names(
        cls,
        names: Sequence[str],
        *,
        type: str = "aux",
        unit: str | Sequence[str] = "a.u.",
        start_idx: int = 0,
    ) -> ChannelTable:
        """Construct a channel table from names.

        Parameters
        ----------
        names : sequence of str
            Channel names in table order.
        type : str, default="aux"
            Channel type assigned to every entry.
        unit : str, default="a.u."
            Unit assigned to every entry.
        start_index : int, default=0
            Index assigned to the first channel.

        Returns
        -------
        neurale.data.channels.ChannelTable
            Constructed table.

        Raises
        ------
        neurale.exceptions.ValidationError
            If generated channel metadata are invalid or names are duplicated.
        """
        units = [unit] * len(names) if isinstance(unit, str) else list(unit)
        if len(units) != len(names):
            raise ValidationError("unit length must match channel names length.")
        return cls(
            [
                ChannelInfo(
                    name=name,
                    index=start_idx + idx,
                    type=type,
                    unit=units[idx],
                )
                for idx, name in enumerate(names)
            ]
        )

    @classmethod
    def default(
        cls,
        n_channels: int,
        *,
        prefix: str = "ch",
        type: str = "aux",
        unit: str = "a.u.",
    ) -> ChannelTable:
        """Construct a table with stable generated channel names.

        Parameters
        ----------
        n_channels : int
            Number of channels to generate.
        prefix : str, default="ch"
            Prefix used for generated names.
        type : str, default="aux"
            Channel type assigned to every entry.
        unit : str or sequence of str, default="a.u."
            Shared unit or one unit per generated entry.

        Returns
        -------
        neurale.data.channels.ChannelTable
            Table with names such as ``ch000`` and ``ch001``.

        Raises
        ------
        neurale.exceptions.ValidationError
            If ``n_channels`` is negative or generated metadata are invalid.
        """
        if n_channels < 0:
            raise ValidationError("n_channels must be non-negative.")
        width = max(3, len(str(max(n_channels - 1, 0))))
        names = [f"{prefix}{idx:0{width}d}" for idx in range(n_channels)]
        return cls.from_names(names, type=type, unit=unit)


@dataclass(slots=True)
class ElectrodeArray:
    """Describe electrode or contact-array topology.

    Parameters
    ----------
    name : str
        Non-empty electrode-array name.
    type : str
        Electrode-array type.
    contacts : list of int
        Unique non-negative contact identifiers.
    channels : list of int
        Unique non-negative associated channel identifiers.
    shape : tuple of int or None, optional
        Positive ``(rows, columns)`` layout shape.
    map_channel_to_contact : dict of int to int, optional
        Mapping from channel identifiers to contact identifiers.
    contact_mask : numpy.ndarray or None, optional
        One- or two-dimensional boolean mask of active contacts.
    spacing : tuple of float or None, optional
        Two-dimensional contact spacing.
    spacing_unit : str or None, optional
        Unit used for ``spacing``.
    attrs : dict, optional
        Additional application-specific metadata.

    Raises
    ------
    neurale.exceptions.ValidationError
        If identifiers, shape, mapping, mask, spacing, or units are invalid.
    """

    name: str
    type: str
    contacts: list[int]
    channels: list[int]
    shape: tuple[int, int] | None = None
    map_channel_to_contact: dict[int, int] = field(default_factory=dict)
    contact_mask: np.ndarray | None = None
    spacing: tuple[float, float] | None = None
    spacing_unit: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = ensure_non_empty_string(self.name, "electrode name")
        self.type = ensure_non_empty_string(self.type, "electrode type")
        self.contacts = [validate_integer(item, "contact", minimum=0) for item in self.contacts]
        self.channels = [validate_integer(item, "channel", minimum=0) for item in self.channels]
        if len(self.contacts) != len(set(self.contacts)):
            raise ValidationError("electrode contacts must be unique.")
        if len(self.channels) != len(set(self.channels)):
            raise ValidationError("electrode channels must be unique.")
        if self.shape is not None:
            if len(self.shape) != 2 or self.shape[0] <= 0 or self.shape[1] <= 0:
                raise ValidationError("electrode shape must be a positive (rows, cols) tuple.")
            self.shape = (int(self.shape[0]), int(self.shape[1]))
        self.map_channel_to_contact = {
            validate_integer(channel, "map channel", minimum=0): validate_integer(
                contact, "map contact", minimum=0
            )
            for channel, contact in self.map_channel_to_contact.items()
        }
        if self.contact_mask is not None:
            mask = np.asarray(self.contact_mask, dtype=bool)
            if mask.ndim not in (1, 2):
                raise ValidationError("contact_mask must be a 1D or 2D boolean array.")
            self.contact_mask = mask
        if self.spacing is not None:
            if len(self.spacing) != 2:
                raise ValidationError("spacing must contain two values.")
            self.spacing = (
                ensure_finite_float(self.spacing[0], "spacing"),
                ensure_finite_float(self.spacing[1], "spacing"),
            )
        if self.spacing_unit is not None:
            self.spacing_unit = ensure_non_empty_string(self.spacing_unit, "spacing_unit")
        self.attrs = copy_attrs(self.attrs)
