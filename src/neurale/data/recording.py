#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Top-level neural recording container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neurale.exceptions import ValidationError

from ._helpers import ValidatedNamedDict, copy_attrs
from .arrays import FeatureMatrix, SignalArray, SpikeTrain
from .channels import ElectrodeArray
from .events import EventSeries, TrialTable


@dataclass(slots=True)
class Recording:
    """Collect signals, events, trials, features, spikes, and metadata.

    Parameters
    ----------
    signals : dict of str to neurale.data.arrays.SignalArray, optional
        Named continuous signals.
    events : neurale.data.events.EventSeries or None, optional
        Recording-wide events.
    trials : neurale.data.events.TrialTable or None, optional
        Task-level trials.
    features : dict of str to neurale.data.arrays.FeatureMatrix, optional
        Named feature matrices.
    spikes : dict of str to neurale.data.arrays.SpikeTrain, optional
        Named spike-train collections.
    metadata : dict, optional
        Recording-level metadata.
    subject : dict or None, optional
        Subject metadata.
    session : dict or None, optional
        Session metadata.
    devices : dict of str to dict, optional
        Device metadata keyed by device name.
    electrodes : dict of str to neurale.data.channels.ElectrodeArray, optional
        Electrode topology metadata keyed by array name.

    Raises
    ------
    neurale.exceptions.ValidationError
        If mapping names are empty or values have incompatible types.
    """

    signals: dict[str, SignalArray] = field(default_factory=dict)
    events: EventSeries | None = None
    trials: TrialTable | None = None
    features: dict[str, FeatureMatrix] = field(default_factory=dict)
    spikes: dict[str, SpikeTrain] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    subject: dict[str, Any] | None = None
    session: dict[str, Any] | None = None
    devices: dict[str, dict[str, Any]] = field(default_factory=dict)
    electrodes: dict[str, ElectrodeArray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.signals = ValidatedNamedDict(self.signals, value_type=SignalArray, label="signal")
        self.features = ValidatedNamedDict(self.features, value_type=FeatureMatrix, label="feature")
        self.spikes = ValidatedNamedDict(self.spikes, value_type=SpikeTrain, label="spike train")
        self.devices = dict(self.devices)
        self.electrodes = ValidatedNamedDict(
            self.electrodes,
            value_type=ElectrodeArray,
            label="electrode",
        )
        self.metadata = copy_attrs(self.metadata)
        self.subject = None if self.subject is None else dict(self.subject)
        self.session = None if self.session is None else dict(self.session)

    def signal(self, name: str = "neural") -> SignalArray:
        """Return a named signal.

        Parameters
        ----------
        name : str, default="neural"
            Signal name.

        Returns
        -------
        neurale.data.arrays.SignalArray
            Requested signal.

        Raises
        ------
        KeyError
            If ``name`` is not present in the recording.
        """
        try:
            return self.signals[name]
        except KeyError as exc:
            raise KeyError(f"unknown signal: {name!r}") from exc

    def add_signal(self, name: str, signal: SignalArray) -> None:
        """Add or replace a named signal.

        Parameters
        ----------
        name : str
            Non-empty signal name.
        signal : neurale.data.arrays.SignalArray
            Signal to store.

        Raises
        ------
        neurale.exceptions.ValidationError
            If ``name`` is empty or ``signal`` has the wrong type.
        """
        if not name:
            raise ValidationError("signal name must be a non-empty string.")
        if not isinstance(signal, SignalArray):
            raise ValidationError("signal must be a SignalArray.")
        self.signals[name] = signal

    @classmethod
    def from_arrays(
        cls,
        data: np.ndarray,
        *,
        fs: float,
        time: np.ndarray | None = None,
        t0: float | None = 0.0,
        channel_names: list[str] | None = None,
        channel_type: str = "aux",
        unit: str | list[str] = "a.u.",
        signal_name: str = "neural",
        attrs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        subject: dict[str, Any] | None = None,
        session: dict[str, Any] | None = None,
    ) -> Recording:
        """Construct a recording from one sample-by-channel array.

        Parameters
        ----------
        data : numpy.ndarray
            Two-dimensional array with shape ``(n_samples, n_channels)``.
        fs : float
            Sampling rate in hertz.
        time : numpy.ndarray or None, optional
            Optional timestamp vector with one value per sample.
        t0 : float or None, default=0.0
            Time of the first sample when timestamps are derived.
        channel_names : list of str or None, optional
            Channel names. Stable names are generated when omitted.
        channel_type : str, default="aux"
            Type assigned to all generated channel metadata.
        unit : str or list of str, default="a.u."
            Shared unit or one unit per channel.
        signal_name : str, default="neural"
            Name used for the generated signal.
        attrs : dict or None, optional
            Metadata attached to the generated signal.
        metadata : dict or None, optional
            Recording-level metadata.
        subject : dict or None, optional
            Subject metadata.
        session : dict or None, optional
            Session metadata.

        Returns
        -------
        Recording
            Recording containing the generated signal and channel table.

        Raises
        ------
        neurale.exceptions.ValidationError
            If data shape, channel names, units, or timing metadata are invalid.
        """
        if np.asarray(data).ndim != 2:
            raise ValidationError("data must be a 2D array.")
        signal = SignalArray.from_array(
            data,
            fs=fs,
            time=time,
            t0=t0,
            channel_names=channel_names,
            channel_types=channel_type,
            units=unit,
            name=signal_name,
            attrs={} if attrs is None else attrs,
        )
        return cls(
            signals={signal_name: signal},
            metadata={} if metadata is None else metadata,
            subject=subject,
            session=session,
        )
