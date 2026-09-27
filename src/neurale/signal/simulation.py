#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Native sampled-signal and intent-driven neural-signal generation.

All generated arrays use ``(n_samples, n_channels)`` order and ``numpy.float64``.
``SignalGenerator`` is stateless. ``NeuralSignalGenerator`` keeps physiological
surrogate state and advances its contiguous sample position internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, final

import numpy as np

from neurale._native_loader import load_native_namespace
from neurale._validation import validate_integer, validate_number, validate_positive_float
from neurale.exceptions import ValidationError

_MAX_SAMPLE_IDX = 2**64 - 1
_DTYPE = np.dtype(np.float64)


def _native_simulation() -> Any:
    return load_native_namespace("signal.simulation")


def _native_generator(generator: SignalGenerator) -> object:
    """Return the prepared native generator for an internal native consumer."""
    if not isinstance(generator, SignalGenerator):
        raise ValidationError("generator must be a SignalGenerator.")
    return generator._native


def _dtype(value: np.dtype[Any] | type[Any] | str) -> np.dtype[np.float64]:
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("dtype must be numpy.float64.") from exc
    if dtype != _DTYPE:
        raise ValidationError("dtype must be numpy.float64.")
    return _DTYPE


def _channels(value: int) -> int:
    return validate_integer(value, "n_channels", minimum=1)


def _rate(value: float) -> float:
    return validate_positive_float(value, "fs")


def _finite_array(value: object, name: str) -> np.ndarray:
    try:
        source = np.asarray(value)
        if np.iscomplexobj(source) or not np.issubdtype(source.dtype, np.number):
            raise TypeError
        arr = np.asarray(source, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain finite real values.") from exc
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must contain finite real values.")
    return arr if arr.ndim == 0 else np.ascontiguousarray(arr)


def _channel_values(value: object, name: str, n_channels: int) -> np.ndarray:
    arr = _finite_array(value, name)
    if arr.ndim == 0:
        return np.full(n_channels, float(arr), dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != n_channels:
        raise ValidationError(f"{name} must be scalar or have shape ({n_channels},).")
    return arr


def _tone_frequencies(value: object, n_channels: int) -> np.ndarray:
    arr = _finite_array(value, "frequencies")
    if arr.ndim == 0:
        return np.full((1, n_channels), float(arr), dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] == 0:
            raise ValidationError("frequencies must contain at least one tone.")
        return np.repeat(arr[:, None], n_channels, axis=1)
    if arr.ndim != 2 or arr.shape[1] != n_channels or arr.shape[0] == 0:
        raise ValidationError(
            f"frequencies must be scalar, 1D tones, or have shape (n_tones, {n_channels})."
        )
    return arr


def _tone_parameter(
    value: object,
    name: str,
    n_tones: int,
    n_channels: int,
) -> np.ndarray:
    arr = _finite_array(value, name)
    if arr.ndim == 0:
        return np.full((n_tones, n_channels), float(arr), dtype=np.float64)
    if arr.ndim == 1 and arr.shape[0] == n_tones:
        return np.repeat(arr[:, None], n_channels, axis=1)
    if arr.ndim != 2 or arr.shape != (n_tones, n_channels):
        raise ValidationError(
            f"{name} must be scalar, have shape ({n_tones},), or have shape "
            f"({n_tones}, {n_channels})."
        )
    return arr


@final
class SignalGenerator:
    """Stateless deterministic native sampled-signal generator.

    Construct generators with the named class methods. ``generate_into``
    exposes the caller-buffer native generation path; ``generate`` is an
    allocating convenience for offline use and tests. Strict realtime simulated
    acquisition consumes the prepared native generator directly.
    """

    __slots__ = ("_finite_length", "_fs", "_n_channels", "_native")

    def __init__(self) -> None:
        raise TypeError("use a named SignalGenerator constructor")

    @classmethod
    def _create(
        cls,
        native: object,
        n_channels: int,
        fs: float,
        finite_length: int | None = None,
    ) -> SignalGenerator:
        generator = object.__new__(cls)
        generator._native = native
        generator._n_channels = n_channels
        generator._fs = fs
        generator._finite_length = finite_length
        return generator

    @classmethod
    def zeros(
        cls,
        n_channels: int,
        fs: float,
        *,
        dtype: np.dtype[Any] | type[Any] | str = np.float64,
    ) -> SignalGenerator:
        """Create a zero-valued generator."""
        n_channels = _channels(n_channels)
        fs = _rate(fs)
        _dtype(dtype)
        native = _native_simulation().SignalGenerator.zeros(n_channels, fs)
        return cls._create(native, n_channels, fs)

    @classmethod
    def constant(
        cls,
        n_channels: int,
        fs: float,
        value: object,
        *,
        dtype: np.dtype[Any] | type[Any] | str = np.float64,
    ) -> SignalGenerator:
        """Create a generator with one constant value per channel."""
        n_channels = _channels(n_channels)
        fs = _rate(fs)
        _dtype(dtype)
        values = _channel_values(value, "value", n_channels)
        native = _native_simulation().SignalGenerator.constant(n_channels, fs, values)
        return cls._create(native, n_channels, fs)

    @classmethod
    def sine(
        cls,
        n_channels: int,
        fs: float,
        freq: object,
        *,
        amp: object = 1.0,
        phase: object = 0.0,
        dtype: np.dtype[Any] | type[Any] | str = np.float64,
    ) -> SignalGenerator:
        """Create one sine tone per channel.

        ``frequency``, ``amplitude``, and ``phase`` may be scalar or contain
        one value per channel. Phase is expressed in radians.
        """
        n_channels = _channels(n_channels)
        fs = _rate(fs)
        _dtype(dtype)
        freqs = _channel_values(freq, "frequency", n_channels)[None, :]
        amps = _channel_values(amp, "amplitude", n_channels)[None, :]
        phases = _channel_values(phase, "phase", n_channels)[None, :]
        return cls._from_tones(n_channels, fs, freqs, amps, phases)

    @classmethod
    def tones(
        cls,
        n_channels: int,
        fs: float,
        freqs: object,
        *,
        amps: object = 1.0,
        phases: object = 0.0,
        dtype: np.dtype[Any] | type[Any] | str = np.float64,
    ) -> SignalGenerator:
        """Create a sum of known tones.

        One-dimensional frequencies are tones shared by all channels. A
        2D frequency array uses ``(tone, channel)`` order.
        Amplitudes and phases may be scalar, one value per tone, or the same
        2D shape.
        """
        n_channels = _channels(n_channels)
        fs = _rate(fs)
        _dtype(dtype)
        freq_matrix = _tone_frequencies(freqs, n_channels)
        n_tones = freq_matrix.shape[0]
        amp_matrix = _tone_parameter(amps, "amps", n_tones, n_channels)
        phase_matrix = _tone_parameter(phases, "phases", n_tones, n_channels)
        return cls._from_tones(
            n_channels,
            fs,
            freq_matrix,
            amp_matrix,
            phase_matrix,
        )

    @classmethod
    def _from_tones(
        cls,
        n_channels: int,
        fs: float,
        freqs: np.ndarray,
        amps: np.ndarray,
        phases: np.ndarray,
    ) -> SignalGenerator:
        if np.any(freqs < 0.0) or np.any(freqs > fs / 2.0):
            raise ValidationError("frequencies must be in [0, fs / 2].")
        native = _native_simulation().SignalGenerator.tones(
            n_channels,
            fs,
            freqs.shape[0],
            freqs,
            amps,
            phases,
        )
        return cls._create(native, n_channels, fs)

    @classmethod
    def noise(
        cls,
        n_channels: int,
        fs: float,
        *,
        seed: int,
        low: float = -1.0,
        high: float = 1.0,
        dtype: np.dtype[Any] | type[Any] | str = np.float64,
    ) -> SignalGenerator:
        """Create deterministic counter-based uniform noise in ``[low, high)``."""
        n_channels = _channels(n_channels)
        fs = _rate(fs)
        _dtype(dtype)
        seed = int(
            validate_number(
                seed,
                "seed",
                kind="integer",
                minimum=0,
                maximum=_MAX_SAMPLE_IDX,
            )
        )
        low = float(validate_number(low, "low", kind="real", coerce=True))
        high = float(validate_number(high, "high", kind="real", coerce=True))
        if not low < high or not np.isfinite(high - low):
            raise ValidationError("low and high must define a finite positive span.")
        native = _native_simulation().SignalGenerator.noise(n_channels, fs, seed, low, high)
        return cls._create(native, n_channels, fs)

    @classmethod
    def samples(
        cls,
        n_channels: int,
        fs: float,
        x: np.ndarray,
        *,
        repeat: bool = False,
        dtype: np.dtype[Any] | type[Any] | str = np.float64,
    ) -> SignalGenerator:
        """Create a finite or repeating generator from sample-major values."""
        n_channels = _channels(n_channels)
        fs = _rate(fs)
        _dtype(dtype)
        if not isinstance(repeat, bool):
            raise ValidationError("repeat must be a bool.")
        if not isinstance(x, np.ndarray) or x.dtype != _DTYPE:
            raise ValidationError("x must be a numpy.ndarray with dtype float64.")
        if x.ndim == 1:
            if n_channels != 1:
                raise ValidationError("1D x requires n_channels=1.")
            x = x[:, None]
        if x.ndim != 2 or x.shape[1] != n_channels or x.shape[0] == 0:
            raise ValidationError(f"x must have non-empty shape (n_samples, {n_channels}).")
        if not np.all(np.isfinite(x)):
            raise ValidationError("x must contain finite real values.")
        x = np.ascontiguousarray(x)
        native = _native_simulation().SignalGenerator.samples(n_channels, fs, x.shape[0], x, repeat)
        finite_length = None if repeat else x.shape[0]
        return cls._create(native, n_channels, fs, finite_length)

    @property
    def n_channels(self) -> int:
        """Number of output channels."""
        return self._n_channels

    @property
    def sample_rate(self) -> float:
        """Sampling rate in hertz."""
        return self._fs

    @property
    def dtype(self) -> np.dtype[np.float64]:
        """Native output dtype (currently always ``numpy.float64``)."""
        return _DTYPE

    def generate(self, start: int, count: int) -> np.ndarray:
        """Allocate and return samples for ``[start, start + count)``."""
        start, count = self._range(start, count)
        output = np.empty((count, self.n_channels), dtype=np.float64)
        self._generate_into(start, output)
        return output

    def generate_into(self, output: np.ndarray, *, start: int) -> None:
        """Fill a prepared sample-major output array without allocating natively."""
        if not isinstance(output, np.ndarray):
            raise ValidationError("output must be a numpy.ndarray.")
        if output.dtype != _DTYPE:
            raise ValidationError("output dtype must be float64.")
        if output.ndim != 2 or output.shape[1] != self.n_channels:
            raise ValidationError(f"output must have shape (n_samples, {self.n_channels}).")
        if not output.flags.c_contiguous:
            raise ValidationError("output must be C-contiguous.")
        if not output.flags.writeable:
            raise ValidationError("output must be writable.")
        start, _ = self._range(start, output.shape[0])
        self._generate_into(start, output)

    def _generate_into(self, start: int, output: np.ndarray) -> None:
        try:
            self._native.generate_into(start, output)
        except (ValueError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc

    def _range(self, start: int, count: int) -> tuple[int, int]:
        start = int(
            validate_number(
                start,
                "start",
                kind="integer",
                minimum=0,
                maximum=_MAX_SAMPLE_IDX,
            )
        )
        count = validate_integer(count, "count", minimum=0)
        if count and count - 1 > _MAX_SAMPLE_IDX - start:
            raise ValidationError("requested absolute sample range overflows uint64.")
        if self._finite_length is not None and start + count > self._finite_length:
            raise ValidationError("requested range exceeds finite supplied samples.")
        return start, count


@dataclass(frozen=True, slots=True)
class NeuralPopulationConfig:
    """Hidden spiking-population parameters."""

    units_per_channel: int = 4
    baseline_rate_hz: float = 12.0
    intent_rate_gain_hz: float = 30.0
    max_rate_hz: float = 120.0
    refractory_seconds: float = 0.001
    preferred_direction_jitter_radians: float = 0.08


@dataclass(frozen=True, slots=True)
class SpikeWaveformConfig:
    """Biphasic waveform and exponential spread along linear channel order."""

    amplitude_volts: float = 80e-6
    duration_seconds: float = 0.0025
    spatial_decay_channels: float = 0.75


@dataclass(frozen=True, slots=True)
class LfpConfig:
    """Damped stochastic narrow-band LFP surrogate parameters."""

    frequency_hz: float = 10.0
    damping_time_seconds: float = 0.25
    std_volts: float = 15e-6
    spatial_correlation: float = 0.6


@dataclass(frozen=True, slots=True)
class BackgroundNoiseConfig:
    """Temporally colored and spatially correlated background noise."""

    std_volts: float = 5e-6
    time_constant_seconds: float = 0.004
    spatial_correlation: float = 0.25


@dataclass(frozen=True, slots=True)
class NonstationarityConfig:
    """Slow correlated multiplicative firing-rate modulation."""

    std_fraction: float = 0.15
    time_constant_seconds: float = 30.0
    rate_correlation: float = 0.15


@dataclass(frozen=True, slots=True)
class NeuralDriftConfig:
    """Endpoint change applied as drift progress moves from zero to one.

    Extremely narrow finite per-unit limits relative to the requested standard
    deviation are rejected instead of being approximated by clipping samples.
    """

    rotation_degrees: float = 45.0
    tuning_gain_scale: float = 1.0
    baseline_rate_shift_hz: float = 0.0
    per_unit_rotation_std_degrees: float = 0.0
    per_unit_rotation_limit_degrees: float | None = None


@dataclass(frozen=True, slots=True)
class NeuralSignalConfig:
    """Complete configuration for the neural signal surrogate."""

    n_channels: int = 16
    fs: float = 30_000.0
    seed: int = 1
    population: NeuralPopulationConfig = field(default_factory=NeuralPopulationConfig)
    waveform: SpikeWaveformConfig = field(default_factory=SpikeWaveformConfig)
    lfp: LfpConfig = field(default_factory=LfpConfig)
    noise: BackgroundNoiseConfig = field(default_factory=BackgroundNoiseConfig)
    nonstationarity: NonstationarityConfig = field(default_factory=NonstationarityConfig)
    drift: NeuralDriftConfig = field(default_factory=NeuralDriftConfig)


@dataclass(frozen=True, slots=True)
class NeuralSignalBlock:
    """Continuous signal and the hidden spike raster that produced it."""

    signal: np.ndarray
    spikes: np.ndarray


def _intent(value: Sequence[float] | np.ndarray) -> tuple[float, float]:
    arr = _finite_array(value, "intent")
    if arr.shape != (2,):
        raise ValidationError("intent must have shape (2,).")
    return float(arr[0]), float(arr[1])


def _native_neural_config(config: NeuralSignalConfig) -> object:
    if not isinstance(config, NeuralSignalConfig):
        raise ValidationError("config must be a NeuralSignalConfig.")
    nested = (
        (config.population, NeuralPopulationConfig, "population"),
        (config.waveform, SpikeWaveformConfig, "waveform"),
        (config.lfp, LfpConfig, "lfp"),
        (config.noise, BackgroundNoiseConfig, "noise"),
        (config.nonstationarity, NonstationarityConfig, "nonstationarity"),
        (config.drift, NeuralDriftConfig, "drift"),
    )
    for value, expected, name in nested:
        if not isinstance(value, expected):
            raise ValidationError(f"config.{name} must be a {expected.__name__}.")
    native = _native_simulation().NeuralSignalConfig()
    native.n_channels = config.n_channels
    native.fs = config.fs
    native.seed = config.seed
    native.units_per_channel = config.population.units_per_channel
    native.baseline_rate_hz = config.population.baseline_rate_hz
    native.intent_rate_gain_hz = config.population.intent_rate_gain_hz
    native.max_rate_hz = config.population.max_rate_hz
    native.refractory_seconds = config.population.refractory_seconds
    native.preferred_direction_jitter_radians = config.population.preferred_direction_jitter_radians
    native.spike_amplitude_volts = config.waveform.amplitude_volts
    native.spike_duration_seconds = config.waveform.duration_seconds
    native.spike_spatial_decay_channels = config.waveform.spatial_decay_channels
    native.lfp_frequency_hz = config.lfp.frequency_hz
    native.lfp_damping_time_seconds = config.lfp.damping_time_seconds
    native.lfp_std_volts = config.lfp.std_volts
    native.lfp_spatial_correlation = config.lfp.spatial_correlation
    native.background_noise_std_volts = config.noise.std_volts
    native.background_time_constant_seconds = config.noise.time_constant_seconds
    native.background_spatial_correlation = config.noise.spatial_correlation
    native.nonstationarity_std_fraction = config.nonstationarity.std_fraction
    native.nonstationarity_time_constant_seconds = config.nonstationarity.time_constant_seconds
    native.rate_correlation = config.nonstationarity.rate_correlation
    native.drift_rotation_degrees = config.drift.rotation_degrees
    native.drift_per_unit_rotation_std_degrees = config.drift.per_unit_rotation_std_degrees
    native.drift_per_unit_rotation_limit_degrees = (
        float("inf")
        if config.drift.per_unit_rotation_limit_degrees is None
        else config.drift.per_unit_rotation_limit_degrees
    )
    native.drift_tuning_gain_scale = config.drift.tuning_gain_scale
    native.drift_baseline_rate_shift_hz = config.drift.baseline_rate_shift_hz
    return native


@final
class NeuralSignalGenerator:
    """Stateful native generator for continuous signals with hidden spikes.

    The generated voltage combines intent-tuned spike waveforms, colored
    spatially correlated background noise, and a damped stochastic LFP
    surrogate. The model captures selected signal statistics; it is not a
    biophysical tissue or forward model. Each call accepts a finite two-value
    intent and a ``drift_progress`` value in ``[0, 1]``. The generator advances
    its absolute sample index internally.
    """

    __slots__ = ("_config", "_native")

    def __init__(self, config: NeuralSignalConfig | None = None) -> None:
        if config is None:
            config = NeuralSignalConfig()
        try:
            native_config = _native_neural_config(config)
            self._native = _native_simulation().NeuralSignalGenerator(native_config)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc
        self._config = config

    @property
    def config(self) -> NeuralSignalConfig:
        return self._config

    @property
    def n_channels(self) -> int:
        return int(self._native.n_channels)

    @property
    def n_units(self) -> int:
        return int(self._native.n_units)

    @property
    def sample_rate(self) -> float:
        return float(self._native.sample_rate)

    @property
    def sample_index(self) -> int:
        """Absolute index that will be assigned to the next generated sample."""
        return int(self._native.sample_index)

    @property
    def drift_fingerprint(self) -> int:
        """Deterministic identity of the prepared per-unit drift endpoints."""
        return int(self._native.drift_fingerprint)

    def generate(
        self,
        count: int,
        intent: Sequence[float] | np.ndarray,
        *,
        drift_progress: float = 0.0,
    ) -> np.ndarray:
        """Allocate and return one contiguous sample-major voltage block."""
        count = self._count(count)
        output = np.empty((count, self.n_channels), dtype=np.float64)
        self._generate_into(intent, drift_progress, output, None)
        return output

    def generate_with_truth(
        self,
        count: int,
        intent: Sequence[float] | np.ndarray,
        *,
        drift_progress: float = 0.0,
    ) -> NeuralSignalBlock:
        """Return voltage plus the hidden unit spike raster used to produce it."""
        count = self._count(count)
        output = np.empty((count, self.n_channels), dtype=np.float64)
        spikes = np.empty((count, self.n_units), dtype=np.uint8)
        self._generate_into(intent, drift_progress, output, spikes)
        return NeuralSignalBlock(output, spikes)

    def generate_into(
        self,
        output: np.ndarray,
        *,
        intent: Sequence[float] | np.ndarray,
        drift_progress: float = 0.0,
        spike_output: np.ndarray | None = None,
    ) -> None:
        """Fill prepared signal and optional truth buffers without native allocation."""
        if not isinstance(output, np.ndarray) or output.dtype != _DTYPE:
            raise ValidationError("output must be a numpy.ndarray with dtype float64.")
        if output.ndim != 2 or output.shape[1] != self.n_channels:
            raise ValidationError(f"output must have shape (n_samples, {self.n_channels}).")
        if not output.flags.c_contiguous or not output.flags.writeable:
            raise ValidationError("output must be writable and C-contiguous.")
        if spike_output is not None:
            if not isinstance(spike_output, np.ndarray) or spike_output.dtype != np.uint8:
                raise ValidationError("spike_output must be a numpy.ndarray with dtype uint8.")
            if spike_output.shape != (output.shape[0], self.n_units):
                raise ValidationError(
                    f"spike_output must have shape ({output.shape[0]}, {self.n_units})."
                )
            if not spike_output.flags.c_contiguous or not spike_output.flags.writeable:
                raise ValidationError("spike_output must be writable and C-contiguous.")
        self._count(output.shape[0])
        self._generate_into(intent, drift_progress, output, spike_output)

    def _generate_into(
        self,
        intent: Sequence[float] | np.ndarray,
        drift_progress: float,
        output: np.ndarray,
        spike_output: np.ndarray | None,
    ) -> None:
        intent_x, intent_y = _intent(intent)
        drift_progress = float(
            validate_number(
                drift_progress,
                "drift_progress",
                kind="real",
                minimum=0.0,
                maximum=1.0,
                coerce=True,
            )
        )
        try:
            self._native.generate_into(
                intent_x,
                intent_y,
                drift_progress,
                output,
                spike_output,
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValidationError(str(exc)) from exc

    def _count(self, count: int) -> int:
        count = validate_integer(count, "count", minimum=0)
        if count > _MAX_SAMPLE_IDX - self.sample_index:
            raise ValidationError("requested absolute sample range overflows uint64.")
        return count

    def reset(self) -> None:
        """Restore the deterministic initial state and sample index zero."""
        self._native.reset()


__all__ = [
    "BackgroundNoiseConfig",
    "LfpConfig",
    "NeuralDriftConfig",
    "NeuralPopulationConfig",
    "NeuralSignalBlock",
    "NeuralSignalConfig",
    "NeuralSignalGenerator",
    "NonstationarityConfig",
    "SignalGenerator",
    "SpikeWaveformConfig",
]
