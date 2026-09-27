# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import neurale.signal.simulation as simulation
from neurale.exceptions import ValidationError
from neurale.signal.simulation import (
    BackgroundNoiseConfig,
    LfpConfig,
    NeuralDriftConfig,
    NeuralPopulationConfig,
    NeuralSignalConfig,
    NeuralSignalGenerator,
    NonstationarityConfig,
    SpikeWaveformConfig,
)


def test_drift_config_preserves_original_positional_arguments() -> None:
    config = NeuralDriftConfig(45.0, 0.5, 2.0)
    assert config == NeuralDriftConfig(
        rotation_degrees=45.0, tuning_gain_scale=0.5, baseline_rate_shift_hz=2.0
    )
    assert config.per_unit_rotation_std_degrees == 0.0
    assert config.per_unit_rotation_limit_degrees is None


def quiet_config(**kwargs: object) -> NeuralSignalConfig:
    defaults: dict[str, object] = {
        "n_channels": 2,
        "fs": 1_000.0,
        "seed": 41,
        "population": NeuralPopulationConfig(
            units_per_channel=2,
            baseline_rate_hz=20.0,
            intent_rate_gain_hz=30.0,
            max_rate_hz=100.0,
            preferred_direction_jitter_radians=0.0,
        ),
        "waveform": SpikeWaveformConfig(duration_seconds=0.004),
        "lfp": LfpConfig(frequency_hz=20.0, std_volts=0.0),
        "noise": BackgroundNoiseConfig(std_volts=0.0),
        "nonstationarity": NonstationarityConfig(std_fraction=0.0),
    }
    defaults.update(kwargs)
    return NeuralSignalConfig(**defaults)


def test_public_surface_contains_only_user_facing_types() -> None:
    assert set(simulation.__all__) == {
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
    }
    for removed in (
        "DriftState",
        "Intent2D",
        "NeuralGenerationSummary",
        "PositionIntentConfig",
        "position_intent",
    ):
        assert not hasattr(simulation, removed)


def test_configuration_is_immutable_and_intent_is_validated() -> None:
    config = quiet_config()
    with pytest.raises(FrozenInstanceError):
        config.fs = 2_000.0  # type: ignore[misc]
    generator = NeuralSignalGenerator(config)
    assert generator.sample_index == 0
    assert not hasattr(generator, "dtype")
    assert not hasattr(generator, "expected_sample")
    assert not hasattr(generator, "last_summary")
    with pytest.raises(ValidationError, match="shape"):
        generator.generate(1, (1.0,))


def test_generator_is_reproducible_and_chunk_invariant() -> None:
    config = quiet_config()
    intent = (0.7, -0.2)
    whole_generator = NeuralSignalGenerator(config)
    whole = whole_generator.generate_with_truth(257, intent, drift_progress=0.3)

    chunked_generator = NeuralSignalGenerator(config)
    first = chunked_generator.generate_with_truth(83, intent, drift_progress=0.3)
    second = chunked_generator.generate_with_truth(174, intent, drift_progress=0.3)
    np.testing.assert_array_equal(np.vstack((first.signal, second.signal)), whole.signal)
    np.testing.assert_array_equal(np.vstack((first.spikes, second.spikes)), whole.spikes)
    assert whole.signal.shape == (257, 2)
    assert whole.spikes.shape == (257, 4)
    assert whole.signal.dtype == np.float64
    assert whole.spikes.dtype == np.uint8

    chunked_generator.reset()
    np.testing.assert_array_equal(
        chunked_generator.generate(257, intent, drift_progress=0.3), whole.signal
    )
    assert chunked_generator.sample_index == 257


def test_hidden_spikes_produce_a_biphasic_voltage_waveform() -> None:
    amplitude = 100e-6
    config = quiet_config(
        n_channels=1,
        population=NeuralPopulationConfig(
            units_per_channel=2,
            baseline_rate_hz=80.0,
            intent_rate_gain_hz=0.0,
            max_rate_hz=100.0,
            preferred_direction_jitter_radians=0.0,
        ),
        waveform=SpikeWaveformConfig(
            amplitude_volts=amplitude,
            duration_seconds=0.006,
            spatial_decay_channels=1.0,
        ),
    )
    block = NeuralSignalGenerator(config).generate_with_truth(10_000, (0.0, 0.0))
    spike_rows = np.flatnonzero(np.any(block.spikes != 0, axis=1))
    usable = spike_rows[spike_rows < block.signal.shape[0] - 6]
    triggered = np.stack([block.signal[row : row + 6, 0] for row in usable])
    average = triggered.mean(axis=0)
    assert usable.size > 500
    assert average.min() < -0.5 * amplitude
    assert average.max() > 0.05 * amplitude
    assert int(block.spikes.sum()) > 500


def test_drift_rotates_population_tuning() -> None:
    config = quiet_config(
        n_channels=4,
        population=NeuralPopulationConfig(
            units_per_channel=1,
            baseline_rate_hz=0.0,
            intent_rate_gain_hz=80.0,
            max_rate_hz=80.0,
            preferred_direction_jitter_radians=0.0,
        ),
        drift=NeuralDriftConfig(rotation_degrees=90.0),
    )
    baseline = NeuralSignalGenerator(config).generate_with_truth(10_000, (1.0, 0.0))
    drifted = NeuralSignalGenerator(config).generate_with_truth(
        10_000, (1.0, 0.0), drift_progress=1.0
    )
    baseline_counts = baseline.spikes.sum(axis=0)
    drifted_counts = drifted.spikes.sum(axis=0)
    assert baseline_counts[0] > 5 * baseline_counts[3]
    assert drifted_counts[3] > 5 * drifted_counts[0]


def test_per_unit_drift_is_seeded_bounded_and_fingerprinted() -> None:
    drift = NeuralDriftConfig(
        rotation_degrees=0.0,
        per_unit_rotation_std_degrees=30.0,
        per_unit_rotation_limit_degrees=60.0,
    )
    first = NeuralSignalGenerator(quiet_config(seed=73, drift=drift))
    repeated = NeuralSignalGenerator(quiet_config(seed=73, drift=drift))
    different = NeuralSignalGenerator(quiet_config(seed=74, drift=drift))
    no_random_drift = NeuralSignalGenerator(
        quiet_config(seed=73, drift=NeuralDriftConfig(rotation_degrees=0.0))
    )

    assert first.drift_fingerprint == repeated.drift_fingerprint
    assert first.drift_fingerprint != different.drift_fingerprint
    assert first.drift_fingerprint != no_random_drift.drift_fingerprint
    np.testing.assert_array_equal(
        first.generate(257, (0.7, -0.2), drift_progress=1.0),
        repeated.generate(257, (0.7, -0.2), drift_progress=1.0),
    )


def test_lfp_surrogate_has_the_configured_spectral_peak() -> None:
    fs = 1_000.0
    frequency = 20.0
    config = quiet_config(
        n_channels=1,
        fs=fs,
        population=NeuralPopulationConfig(
            units_per_channel=1,
            baseline_rate_hz=0.0,
            intent_rate_gain_hz=0.0,
            max_rate_hz=1.0,
        ),
        lfp=LfpConfig(
            frequency_hz=frequency,
            damping_time_seconds=0.5,
            std_volts=20e-6,
            spatial_correlation=0.0,
        ),
    )
    signal = NeuralSignalGenerator(config).generate(20_000, (0.0, 0.0))[2_000:, 0]
    frequencies = np.fft.rfftfreq(signal.size, 1.0 / fs)
    power = np.abs(np.fft.rfft(signal)) ** 2
    band = (frequencies >= 5.0) & (frequencies <= 50.0)
    peak = frequencies[band][np.argmax(power[band])]
    assert abs(peak - frequency) < 2.0


def test_background_noise_has_configured_spatial_correlation() -> None:
    correlation = 0.7
    config = quiet_config(
        n_channels=4,
        population=NeuralPopulationConfig(
            units_per_channel=1,
            baseline_rate_hz=0.0,
            intent_rate_gain_hz=0.0,
            max_rate_hz=1.0,
        ),
        noise=BackgroundNoiseConfig(
            std_volts=10e-6,
            time_constant_seconds=0.005,
            spatial_correlation=correlation,
        ),
    )
    signal = NeuralSignalGenerator(config).generate(30_000, (0.0, 0.0))[1_000:]
    matrix = np.corrcoef(signal, rowvar=False)
    observed = matrix[np.triu_indices(4, k=1)].mean()
    assert abs(observed - correlation) < 0.08


def test_background_noise_starts_in_its_stationary_distribution() -> None:
    std_volts = 10e-6
    config = quiet_config(
        n_channels=128,
        population=NeuralPopulationConfig(
            units_per_channel=1,
            baseline_rate_hz=0.0,
            intent_rate_gain_hz=0.0,
            max_rate_hz=1.0,
        ),
        noise=BackgroundNoiseConfig(
            std_volts=std_volts,
            time_constant_seconds=1.0,
            spatial_correlation=0.0,
        ),
    )
    first_sample = NeuralSignalGenerator(config).generate(1, (0.0, 0.0))[0]
    assert 0.75 * std_volts < first_sample.std() < 1.25 * std_volts


def test_slow_rate_nonstationarity_is_temporally_correlated() -> None:
    config = quiet_config(
        n_channels=8,
        population=NeuralPopulationConfig(
            units_per_channel=4,
            baseline_rate_hz=40.0,
            intent_rate_gain_hz=0.0,
            max_rate_hz=120.0,
        ),
        nonstationarity=NonstationarityConfig(
            std_fraction=0.8,
            time_constant_seconds=1.0,
            rate_correlation=1.0,
        ),
    )
    spikes = NeuralSignalGenerator(config).generate_with_truth(20_000, (0.0, 0.0)).spikes
    counts = spikes.reshape(200, 100, -1).sum(axis=(1, 2)).astype(np.float64)
    lag_one = np.corrcoef(counts[:-1], counts[1:])[0, 1]
    assert lag_one > 0.3


def test_invalid_drift_and_noncontiguous_output_are_rejected() -> None:
    with pytest.raises(ValidationError, match="too narrow"):
        NeuralSignalGenerator(
            quiet_config(
                drift=NeuralDriftConfig(
                    rotation_degrees=0.0,
                    per_unit_rotation_std_degrees=1_000.0,
                    per_unit_rotation_limit_degrees=0.1,
                )
            )
        )
    generator = NeuralSignalGenerator(quiet_config())
    output = np.empty((8, 4), dtype=np.float64)[:, ::2]
    with pytest.raises(ValidationError, match="C-contiguous"):
        generator.generate_into(output, intent=(0.0, 0.0))
    with pytest.raises(ValidationError, match=r"at most 1\.0"):
        generator.generate(1, (0.0, 0.0), drift_progress=1.1)
