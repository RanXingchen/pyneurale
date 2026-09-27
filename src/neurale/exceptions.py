#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Define the public exception hierarchy used throughout PyNeurale."""


class NeuraleError(Exception):
    """Base class for PyNeurale-specific exceptions."""


class ConfigurationError(NeuraleError, ValueError):
    """Report invalid runtime or project configuration."""


class DependencyError(NeuraleError, ImportError):
    """Report an unavailable or unusable dependency."""


class NativeUnavailableError(NeuraleError, RuntimeError):
    """Report that a requested native implementation is unavailable."""


class NativeExtensionError(DependencyError):
    """Report that the native extension cannot be loaded or is incompatible."""


class DeviceUnavailableError(NativeUnavailableError):
    """Report that a requested execution device is unavailable."""


class ValidationError(NeuraleError, ValueError):
    """Report invalid user-provided data or parameters."""


class StreamError(NeuraleError, RuntimeError):
    """Base class for streaming lifecycle and execution failures."""


class StreamStateError(StreamError):
    """Report an operation that is invalid for the current stream state."""


class StreamTimeoutError(StreamError, TimeoutError):
    """Report that a streaming operation exceeded its configured timeout."""


class BackpressureError(StreamError):
    """Report that a strict bounded queue cannot accept another frame."""


class StreamExecutionError(StreamError):
    """Report failure of a streaming component or worker."""


class RecorderError(StreamError):
    """Base class for streaming-to-NRF recording failures.

    Defined here rather than in :mod:`neurale.recording` because two layers
    raise it. The recording plan's document form is fixed by the NRF
    ``neurale.native_replay`` extension, so :mod:`neurale.io.nrf` parses and
    validates it -- and an error type that a lower layer raises cannot be owned
    by a higher one without inverting the dependency. The recorder's own
    hierarchy is unchanged and :mod:`neurale.recording` re-exports both names.
    """


class RecorderConfigError(RecorderError, ValueError):
    """The recording could not be described as a valid NRF session."""
