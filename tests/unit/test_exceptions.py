#!/usr/bin/env python3

from neurale.exceptions import (
    BackpressureError,
    ConfigurationError,
    DependencyError,
    DeviceUnavailableError,
    NativeExtensionError,
    NativeUnavailableError,
    NeuraleError,
    StreamError,
    StreamExecutionError,
    StreamStateError,
    StreamTimeoutError,
    ValidationError,
)


def test_exception_hierarchy() -> None:
    assert issubclass(ConfigurationError, NeuraleError)
    assert issubclass(ConfigurationError, ValueError)
    assert issubclass(DependencyError, NeuraleError)
    assert issubclass(DependencyError, ImportError)
    assert issubclass(NativeUnavailableError, NeuraleError)
    assert issubclass(NativeUnavailableError, RuntimeError)
    assert issubclass(NativeExtensionError, DependencyError)
    assert issubclass(DeviceUnavailableError, NativeUnavailableError)
    assert issubclass(ValidationError, NeuraleError)
    assert issubclass(ValidationError, ValueError)
    assert issubclass(StreamError, NeuraleError)
    assert issubclass(StreamError, RuntimeError)
    assert issubclass(StreamStateError, StreamError)
    assert issubclass(StreamExecutionError, StreamError)
    assert issubclass(StreamTimeoutError, StreamError)
    assert issubclass(StreamTimeoutError, TimeoutError)
    assert issubclass(BackpressureError, StreamError)
