# SPDX-License-Identifier: MIT
"""Small, SDK-independent device provider contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True)
class Signal:
    """One immutable signal. Event signals use rate=0 and max_samples=1.

    Array shape is (samples, channels) for sample_major, otherwise
    (channels, samples). Raw ADC counts should use unit='unspecified'.
    """

    name: str
    channels: int
    sample_rate: int | float | Fraction
    max_samples: int
    dtype: str = "float64"
    layout: str = "sample_major"
    kind: str = "sampled"
    unit: str = "unspecified"
    sample_counter: bool = False
    clock_domain: int = 1
    channel_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class NativeProvider:
    """Entry-point factory result identifying an independently built library."""

    library: str | Path


@dataclass(frozen=True)
class PythonProvider:
    """Importable module:factory instantiated only inside the acquisition process."""

    factory: str


class Ingress(Protocol):
    @property
    def cancelled(self) -> bool: ...

    def publish(self, signal_index: int, data: Any, metadata: dict[str, int] = ...) -> None: ...

    def publish_frame(self, blocks: Sequence[tuple[int, Any, dict[str, int]]]) -> None: ...

    def gap(
        self, signal_index: int, restart: bool = False, metadata: dict[str, int] = ...
    ) -> None: ...

    def finish(self) -> None: ...

    def fail(self) -> None: ...


class Provider(Protocol):
    """Python SDK adapter. start returns promptly; close joins every SDK worker."""

    def open(self, config: dict[str, Any]) -> None: ...

    def describe(self) -> Sequence[Signal]: ...

    def start(self, ingress: Ingress) -> None: ...

    def cancel(self) -> None: ...

    def close(self) -> None: ...


def get_include() -> str:
    """Directory containing the wheel-shipped C ABI and header-only helpers."""
    return str(Path(__file__).with_name("sdk"))


def _native_signals(signals: Sequence[Signal]) -> list[Any]:
    from neurale.streaming import _native as n

    if not signals or len({s.name for s in signals}) != len(signals):
        raise ValueError("device signals must be nonempty and have unique names")
    result = []
    for index, signal in enumerate(signals):
        if not signal.name or signal.kind not in ("sampled", "event"):
            raise ValueError("device signal needs a name and sampled/event kind")
        if signal.clock_domain < 1:
            raise ValueError("clock_domain must be positive")
        rate = Fraction(str(signal.sample_rate))
        try:
            value = n.SignalSchema(
                index + 1,
                getattr(n.SignalDType, signal.dtype.upper()),
                signal.channels,
                signal.max_samples,
                signal.max_samples,
                n.RationalRate(rate.numerator, rate.denominator),
                signal.clock_domain,
                layout=getattr(n.SignalLayout, signal.layout.upper()),
                kind=getattr(n.SignalKind, signal.kind.upper()),
                physical_unit=getattr(n.PhysicalUnit, signal.unit.upper()),
                device_tick_tracking=(
                    n.DeviceTickTracking.SAMPLE_COUNTER
                    if signal.sample_counter
                    else n.DeviceTickTracking.UNAVAILABLE
                ),
                channel_names=list(signal.channel_names),
            )
        except AttributeError as error:
            raise ValueError("unsupported device dtype, layout, kind, or unit") from error
        result.append(value)
    return result


def host_time_ns() -> int:
    """Read the default native runtime monotonic clock, in nanoseconds."""
    from neurale._native_loader import load_native_namespace

    return load_native_namespace("devices").host_time_ns()
