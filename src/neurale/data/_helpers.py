#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Internal validation and normalization helpers for data models."""

from __future__ import annotations

import copy
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, TypeVar

import numpy as np

from neurale.exceptions import ValidationError

ALLOWED_CHANNEL_TYPES = frozenset(
    {
        "ecog",
        "eeg",
        "seeg",
        "lfp",
        "spike",
        "emg",
        "eog",
        "ttl",
        "analog",
        "imu",
        "behavior",
        "target",
        "decoder",
        "aux",
    }
)

_T = TypeVar("_T")


class FrozenMapping(Mapping[str, Any]):
    # Small immutable and pickle-friendly mapping used for public metadata.

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, Any] | None = None) -> None:
        self._values = {} if values is None else dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __reduce__(self):
        # Route through ``_rebuild_frozen_mapping`` so unpickle and deep copy
        # re-freeze nested ndarrays. Without this, ``np.ndarray`` reconstruction
        # restores writable arrays inside the mapping, breaking the deep
        # immutability contract for every object that stores a ``FrozenMapping``
        # (SignalArray.attrs, ChannelInfo.attrs, Clock.attrs, Event.attrs, ...).
        return _rebuild_frozen_mapping, (self._values,)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._values!r})"


class ValidatedNamedDict(dict[str, _T]):
    def __init__(
        self,
        values: dict[str, _T] | None,
        *,
        value_type: type[_T],
        label: str,
    ) -> None:
        self._value_type = value_type
        self._label = label
        super().__init__()
        self.update({} if values is None else values)

    def __setitem__(self, key: str, value: _T) -> None:
        self._validate_item(key, value)
        super().__setitem__(key, value)

    def _validate_item(self, key: str, value: _T) -> None:
        if not isinstance(key, str) or key == "":
            raise ValidationError(f"{self._label} names must be non-empty strings.")
        if not isinstance(value, self._value_type):
            raise ValidationError(
                f"{self._label} values must be {self._value_type.__name__} instances."
            )

    def update(self, values=(), /, **kwargs: _T) -> None:
        pending = dict(values, **kwargs)
        for key, value in pending.items():
            self._validate_item(key, value)
        super().update(pending)

    def setdefault(self, key: str, default: _T | None = None) -> _T:
        if key in self:
            return self[key]
        self[key] = default  # type: ignore[assignment]
        return self[key]

    def __ior__(self, other):
        self.update(other)
        return self


def ensure_non_empty_string(value: str, name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValidationError(f"{name} must be a non-empty string.")
    return value


def ensure_positive_float(value: float | None, name: str) -> float | None:
    if value is None:
        return None
    value = ensure_finite_float(value, name)
    if value <= 0:
        raise ValidationError(f"{name} must be positive.")
    return value


def ensure_non_negative_float(value: float, name: str) -> float:
    value = ensure_finite_float(value, name)
    if value < 0:
        raise ValidationError(f"{name} must be non-negative.")
    return value


def ensure_finite_float(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValidationError(f"{name} must be a finite real number.")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must be a finite real number.") from exc
    if not np.isfinite(converted):
        raise ValidationError(f"{name} must be finite.")
    return converted


def as_2d_array(data: Any, name: str) -> np.ndarray:
    # Convert input data to a 2D NumPy array.
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValidationError(f"{name} must be a 2D array.")
    if not (np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_)):
        raise ValidationError(f"{name} must contain numeric values.")
    return arr


def as_1d_float_array(data: Any, name: str) -> np.ndarray:
    try:
        arr = np.asarray(data, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValidationError(f"{name} must contain numeric values.") from exc
    if arr.ndim != 1:
        raise ValidationError(f"{name} must be a 1D array.")
    if not np.all(np.isfinite(arr)):
        raise ValidationError(f"{name} must contain only finite values.")
    return arr


def ensure_monotonic_non_decreasing(values: np.ndarray, name: str) -> None:
    if values.size > 1 and np.any(np.diff(values) < 0):
        raise ValidationError(f"{name} must be monotonic non-decreasing.")


def validate_unit(unit: str | Sequence[str], n_items: int, name: str) -> str | list[str]:
    if isinstance(unit, str):
        return ensure_non_empty_string(unit, name)

    units = list(unit)
    if len(units) != n_items:
        raise ValidationError(f"{name} must contain {n_items} entries.")
    for idx, item in enumerate(units):
        ensure_non_empty_string(item, f"{name}[{idx}]")
    return units


def copy_attrs(attrs: dict[str, Any] | None) -> dict[str, Any]:
    return {} if attrs is None else dict(attrs)


_IMMUTABLE_DATACLASS_TYPES: tuple[type, ...] | None = None


def _admitted_dataclass_types() -> tuple[type, ...]:
    # A ``frozen=True`` dataclass is only *shallowly* frozen: it forbids
    # rebinding fields but not mutating the objects its fields reference (a
    # ``list`` field stays mutable), and ``copy.deepcopy`` of a frozen dataclass
    # that holds a read-only ndarray restores it as writable. Blindly trusting
    # any frozen dataclass therefore breaks deep immutability. Only specific
    # project data types whose fields are themselves deeply immutable --
    # scalars, tuples, and ``FrozenMapping`` (never a direct ndarray) -- are
    # admitted; their ndarrays live inside ``FrozenMapping`` attrs, which
    # ``_rebuild_frozen_mapping`` re-freezes on pickle/deepcopy. The import is
    # lazy because ``channels`` and ``time`` import from this module, and the
    # result is cached so it runs at most once.
    global _IMMUTABLE_DATACLASS_TYPES
    if _IMMUTABLE_DATACLASS_TYPES is None:
        from .channels import ChannelInfo, ChannelTable
        from .time import Clock

        _IMMUTABLE_DATACLASS_TYPES = (ChannelInfo, ChannelTable, Clock)
    return _IMMUTABLE_DATACLASS_TYPES


def freeze_metadata(
    value: Any,
) -> Any:
    # Implements the deep-freeze contract documented in :mod:`neurale.data`.
    # Rejects mutable leaf types the frozen containers cannot protect -- e.g.
    # ``bytearray``, an arbitrary ``frozen=True`` dataclass with mutable
    # fields, or a non-frozen dataclass -- so a caller cannot stash a mutable
    # payload behind the "immutable metadata" contract.
    if isinstance(value, Mapping):
        return FrozenMapping({key: freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_metadata(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return freeze_metadata(value.tolist())
        return immutable_array_copy(value)
    if isinstance(value, np.void):
        raise ValidationError("structured NumPy scalars are not supported as metadata values.")
    if value is None or isinstance(value, (bool, int, float, str, bytes, np.generic)):
        return value
    if isinstance(value, _admitted_dataclass_types()):
        return copy.deepcopy(value)
    raise ValidationError(
        f"metadata value of type {type(value).__name__!r} is not supported; "
        "attrs only accepts None, bool, int, float, str, bytes, numpy scalars, "
        "mappings, sequences, sets, ndarrays (object arrays are converted to "
        "immutable containers), and the project "
        "frozen dataclasses (ChannelInfo, ChannelTable, Clock)."
    )


def _rebuild_frozen_mapping(values: Mapping[str, Any]) -> FrozenMapping:
    # Module-level rebuild entry point referenced by ``FrozenMapping.__reduce__``.
    # Pickle resolves it by qualified name, so it must live at module scope.
    # Re-entering ``freeze_metadata`` re-freezes every nested ndarray (and
    # re-wraps nested mappings) so unpickle and deep copy restore deep
    # immutability instead of leaving writable arrays inside the mapping.
    return freeze_metadata(values)


def immutable_array_copy(value: np.ndarray) -> np.ndarray:
    # Back arrays with immutable bytes so writeability cannot be re-enabled.
    arr = np.asarray(value)
    order = "F" if arr.flags.f_contiguous and not arr.flags.c_contiguous else "C"
    buffer = arr.tobytes(order=order)
    return np.frombuffer(buffer, dtype=arr.dtype).reshape(
        arr.shape,
        order=order,
    )


def _array_root_buffer(arr: np.ndarray) -> Any:
    # Walk the ``.base`` view chain to the object that ultimately owns the
    # memory. For an array produced by ``immutable_array_copy`` the root is the
    # ``bytes`` buffer returned by ``ndarray.tobytes``; for a view of a writable
    # ndarray the root is that writable ndarray. A short cycle guard caps the
    # walk so a pathological self-referential chain cannot loop forever.
    current: Any = arr
    for _ in range(32):
        base = getattr(current, "base", None)
        if base is None or base is current:
            return current
        current = base
    return current  # pragma: no cover - pathological cycle guard


def is_strongly_immutable(arr: np.ndarray) -> bool:
    # True only when ``array`` cannot be made writable again. ``flags.writeable
    # == False`` is insufficient: a view of a writable base can be marked
    # read-only yet still allow ``flags.writeable = True`` (and then mutate the
    # base through the view). The robust proof is that the ultimate backing
    # buffer is an immutable ``bytes`` object -- NumPy cannot expose a ``bytes``
    # buffer for writing, so neither the array nor any of its views can be
    # re-enabled. ``bytearray`` is deliberately excluded because it is mutable.
    if arr.flags.writeable:
        return False
    return isinstance(_array_root_buffer(arr), bytes)


def build_sample_time_grid(
    time: np.ndarray | None,
    t0: float,
    n_samples: int,
    fs: float | None,
) -> np.ndarray:
    # Build a sample time vector for a sampled container. When ``time`` is
    # ``None`` the grid is derived from ``t0`` and ``fs``; otherwise
    # the supplied vector is validated, copied, and checked for length and
    # monotonicity. The returned array is made read-only so callers cannot
    # mutate the time axis in place.
    if time is None:
        if fs is None:
            raise ValidationError("fs must be provided when time is omitted.")
        built = t0 + np.arange(n_samples, dtype=float) / fs
    else:
        built = as_1d_float_array(time, "time").copy()
        if built.size != n_samples:
            raise ValidationError("time length must match data.shape[0].")
        ensure_monotonic_non_decreasing(built, "time")
    built.flags.writeable = False
    return built
