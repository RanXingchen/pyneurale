from pathlib import Path

from neurale.devices import NativeProvider


def provider():
    candidates = tuple(Path(__file__).parent.glob("*provider.*"))
    if len(candidates) != 1:
        raise ImportError("expected one native provider library")
    return NativeProvider(candidates[0])
