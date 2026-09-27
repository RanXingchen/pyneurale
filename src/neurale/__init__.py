"""PyNeurale public package.

The top-level package intentionally stays lightweight. Import feature modules
explicitly so importing :mod:`neurale` never initializes optional native,
hardware, GUI, or heavy scientific-computing dependencies.
"""

try:
    from ._version import __version__
except ModuleNotFoundError:
    __version__ = "0+unknown"

__all__ = ["__version__"]
