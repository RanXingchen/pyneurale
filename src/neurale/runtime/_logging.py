#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2026 pyneurale contributors
# SPDX-License-Identifier: MIT

"""Configure logging for the ``neurale`` namespace."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import RLock
from typing import IO, ClassVar, Literal

LOGGER_NAME = "neurale"
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOCK = RLock()
_CONFIGURED = False


class _ColorFormatter(logging.Formatter):
    _COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    _RESET: ClassVar[str] = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self._COLORS.get(record.levelno)
        if color is None:
            return message
        return f"{color}{message}{self._RESET}"


def _root_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def _owned_handler(handler: logging.Handler) -> logging.Handler:
    handler._neurale_owned = True
    return handler


def _clear_owned_handlers(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_neurale_owned", False):
            logger.removeHandler(handler)
            handler.close()


def _make_formatter(
    *,
    fmt: str,
    datefmt: str,
    color: bool,
) -> logging.Formatter:
    formatter_class = _ColorFormatter if color else logging.Formatter
    return formatter_class(fmt=fmt, datefmt=datefmt)


def _require_non_negative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def configure_logging(
    *,
    level: int | str = logging.INFO,
    stream: IO[str] | None = None,
    log_file: str | Path | None = None,
    console: bool = True,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATE_FORMAT,
    color: bool | None = None,
    file_mode: Literal["a", "w"] = "a",
    max_bytes: int = 0,
    n_backups: int = 0,
    force: bool = False,
) -> logging.Logger:
    """Configure logging for the ``neurale`` logger tree.

    The process-wide root logger is not modified.

    Parameters
    ----------
    level : int or str, default=logging.INFO
        Logging threshold applied to the package logger.
    stream : IO[str] or None, optional
        Console stream. Defaults to :data:`sys.stderr`.
    log_file : str or pathlib.Path or None, optional
        File receiving a second copy of log records.
    console : bool, default=True
        Whether to install a console handler.
    fmt : str, default=DEFAULT_FORMAT
        Logging format string.
    datefmt : str, default=DEFAULT_DATE_FORMAT
        Datetime format string.
    color : bool or None, optional
        Enable ANSI colors. If ``None``, detect support from the stream.
    file_mode : {"a", "w"}, default="a"
        Append to or replace ``log_file`` when rotation is disabled.
    max_bytes : int, default=0
        Rotate the file when it exceeds this size. Zero disables rotation.
    n_backups : int, default=0
        Number of rotated files retained when ``max_bytes`` is positive.
    force : bool, default=False
        Replace existing PyNeurale-managed handlers when ``True``.

    Returns
    -------
    logging.Logger
        Configured package root logger.

    Notes
    -----
    Repeated calls without ``force=True`` preserve existing handlers and only
    update the logger level.
    """

    global _CONFIGURED

    if file_mode not in ("a", "w"):
        raise ValueError("file_mode must be 'a' or 'w'.")
    _require_non_negative_int(max_bytes, "max_bytes")
    _require_non_negative_int(n_backups, "n_backups")

    with _LOCK:
        logger = _root_logger()

        if _CONFIGURED and not force:
            logger.setLevel(level)
            return logger

        _clear_owned_handlers(logger)

        logger.setLevel(level)
        logger.propagate = False

        handlers: list[logging.Handler] = []

        if console:
            target_stream = stream if stream is not None else sys.stderr
            use_color = (
                bool(getattr(target_stream, "isatty", lambda: False)()) if color is None else color
            )
            handler = logging.StreamHandler(target_stream)
            handler.setFormatter(_make_formatter(fmt=fmt, datefmt=datefmt, color=use_color))
            handlers.append(_owned_handler(handler))

        if log_file is not None:
            path = Path(log_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler
            if max_bytes > 0:
                handler = RotatingFileHandler(
                    path,
                    mode="a",
                    maxBytes=max_bytes,
                    backupCount=n_backups,
                    encoding="utf-8",
                )
            else:
                handler = logging.FileHandler(path, mode=file_mode, encoding="utf-8")
            handler.setFormatter(_make_formatter(fmt=fmt, datefmt=datefmt, color=False))
            handlers.append(_owned_handler(handler))

        if not handlers:
            handlers.append(_owned_handler(logging.NullHandler()))

        for handler in handlers:
            logger.addHandler(handler)

        _CONFIGURED = True
        return logger


def reset_logging() -> None:
    """Remove handlers installed by :func:`configure_logging`.

    Notes
    -----
    Handlers installed by other libraries or application code are not modified.
    """

    global _CONFIGURED

    with _LOCK:
        _clear_owned_handlers(_root_logger())
        _root_logger().propagate = True
        _CONFIGURED = False


def is_logging_configured() -> bool:
    """Return whether PyNeurale logging has been configured.

    Returns
    -------
    bool
        ``True`` after configuration and before :func:`reset_logging` is called.
    """

    return _CONFIGURED
