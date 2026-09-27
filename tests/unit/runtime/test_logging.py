#!/usr/bin/env python3

from __future__ import annotations

import logging
from io import StringIO

from neurale.runtime import (
    configure_logging,
    is_logging_configured,
    reset_logging,
)


def teardown_function() -> None:
    reset_logging()


def test_configure_logging_writes_to_stream() -> None:
    stream = StringIO()

    configure_logging(level=logging.INFO, stream=stream, color=False, force=True)
    logging.getLogger("neurale.runtime.test").info("runtime logger ready")

    output = stream.getvalue()
    assert "INFO" in output
    assert "neurale.runtime.test" in output
    assert "runtime logger ready" in output


def test_configure_logging_is_idempotent_without_force() -> None:
    stream = StringIO()
    logger = configure_logging(stream=stream, color=False, force=True)
    n_handlers = len(logger.handlers)

    same_logger = configure_logging(stream=StringIO(), color=False)

    assert same_logger is logger
    assert len(logger.handlers) == n_handlers


def test_force_reconfigure_replaces_handlers() -> None:
    first_stream = StringIO()
    second_stream = StringIO()

    configure_logging(stream=first_stream, color=False, force=True)
    configure_logging(stream=second_stream, color=False, force=True)
    logging.getLogger("neurale.runtime").warning("new target")

    assert first_stream.getvalue() == ""
    assert "new target" in second_stream.getvalue()


def test_configure_logging_can_write_to_file(tmp_path) -> None:
    log_file = tmp_path / "neurale.log"

    configure_logging(console=False, log_file=log_file, force=True)
    logging.getLogger("neurale.runtime.file").error("file target")
    reset_logging()

    content = log_file.read_text(encoding="utf-8")
    assert "ERROR" in content
    assert "neurale.runtime.file" in content
    assert "file target" in content


def test_file_logging_appends_and_can_rotate(tmp_path) -> None:
    log_file = tmp_path / "neurale.log"
    log_file.write_text("existing\n", encoding="utf-8")

    configure_logging(
        console=False,
        log_file=log_file,
        max_bytes=80,
        n_backups=1,
        force=True,
    )
    logger = logging.getLogger("neurale.runtime.file")
    logger.error("x" * 100)
    reset_logging()

    assert (tmp_path / "neurale.log.1").exists()


def test_unconfigured_package_logging_propagates_to_application() -> None:
    reset_logging()
    logger = logging.getLogger("neurale")
    assert logger.propagate


def test_reset_logging_is_idempotent() -> None:
    configure_logging(force=True)
    assert is_logging_configured()

    reset_logging()
    reset_logging()

    assert not is_logging_configured()


def test_reset_logging_preserves_user_handlers() -> None:
    logger = logging.getLogger("neurale")
    user_handler = logging.NullHandler()
    logger.addHandler(user_handler)

    try:
        configure_logging(force=True)
        reset_logging()
        assert user_handler in logger.handlers
    finally:
        logger.removeHandler(user_handler)
        user_handler.close()
