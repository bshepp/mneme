"""Tests for utils.logging."""

import logging

import pytest

from mneme.utils.logging import (
    LoggerMixin,
    configure_logging_from_config,
    create_experiment_logger,
    get_logger,
    log_execution_time,
    log_function_call,
    set_log_level,
    setup_logging,
)


def test_setup_logging_console_only():
    logger = setup_logging(level=logging.DEBUG, console=True)
    assert logger.name == "mneme"
    assert logger.level == logging.DEBUG
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_setup_logging_file(tmp_path):
    log_file = tmp_path / "logs" / "x.log"
    logger = setup_logging(level=logging.INFO, log_file=log_file, console=False)
    logger.info("hello")
    for h in logger.handlers:
        h.flush()
    assert log_file.exists()


def test_setup_logging_rotation(tmp_path):
    from logging.handlers import RotatingFileHandler

    log_file = tmp_path / "rot.log"
    logger = setup_logging(
        level=logging.INFO, log_file=log_file, console=False, rotation=True, max_bytes=1024
    )
    assert any(isinstance(h, RotatingFileHandler) for h in logger.handlers)


def test_get_logger_namespaced():
    logger = get_logger("foo")
    assert logger.name == "mneme.foo"


def test_logger_mixin():
    class Sample(LoggerMixin):
        pass

    s = Sample()
    assert s.logger.name == "mneme.Sample"


def test_log_function_call_success():
    @log_function_call
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_log_function_call_propagates_error():
    @log_function_call
    def boom():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        boom()


def test_log_execution_time_success():
    @log_execution_time
    def f():
        return 1

    assert f() == 1


def test_log_execution_time_propagates_error():
    @log_execution_time
    def boom():
        raise ValueError("x")

    with pytest.raises(ValueError):
        boom()


def test_set_log_level_string_and_int():
    setup_logging(level=logging.INFO, console=True)
    set_log_level("WARNING")
    assert logging.getLogger("mneme").level == logging.WARNING
    set_log_level(logging.INFO)
    assert logging.getLogger("mneme").level == logging.INFO


def test_configure_logging_from_config(tmp_path):
    cfg = {
        "logging": {
            "level": "DEBUG",
            "console": False,
            "file": str(tmp_path / "x.log"),
        }
    }
    configure_logging_from_config(cfg)
    assert logging.getLogger("mneme").level == logging.DEBUG


def test_create_experiment_logger(tmp_path):
    logger = create_experiment_logger("exp1", tmp_path)
    assert isinstance(logger, logging.Logger)
    assert tmp_path.exists()
