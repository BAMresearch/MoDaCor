# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging

import pytest

from modacor.dataclasses.messagehandler import MessageHandler
from modacor.server.logging_setup import configure_server_logging


@pytest.fixture
def isolated_logging(monkeypatch):
    # Keep pytest's handlers alive and restore all logger configuration after
    # testing the setup function that owns logging in a dedicated CLI process.
    for name in (
        "modacor.tests.early",
        "modacor.tests.late",
        "EarlyHandler",
        "LateHandler",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "modacor.server.logging_setup",
    ):
        logging.getLogger(name)
    for logger in [logging.getLogger(), *list(logging.root.manager.loggerDict.values())]:
        if isinstance(logger, logging.Logger):
            monkeypatch.setattr(logger, "handlers", list(logger.handlers) if logger.name != "root" else [])
            monkeypatch.setattr(logger, "level", logger.level)
            monkeypatch.setattr(logger, "propagate", logger.propagate)
    yield
    for handler in logging.getLogger().handlers:
        handler.close()
    logging.getLogger().handlers = []


@pytest.mark.parametrize("repeat", [False, True])
def test_serve_logs_once_per_destination_for_early_and_late_handlers(isolated_logging, tmp_path, capsys, repeat):
    early = MessageHandler(name="modacor.tests.early")
    bare_early = MessageHandler(name="EarlyHandler")
    # Uvicorn may already have a console destination when embedded.
    logging.getLogger("uvicorn").addHandler(logging.StreamHandler())
    path = tmp_path / "logs" / "runtime.log"
    configure_server_logging(log_file=path, log_level="debug")
    if repeat:
        configure_server_logging(log_file=path, log_level="debug")
    late = MessageHandler(name="modacor.tests.late")
    bare_late = MessageHandler(name="LateHandler")
    loggers = [
        early.logger,
        late.logger,
        bare_early.logger,
        bare_late.logger,
        logging.getLogger("uvicorn.error"),
        logging.getLogger("uvicorn.access"),
    ]
    for index, logger in enumerate(loggers):
        logger.debug("unique-diagnostic-%s", index)
    stderr = capsys.readouterr().err
    contents = path.read_text()
    for index, logger in enumerate(loggers):
        marker = f"unique-diagnostic-{index}"
        assert stderr.count(marker) == 1
        assert contents.count(marker) == 1
        assert f" - {logger.name} - DEBUG - {marker}" in stderr


def test_serve_log_level_applies_to_message_handlers(isolated_logging, tmp_path, capsys):
    path = tmp_path / "runtime.log"
    configure_server_logging(log_file=path, log_level="warning")
    handler = MessageHandler(name="LateHandler")
    handler.info("filtered-info-record")
    handler.warning("visible-warning-record")
    stderr = capsys.readouterr().err
    assert "filtered-info-record" not in stderr + path.read_text()
    assert stderr.count("visible-warning-record") == 1
    assert path.read_text().count("visible-warning-record") == 1
