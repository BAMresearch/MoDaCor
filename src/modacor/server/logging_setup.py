# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from pathlib import Path

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def configure_server_logging(*, log_file: str | Path | None = None, log_level: str = "info") -> None:
    """Configure the serve process with one console and optional file destination.

    Uvicorn must use ``log_config=None`` so it keeps this configuration.
    MessageHandler delegates to these root handlers while the server runs.
    """
    level = logging.getLevelNamesMapping()[log_level.upper()]
    formatter = logging.Formatter(LOG_FORMAT)
    console = logging.StreamHandler()
    console.set_name("modacor.runtime.console")
    handlers: list[logging.Handler] = [console]
    if log_file is not None:
        path = Path(log_file).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
    # This function owns logging in the CLI server process. Remove existing
    # MessageHandler/uvicorn console destinations before installing root ones.
    for name, logger in list(logging.root.manager.loggerDict.items()):
        if not isinstance(logger, logging.Logger):
            continue
        message_handlers = [handler for handler in logger.handlers if handler.name == "modacor.message_handler.console"]
        managed = name in {"modacor", "uvicorn"} or name.startswith(("modacor.", "uvicorn."))
        for handler in list(logger.handlers) if managed else message_handlers:
            logger.removeHandler(handler)
            handler.close()
        if managed or message_handlers:
            logger.propagate = True
            logger.setLevel(logging.NOTSET)
    logging.basicConfig(level=level, handlers=handlers, force=True)
    if log_file is not None:
        logging.getLogger(__name__).info("MoDaCor runtime log: %s", path)
