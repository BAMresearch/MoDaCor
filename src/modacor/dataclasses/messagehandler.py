# SPDX-License-Identifier: BSD-3-Clause
# /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

__coding__ = "utf-8"
__authors__ = ["Malte Storm", "Tim Snow", "Brian R. Pauw"]  # add names to the list as appropriate
__copyright__ = "Copyright 2025, The MoDaCor team"
__date__ = "16/11/2025"
__status__ = "Development"  # "Development", "Production"
# end of header and standard imports


import logging

_default_handler: MessageHandler | None = None


def get_default_handler(level: int = logging.INFO) -> MessageHandler:
    """
    MoDaCor-wide default message handler. Useful for overarching logging like in the pipeline runner.
    For specific modules or classes, it's better to create dedicated named MessageHandler instances.
    """
    global _default_handler
    if _default_handler is None:
        _default_handler = MessageHandler(level=level, name="MoDaCor")
    return _default_handler


class MessageHandler:
    """
    A simple class to handle logging messages at different levels.
    This class should be replaced to match the messaging system used at a given location.

    Args:
        level (int): The logging level to use. Defaults to logging.INFO.
        name (str): Logger name (typically __name__).
    """

    def __init__(self, level: int = logging.INFO, name: str = "MoDaCor", **kwargs):
        self.level = level
        self.name = name

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Avoid adding multiple console handlers if this handler is created multiple times
        if not any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def log(self, message: str, level: int = None) -> None:
        if level is None:
            level = self.level
        self.logger.log(msg=message, level=level)

    def info(self, message: str):
        self.log(message, level=logging.INFO)

    def warning(self, message: str):
        self.log(message, level=logging.WARNING)

    def error(self, message: str):
        self.log(message, level=logging.ERROR)

    def critical(self, message: str):
        self.log(message, level=logging.CRITICAL)

    def debug(self, message: str):
        self.log(message, level=logging.DEBUG)
