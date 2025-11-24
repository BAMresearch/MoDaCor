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

# logger = logging.getLogger(__name__)


class MessageHandler:
    """
    A simple class to handle logging messages at different levels.
    This class should be replaced to match the messaging system used at a given location.

    Args:
        level (int): The logging level to use. Defaults to logging.INFO.
    """

    def __init__(self, level: int = logging.INFO, name: str = "MoDaCor", **kwargs):
        self.level = level
        self.name = name

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        self.consoleLogHandler = logging.StreamHandler()
        self.consoleLogHandler.setLevel(level)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.consoleLogHandler.setFormatter(formatter)
        self.logger.addHandler(self.consoleLogHandler)

    def log(self, message: str, level: int = None) -> None:  # , name: str = None):
        if level is None:
            level = self.level

        # if name is None:
        #     name = self.name
        # does not take a name: # self.logger = logging.getLogger(name)
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
