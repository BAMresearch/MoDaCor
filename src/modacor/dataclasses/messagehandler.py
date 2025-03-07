# src/modacor/dataclasses/messagehandler.py
# # -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)


class MessageHandler:
    """
    A simple class to handle logging messages at different levels.
    This class should be replaced to match the messaging system used at a given location.

    Args:
        level (int): The logging level to use. Defaults to logging.INFO.
    """
    def __init__(self, level: int = logging.INFO, **kwargs):
        self.level = level

    def log(self, message: str, level: int = None):
        if level is None:
            level = self.level
        logger.log(level, message)

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
    
