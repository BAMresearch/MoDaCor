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
    def __init__(self, level: int = logging.INFO, name: str = 'MoDaCor', **kwargs):
        self.level = level
        self.name = name

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        self.consoleLogHandler = logging.StreamHandler()
        self.consoleLogHandler.setLevel(level)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.consoleLogHandler.setFormatter(formatter)
        self.logger.addHandler(self.consoleLogHandler)

    def log(self, message: str, level: int = None, name: str = None):
        if level is None:
            level = self.level

        if name is None:
            name = self.name
        
        self.logger(message, level=level, name=name)

    def info(self, message: str):
        self.log(message, level=logging.INFO, name='MoDaCor')

    def warning(self, message: str):
        self.log(message, level=logging.WARNING, name='MoDaCor')

    def error(self, message: str):
        self.log(message, level=logging.ERROR, name='MoDaCor')

    def critical(self, message: str):
        self.log(message, level=logging.CRITICAL, name='MoDaCor')
    
    def debug(self, message: str):
        self.log(message, level=logging.DEBUG, name='MoDaCor')
