import json
import sys
import logging
import os
from definitions import Definitions


class LoggerFactory:
    __configuration_file_path = "log.config"
    __loggers = dict()

    @staticmethod
    def configure(logger_name):
        path = os.path.join(Definitions.ROOT_DIR, LoggerFactory.__configuration_file_path)
        with open(path) as file:
            configuration_data = json.load(file)
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.getLevelName(configuration_data["level"]))

        handler_types = configuration_data["handlerTypes"]
        for handler_type in handler_types:
            if handler_type == "none":
                handler = logging.NullHandler()
                logger.addHandler(handler)
            if handler_type == "console":
                handler = logging.StreamHandler(stream=sys.stdout)
                handler.setFormatter(logging.Formatter(configuration_data["format"]))
                logger.addHandler(handler)
            if handler_type == "file":
                if configuration_data["appendToFile"] is True:
                    mode = 'a'
                else:
                    mode = 'w'
                file_path = os.path.join(Definitions.ROOT_DIR, configuration_data["filePath"])
                handler = logging.FileHandler(file_path, mode)
                handler.setFormatter(logging.Formatter(configuration_data["format"]))
                logger.addHandler(handler)
        LoggerFactory.__loggers[logger_name] = logger
        return logger

    @staticmethod
    def get_logger(logger_name):
        if logger_name in LoggerFactory.__loggers:
            return LoggerFactory.__loggers[logger_name]
        return LoggerFactory.configure(logger_name)
