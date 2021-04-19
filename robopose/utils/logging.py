import logging
import time
from datetime import timedelta


class ElapsedFormatter():

    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = record.created - self.start_time
        elapsed = timedelta(seconds=elapsed_seconds)
        return "{} - {}".format(elapsed, record.getMessage())


def get_logger(name):
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    handler.setFormatter(ElapsedFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def set_logging_level(level):
    if 'level' == 'debug':
        loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
        for logger in loggers:
            if 'robopose' in logger.name:
                logger.setLevel(logging.DEBUG)
    else:
        pass
    return
