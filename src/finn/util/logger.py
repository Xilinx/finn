# Description: This module provides a logger class and decorators for logging function calls and performance.
#              outputs are written to both console and a file named 'finn.log'.
#              the log level can be set using the verbosity argument in the configure_logging method.
#              verbosity can be set to 0, 1, 2, or 3 for logging levels WARNING, INFO, DEBUG, and NOTSET respectively.
# usage example:
#       from logger import finn_logger, log_func, log_func_perf
#       finn_logger.configure_logging(verbosity)
#       logger = finn_logger.get_logger(__name__)
#       @log_func(__name__)
#       @log_func_perf(__name__)
#       def my_function():
#           logger.info("Hello, world!")
#           logger.warning("This is a warning!")
#           logger.error("This is an error!")
#           logger.debug("This is a debug message!")
#           return 42

import logging
import logging.config
import functools
import time

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'brief': {
            'format': '[%(levelname)-6s] %(asctime)s - %(message)s',
            'datefmt': '%d-%m-%Y %H:%M:%S'
        },
        'precise': {
            'format': '[%(levelname)-6s] %(asctime)s - %(name)s - %(message)s [in %(pathname)s:%(lineno)d]',
            'datefmt': '%d-%m-%Y %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'brief',
            'level': 'INFO',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'finn.log',
            'formatter': 'precise',
            'level': 'DEBUG',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        }
    }
}

class FinnLogger:
    def __init__(self, config):
        self.config = config

    def get_logging_level(self, verbosity):
        if verbosity == 0:
            return logging.WARNING
        elif verbosity == 1:
            return logging.INFO
        elif verbosity == 2:
            return logging.DEBUG
        else:
            return logging.NOTSET

    def configure_logging(self, verbosity):
        logging_level = self.get_logging_level(verbosity)
        self.config['handlers']['console']['level'] = logging_level
        self.config['handlers']['file']['level'] = logging_level
        logging.config.dictConfig(self.config)

    def get_logger(self, name):
        return logging.getLogger(name)


finn_logger = FinnLogger(LOGGING_CONFIG)

# Decorator for logging function calls
def log_func(logger_name):
    def decorator_log_func(func):
        logger = finn_logger.get_logger(logger_name)

        @functools.wraps(func)
        def wrapper_log_func(*args, **kwargs):
            logger.debug(f"Entering {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"Exiting {func.__name__}")
                return result
            except Exception as e:
                logger.error(f"Exception in {func.__name__}: {e}", exc_info=True)
                raise
        return wrapper_log_func
    return decorator_log_func

# Decorator for logging function performance
def log_func_perf(logger_name):
    def decorator_log_func_perf(func):
        logger = finn_logger.get_logger(logger_name)

        @functools.wraps(func)
        def wrapper_log_func_perf(*args, **kwargs):
            start_time = time.time()
            logger.debug(f"Starting {func.__name__}")
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.debug(f"Finished {func.__name__} in {elapsed_time:.4f} seconds")
                return result
            except Exception as e:
                end_time = time.time()
                elapsed_time = end_time - start_time
                logger.error(f"Exception in {func.__name__} after {elapsed_time:.4f} seconds: {e}", exc_info=True)
                raise
        return wrapper_log_func_perf
    return decorator_log_func_perf