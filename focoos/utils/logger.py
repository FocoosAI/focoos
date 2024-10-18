import logging
import logging.config
from functools import cache



class ColoredFormatter(logging.Formatter):
    log_format = (
        "[%(asctime)s][%(levelname)s][%(name)s]: %(message)s"
    )
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[31;20m"
    blue = "\x1B[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: yellow + log_format + reset,
        logging.INFO: blue + log_format + reset,
        logging.WARNING: purple + log_format + reset,
        logging.ERROR: bold_red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%m/%d %H:%M")
        return formatter.format(record)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "color": {
            # Assicurati di mettere il percorso completo al LogFormatter
            "()": ColoredFormatter,
            # "use_colors": True,
        },
    },
    "handlers": {
        "default": {
            "class": "logging.StreamHandler",
            "formatter": "color",
            "level": "DEBUG",
        },
    },
    "root": {  # Configura il logger di default (root)
        "handlers": ["default"],
        "level": "INFO",
    },
    "loggers": {
        "focoos": {
            "handlers": ["default"],
            "level": "DEBUG",
            "propagate": False,
        },
        "matplotlib": {"level": "WARNING"},
        "botocore": {"level": "INFO"},
    },
}


@cache
def get_logger(name="focoos", level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logging.getLogger(name)


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = get_logger()
    return logger