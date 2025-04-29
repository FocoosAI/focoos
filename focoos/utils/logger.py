import atexit
import logging
import logging.config
import os
import sys
import time
from functools import cache
from typing import Counter, Optional

from tabulate import tabulate

from focoos.config import FOCOOS_CONFIG, LogLevel


class ColoredFormatter(logging.Formatter):
    log_format = "[%(asctime)s][%(levelname)s][%(name)s]: %(message)s"
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[1;33m"
    red = "\x1b[31;20m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: yellow + log_format + reset,
        logging.INFO: green + log_format + reset,
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
            "level": FOCOOS_CONFIG.focoos_log_level,
        },
    },
    "root": {  # Configura il logger di default (root)
        "handlers": ["default"],
        "level": "INFO",
    },
    "loggers": {
        "focoos": {
            "handlers": ["default"],
            "level": FOCOOS_CONFIG.focoos_log_level,
            "propagate": False,
        },
        "matplotlib": {"level": "WARNING"},
        "botocore": {"level": "INFO"},
        "fvcore": {"level": "DEBUG"},
    },
}


def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


@cache
def get_logger(name="focoos", level: Optional[LogLevel] = None) -> logging.Logger:
    """
    Get a logger instance with the specified name and logging level.

    Args:
        name (str, optional): The name of the logger. Defaults to "focoos".
        level (LogLevel, optional): The logging level to set. If None, uses the level from FOCOOS_CONFIG (default to DEBUG).
            Must be one of the standard Python logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns:
        logging.Logger: A configured logger instance with the specified name and level.

    Example:
        >>> logger = get_logger("my_module", logging.INFO)
        >>> logger.info("This is an info message")
    """
    level = level or FOCOOS_CONFIG.focoos_log_level
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger


D2_LOG_BUFFER_SIZE_KEY: str = "D2_LOG_BUFFER_SIZE"

DEFAULT_LOG_BUFFER_SIZE: int = 1024 * 1024  # 1MB


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = get_logger()
    return logger


def _find_caller():
    """
    Returns:
        str: module name of the caller
        tuple: a hashable key to be used to identify different callers
    """
    frame = sys._getframe(2)
    while frame:
        code = frame.f_code
        if os.path.join("utils", "logger.") not in code.co_filename:
            mod_name = frame.f_globals["__name__"]
            if mod_name == "__main__":
                mod_name = "detectron2"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


_LOG_COUNTER = Counter()
_LOG_TIMER = {}


def log_first_n(lvl, msg, n=1, *, name=None, key="caller"):
    """
    Log only for the first n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
        key (str or tuple[str]): the string(s) can be one of "caller" or
            "message", which defines how to identify duplicated logs.
            For example, if called with `n=1, key="caller"`, this function
            will only log the first call from the same caller, regardless of
            the message content.
            If called with `n=1, key="message"`, this function will log the
            same content only once, even if they are called from different places.
            If called with `n=1, key=("caller", "message")`, this function
            will not log only if the same caller has logged the same message before.
    """
    if isinstance(key, str):
        key = (key,)
    assert len(key) > 0

    caller_module, caller_key = _find_caller()  # type: ignore
    hash_key = ()
    if "caller" in key:
        hash_key = hash_key + caller_key
    if "message" in key:
        hash_key = hash_key + (msg,)

    _LOG_COUNTER[hash_key] += 1
    if _LOG_COUNTER[hash_key] <= n:
        logging.getLogger(name or caller_module).log(lvl, msg)


def log_every_n(lvl, msg, n=1, *, name=None):
    """
    Log once per n times.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()  # type: ignore
    _LOG_COUNTER[key] += 1
    if n == 1 or _LOG_COUNTER[key] % n == 1:
        get_logger(name or caller_module).log(lvl, msg)


def log_every_n_seconds(lvl, msg, n=1, *, name=None):
    """
    Log no more than once per n seconds.

    Args:
        lvl (int): the logging level
        msg (str):
        n (int):
        name (str): name of the logger to use. Will use the caller's module by default.
    """
    caller_module, key = _find_caller()  # type: ignore
    if key is None or caller_module is None:
        return
    last_logged = _LOG_TIMER.get(key, None)
    current_time = time.time()
    if last_logged is None or current_time - last_logged >= n:
        get_logger(name or caller_module).log(lvl, msg)
        _LOG_TIMER[key] = current_time


def _get_log_stream_buffer_size(filename: str) -> int:
    if "://" not in filename:
        # Local file, no extra caching is necessary
        return -1
    # Remote file requires a larger cache to avoid many small writes.
    if D2_LOG_BUFFER_SIZE_KEY in os.environ:
        return int(os.environ[D2_LOG_BUFFER_SIZE_KEY])
    return DEFAULT_LOG_BUFFER_SIZE


@cache
def _cached_log_stream(filename):
    # use 1K buffer if writing to cloud storage
    io = open(filename, "a", buffering=_get_log_stream_buffer_size(filename))
    atexit.register(io.close)
    return io


def add_file_logging(
    logger: logging.Logger,
    verbose=True,
    output="log.txt",
    rank=0,
):
    level = logging.DEBUG if verbose else logging.INFO
    if output.endswith(".txt") or output.endswith(".log"):
        output = output
    else:
        output = os.path.join(output, "log.txt")

    if os.path.exists(output):
        os.remove(output)

    distributed_rank = rank
    if distributed_rank > 0:
        output = output + ".rank{}".format(distributed_rank)
    dirname = os.path.dirname(output)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    fh = logging.StreamHandler(_cached_log_stream(output))
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%m/%d %H:%M"))
    logger.addHandler(fh)
