import logging
import logging.config
import os
import sys
import time
from contextlib import contextmanager
from functools import cache
from typing import Counter, Optional

from tabulate import tabulate

from focoos.config import FOCOOS_CONFIG, LogLevel

D2_LOG_BUFFER_SIZE_KEY: str = "D2_LOG_BUFFER_SIZE"

DEFAULT_LOG_BUFFER_SIZE: int = 1024 * 1024  # 1MB

LOG_FORMAT = "[%(asctime)s][%(levelname)s][%(name)s]: %(message)s"

_LOG_COUNTER = Counter()
_LOG_TIMER = {}


class ColoredFormatter(logging.Formatter):
    """
    A custom formatter that adds color to log messages based on their level.

    This formatter applies different colors to log messages depending on their severity level:
    - DEBUG: yellow
    - INFO: green
    - WARNING: purple
    - ERROR: bold red
    - CRITICAL: bold red

    The format follows the standard LOG_FORMAT pattern with added ANSI color codes.
    """

    log_format = LOG_FORMAT
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
        """
        Format the log record with appropriate colors.

        Args:
            record: The log record to format

        Returns:
            str: The formatted log message with color codes
        """
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%m/%d %H:%M")
        return formatter.format(record)


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "color": {
            # Make sure to use the full path to the LogFormatter
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
    "root": {  # Configure the default (root) logger
        "handlers": ["default"],
        "level": "INFO",
    },
    "loggers": {
        # General configuration for all focoos.* loggers
        "focoos": {
            "handlers": ["default"],
            "level": FOCOOS_CONFIG.focoos_log_level,
            "propagate": False,  # Don't propagate to the root logger
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


def _setup_logging():
    """
    Configure the logging system using the LOGGING_CONFIG dictionary.

    This function initializes the logging system with the predefined configuration,
    setting up formatters, handlers, and logger levels.
    """
    logging.config.dictConfig(LOGGING_CONFIG)


def _find_caller():
    """
    Find the calling module and location in the stack.

    This function walks up the call stack to find the first frame that is not
    part of the logger module itself, to identify where the logging call originated.

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
                mod_name = "focoos"
            return mod_name, (code.co_filename, frame.f_lineno, code.co_name)
        frame = frame.f_back


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
        get_logger(name or caller_module).log(lvl, msg)


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


class TeeStream:
    """
    A stream wrapper that duplicates output to both the original stream and a log file.

    This class is used to capture output that would normally go to stdout or stderr
    and also write it to a log file, allowing for both console display and logging.

    Args:
        original_stream: The original output stream (typically sys.stdout or sys.stderr)
        log_file: The file object to which output should also be written
    """

    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, data):
        """
        Write data to both the original stream and the log file.

        Args:
            data: The data to write
        """
        self.original_stream.write(data)
        self.log_file.write(data)

    def flush(self):
        """Flush both the original stream and the log file."""
        self.original_stream.flush()
        self.log_file.flush()


@contextmanager
def capture_all_output(log_path="output.txt", rank=0):
    """
    Context manager that captures all stdout, stderr, and logging output to a file.

    This function redirects standard output streams and logging to a specified file,
    which is useful for capturing all program output during execution. It's particularly
    helpful in distributed environments where each process can have its own log file.

    Args:
        log_path (str): Path to the log file or directory. If a directory is provided,
                       a file named "log.txt" will be created in that directory.
        rank (int): Process rank in distributed training. Used to create rank-specific
                   log files when running with multiple processes.

    Yields:
        None: This context manager doesn't yield a value, but sets up the logging
              environment for the duration of the context.

    Example:
        >>> with capture_all_output("logs/run1"):
        >>>     print("This will go to both console and log file")
        >>>     logger.info("So will this log message")
    """
    # Handle the output path
    if log_path.endswith(".txt") or log_path.endswith(".log"):
        output = log_path
    else:
        output = os.path.join(log_path, "log.txt")

    # Modify the path based on rank
    distributed_rank = rank
    if distributed_rank > 0:
        base, ext = os.path.splitext(output)
        output = f"{base}.rank{distributed_rank}{ext}"

    # Create directory if needed
    dirname = os.path.dirname(output)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)

    # Remove file if it already exists
    if os.path.exists(output):
        os.remove(output)
    # Open file for stdout/stderr
    log_file = open(output, "a", buffering=1)  # line-buffered

    # Create tee streams
    tee_stdout = TeeStream(sys.stdout, log_file)
    tee_stderr = TeeStream(sys.stderr, log_file)

    # Redirect sys
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    # Redirect logging with FileHandler
    logger_handler = logging.FileHandler(output)
    # Set level based on rank
    if distributed_rank > 0:
        logger_handler.setLevel(logging.WARNING)
    else:
        logger_handler.setLevel(logging.DEBUG)

    logger_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%m/%d %H:%M"))

    # Attach handler to both root logger and focoos logger
    logging.getLogger().addHandler(logger_handler)

    # Add handler to focoos logger as well
    focoos_logger = get_logger()
    focoos_logger.addHandler(logger_handler)

    try:
        yield  # Enter block
    finally:
        # Restore
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        # Close logger handler
        logging.getLogger().removeHandler(logger_handler)

        # Remove handler from focoos logger
        focoos_logger = get_logger()
        focoos_logger.removeHandler(logger_handler)

        logger_handler.close()

        # Close file
        log_file.close()
