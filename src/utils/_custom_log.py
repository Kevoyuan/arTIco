import logging
import sys
from pathlib import Path
from typing import Literal

FORMAT_STR = "%(asctime)s %(processName)-30s %(threadName)-30s %(filename)-45s %(levelname)-19s %(message)s"


class Colors:
    """https://gist.github.com/rene-d/9e584a7dd2935d0f461904b9f2950007
    ANSI color codes"""

    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"


class ColorFormatter(logging.Formatter):
    # Change this dictionary to suit your coloring needs!
    COLORS = {
        "DEBUG": Colors.DARK_GRAY,
        "INFO": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.PURPLE,
        "CRITICAL": Colors.LIGHT_RED,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Adds ANSI color codes to log message

        Args:
            record (logging.LogRecord): log record

        Returns:
            str: formatted log message
        """
        color = self.COLORS.get(record.levelname, "")
        if color:
            record.name = color + record.name + Colors.END
            record.levelname = color + record.levelname + Colors.END
            record.msg = color + record.msg + Colors.END
            record.threadName = color + record.threadName + Colors.END
            record.processName = color + record.processName + Colors.END
            record.filename = color + record.filename + Colors.END
        return logging.Formatter.format(self, record)


class Plain(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        """Removes ANSI color codes from log message

        Args:
            record (logging.LogRecord): log record

        Returns:
            str: unformatted log message
        """
        for col in (Colors.__dict__[c] for c in Colors.__dict__ if not c.startswith("__") and not c.endswith("__")):
            record.name = record.name.replace(col, "")
            record.levelname = record.levelname.replace(col, "")
            record.msg = record.msg.replace(col, "")
            record.threadName = record.threadName.replace(col, "")
            record.processName = record.processName.replace(col, "")
            record.filename = record.filename.replace(col, "")
        return logging.Formatter.format(self, record)


def init_logger(
    log_lvl: Literal[10, 20, 30, 40, 50] = 10, start_msg: str = "Start Log Stream", hard_reset: bool = False
) -> logging.Logger:
    """Initialize logging

    Args:
        log_lvl (Literal[10, 20, 30, 40, 50], optional): log level. Defaults to 10.
        start_msg (str, optional): Init message. Defaults to "START".
        hard_reset (bool, optional): Reset all root log handlers. Defaults to False.

    Returns:
        logging.Logger: logger
    """
    global FORMAT_STR

    # set logger
    logger = logging.getLogger()
    logger.setLevel(log_lvl)

    # check existing
    if hard_reset:
        logger.handlers.clear()
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                logger.debug("Stream logger already initialized - override")
                logger.removeHandler(handler)

    # create new
    console = logging.StreamHandler(stream=sys.stdout)
    color_formatter = ColorFormatter(FORMAT_STR)
    console.setFormatter(color_formatter)
    logger.addHandler(console)
    logger.propagate = False
    logger.info(start_msg)

    return logger


def file_handler(fpath: Path) -> logging.FileHandler:
    """Create file handler

    Args:
        fpath (Path): Path to log file

    Returns:
        logging.FileHandler: file handler
    """

    global FORMAT_STR
    f_handler = logging.FileHandler(filename=fpath, encoding="ascii")
    formatter = Plain(FORMAT_STR)
    f_handler.setFormatter(formatter)

    return f_handler
