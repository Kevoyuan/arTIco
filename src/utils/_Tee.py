import sys
from logging import Logger
from pathlib import Path
from typing import Union
import logging

src_dir = str(Path(__file__).absolute().parents[2])
if src_dir not in set(sys.path):
    sys.path.append(src_dir)
del src_dir
import src.utils._custom_log as custom_log


class _Logger(object):
    def __init__(self, fpath: Path):
        """Copies stream to stream and file

        Args:
            fpath (Path): path to log file
        """
        self.__terminal = sys.stdout
        self.__fpath = fpath

    def write(self, message: str):
        """Write to file and out stream

        Args:
            message (str): from std stream
        """
        with open(self.__fpath, "a", encoding="UTF-32") as log:
            self.__terminal.write(message)
            log.write(message)

    def flush(self):
        """Flush stream"""
        self.__terminal.flush()


class Tee(object):
    def __init__(self, fpath: Path, log: Union[Logger, None] = None):
        """Copies stdout / stderr to file and output

        Args:
            fpath (Path): path to log file
        """
        self.__log: Logger = custom_log.init_logger(log_lvl=10) if log is None else log
        # preserve stream for reset
        self.__stdout_orig = sys.stdout
        self.__stderr_orig = sys.stderr

        # clean log file
        if fpath.is_file():
            fpath.unlink()
        self.__log_path = fpath

        self.__f_handler: Union[None, logging.FileHandler] = None

    def __enter__(self):
        """with call, set streams on copy"""
        self.__log.info("Write STD stream additionally in %s", self.__log_path)
        sys.stdout = _Logger(fpath=self.__log_path)
        sys.stderr = sys.stdout

        has_stream = any(
            [
                isinstance(handler, logging.FileHandler) and handler.baseFilename == str(self.__log_path.absolute())
                for handler in self.__log.handlers
            ]
        )
        if has_stream:
            self.__log.debug("Logger streams already in %s", self.__log_path)
        else:
            self.__log.debug("Start additional logger stream into %s", self.__log_path)
            self.__f_handler = custom_log.file_handler(fpath=self.__log_path)
            self.__log.addHandler(self.__f_handler)

    def __exit__(self, type, value, traceback):
        """Reset on original stream handler

        Args:
            type (_type_): default from context manager
            value (_type_): default from context manager
            traceback (_type_): default from context manager
        """
        sys.stdout = self.__stdout_orig
        sys.stderr = self.__stderr_orig
        self.__log.removeHandler(self.__f_handler)
        self.__f_handler.close()

        self.__log.info("Switch back STD stream")


def test():
    """Example"""
    print("Before")
    log2 = logging.getLogger()
    log2.setLevel(10)
    log2.addHandler(logging.StreamHandler(stream=sys.stdout))
    log2.info("Test2")

    log = custom_log.init_logger(log_lvl=10)

    with Tee(fpath=Path("test.log"), log=log):
        print("i2n")
        log.debug("ho")
        log.info("hi")
        log3 = logging.getLogger()
        log3.setLevel(10)
        log3.addHandler(logging.StreamHandler(stream=sys.stdout))
        log3.info("Test3")
        log.warning("sd")
        log.error("er")
        log.critical("dsgf")

    print("Out")
    log.info("Done")


if __name__ == "__main__":
    test()
