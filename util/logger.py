import logging
from asyncio.tasks import create_task
from enum import Enum
from sys import exc_info
from typing import Optional, Union

from httpx import AsyncClient


class CriticalLevel(Enum):
    MODERATE = 1
    HIGH = 2
    WAKE_ME_UP = 99


class ANSIColors(str, Enum):
    BLACK = "\u001b[30m"
    RED = "\u001b[31m"
    RED_BOLD = "\x1b[31;1m"
    GREEN = "\u001b[32m"
    YELLOW = "\u001b[33m"
    BLUE = "\u001b[34m"
    MAGENTA = "\u001b[35m"
    CYAN = "\u001b[36m"
    WHITE = "\u001b[37m"
    RESET = "\u001b[0m"


def cprint(message: Union[str, Exception], color: ANSIColors):
    print(f"{color}{message}{ANSIColors.RESET}")


class CustomFormatter(logging.Formatter):

    custom_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: custom_format,
        logging.INFO: ANSIColors.CYAN + custom_format + ANSIColors.RESET,
        logging.WARNING: ANSIColors.YELLOW + custom_format + ANSIColors.RESET,
        logging.ERROR: ANSIColors.RED + custom_format + ANSIColors.RESET,
        logging.CRITICAL: ANSIColors.RED_BOLD + custom_format + ANSIColors.RESET,
    }

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    def __init__(self, log_level: int) -> None:
        self._logger = logging.getLogger("dlibin")
        self._logger.setLevel(log_level)
        sh = logging.StreamHandler()
        sh.setLevel(log_level)
        sh.setFormatter(CustomFormatter())
        self._logger.addHandler(sh)

    def debug(self, message: str):
        self._logger.debug(message)

    def info(self, message: str):
        self._logger.info(message)

    def warn(self, message: str):
        self._logger.warning(message)

    def error(self, message: Union[str, Exception]):
        self._logger.error(message, exc_info=exc_info() != (None, None, None))


log_level = logging.DEBUG
app_logger = Logger(log_level)
