# a user can manipulate/configure logger from project libraries like any other Python object

import logging
from logging import Formatter, NullHandler
import colorama
from colorama import Back, Fore, Style

COLORS = {"DEBUG": Fore.BLUE,
          "INFO": Fore.BLACK,
          "WARNING": Fore.YELLOW,
          "ERROR": Fore.RED,
          "CRITICAL": Fore.MAGENTA}

class CustomColoredFormatter(Formatter):
    def __init__(self, *, format, use_color):
        Formatter.__init__(self, fmt=format)
        self.use_color = use_color

    def format(self, record):
        msg = super().format(record)
        if self.use_color:
            levelname = record.levelname
            if hasattr(record, "color"):
                return f"{record.color}{msg}{Style.RESET_ALL}"
            if levelname in COLORS:
                return f"{COLORS[levelname]}{msg}{Style.RESET_ALL}"
        return msg

    
logging.getLogger(__name__).addHandler(NullHandler())
