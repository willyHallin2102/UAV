"""
    logs / formatters.py
    --------------------
    The formatters are aiming for minimal allocations and is without stacking
    inspections and instead rely on python's built in logging which already is
    sufficiently efficient. JSON serialization is conducted using orjson for 
    fast binary operations.
"""
from __future__ import annotations

import os
import logging
import orjson

from datetime import datetime
from logging import LogRecord



class BaseFormatter(logging.Formatter):
    __slots__ = ()

    @staticmethod
    def format_timestamp(created: float) -> str:
        return (
            datetime.utcfromtimestamp(created)
            .isoformat(timespec="milliseconds")
            + "Z"
        )



class JsonFormatter(BaseFormatter):
    """
    """
    __slots__ = ("pid",)

    def __init__(self):
        """
            Initialize JSON - Formatter Instance
        """
        super().__init__()
        self.pid = os.getpid()
    

    def format(self, record: LogRecord) -> str:
        payload = {
            "timestamp": self.format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),

            # caller information
            "module": record.module,
            "function": record.funcName,
            "file": record.filename,
            "line": record.lineno,

            # process/thread
            "pid": self.pid,
            "thread": record.threadName,
        }

        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
        }

        for key, value in record.__dict__.items():

            if key not in reserved and key not in payload:
                payload[key] = value
        
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        
        return orjson.dumps(
            payload, option=orjson.OPT_APPEND_NEWLINE
        ).decode("utf-8")



class ConsoleFormatter(BaseFormatter):
    """"""
    __slots__ = ()
    COLORS = {
        "DEBUG"     : "\033[96m",   # Cyan
        "INFO"      : "\033[94m",   # Blue
        "WARNING"   : "\033[93m",   # Yellow
        "ERROR"     : "\033[91m",   # Red
        "CRITICAL"  : "\033[1;41m", # Bold White on Red
        "RESET"     : "\033[0m",    # No Color
    }

    def format(self, record: LogRecord) -> str:
        """"""
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        message = record.getMessage().replace("\n", "\\n",)
        return (
            f"{color}{record.levelname:<8}{reset} | "
            f"{record.filename}:{record.lineno} | "
            f"{record.funcName} | "
            f"{message}"
        )
