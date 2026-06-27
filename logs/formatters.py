"""
    logs / formatters.py
    --------------------
"""
from __future__ import annotations

import os
import sys
import logging
import orjson
import inspect

from datetime import datetime
from logging import LogRecord



class BaseFormatter(logging.Formatter):
    """
    Base formatter with common utilities
    """
    __slots__ = ()


    @staticmethod
    def format_timestamp(created: float) -> str:
        """
        Format timestamp in ISO format with milliseconds and Z suffix
        """
        return (
            datetime.utcfromtimestamp(created)
            .isoformat(timespec="milliseconds") + "Z"
        )



class JsonFormatter(BaseFormatter):
    """
    JSON formatter for structured logging
    """
    __slots__ = ("pid",)


    def __init__(self):
        """
        Initialize the JSON formatter
        """
        super().__init__()  # Now calls logging.Formatter.__init__
        self.pid = os.getpid()


    def _get_caller_info(self, record: LogRecord):
        """
        Extract caller class and method information
        """
        class_name = getattr(record, 'class_name', None)
        method_name = getattr(record, 'method_name', None)

        # If not explicitly set, try and infer from the stack
        if class_name is None and method_name is None:
            
            # Get the frame that called the logging method
            # Skips frames: this format method and the logging method
            frame = inspect.currentframe()
            try:

                # Walk up the stack to find the caller
                # Skip the logger internal frames
                f = frame
                while f:

                    # Skip frames from the logging module and this module
                    if f.f_code.co_filename.endswith((
                        'logging/__init__.py', 'logging.py', 'formatters.py'
                    )):
                        f = f.f_back
                        continue

                    # Use the function name if method_name not found
                    if not method_name:
                        method_name = f.f_code.co_name
                    
                    f = f.f_back

            finally:
                del frame

        return class_name, method_name


    def format(self, record: LogRecord) -> str:
        """
        Format the log record as JSON
        """
        # Get class and method info
        class_name, method_name = self._get_caller_info(record)
        
        payload = {
            # Class / Method context
            "class": class_name or "unknown",
            "method": method_name or record.funcName,

            # Caller information
            "module": record.module,
            "function": record.funcName,
            "file": record.filename,
            "line": record.lineno,

            "timestamp": self.format_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),

            # Process / Thread
            "pid": self.pid,
            "thread": record.threadName,
        }

        # Reserved attributes that shouldn't be added to extra fields
        reserved = {
            "name", "msg", "args", "levelname", "levelno",
            "pathname", "filename", "module", "exc_info", "exc_text",
            "stack_info", "lineno", "funcName", "created", "msecs",
            "relativeCreated", "thread", "threadName", "processName",
            "process", "class_name", "method_name"
        }

        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in reserved and key not in payload:
                payload[key] = value

        # Add exception info if present
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return orjson.dumps(
            payload, option=orjson.OPT_APPEND_NEWLINE
        ).decode("utf-8")



class ConsoleFormatter(BaseFormatter):
    """
    Console formatter with colored output
    """
    __slots__ = ()
    COLORS = {
        "DEBUG"     : "\033[1;36m",         # Bold Cyan
        "INFO"      : "\033[1;34m",         # Bold Blue
        "WARNING"   : "\033[1;33m",         # Bold Yellow
        "ERROR"     : "\033[1;31m",         # Bold Red
        "CRITICAL"  : "\033[1;41m\033[37m", # Bold White on Red
        "RESET"     : "\033[0m"             # Reset the coloring
    }

    def format(self, record: LogRecord) -> str:
        """
        Format the log record for console output with colors
        """
        color = self.COLORS.get(record.levelname, "")
        reset = self.COLORS["RESET"]

        # Get class/method context
        class_name = getattr(record, 'class_name', '')
        method_name = getattr(record, 'method_name', record.funcName)

        context = f"{class_name}.{method_name}" if class_name else method_name

        # Escape newlines in the message
        message = record.getMessage().replace("\n", "\\n")

        return (
            f"{color}{record.levelname:<8}{reset} | "
            f"{record.filename}:{record.lineno} | "
            f"{context} | "
            f"{message}"
        )