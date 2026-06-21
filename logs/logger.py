"""
    logs / logger.py
    ----------------
"""
from __future__ import annotations

import atexit
import logging
import sys
import time

from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Lock
from typing import Any, Dict, List

from logging.handlers import QueueHandler, QueueListener,RotatingFileHandler
from logs.formatters import JsonFormatter, ConsoleFormatter



class LogLevel(Enum):
    DEBUG       = logging.DEBUG
    INFO        = logging.INFO
    WARNING     = logging.WARNING
    ERROR       = logging.ERROR
    CRITICAL    = logging.CRITICAL



class Logger:
    """
    """
    _instances: Dict[str, "Logger"] = {}
    _instance_lock = Lock()

    _queue: Queue | None = None
    _listener: QueueListener | None = None
    _handlers: List[logging.Handler] = []

    _initialized = False
    _shutdown = False

    _init_lock = Lock()


    def __new__(cls, name: str, *args, **kwargs):
        with cls._instance_lock:

            if name in cls._instances:
                return cls._instances[name]

            instance = super().__new__(cls)
            cls._instances[name] = instance
            
            return instance
    

    def __init__(self,
        name: str, level: LogLevel = LogLevel.INFO, use_console: bool = True,
        to_disk: bool = True, max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5, directory: str | Path = "logs"
    ):
        """
            Initialize Logger Instance
        """
        if getattr(self, "_configured", False):
            return

        self._configured = True

        if Logger._shutdown:
            raise RuntimeError("Logging system has already been shut down")

        self.name = name

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level.value)
        self.logger.propagate = False

        Logger._ensure_infrastructure(
            directory=self.directory,
            use_console=use_console,
            to_disk=to_disk,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )

        if not any(isinstance(h, QueueHandler) for h in self.logger.handlers):
            self.logger.addHandler(QueueHandler(Logger._queue))
    

    @classmethod
    def _ensure_infrastructure(cls, *,
        directory: Path, use_console: bool, to_disk: bool,
        max_bytes: int, backup_count: int,
    ):
        """
        """
        if cls._initialized:
            return

        with cls._init_lock:
            
            if cls._initialized:
                return

            cls._queue = Queue(maxsize=10_000)
            handlers = []

            if to_disk:
                logfile = directory / "app.log"

                file_handler = RotatingFileHandler(
                    filename=logfile,
                    maxBytes=max_bytes,
                    backupCount=backup_count,
                    encoding="utf-8",
                )

                file_handler.setFormatter(JsonFormatter())
                handlers.append(file_handler)

            if use_console:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setFormatter(ConsoleFormatter())
                handlers.append(console_handler)

            cls._handlers = handlers
            cls._listener = QueueListener(
                cls._queue, *handlers, respect_handler_level=True,
            )

            cls._listener.start()
            atexit.register(cls.shutdown)

            cls._initialized = True
    

    def log(self, level: LogLevel | int, msg: str, *args, **extra):
        lvl = level.value if isinstance(level, LogLevel) else int(level)
        self.logger.log(lvl, msg, *args, extra=extra, stacklevel=3)


    def debug(self, msg, *args, **extra):
        self.log(LogLevel.DEBUG, msg, *args, **extra)

    def info(self, msg, *args, **extra):
        self.log(LogLevel.INFO, msg, *args, **extra)

    def warning(self, msg, *args, **extra):
        self.log(LogLevel.WARNING, msg, *args, **extra)

    def error(self, msg, *args, **extra):
        self.log(LogLevel.ERROR, msg, *args, **extra)

    def critical(self, msg, *args, **extra):
        self.log(LogLevel.CRITICAL, msg, *args, **extra)

    def exception(self, msg: str, *args, **extra):
        self.logger.exception(msg, *args, extra=extra, stacklevel=2)
    

    @contextmanager
    def time_block(self,label: str, level: LogLevel = LogLevel.INFO):
        start = time.perf_counter()

        try:
            yield

        finally:
            elapsed = (time.perf_counter() - start)
            self.log(level,"%s completed in %.6fs",label,elapsed)


    @contextmanager
    def catch(self, message: str):
        try:
            yield

        except Exception:
            self.exception(message)
            raise
    

    @classmethod
    def get_logger(cls, name: str = "app", **kwargs) -> "Logger":
        return cls(name, **kwargs)

    @classmethod
    def shutdown(cls):
        if cls._shutdown:
            return

        cls._shutdown = True

        if cls._listener:
            cls._listener.stop()

        for handler in cls._handlers:
            try:
                handler.close()
            except Exception:
                pass

        logging.shutdown()
