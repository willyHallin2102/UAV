"""
    logs/logger.py
    --------------
    This module contains a unified logging setup for this project. 
    It ensures that all logs messages are written in console and 
    rotating logs files, under `<project_root/logs/history>/...`

    Features:
    ---------
    - Console + rotating file logging
    - timestamped default filenames
    - automatic directory creation within `../history`
    - avoid duplicating logs automatically (regardless of recalls)

    Examples:
    ---------
        >>> from logs.logger import setup_logger, get_logger
        >>> setup_logging("directory", logger_level="DEBUG")
        >>> logger = get_logger(__name__)
        >>> log.info("Data pipeline started")
        >>> log.debug("Custom debug message `here`")
    
    Logs will always be stored under:
        `<project_root>/logs/history/directory/run_YYYYMMDD_HHMMSS.log`
"""

import logging
from logging.handlers import RotatingFileHandler

import sys
import os

from pathlib import Path
from datetime import datetime
from typing import Optional, Union


# Root directory: where THIS file (logs.logger.py) is located
ROOT_DIRECTORY  : Path = Path(__file__).resolve().parent
DIRECTORY       : Path = ROOT_DIRECTORY / "history"


def setup_logging(
    logger_directory    : Union[str, Path] = "dir",
    logger_level        : str   = "INFO",
    logger_name         : Optional[str] = None,
    max_bytes           : int   = 5 * 1024 * 1024,  # 5MB
    backup_count        : int   = 5
) -> logging.Logger:
    """
        Configure logging with console + rotating file handlers.
        All logs are written to:
            `<logs/logger.py>/history/<logger_directory>`
        
        Args:
            logger_directory:   sub-directory under `history/` 
                                where logs will be stored.
            logger_level:   Logging verbosity . One of
                            `"DEBUG"`, ..., `"CRITICAL"`
            logger_name:    User-defined filename for the log
                            if None, a timestamped name 
                            (`run_YYYYMMDD_HHMMSS.log`) will
                            be generated.
            max_bytes:  Maximum size (bytes) of each log file
                        before rotation begins.
            backup_count:   Number of maximal rotational log
                            files to keep, then it drops logs.
        
        Returns:
        --------
        logging.Logger: The logger instance configured with handlers
    """
    # Ensure logs go into history/<logger_directory>
    logger_directory = DIRECTORY / Path(logger_directory).name
    os.makedirs(logger_directory, exist_ok=True)

    if logger_name is None:
        logger_name = f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logger_file = logger_directory / logger_name

    # Format for logging messages
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(logger_file, maxBytes=max_bytes,
                                       backupCount=backup_count)
    file_handler.setFormatter(formatter)

    # Logger 
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, logger_level.upper(), logging.INFO))
    logger.handlers.clear() # In case Recall, don't duplicate logs
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialized. Log File: `{logger_file}`")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
        Retrieve a logger for a specific module.

        Typically called inside each script/module:
        Example:
        --------
            >>> from logs.logger import get_logger
            >>> log = get_logger(__name__)
            >>> log.debug("Debugging this module...")

        Args:
        -----
        name:   The module name, usually `__name__`.

        Returns:
        --------
            logging.Logger: Logger scoped to the provided name.
        --------
    """
    return logging.getLogger(name)
