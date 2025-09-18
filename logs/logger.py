"""
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
        Configurations logging with console + rotating files 
        to separate logs within `backup_count` files to 
        sequentially save the logs. The logs will always be 
        stored in `<logs/logger.py>/"history/..."`.

        Args:
        -----
            logger_directory:   Directory assign to within 
            logger_level:   Logging verbosity, the amount of user
                            feedback the logger should return during
                            runtime, eligible options,
                            "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            logger_name:    Custom log-name. If None, the timestamp will be
                            assign.
            max_bytes:  Max size (bytes) per log file before rotation.
            backup_count:   Number of rotating logs files to keep.
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
    # """
    #     Retrieve a logger for a specific module.
    # """
    return logging.getLogger(name)
