"""
    logs/logger.py
    --------------
    Logger is intended as abstraction from redefine logging 
    information within multiple scripts, instead this is 
    aiming to provide a consistent logging structure throughout
    the project.
"""
import logging
import sys
import os

from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional, Union



def setup_logging(
    logger_directory    : Union[str, Path] = "logs",
    logger_level        : str = "INFO",
    logger_name         : Optional[str] = None,
    max_bytes           : int = 5 * 1024 * 1024, # 5MB
    backup_count        : int = 5
) -> logging.Logger:
    """
        Configure logging with console + rotating file handlers.

        Args:
        -----
            logging_directory:  Directory where the logs are stored
            logging_level:  Logging-Level, (_verbosity_) for the 
                            amount of feedback provided, such as
                            (`"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`).
            logging_name:   Optional custom log filename. If default at 
                            `None`, the timestamped file will be used.
            max_bytes:  Max size (bytes) per log file before it start to
                        rotate.
            backup_count:   Number of rotated logs files to keep.
        
        Returns:
            logger: Configured logger.
    """
    os.makedirs(logger_directory, exist_ok=True)
    if logger_name is None: 
        logger_name = f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger_file = Path(logger_directory) / logger_name

    # Format for logging messages
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handlers
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Rotating File Handler --
    file_handler = RotatingFileHandler(
        logger_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setFormatter(formatter)

    # Logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, logger_level.upper(), logging.INFO))
    
    # This is essential for avoid duplicate logs, if re-called
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging Initialized. Log File: `{logger_file}`")
    return logger


def get_logger(name: str) -> logging.Logger:
    """
        Retrieve a logger for a specific module.
        Example:
        --------
            >>> logger = get_logger(__name__)
        --------
        This `(__name__)` refer to the module, hence the logger 
        getting defined based on the module name.
    """
    return logging.getLogger(name)

