"""Logging utility - records logs to file and stdout for future analysis."""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_FILE = LOG_DIR / os.getenv("LOG_FILE", "stock_analysis.log")
MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", "5242880"))  # 5 MB default
BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "5"))

_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with file and console handlers."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    formatter = logging.Formatter(_FORMAT)
    logger.setLevel(level)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (rotating) for future analysis
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError as e:
        logger.warning("Could not create log file %s: %s. Logging to console only.", LOG_FILE, e)

    return logger
