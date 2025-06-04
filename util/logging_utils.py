"""Utility helpers for application wide logging."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FILE = Path("robot.log")


def get_robot_logger(name: str | None = None) -> logging.Logger:
    """Return a logger configured to log robot events to a file."""
    logger = logging.getLogger(name if name else "robot")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    _LOG_FILE.parent.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=1_000_000, backupCount=3
    )
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


def warn_if_overrun(name: str, elapsed: float, expected: float) -> None:
    """Log a warning if elapsed time exceeds expected period."""
    if elapsed > expected:
        over_by = elapsed - expected
        logger = get_robot_logger(__name__)
        logger.warning(
            "%s overrun by %.4fs (elapsed %.4fs, expected %.4fs)",
            name,
            over_by,
            elapsed,
            expected,
        )
