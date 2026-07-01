import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logger(name: str, log_dir: Path, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / f"{name}.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


app_logger = None
indexing_logger = None


def get_app_logger():
    global app_logger
    if app_logger is None:
        from app.config import config
        app_logger = setup_logger("app", config.logs_dir)
    return app_logger


def get_indexing_logger():
    global indexing_logger
    if indexing_logger is None:
        from app.config import config
        indexing_logger = setup_logger("indexing", config.logs_dir)
    return indexing_logger
