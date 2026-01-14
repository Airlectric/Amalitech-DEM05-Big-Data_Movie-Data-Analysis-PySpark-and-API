import logging
from logging.handlers import RotatingFileHandler
import os

# Project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Log directory
log_dir = os.path.join(project_root, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Logger creation
logger = logging.getLogger('movie_project_logger')
logger.setLevel(logging.DEBUG)  # capture all logs, handlers will filter

# Avoid duplicate logs
if not logger.handlers:

    # Console handler (all logs)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
    )
    console.setFormatter(console_formatter)
    logger.addHandler(console)

    # File handlers for specific levels
    level_config = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
    }

    for name, level in level_config.items():
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"),
            maxBytes=2_000_000,
            backupCount=5
        )
        file_handler.setLevel(level)
        # filter to only log the specific level (INFO, WARNING, ERROR)
        file_handler.addFilter(lambda record, lvl=level: record.levelno == lvl)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
