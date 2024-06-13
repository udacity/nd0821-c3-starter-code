import json
import logging

# Load config.json and correct path variable
with open("config.json", "r") as f:
    config = json.load(f)


def get_log_level(level_str):
    levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    # Default to INFO if level_str is not found
    return levels.get(level_str.upper(), logging.INFO)


def setup_logging(module):
    # Default to INFO if not specified
    log_level_str = config.get("log_level", "INFO")
    log_level = get_log_level(log_level_str)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Create console handler and set level to log_level
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)

    # Create file handler and set level to log_level
    file_handler = logging.FileHandler("app.log", mode="w")
    file_handler.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add formatter to ch and fh
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add ch and fh to logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


logger = setup_logging(__name__)
