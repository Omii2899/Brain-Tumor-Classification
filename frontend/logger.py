import logging
from datetime import datetime
import os

def setup_logging():
    # Create logger
    logger = logging.getLogger("unique")
    logger.setLevel(logging.INFO)  # Set level to INFO to capture both info and error messages

    # Create a file handler
    file_handler = logging.FileHandler('logs.log')
    file_handler.setLevel(logging.INFO)

    # Create a console handler for output to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
