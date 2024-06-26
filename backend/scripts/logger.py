import logging
from datetime import datetime
import os


def setup_logging():
    logger = logging.getLogger("brain_tumor_api")  # Use a unique name for your logger
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Prevent logging from propagating to the root logger
        logger.propagate = False

        # Create handler for combined logging
        file_path = './logs.log'  # Save logs in the same directory
        combined_handler = logging.FileHandler(file_path)
        combined_handler.setLevel(logging.INFO)  # Capture all logs above INFO level

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        combined_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(combined_handler)

    return logger

