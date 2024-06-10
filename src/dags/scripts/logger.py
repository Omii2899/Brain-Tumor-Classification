import logging
from datetime import datetime

def setup_logging():
    
    
    logger = logging.getLogger(__name__)
    
    # Set level to INFO to capture both info and error messages
    logger.setLevel(logging.INFO) 
    
    if not logger.handlers:
        # Get the current date and time
        now = datetime.now()

        # Format the datetime object to a string
        timestamp_str = now.strftime("%Y_%m_%d_%H_%M_%S")

        log_file_name = 'log_'+ timestamp_str
        # Create handler for combined logging
        combined_handler = logging.FileHandler('./src/dags/'+log_file_name+'.log')
        combined_handler.setLevel(logging.INFO)  # Capture all logs above INFO level

        # Create formatter and add it to the handler
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        combined_handler.setFormatter(formatter)

        # Add handler to the logger
        logger.addHandler(combined_handler)

    return logger
