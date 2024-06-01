import logging
import json

##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f)

def get_log_level(level_str):
    levels = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    return levels.get(level_str.upper(), logging.INFO)  # Default to INFO if level_str is not found


def setup_logging(module):
    log_level_str = config.get('log_level', 'INFO')  # Default to INFO if not specified
    log_level = get_log_level(log_level_str)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create console handler and set level to log_level
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Create file handler and set level to log_level
    fh = logging.FileHandler('app.log', mode='w')
    fh.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add formatter to ch and fh
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    
    # Add ch and fh to logger
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger

logger = setup_logging(__name__)