import logging
import os
from datetime import datetime
import yaml

def setup_logger(log_path='logs/traffic_analysis.log', log_level='INFO'):
    """Set up logger for the project"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Create a custom logger
    logger = logging.getLogger('traffic_analysis')
    
    # Set level based on parameter
    level = getattr(logging, log_level.upper())
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    
    # Create formatters and add it to handlers
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def get_logger():
    """Get the configured logger"""
    # Load config if exists
    config_path = 'config/config.yaml'
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            log_path = config['logging']['log_path']
            log_level = config['logging']['level']
    else:
        log_path = 'logs/traffic_analysis.log'
        log_level = 'INFO'
    
    # Check if logger is already configured
    logger = logging.getLogger('traffic_analysis')
    if not logger.handlers:
        logger = setup_logger(log_path, log_level)
    
    return logger