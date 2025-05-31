import logging
import os
import sys
import time
from typing import Optional
from .helpers import ensure_directory


class Logger:
    """Custom logger with file and console output"""
    
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    def __init__(self, name: str = "3DMesh", log_level: str = "INFO", 
                 log_to_file: bool = True, log_dir: str = "logs"):
        """Initialize logger with specified name and level"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.LEVELS.get(log_level.upper(), logging.INFO))
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self._get_formatter())
        self.logger.addHandler(console_handler)
        
        # Create file handler if requested
        if log_to_file:
            self.log_file = self._setup_log_file(log_dir)
            if self.log_file:
                file_handler = logging.FileHandler(self.log_file)
                file_handler.setFormatter(self._get_formatter())
                self.logger.addHandler(file_handler)
                
    def _get_formatter(self) -> logging.Formatter:
        """Create a formatter for log messages"""
        return logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
    def _setup_log_file(self, log_dir: str) -> Optional[str]:
        """Set up log file with timestamp in filename"""
        try:
            if not ensure_directory(log_dir):
                return None
                
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            log_file = os.path.join(log_dir, f"3dmesh_{timestamp}.log")
            return log_file
        except Exception as e:
            print(f"Error setting up log file: {str(e)}")
            return None
            
    def debug(self, message: str) -> None:
        """Log a debug message"""
        self.logger.debug(message)
        
    def info(self, message: str) -> None:
        """Log an info message"""
        self.logger.info(message)
        
    def warning(self, message: str) -> None:
        """Log a warning message"""
        self.logger.warning(message)
        
    def error(self, message: str) -> None:
        """Log an error message"""
        self.logger.error(message)
        
    def critical(self, message: str) -> None:
        """Log a critical message"""
        self.logger.critical(message)
        
    def exception(self, message: str) -> None:
        """Log an exception with traceback"""
        self.logger.exception(message)


# Create a global logger instance for easy import
app_logger = Logger()


# Convenience functions
def debug(message: str) -> None:
    """Log a debug message using the global logger"""
    app_logger.debug(message)
    
def info(message: str) -> None:
    """Log an info message using the global logger"""
    app_logger.info(message)
    
def warning(message: str) -> None:
    """Log a warning message using the global logger"""
    app_logger.warning(message)
    
def error(message: str) -> None:
    """Log an error message using the global logger"""
    app_logger.error(message)
    
def critical(message: str) -> None:
    """Log a critical message using the global logger"""
    app_logger.critical(message)
    
def exception(message: str) -> None:
    """Log an exception with traceback using the global logger"""
    app_logger.exception(message)
