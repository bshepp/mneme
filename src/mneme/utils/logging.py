"""Logging utilities for Mneme."""

import logging
import sys
from typing import Optional, Union
from pathlib import Path
import os


def setup_logging(
    level: Union[str, int] = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    console: bool = True,
    rotation: bool = False,
    max_bytes: int = 10*1024*1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup logging configuration for Mneme.
    
    Parameters
    ----------
    level : str or int
        Logging level
    format_string : str, optional
        Custom format string
    log_file : str or Path, optional
        Path to log file
    console : bool
        Whether to log to console
    rotation : bool
        Whether to use rotating file handler
    max_bytes : int
        Maximum bytes per log file (for rotation)
    backup_count : int
        Number of backup files to keep
        
    Returns
    -------
    logger : logging.Logger
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('mneme')
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        if rotation:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get logger with specific name.
    
    Parameters
    ----------
    name : str
        Logger name
        
    Returns
    -------
    logger : logging.Logger
        Logger instance
    """
    return logging.getLogger(f'mneme.{name}')


class LoggerMixin:
    """Mixin class to add logging capabilities."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return get_logger(self.__class__.__name__)


def log_function_call(func):
    """Decorator to log function calls."""
    def wrapper(*args, **kwargs):
        logger = get_logger('function_calls')
        logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} returned successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {e}")
            raise
    
    return wrapper


def log_execution_time(func):
    """Decorator to log function execution time."""
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger('performance')
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"{func.__name__} executed in {execution_time:.3f}s")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"{func.__name__} failed after {execution_time:.3f}s: {e}")
            raise
    
    return wrapper


def configure_logging_from_config(config: dict) -> None:
    """
    Configure logging from configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Logging configuration
    """
    logging_config = config.get('logging', {})
    
    setup_logging(
        level=getattr(logging, logging_config.get('level', 'INFO')),
        format_string=logging_config.get('format'),
        log_file=logging_config.get('file'),
        console=logging_config.get('console', True),
        rotation=logging_config.get('rotation', {}).get('enabled', False),
        max_bytes=logging_config.get('rotation', {}).get('max_bytes', 10*1024*1024),
        backup_count=logging_config.get('rotation', {}).get('backup_count', 5)
    )


def set_log_level(level: Union[str, int]) -> None:
    """
    Set logging level for all Mneme loggers.
    
    Parameters
    ----------
    level : str or int
        Logging level
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    logger = logging.getLogger('mneme')
    logger.setLevel(level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


def create_experiment_logger(experiment_id: str, log_dir: Union[str, Path]) -> logging.Logger:
    """
    Create logger for specific experiment.
    
    Parameters
    ----------
    experiment_id : str
        Experiment identifier
    log_dir : str or Path
        Directory for log files
        
    Returns
    -------
    logger : logging.Logger
        Experiment logger
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(f'mneme.experiment.{experiment_id}')
    logger.setLevel(logging.INFO)
    
    # File handler for experiment
    log_file = log_dir / f"{experiment_id}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    return logger