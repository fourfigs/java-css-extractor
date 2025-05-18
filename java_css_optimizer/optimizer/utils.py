"""
Utility functions for the Java CSS Optimizer.
"""

import signal
import time
import logging
from contextlib import contextmanager
from typing import Callable, Any
from functools import wraps
import os
import shutil
from pathlib import Path

from java_css_optimizer.optimizer.exceptions import TimeoutException
from java_css_optimizer.optimizer.constants import (
    MAX_RETRIES, RETRY_DELAY, BACKUP_SUFFIX, ENCODING
)

logger = logging.getLogger(__name__)

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutException("Operation timed out")

@contextmanager
def operation_timeout(timeout: int):
    """Context manager for operation timeout."""
    # Set the alarm
    signal.alarm(timeout)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def retry_operation(operation: Callable, *args, **kwargs) -> Any:
    """Retry an operation with exponential backoff."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES):
        try:
            return operation(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (2 ** attempt))
                continue
            raise last_exception

def create_backup(file_path: Path) -> Path:
    """Create a backup of a file."""
    backup_path = file_path.with_suffix(file_path.suffix + BACKUP_SUFFIX)
    try:
        shutil.copy2(file_path, backup_path)
        return backup_path
    except Exception as e:
        logger.error(f"Error creating backup of {file_path}: {str(e)}")
        raise

def restore_backup(backup_path: Path, original_path: Path) -> None:
    """Restore a file from backup."""
    try:
        shutil.copy2(backup_path, original_path)
    except Exception as e:
        logger.error(f"Error restoring backup {backup_path}: {str(e)}")
        raise

def safe_write_file(file_path: Path, content: str) -> None:
    """Safely write content to a file with backup."""
    backup_path = None
    try:
        # Create backup if file exists
        if file_path.exists():
            backup_path = create_backup(file_path)
        
        # Write new content
        with open(file_path, 'w', encoding=ENCODING) as f:
            f.write(content)
        
        # Remove backup if write was successful
        if backup_path and backup_path.exists():
            backup_path.unlink()
            
    except Exception as e:
        # Restore from backup if write failed
        if backup_path and backup_path.exists():
            restore_backup(backup_path, file_path)
        raise

def ensure_directory(path: Path) -> None:
    """Ensure a directory exists."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {path}: {str(e)}")
        raise

def get_file_size(file_path: Path) -> int:
    """Get the size of a file in bytes."""
    try:
        return file_path.stat().st_size
    except Exception as e:
        logger.error(f"Error getting size of {file_path}: {str(e)}")
        raise

def get_directory_size(directory: Path) -> int:
    """Get the total size of all files in a directory."""
    try:
        return sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())
    except Exception as e:
        logger.error(f"Error getting size of directory {directory}: {str(e)}")
        raise

def count_files(directory: Path, pattern: str = "*.java") -> int:
    """Count files in a directory matching a pattern."""
    try:
        return len(list(directory.rglob(pattern)))
    except Exception as e:
        logger.error(f"Error counting files in {directory}: {str(e)}")
        raise

def cleanup_temp_files(directory: Path) -> None:
    """Clean up temporary files in a directory."""
    try:
        if directory.exists():
            shutil.rmtree(directory)
    except Exception as e:
        logger.error(f"Error cleaning up temporary directory {directory}: {str(e)}")
        raise 