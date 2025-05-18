"""
Custom exceptions for the Java CSS Optimizer.
"""

class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass

class FileSizeException(Exception):
    """Exception raised when a file exceeds the maximum size limit."""
    pass

class DirectorySizeException(Exception):
    """Exception raised when a directory exceeds the maximum size limit."""
    pass

class FileCountException(Exception):
    """Exception raised when a directory contains too many files."""
    pass

class CSSRuleLimitException(Exception):
    """Exception raised when too many CSS rules are generated."""
    pass

class JavaBehaviorLimitException(Exception):
    """Exception raised when too many Java behaviors are generated."""
    pass

class LockAcquisitionException(Exception):
    """Exception raised when a file lock cannot be acquired."""
    pass

class BackupException(Exception):
    """Exception raised when file backup operations fail."""
    pass

class ContentGenerationException(Exception):
    """Exception raised when content generation fails."""
    pass

class AnalysisException(Exception):
    """Exception raised when file analysis fails."""
    pass 