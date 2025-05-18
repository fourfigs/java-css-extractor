"""
Constants for the Java CSS Optimizer.
"""

# Worker configuration
MAX_WORKERS = 8
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
OPERATION_TIMEOUT = 300  # 5 minutes
CACHE_SIZE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

# CSS configuration
CSS_INDENT = "    "  # 4 spaces
CSS_SELECTOR_SEPARATOR = ",\n"
CSS_RULE_SEPARATOR = "\n\n"

# Java configuration
JAVA_INDENT = "    "  # 4 spaces
JAVA_IMPORTS = [
    "javax.swing.*",
    "java.awt.*",
    "java.awt.event.*"
]

# File operations
BACKUP_SUFFIX = ".bak"
TEMP_DIR = "temp_optimizer"
ENCODING = "utf-8"

# Performance thresholds
MAX_DIRECTORY_SIZE = 100 * 1024 * 1024  # 100MB
MAX_FILES_PER_DIRECTORY = 1000
MAX_CSS_RULES_PER_FILE = 1000
MAX_JAVA_BEHAVIORS_PER_FILE = 100 