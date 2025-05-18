#!/usr/bin/env python3
"""
Core optimization logic for Java CSS Optimizer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from functools import lru_cache
import threading
from collections import defaultdict
import time
import weakref
import queue
import os
import shutil
from contextlib import contextmanager
import signal
from typing import Callable

from java_css_optimizer.analyzer import JavaAnalyzer, StylePattern, AnalysisResult, ImportedFile
from java_css_optimizer.config import Config
from java_css_optimizer.optimizer.constants import (
    MAX_WORKERS, MAX_FILE_SIZE, OPERATION_TIMEOUT,
    CACHE_SIZE, MAX_RETRIES, RETRY_DELAY
)
from java_css_optimizer.optimizer.exceptions import TimeoutException
from java_css_optimizer.optimizer.utils import (
    timeout_handler, operation_timeout, retry_operation
)
from java_css_optimizer.optimizer.models import OptimizationResult
from java_css_optimizer.optimizer.generators import (
    CSSGenerator, JavaGenerator, ContentGenerator
)

class JavaOptimizer:
    """Core optimizer class that coordinates the optimization process."""
    
    def __init__(self, config: Config, max_workers: int = MAX_WORKERS):
        self.config = config
        self.analyzer = JavaAnalyzer(max_workers=max_workers)
        self.logger = logging.getLogger(__name__)
        self.max_workers = min(max_workers, MAX_WORKERS)
        self._css_cache = weakref.WeakKeyDictionary()
        self._java_cache = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()
        self._file_locks = defaultdict(threading.Lock)
        self._temp_dir = Path("temp_optimizer")
        self._operation_timeout = OPERATION_TIMEOUT
        self._retry_count = 0
        self._processed_imports = set()
        
        # Initialize generators
        self.css_generator = CSSGenerator(config)
        self.java_generator = JavaGenerator(config)
        self.content_generator = ContentGenerator(config)
        
        # Set up signal handlers for timeouts
        signal.signal(signal.SIGALRM, timeout_handler)
    
    def __del__(self):
        """Clean up temporary files."""
        try:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    @lru_cache(maxsize=CACHE_SIZE)
    def optimize_directory(self, directory: Path) -> List[OptimizationResult]:
        """Optimize all Java files in a directory."""
        results = []
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Check directory size
            total_size = sum(f.stat().st_size for f in directory.rglob("*.java"))
            if total_size > MAX_FILE_SIZE:
                warnings.append(f"Directory size ({total_size} bytes) exceeds maximum limit")
            
            # Collect all Java files
            java_files = list(directory.rglob("*.java"))
            
            # Process files in parallel with timeout
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_file = {
                    executor.submit(retry_operation, self.optimize_file, java_file): java_file
                    for java_file in java_files
                }
                
                for future in as_completed(future_to_file):
                    java_file = future_to_file[future]
                    try:
                        with operation_timeout(self._operation_timeout):
                            result = future.result()
                            results.append(result)
                            
                            # Process imported files
                            self._process_imported_files(result)
                            
                    except TimeoutError:
                        error_msg = f"Operation timed out for {java_file}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                    except Exception as e:
                        error_msg = f"Error optimizing {java_file}: {str(e)}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
            
            # Calculate directory-wide statistics
            total_time = time.time() - start_time
            total_changes = sum(r.changes_made for r in results)
            total_css_rules = sum(len(r.css_rules) for r in results if r.css_rules)
            
            self.logger.info(f"Directory optimization completed in {total_time:.2f}s")
            self.logger.info(f"Total changes: {total_changes}")
            self.logger.info(f"Total CSS rules: {total_css_rules}")
            
            if errors:
                self.logger.warning(f"Encountered {len(errors)} errors during optimization")
            if warnings:
                self.logger.warning(f"Encountered {len(warnings)} warnings during optimization")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error optimizing directory {directory}: {str(e)}")
            raise
    
    def _process_imported_files(self, result: OptimizationResult) -> None:
        """Process imported files for optimization."""
        try:
            for imported in result.imported_files:
                if imported.path in self._processed_imports:
                    continue
                
                # Check if imported file needs optimization
                if self._needs_optimization(imported):
                    # Optimize the imported file
                    imported_result = self.optimize_file(imported.path)
                    
                    # Update the original result with imported file changes
                    self._update_result_with_imports(result, imported_result)
                    
                    # Mark as processed
                    self._processed_imports.add(imported.path)
                    
        except Exception as e:
            self.logger.error(f"Error processing imported files: {str(e)}")
    
    def _needs_optimization(self, imported: ImportedFile) -> bool:
        """Check if an imported file needs optimization."""
        try:
            # Check if file has style patterns
            if not imported.style_patterns:
                return False
            
            # Check if file has behaviors that can be converted to CSS
            for pattern in imported.style_patterns:
                if pattern.css_property and pattern.css_value:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking optimization needs: {str(e)}")
            return False
    
    def _update_result_with_imports(self, result: OptimizationResult, imported_result: OptimizationResult) -> None:
        """Update optimization result with imported file changes."""
        try:
            # Add imported CSS rules
            result.css_rules.extend(imported_result.css_rules)
            
            # Add imported Java behaviors
            if imported_result.java_behaviors:
                if not result.java_behaviors:
                    result.java_behaviors = []
                result.java_behaviors.extend(imported_result.java_behaviors)
            
            # Update changes count
            result.changes_made += imported_result.changes_made
            
            # Merge errors and warnings
            result.errors.extend(imported_result.errors)
            result.warnings.extend(imported_result.warnings)
            
        except Exception as e:
            self.logger.error(f"Error updating result with imports: {str(e)}")
    
    def optimize_file(self, file_path: Path) -> OptimizationResult:
        """Optimize a single Java file."""
        self.logger.info(f"Optimizing {file_path}")
        start_time = time.time()
        errors = []
        warnings = []
        
        # Check file size
        try:
            file_size = file_path.stat().st_size
            if file_size > MAX_FILE_SIZE:
                warnings.append(f"File size ({file_size} bytes) exceeds maximum limit")
        except Exception as e:
            self.logger.warning(f"Error checking file size: {str(e)}")
        
        # Get file-specific lock with timeout
        lock = self._file_locks[file_path]
        if not lock.acquire(timeout=10):  # 10 second timeout for lock acquisition
            error_msg = f"Timeout acquiring lock for {file_path}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            return OptimizationResult(
                original_file=file_path,
                optimized_file=file_path,
                css_file=file_path.with_suffix('.css'),
                errors=errors,
                warnings=warnings
            )
        
        try:
            with operation_timeout(self._operation_timeout):
                # Analyze the file
                analysis = self.analyzer.analyze_file(file_path)
                
                # Read original content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Generate optimized content in parallel with timeout
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    optimized_future = executor.submit(
                        retry_operation,
                        self.content_generator.generate_optimized_content,
                        content,
                        analysis
                    )
                    css_future = executor.submit(
                        retry_operation,
                        self.css_generator.generate_css_content,
                        analysis
                    )
                    java_future = executor.submit(
                        retry_operation,
                        self.java_generator.generate_java_behaviors,
                        analysis
                    ) if analysis.requires_java else None
                    
                    try:
                        optimized_content = optimized_future.result(timeout=self._operation_timeout)
                        css_content = css_future.result(timeout=self._operation_timeout)
                        java_content = java_future.result(timeout=self._operation_timeout) if java_future else None
                    except TimeoutError:
                        error_msg = f"Operation timed out while generating content for {file_path}"
                        self.logger.error(error_msg)
                        errors.append(error_msg)
                        raise
                
                # Determine output paths
                output_dir = self.config.css_output or file_path.parent
                output_dir.mkdir(parents=True, exist_ok=True)
                
                css_file = output_dir / f"{file_path.stem}.css"
                optimized_file = output_dir / f"{file_path.stem}_optimized.java"
                java_file = output_dir / f"{file_path.stem}_behaviors.java" if java_content else None
                
                # Calculate optimization statistics
                original_size = len(content)
                optimized_size = len(optimized_content)
                css_size = len(css_content)
                java_size = len(java_content) if java_content else 0
                
                optimization_stats = {
                    'css_size_reduction': ((original_size - css_size) / original_size) * 100 if original_size > 0 else 0,
                    'java_size_reduction': ((original_size - java_size) / original_size) * 100 if original_size > 0 else 0,
                    'duplicate_rules_removed': self.css_generator.count_duplicate_rules(analysis.style_patterns)
                }
                
                # Save files with backup and retry
                retry_operation(self._save_file_with_backup, css_file, css_content)
                retry_operation(self._save_file_with_backup, optimized_file, optimized_content)
                if java_file:
                    retry_operation(self._save_file_with_backup, java_file, java_content)
                
                # Calculate performance metrics
                end_time = time.time()
                performance_metrics = {
                    'optimization_time': end_time - start_time,
                    'css_generation_time': css_future.result_time if hasattr(css_future, 'result_time') else 0,
                    'java_generation_time': java_future.result_time if java_future and hasattr(java_future, 'result_time') else 0
                }
                
                return OptimizationResult(
                    original_file=file_path,
                    optimized_file=optimized_file,
                    css_file=css_file,
                    java_file=java_file,
                    changes_made=len(analysis.style_patterns),
                    css_rules=[self.css_generator.pattern_to_css_rule(p) for p in analysis.style_patterns],
                    java_behaviors=[self.java_generator.behavior_to_java(b) for b in analysis.requires_java] if analysis.requires_java else None,
                    performance_metrics=performance_metrics,
                    optimization_stats=optimization_stats,
                    errors=errors,
                    warnings=warnings
                )
                
        except Exception as e:
            error_msg = f"Error optimizing {file_path}: {str(e)}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            raise
        finally:
            # Clean up file lock
            lock.release()
            self._file_locks.pop(file_path, None)
    
    def _save_file_with_backup(self, file_path: Path, content: str) -> None:
        """Save a file with backup."""
        backup_path = None
        try:
            # Create backup if file exists
            if file_path.exists():
                backup_path = file_path.with_suffix(file_path.suffix + '.bak')
                shutil.copy2(file_path, backup_path)
            
            # Write new content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Remove backup if write was successful
            if file_path.exists() and backup_path and backup_path.exists():
                backup_path.unlink()
                
        except Exception as e:
            # Restore from backup if write failed
            if backup_path and backup_path.exists():
                shutil.copy2(backup_path, file_path)
            raise 