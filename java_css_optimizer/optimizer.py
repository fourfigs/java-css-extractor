#!/usr/bin/env python3
"""
Java code optimizer that transforms UI styling to CSS.
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
import json
import csscompressor
import weakref
import queue
import os
import shutil
from contextlib import contextmanager
import signal
from typing import Callable

from java_css_optimizer.analyzer import JavaAnalyzer, StylePattern, AnalysisResult
from java_css_optimizer.config import Config

# Constants for timeouts and limits
MAX_WORKERS = 8
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
OPERATION_TIMEOUT = 300  # 5 minutes
CACHE_SIZE = 1000
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

@dataclass
class OptimizationResult:
    """Results of Java code optimization."""
    original_file: Path
    optimized_file: Path
    css_file: Path
    java_file: Optional[Path] = None
    changes_made: int = 0
    css_rules: List[Dict[str, Any]] = None
    java_behaviors: List[Dict[str, Any]] = None
    performance_metrics: Dict[str, float] = None
    optimization_stats: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.css_rules is None:
            self.css_rules = []
        if self.java_behaviors is None:
            self.java_behaviors = []
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.optimization_stats is None:
            self.optimization_stats = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
    
    def summary(self) -> str:
        """Generate a summary of the optimization results."""
        summary = [
            f"Optimization Summary:",
            f"  Original file: {self.original_file}",
            f"  Optimized file: {self.optimized_file}",
            f"  CSS file: {self.css_file}",
            f"  Changes made: {self.changes_made}",
            f"  CSS rules generated: {len(self.css_rules) if self.css_rules else 0}"
        ]
        
        if self.java_file:
            summary.extend([
                f"  Java behaviors file: {self.java_file}",
                f"  Java behaviors: {len(self.java_behaviors) if self.java_behaviors else 0}"
            ])
        
        if self.performance_metrics:
            summary.extend([
                f"  Performance Metrics:",
                f"    Optimization time: {self.performance_metrics.get('optimization_time', 0):.2f}s",
                f"    CSS generation time: {self.performance_metrics.get('css_generation_time', 0):.2f}s",
                f"    Java generation time: {self.performance_metrics.get('java_generation_time', 0):.2f}s"
            ])
        
        if self.optimization_stats:
            summary.extend([
                f"  Optimization Stats:",
                f"    CSS size reduction: {self.optimization_stats.get('css_size_reduction', 0)}%",
                f"    Java size reduction: {self.optimization_stats.get('java_size_reduction', 0)}%",
                f"    Duplicate rules removed: {self.optimization_stats.get('duplicate_rules_removed', 0)}"
            ])
        
        if self.errors:
            summary.extend([
                f"  Errors:",
                *[f"    - {error}" for error in self.errors]
            ])
        
        if self.warnings:
            summary.extend([
                f"  Warnings:",
                *[f"    - {warning}" for warning in self.warnings]
            ])
        
        return "\n".join(summary)

class TimeoutException(Exception):
    """Exception raised when an operation times out."""
    pass

def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutException("Operation timed out")

class JavaOptimizer:
    """Optimizes Java code by extracting styles to CSS."""
    
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
        
        # Create temp directory if it doesn't exist
        self._temp_dir.mkdir(exist_ok=True)
        
        # Load component-specific CSS templates
        self.component_css = self._load_component_templates()
        
        # Load behavior-specific CSS
        self.behavior_css = self._load_behavior_templates()
        
        # Set up signal handlers for timeouts
        signal.signal(signal.SIGALRM, timeout_handler)
    
    def __del__(self):
        """Clean up temporary files."""
        try:
            if self._temp_dir.exists():
                shutil.rmtree(self._temp_dir)
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary directory: {str(e)}")
    
    @contextmanager
    def _operation_timeout(self, timeout: int = None):
        """Context manager for operation timeout."""
        if timeout is None:
            timeout = self._operation_timeout
        
        # Set the alarm
        signal.alarm(timeout)
        try:
            yield
        finally:
            # Disable the alarm
            signal.alarm(0)
    
    def _retry_operation(self, operation: Callable, *args, **kwargs) -> Any:
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
                    executor.submit(self._retry_operation, self.optimize_file, java_file): java_file
                    for java_file in java_files
                }
                
                for future in as_completed(future_to_file):
                    java_file = future_to_file[future]
                    try:
                        with self._operation_timeout():
                            result = future.result()
                            results.append(result)
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
            with self._operation_timeout():
                # Analyze the file
                analysis = self.analyzer.analyze_file(file_path)
                
                # Read original content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Generate optimized content in parallel with timeout
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    optimized_future = executor.submit(self._retry_operation, self._generate_optimized_content, content, analysis)
                    css_future = executor.submit(self._retry_operation, self._generate_css_content, analysis)
                    java_future = executor.submit(self._retry_operation, self._generate_java_behaviors, analysis) if analysis.requires_java else None
                    
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
                    'duplicate_rules_removed': self._count_duplicate_rules(analysis.style_patterns)
                }
                
                # Save files with backup and retry
                self._retry_operation(self._save_file_with_backup, css_file, css_content)
                self._retry_operation(self._save_file_with_backup, optimized_file, optimized_content)
                if java_file:
                    self._retry_operation(self._save_file_with_backup, java_file, java_content)
                
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
                    css_rules=[self._pattern_to_css_rule(p) for p in analysis.style_patterns],
                    java_behaviors=[self._behavior_to_java(b) for b in analysis.requires_java] if analysis.requires_java else None,
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
    
    def _load_component_templates(self) -> Dict[str, Dict[str, str]]:
        """Load component-specific CSS templates."""
        return {
            'menu': {
                'base': """
                    display: flex;
                    flex-direction: column;
                    list-style: none;
                    padding: 0;
                    margin: 0;
                """,
                'item': """
                    padding: 8px 16px;
                    cursor: pointer;
                    transition: background-color 0.2s;
                """,
                'hover': """
                    &:hover {
                        background-color: #f0f0f0;
                    }
                """
            },
            'gallery': {
                'base': """
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 16px;
                    padding: 16px;
                """,
                'item': """
                    position: relative;
                    overflow: hidden;
                    border-radius: 4px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                """
            },
            'form': {
                'base': """
                    display: flex;
                    flex-direction: column;
                    gap: 16px;
                    padding: 16px;
                """,
                'input': """
                    padding: 8px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                """,
                'label': """
                    font-weight: bold;
                    margin-bottom: 4px;
                """
            },
            'navigation': {
                'base': """
                    display: flex;
                    align-items: center;
                    padding: 8px 16px;
                    background-color: #f8f8f8;
                    border-bottom: 1px solid #ddd;
                """,
                'item': """
                    padding: 8px 16px;
                    cursor: pointer;
                    border-radius: 4px;
                """
            },
            'modal': {
                'base': """
                    position: fixed;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    background: white;
                    padding: 24px;
                    border-radius: 8px;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                """,
                'overlay': """
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0,0,0,0.5);
                """
            }
        }
    
    def _load_behavior_templates(self) -> Dict[str, str]:
        """Load behavior-specific CSS templates."""
        return {
            'animation': """
                transition: all 0.3s ease;
            """,
            'hover': """
                &:hover {
                    opacity: 0.8;
                }
            """,
            'click': """
                cursor: pointer;
                user-select: none;
            """,
            'scroll': """
                overflow: auto;
                scroll-behavior: smooth;
            """
        }
    
    def _generate_optimized_content(self, content: str, analysis: AnalysisResult) -> str:
        """Generate optimized Java code by replacing style methods with CSS classes."""
        try:
            lines = content.splitlines()
            
            # Sort patterns by line number in reverse to avoid offset issues
            patterns = sorted(analysis.style_patterns, key=lambda p: p.line_number, reverse=True)
            
            # Create CSS class names with caching
            class_mapping = self._create_class_mapping(patterns)
            
            # Replace style methods with CSS class assignments
            for pattern in patterns:
                if pattern.css_property:
                    line_idx = pattern.line_number - 1
                    if 0 <= line_idx < len(lines):
                        # Replace the style method with CSS class assignment
                        new_line = self._replace_style_method(
                            lines[line_idx],
                            pattern,
                            class_mapping[pattern]
                        )
                        lines[line_idx] = new_line
            
            return "\n".join(lines)
        except Exception as e:
            self.logger.error(f"Error generating optimized content: {str(e)}")
            raise
    
    def _generate_css_content(self, analysis: AnalysisResult) -> str:
        """Generate CSS content from analysis results."""
        try:
            css_rules = []
            
            # Add component-specific CSS
            for pattern in analysis.style_patterns:
                if pattern.component_type and pattern.component_type in self.component_css:
                    css_rules.extend(self._generate_component_css(pattern))
            
            # Add behavior-specific CSS
            for pattern in analysis.style_patterns:
                if pattern.behavior_type and pattern.behavior_type in self.behavior_css:
                    css_rules.extend(self._generate_behavior_css(pattern))
            
            # Add style-specific CSS
            for pattern in analysis.style_patterns:
                if pattern.css_property and pattern.css_value:
                    css_rules.append(self._generate_style_css(pattern))
            
            # Remove duplicate rules
            css_rules = self._remove_duplicate_rules(css_rules)
            
            # Compress CSS
            css_content = "\n".join(css_rules)
            if self.config.compress_css:
                try:
                    css_content = csscompressor.compress(css_content)
                except Exception as e:
                    self.logger.warning(f"Error compressing CSS: {str(e)}")
            
            return css_content
        except Exception as e:
            self.logger.error(f"Error generating CSS content: {str(e)}")
            raise
    
    def _remove_duplicate_rules(self, rules: List[str]) -> List[str]:
        """Remove duplicate CSS rules."""
        try:
            seen = set()
            unique_rules = []
            
            for rule in rules:
                # Normalize rule for comparison
                normalized = re.sub(r'\s+', ' ', rule.strip())
                if normalized not in seen:
                    seen.add(normalized)
                    unique_rules.append(rule)
            
            return unique_rules
        except Exception as e:
            self.logger.error(f"Error removing duplicate rules: {str(e)}")
            return rules
    
    def _count_duplicate_rules(self, patterns: List[StylePattern]) -> int:
        """Count duplicate CSS rules in patterns."""
        try:
            seen = set()
            duplicates = 0
            
            for pattern in patterns:
                if pattern.css_property and pattern.css_value:
                    rule = f"{pattern.css_property}: {pattern.css_value}"
                    if rule in seen:
                        duplicates += 1
                    else:
                        seen.add(rule)
            
            return duplicates
        except Exception as e:
            self.logger.error(f"Error counting duplicate rules: {str(e)}")
            return 0
    
    def _generate_component_css(self, pattern: StylePattern) -> List[str]:
        """Generate component-specific CSS rules."""
        try:
            css = []
            component = self.component_css[pattern.component_type]
            
            # Add base component styles
            css.append(f".{pattern.class_name} {{")
            css.append(component['base'])
            css.append("}")
            
            # Add child component styles
            if pattern.child_components:
                for child in pattern.child_components:
                    css.append(f".{pattern.class_name} .{child} {{")
                    css.append(component.get('item', ''))
                    css.append("}")
            
            return css
        except Exception as e:
            self.logger.error(f"Error generating component CSS: {str(e)}")
            return []
    
    def _generate_behavior_css(self, pattern: StylePattern) -> List[str]:
        """Generate behavior-specific CSS rules."""
        try:
            css = []
            behavior = self.behavior_css[pattern.behavior_type]
            
            css.append(f".{pattern.class_name} {{")
            css.append(behavior)
            css.append("}")
            
            return css
        except Exception as e:
            self.logger.error(f"Error generating behavior CSS: {str(e)}")
            return []
    
    def _generate_style_css(self, pattern: StylePattern) -> str:
        """Generate style-specific CSS rule."""
        try:
            return f".{pattern.class_name} {{ {pattern.css_property}: {pattern.css_value}; }}"
        except Exception as e:
            self.logger.error(f"Error generating style CSS: {str(e)}")
            return ""
    
    def _generate_java_behaviors(self, analysis: AnalysisResult) -> str:
        """Generate Java code for behaviors that can't be handled by CSS."""
        try:
            java = []
            
            # Add imports
            java.append("import javax.swing.*;")
            java.append("import java.awt.*;")
            java.append("import java.awt.event.*;")
            java.append("")
            
            # Create behavior handler class
            java.append("public class BehaviorHandler {")
            
            # Add behavior methods
            for behavior in analysis.requires_java:
                java.extend(self._generate_behavior_method(behavior))
            
            java.append("}")
            
            return "\n".join(java)
        except Exception as e:
            self.logger.error(f"Error generating Java behaviors: {str(e)}")
            raise
    
    def _generate_behavior_method(self, behavior: str) -> List[str]:
        """Generate Java method for a specific behavior."""
        try:
            # Check cache first
            if behavior in self._java_cache:
                return self._java_cache[behavior]
            
            methods = {
                'drag_drop': """
    public void setupDragDrop(JComponent component) {
        component.setTransferHandler(new TransferHandler("text"));
        component.addMouseListener(new MouseAdapter() {
            public void mousePressed(MouseEvent e) {
                JComponent c = (JComponent) e.getSource();
                TransferHandler th = c.getTransferHandler();
                th.exportAsDrag(c, e, TransferHandler.COPY);
            }
        });
    }
""",
                'validation': """
    public void setupValidation(JTextField field, String pattern) {
        field.getDocument().addDocumentListener(new DocumentListener() {
            public void changedUpdate(DocumentEvent e) { validate(); }
            public void removeUpdate(DocumentEvent e) { validate(); }
            public void insertUpdate(DocumentEvent e) { validate(); }
            
            private void validate() {
                String text = field.getText();
                if (!text.matches(pattern)) {
                    field.setBackground(Color.PINK);
                } else {
                    field.setBackground(Color.WHITE);
                }
            }
        });
    }
"""
            }
            
            result = [methods.get(behavior, f"    // TODO: Implement {behavior} behavior")]
            
            # Cache the result
            with self._lock:
                self._java_cache[behavior] = result
            
            return result
        except Exception as e:
            self.logger.error(f"Error generating behavior method: {str(e)}")
            return [f"    // Error generating {behavior} behavior: {str(e)}"]
    
    def _create_class_mapping(self, patterns: List[StylePattern]) -> Dict[StylePattern, str]:
        """Create mapping from patterns to CSS class names."""
        try:
            mapping = {}
            used_names = set()
            
            for pattern in patterns:
                if pattern.css_property:
                    # Generate a unique class name
                    base_name = f"{pattern.class_name}_{pattern.css_property}"
                    class_name = base_name
                    counter = 1
                    
                    while class_name in used_names:
                        class_name = f"{base_name}_{counter}"
                        counter += 1
                    
                    mapping[pattern] = class_name
                    used_names.add(class_name)
            
            return mapping
        except Exception as e:
            self.logger.error(f"Error creating class mapping: {str(e)}")
            return {}
    
    def _replace_style_method(self, line: str, pattern: StylePattern, css_class: str) -> str:
        """Replace a style method call with CSS class assignment."""
        try:
            # Find the style method call
            method_call = f"{pattern.method_name}({', '.join(pattern.arguments)})"
            
            # Replace with CSS class assignment
            if "setStyle" in line:
                # If there's already a setStyle call, add to it
                return re.sub(
                    r'setStyle\("([^"]*)"\)',
                    lambda m: f'setStyle("{m.group(1)} {css_class}")',
                    line
                )
            else:
                # Otherwise, add a new setStyle call
                return line.replace(
                    method_call,
                    f'setStyle("{css_class}")'
                )
        except Exception as e:
            self.logger.error(f"Error replacing style method: {str(e)}")
            return line
    
    def _pattern_to_css_rule(self, pattern: StylePattern) -> Dict[str, Any]:
        """Convert a style pattern to a CSS rule dictionary."""
        try:
            return {
                "class": pattern.class_name,
                "property": pattern.css_property,
                "value": pattern.css_value,
                "component_type": pattern.component_type,
                "behavior_type": pattern.behavior_type,
                "original_method": pattern.method_name,
                "line": pattern.line_number
            }
        except Exception as e:
            self.logger.error(f"Error converting pattern to CSS rule: {str(e)}")
            return {}
    
    def _behavior_to_java(self, behavior: str) -> Dict[str, Any]:
        """Convert a behavior to Java implementation details."""
        try:
            return {
                "type": behavior,
                "method": f"setup{behavior.title().replace('_', '')}",
                "description": f"Implementation of {behavior} behavior"
            }
        except Exception as e:
            self.logger.error(f"Error converting behavior to Java: {str(e)}")
            return {} 