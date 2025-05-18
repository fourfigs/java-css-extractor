#!/usr/bin/env python3
"""
Java code analyzer for CSS optimization.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
import re
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import threading
from collections import defaultdict
import time
import queue
import weakref
import os
import shutil
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@dataclass
class ImportedFile:
    """Represents an imported Java file."""
    path: Path
    package: str
    class_name: str
    imports: List[str]
    style_patterns: List['StylePattern'] = field(default_factory=list)
    dependencies: List['ImportedFile'] = field(default_factory=list)
    is_analyzed: bool = False

@dataclass
class StylePattern:
    """Represents a style pattern found in Java code."""
    class_name: str
    method_name: str
    arguments: List[str]
    line_number: int
    css_property: Optional[str] = None
    css_value: Optional[str] = None
    component_type: Optional[str] = None
    behavior_type: Optional[str] = None
    parent_component: Optional[str] = None
    child_components: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0

@dataclass
class AnalysisResult:
    """Results of Java code analysis."""
    file_path: Path
    style_patterns: List[StylePattern]
    component_hierarchy: Dict[str, List[str]]
    behavior_patterns: List[Dict[str, Any]]
    requires_java: List[str]
    imported_files: List[ImportedFile]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class JavaAnalyzer:
    """Analyzes Java code for style patterns and component hierarchy."""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self._cache = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()
        self._file_locks = defaultdict(threading.Lock)
        self._import_cache = weakref.WeakKeyDictionary()
        self._classpath_cache = weakref.WeakKeyDictionary()
        self._pattern_cache = weakref.WeakKeyDictionary()
        self._pattern_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # Compile regex patterns
        self.import_pattern = re.compile(r'import\s+([\w.]+);')
        self.package_pattern = re.compile(r'package\s+([\w.]+);')
        self.class_pattern = re.compile(r'(?:public|private|protected)?\s+class\s+(\w+)')
        
        # Compile regex patterns for better performance
        self.component_patterns = self._compile_patterns({
            'menu': r'JMenu|MenuBar|PopupMenu',
            'gallery': r'JList|JTable|GridLayout',
            'form': r'JForm|JPanel.*form',
            'navigation': r'JTabbedPane|JTree|JMenuBar',
            'modal': r'JDialog|JOptionPane',
            'carousel': r'CardLayout.*swipe|JPanel.*carousel',
            'accordion': r'JPanel.*accordion|JSplitPane',
            'toolbar': r'JToolBar|JPanel.*toolbar',
            'sidebar': r'JSplitPane.*sidebar|JPanel.*sidebar',
            'dropdown': r'JComboBox|JPopupMenu',
            'tabs': r'JTabbedPane|JPanel.*tabs',
            'grid': r'GridLayout|JTable',
            'list': r'JList|JTree',
            'form': r'JForm|JPanel.*form',
            'chart': r'JFreeChart|ChartPanel',
            'map': r'JMapViewer|MapPanel',
            'calendar': r'JCalendar|CalendarPanel',
            'wizard': r'JPanel.*wizard|CardLayout.*wizard',
            'dashboard': r'JPanel.*dashboard|GridLayout.*dashboard'
        })
        
        self.behavior_patterns = self._compile_patterns({
            'animation': r'Timer|Animation|Transition',
            'drag_drop': r'DragSource|DropTarget|TransferHandler',
            'resize': r'ComponentListener|ResizeListener',
            'hover': r'MouseListener.*hover|MouseMotionListener',
            'click': r'MouseListener|ActionListener',
            'scroll': r'JScrollPane|ScrollListener',
            'keyboard': r'KeyListener|KeyBinding',
            'focus': r'FocusListener|FocusTraversalPolicy',
            'validation': r'InputVerifier|DocumentListener',
            'autocomplete': r'JComboBox.*auto|AutoComplete',
            'infinite_scroll': r'ScrollListener.*load|JList.*infinite',
            'lazy_load': r'ImageObserver|AsyncImageLoader',
            'sort': r'TableRowSorter|ListModel.*sort',
            'filter': r'RowFilter|ListModel.*filter',
            'search': r'DocumentListener.*search|SearchField',
            'zoom': r'MouseWheelListener.*zoom|ZoomController',
            'pan': r'MouseListener.*pan|PanController',
            'tooltip': r'ToolTipManager|JToolTip',
            'context_menu': r'MouseListener.*popup|JPopupMenu',
            'modal': r'JDialog.*show|JOptionPane.*show'
        })
        
        self.event_handlers = self._compile_patterns({
            'mouse': r'MouseListener|MouseMotionListener|MouseWheelListener',
            'keyboard': r'KeyListener|KeyBinding',
            'focus': r'FocusListener|FocusTraversalPolicy',
            'window': r'WindowListener|WindowStateListener',
            'component': r'ComponentListener|ContainerListener',
            'document': r'DocumentListener|UndoableEditListener',
            'action': r'ActionListener|Action',
            'change': r'ChangeListener|ListSelectionListener',
            'item': r'ItemListener|ListSelectionListener',
            'tree': r'TreeSelectionListener|TreeExpansionListener'
        })
    
    def _compile_patterns(self, patterns: Dict[str, str]) -> Dict[str, re.Pattern]:
        """Compile regex patterns for better performance."""
        return {k: re.compile(v, re.MULTILINE) for k, v in patterns.items()}
    
    def analyze_file(self, file_path: Path) -> AnalysisResult:
        """Analyze a Java file for style patterns and component hierarchy."""
        try:
            # Check cache first
            if file_path in self._cache:
                return self._cache[file_path]
            
            # Get file-specific lock
            lock = self._file_locks[file_path]
            if not lock.acquire(timeout=10):
                raise TimeoutError(f"Timeout acquiring lock for {file_path}")
            
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract package and class name
                package = self._extract_package(content)
                class_name = self._extract_class_name(content)
                
                # Find imports and analyze imported files
                imports = self._find_imports(content)
                imported_files = self._analyze_imports(file_path, imports)
                
                # Analyze style patterns
                style_patterns = self._find_style_patterns(content)
                
                # Analyze component hierarchy
                component_hierarchy = self._analyze_component_hierarchy(content, imported_files)
                
                # Analyze behavior patterns
                behavior_patterns = self._find_behavior_patterns(content, imported_files)
                
                # Determine Java requirements
                requires_java = self._determine_java_requirements(behavior_patterns)
                
                # Calculate confidence scores
                self._calculate_confidence_scores(style_patterns, imported_files)
                
                # Create analysis result
                result = AnalysisResult(
                    file_path=file_path,
                    style_patterns=style_patterns,
                    component_hierarchy=component_hierarchy,
                    behavior_patterns=behavior_patterns,
                    requires_java=requires_java,
                    imported_files=imported_files
                )
                
                # Cache the result
                with self._lock:
                    self._cache[file_path] = result
                
                return result
                
            finally:
                lock.release()
                self._file_locks.pop(file_path, None)
                
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {str(e)}")
            raise
    
    def _extract_package(self, content: str) -> str:
        """Extract package name from Java file content."""
        match = self.package_pattern.search(content)
        return match.group(1) if match else ""
    
    def _extract_class_name(self, content: str) -> str:
        """Extract class name from Java file content."""
        match = self.class_pattern.search(content)
        return match.group(1) if match else ""
    
    def _find_imports(self, content: str) -> List[str]:
        """Find all imports in Java file content."""
        return self.import_pattern.findall(content)
    
    def _analyze_imports(self, file_path: Path, imports: List[str]) -> List[ImportedFile]:
        """Analyze imported Java files."""
        imported_files = []
        
        for imp in imports:
            try:
                # Check import cache
                if imp in self._import_cache:
                    imported_files.append(self._import_cache[imp])
                    continue
                
                # Find imported file
                imported_file = self._find_imported_file(file_path, imp)
                if imported_file:
                    # Analyze the imported file
                    analysis = self.analyze_file(imported_file)
                    
                    # Create ImportedFile object
                    imported = ImportedFile(
                        path=imported_file,
                        package=imp.rsplit('.', 1)[0],
                        class_name=imp.rsplit('.', 1)[1],
                        imports=analysis.imported_files,
                        style_patterns=analysis.style_patterns,
                        is_analyzed=True
                    )
                    
                    imported_files.append(imported)
                    
                    # Cache the result
                    with self._lock:
                        self._import_cache[imp] = imported
                    
            except Exception as e:
                self.logger.warning(f"Error analyzing import {imp}: {str(e)}")
                continue
        
        return imported_files
    
    def _find_imported_file(self, current_file: Path, import_path: str) -> Optional[Path]:
        """Find the physical file for an import."""
        try:
            # Check classpath cache
            if import_path in self._classpath_cache:
                return self._classpath_cache[import_path]
            
            # Convert import path to file path
            file_path = current_file.parent / f"{import_path.replace('.', '/')}.java"
            
            if file_path.exists():
                # Cache the result
                with self._lock:
                    self._classpath_cache[import_path] = file_path
                return file_path
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Error finding imported file for {import_path}: {str(e)}")
            return None
    
    def _find_style_patterns(self, content: str) -> List[StylePattern]:
        """Find style patterns in Java code."""
        patterns = []
        lines = content.splitlines()
        
        # Use a more robust pattern matching approach
        style_methods = defaultdict(list)
        for i, line in enumerate(lines, 1):
            try:
                # Look for style method calls with better pattern matching
                style_match = re.search(r'(\w+)(?:\.|->)(set\w+)\((.*?)\)', line)
                if style_match:
                    class_name = style_match.group(1)
                    method_name = style_match.group(2)
                    arguments = [arg.strip() for arg in style_match.group(3).split(',')]
                    
                    # Get context from surrounding lines
                    context = self._get_context(lines, i)
                    
                    # Determine component type with context
                    component_type = self._detect_component_type(class_name, lines, context)
                    
                    # Detect behavior with context
                    behavior_type = self._detect_behavior(class_name, lines, context)
                    
                    # Find parent component with improved accuracy
                    parent = self._find_parent_component(class_name, lines, context)
                    
                    # Find child components with improved accuracy
                    children = self._find_child_components(class_name, lines, context)
                    
                    # Find event handlers with improved accuracy
                    handlers = self._find_event_handlers(class_name, lines, context)
                    
                    pattern = StylePattern(
                        class_name=class_name,
                        method_name=method_name,
                        arguments=arguments,
                        line_number=i,
                        component_type=component_type,
                        behavior_type=behavior_type,
                        parent_component=parent,
                        child_components=children,
                        event_handlers=handlers,
                        context=context
                    )
                    
                    patterns.append(pattern)
                    style_methods[class_name].append(pattern)
            except Exception as e:
                self.logger.warning(f"Error processing line {i}: {str(e)}")
                continue
        
        return patterns
    
    def _get_context(self, lines: List[str], line_number: int, context_lines: int = 3) -> Dict[str, Any]:
        """Get context from surrounding lines."""
        try:
            start = max(0, line_number - context_lines - 1)
            end = min(len(lines), line_number + context_lines)
            
            return {
                'previous_lines': lines[start:line_number-1],
                'next_lines': lines[line_number:end],
                'indentation': len(lines[line_number-1]) - len(lines[line_number-1].lstrip()),
                'surrounding_code': '\n'.join(lines[start:end])
            }
        except Exception as e:
            self.logger.warning(f"Error getting context for line {line_number}: {str(e)}")
            return {
                'previous_lines': [],
                'next_lines': [],
                'indentation': 0,
                'surrounding_code': ''
            }
    
    def _detect_component_type(self, class_name: str, lines: List[str], context: Dict[str, Any]) -> Optional[str]:
        """Detect the type of UI component with improved accuracy."""
        try:
            # Check cache first with lock
            cache_key = f"{class_name}_{hash(context['surrounding_code'])}"
            with self._pattern_lock:
                if cache_key in self._pattern_cache:
                    return self._pattern_cache[cache_key]
            
            # Analyze context for better component detection
            for comp_type, pattern in self.component_patterns.items():
                # Check surrounding code
                if pattern.search(context['surrounding_code']):
                    with self._pattern_lock:
                        self._pattern_cache[cache_key] = comp_type
                    return comp_type
                
                # Check class name patterns
                if re.search(rf'{comp_type}|{comp_type.title()}', class_name):
                    with self._pattern_lock:
                        self._pattern_cache[cache_key] = comp_type
                    return comp_type
            
            return None
        except Exception as e:
            self.logger.warning(f"Error detecting component type for {class_name}: {str(e)}")
            return None
    
    def _detect_behavior(self, class_name: str, lines: List[str], context: Dict[str, Any]) -> Optional[str]:
        """Detect behavior patterns with improved accuracy."""
        try:
            # Check cache first with lock
            cache_key = f"{class_name}_behavior_{hash(context['surrounding_code'])}"
            with self._pattern_lock:
                if cache_key in self._pattern_cache:
                    return self._pattern_cache[cache_key]
            
            # Analyze context for better behavior detection
            for behavior, pattern in self.behavior_patterns.items():
                if pattern.search(context['surrounding_code']):
                    with self._pattern_lock:
                        self._pattern_cache[cache_key] = behavior
                    return behavior
            
            return None
        except Exception as e:
            self.logger.warning(f"Error detecting behavior for {class_name}: {str(e)}")
            return None
    
    def _find_parent_component(self, class_name: str, lines: List[str], context: Dict[str, Any]) -> Optional[str]:
        """Find parent component with improved accuracy."""
        try:
            # Look in surrounding context first
            for line in context['previous_lines']:
                if f"add({class_name})" in line or f"addComponent({class_name})" in line:
                    parent_match = re.search(r'(\w+)\.add', line)
                    if parent_match:
                        return parent_match.group(1)
            
            # Look in the entire file if not found in context
            for line in lines:
                if f"add({class_name})" in line or f"addComponent({class_name})" in line:
                    parent_match = re.search(r'(\w+)\.add', line)
                    if parent_match:
                        return parent_match.group(1)
            
            return None
        except Exception as e:
            self.logger.warning(f"Error finding parent component for {class_name}: {str(e)}")
            return None
    
    def _find_child_components(self, class_name: str, lines: List[str], context: Dict[str, Any]) -> List[str]:
        """Find child components with improved accuracy."""
        try:
            children = set()
            
            # Look in surrounding context first
            for line in context['next_lines']:
                if f"{class_name}.add(" in line:
                    child_match = re.search(r'add\((\w+)\)', line)
                    if child_match:
                        children.add(child_match.group(1))
            
            # Look in the entire file if not found in context
            for line in lines:
                if f"{class_name}.add(" in line:
                    child_match = re.search(r'add\((\w+)\)', line)
                    if child_match:
                        children.add(child_match.group(1))
            
            return list(children)
        except Exception as e:
            self.logger.warning(f"Error finding child components for {class_name}: {str(e)}")
            return []
    
    def _find_event_handlers(self, class_name: str, lines: List[str], context: Dict[str, Any]) -> Dict[str, str]:
        """Find event handlers with improved accuracy."""
        try:
            handlers = {}
            
            # Look in surrounding context first
            for line in context['surrounding_code'].split('\n'):
                for event_type, pattern in self.event_handlers.items():
                    if f"{class_name}.add" in line and pattern.search(line):
                        handlers[event_type] = line.strip()
            
            # Look in the entire file if not found in context
            for line in lines:
                for event_type, pattern in self.event_handlers.items():
                    if f"{class_name}.add" in line and pattern.search(line):
                        handlers[event_type] = line.strip()
            
            return handlers
        except Exception as e:
            self.logger.warning(f"Error finding event handlers for {class_name}: {str(e)}")
            return {}
    
    def _analyze_component_hierarchy(self, content: str, imported_files: List[ImportedFile]) -> Dict[str, List[str]]:
        """Analyze component hierarchy including imported components."""
        hierarchy = defaultdict(list)
        
        # Analyze local components
        lines = content.splitlines()
        for line in lines:
            add_match = re.search(r'(\w+)\.add\((\w+)\)', line)
            if add_match:
                parent = add_match.group(1)
                child = add_match.group(2)
                hierarchy[parent].append(child)
        
        # Add imported components
        for imported in imported_files:
            for pattern in imported.style_patterns:
                if pattern.component_type:
                    hierarchy[pattern.component_type].extend(pattern.child_components)
        
        return dict(hierarchy)
    
    def _find_behavior_patterns(self, content: str, imported_files: List[ImportedFile]) -> List[Dict[str, Any]]:
        """Find behavior patterns including those from imported files."""
        patterns = []
        
        # Analyze local behaviors
        lines = content.splitlines()
        for line in lines:
            for behavior, pattern in self.behavior_patterns.items():
                if pattern.search(line):
                    patterns.append({
                        'type': behavior,
                        'class': '',
                        'method': '',
                        'context': self._get_context(lines, lines.index(line))
                    })
        
        # Add imported behaviors
        for imported in imported_files:
            for pattern in imported.style_patterns:
                if pattern.behavior_type:
                    patterns.append({
                        'type': pattern.behavior_type,
                        'class': imported.class_name,
                        'method': pattern.method_name,
                        'context': pattern.context
                    })
        
        return patterns
    
    def _determine_java_requirements(self, behavior_patterns: List[Dict[str, Any]]) -> List[str]:
        """Determine which behaviors require Java implementation."""
        requires_java = []
        
        # Behaviors that typically require Java
        java_behaviors = {
            'drag_drop', 'validation', 'autocomplete',
            'infinite_scroll', 'lazy_load', 'sort',
            'filter', 'search', 'zoom', 'pan'
        }
        
        for behavior in behavior_patterns:
            if behavior['type'] in java_behaviors:
                requires_java.append(behavior['type'])
        
        return requires_java
    
    def _calculate_confidence_scores(self, style_patterns: List[StylePattern], imported_files: List[ImportedFile]) -> None:
        """Calculate confidence scores for style patterns."""
        # ... existing confidence score calculation code ...
        pass 