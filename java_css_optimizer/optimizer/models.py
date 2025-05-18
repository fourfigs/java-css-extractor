"""
Data models for the Java CSS Optimizer.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional

@dataclass
class OptimizationResult:
    """Results of Java code optimization."""
    original_file: Path
    optimized_file: Path
    css_file: Path
    java_file: Optional[Path] = None
    changes_made: int = 0
    css_rules: List[Dict[str, Any]] = field(default_factory=list)
    java_behaviors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_stats: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate a summary of the optimization results."""
        summary = [
            f"Optimization Summary:",
            f"  Original file: {self.original_file}",
            f"  Optimized file: {self.optimized_file}",
            f"  CSS file: {self.css_file}",
            f"  Changes made: {self.changes_made}",
            f"  CSS rules generated: {len(self.css_rules)}"
        ]
        
        if self.java_file:
            summary.extend([
                f"  Java behaviors file: {self.java_file}",
                f"  Java behaviors: {len(self.java_behaviors)}"
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

@dataclass
class CSSRule:
    """CSS rule data model."""
    selector: str
    properties: Dict[str, str]
    component_type: Optional[str] = None
    behavior_type: Optional[str] = None
    parent_selector: Optional[str] = None
    media_query: Optional[str] = None
    
    def to_string(self, indent: str = "    ") -> str:
        """Convert the rule to a CSS string."""
        lines = []
        
        # Add media query if present
        if self.media_query:
            lines.append(f"@media {self.media_query} {{")
            indent = indent * 2
        
        # Add selector
        selector = self.selector
        if self.parent_selector:
            selector = f"{self.parent_selector} {selector}"
        lines.append(f"{indent}{selector} {{")
        
        # Add properties
        for prop, value in self.properties.items():
            lines.append(f"{indent}{indent}{prop}: {value};")
        
        # Close rule
        lines.append(f"{indent}}}")
        
        # Close media query if present
        if self.media_query:
            lines.append("}")
        
        return "\n".join(lines)

@dataclass
class JavaBehavior:
    """Java behavior data model."""
    type: str
    method_name: str
    parameters: List[str]
    return_type: str
    body: str
    imports: List[str] = field(default_factory=list)
    
    def to_string(self, indent: str = "    ") -> str:
        """Convert the behavior to a Java method string."""
        lines = []
        
        # Add method signature
        params = ", ".join(self.parameters)
        lines.append(f"{indent}public {self.return_type} {self.method_name}({params}) {{")
        
        # Add method body
        for line in self.body.splitlines():
            lines.append(f"{indent}{indent}{line}")
        
        # Close method
        lines.append(f"{indent}}}")
        
        return "\n".join(lines) 