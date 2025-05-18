#!/usr/bin/env python3
"""
Reporting functionality for Java CSS Optimizer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import json
from datetime import datetime

@dataclass
class OptimizationReport:
    """Report of optimization results."""
    timestamp: str
    input_files: List[str]
    output_files: List[str]
    css_file: str
    total_changes: int
    style_patterns: Dict[str, int]
    optimization_level: int
    duration: float

class Reporter:
    """Handles generation of optimization reports."""
    
    def __init__(self, output_dir: Path):
        """Initialize reporter with output directory."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_report(self, results: List[Dict[str, Any]], 
                       css_file: str,
                       optimization_level: int,
                       duration: float) -> Path:
        """Generate an optimization report."""
        # Collect statistics
        total_changes = sum(r['changes'] for r in results)
        style_patterns = {}
        for result in results:
            for pattern in result['css_rules']:
                style_patterns[pattern['type']] = style_patterns.get(pattern['type'], 0) + 1
        
        # Create report
        report = OptimizationReport(
            timestamp=datetime.now().isoformat(),
            input_files=[str(r['original_file']) for r in results],
            output_files=[str(r['optimized_file']) for r in results],
            css_file=css_file,
            total_changes=total_changes,
            style_patterns=style_patterns,
            optimization_level=optimization_level,
            duration=duration
        )
        
        # Save report
        report_path = self.output_dir / f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self._report_to_dict(report), f, indent=2)
        
        return report_path
    
    def _report_to_dict(self, report: OptimizationReport) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'timestamp': report.timestamp,
            'input_files': report.input_files,
            'output_files': report.output_files,
            'css_file': report.css_file,
            'total_changes': report.total_changes,
            'style_patterns': report.style_patterns,
            'optimization_level': report.optimization_level,
            'duration': report.duration
        }
    
    def print_summary(self, report: OptimizationReport) -> None:
        """Print a summary of the optimization report."""
        print("\nOptimization Summary:")
        print("=" * 50)
        print(f"Timestamp: {report.timestamp}")
        print(f"Duration: {report.duration:.2f} seconds")
        print(f"Optimization Level: {report.optimization_level}")
        print(f"\nFiles Processed: {len(report.input_files)}")
        print(f"Total Changes: {report.total_changes}")
        print(f"\nCSS File: {report.css_file}")
        print("\nStyle Patterns Found:")
        for pattern, count in report.style_patterns.items():
            print(f"  - {pattern}: {count}")
        print("=" * 50) 