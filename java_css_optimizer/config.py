#!/usr/bin/env python3
"""
Configuration management for Java CSS Optimizer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

@dataclass
class Config:
    """Configuration for Java CSS Optimizer."""
    optimization_level: int = 2
    css_output: Optional[Path] = None
    css_name: Optional[str] = None
    preserve_comments: bool = True
    rules: Dict[str, Any] = None
    
    def __post_init__(self):
        """Initialize default rules if not provided."""
        if self.rules is None:
            self.rules = self._default_rules()
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'Config':
        """Load configuration from a YAML file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            return cls(
                optimization_level=data.get('optimization', {}).get('level', 2),
                css_output=Path(data.get('optimization', {}).get('css_output', '')),
                css_name=data.get('optimization', {}).get('css_name'),
                preserve_comments=data.get('optimization', {}).get('preserve_comments', True),
                rules=data.get('rules', {})
            )
        except Exception as e:
            logging.error(f"Error loading config from {file_path}: {str(e)}")
            return cls()
    
    def _default_rules(self) -> Dict[str, Any]:
        """Get default optimization rules."""
        return {
            'color': {
                'pattern': 'setColor|setBackground|setForeground',
                'css_property': 'color|background-color'
            },
            'font': {
                'pattern': 'setFont|setFontSize|setFontStyle',
                'css_property': 'font|font-size|font-style'
            },
            'layout': {
                'pattern': 'setLayout|setAlignment|setMargin',
                'css_property': 'display|text-align|margin'
            },
            'border': {
                'pattern': 'setBorder|setBorderColor|setBorderWidth',
                'css_property': 'border|border-color|border-width'
            },
            'padding': {
                'pattern': 'setPadding|setInsets',
                'css_property': 'padding'
            },
            'size': {
                'pattern': 'setSize|setPreferredSize|setMinimumSize',
                'css_property': 'width|height'
            },
            'position': {
                'pattern': 'setLocation|setBounds',
                'css_property': 'position|top|left'
            },
            'opacity': {
                'pattern': 'setOpacity|setAlpha',
                'css_property': 'opacity'
            },
            'cursor': {
                'pattern': 'setCursor',
                'css_property': 'cursor'
            },
            'text': {
                'pattern': 'setText|setLabel',
                'css_property': 'content'
            }
        }
    
    def save(self, file_path: Path) -> None:
        """Save configuration to a YAML file."""
        data = {
            'optimization': {
                'level': self.optimization_level,
                'css_output': str(self.css_output) if self.css_output else None,
                'css_name': self.css_name,
                'preserve_comments': self.preserve_comments
            },
            'rules': self.rules
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_rule_pattern(self, rule_name: str) -> Optional[str]:
        """Get the pattern for a specific rule."""
        return self.rules.get(rule_name, {}).get('pattern')
    
    def get_css_property(self, rule_name: str) -> Optional[str]:
        """Get the CSS property for a specific rule."""
        return self.rules.get(rule_name, {}).get('css_property')
    
    def add_rule(self, name: str, pattern: str, css_property: str) -> None:
        """Add a new optimization rule."""
        self.rules[name] = {
            'pattern': pattern,
            'css_property': css_property
        }
    
    def remove_rule(self, name: str) -> None:
        """Remove an optimization rule."""
        self.rules.pop(name, None) 