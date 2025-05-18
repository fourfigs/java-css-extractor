yes
import unittest
import tempfile
import os
import json
from pathlib import Path
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import psutil

from java_css_optimizer.optimizer.generators import (
    AdvancedCache, CacheConfig, CacheStrategy,
    TemplateManager, CSSGenerator, JavaGenerator,
    ContentGenerator
)
from java_css_optimizer.analyzer import StylePattern, AnalysisResult
from java_css_optimizer.config import Config
from java_css_optimizer.optimizer.exceptions import (
    MemoryLimitExceeded, CacheLimitExceeded,
    CSSRuleLimitException, JavaBehaviorLimitException
)

class TestAdvancedCache(unittest.TestCase):
    def setUp(self):
        self.config = CacheConfig(
            max_size=100,
            strategy=CacheStrategy.LRU,
            ttl=1.0,
            enable_persistence=True,
            persistence_path=Path(tempfile.mkdtemp())
        )
        self.cache = AdvancedCache(self.config)
    
    def tearDown(self):
        self.cache.clear()
        if self.config.persistence_path.exists():
            for file in self.config.persistence_path.glob('*.json'):
                file.unlink()
            self.config.persistence_path.rmdir()
    
    def test_basic_operations(self):
        # Test put and get
        self.cache.put("key1", "value1", tags={"test"})
        self.assertEqual(self.cache.get("key1"), "value1")
        
        # Test cache miss
        self.assertIsNone(self.cache.get("nonexistent"))
        
        # Test cache stats
        stats = self.cache.get_stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
    
    def test_ttl_expiration(self):
        self.cache.put("key1", "value1")
        time.sleep(1.1)  # Wait for TTL to expire
        self.assertIsNone(self.cache.get("key1"))
    
    def test_tag_operations(self):
        self.cache.put("key1", "value1", tags={"tag1"})
        self.cache.put("key2", "value2", tags={"tag1", "tag2"})
        
        # Test get_by_tag
        self.assertEqual(len(self.cache.get_by_tag("tag1")), 2)
        self.assertEqual(len(self.cache.get_by_tag("tag2")), 1)
        
        # Test invalidate_by_tag
        self.cache.invalidate_by_tag("tag1")
        self.assertEqual(len(self.cache.get_by_tag("tag1")), 0)
        self.assertEqual(len(self.cache.get_by_tag("tag2")), 1)
    
    def test_memory_limit(self):
        large_value = "x" * (self.config.max_memory_usage + 1)
        with self.assertRaises(CacheLimitExceeded):
            self.cache.put("key1", large_value)

class TestTemplateManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = Config({
            'template_dir': str(self.temp_dir)
        })
        self.manager = TemplateManager(self.config)
    
    def tearDown(self):
        self.manager.__del__()
        if self.temp_dir.exists():
            for file in self.temp_dir.glob('**/*'):
                if file.is_file():
                    file.unlink()
            self.temp_dir.rmdir()
    
    def test_template_operations(self):
        # Test default template creation
        template = self.manager.get_template('components', 'default')
        self.assertIsNotNone(template)
        self.assertIn('base', template)
        
        # Test template validation
        self.assertTrue(self.manager._validate_component_template(template))
        
        # Test invalid template
        invalid_template = {'invalid': 'template'}
        self.assertFalse(self.manager._validate_component_template(invalid_template))
    
    def test_template_persistence(self):
        template = {
            'base': 'test',
            'item': 'test',
            'hover': 'test',
            'active': 'test'
        }
        
        # Save template
        asyncio.run(self.manager._save_template('components', 'test', template))
        
        # Load template
        loaded = self.manager.get_template('components', 'test')
        self.assertEqual(loaded, template)

class TestCSSGenerator(unittest.TestCase):
    def setUp(self):
        self.config = Config({
            'cache_dir': tempfile.mkdtemp(),
            'compress_css': True
        })
        self.generator = CSSGenerator(self.config)
        
        # Create test analysis result
        self.analysis = AnalysisResult(
            style_patterns=[
                StylePattern(
                    class_name="test-class",
                    css_property="color",
                    css_value="red",
                    component_type="button",
                    behavior_type="hover",
                    has_hover=True,
                    has_active=True
                )
            ],
            requires_java=False
        )
    
    def tearDown(self):
        self.generator.__del__()
        if Path(self.config['cache_dir']).exists():
            for file in Path(self.config['cache_dir']).glob('**/*'):
                if file.is_file():
                    file.unlink()
            Path(self.config['cache_dir']).rmdir()
    
    def test_css_generation(self):
        css = self.generator.generate_css_content(self.analysis)
        self.assertIsNotNone(css)
        self.assertIn(".test-class", css)
        self.assertIn("color: red", css)
    
    def test_css_validation(self):
        # Test valid CSS
        self.assertTrue(self.generator._validate_css_property("color"))
        self.assertTrue(self.generator._validate_css_value("red"))
        
        # Test invalid CSS
        self.assertFalse(self.generator._validate_css_property(""))
        self.assertFalse(self.generator._validate_css_value(""))
    
    def test_duplicate_removal(self):
        rules = [
            ".test { color: red; }",
            ".test { color: red; }",
            ".test { color: blue; }"
        ]
        unique = self.generator._remove_duplicate_rules(rules)
        self.assertEqual(len(unique), 2)

class TestJavaGenerator(unittest.TestCase):
    def setUp(self):
        self.config = Config({
            'cache_dir': tempfile.mkdtemp()
        })
        self.generator = JavaGenerator(self.config)
        
        # Create test analysis result
        self.analysis = AnalysisResult(
            style_patterns=[],
            requires_java=['drag_drop', 'validation']
        )
    
    def tearDown(self):
        self.generator.__del__()
        if Path(self.config['cache_dir']).exists():
            for file in Path(self.config['cache_dir']).glob('**/*'):
                if file.is_file():
                    file.unlink()
            Path(self.config['cache_dir']).rmdir()
    
    def test_java_generation(self):
        java = self.generator.generate_java_behaviors(self.analysis)
        self.assertIsNotNone(java)
        self.assertIn("public class BehaviorHandler", java)
        self.assertIn("setupDragDrop", java)
        self.assertIn("setupValidation", java)
    
    def test_behavior_conversion(self):
        behavior = {
            'type': 'drag_drop',
            'class': 'TestComponent',
            'method': 'setupDragDrop',
            'context': {}
        }
        java = self.generator.behavior_to_java(behavior)
        self.assertIsNotNone(java)
        self.assertIn("TestComponent", java)

class TestContentGenerator(unittest.TestCase):
    def setUp(self):
        self.config = Config({
            'cache_dir': tempfile.mkdtemp()
        })
        self.generator = ContentGenerator(self.config)
        
        # Create test content and analysis
        self.content = """
        public class TestComponent extends JComponent {
            public void init() {
                setColor(Color.RED);
                setStyle("background: blue");
            }
        }
        """
        self.analysis = AnalysisResult(
            style_patterns=[
                StylePattern(
                    class_name="test-class",
                    css_property="color",
                    css_value="red",
                    method_name="setColor",
                    arguments=["Color.RED"]
                )
            ],
            requires_java=False
        )
    
    def tearDown(self):
        self.generator.__del__()
        if Path(self.config['cache_dir']).exists():
            for file in Path(self.config['cache_dir']).glob('**/*'):
                if file.is_file():
                    file.unlink()
            Path(self.config['cache_dir']).rmdir()
    
    def test_content_optimization(self):
        optimized = self.generator.generate_optimized_content(self.content, self.analysis)
        self.assertIsNotNone(optimized)
        self.assertIn("setStyle", optimized)
        self.assertNotIn("setColor", optimized)
    
    def test_class_mapping(self):
        mapping = self.generator._create_class_mapping(self.analysis)
        self.assertIsNotNone(mapping)
        self.assertTrue(len(mapping) > 0)

if __name__ == '__main__':
    unittest.main() 